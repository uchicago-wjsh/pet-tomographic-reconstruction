#!/usr/bin/env python3
"""Convert GATE ROOT file(s) into a list-mode .npz, imposing a finite CTR.

The counterpart to ``make_listmode.py``, which samples a directory of
equal-length chunks to emulate a SHORTER acquisition. That is meaningless for a
fixed-primaries run, so this one takes the files it is given and uses all of
them -- e.g. ``macros/main_point_center.mac``, chunked to keep GATE's memory
bounded. Pass several files and they are concatenated into one listmode; the
emitted-primary counts add up, and per-file eventIDs are never compared across
files.

Each coincidence must survive, in order:

  * trues only       eventID1 == eventID2, dropping randoms (``--keep-randoms``
                     turns this off). A fixed-primaries run spaces its events
                     far wider than the coincidence window, so it has none.
  * energy window    both singles, in keV. The simulation itself only cut at
                     the alpha=3 window (357.7-664.3 keV), so anything
                     narrower is applied here.
  * cross-panel      the two singles must land on different plates
                     (``--no-cross-panel`` turns this off).

Timing. ``digitizer.mac`` inserts no timeResolution module, so the ROOT times
carry no electronics jitter and the CTR is imposed here: Gaussian noise of
sigma = CTR_FWHM / (2.3548 * sqrt(2)) is added INDEPENDENTLY to each arm, so
their difference dt = t1 - t2 acquires exactly CTR_FWHM.

Two time bases are available (``--time-source``):

  readout          (default) GATE's own Coincidences time1/time2. The readout
                   policy is TakeEnergyWinner, i.e. the EARLIEST contributing
                   hit time in the module.
  energy-weighted  sum_i(edep_i * t_i) / sum_i(edep_i) per module, rebuilt
                   from the Hits tree -- the definition this project specifies
                   (see scripts/module_time.py and the README). Requires the
                   run to have stored Hits.

The two differ by ~33 ps rms on the third of events that deposit in more than
one crystal, which is NOT negligible next to a 40 ps CTR (12 ps per arm), so
prefer ``--time-source energy-weighted`` whenever Hits are available.

Output keys match reconstruct_mlem_pytomography.py: ``detector_ids`` (N,2 flat
LUT ids) and ``tof_seconds`` (= time1 - time2).

    python reconstruction/root_to_listmode.py root/point_center/point_1e6.root \
        --ctr-ps 40 --time-source energy-weighted \
        --out results/listmode_point/lm_point_1e6_ctr40ps.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
COINC_BRANCHES = ["runID", "eventID1", "eventID2", "energy1", "energy2",
                  "time1", "time2", "level1ID1", "level2ID1", "level1ID2", "level2ID2"]
HIT_BRANCHES = ["runID", "eventID", "level1ID", "time", "edep"]
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))      # 1 / 2.3548


def load_panel_map(lut_json: Path):
    """Return (module -> panel index, crystals per module) from the LUT sidecar."""
    meta = json.loads(lut_json.read_text())
    panel = np.full(meta["n_modules"], -1, dtype=np.int16)
    for index, plate in enumerate(meta["plates"].values()):
        first, last = plate["module_id_range"]
        panel[first:last + 1] = index
    if (panel < 0).any():
        raise ValueError(f"{lut_json}: 'plates' does not cover every module")
    return panel, meta["n_crystals_per_module"]


def emitted_primaries(f) -> int | None:
    """Number of primary events GATE generated, from its ``pet_data`` tree.

    Read ``pet_data``, NOT the identically named TH1D objects at the top of the
    file -- those are 100-bin histograms spanning 0..9e11, so they locate the
    count only to within 9e9. scripts/sensitivity.py reads the same branch.

    GATE's own counter is one short of the number of events actually beamed on:
    GateToRoot increments ``nbPrimaries`` per event and then subtracts 1 at the
    end of every run ("Number of primaries increase too much at each end of
    run", GateToRoot.cc:679). A 1e6-event run therefore records 999,999. The
    value is returned unchanged, so it matches sensitivity.py; the discrepancy
    is 1 part in 1e6 and affects nothing.
    """
    if "pet_data" not in {k.split(";")[0] for k in f.keys()}:
        return None
    values = f["pet_data"]["total_nb_primaries"].array(library="np")
    return int(values[0]) if values.size else None


def energy_weighted_times(f) -> tuple[np.ndarray, np.ndarray]:
    """(sorted (run,event,module) keys, energy-weighted time) from the Hits tree.

    GATE's readout groups every hit of one event inside one module into a
    single Single, so (runID, eventID, level1ID) identifies an arm uniquely.
    """
    trees = {k.split(";")[0] for k in f.keys()}
    if "Hits" not in trees:
        raise SystemExit("--time-source energy-weighted needs a Hits tree; "
                         "re-run with macros/output_hits.mac")
    h = f["Hits"].arrays(HIT_BRANCHES, library="np")
    keep = h["edep"] > 0
    if not keep.any():
        raise SystemExit("Hits tree has no entries with edep > 0")

    key = np.stack([h["runID"][keep].astype(np.int64),
                    h["eventID"][keep].astype(np.int64),
                    h["level1ID"][keep].astype(np.int64)], axis=1)
    t, e = h["time"][keep], h["edep"][keep]

    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    key, t, e = key[order], t[order], e[order]
    new = np.ones(len(key), dtype=bool)
    new[1:] = np.any(key[1:] != key[:-1], axis=1)
    gid = np.cumsum(new) - 1
    n_groups = int(gid[-1]) + 1

    esum = np.bincount(gid, weights=e, minlength=n_groups)
    et = np.bincount(gid, weights=e * t, minlength=n_groups)
    return key[new], et / esum                      # keys are already sorted


def lookup(sorted_keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Row-wise lookup of `query` in lexicographically sorted `sorted_keys`."""
    # Pack the three int64 columns into one sortable key. Module ids are < 2^16
    # and event ids < 2^31 in every run this is used for, but check rather than
    # assume: a silent wrap here would scramble the times.
    for arr in (sorted_keys, query):
        if arr[:, 1].max() >= 1 << 31 or arr[:, 2].max() >= 1 << 16:
            raise ValueError("eventID or moduleID too large to pack")
    def pack(a):
        return (a[:, 0].astype(np.int64) << 47) | (a[:, 1].astype(np.int64) << 16) \
            | a[:, 2].astype(np.int64)
    keys, q = pack(sorted_keys), pack(query)
    idx = np.searchsorted(keys, q)
    ok = (idx < keys.size) & (keys[np.clip(idx, 0, keys.size - 1)] == q)
    if not ok.all():
        raise SystemExit(f"{(~ok).sum():,} of {q.size:,} coincidence arms have no "
                         "matching Hits group -- the ROOT file is inconsistent")
    return values[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, nargs="+",
                   help="GATE ROOT file(s) with a Coincidences tree; several are "
                        "concatenated into one listmode")
    p.add_argument("--out", type=Path, required=True, help="output list-mode .npz")
    p.add_argument("--ctr-ps", type=float, default=40.0,
                   help="coincidence timing resolution FWHM in ps, i.e. the FWHM of "
                        "the dt error (NOT the single-detector resolution). "
                        "0 disables blurring. Default 40.")
    p.add_argument("--time-source", choices=("readout", "energy-weighted"),
                   default="energy-weighted",
                   help="base time before blurring (default: energy-weighted)")
    p.add_argument("--energy-window", type=float, nargs=2, default=(459.9, 562.1),
                   metavar=("LO", "HI"),
                   help="keV window applied to both singles (default: the alpha=1 window)")
    p.add_argument("--keep-randoms", action="store_true",
                   help="do not require eventID1 == eventID2")
    p.add_argument("--no-cross-panel", action="store_true",
                   help="do not require the two singles to be on different plates")
    p.add_argument("--lut-json", type=Path, default=HERE / "scanner_lut.json",
                   help="LUT sidecar, for the module -> panel map")
    p.add_argument("--seed", type=int, default=20260909)
    return p.parse_args()


def process_file(path: Path, args, panel, n_crystals, e_lo, e_hi):
    """One ROOT file -> (detector_ids, dt_before_blurring, cut counts, n_emitted).

    Everything that depends on eventID happens inside this function, because
    GATE restarts eventID at 0 in every file: comparing or joining them across
    files would silently mismatch arms once the run is chunked.
    """
    with uproot.open(path) as f:
        a = f["Coincidences"].arrays(COINC_BRANCHES, library="np")
        n_emitted = emitted_primaries(f)
        hit_keys, hit_times = (energy_weighted_times(f)
                               if args.time_source == "energy-weighted" else (None, None))

    prompts = a["eventID1"].size
    keep = (np.ones(prompts, dtype=bool) if args.keep_randoms
            else a["eventID1"] == a["eventID2"])
    counts = [prompts, int(keep.sum())]
    for side in ("1", "2"):
        keep &= (a[f"energy{side}"] >= e_lo) & (a[f"energy{side}"] <= e_hi)
    counts.append(int(keep.sum()))
    if not args.no_cross_panel:
        keep &= panel[a["level1ID1"]] != panel[a["level1ID2"]]
    counts.append(int(keep.sum()))

    detector_ids = np.stack(
        [a[f"level1ID{s}"][keep] * n_crystals + a[f"level2ID{s}"][keep] for s in ("1", "2")],
        axis=1,
    ).astype(np.int32)

    if args.time_source == "energy-weighted":
        run = a["runID"][keep].astype(np.int64)
        arms = []
        for s in ("1", "2"):
            query = np.stack([run, a[f"eventID{s}"][keep].astype(np.int64),
                              a[f"level1ID{s}"][keep].astype(np.int64)], axis=1)
            arms.append(lookup(hit_keys, hit_times, query))
        t1, t2 = arms
    else:
        t1, t2 = a["time1"][keep], a["time2"][keep]

    return detector_ids, t1 - t2, np.asarray(counts, dtype=np.int64), n_emitted


def main() -> None:
    args = parse_args()
    panel, n_crystals = load_panel_map(args.lut_json)
    e_lo, e_hi = (e / 1000.0 for e in args.energy_window)      # keV -> MeV, as in ROOT
    rng = np.random.default_rng(args.seed)

    ids, dts = [], []
    totals = np.zeros(4, dtype=np.int64)
    n_emitted = 0
    n_emitted_known = True
    for i, path in enumerate(args.root, 1):
        f_ids, f_dt, f_counts, f_emitted = process_file(
            path, args, panel, n_crystals, e_lo, e_hi)
        ids.append(f_ids)
        dts.append(f_dt)
        totals += f_counts
        if f_emitted is None:
            n_emitted_known = False
        else:
            n_emitted += f_emitted
        if len(args.root) > 1:
            print(f"  [{i}/{len(args.root)}] {path.name}: "
                  f"{f_counts[0]:,} prompts -> {f_counts[3]:,} kept "
                  f"(running total {totals[3]:,})", flush=True)

    detector_ids = np.concatenate(ids)
    dt_clean = np.concatenate(dts)
    del ids, dts
    prompts, trues, in_window, cross = totals
    if cross == 0:
        raise SystemExit("no coincidences survived the cuts")

    # Blur once, over the whole concatenated set, so the noise is one draw from
    # one generator rather than a per-chunk sequence.
    sigma_arm = args.ctr_ps * 1e-12 * FWHM_TO_SIGMA / np.sqrt(2.0)
    tof = dt_clean
    if sigma_arm > 0:
        tof = tof + rng.normal(0.0, sigma_arm, tof.size) - rng.normal(0.0, sigma_arm, tof.size)
    tof = tof.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        detector_ids=detector_ids,
        tof_seconds=tof,
        ctr_ps=args.ctr_ps,
        time_source=args.time_source,
        energy_window_kev=np.asarray(args.energy_window),
        n_emitted_primaries=n_emitted if n_emitted_known else -1,
        n_prompts=prompts,
        seed=args.seed,
        source_file=np.asarray([str(p) for p in args.root]),
    )

    print(f"\nsource files   {len(args.root)}")
    if n_emitted_known:
        print(f"emitted        {n_emitted:,} primary events "
              f"(back-to-back: {2 * n_emitted:,} gammas)")
    print(f"prompts        {prompts:,}")
    print(f"trues          {trues:,}  ({100 * trues / prompts:.2f}%)")
    print(f"in energy win  {in_window:,}  ({100 * in_window / prompts:.2f}%)")
    print(f"cross-panel    {cross:,}  ({100 * cross / prompts:.2f}%)   <- written")
    if n_emitted_known and n_emitted:
        print(f"sensitivity    {100 * cross / n_emitted:.4f}% of emitted annihilations")

    # The intrinsic dt spread is NOT zero (depth of interaction, scatter, and
    # the sorter emitting the earlier single as arm 1), so the width of the
    # written dt is the CTR and that spread in quadrature -- report both rather
    # than one number that could be read as the CTR.
    rms_before, rms_after = dt_clean.std() * 1e12, tof.astype(np.float64).std() * 1e12
    print(f"\ntime source    {args.time_source}")
    print(f"dt before CTR  mean {dt_clean.mean() * 1e12:+.2f} ps  "
          f"rms {rms_before:.2f} ps   (geometry + DOI + scatter only)")
    print(f"CTR imposed    {args.ctr_ps:g} ps FWHM on dt "
          f"= {sigma_arm * 1e12:.2f} ps sigma independently on each arm")
    print(f"dt written     mean {tof.mean() * 1e12:+.2f} ps  rms {rms_after:.2f} ps  "
          f"= total FWHM {2.3548 * rms_after:.1f} ps")
    if rms_after > rms_before:
        print(f"               quadrature check: "
              f"sqrt({rms_after:.2f}^2 - {rms_before:.2f}^2) x 2.3548 = "
              f"{2.3548 * np.sqrt(rms_after**2 - rms_before**2):.1f} ps "
              f"vs {args.ctr_ps:g} ps requested")
    print(f"\nSaved {args.out} ({args.out.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
