#!/usr/bin/env python3
"""Sample the cylinder acquisition into a list-mode .npz for reconstruction.

root/cylinder holds 2400 ROOT chunks of 0.25 s each = 600 s in total, and
every chunk is an independent acquisition of the same source. Sampling files
is therefore the same as sampling time: ``--seconds 60`` draws 240 chunks.

Each coincidence must survive, in order:

  * trues only       eventID1 == eventID2, which drops the randoms
  * energy window    both singles, in keV. The simulation itself only cut at
                     the alpha=3 window (357.7-664.3 keV), so anything
                     narrower is applied here.
  * cross-panel      the two singles must land on different plates
  * TOF blurring     digitizer.mac has no timeResolution module, so the ROOT
                     times are exact. The time difference is smeared here
                     with the requested coincidence timing resolution (CTR),
                     i.e. sigma = CTR_FWHM / 2.355 applied to dt directly --
                     pass the CTR, not the single-detector resolution.

Output keys match reconstruct_mlem_pytomography.py: ``detector_ids`` (N,2 flat
LUT ids) and ``tof_seconds`` (= time1 - time2, so <= 0 before blurring).

    python reconstruction/make_listmode.py --seconds 60 --ctr-ps 283 \
        --out results/listmode_60s.npz

Memory scales with the events kept (~12 bytes each). 60 s is comfortable;
sampling the full 600 s produces >1e8 events, so run that on a compute node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
BRANCHES = ["eventID1", "eventID2", "energy1", "energy2", "time1", "time2",
            "level1ID1", "level2ID1", "level1ID2", "level2ID2"]
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


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


def read_chunk(path: Path, panel: np.ndarray, n_crystals: int, e_lo: float, e_hi: float):
    """One ROOT chunk -> (detector_ids, dt_seconds, cut counts)."""
    a = uproot.open(path)["Coincidences"].arrays(BRANCHES, library="np")

    keep = a["eventID1"] == a["eventID2"]
    counts = [keep.size, int(keep.sum())]
    for side in ("1", "2"):
        keep &= (a[f"energy{side}"] >= e_lo) & (a[f"energy{side}"] <= e_hi)
    counts.append(int(keep.sum()))
    keep &= panel[a["level1ID1"]] != panel[a["level1ID2"]]
    counts.append(int(keep.sum()))

    detector_ids = np.stack(
        [a[f"level1ID{s}"][keep] * n_crystals + a[f"level2ID{s}"][keep] for s in ("1", "2")],
        axis=1,
    ).astype(np.int32)
    return detector_ids, a["time1"][keep] - a["time2"][keep], np.asarray(counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-dir", type=Path, default=HERE.parent / "root" / "cylinder")
    p.add_argument("--out", type=Path, required=True, help="output list-mode .npz")
    p.add_argument("--seconds", type=float, default=None,
                   help="acquisition time to sample (default: all of it)")
    p.add_argument("--total-seconds", type=float, default=600.0,
                   help="acquisition time the whole directory represents")
    p.add_argument("--ctr-ps", type=float, default=283.0,
                   help="coincidence timing resolution FWHM in ps (NOT the single-detector "
                        "resolution; a 200 ps single-photon digitizer gives CTR ~= 283 ps). "
                        "0 disables blurring.")
    p.add_argument("--energy-window", type=float, nargs=2, default=(459.9, 562.1),
                   metavar=("LO", "HI"),
                   help="keV window applied to both singles (default: the alpha=1 window)")
    p.add_argument("--lut-json", type=Path, default=HERE / "scanner_lut.json",
                   help="LUT sidecar, for the module -> panel map")
    p.add_argument("--seed", type=int, default=20260804)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.root_dir.glob("*.root"))
    if not files:
        raise SystemExit(f"no ROOT files in {args.root_dir}")

    seconds_per_file = args.total_seconds / len(files)
    n_files = len(files) if args.seconds is None else round(args.seconds / seconds_per_file)
    n_files = int(np.clip(n_files, 1, len(files)))
    rng = np.random.default_rng(args.seed)
    chosen = [files[i] for i in sorted(rng.choice(len(files), n_files, replace=False))]
    sampled_seconds = n_files * seconds_per_file
    print(f"Sampling {n_files} of {len(files)} chunks "
          f"({seconds_per_file:g} s each) -> {sampled_seconds:g} s of acquisition")

    panel, n_crystals = load_panel_map(args.lut_json)
    e_lo, e_hi = (e / 1000.0 for e in args.energy_window)      # keV -> MeV, as in ROOT

    ids, dts, totals = [], [], np.zeros(4, dtype=np.int64)
    for i, path in enumerate(chosen, 1):
        chunk_ids, chunk_dt, counts = read_chunk(path, panel, n_crystals, e_lo, e_hi)
        ids.append(chunk_ids)
        dts.append(chunk_dt)
        totals += counts
        if i % 25 == 0 or i == n_files:
            print(f"  [{i}/{n_files}] kept {totals[3]:,} of {totals[0]:,} prompts")

    detector_ids = np.concatenate(ids)
    tof = np.concatenate(dts)
    del ids, dts

    sigma = args.ctr_ps * 1e-12 * FWHM_TO_SIGMA
    if sigma > 0:
        tof = tof + rng.normal(0.0, sigma, tof.size)
    tof = tof.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        detector_ids=detector_ids,
        tof_seconds=tof,
        sampled_seconds=sampled_seconds,
        n_files=n_files,
        ctr_ps=args.ctr_ps,
        energy_window_kev=np.asarray(args.energy_window),
        seed=args.seed,
        source_files=np.asarray([f.name for f in chosen]),
    )

    prompts, trues, in_window, cross = totals
    print(f"\nprompts        {prompts:,}")
    print(f"trues          {trues:,}  ({100 * trues / prompts:.1f}%)")
    print(f"in energy win  {in_window:,}  ({100 * in_window / prompts:.1f}%)")
    print(f"cross-panel    {cross:,}  ({100 * cross / prompts:.1f}%)   <- written")
    print(f"rate           {cross / sampled_seconds:,.0f} events/s")
    print(f"TOF blurring   CTR {args.ctr_ps:g} ps FWHM -> sigma {sigma * 1e12:.1f} ps")
    print(f"Saved {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
