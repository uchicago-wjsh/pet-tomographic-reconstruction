#!/usr/bin/env python3
"""Sample the Vision 600 cylinder acquisition into a list-mode .npz.

The Vision 600 counterpart of make_listmode.py. Same campaign shape
(root/cyl_v600 holds 2400 chunks of 0.25 s = 600 s, each an independent
acquisition of the same source), same cuts, same output contract -- but the
scanner is a cylindricalPET, which changes two things:

  * DETECTOR IDS. The ROOT branches are rsectorID/moduleID/submoduleID/
    crystalID, not the panel scanner's level1ID/level2ID. IDs are built with
    pytomography's get_detector_ids_from_trans_axial_ids from the SAME `info`
    dict that vision600_lut.py used for the LUT, so the two cannot disagree.

  * THE GEOMETRIC CUT. There are no panels, so "the two singles must be on
    different plates" has no meaning. The ring analogue is a minimum RSECTOR
    SEPARATION: coincidences within the same or an adjacent rsector subtend
    almost no FOV, are dominated by inter-detector scatter, and are routinely
    excluded. --min-rsector-difference 0 keeps everything.

Applied to every coincidence, in order:
  * trues only       eventID1 == eventID2, dropping randoms
  * energy window    both singles, in keV
  * rsector cut      |rsector1 - rsector2| (cyclic) >= --min-rsector-difference
  * TOF blurring     digitizer_vision600.mac has no timeResolution module, so
                     the times are exact; dt is smeared with the requested
                     coincidence timing resolution (sigma = CTR_FWHM / 2.355
                     applied to dt directly -- pass the CTR, not the
                     single-detector resolution)

Output keys match reconstruct_mlem_pytomography.py: `detector_ids` (N,2 flat
LUT ids) and `tof_seconds` (= time1 - time2).

    python reconstruction/make_listmode_vision600.py --seconds 60 --ctr-ps 283 \
        --out results/listmode_v600/lm_60s_ctr283ps.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import uproot

HERE = Path(__file__).resolve().parent
BRANCHES = ["eventID1", "eventID2", "energy1", "energy2", "time1", "time2",
            "rsectorID1", "moduleID1", "submoduleID1", "crystalID1",
            "rsectorID2", "moduleID2", "submoduleID2", "crystalID2"]
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def load_info(lut_json: Path) -> dict:
    """The geometry dict vision600_lut.py stored alongside the LUT."""
    meta = json.loads(lut_json.read_text())
    if "info" not in meta:
        raise KeyError(f"{lut_json} has no 'info' block; rebuild with vision600_lut.py")
    return meta["info"]


def detector_ids(a, side: str, info: dict) -> np.ndarray:
    """Flat LUT detector IDs for one arm, via pytomography's own mapping."""
    from pytomography.io.PET.shared import get_detector_ids_from_trans_axial_ids

    t = lambda k: torch.as_tensor(a[f"{k}ID{side}"].astype(np.int64))
    rs, mo, sm, cr = t("rsector"), t("module"), t("submodule"), t("crystal")
    ids = get_detector_ids_from_trans_axial_ids(
        cr % info["crystalTransNr"], sm % info["submoduleTransNr"],
        mo % info["moduleTransNr"], rs % info["rsectorTransNr"],
        cr // info["crystalTransNr"], sm // info["submoduleTransNr"],
        mo // info["moduleTransNr"], rs // info["rsectorTransNr"],
        info,
    )
    return ids.numpy().astype(np.int64)


def read_chunk(path: Path, info: dict, e_lo: float, e_hi: float, min_rsec: int):
    """One ROOT chunk -> (detector_ids, dt_seconds, cumulative cut counts)."""
    a = uproot.open(path)["Coincidences"].arrays(BRANCHES, library="np")

    keep = a["eventID1"] == a["eventID2"]
    counts = [keep.size, int(keep.sum())]
    for side in ("1", "2"):
        keep &= (a[f"energy{side}"] >= e_lo) & (a[f"energy{side}"] <= e_hi)
    counts.append(int(keep.sum()))

    if min_rsec > 0:
        n_rs = int(info["rsectorTransNr"])
        d = np.abs(a["rsectorID1"].astype(np.int64) - a["rsectorID2"].astype(np.int64))
        keep &= np.minimum(d, n_rs - d) >= min_rsec      # cyclic distance around the ring
    counts.append(int(keep.sum()))

    sub = {k: v[keep] for k, v in a.items()}
    det = np.stack([detector_ids(sub, "1", info), detector_ids(sub, "2", info)], axis=1)
    return det.astype(np.int32), sub["time1"] - sub["time2"], np.asarray(counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-dir", type=Path, default=HERE.parent / "root" / "cyl_v600")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=None,
                   help="acquisition time to sample (default: all of it)")
    p.add_argument("--total-seconds", type=float, default=600.0)
    p.add_argument("--ctr-ps", type=float, default=283.0,
                   help="coincidence timing resolution FWHM in ps (NOT the single-detector "
                        "resolution). 0 disables blurring.")
    p.add_argument("--energy-window", type=float, nargs=2, default=(459.9, 562.1),
                   metavar=("LO", "HI"),
                   help="keV window on both singles (default: the alpha=1 window, "
                        "matching the panel campaign)")
    p.add_argument("--min-rsector-difference", type=int, default=0,
                   help="Reject coincidences whose rsectors are closer than this around "
                        "the ring. 0 keeps everything.")
    p.add_argument("--lut-json", type=Path, default=HERE / "vision600_lut.json")
    p.add_argument("--seed", type=int, default=20260804)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.root_dir.glob("*.root"))
    if not files:
        raise SystemExit(f"no ROOT files in {args.root_dir}")

    info = load_info(args.lut_json)
    seconds_per_file = args.total_seconds / len(files)
    n_files = len(files) if args.seconds is None else round(args.seconds / seconds_per_file)
    n_files = int(np.clip(n_files, 1, len(files)))
    rng = np.random.default_rng(args.seed)
    chosen = [files[i] for i in sorted(rng.choice(len(files), n_files, replace=False))]
    sampled_seconds = n_files * seconds_per_file
    print(f"Sampling {n_files} of {len(files)} chunks ({seconds_per_file:g} s each) "
          f"-> {sampled_seconds:g} s of acquisition")
    print(f"Scanner: {info['NrCrystalsPerRing']} crystals/ring x {info['NrRings']} rings")

    e_lo, e_hi = (e / 1000.0 for e in args.energy_window)      # keV -> MeV, as in ROOT
    ids, dts, totals = [], [], np.zeros(4, dtype=np.int64)
    for i, path in enumerate(chosen, 1):
        d, dt, c = read_chunk(path, info, e_lo, e_hi, args.min_rsector_difference)
        ids.append(d); dts.append(dt); totals += c
        if i % 25 == 0 or i == n_files:
            print(f"  [{i}/{n_files}] kept {totals[3]:,} of {totals[0]:,} prompts", flush=True)

    detector = np.concatenate(ids)
    tof = np.concatenate(dts)
    del ids, dts

    sigma = args.ctr_ps * 1e-12 * FWHM_TO_SIGMA
    if sigma > 0:
        tof = tof + rng.normal(0.0, sigma, tof.size)
    tof = tof.astype(np.float32)

    n_det = int(info["NrCrystalsPerRing"]) * int(info["NrRings"])
    if detector.size and detector.max() >= n_det:
        raise ValueError(f"detector id {detector.max()} >= n_det {n_det}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, detector_ids=detector, tof_seconds=tof,
             sampled_seconds=sampled_seconds, n_files=n_files, ctr_ps=args.ctr_ps,
             energy_window_kev=np.asarray(args.energy_window),
             min_rsector_difference=args.min_rsector_difference,
             seed=args.seed, scanner="vision600",
             source_files=np.asarray([f.name for f in chosen]))

    prompts, trues, in_window, kept = totals
    print(f"\nprompts        {prompts:,}")
    print(f"trues          {trues:,}  ({100*trues/prompts:.1f}%)")
    print(f"in energy win  {in_window:,}  ({100*in_window/prompts:.1f}%)")
    print(f"rsector cut    {kept:,}  ({100*kept/prompts:.1f}%)   <- written")
    print(f"rate           {kept/sampled_seconds:,.0f} events/s")
    print(f"TOF blurring   CTR {args.ctr_ps:g} ps FWHM -> sigma {sigma*1e12:.1f} ps")
    print(f"Saved {args.out} ({args.out.stat().st_size/2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
