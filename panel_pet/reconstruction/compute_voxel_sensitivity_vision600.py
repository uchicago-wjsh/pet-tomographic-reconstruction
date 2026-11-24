#!/usr/bin/env python3
"""Precompute the MLEM sensitivity image (A^T 1) for the Biograph Vision 600.

The Vision 600 counterpart of compute_voxel_sensitivity.py. The backprojection
itself is identical -- it imports compute_sensitivity from that module, so the
numerics stay byte-for-byte the same -- and only the LOR SET differs:

  * The panel scanner enumerates CROSS-PANEL pairs, because a LOR joining two
    crystals of one flat panel carries no FOV information.
  * A full ring has no panels. Every unordered pair of distinct crystals is a
    valid LOR; the only conventional exclusion is a minimum RSECTOR separation,
    since coincidences within the same or an adjacent rsector subtend almost no
    FOV and are dominated by inter-detector scatter.

SCALE. 60,800 detectors give C(60800,2) = 1.85e9 pairs -- 30 GB just for the
int64 ids, larger than the panel scanner's 1.0e9. Exact enumeration is
therefore off the table in practice; --n-sensitivity-lors samples instead. 5e7
LORs give ~1e4 deposits per voxel on a 2 mm grid, which is ample.

As in the panel version the map is saved BEFORE pytomography's
`norm_BP < 1e-7 -> 1e7` substitution, so the file stays a faithful A^T 1; the
reconstruction applies that clamp on load exactly as the class does.

    python reconstruction/compute_voxel_sensitivity_vision600.py \
        --grid-size 112 --grid-size-z 128 --voxel-size-mm 2.0 \
        --n-sensitivity-lors 50000000 \
        --out results/sensitivity_v600/sens_v600.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pytomography

from compute_voxel_sensitivity import compute_sensitivity
from reconstruct_mlem_pytomography import object_origin, sensitivity_geometry_key

HERE = Path(__file__).resolve().parent


def rsector_of(detector_id: np.ndarray, info: dict) -> np.ndarray:
    """Rsector index of a flat detector id.

    get_detector_ids_from_trans_axial_ids packs the id as
    ring * n_per_ring + within_ring, and within_ring counts crystal, then
    submodule, then module, then rsector -- so the rsector is the within-ring
    index divided by the crystals each rsector contributes.
    """
    n_per_ring = int(info["NrCrystalsPerRing"])
    per_rsector = (int(info["crystalTransNr"]) * int(info["submoduleTransNr"])
                   * int(info["moduleTransNr"]))
    return (detector_id % n_per_ring) // per_rsector


def build_ring_sensitivity_pairs(n_det: int, info: dict, n_pairs: int, seed: int,
                                 min_rsector_difference: int):
    """Uniform random sample of valid ring LORs (i != j, rsector cut applied)."""
    if n_pairs <= 0:
        total = n_det * (n_det - 1) // 2
        raise SystemExit(
            f"Exact enumeration would need {total:,} pairs "
            f"({total*16/2**30:.0f} GiB of ids). Pass --n-sensitivity-lors N."
        )
    rng = np.random.default_rng(seed)
    n_rs = int(info["rsectorTransNr"])
    out = np.empty((n_pairs, 2), dtype=np.int64)
    filled = 0
    while filled < n_pairs:
        need = n_pairs - filled
        a = rng.integers(0, n_det, int(need * 1.4) + 16)
        b = rng.integers(0, n_det, a.size)
        ok = a != b
        if min_rsector_difference > 0:
            d = np.abs(rsector_of(a, info) - rsector_of(b, info))
            ok &= np.minimum(d, n_rs - d) >= min_rsector_difference
        a, b = a[ok][:need], b[ok][:need]
        out[filled:filled + a.size, 0] = np.minimum(a, b)
        out[filled:filled + a.size, 1] = np.maximum(a, b)
        filled += a.size
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lut", type=Path, default=HERE / "vision600_lut.npy")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid-size", type=int, required=True)
    p.add_argument("--grid-size-z", type=int, default=None)
    p.add_argument("--voxel-size-mm", type=float, required=True)
    p.add_argument("--voxel-size-z-mm", type=float, default=None)
    p.add_argument("--n-sensitivity-lors", type=int, default=50_000_000)
    p.add_argument("--sensitivity-seed", type=int, default=20260723)
    p.add_argument("--min-rsector-difference", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"pytomography {pytomography.__version__} on device: {pytomography.device}")

    lut_np = np.load(args.lut).astype(np.float32)
    lut = torch.from_numpy(lut_np)
    info = json.loads(args.lut.with_suffix(".json").read_text())["info"]
    n_det = int(lut.shape[0])

    gz = args.grid_size_z if args.grid_size_z is not None else args.grid_size
    vz = args.voxel_size_z_mm if args.voxel_size_z_mm is not None else args.voxel_size_mm
    shape = (args.grid_size, args.grid_size, gz)
    dr = (args.voxel_size_mm, args.voxel_size_mm, vz)

    ids = build_ring_sensitivity_pairs(n_det, info, args.n_sensitivity_lors,
                                       args.sensitivity_seed, args.min_rsector_difference)
    sens_ids = torch.from_numpy(ids).long()
    weights = torch.ones(sens_ids.shape[0], dtype=torch.float32)
    total = n_det * (n_det - 1) // 2

    print(f"Grid: {shape} @ {dr} mm (FOV {[round(s*d,1) for s,d in zip(shape,dr)]} mm)")
    print(f"Scanner: {n_det:,} detectors, {info['NrCrystalsPerRing']} x {info['NrRings']}")
    print(f"Sensitivity LORs: {len(sens_ids):,} sampled of {total:,} possible "
          f"(min rsector difference {args.min_rsector_difference})")

    sensitivity = compute_sensitivity(lut, sens_ids, weights, shape, dr,
                                      n_splits=args.n_splits)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, sensitivity)
    uncovered = float(np.mean(sensitivity < 1e-7))
    args.out.with_suffix(".json").write_text(json.dumps({
        "sensitivity_file": str(args.out.resolve()),
        "scanner": "Biograph Vision 600",
        "shape": list(shape), "voxel_size_mm": list(dr),
        "object_origin_mm": object_origin(shape, dr).tolist(),
        "lut": str(args.lut.resolve()), "n_detectors": n_det,
        "n_sensitivity_lors": int(len(sens_ids)),
        "total_possible_pairs": int(total),
        "sensitivity_exact": False,
        "min_rsector_difference": args.min_rsector_difference,
        "sensitivity_seed": args.sensitivity_seed,
        "psf_fwhm_mm": None, "clamp_applied": False,
        "geometry_key": sensitivity_geometry_key(shape, dr, n_det),
        "stats": {"min": float(sensitivity.min()), "max": float(sensitivity.max()),
                  "mean": float(sensitivity.mean()),
                  "frac_below_clamp_threshold": uncovered},
    }, indent=2) + "\n")

    print(f"\nSaved {args.out}  {sensitivity.shape} {sensitivity.dtype}")
    print(f"min/median/max: {sensitivity.min():.6g} / "
          f"{np.median(sensitivity):.6g} / {sensitivity.max():.6g}")
    print(f"Voxels below the 1e-7 clamp threshold: {100*uncovered:.2f}%")


if __name__ == "__main__":
    main()
