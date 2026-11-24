#!/usr/bin/env python3
"""Precompute the MLEM sensitivity image (A^T 1) for the panel scanner.

The output ``.npy`` is loadable by ``reconstruct_mlem_pytomography.py
--sensitivity-map``, which substitutes it for the map ``PETLMSystemMatrix``
would otherwise build in its constructor. That build is the single most
expensive step of a reconstruction and depends only on the geometry -- not on
the data -- so computing it once and reusing it across a duration x CTR sweep
saves it nine times over.

WHAT IS COMPUTED
----------------
Exactly what ``PETLMSystemMatrix._backward_full()`` computes, by the same
route: a NON-TOF Joseph backprojection of a vector of ones (times any
per-pair weights, plus ``pytomography.delta``) along the cross-panel
sensitivity LORs. Non-TOF is correct and not an approximation: summed over
all TOF bins a normalised TOF kernel integrates to 1.

Verified against the class: reproducing ``_backward_full`` by hand matches its
internal ``norm_BP`` to ~1e-7 relative (float32 accumulation order).

THE CLAMP IS DELIBERATELY NOT APPLIED
-------------------------------------
``PETLMSystemMatrix.__init__`` follows ``_backward_full()`` with

    norm_BP[norm_BP < 1e-7] = 1e7

which freezes uncovered voxels at ~0 for every iteration. This script saves
the map BEFORE that substitution, so the file stays a faithful A^T 1 that can
be inspected for coverage holes. The reconstruction applies the clamp on load,
exactly as the class does, so the reconstructed image is unchanged either way.

Use the SAME --grid-size/--voxel-size/--n-sensitivity-lors/--sensitivity-seed
as the reconstruction. The reconstruction validates the geometry recorded in
the sidecar JSON and refuses a mismatch.

    python reconstruction/compute_voxel_sensitivity.py \
        --lut reconstruction/scanner_lut.npy \
        --grid-size 112 --grid-size-z 208 --voxel-size-mm 2.0 \
        --n-sensitivity-lors 50000000 \
        --out results/sensitivity/sens_112x112x208_2mm.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pytomography
import parallelproj

from reconstruct_mlem_pytomography import (
    build_sensitivity_pairs,
    load_lut,
    load_panel_groups,
    object_origin,
    sensitivity_geometry_key,
)

HERE = Path(__file__).resolve().parent


def compute_sensitivity(
    lut: torch.Tensor,
    sensitivity_ids: torch.Tensor,
    weights: torch.Tensor,
    shape: tuple[int, int, int],
    dr: tuple[float, float, float],
    *,
    n_splits: int = 20,
    psf_fwhm_mm: float | None = None,
) -> np.ndarray:
    """Reproduce ``PETLMSystemMatrix._backward_full()`` exactly.

    Same projector, same chunking, same ``+ pytomography.delta``, same
    transform ordering -- so the result is interchangeable with the map the
    class builds internally.
    """
    origin = object_origin(shape, dr)
    device = pytomography.device

    transforms = []
    if psf_fwhm_mm is not None:
        from pytomography.transforms.shared import GaussianFilter
        from pytomography.metadata import ObjectMeta

        transform = GaussianFilter(psf_fwhm_mm, n_sigmas=3.0)
        transform.configure(ObjectMeta(dr=dr, shape=shape), None)
        transforms.append(transform)

    norm_bp = 0
    n_done = 0
    for proj_sub, ids_sub in zip(
        torch.tensor_split(weights, n_splits), torch.tensor_split(sensitivity_ids, n_splits)
    ):
        norm_bp = norm_bp + parallelproj.joseph3d_back(
            lut[ids_sub[:, 0]].to(device),
            lut[ids_sub[:, 1]].to(device),
            shape,
            origin,
            dr,
            proj_sub.to(device) + pytomography.delta,
            num_chunks=4,
        )
        n_done += len(ids_sub)
        print(f"  backprojected {n_done:,}/{len(sensitivity_ids):,} LORs", flush=True)

    # Same order as _backward_full: transforms applied in reverse.
    for transform in transforms[::-1]:
        norm_bp = transform.backward(norm_bp)

    return norm_bp.cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lut", type=Path, default=HERE / "scanner_lut.npy",
                   help="scanner_lut .npy (n_det, 3) in mm")
    p.add_argument("--out", type=Path, required=True, help="output sensitivity .npy")
    # grid -- must match the reconstruction
    p.add_argument("--grid-size", type=int, required=True, help="voxels along x and y")
    p.add_argument("--grid-size-z", type=int, default=None, help="voxels along z (default: grid-size)")
    p.add_argument("--voxel-size-mm", type=float, required=True)
    p.add_argument("--voxel-size-z-mm", type=float, default=None)
    # sensitivity LOR set -- must match the reconstruction
    p.add_argument("--n-sensitivity-lors", type=int, default=0,
                   help="0 = all cross-panel LORs (exact); >0 = proportional random sample. "
                        "This scanner has 1.0e9 cross-panel LORs, so 0 needs ~15 GiB for the "
                        "ids alone plus a long backprojection -- size the job accordingly.")
    p.add_argument("--sensitivity-seed", type=int, default=20260723)
    p.add_argument("--sensitivity-weights", type=Path, default=None,
                   help="Optional .npy of per-pair weights aligned with the sensitivity pairs "
                        "(e.g. flood efficiencies). Default is unit (purely geometric).")
    p.add_argument("--panel-ids", type=Path, default=None,
                   help="Optional .npy of one panel label per detector; overrides the grouping "
                        "from the LUT's .json sidecar.")
    p.add_argument("--psf-fwhm-mm", type=float, default=None,
                   help="Image-space Gaussian PSF FWHM. Must match the reconstruction's "
                        "--psf-fwhm-mm, since the class folds it into the sensitivity too.")
    p.add_argument("--n-splits", type=int, default=20,
                   help="Backprojection chunks (matches _backward_full's default of 20).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"pytomography {pytomography.__version__} on device: {pytomography.device}")

    lut = load_lut(args.lut)
    gz = args.grid_size_z if args.grid_size_z is not None else args.grid_size
    vz = args.voxel_size_z_mm if args.voxel_size_z_mm is not None else args.voxel_size_mm
    shape = (args.grid_size, args.grid_size, gz)
    dr = (args.voxel_size_mm, args.voxel_size_mm, vz)

    panels = load_panel_groups(lut, args.lut, args.panel_ids)
    print("Panels: " + ", ".join(f"{label} ({ids.size})" for label, ids in panels))

    sens_ids_np, total_cross = build_sensitivity_pairs(
        panels, args.n_sensitivity_lors, args.sensitivity_seed
    )
    sens_ids = torch.from_numpy(sens_ids_np).long()
    if args.sensitivity_weights is not None:
        weights = torch.from_numpy(np.load(args.sensitivity_weights).astype(np.float32))
        if weights.shape[0] != sens_ids.shape[0]:
            raise ValueError("--sensitivity-weights length must match the sensitivity pairs")
    else:
        weights = torch.ones(sens_ids.shape[0], dtype=torch.float32)

    exact = len(sens_ids) == total_cross
    print(f"Grid: {shape} @ {dr} mm (FOV {[round(s*d,1) for s,d in zip(shape,dr)]} mm)")
    print(f"Sensitivity LORs: {len(sens_ids):,} of {total_cross:,} cross-panel "
          f"({'exact' if exact else 'sampled'})")

    sensitivity = compute_sensitivity(
        lut, sens_ids, weights, shape, dr,
        n_splits=args.n_splits, psf_fwhm_mm=args.psf_fwhm_mm,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, sensitivity)

    uncovered = float(np.mean(sensitivity < 1e-7))
    meta = {
        "sensitivity_file": str(args.out.resolve()),
        "shape": list(shape),
        "voxel_size_mm": list(dr),
        "object_origin_mm": object_origin(shape, dr).tolist(),
        "lut": str(args.lut.resolve()),
        "n_detectors": int(lut.shape[0]),
        "panels": {label: int(ids.size) for label, ids in panels},
        "n_sensitivity_lors": int(len(sens_ids)),
        "total_cross_panel_lors": int(total_cross),
        "sensitivity_exact": bool(exact),
        "sensitivity_seed": args.sensitivity_seed,
        "sensitivity_weights": (
            str(args.sensitivity_weights) if args.sensitivity_weights else None
        ),
        "psf_fwhm_mm": args.psf_fwhm_mm,
        "clamp_applied": False,
        "geometry_key": sensitivity_geometry_key(shape, dr, int(lut.shape[0])),
        "stats": {
            "min": float(sensitivity.min()),
            "max": float(sensitivity.max()),
            "mean": float(sensitivity.mean()),
            "median": float(np.median(sensitivity)),
            "frac_below_clamp_threshold": uncovered,
        },
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nSaved sensitivity: {args.out}  {sensitivity.shape} {sensitivity.dtype}")
    print(f"Saved metadata:    {meta_path}")
    print(f"min/median/max: {sensitivity.min():.6g} / "
          f"{np.median(sensitivity):.6g} / {sensitivity.max():.6g}")
    print(f"Voxels below the 1e-7 clamp threshold: {100 * uncovered:.2f}%")
    if uncovered > 0.5:
        print("WARNING: over half the grid has no sensitivity coverage. Either the grid "
              "extends well beyond the scanner FOV, or too few sensitivity LORs were used "
              "-- those voxels are frozen at ~0 for every MLEM iteration.")


if __name__ == "__main__":
    main()
