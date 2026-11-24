#!/usr/bin/env python3
"""Interpolate a folded Monte-Carlo sensitivity map onto a finer voxel grid.

The GATE maps are measured on 2 mm voxels. This resamples one onto a finer grid
(1 mm by default) and writes the same full-FOV .npy + .json sidecar that
reconstruction/unfold_mc_sensitivity.py writes, so recon_mlem.py
--mc-sensitivity-map reads it directly (grids identical -> no resampling).

The interpolation is done in the FOLDED octant, before unfolding:

  * Coarse centre i sits at (i+0.5)*d, fine centre j at (j+0.5)*f, so the
    fractional coarse index of a fine voxel is ((j+0.5)*f)/d - 0.5.
  * At the symmetry planes (x, y, z = 0) the fine voxels nearest 0 fall half a
    coarse voxel short of the first centre. Mirroring about 0 mm puts the
    first coarse value on both sides of the plane, so the scanner's mirror
    symmetry is exactly an edge-hold there -- the same result as interpolating
    the unfolded map across 0.
  * At the outer face of the FOV the last fine voxels lie beyond the last
    coarse centre; they are held at the edge value too, not extrapolated, so
    the 1-3% Poisson noise of the edge bin is not amplified.

Trilinear (order=1) keeps the map positive and matches the resampler in
recon_mlem.py. No counts are invented: this is a smooth interpolant of the
2 mm measurement, not a 1 mm measurement.

    python reconstruction/upsample_mc_sensitivity.py \
        --npz results/slab/sens_extended_5mm.npz --alpha 1 --voxel-mm 1.0 \
        --out results/sensitivity/mc_sens_5mm_ext_alpha1_400x400x480_1.0mm.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from unfold_mc_sensitivity import unfold


def upsample_octant(octant: np.ndarray, coarse_mm: float, fine_mm: float) -> np.ndarray:
    ratio = coarse_mm / fine_mm
    n_fine = [int(round(n * ratio)) for n in octant.shape]
    if not np.allclose([n * fine_mm for n in n_fine],
                       [n * coarse_mm for n in octant.shape]):
        raise ValueError(f"{fine_mm} mm does not tile the {coarse_mm} mm grid exactly")
    axes = [((np.arange(n) + 0.5) * fine_mm) / coarse_mm - 0.5 for n in n_fine]
    out = np.empty(n_fine, dtype=np.float64)
    # One output z-plane at a time keeps the coordinate arrays small.
    # mode="nearest" is both the mirror at 0 and the edge-hold at the outer face.
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
    for k, zc in enumerate(axes[2]):
        coords = np.stack([gx, gy, np.full_like(gx, zc)])
        out[:, :, k] = map_coordinates(octant, coords, order=1, mode="nearest",
                                       prefilter=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, required=True, help="folded sensitivity .npz")
    p.add_argument("--alpha", type=int, default=1, choices=(1, 2, 3),
                   help="Energy-window index (1 = 459.9-562.1 keV, the make_listmode.py default)")
    p.add_argument("--voxel-mm", type=float, default=1.0, help="output voxel size")
    p.add_argument("--out", type=Path, required=True, help="output .npy")
    args = p.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    key = f"sensitivity_alpha{args.alpha}"
    if key not in z:
        raise KeyError(f"{args.npz} has no {key}; found {z.files}")
    if not bool(z["folded"]):
        raise ValueError(f"{args.npz} is not folded; this script expects the octant map")
    octant = np.asarray(z[key], dtype=np.float64)
    coarse = float(z["voxel_mm"])

    fine_octant = upsample_octant(octant, coarse, args.voxel_mm)
    full = unfold(fine_octant)

    shape = tuple(int(s) for s in full.shape)
    dr = [args.voxel_mm] * 3
    origin = ((-np.asarray(shape) / 2.0 + 0.5) * np.asarray(dr)).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, full.astype(np.float32))
    args.out.with_suffix(".json").write_text(json.dumps({
        "source_npz": str(args.npz.resolve()),
        "alpha": args.alpha,
        "folded_source_shape": [int(s) for s in octant.shape],
        "source_voxel_size_mm": coarse,
        "shape": list(shape),
        "voxel_size_mm": dr,
        "object_origin_mm": origin,
        "fov_mm": [s * args.voxel_mm for s in shape],
        "interpolation": "trilinear in the folded octant; mirror at x,y,z=0, "
                         "edge-hold (no extrapolation) at the outer FOV face",
        "note": f"Monte-Carlo (GATE) voxel sensitivity measured at {coarse} mm, "
                f"interpolated to {args.voxel_mm} mm and mirrored out of the folded octant.",
    }, indent=2) + "\n")

    for axis, name in enumerate("xyz"):
        asym = np.abs(full - np.flip(full, axis=axis)).max()
        print(f"  mirror symmetry about {name}=0: max |diff| = {asym:.3e}")
    print(f"  {coarse} mm octant: min {octant.min():.4e}  max {octant.max():.4e}  "
          f"mean {octant.mean():.6e}")
    print(f"  {args.voxel_mm} mm octant: min {fine_octant.min():.4e}  max {fine_octant.max():.4e}  "
          f"mean {fine_octant.mean():.6e}")
    print(f"Upsampled {octant.shape} @ {coarse} mm -> {shape} @ {args.voxel_mm} mm "
          f"(FOV {[s * args.voxel_mm for s in shape]} mm)")
    print(f"Saved {args.out}")
    print(f"Saved {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
