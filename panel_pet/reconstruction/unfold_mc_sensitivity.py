#!/usr/bin/env python3
"""Unfold the Monte-Carlo voxel sensitivity map into a plain, full-FOV .npy.

scripts/sensitivity.py stores the map FOLDED: because the scanner is mirror
symmetric about x=0, y=0 and z=0, it histograms |x|,|y|,|z| into a single
positive octant (100^3 @ 2 mm covering 0..200 mm), which gives 8x the counts
per voxel. reconstruction/recon_mlem.py --mc-sensitivity-map wants an ordinary
array on a zero-centred grid, so mirror it back out here, once, rather than in
every array task.

The unfolded grid has 2n voxels per axis with centres at +-(1,3,...,199) mm,
which is exactly pytomography's convention: origin = (-N/2 + 0.5)*dr = -199 mm.

    python reconstruction/unfold_mc_sensitivity.py \
        --npz results/voxel_sensitivity.npz --alpha 1 \
        --out results/sensitivity/mc_sens_alpha1_200x200x200_2.0mm.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unfold(octant: np.ndarray) -> np.ndarray:
    """Mirror a positive-octant map into the full volume.

    Folded voxel i covers [i*d, (i+1)*d] with centre (i+0.5)*d, so the negative
    side is the reverse of the same array -- there is no shared centre plane to
    double-count.
    """
    full = octant
    for axis in range(3):
        full = np.concatenate([np.flip(full, axis=axis), full], axis=axis)
    return full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, required=True, help="voxel_sensitivity.npz")
    p.add_argument("--alpha", type=int, default=1, choices=(1, 2, 3),
                   help="Energy-window index. 1 = 459.9-562.1 keV, which is what "
                        "make_listmode.py applies by default -- match them.")
    p.add_argument("--out", type=Path, required=True, help="output .npy")
    args = p.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    key = f"sensitivity_alpha{args.alpha}"
    if key not in z:
        raise KeyError(f"{args.npz} has no {key}; found {[k for k in z.files]}")
    octant = np.asarray(z[key], dtype=np.float64)
    voxel = float(z["voxel_mm"])
    folded = bool(z["folded"])

    if folded:
        full = unfold(octant)
    else:
        full = octant
        print("map is already unfolded; passing through")

    shape = tuple(int(s) for s in full.shape)
    dr = [voxel] * 3
    origin = ((-np.asarray(shape) / 2.0 + 0.5) * np.asarray(dr)).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, full.astype(np.float32))
    args.out.with_suffix(".json").write_text(json.dumps({
        "source_npz": str(args.npz.resolve()),
        "alpha": args.alpha,
        "folded_source_shape": [int(s) for s in octant.shape],
        "shape": list(shape),
        "voxel_size_mm": dr,
        "object_origin_mm": origin,
        "fov_mm": [s * voxel for s in shape],
        "note": "Monte-Carlo (GATE) voxel sensitivity, mirrored out of the folded octant.",
    }, indent=2) + "\n")

    # Folding asserts mirror symmetry, so the unfolded map must be exactly
    # symmetric -- a cheap check that the mirroring did not slip by a voxel.
    for axis, name in enumerate("xyz"):
        asym = np.abs(full - np.flip(full, axis=axis)).max()
        print(f"  mirror symmetry about {name}=0: max |diff| = {asym:.3e}")

    print(f"Unfolded {octant.shape} -> {shape} @ {voxel} mm "
          f"(FOV {[s*voxel for s in shape]} mm)")
    print(f"Saved {args.out}")
    print(f"Saved {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
