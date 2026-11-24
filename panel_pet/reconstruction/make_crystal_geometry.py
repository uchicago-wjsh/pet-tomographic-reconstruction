#!/usr/bin/env python3
"""Export oriented crystal boxes from this repository's GATE panel geometry."""
import argparse
from pathlib import Path

import numpy as np

import scanner_lut
from pet_projector import CrystalGeometry


def geometry_from_panel_macro(macro, placements=None):
    geom = scanner_lut.load_geometry(Path(macro), placements)
    depth_axis = geom.depth_axis
    face_axes = [axis for axis in range(3) if axis != depth_axis]
    axes = np.repeat(geom.module_rotations[:, :, face_axes + [depth_axis]],
                     geom.n_crystals_per_module, axis=0).copy()
    centers = scanner_lut.build_scanner_lut(geom)
    # Positive depth must point away from the FOV, irrespective of each
    # module's passive placement rotation.
    orientation = np.sign(np.einsum("ni,ni->n", axes[:, :, 2], centers))
    if (orientation == 0).any():
        raise ValueError("cannot infer entrance-face orientation for a crystal centered on the depth-zero plane")
    axes[:, :, 2] *= orientation[:, None]
    sizes = geom.crystal_size_mm[face_axes + [depth_axis]]
    return CrystalGeometry.from_centers(centers, axes, sizes)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", type=Path, default=root / "macros/geometry.mac")
    parser.add_argument("--placements", type=Path)
    parser.add_argument("--lut", type=Path, help="Optional geometric-center LUT to validate detector indexing")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    geometry = geometry_from_panel_macro(args.macro, args.placements)
    if args.lut is not None:
        lut = np.load(args.lut, allow_pickle=False)
        if lut.shape != geometry.centers_mm.shape or not np.allclose(lut, geometry.centers_mm, rtol=0, atol=1e-3):
            raise ValueError("LUT does not match geometric crystal centers in detector-ID order")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    geometry.save(args.out)
    print(f"Saved {geometry.n_detectors:,} crystal frames to {args.out}; sha256={geometry.fingerprint}")


if __name__ == "__main__":
    main()
