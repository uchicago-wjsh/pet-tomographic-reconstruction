#!/usr/bin/env python3
"""Per-crystal front-face frame, so a LOR endpoint can be sampled over the face.

``scanner_lut.npy`` holds one point per crystal -- its geometric CENTRE. Using
that as the LOR endpoint pins every LOR to a lattice: crystal centres sit on the
4 mm crystal pitch, so the midpoint of any pair lands on a 2 mm lattice, and for
a source at the isocentre (where the midpoint IS the event's transverse
position) that lattice prints straight through into the image as a 2 mm comb.

Sampling the endpoint uniformly over the crystal's front FACE breaks that. This
script writes the frame needed to do it:

    front_face_center  (n_det, 3)  centre of the 4 x 4 mm face presented to the
                                   isocentre -- i.e. the LUT centre moved half a
                                   crystal depth inward
    axis1, axis2       (n_det, 3)  the two in-plane unit vectors, in world
                                   coordinates (the module rotation applied to
                                   the two non-depth local axes)
    half1, half2                   half-extents along those axes, mm

so a sample is

    p = front_face_center + axis1 * U(-half1, half1) + axis2 * U(-half2, half2)

Note what this models and what it does not. It replaces "the interaction was at
the crystal centre" with "the photon entered somewhere on the front face",
which removes the lattice and is the usual first-order fix. It is still not the
true interaction point: that is distributed through the crystal DEPTH with an
exponential attenuation profile, which this deliberately does not model -- every
sample sits on the entrance plane.

    python reconstruction/make_frontface_basis.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import scanner_lut as sl

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--macro", type=Path, default=HERE.parent / "macros" / "geometry.mac")
    p.add_argument("--placements", type=Path, default=None)
    p.add_argument("--out", type=Path, default=HERE / "scanner_frontface.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    geom = sl.load_geometry(args.macro, args.placements)

    n_cry = geom.n_crystals_per_module
    depth = geom.depth_axis
    plane = [a for a in range(3) if a != depth]          # the two in-face axes

    # Module rotations take local -> world; the in-face basis is the rotation
    # applied to the two local axes that are not the depth axis.
    e = np.eye(3)
    ax1_mod = geom.module_rotations @ e[plane[0]]        # (n_mod, 3)
    ax2_mod = geom.module_rotations @ e[plane[1]]
    axis1 = np.repeat(ax1_mod, n_cry, axis=0)
    axis2 = np.repeat(ax2_mod, n_cry, axis=0)

    face = sl.build_scanner_lut(geom, depth_mode="frontface").astype(np.float64)
    half1 = float(geom.crystal_size_mm[plane[0]]) / 2.0
    half2 = float(geom.crystal_size_mm[plane[1]]) / 2.0

    np.savez(args.out,
             front_face_center=face.astype(np.float32),
             axis1=axis1.astype(np.float32), axis2=axis2.astype(np.float32),
             half1=half1, half2=half2,
             n_crystals_per_module=n_cry,
             crystal_size_mm=np.asarray(geom.crystal_size_mm, dtype=np.float64),
             depth_axis=depth)

    centers = sl.build_scanner_lut(geom, depth_mode="center").astype(np.float64)
    inward = centers - face
    print(f"detectors            {face.shape[0]:,}")
    print(f"depth axis (local)   {depth}, in-face axes {plane}, "
          f"face {2*half1:g} x {2*half2:g} mm")
    print(f"face centre is       {np.linalg.norm(inward, axis=1).mean():.3f} mm from the "
          f"crystal centre (half of {geom.crystal_depth_mm:g} mm depth)")
    print(f"axes orthonormal     |a1|={np.linalg.norm(axis1,axis=1).max():.6f} "
          f"|a2|={np.linalg.norm(axis2,axis=1).max():.6f} "
          f"max|a1.a2|={np.abs(np.einsum('ij,ij->i',axis1,axis2)).max():.2e}")
    dep = np.einsum('ij,ij->i', np.cross(axis1, axis2), inward)
    print(f"a1 x a2 parallel to depth: min |dot| {np.abs(dep).min():.4f} "
          f"(should equal half the crystal depth, {geom.crystal_depth_mm/2:g})")
    for label, (lo, hi) in geom.plate_slices().items():
        s = slice(lo * n_cry, (hi + 1) * n_cry)
        rng = [f"{face[s, a].min():+.1f}..{face[s, a].max():+.1f}" for a in range(3)]
        print(f"  {label:<7} face x {rng[0]}  y {rng[1]}  z {rng[2]} mm")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
