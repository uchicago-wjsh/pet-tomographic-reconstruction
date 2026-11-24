#!/usr/bin/env python3
"""Build the NEMA IEC body phantom as a voxelised GATE activity map.

Writes data/nema_activity.mhd/.raw (Bq per voxel), read by
macros/source_nema.mac exactly as make_activity_map.py's output is read by
macros/source_weighted.mac.

GEOMETRY (NEMA NU 2 / IEC 61675-1 image-quality phantom, to the standard)
  * body      The transaxial outline is NOT an ellipse. It is built from
                - a semicircle of radius 150 mm forming the top (y >= 0),
                - a flat bottom 140 mm wide at y = -80 mm,
                - two quarter circles of radius 80 mm joining them, centred
                  at (+-70, 0).
              giving 300 mm wide x 230 mm tall, extruded 180 mm along z.
              Note the outline is ASYMMETRIC about y = 0: it spans y from
              -80 to +150 mm. The origin is the centre of the top semicircle,
              which is also the lung axis and the sphere-ring centre, so the
              phantom axis sits on the scanner axis.
  * lung      central cylinder, 50 mm diameter, full length, ZERO activity.
  * spheres   six, INNER diameters 10, 13, 17, 22, 28 and 37 mm, centres on a
              114.4 mm diameter circle in the z = 0 plane, at 60 deg spacing,
              with the 37 mm sphere at 0 deg (+x) and the rest following
              counter-clockwise in ascending order: 37, 10, 13, 17, 22, 28.
              The ROTATION matters here even though the cyclic order would
              look the same either way: this scanner's resolution is
              anisotropic (four panels, not a ring), so which sphere faces
              which panel changes its measured contrast.

ACTIVITY
  Spheres carry the SAME concentration as the cylinder phantom
  (macros/source_cylinder.mac: 2642 Bq/cm3), so lesion contrast is measured
  against a source strength already characterised in this project. Background
  is spheres/ratio, with the NEMA-standard 4:1 default.

  The map is built by SUPERSAMPLING each voxel, so a voxel straddling a sphere
  boundary gets the correct partial volume rather than a hard in/out decision.
  Without it the 10 mm sphere -- 10 voxels across at 1 mm -- would have a
  volume error of several percent, which is exactly the quantity a contrast
  recovery measurement reports.

NO ATTENUATING MATERIAL is created. This project's FOV is vacuum by design
(see macros/geometry.mac), so the phantom is an activity distribution only:
no water, no attenuation, no phantom scatter. That keeps these runs directly
comparable to the cylinder acquisitions and leaves the existing sensitivity
maps valid. A physical water phantom would change the physics and require
attenuation correction, which this pipeline does not do.

    python scripts/make_nema_phantom.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPHERE_DIAMETERS_MM = (10.0, 13.0, 17.0, 22.0, 28.0, 37.0)
SPHERE_RING_DIAMETER_MM = 114.4
# Angle of each sphere in SPHERE_DIAMETERS_MM order. The 37 mm sphere sits at
# 0 deg, so 10 mm starts at 60 deg -- i.e. index k is at 60*(k+1) deg.
SPHERE_ANGLES_DEG = tuple(60.0 * (k + 1) % 360.0 for k in range(6))

TOP_RADIUS_MM = 150.0          # upper semicircle
CORNER_RADIUS_MM = 80.0        # lower corner fillets
FLAT_HALF_WIDTH_MM = TOP_RADIUS_MM - CORNER_RADIUS_MM      # 70 -> 140 mm flat
BODY_LENGTH_MM = 180.0
BODY_WIDTH_MM = 2 * TOP_RADIUS_MM                          # 300
BODY_HEIGHT_MM = TOP_RADIUS_MM + CORNER_RADIUS_MM          # 230
LUNG_DIAMETER_MM = 50.0
CYLINDER_CONC_BQ_PER_CM3 = 2642.0          # macros/source_cylinder.mac CONC


def inside_outline(X, Y):
    """True inside the NEMA transaxial outline. Origin = semicircle centre."""
    upper = (Y >= 0) & (X ** 2 + Y ** 2 <= TOP_RADIUS_MM ** 2)
    ax = np.abs(X)
    lower_flat = (Y < 0) & (Y >= -CORNER_RADIUS_MM) & (ax <= FLAT_HALF_WIDTH_MM)
    lower_round = (Y < 0) & (ax > FLAT_HALF_WIDTH_MM) & \
        ((ax - FLAT_HALF_WIDTH_MM) ** 2 + Y ** 2 <= CORNER_RADIUS_MM ** 2)
    return upper | lower_flat | lower_round


def outline_area_mm2() -> float:
    return (0.5 * np.pi * TOP_RADIUS_MM ** 2                      # top semicircle
            + 2 * FLAT_HALF_WIDTH_MM * CORNER_RADIUS_MM           # flat bottom block
            + 0.5 * np.pi * CORNER_RADIUS_MM ** 2)                # two quarter circles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voxel", type=float, default=1.0, help="map voxel size in mm")
    p.add_argument("--sphere-conc", type=float, default=CYLINDER_CONC_BQ_PER_CM3,
                   help="hot-sphere concentration in Bq/cm3")
    p.add_argument("--ratio", type=float, default=4.0,
                   help="sphere-to-background concentration ratio (NEMA default 4:1)")
    p.add_argument("--sub", type=int, default=3,
                   help="supersampling per axis for partial volume at boundaries")
    p.add_argument("--outdir", type=Path, default=Path("data"))
    p.add_argument("--pad-mm", type=float, default=10.0,
                   help="margin around the body, so the map's edge is not the phantom's")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v = args.voxel

    # Bounding box of the phantom. NOT symmetric in y: the outline runs from
    # -80 (flat bottom) to +150 (top of the semicircle), so a symmetric box
    # would be centred on the wrong plane and waste 70 mm of map.
    lo = np.array([-TOP_RADIUS_MM, -CORNER_RADIUS_MM, -BODY_LENGTH_MM / 2]) - args.pad_mm
    hi = np.array([TOP_RADIUS_MM, TOP_RADIUS_MM, BODY_LENGTH_MM / 2]) + args.pad_mm
    n = np.ceil((hi - lo) / v).astype(int)
    n += n % 2                                   # even along each axis
    centre = (lo + hi) / 2.0
    lo = centre - n * v / 2.0
    origin = lo                                  # lower corner, for GATE setPosition

    # voxel centres
    ax = [lo[a] + (np.arange(n[a]) + 0.5) * v for a in range(3)]
    off = ((np.arange(args.sub) + 0.5) / args.sub - 0.5) * v

    a_sphere = args.sphere_conc / 1000.0         # Bq/cm3 -> Bq/mm3
    a_bg = a_sphere / args.ratio
    conc = np.zeros(tuple(n), dtype=np.float64)  # Bq per mm3, filled by supersampling

    ring_r = SPHERE_RING_DIAMETER_MM / 2.0
    centres = [(ring_r * np.cos(np.deg2rad(a)), ring_r * np.sin(np.deg2rad(a)), 0.0)
               for a in SPHERE_ANGLES_DEG]

    for du in off:
        for dv in off:
            for dw in off:
                X = (ax[0] + du)[:, None, None]
                Y = (ax[1] + dv)[None, :, None]
                Z = (ax[2] + dw)[None, None, :]
                inside_body = inside_outline(X, Y) & (np.abs(Z) <= BODY_LENGTH_MM / 2)
                in_lung = (np.hypot(X, Y) <= LUNG_DIAMETER_MM / 2) & (np.abs(Z) <= BODY_LENGTH_MM / 2)
                sample = np.where(inside_body & ~in_lung, a_bg, 0.0)
                for (cx, cy, cz), d in zip(centres, SPHERE_DIAMETERS_MM):
                    in_sph = ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= (d / 2) ** 2
                    sample = np.where(in_sph & inside_body, a_sphere, sample)
                conc += sample
    conc /= args.sub ** 3

    act = conc * v ** 3                          # Bq per voxel, what GATE reads
    args.outdir.mkdir(parents=True, exist_ok=True)
    act.astype(np.float32).T.tofile(args.outdir / "nema_activity.raw")   # x fastest
    (args.outdir / "nema_activity.mhd").write_text(
        "ObjectType = Image\nNDims = 3\nBinaryData = True\n"
        "BinaryDataByteOrderMSB = False\nCompressedData = False\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        "Offset = 0 0 0\nCenterOfRotation = 0 0 0\n"
        "AnatomicalOrientation = RAI\n"
        f"ElementSpacing = {v:g} {v:g} {v:g}\n"
        f"DimSize = {n[0]} {n[1]} {n[2]}\n"
        "ElementType = MET_FLOAT\nElementDataFile = nema_activity.raw\n")

    # --- report, and check the digitised volumes against the analytic ones ---
    vol_vox = v ** 3 / 1000.0                    # cm3
    tot = float(act.sum())
    sph_vol = sum((4 / 3) * np.pi * (d / 2) ** 3 for d in SPHERE_DIAMETERS_MM) / 1000.0
    body_vol = outline_area_mm2() * BODY_LENGTH_MM / 1000.0
    lung_vol = np.pi * (LUNG_DIAMETER_MM / 2) ** 2 * BODY_LENGTH_MM / 1000.0
    bg_vol = body_vol - lung_vol - sph_vol
    analytic = bg_vol * args.sphere_conc / args.ratio + sph_vol * args.sphere_conc

    print(f"outline    : semicircle r={TOP_RADIUS_MM:g}, corners r={CORNER_RADIUS_MM:g}, "
          f"flat {2*FLAT_HALF_WIDTH_MM:g} mm  -> {BODY_WIDTH_MM:g} x {BODY_HEIGHT_MM:g} mm")
    print(f"spheres    : " + ", ".join(f"{d:g}mm@{a:g}deg"
                                       for d, a in zip(SPHERE_DIAMETERS_MM, SPHERE_ANGLES_DEG)))
    print(f"map        : {tuple(int(x) for x in n)} @ {v:g} mm  "
          f"= {n[0]*v:g} x {n[1]*v:g} x {n[2]*v:g} mm")
    print(f"             x [{lo[0]:g},{lo[0]+n[0]*v:g}]  y [{lo[1]:g},{lo[1]+n[1]*v:g}]  "
          f"z [{lo[2]:g},{lo[2]+n[2]*v:g}] mm")
    print(f"GATE corner: {origin[0]:g} {origin[1]:g} {origin[2]:g} mm  "
          f"(use in source_nema.mac setPosition)")
    print(f"concentration: spheres {args.sphere_conc:g} Bq/cm3 "
          f"(= the cylinder phantom), background {args.sphere_conc/args.ratio:g} "
          f"({args.ratio:g}:1)")
    print()
    print(f"{'compartment':>12s} {'analytic cm3':>13s} {'digitised cm3':>14s} {'error':>8s}")
    # Measure the digitised volumes by ACTIVITY, not by counting "fully inside"
    # voxels: with supersampling a boundary voxel holds a fraction of the sphere
    # concentration, and excluding those undercounts every sphere by its whole
    # surface shell (-10% for this set). Each voxel contributes
    # (conc - a_bg)/(a_sphere - a_bg) of a sphere-volume.
    frac_sph = np.clip((conc - a_bg) / (a_sphere - a_bg), 0.0, 1.0)
    frac_sph[conc <= 0] = 0.0
    dig_sph = float(frac_sph.sum()) * vol_vox
    # Background by ACTIVITY as well: counting (conc > 0) voxels would treat a
    # partially-filled outline-boundary voxel as full and overstate it by ~half
    # the surface shell (+0.6% here).
    dig_bg = (tot - dig_sph * args.sphere_conc) / (args.sphere_conc / args.ratio)
    print(f"{'spheres':>12s} {sph_vol:>13.2f} {dig_sph:>14.2f} "
          f"{100*(dig_sph-sph_vol)/sph_vol:>+7.2f}%")
    print(f"{'background':>12s} {bg_vol:>13.2f} {dig_bg:>14.2f} "
          f"{100*(dig_bg-bg_vol)/bg_vol:>+7.2f}%")
    print()
    print(f"total activity: {tot/1e6:.4f} MBq   (analytic {analytic/1e6:.4f} MBq, "
          f"{100*(tot-analytic)/analytic:+.2f}%)")
    print(f"  for reference the cylinder phantom is 33.2 MBq, so the count rate "
          f"here is ~{100*tot/33.2e6:.0f}% of it")

    (args.outdir / "nema_activity.json").write_text(json.dumps({
        "voxel_mm": v, "dim": [int(x) for x in n],
        "gate_set_position_mm": [float(x) for x in origin],
        "sphere_diameters_mm": list(SPHERE_DIAMETERS_MM),
        "sphere_angles_deg": list(SPHERE_ANGLES_DEG),
        "sphere_ring_diameter_mm": SPHERE_RING_DIAMETER_MM,
        "sphere_centres_mm": [[float(c) for c in cc] for cc in centres],
        "outline": {"top_radius_mm": TOP_RADIUS_MM,
                    "corner_radius_mm": CORNER_RADIUS_MM,
                    "flat_width_mm": 2 * FLAT_HALF_WIDTH_MM,
                    "y_range_mm": [-CORNER_RADIUS_MM, TOP_RADIUS_MM]},
        "body_mm": [BODY_WIDTH_MM, BODY_HEIGHT_MM, BODY_LENGTH_MM],
        "lung_diameter_mm": LUNG_DIAMETER_MM,
        "sphere_conc_bq_per_cm3": args.sphere_conc,
        "background_conc_bq_per_cm3": args.sphere_conc / args.ratio,
        "ratio": args.ratio, "total_activity_bq": tot,
        "attenuating_material": False,
    }, indent=2) + "\n")
    print(f"\nwrote {args.outdir/'nema_activity.mhd'} (+ .raw, {act.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
