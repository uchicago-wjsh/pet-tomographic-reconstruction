#!/usr/bin/env python3
"""Build the scanner LUT for the Biograph Vision 600 (cylindricalPET).

Unlike the four-panel scanner -- whose genericRepeater geometry has no
equivalent in pytomography and needs reconstruction/scanner_lut.py -- the
Vision 600 is a textbook cylindricalPET, so pytomography builds the LUT itself:

    get_detector_info(macro)  ->  info dict  ->  get_scanner_LUT(info)

The payoff is that reconstruction/make_listmode_vision600.py derives its
detector IDs from the SAME `info` dict via the same
get_detector_ids_from_trans_axial_ids, so the LUT and the listmode indices
cannot disagree.

TWO SNAGS, both handled here:

1. get_detector_info hardcodes the volume names 'crystal', 'submodule',
   'module' and 'rsector' when it looks up repeaters, but macros/vision600.mac
   names them box4..box1 (and geometry_vision600.mac says that file is used as
   provided). Pointing it at vision600.mac raises KeyError: 'crystalTransNr'.
   So a RENAMED COPY is written here and used only for LUT construction; the
   simulation macro is untouched. This is safe because the ROOT branches are
   named from the SYSTEM level (rsectorID/moduleID/...), not from the volume
   names, so the copy changes nothing about the data.

2. mean_interaction_depth sets where along the 20 mm crystal the LUT places
   the interaction point: radius = 433 - 20/2 + depth. The default 0 puts every
   crystal on its inner FACE. The physical mean depth of interaction for 20 mm
   LSO at 511 keV (mu ~= 0.083/mm) is

       1/mu - d*exp(-mu*d)/(1 - exp(-mu*d)) = 12.05 - 4.69 = 7.36 mm

   so 7.4 mm is the default here. (The panel scanner's 5 mm crystals put this
   at 2.34 mm vs a 2.5 mm geometric centre -- a 0.16 mm difference, which is
   why it never mattered there and does here.)

    python reconstruction/vision600_lut.py \
        --macro macros/vision600.mac \
        --out reconstruction/vision600_lut.npy
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RENAME = {"box1": "rsector", "box2": "module", "box3": "submodule", "box4": "crystal"}


def write_renamed_macro(src: Path, dst: Path) -> Path:
    """Copy the scanner macro with box1..box4 renamed to the system level names."""
    text = src.read_text()
    for old, new in RENAME.items():
        text = re.sub(rf"\b{old}\b", new, text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        f"# AUTO-GENERATED from {src} by vision600_lut.py -- do not edit.\n"
        f"# Only get_detector_info() reads this; the simulation uses {src.name}.\n"
        + text
    )
    return dst


def jsonable(info: dict) -> dict:
    out = {}
    for k, v in info.items():
        out[k] = v.item() if isinstance(v, (np.generic,)) else (
            v.tolist() if isinstance(v, np.ndarray) else v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--macro", type=Path, default=HERE.parent / "macros" / "vision600.mac")
    p.add_argument("--out", type=Path, default=HERE / "vision600_lut.npy")
    p.add_argument("--mean-interaction-depth-mm", type=float, default=7.4,
                   help="Depth into the 20 mm crystal at which the LUT places the "
                        "interaction. 0 = inner face, 10 = geometric centre, "
                        "7.4 = physical mean DOI for LSO at 511 keV (default).")
    p.add_argument("--keep-pytomography-z-flip", action="store_true",
                   help="Keep get_scanner_LUT's axial negation instead of undoing it. "
                        "Only for reproducing pytomography's default behaviour; it "
                        "mirrors the reconstruction in z.")
    p.add_argument("--min-rsector-difference", type=int, default=0,
                   help="Stored in the info dict; the listmode builder uses it to "
                        "reject coincidences between neighbouring rsectors.")
    args = p.parse_args()

    from pytomography.io.PET.gate import get_detector_info
    from pytomography.io.PET.shared import get_scanner_LUT

    renamed = write_renamed_macro(args.macro, args.out.parent / "_vision600_renamed.mac")
    info = get_detector_info(
        str(renamed),
        init_volume_name="crystal",
        mean_interaction_depth=args.mean_interaction_depth_mm,
        min_rsector_difference=args.min_rsector_difference,
    )
    lut = get_scanner_LUT(info).numpy().astype(np.float32)

    # pytomography's get_scanner_LUT ends with
    #     XYZ = vstack([XY.T, -Z.unsqueeze(0)]).T   # "for some reason Z needs
    #                                               #  to be negative?"
    # i.e. it negates the axial coordinate on a guess. Checked against this
    # simulation: with the raw LUT, |z_lut - z_gate| has median 128.3 mm while
    # |z_lut + z_gate| has median 0.77 mm (sub-crystal-pitch) and the
    # correlation is -0.9999. So the LUT is mirrored in z relative to GATE.
    # Left alone it reconstructs the image FLIPPED IN Z -- invisible for a
    # z-symmetric phantom like this cylinder, wrong for anything else.
    if not args.keep_pytomography_z_flip:
        lut[:, 2] *= -1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, lut)

    r = np.hypot(lut[:, 0], lut[:, 1])
    meta = {
        "lut_file": str(args.out.resolve()),
        "lut_shape": list(lut.shape),
        "units": "mm",
        "scanner": "Biograph Vision 600",
        "system": "cylindricalPET",
        "gate_macro": str(args.macro.resolve()),
        "renamed_macro": str(renamed.resolve()),
        "mean_interaction_depth_mm": args.mean_interaction_depth_mm,
        "pytomography_z_flip_undone": not args.keep_pytomography_z_flip,
        "n_detectors": int(lut.shape[0]),
        "n_crystals_per_ring": int(info["NrCrystalsPerRing"]),
        "n_rings": int(info["NrRings"]),
        "radius_mm": float(info["radius"]),
        "radial_extent_mm": [float(r.min()), float(r.max())],
        "axial_extent_mm": [float(lut[:, 2].min()), float(lut[:, 2].max())],
        "gate_id_hierarchy": {
            "rsectorID": "rsector", "moduleID": "module",
            "submoduleID": "submodule", "crystalID": "crystal",
        },
        "detector_id_source": (
            "pytomography get_scanner_LUT / get_detector_ids_from_trans_axial_ids"
        ),
        "info": jsonable(info),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Scanner: Biograph Vision 600 (cylindricalPET)")
    print(f"  {info['NrCrystalsPerRing']} crystals/ring x {info['NrRings']} rings "
          f"= {lut.shape[0]:,} detectors")
    print(f"  radius {info['radius']:.1f} mm (mean interaction depth "
          f"{args.mean_interaction_depth_mm} mm)")
    print(f"  radial extent {r.min():.2f} .. {r.max():.2f} mm")
    print(f"  axial extent  {lut[:,2].min():.2f} .. {lut[:,2].max():.2f} mm")
    print(f"Saved {args.out}")
    print(f"Saved {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
