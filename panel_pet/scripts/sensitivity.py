#!/usr/bin/env python
"""
Voxel-wise sensitivity of the four-panel PET scanner.

Reads one or more GATE ROOT files, assigns every detected coincidence to a
2 mm voxel using the KNOWN annihilation position (sourcePosX/Y/Z1), and
divides by the number of annihilations emitted into that voxel.

    sensitivity(voxel) = detected coincidences / emitted annihilations

The source is uniform over the 40 x 40 x 40 cm FOV, so the emitted count per
voxel is (total emitted) / (number of voxels). The total emitted is read from
GATE's `pet_data` tree, not assumed from activity x duration.

Three energy windows are applied, all from the same simulation. With 10%
energy resolution at 511 keV the half-width is alpha * 0.1 * 511 keV:

    alpha = 1 : [459.9, 562.1] keV
    alpha = 2 : [408.8, 613.2] keV
    alpha = 3 : [357.7, 664.3] keV

Both arms of a coincidence must fall inside the window. Random coincidences
are removed by requiring eventID1 == eventID2.

Usage
-----
    python scripts/sensitivity.py output/run*.root -o results/sens.npz

    # fold the 8-fold mirror symmetry of the geometry to get 8x the counts
    # per unique voxel (the scanner is symmetric in x, y and z):
    python scripts/sensitivity.py output/run*.root -o results/sens.npz --fold

    # coarser voxels when 2 mm is not statistically affordable:
    python scripts/sensitivity.py output/run*.root -o results/sens.npz --voxel 10
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import uproot

FOV_MM = 400.0            # 40 cm cube enclosed by the panels
E_REF_KEV = 511.0
E_RES = 0.10              # 10% FWHM at 511 keV
ALPHAS = (1, 2, 3)

BRANCHES = ["eventID1", "eventID2", "energy1", "energy2",
            "sourcePosX1", "sourcePosY1", "sourcePosZ1"]


def energy_window(alpha: int) -> tuple[float, float]:
    """Window half-width is alpha * (resolution * 511 keV)."""
    half = alpha * E_RES * E_REF_KEV
    return E_REF_KEV - half, E_REF_KEV + half


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="GATE ROOT files (globs allowed)")
    p.add_argument("-o", "--out", default="results/sensitivity.npz",
                   help="output .npz")
    p.add_argument("--fov", type=float, nargs="+", default=[FOV_MM],
                   help="FOV side in mm that the source fills and the map "
                        "covers. Give ONE value for a cube, or THREE (x y z) for "
                        "an anisotropic FOV -- the Vision 600 wants 400 400 256, "
                        "matching the panels radially while staying inside its "
                        "+-128.8 mm axial FOV. Legacy help follows: "
                        "covers. Default 400 (the four-panel scanner). The Vision "
                        "600 uses 256, matching its +-128.8 mm axial FOV.")
    p.add_argument("--voxel", type=float, default=2.0,
                   help="voxel size in mm (default 2)")
    p.add_argument("--fold", action="store_true",
                   help="fold the x/y/z mirror symmetry (8x counts per voxel)")
    p.add_argument("--accumulate", action="store_true",
                   help="add these counts into an existing -o file instead of "
                        "overwriting. Lets a job histogram one chunk at a time "
                        "and keep a single small part file, so a job killed "
                        "part-way still leaves valid partial results.")
    p.add_argument("--weights", default=None,
                   help="folded .npy of relative emission weights, for runs "
                        "using a non-uniform (importance-sampled) source. "
                        "Without it the source is assumed uniform.")
    args = p.parse_args()

    paths: list[str] = []
    for pattern in args.roots:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])
    paths = [q for q in paths if Path(q).exists()]
    if not paths:
        print("no input ROOT files found", file=sys.stderr)
        return 1

    fov = np.asarray(args.fov, dtype=float)
    if fov.size == 1:
        fov = np.repeat(fov, 3)
    elif fov.size != 3:
        print("--fov takes 1 value (cube) or 3 (x y z)", file=sys.stderr)
        return 1
    n_vox = np.rint(fov / args.voxel).astype(int)
    half = fov / 2.0
    print(f"{len(paths)} file(s); {args.voxel:g} mm voxels -> "
          f"{'x'.join(map(str, n_vox))} = {int(np.prod(n_vox)):,} voxels over the "
          f"{'x'.join(f'{c/10:g}' for c in fov)} cm FOV")

    # The scanner is mirror-symmetric about x=0, y=0 and z=0, so folding the
    # annihilation positions into the positive octant gives 8x the counts per
    # unique voxel for free. Histogram |x|,|y|,|z| directly on a half-width
    # grid -- that is the fold, done exactly.
    if args.fold:
        if np.any(n_vox % 2):
            print("--fold needs an even voxel count across every axis",
                  file=sys.stderr)
            return 1
        n_bin = n_vox // 2
        edges = [np.linspace(0.0, half[i], n_bin[i] + 1) for i in range(3)]
        print(f"folding x/y/z mirror symmetry -> {int(np.prod(n_bin)):,} unique voxels")
    else:
        n_bin = n_vox
        edges = [np.linspace(-half[i], half[i], n_bin[i] + 1) for i in range(3)]
    shape = tuple(int(v) for v in n_bin)

    counts = {a: np.zeros(shape, dtype=np.int64) for a in ALPHAS}
    emitted_total = 0.0
    n_coinc = n_true = 0

    if args.accumulate and Path(args.out).exists():
        prev = np.load(args.out)
        if prev["counts_alpha1"].shape != shape:
            print(f"{args.out}: shape {prev['counts_alpha1'].shape} != {shape}",
                  file=sys.stderr)
            return 1
        for a in ALPHAS:
            counts[a] += prev[f"counts_alpha{a}"].astype(np.int64)
        emitted_total += float(prev["emitted_total"])
        print(f"accumulating into {args.out} "
              f"({emitted_total:,.0f} decays already there)")

    for path in paths:
        with uproot.open(path) as f:
            emitted_total += float(
                f["pet_data"].arrays(library="np")["total_nb_primaries"][0])
            tree = f["Coincidences"]
            if tree.num_entries == 0:
                continue
            for chunk in tree.iterate(BRANCHES, library="np", step_size="200 MB"):
                n_coinc += len(chunk["eventID1"])
                # true coincidences only: both singles from the same annihilation
                keep = chunk["eventID1"] == chunk["eventID2"]
                n_true += int(keep.sum())
                if not keep.any():
                    continue
                e1 = chunk["energy1"][keep] * 1000.0     # MeV -> keV
                e2 = chunk["energy2"][keep] * 1000.0
                x = chunk["sourcePosX1"][keep]
                y = chunk["sourcePosY1"][keep]
                z = chunk["sourcePosZ1"][keep]

                if args.fold:
                    x, y, z = np.abs(x), np.abs(y), np.abs(z)

                for a in ALPHAS:
                    lo, hi = energy_window(a)
                    sel = (e1 >= lo) & (e1 <= hi) & (e2 >= lo) & (e2 <= hi)
                    if not sel.any():
                        continue
                    h, _ = np.histogramdd(
                        np.stack([x[sel], y[sel], z[sel]], axis=1),
                        bins=edges)
                    counts[a] += h.astype(np.int64)

    if emitted_total <= 0:
        print("no emitted annihilations recorded", file=sys.stderr)
        return 1

    # Emissions per voxel. For a uniform source this is a scalar; for a
    # weighted source it is an array.
    #
    #   full FOV:  N_v = A_v / sum(A_full) * N_total
    #   folded:    bin b pools its 8 mirrors, all with the same A, so
    #              N_b = 8*A_b / sum(A_full) * N_total = A_b / sum_b(A_b) * N_total
    #   (sum_b N_b = N_total, i.e. the weights just redistribute the same decays)
    if args.weights:
        w = np.load(args.weights)
        if w.shape != shape:
            print(f"weights {w.shape} do not match the {shape} grid",
                  file=sys.stderr)
            return 1
        emitted_per_voxel = (w / w.sum()) * emitted_total
    else:
        emitted_per_voxel = emitted_total / float(np.prod(n_bin))
    print(f"\nemitted annihilations : {emitted_total:,.0f}")
    print(f"coincidences in files : {n_coinc:,}  "
          f"(trues {n_true:,}, randoms {n_coinc-n_true:,})")
    _epv = np.asarray(emitted_per_voxel)
    _desc = (f"{float(_epv):,.1f}" if _epv.ndim == 0
             else f"{_epv.min():,.1f} .. {_epv.max():,.1f}  (weighted)")
    print(f"emitted per voxel     : {_desc}"
          f"{'  (8 mirrored voxels combined)' if args.fold else ''}")

    # A source that fills only part of the grid (the axial end-slab runs, which
    # emit nothing for |z| < 200 mm so the measured centre is not re-simulated)
    # leaves emitted_per_voxel = 0 outside its support. Those bins carry no
    # information: report over the SUPPORT only, so "min counts" stays a usable
    # progress signal instead of being pinned at 0 by bins nothing was aimed at.
    # For a scalar or strictly positive emitted_per_voxel the support is the
    # whole grid and every number below is unchanged.
    support = np.broadcast_to(np.asarray(emitted_per_voxel) > 0, shape)
    n_support = int(support.sum())
    if n_support < support.size:
        print(f"support               : {n_support:,} of {support.size:,} bins "
              f"({100.0*n_support/support.size:.1f}%) have emission; "
              f"statistics below cover those only")

    results = {}
    print(f"\n{'window':>8} {'range (keV)':>18} {'detected':>14} {'efficiency':>11} "
          f"{'sens mean':>11} {'sens min':>10} {'min counts':>11} {'err@min':>9}")
    print("-" * 100)
    for a in ALPHAS:
        c = counts[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            sens = np.where(support, c / np.where(support, emitted_per_voxel, 1.0),
                            np.nan)
        det = int(c.sum())
        lo, hi = energy_window(a)
        cmin = int(c[support].min()) if n_support else 0
        err = (1.0 / np.sqrt(cmin) * 100.0) if cmin > 0 else float("inf")
        print(f"alpha={a:<2d} [{lo:7.1f},{hi:7.1f}] {det:14,d} "
              f"{det/emitted_total*100:10.3f}% {sens[support].mean()*100:10.4f}% "
              f"{sens[support].min()*100:9.4f}% {cmin:11,d} {err:8.1f}%")
        results[f"counts_alpha{a}"] = c.astype(np.int32)
        results[f"sensitivity_alpha{a}"] = sens

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        emitted_total=emitted_total,
        emitted_per_voxel=emitted_per_voxel,
        voxel_mm=args.voxel,
        # scalar when cubic (what every existing reader expects),
        # 3-vector only when the axes genuinely differ
        fov_mm=(float(fov[0]) if np.allclose(fov, fov[0]) else fov),
        folded=args.fold,
        weights_file=(args.weights or ""),
        alphas=np.array(ALPHAS),
        **results)
    print(f"\nwrote {out}")

    # How much more simulation would the 1% target need?
    print("\n--- statistics needed for 1% error (10,000 counts) in EVERY voxel ---")
    for a in ALPHAS:
        c = results[f"counts_alpha{a}"]
        cmin = int(c[support].min()) if n_support else 0
        if cmin == 0:
            print(f"  alpha={a}: some voxels still empty; cannot extrapolate reliably")
            continue
        factor = 10000.0 / cmin
        print(f"  alpha={a}: worst voxel has {cmin:,} counts -> need "
              f"{factor:,.0f}x more emitted "
              f"({emitted_total*factor:.3g} annihilations total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
