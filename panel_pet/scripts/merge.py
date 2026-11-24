#!/usr/bin/env python
"""
Merge per-task voxel count files from sensitivity.py into the final result.

Production runs histogram each task's coincidences immediately and delete the
ROOT file (at full statistics the ROOT output would be several TB), so the
campaign leaves behind many small .npz count files. This sums them and
reports the sensitivity and its precision.

    python scripts/merge.py 'results/parts/*.npz' -o results/sensitivity.npz
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ALPHAS = (1, 2, 3)
E_REF_KEV, E_RES = 511.0, 0.10


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="+", help="per-task .npz files (globs allowed)")
    p.add_argument("-o", "--out", default="results/sensitivity.npz")
    p.add_argument("--target", type=int, default=10000,
                   help="counts per voxel required (default 10000 -> 1%%)")
    p.add_argument("--rebin", type=int, default=1,
                   help="combine NxNxN voxels (e.g. 5 turns 2 mm into 10 mm), "
                        "trading resolution for counts per voxel")
    args = p.parse_args()

    paths: list[str] = []
    for pattern in args.parts:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])
    paths = [q for q in paths if Path(q).exists()]
    if not paths:
        print("no .npz parts found", file=sys.stderr)
        return 1

    total = {a: None for a in ALPHAS}
    emitted = 0.0
    # Emitted decays accumulate SEPARATELY per emission profile, because the
    # uniform runs and the weighted runs spread their decays differently. The
    # denominator is the sum of what each profile delivered to each voxel.
    emitted_by_profile: dict[str, float] = {}
    meta = None
    for q in paths:
        d = np.load(q)
        e = float(d["emitted_total"])
        emitted += e
        prof = str(d["weights_file"]) if "weights_file" in d.files else ""
        emitted_by_profile[prof] = emitted_by_profile.get(prof, 0.0) + e
        if meta is None:
            meta = {k: d[k] for k in ("voxel_mm", "fov_mm", "folded")}
        elif (float(d["voxel_mm"]) != float(meta["voxel_mm"])
              or bool(d["folded"]) != bool(meta["folded"])):
            print(f"{q}: voxel/fold settings differ from the first part",
                  file=sys.stderr)
            return 1
        for a in ALPHAS:
            c = d[f"counts_alpha{a}"].astype(np.int64)
            total[a] = c if total[a] is None else total[a] + c

    folded = bool(meta["folded"])
    voxel = float(meta["voxel_mm"])

    if args.rebin > 1:
        n0 = total[1].shape[0]
        if n0 % args.rebin:
            print(f"--rebin {args.rebin} does not divide the {n0}^3 grid",
                  file=sys.stderr)
            return 1
        m = n0 // args.rebin
        r = args.rebin
        for a in ALPHAS:
            total[a] = (total[a].reshape(m, r, m, r, m, r)
                                .sum(axis=(1, 3, 5)))
        voxel *= r
        print(f"rebinned {r}x{r}x{r}: {n0}^3 -> {m}^3 voxels "
              f"({voxel/r:g} mm -> {voxel:g} mm)")

    grid = total[1].shape          # NOT a cube in general
    n_vox_total = int(np.prod(grid))
    # N_v = sum over profiles of (that profile's share of voxel v) x (its decays)
    emitted_per_voxel = np.zeros(total[1].shape, dtype=np.float64)
    for prof, e in sorted(emitted_by_profile.items()):
        if prof:
            w = np.load(prof)
            if args.rebin > 1:
                r2 = args.rebin
                a2, b2, c2 = (d // r2 for d in w.shape)
                w = w.reshape(a2, r2, b2, r2, c2, r2).sum(axis=(1, 3, 5))
            emitted_per_voxel += (w / w.sum()) * e
            print(f"  profile {Path(prof).name:<28} {e:>18,.0f} decays (weighted)")
        else:
            emitted_per_voxel += e / float(n_vox_total)
            print(f"  profile {'uniform':<28} {e:>18,.0f} decays")

    print(f"merged {len(paths)} part(s)")
    print(f"voxel size            : {voxel:g} mm"
          f"{'  (x/y/z symmetry folded)' if folded else ''}")
    print(f"grid                  : {'x'.join(map(str, grid))} = "
          f"{n_vox_total:,} voxels")
    print(f"emitted annihilations : {emitted:,.0f}")
    print(f"emitted per voxel     : {emitted_per_voxel.min():,.1f} .. "
          f"{emitted_per_voxel.max():,.1f}")

    # A campaign whose source fills only part of the grid -- the axial end-slab
    # runs emit nothing for |z| < 200 mm, so the already-measured centre is not
    # re-simulated -- leaves emitted_per_voxel = 0 outside its support. Those
    # bins hold no information; reporting over them would pin "min counts" at 0
    # forever and hide real progress. Restrict every statistic to the support.
    # When every voxel is emitted into, this is the whole grid and nothing below
    # changes.
    support = emitted_per_voxel > 0
    n_support = int(support.sum())
    if n_support < support.size:
        print(f"support               : {n_support:,} of {support.size:,} bins "
              f"({100.0*n_support/support.size:.1f}%) have emission; "
              f"statistics below cover those only")

    print(f"\n{'window':>8} {'range (keV)':>18} {'detected':>16} {'efficiency':>11} "
          f"{'sens mean':>11} {'sens min':>10} {'sens max':>10} "
          f"{'min counts':>11} {'err@min':>9}")
    print("-" * 116)
    results = {}
    for a in ALPHAS:
        c = total[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            sens = np.where(support,
                            c / np.where(support, emitted_per_voxel, 1.0), np.nan)
        half = a * E_RES * E_REF_KEV
        cmin = int(c[support].min()) if n_support else 0
        err = (100.0 / np.sqrt(cmin)) if cmin > 0 else float("inf")
        print(f"alpha={a:<2d} [{E_REF_KEV-half:7.1f},{E_REF_KEV+half:7.1f}] "
              f"{int(c.sum()):16,d} {c.sum()/emitted*100:10.3f}% "
              f"{sens[support].mean()*100:10.4f}% {sens[support].min()*100:9.4f}% "
              f"{sens[support].max()*100:9.4f}% {cmin:11,d} {err:8.2f}%")
        results[f"counts_alpha{a}"] = c.astype(np.int64)
        results[f"sensitivity_alpha{a}"] = sens

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emitted_total=emitted,
                        emitted_per_voxel=emitted_per_voxel,
                        # fov_mm is a scalar for a cube and a 3-vector when the axes
        # differ; float() would crash on the latter
        voxel_mm=voxel,
        fov_mm=(float(meta["fov_mm"]) if np.ndim(meta["fov_mm"]) == 0
                else np.asarray(meta["fov_mm"], dtype=float)),
                        folded=folded, alphas=np.array(ALPHAS), **results)
    print(f"\nwrote {out}")

    print(f"\n--- progress toward {args.target:,} counts in EVERY voxel ---")
    for a in ALPHAS:
        c = results[f"counts_alpha{a}"]
        cmin = int(c[support].min()) if n_support else 0
        frac = (np.mean(c[support] >= args.target) * 100.0) if n_support else 0.0
        print(f"  alpha={a}: worst voxel {cmin:,} counts; "
              f"{frac:.1f}% of voxels already at target", end="")
        if cmin > 0:
            need = emitted * (args.target / cmin)
            print(f"; need ~{need:.3g} emitted total "
                  f"({need/emitted:.1f}x current)")
        else:
            print("; some voxels still empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
