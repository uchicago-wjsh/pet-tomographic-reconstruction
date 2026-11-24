#!/usr/bin/env python3
"""Split one list-mode acquisition into N equal, disjoint sub-acquisitions.

The point is to turn a single long scan into N statistically independent
repeats of a shorter one, so that per-voxel mean and variance across the
repeats measure the *noise of a T/N second scan* rather than anything about
the reconstruction algorithm.

Why a contiguous split by event index is the right thing here
------------------------------------------------------------
``make_listmode.py`` concatenates its ROOT chunks in sorted file order, and
every chunk is an independent 1.0 s acquisition of the same static source.
Event order in the npz is therefore chunk order, i.e. acquisition order, so
slice ``k`` of ``N`` equal contiguous slices is the events of (T/N) seconds of
scanning -- the boundaries land inside a chunk rather than exactly on one, but
the chunks are i.i.d. replicas so a partial chunk is no different from a whole
one. The per-chunk event counts are not stored in the npz and the ROOT files
this one was built from have been deleted, so cutting exactly on chunk
boundaries is not available; it would also not be more correct.

The one consequence worth naming: fixing the event COUNT per subgroup makes the
N subgroups multinomial rather than Poisson, which removes the run-to-run
fluctuation of the *total* count. Per voxel that scales the variance by
(1 - p_v) with p_v the fraction of all events landing in that voxel -- here
p_v < 1e-5, so it is a <0.001% effect and irrelevant to the SNR maps. Sorting
by event index rather than shuffling is deliberate: a shuffle would mix all
600 s into every subgroup, which still gives independent subsets but no longer
corresponds to ten consecutive 60 s scans.

    python reconstruction/split_listmode.py \
        results/listmode_nema/lm_nema_600s_ctr40ps.npz \
        --n-groups 10 --out-dir results/listmode_nema_sub60s \
        --prefix lm_nema_sub
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Metadata carried through unchanged: these describe the measurement, not its
# length, so they must not be recomputed per subgroup.
PASSTHROUGH = ("ctr_ps", "energy_window_kev", "seed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("listmode", type=Path, help="input list-mode .npz")
    p.add_argument("--n-groups", type=int, default=10)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="lm_sub")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    z = np.load(args.listmode, allow_pickle=False)
    ids = z["detector_ids"]
    tof = z["tof_seconds"]
    n = ids.shape[0]
    if tof.shape[0] != n:
        raise SystemExit(f"detector_ids has {n} rows but tof_seconds has {tof.shape[0]}")

    total_seconds = float(z["sampled_seconds"])
    seconds_each = total_seconds / args.n_groups
    src = z["source_files"] if "source_files" in z.files else None

    # round(), not floor(): the last group would otherwise absorb the remainder
    # and be up to n_groups-1 events longer than the rest. Here it costs
    # nothing either way, but an unequal group is a silent asymmetry in a
    # variance calculation and those are expensive to notice later.
    edges = np.round(np.linspace(0, n, args.n_groups + 1)).astype(np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.listmode.name}: {n:,} events, {total_seconds:g} s "
          f"-> {args.n_groups} x {seconds_each:g} s")

    for k in range(args.n_groups):
        lo, hi = int(edges[k]), int(edges[k + 1])
        out = args.out_dir / f"{args.prefix}{k:02d}_{seconds_each:g}s_ctr{float(z['ctr_ps']):g}ps.npz"
        payload = {
            "detector_ids": ids[lo:hi],
            "tof_seconds": tof[lo:hi],
            "sampled_seconds": np.float64(seconds_each),
            "n_files": np.int64(round(float(z["n_files"]) / args.n_groups))
                       if "n_files" in z.files else np.int64(0),
            "subgroup": np.int64(k),
            "n_subgroups": np.int64(args.n_groups),
            "parent_listmode": np.asarray(args.listmode.name),
            "event_range": np.asarray([lo, hi], dtype=np.int64),
        }
        for key in PASSTHROUGH:
            if key in z.files:
                payload[key] = z[key]
        if src is not None:
            # Approximate provenance: the chunks this stretch of the
            # acquisition came from, to within the one chunk each boundary
            # falls inside.
            per = len(src) / args.n_groups
            payload["source_files"] = src[int(round(k * per)):int(round((k + 1) * per))]
        np.savez(out, **payload)
        print(f"  [{k}] events {lo:,}..{hi:,}  ({hi - lo:,})  "
              f"-> {out.name}  {out.stat().st_size / 2**20:.0f} MiB")

    print("done")


if __name__ == "__main__":
    main()
