#!/usr/bin/env python3
"""Precompute the TOF forward projection matrix A as a sparse CSR matrix.

Rows are the unique (detector_lo, detector_hi, tof_bin) triples in the
list-mode file; columns are the flattened voxels of the reconstruction grid,
so only voxels IN the grid are ever represented. The output is loadable by
``reconstruct_mlem_pytomography.py --system-matrix-dir``.

HOW A ROW IS OBTAINED
---------------------
Backprojecting a unit value for one (detector pair, TOF bin) returns exactly
that row of A, since ``A^T e_i`` is the i-th row of A. This is verified
bit-exact against the on-the-fly projector: ``A @ f`` reproduces
``joseph3d_fwd_tof_lm(...)`` with 0.0 absolute difference. The matrix is
therefore the same operator the reconstruction uses, not an approximation of
it, and stays consistent with the parallelproj sensitivity map.

SCALE -- READ THIS BEFORE BUILDING
----------------------------------
Measured on this scanner (112x112x208 @ 2 mm, real coincidences):

    CTR      sigma    nnz/row
    40 ps   2.55 mm     29.8
    80 ps   5.09 mm     55.5
   120 ps   7.64 mm     80.6

and crucially the unique-rows-per-event ratio is 1.000: with 51840 detectors
and hundreds of TOF bins, essentially no two events share a row, so
aggregation saves nothing here. A full 600 s acquisition (4.7e8 events) would
therefore need 105-284 GB at 8 bytes per nonzero. The matrix is a
development, validation and small-study tool at this scanner's scale -- NOT a
production substitute for the on-the-fly projector. Use --max-events or
--max-rows, and the printed estimate, to stay inside your disk budget.

    python reconstruction/compute_sparse_system_matrix.py \
        --listmode results/listmode/lm_120s_ctr120ps.npz \
        --lut reconstruction/scanner_lut.npy \
        --grid-size 112 --grid-size-z 208 --voxel-size-mm 2.0 \
        --tof-num-bins 175 --tof-bin-width-ps 30 --tof-fwhm-ps 120 \
        --max-events 200000 --n-workers 8 \
        --output-dir results/system_matrix/120s_ctr120ps
"""
from __future__ import annotations

import os

# MUST precede the parallelproj import. Each row is a single-LOR backprojection,
# so the OpenMP thread team costs far more than the work it parallelises: on a
# 64-core node the default gives 8.1 rows/s, OMP_NUM_THREADS=1 gives 2376 rows/s
# -- a 290x speedup. Rows are parallelised across PROCESSES instead (--n-workers).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

import parallelproj

from reconstruct_mlem_pytomography import (
    load_lut,
    make_tof_meta,
    object_origin,
    row_keys,
    tof_seconds_to_bins,
)

HERE = Path(__file__).resolve().parent
_WORKER: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #
def tof_bbox(x0, x1, shape, origin, dr, signed_bin, tof):
    """Voxel-index bounds [lo, hi) of a TOF row's support, plus a halo.

    A TOF kernel truncated at n_sigmas occupies a short segment of the LOR
    (15-46 mm here) inside a 224x224x416 mm grid, so backprojecting into the
    full grid and scanning it is almost all waste: measured, the
    ``flatnonzero`` over 2.6e6 voxels costs 6 ms/row while the projector call
    itself costs 0.4 ms.

    The kernel centre sits at ``+signed_bin * bin_width + center_offset`` along
    the unit vector from x0 to x1 -- sign verified empirically against the
    weighted centroid of a full-grid backprojection.
    """
    bin_width_mm, sigma, center_offset, n_sigmas, _ = tof
    d = (x1 - x0)[0]
    length = float(np.linalg.norm(d))
    u = d / length
    centre_w = 0.5 * (x0[0] + x1[0]) + u * (
        signed_bin * float(bin_width_mm) + float(center_offset[0])
    )
    half = float(n_sigmas) * float(sigma[0]) + 0.5 * float(bin_width_mm)
    ends = np.stack([centre_w - u * half, centre_w + u * half])

    # A 2-voxel halo covers Joseph's bilinear footprint and any rounding.
    lo_w = ends.min(axis=0) - 2.0 * dr
    hi_w = ends.max(axis=0) + 2.0 * dr
    lo = np.clip(np.floor((lo_w - origin) / dr).astype(np.int64), 0, np.asarray(shape))
    hi = np.clip(np.ceil((hi_w - origin) / dr).astype(np.int64) + 1, 0, np.asarray(shape))
    return lo, hi


def row_voxel_weights(row, lut, shape, origin, dr, tof, *, full_grid: bool = False):
    """Columns and values of one matrix row: A[row, :].

    ``tof`` is None for a non-TOF matrix, else
    (bin_width_mm, sigma_mm, center_offset, n_sigmas, tof_centre_offset_bins).
    ``full_grid=True`` forces the naive whole-grid extraction; it is only used
    by the self-check that proves the bounding box loses nothing.
    """
    x0 = lut[int(row[0])][None, :]
    x1 = lut[int(row[1])][None, :]
    one = np.ones(1, dtype=np.float32)

    if tof is None:
        # A non-TOF LOR crosses the whole grid, so there is nothing to bound.
        image = parallelproj.joseph3d_back(x0, x1, shape, origin, dr, one)
        flat = np.asarray(image, dtype=np.float32).ravel()
        cols = np.flatnonzero(flat).astype(np.int32)
        return cols, flat[cols]

    bin_width_mm, sigma, center_offset, n_sigmas, centre = tof
    # pytomography's forward passes `stored_bin - (num_bins-1)//2`; the row must
    # use the identical signed bin or the kernel sits in the wrong place.
    signed_bin = int(row[2]) - centre
    signed = np.asarray([signed_bin], dtype=np.int32)

    if full_grid:
        sub_shape, sub_origin, lo = shape, origin, None
    else:
        lo, hi = tof_bbox(x0, x1, shape, origin, dr, signed_bin, tof)
        if np.any(hi <= lo):
            return np.empty(0, np.int32), np.empty(0, np.float32)
        sub_shape = tuple(int(v) for v in (hi - lo))
        sub_origin = (origin + lo * dr).astype(np.float32)

    image = parallelproj.joseph3d_back_tof_lm(
        x0, x1, sub_shape, sub_origin, dr, one,
        bin_width_mm, sigma, center_offset, n_sigmas, signed,
    )
    flat = np.asarray(image, dtype=np.float32).ravel()
    local = np.flatnonzero(flat)
    if lo is None:
        return local.astype(np.int32), flat[local]

    # Map sub-grid indices back to flat indices in the full grid. C-order is
    # monotonic under this mapping, so the columns stay sorted -- which the CSR
    # streaming writer relies on.
    i, j, k = np.unravel_index(local, sub_shape)
    cols = ((i + lo[0]) * shape[1] + (j + lo[1])) * shape[2] + (k + lo[2])
    return cols.astype(np.int32), flat[local]


def build_chunk(task):
    start, rows = task
    cols_all, vals_all, counts = [], [], np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        cols, vals = row_voxel_weights(
            row, _WORKER["lut"], _WORKER["shape"], _WORKER["origin"],
            _WORKER["dr"], _WORKER["tof"],
        )
        counts[i] = cols.size
        if cols.size:
            cols_all.append(cols)
            vals_all.append(vals)
    cols = np.concatenate(cols_all) if cols_all else np.empty(0, np.int32)
    vals = np.concatenate(vals_all) if vals_all else np.empty(0, np.float32)
    return start, len(rows), cols, vals, counts


def init_worker(lut, shape, origin, dr, tof):
    _WORKER.update(lut=lut, shape=shape, origin=origin, dr=dr, tof=tof)


def build_and_save_csr(out_path, rows, lut, shape, origin, dr, tof,
                       *, chunk_size, n_workers):
    """Build the CSR streaming straight to disk; only indptr stays resident.

    ``pool.map`` preserves input order and every row's columns come from
    ``flatnonzero`` (already sorted and unique), so chunks can be appended in
    CSR layout as they complete -- no COO triplets, no concatenate/tocsr copies.
    """
    n_rows = len(rows)
    n_cols = int(np.prod(shape))
    idx_tmp = out_path.parent / (out_path.name + ".indices.tmp")
    dat_tmp = out_path.parent / (out_path.name + ".data.tmp")
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    nnz, next_row, done = 0, 0, 0

    tasks = [(s, rows[s:s + chunk_size]) for s in range(0, n_rows, chunk_size)]
    args = (lut, shape, origin, dr, tof)

    with open(idx_tmp, "wb") as fi, open(dat_tmp, "wb") as fd:
        if n_workers == 1:
            init_worker(*args)
            results = map(build_chunk, tasks)
        else:
            pool = ProcessPoolExecutor(max_workers=n_workers,
                                       initializer=init_worker, initargs=args)
            results = pool.map(build_chunk, tasks)

        for start, span, cols, vals, counts in results:
            cols.tofile(fi)
            vals.tofile(fd)
            indptr[next_row + 1: next_row + 1 + span] = nnz + np.cumsum(counts)
            nnz = int(indptr[next_row + span])
            next_row += span
            done += span
            if done % (chunk_size * 20) == 0 or done == n_rows:
                print(f"  rows {done:,}/{n_rows:,}  nnz={nnz:,}", flush=True)
        if n_workers > 1:
            pool.shutdown()

    try:
        if nnz:
            indices = np.memmap(idx_tmp, dtype=np.int32, mode="r", shape=(nnz,))
            data = np.memmap(dat_tmp, dtype=np.float32, mode="r", shape=(nnz,))
        else:
            indices = np.empty(0, np.int32)
            data = np.empty(0, np.float32)
        index_dtype = np.int32 if nnz < np.iinfo(np.int32).max else np.int64
        if index_dtype == np.int64:
            indices = np.asarray(indices, dtype=np.int64)
        matrix = sparse.csr_matrix(
            (data, indices, indptr.astype(index_dtype, copy=False)),
            shape=(n_rows, n_cols),
        )
        sparse.save_npz(out_path, matrix, compressed=True)
    finally:
        matrix = indices = data = None
        for tmp in (idx_tmp, dat_tmp):
            tmp.unlink(missing_ok=True)
    return nnz


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--listmode", type=Path, required=True)
    p.add_argument("--lut", type=Path, default=HERE / "scanner_lut.npy")
    p.add_argument("--output-dir", type=Path, required=True)
    # grid -- must match the reconstruction
    p.add_argument("--grid-size", type=int, required=True)
    p.add_argument("--grid-size-z", type=int, default=None)
    p.add_argument("--voxel-size-mm", type=float, required=True)
    p.add_argument("--voxel-size-z-mm", type=float, default=None)
    # TOF -- must match the reconstruction
    p.add_argument("--no-tof", action="store_true",
                   help="Build a non-TOF matrix even though the listmode has tof_seconds.")
    p.add_argument("--tof-num-bins", type=int, default=201)
    p.add_argument("--tof-bin-width-ps", type=float, default=30.0)
    p.add_argument("--tof-fwhm-ps", type=float, default=283.0)
    p.add_argument("--tof-nsigmas", type=float, default=3.0)
    # scale control
    p.add_argument("--max-events", type=int, default=None,
                   help="Use only the first N events. The main lever on matrix size.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap unique rows after aggregation.")
    p.add_argument("--max-gib", type=float, default=32.0,
                   help="Refuse to build an estimated matrix larger than this. Raise "
                        "deliberately; a full 600 s acquisition is 105-284 GiB.")
    p.add_argument("--verify-bbox", type=int, default=64,
                   help="Check this many random rows against the naive full-grid "
                        "extraction before building, and abort on any mismatch. 0 skips.")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--n-workers", type=int, default=1,
                   help="Worker processes. Each runs single-threaded (see the "
                        "OMP_NUM_THREADS note at the top).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lut = load_lut(args.lut)
    n_det = int(lut.shape[0])
    lm = np.load(args.listmode, allow_pickle=True)
    det = lm["detector_ids"].astype(np.int64)
    if args.max_events is not None:
        det = det[: args.max_events]

    gz = args.grid_size_z if args.grid_size_z is not None else args.grid_size
    vz = args.voxel_size_z_mm if args.voxel_size_z_mm is not None else args.voxel_size_mm
    shape = (args.grid_size, args.grid_size, gz)
    dr = np.asarray([args.voxel_size_mm, args.voxel_size_mm, vz], dtype=np.float32)
    origin = object_origin(shape, dr).astype(np.float32)

    has_tof = (not args.no_tof) and ("tof_seconds" in lm)
    tof_meta = None
    if has_tof:
        tof = lm["tof_seconds"].astype(np.float64)
        if args.max_events is not None:
            tof = tof[: args.max_events]
        tof_meta, bin_width_mm, fwhm_mm = make_tof_meta(
            args.tof_num_bins, args.tof_bin_width_ps, args.tof_fwhm_ps, args.tof_nsigmas
        )

    # Canonicalise events into rows EXACTLY as the reconstruction does.
    id1 = torch.from_numpy(det[:, 0])
    id2 = torch.from_numpy(det[:, 1])
    keep = (id1 != id2).numpy()
    if not keep.all():
        det = det[keep]
        id1, id2 = torch.from_numpy(det[:, 0]), torch.from_numpy(det[:, 1])
        if has_tof:
            tof = tof[keep]
        print(f"Dropped {int((~keep).sum())} same-detector events")
    idx = (id1 < id2).numpy()
    lo = np.where(idx, det[:, 0], det[:, 1])
    hi = np.where(idx, det[:, 1], det[:, 0])
    if has_tof:
        tof_signed = tof.copy()
        tof_signed[~idx] *= -1
        bins = tof_seconds_to_bins(tof_signed, tof_meta).numpy()
    else:
        bins = np.zeros(len(lo), dtype=np.int64)

    keys = row_keys(lo, hi, bins, n_det, args.tof_num_bins if has_tof else 1)
    rows_unique, inverse = np.unique(keys, return_inverse=True)
    n_rows_all = len(rows_unique)
    order = np.argsort(keys, kind="stable")
    first = order[np.searchsorted(keys[order], rows_unique)]
    rows = np.column_stack([lo[first], hi[first], bins[first]]).astype(np.int64)
    counts = np.bincount(inverse, minlength=n_rows_all).astype(np.int64)

    if args.max_rows is not None and n_rows_all > args.max_rows:
        rows = rows[: args.max_rows]
        counts = counts[: args.max_rows]
    n_rows = len(rows)

    # Estimate before committing: nnz/row is set by the TOF kernel width.
    sigma_mm = (args.tof_fwhm_ps * 0.299792458 / 2 / 2.3548) if has_tof else None
    est_nnz_per_row = (
        min(np.prod(shape), 4 * (6 * sigma_mm / float(dr.min())) + 8) if has_tof
        else 4 * float(np.linalg.norm(np.asarray(shape)))
    )
    est_gib = n_rows * est_nnz_per_row * 8 / 2**30
    print(f"Events: {len(det):,}   unique rows: {n_rows_all:,} "
          f"({n_rows_all/max(len(det),1):.3f} rows/event)   building: {n_rows:,}")
    print(f"Grid: {shape} @ {tuple(float(v) for v in dr)} mm -> {int(np.prod(shape)):,} voxels")
    print(f"TOF: {has_tof}" + (f"  sigma={sigma_mm:.2f} mm" if has_tof else ""))
    print(f"Estimated: ~{est_nnz_per_row:.0f} nnz/row, ~{est_gib:.1f} GiB uncompressed")
    if est_gib > args.max_gib:
        raise SystemExit(
            f"Estimated {est_gib:.1f} GiB exceeds --max-gib {args.max_gib:g}. Reduce "
            f"--max-events/--max-rows or a coarser grid, or raise --max-gib deliberately."
        )
    # ~7600 rows/s/core measured on this grid with the bounding-box extraction
    # (the naive full-grid scan manages 156).
    rate = 7600 * args.n_workers
    print(f"Build rate ~7600 rows/s/worker; {n_rows:,} rows with "
          f"{args.n_workers} worker(s) ~ {n_rows/rate/60:.1f} min")

    tof_arg = None
    if has_tof:
        # PETTOFMeta already stores sigma/center_offset as 1-element tensors --
        # wrapping them in a list makes them 2-D and parallelproj rejects it.
        tof_arg = (
            float(tof_meta.bin_width),
            np.asarray(tof_meta.sigma, dtype=np.float32).reshape(-1),
            np.asarray(tof_meta.center_offset, dtype=np.float32).reshape(-1),
            float(args.tof_nsigmas),
            (args.tof_num_bins - 1) // 2,
        )

    if args.verify_bbox and has_tof:
        # Prove on real rows that the bounding box drops nothing before
        # committing hours to a build that would be silently truncated.
        rng = np.random.default_rng(0)
        sample = rows[rng.choice(n_rows, min(args.verify_bbox, n_rows), replace=False)]
        mismatches = 0
        for row in sample:
            c1, v1 = row_voxel_weights(row, lut.numpy(), shape, origin, dr, tof_arg,
                                       full_grid=True)
            c2, v2 = row_voxel_weights(row, lut.numpy(), shape, origin, dr, tof_arg)
            if not (np.array_equal(c1, c2) and np.array_equal(v1, v2)):
                mismatches += 1
        if mismatches:
            raise SystemExit(
                f"Bounding-box extraction disagrees with the full-grid extraction on "
                f"{mismatches}/{len(sample)} rows. Refusing to build a truncated matrix."
            )
        print(f"Bounding-box self-check: {len(sample)} rows, bit-identical to full grid")

    matrix_path = args.output_dir / "projection_matrix.npz"
    print("Building CSR (streaming to disk)...")
    nnz = build_and_save_csr(
        matrix_path, rows, lut.numpy(), shape, origin, dr, tof_arg,
        chunk_size=args.chunk_size, n_workers=args.n_workers,
    )

    # Row ids go in a binary .npz, not JSON: at millions of rows a JSON list is
    # both enormous and slow to parse.
    np.savez_compressed(
        args.output_dir / "projection_matrix_rows.npz",
        rows=rows.astype(np.int32), counts=counts.astype(np.int64),
    )
    meta = {
        "matrix_file": str(matrix_path.resolve()),
        "rows_file": str((args.output_dir / "projection_matrix_rows.npz").resolve()),
        "n_rows": int(n_rows),
        "n_rows_all": int(n_rows_all),
        "n_cols": int(np.prod(shape)),
        "nnz": int(nnz),
        "shape": list(shape),
        "voxel_size_mm": [float(v) for v in dr],
        "object_origin_mm": origin.tolist(),
        "lut": str(args.lut.resolve()),
        "n_detectors": n_det,
        "listmode": str(args.listmode.resolve()),
        "n_events_used": int(len(det)),
        "max_events": args.max_events,
        "max_rows": args.max_rows,
        "has_tof": bool(has_tof),
        "tof_num_bins": args.tof_num_bins if has_tof else None,
        "tof_bin_width_ps": args.tof_bin_width_ps if has_tof else None,
        "tof_fwhm_ps": args.tof_fwhm_ps if has_tof else None,
        "tof_nsigmas": args.tof_nsigmas if has_tof else None,
        "tof_centre_bin_offset": ((args.tof_num_bins - 1) // 2) if has_tof else None,
        "projector": "parallelproj_joseph_tof_lm" if has_tof else "parallelproj_joseph",
        "row_key_formula": "(lo * n_detectors + hi) * tof_num_bins + tof_bin",
    }
    (args.output_dir / "projection_matrix_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    size_gib = matrix_path.stat().st_size / 2**30
    print(f"\nSaved {matrix_path}  nnz={nnz:,} ({nnz/max(n_rows,1):.1f}/row), "
          f"{size_gib:.2f} GiB on disk")
    print(f"Saved {args.output_dir / 'projection_matrix_rows.npz'}")
    print(f"Saved {args.output_dir / 'projection_matrix_meta.json'}")


if __name__ == "__main__":
    main()
