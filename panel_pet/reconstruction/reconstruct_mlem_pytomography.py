#!/usr/bin/env python3
"""TOF list-mode MLEM for the panel scanner, built entirely on pytomography.

Works with any number of detector panels (the two-panel and four-panel
geometries both drop straight in). Panels matter in exactly one place: the
MLEM sensitivity is built from cross-panel LORs, so the detectors have to be
grouped by panel. See `load_panel_groups`.

With --lor-samples K --crystal-geometry FILE, use the finite-crystal
pet_projector adapter for matched sampled forward/back projection and
sensitivity. See pet_projector/README.md for the geometry exporter and model.

This replaces reconstruct_mlem_from_listmode_tof.py (the hand-rolled parallelproj
pipeline). The key reason to switch: pytomography derives the MLEM sensitivity
(the denominator, A^T 1) by back-projecting through the SAME PETLMSystemMatrix
used for the forward projection. Forward and sensitivity therefore share one
operator by construction, so the forward/sensitivity `subsample_face` mismatch
that produced the z=+-2 mm dark rows is impossible here.

Inputs\
  - listmode .npz with `detector_ids` (N,2, flat ids into the LUT) and
    `tof_seconds` (= time1 - time2; GATE stores single 1 as the earlier hit, so
    values are <= 0 -- handled by the canonicalize + sign-flip below).
  - scanner_lut .npy of shape (n_det, 3) in mm (crystal centres).


"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import MLEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.callbacks import Callback
from pet_projector.cli import (
    add_projector_arguments, make_system_matrix, validate_projector_arguments,
)

C_MM_PER_PS = 0.299792458          # speed of light
HALF_C = C_MM_PER_PS / 2.0         # position along LOR = (c/2) * dt


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class SaveIterationsCallback(Callback):
    """Save the image (NIfTI) at specific 1-indexed iterations.

    pytomography calls callback.run(object, n_iter, n_subset); n_iter is 0-indexed.
    """

    def __init__(self, save_stub: Path, save_iters: list[int], affine: np.ndarray):
        self.save_stub = str(save_stub)
        self.save_iters = set(save_iters)
        self.affine = affine

    def run(self, object, n_iter, n_subset=None):
        it = n_iter + 1
        if it in self.save_iters:
            nib.save(
                nib.Nifti1Image(object.detach().cpu().numpy(), self.affine),
                f"{self.save_stub}_iter{it:03d}.nii",
            )
        return object


def load_lut(path: Path) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"LUT must be (n_det, 3); got {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32))


def object_affine(shape, dr) -> np.ndarray:
    """Voxel->physical affine for a grid centred at the origin (NIfTI only)."""
    A = np.eye(4, dtype=np.float64)
    for i in range(3):
        A[i, i] = dr[i]
        A[i, 3] = -0.5 * (shape[i] - 1) * dr[i]
    return A


def object_origin(shape, dr) -> np.ndarray:
    """World coordinate of voxel [0,0,0], as PETLMSystemMatrix defines it.

    Must stay byte-identical to ``PETLMSystemMatrix.__init__``'s
    ``self.object_origin``; anything precomputed against a different origin is
    silently misaligned rather than wrong-looking.
    """
    return (-np.asarray(shape) / 2 + 0.5) * np.asarray(dr)


def row_keys(lo, hi, bins, n_detectors: int, num_bins: int) -> np.ndarray:
    """Canonical int64 key identifying a (detector pair, TOF bin) matrix row.

    lo < hi always (events are canonicalised before this is called), so one
    physical LOR maps to exactly one key. Max value here is
    51840^2 * 521 ~= 1.4e12, comfortably inside int64.
    """
    return (
        np.asarray(lo, dtype=np.int64) * n_detectors + np.asarray(hi, dtype=np.int64)
    ) * num_bins + np.asarray(bins, dtype=np.int64)


def sensitivity_geometry_key(shape, dr, n_detectors: int) -> str:
    """Compact identity for everything a sensitivity map depends on.

    The sensitivity is a function of the grid and the scanner alone (not of the
    data or the TOF settings), so this is what a precomputed map must agree on.
    """
    grid = "x".join(str(int(s)) for s in shape)
    voxel = "x".join(f"{float(v):g}" for v in dr)
    return f"grid={grid};voxel_mm={voxel};n_det={int(n_detectors)}"


def make_tof_meta(num_bins: int, bin_width_ps: float, fwhm_ps: float, n_sigmas: float):
    """PETTOFMeta with bins/fwhm expressed as spatial mm along the LOR."""
    bin_width_mm = bin_width_ps * HALF_C
    tof_range_mm = num_bins * bin_width_mm            # PETTOFMeta bin_width = range/num_bins
    fwhm_mm = fwhm_ps * HALF_C
    return PETTOFMeta(num_bins, tof_range_mm, fwhm_mm, n_sigmas=n_sigmas), bin_width_mm, fwhm_mm


def tof_seconds_to_bins(tof_seconds: np.ndarray, tof_meta: PETTOFMeta) -> torch.Tensor:
    edges = tof_meta.bin_edges
    edges = edges.detach().cpu().numpy() if torch.is_tensor(edges) else np.asarray(edges)
    tof_pos_mm = tof_seconds.astype(np.float64) * 1e12 * HALF_C   # s -> ps -> mm
    ids = np.clip(np.digitize(tof_pos_mm, edges) - 1, 0, tof_meta.num_bins - 1)
    return torch.as_tensor(ids, dtype=torch.long)


Panels = list[tuple[str, np.ndarray]]      # [(label, detector_ids), ...]


def infer_panel_groups(lut: torch.Tensor, min_fraction: float = 0.01) -> Panels:
    """Group detectors into panels geometrically, for any panel count.

    Every panel in these scanners is planar, so its detectors share one
    constant coordinate. Repeatedly peel off the largest remaining coplanar
    set: for the two-panel scanner that finds z=+-210, for the four-panel one
    it finds y=+-202.5 (primaries) and x=+-227.5 (sides).
    """
    xyz = np.round(lut.numpy().astype(np.float64), 3)
    remaining = np.ones(len(xyz), dtype=bool)
    groups: Panels = []

    while remaining.any():
        best_axis, best_value, best_count = 0, 0.0, -1
        for axis in range(3):
            values, counts = np.unique(xyz[remaining, axis], return_counts=True)
            k = int(counts.argmax())
            if int(counts[k]) > best_count:
                best_axis, best_value, best_count = axis, float(values[k]), int(counts[k])
        if best_count < max(1, int(min_fraction * len(xyz))):
            raise ValueError(
                f"Panel detection stalled: the largest remaining plane holds only "
                f"{best_count} of {len(xyz)} detectors. Pass --panel-ids with an "
                f"explicit per-detector panel label array."
            )
        selected = remaining & (xyz[:, best_axis] == best_value)
        groups.append(
            (f"{'xyz'[best_axis]}={best_value:g}", np.flatnonzero(selected).astype(np.int64))
        )
        remaining &= ~selected

    return sorted(groups, key=lambda item: int(item[1][0]))


def load_panel_groups(lut: torch.Tensor, lut_path: Path, panel_ids_path: Path | None) -> Panels:
    """Group detector ids by panel, from the most trustworthy source available.

    1. ``--panel-ids``: an .npy of length n_det holding one label per detector.
    2. The LUT's sidecar ``.json`` (written by scanner_lut.py), whose ``plates``
       block carries an exact detector-id range per panel.
    3. A cylindricalPET sidecar: one detector set, permitting all distinct pairs.
    4. Geometric inference (:func:`infer_panel_groups`) for planar scanners.
    """
    n_det = int(lut.shape[0])

    if panel_ids_path is not None:
        labels = np.load(panel_ids_path)
        if labels.shape[0] != n_det:
            raise ValueError(
                f"--panel-ids has {labels.shape[0]} entries but the LUT has {n_det} detectors"
            )
        groups = [
            (str(value), np.flatnonzero(labels == value).astype(np.int64))
            for value in np.unique(labels)
        ]
        return sorted(groups, key=lambda item: int(item[1][0]))

    sidecar = lut_path.with_suffix(".json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text())
        # A ring also contains large coplanar sets (especially axial rings).
        # Inferring panels from those sets incorrectly excludes same-ring LORs
        # from A^T 1 although those events remain in the forward model.
        if metadata.get("system") == "cylindricalPET":
            return [("ring", np.arange(n_det, dtype=np.int64))]
        plates = metadata.get("plates") or {}
        groups = [
            (label, np.arange(spec["detector_id_range"][0], spec["detector_id_range"][1] + 1,
                              dtype=np.int64))
            for label, spec in plates.items()
            if "detector_id_range" in spec
        ]
        covered = sum(len(ids) for _, ids in groups)
        if groups and covered == n_det:
            return sorted(groups, key=lambda item: int(item[1][0]))
        if groups:
            print(f"WARNING: {sidecar} 'plates' covers {covered}/{n_det} detectors; "
                  "falling back to geometric panel detection.")

    return infer_panel_groups(lut)


def build_sensitivity_pairs(panels: Panels, n_pairs, seed):
    """Cross-panel detector pairs used for the MLEM sensitivity (A^T 1).

    Every unordered pair of distinct panels contributes; self-panel pairs are
    excluded, since a LOR joining two crystals of the same flat panel carries
    no information about the FOV.

    n_pairs <= 0  -> enumerate ALL cross-panel pairs (exact, the true all-LOR set).
    n_pairs  > 0  -> random sample of that size, allocated across the panel
                     pairs in proportion to how many LORs each contributes, so
                     the sample stays unbiased over the full cross-panel set.

    Consistency with the forward projection is guaranteed either way (same
    system matrix); n_pairs only trades sampling smoothness against build time.

    Exact enumeration is not capped. It is genuinely large for a panel scanner
    (this one has 1.0e9 cross-panel LORs = ~15 GiB of ids), so size the job's
    memory accordingly, or precompute the map once with
    reconstruction/compute_voxel_sensitivity.py and pass --sensitivity-map.
    """
    if len(panels) == 1:
        # A FULL RING (or any single contiguous detector set) has no panel
        # structure: every unordered pair of distinct crystals is a valid LOR,
        # so there is nothing to exclude and "cross-panel" does not apply.
        det = np.sort(np.asarray(panels[0][1]))   # sorted, so i<j gives id_a<id_b
        n = int(det.size)
        total = n * (n - 1) // 2
        if n_pairs is None or n_pairs <= 0 or n_pairs >= total:
            # EXACT: every unordered pair of distinct crystals. This used to be
            # refused outright. It is merely large, not impossible -- for the
            # Vision 600, 1,848,289,600 pairs = 27.5 GiB of int64 ids -- so the
            # caller must size the job's memory, and gets told what that is.
            # Worth the cost: this map is the MLEM denominator, and a sample
            # leaves fixed-pattern noise that no number of iterations averages
            # away (a 50e6 sample measured 2.1% RMS on the four-panel scanner).
            #
            # Built in row blocks rather than via np.triu_indices, which would
            # allocate two full index arrays and then a third for the stack --
            # 3x the peak for the same result.
            print(f"Enumerating all {total:,} pairs of a single detector set "
                  f"exactly ({total * 16 / 2**30:.1f} GiB for the id array)",
                  flush=True)
            ids = np.empty((total, 2), dtype=np.int64)
            pos = 0
            BLOCK = 512
            for i0 in range(0, n, BLOCK):
                i1 = min(i0 + BLOCK, n)
                idx = np.arange(i0, i1)
                counts = n - idx - 1              # partners of each i
                tot = int(counts.sum())
                if tot == 0:
                    continue
                ids[pos:pos + tot, 0] = np.repeat(det[idx], counts)
                # concatenated aranges (i+1 .. n-1) for every i in the block,
                # without a Python loop over i
                starts = idx + 1
                offs = np.concatenate(([0], np.cumsum(counts)[:-1]))
                seq = (np.arange(tot) - np.repeat(offs, counts)
                       + np.repeat(starts, counts))
                ids[pos:pos + tot, 1] = det[seq]
                pos += tot
            if pos != total:
                raise RuntimeError(f"enumerated {pos:,}, expected {total:,}")
            return ids, total
        rng = np.random.default_rng(seed)
        ids = np.empty((n_pairs, 2), dtype=np.int64)
        filled = 0
        while filled < n_pairs:
            need = n_pairs - filled
            a = det[rng.integers(0, n, need)]
            b = det[rng.integers(0, n, need)]
            ok = a != b
            a, b = a[ok], b[ok]
            ids[filled:filled + a.size, 0] = np.minimum(a, b)
            ids[filled:filled + a.size, 1] = np.maximum(a, b)
            filled += a.size
        return ids, total

    if len(panels) < 2:
        raise ValueError(f"Need at least 1 detector group; found {len(panels)}")

    combos = [(i, j) for i in range(len(panels)) for j in range(i + 1, len(panels))]
    counts = np.asarray(
        [int(panels[i][1].size) * int(panels[j][1].size) for i, j in combos], dtype=np.int64
    )
    total = int(counts.sum())

    if n_pairs is not None and 0 < n_pairs < total:
        rng = np.random.default_rng(seed)
        allocation = rng.multinomial(n_pairs, counts / total)
        ids = np.empty((int(allocation.sum()), 2), dtype=np.int64)
        offset = 0
        for (i, j), take in zip(combos, allocation):
            take = int(take)
            if take == 0:
                continue
            a, b = panels[i][1], panels[j][1]
            ids[offset:offset + take, 0] = a[rng.integers(0, a.size, take)]
            ids[offset:offset + take, 1] = b[rng.integers(0, b.size, take)]
            offset += take
        return ids, total

    print(f"Enumerating all {total:,} cross-panel LORs exactly "
          f"({total * 16 / 2**30:.1f} GiB for the id array alone)", flush=True)
    ids = np.empty((total, 2), dtype=np.int64)
    offset = 0
    for (i, j), count in zip(combos, counts):
        count = int(count)
        a, b = panels[i][1], panels[j][1]
        ids[offset:offset + count, 0] = np.repeat(a, b.size)
        ids[offset:offset + count, 1] = np.tile(b, a.size)
        offset += count
    return ids, total


class PrecomputedSensitivitySystemMatrix(PETLMSystemMatrix):
    """PETLMSystemMatrix that takes its sensitivity image from a file.

    ``PETLMSystemMatrix.__init__`` calls ``self._backward_full()``
    unconditionally, and that call is the expensive part of setting up a
    reconstruction. Overriding the METHOD -- rather than overwriting
    ``norm_BP`` after construction -- means the backprojection never runs at
    all, while everything ``__init__`` does around it still applies, in
    particular the ``norm_BP < 1e-7 -> 1e7`` substitution. A precomputed map is
    therefore treated identically to one built in place.
    """

    def __init__(self, *args, precomputed_sensitivity: torch.Tensor | None = None, **kwargs):
        self._precomputed_sensitivity = precomputed_sensitivity
        super().__init__(*args, **kwargs)

    def _backward_full(self, N_splits: int = 20):       # noqa: N803 (parent's signature)
        if self._precomputed_sensitivity is not None:
            return self._precomputed_sensitivity
        return super()._backward_full(N_splits)


def load_sensitivity_map(
    path: Path, shape, dr, n_detectors: int, psf_fwhm_mm: float | None
) -> torch.Tensor:
    """Load a precomputed sensitivity image and refuse any geometry mismatch.

    A map built for a different grid is not detectably wrong at runtime -- it
    just biases every iteration -- so every field it depends on is checked.
    """
    image = np.load(path)
    expected = tuple(int(s) for s in shape)
    if image.shape != expected:
        raise ValueError(
            f"--sensitivity-map {path} has shape {image.shape}, but this "
            f"reconstruction uses {expected}"
        )

    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        want = sensitivity_geometry_key(shape, dr, n_detectors)
        got = meta.get("geometry_key")
        if got is not None and got != want:
            raise ValueError(
                f"--sensitivity-map {path} was built for a different geometry.\n"
                f"  map:            {got}\n"
                f"  reconstruction: {want}"
            )
        if meta.get("psf_fwhm_mm") != psf_fwhm_mm:
            raise ValueError(
                f"--sensitivity-map {path} was built with psf_fwhm_mm="
                f"{meta.get('psf_fwhm_mm')}, but this reconstruction uses "
                f"{psf_fwhm_mm}. The class folds the PSF into the sensitivity, "
                f"so these must agree."
            )
        print(f"Sensitivity map: {path} ({meta.get('n_sensitivity_lors', '?'):,} LORs, "
              f"{'exact' if meta.get('sensitivity_exact') else 'sampled'})")
    else:
        print(f"Sensitivity map: {path} (no .json sidecar; geometry unverified "
              f"beyond array shape)")

    return torch.from_numpy(image.astype(np.float32))


class SparseForwardSystemMatrix(PrecomputedSensitivitySystemMatrix):
    """Forward/back projection by a precomputed sparse A instead of ray tracing.

    The matrix rows are unique (detector pair, TOF bin) triples, while
    pytomography's forward must return one value PER EVENT, so events are
    mapped onto rows by ``row_index``. Writing that selection as a matrix S
    (S[e, row_index[e]] = 1), the forward is ``S A f`` and the exact adjoint is
    ``A^T S^T``, where ``S^T g`` is a bincount of g over row_index. Getting
    that bincount wrong would break the adjoint identity and bias MLEM, so it
    is asserted in the tests rather than assumed.
    """

    def __init__(self, *args, matrix, row_index: np.ndarray, **kwargs):
        self.matrix = matrix
        self.row_index = np.asarray(row_index, dtype=np.int64)
        super().__init__(*args, **kwargs)

    def _rows_for(self, subset_idx):
        if subset_idx is None:
            return self.row_index
        return self.row_index[self.subset_indices_array[subset_idx].cpu().numpy()]

    def forward(self, object, subset_idx=None):
        obj = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            obj = transform.forward(obj)
        f = obj.detach().cpu().numpy().ravel().astype(np.float32)
        per_row = self.matrix @ f
        return torch.from_numpy(per_row[self._rows_for(subset_idx)]).to(self.output_device)

    def backward(self, proj, subset_idx=None, return_norm_constant=False):
        values = proj.detach().cpu().numpy().astype(np.float64)
        rows = self._rows_for(subset_idx)
        if values.size == 1 and rows.size != 1:      # broadcast scalar projections
            values = np.full(rows.size, float(values), dtype=np.float64)
        per_row = np.bincount(rows, weights=values, minlength=self.matrix.shape[0])
        image = self.matrix.T @ per_row.astype(np.float32)
        BP = torch.from_numpy(
            np.asarray(image, dtype=np.float32).reshape(self.object_meta.shape)
        ).to(pytomography.device)

        norm_constant = self.compute_normalization_factor(subset_idx)
        for transform in self.obj2obj_transforms[::-1]:
            if return_norm_constant:
                BP, norm_constant = transform.backward(BP, norm_constant=norm_constant)
            else:
                BP = transform.backward(BP)
        if return_norm_constant:
            return BP.to(self.output_device), norm_constant.to(self.output_device)
        return BP.to(self.output_device)


def load_system_matrix(matrix_dir: Path, shape, dr, n_detectors, tof_args, event_keys):
    """Load the sparse A and map this run's events onto its rows.

    Returns (matrix, row_index, keep_mask). Events whose row is absent from the
    matrix -- which happens whenever it was built with --max-events/--max-rows --
    are reported and dropped rather than silently mis-projected.
    """
    from scipy import sparse as _sparse

    meta = json.loads((matrix_dir / "projection_matrix_meta.json").read_text())
    expected = {
        "shape": [int(s) for s in shape],
        "voxel_size_mm": [float(v) for v in dr],
        "n_detectors": int(n_detectors),
    }
    for key, want in expected.items():
        got = meta.get(key)
        if got != want:
            raise ValueError(
                f"{matrix_dir}: {key} is {got} but this reconstruction uses {want}"
            )
    for key, want in tof_args.items():
        got = meta.get(key)
        if got != want:
            raise ValueError(
                f"{matrix_dir}: {key} is {got} but this reconstruction uses {want}. "
                f"The matrix bakes in the TOF kernel, so these must agree."
            )

    matrix = _sparse.load_npz(matrix_dir / "projection_matrix.npz").tocsr()
    if matrix.shape != (meta["n_rows"], meta["n_cols"]):
        raise ValueError(f"{matrix_dir}: matrix shape {matrix.shape} disagrees with metadata")

    rows = np.load(matrix_dir / "projection_matrix_rows.npz")["rows"].astype(np.int64)
    num_bins = meta.get("tof_num_bins") or 1
    matrix_keys = row_keys(rows[:, 0], rows[:, 1], rows[:, 2], int(n_detectors), int(num_bins))
    order = np.argsort(matrix_keys)
    pos = np.searchsorted(matrix_keys[order], event_keys)
    pos = np.clip(pos, 0, len(order) - 1)
    row_index = order[pos]
    keep = matrix_keys[row_index] == event_keys

    print(f"System matrix: {matrix_dir}")
    print(f"  {matrix.shape[0]:,} rows x {matrix.shape[1]:,} voxels, nnz={matrix.nnz:,} "
          f"({matrix.nnz / max(matrix.shape[0],1):.1f}/row)")
    if not keep.all():
        print(f"  WARNING: {int((~keep).sum()):,} of {keep.size:,} events "
              f"({100*float((~keep).mean()):.1f}%) have no row in this matrix and are "
              f"DROPPED. It was built with max_events={meta.get('max_events')}, "
              f"max_rows={meta.get('max_rows')}.")
    return matrix, row_index, keep


def save_preview(image: np.ndarray, out_png: Path, voxel_mm: float, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xm, ym, zm = (s // 2 for s in image.shape)
    half = [voxel_mm * s / 2 for s in image.shape]
    views = [
        ("x-z (y-mid)", image[:, ym, :].T, (-half[0], half[0], -half[2], half[2]), "x", "z"),
        ("x-y (z-mid)", image[:, :, zm].T, (-half[0], half[0], -half[1], half[1]), "x", "y"),
        ("y-z (x-mid)", image[xm, :, :].T, (-half[1], half[1], -half[2], half[2]), "y", "z"),
        ("MIP over y", image.max(axis=1).T, (-half[0], half[0], -half[2], half[2]), "x", "z"),
    ]
    pos = image[image > 0]
    vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, (name, data, ext, xl, yl) in zip(axes.ravel(), views):
        im = ax.imshow(data, origin="lower", cmap="inferno", vmax=vmax, extent=ext, aspect="equal")
        ax.set_title(name); ax.set_xlabel(f"{xl} [mm]"); ax.set_ylabel(f"{yl} [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listmode", type=Path, required=True, help="listmode .npz (detector_ids, tof_seconds)")
    p.add_argument("--lut", type=Path, required=True, help="scanner_lut .npy (n_det, 3) in mm")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--save-iters", type=int, nargs="+", default=None,
                   help="1-indexed iterations to also save (default: only the final image)")
    # grid
    p.add_argument("--grid-size", type=int, default=25, help="voxels along x and y")
    p.add_argument("--grid-size-z", type=int, default=None, help="voxels along z (default: grid-size)")
    p.add_argument("--voxel-size-mm", type=float, default=2.0)
    p.add_argument("--voxel-size-z-mm", type=float, default=None)
    # TOF
    p.add_argument("--tof-num-bins", type=int, default=201)
    p.add_argument("--tof-bin-width-ps", type=float, default=30.0)
    p.add_argument("--tof-fwhm-ps", type=float, default=283.0,
                   help="Coincidence timing resolution FWHM. For a 200 ps single-photon "
                        "digitizer, the coincidence CTR is sqrt(2)*200 ~= 283 ps.")
    p.add_argument("--tof-nsigmas", type=float, default=3.0)
    # sensitivity
    p.add_argument("--n-sensitivity-lors", type=int, default=0,
                   help="0 = all cross-panel LORs (exact); >0 = random sample of that size, "
                        "spread across panel pairs in proportion to their LOR counts. "
                        "Exact is impractical beyond ~2 panels (the four-panel scanner has "
                        "~1.0e9 cross-panel LORs); sample instead.")
    p.add_argument("--panel-ids", type=Path, default=None,
                   help="Optional .npy of one panel label per detector. Overrides the panel "
                        "grouping otherwise taken from the LUT's .json sidecar or inferred "
                        "geometrically. Only needed for non-planar panels.")
    p.add_argument("--system-matrix-dir", type=Path, default=None,
                   help="Directory from compute_sparse_system_matrix.py holding "
                        "projection_matrix.npz. Replaces on-the-fly ray tracing with a "
                        "sparse matmul. Only practical for modest event counts -- see that "
                        "script's docstring for the size numbers.")
    p.add_argument("--sensitivity-map", type=Path, default=None,
                   help="Precomputed sensitivity image .npy from "
                        "compute_voxel_sensitivity.py. Substitutes for the map the system "
                        "matrix would build in its constructor -- the most expensive setup "
                        "step, and identical across a sweep that shares one geometry. "
                        "Makes --n-sensitivity-lors/--sensitivity-seed/--sensitivity-weights "
                        "irrelevant.")
    p.add_argument("--sensitivity-seed", type=int, default=20260723)
    p.add_argument("--sensitivity-weights", type=Path, default=None,
                   help="Optional .npy of per-pair weights aligned with the sensitivity pairs "
                        "(e.g. flood efficiencies). Rarely needed; default is unit (geometric).")
    # optional resolution model + performance
    p.add_argument("--psf-fwhm-mm", type=float, default=None,
                   help="Optional image-space Gaussian PSF FWHM (mm) to approximate finite-crystal "
                        "resolution lost by dropping sub-LOR sampling.")
    p.add_argument("--n-splits", type=int, default=1,
                   help="Split projections into N chunks to cap memory (raise for large sensitivity sets / GPU).")
    p.add_argument("--max-events", type=int, default=None, help="debug: cap number of events")
    add_projector_arguments(p)
    args = p.parse_args()
    validate_projector_arguments(args)
    return args


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = pytomography.device
    print(f"pytomography {pytomography.__version__} on device: {device}")

    # ---- geometry + data ----
    lut = load_lut(args.lut)
    n_det = lut.shape[0]
    lm = np.load(args.listmode, allow_pickle=True)
    if "detector_ids" not in lm or "tof_seconds" not in lm:
        raise KeyError(f"{args.listmode} must contain 'detector_ids' and 'tof_seconds'")
    det = lm["detector_ids"].astype(np.int64)
    tof = lm["tof_seconds"].astype(np.float64)
    if args.max_events is not None:
        det, tof = det[: args.max_events], tof[: args.max_events]
    if det.max() >= n_det:
        raise ValueError(f"detector id {det.max()} >= n_det {n_det} (LUT/listmode mismatch)")
    print(f"Events: {len(det)} | detectors: {n_det}")

    # ---- object grid ----
    gz = args.grid_size_z if args.grid_size_z is not None else args.grid_size
    vz = args.voxel_size_z_mm if args.voxel_size_z_mm is not None else args.voxel_size_mm
    shape = (args.grid_size, args.grid_size, gz)
    dr = (args.voxel_size_mm, args.voxel_size_mm, vz)
    object_meta = ObjectMeta(dr=dr, shape=shape)
    affine = object_affine(shape, dr)
    print(f"Grid: {shape} @ {dr} mm (FOV {[round(s*d,1) for s,d in zip(shape,dr)]} mm)")

    # ---- TOF metadata ----
    tof_meta, bin_width_mm, fwhm_mm = make_tof_meta(
        args.tof_num_bins, args.tof_bin_width_ps, args.tof_fwhm_ps, args.tof_nsigmas
    )
    print(f"TOF: {args.tof_num_bins} bins x {bin_width_mm:.3f} mm "
          f"(range +-{args.tof_num_bins*bin_width_mm/2:.0f} mm); FWHM {fwhm_mm:.2f} mm")

    # ---- LOR list (canonicalize by id; flip TOF sign on swap so the signed bin
    #      is defined relative to the smaller->larger id direction) ----
    id1 = torch.from_numpy(det[:, 0])
    id2 = torch.from_numpy(det[:, 1])
    keep = id1 != id2
    if not bool(keep.all()):
        id1, id2, tof = id1[keep], id2[keep], tof[keep.numpy()]
        print(f"Dropped {int((~keep).sum())} same-detector events")
    idx = id1 < id2
    tof_signed = tof.copy()
    tof_signed[~idx.numpy()] *= -1
    tof_bins = tof_seconds_to_bins(tof_signed, tof_meta)
    centre = args.tof_num_bins // 2
    print(f"TOF-bin balance (below/above centre): "
          f"{float((tof_bins<centre).float().mean()):.3f} / {float((tof_bins>centre).float().mean()):.3f}")
    id_pair = torch.stack([torch.where(idx, id1, id2), torch.where(idx, id2, id1), tof_bins], dim=-1)

    # ---- optional precomputed sparse forward matrix ----
    sparse_matrix, row_index = None, None
    if args.system_matrix_dir is not None:
        event_keys = row_keys(
            id_pair[:, 0].numpy(), id_pair[:, 1].numpy(), id_pair[:, 2].numpy(),
            int(n_det), args.tof_num_bins,
        )
        sparse_matrix, row_index, keep_rows = load_system_matrix(
            args.system_matrix_dir, shape, dr, int(n_det),
            {
                "tof_num_bins": args.tof_num_bins,
                "tof_bin_width_ps": args.tof_bin_width_ps,
                "tof_fwhm_ps": args.tof_fwhm_ps,
                "tof_nsigmas": args.tof_nsigmas,
            },
            event_keys,
        )
        if not keep_rows.all():
            id_pair = id_pair[torch.from_numpy(keep_rows)]
            row_index = row_index[keep_rows]
            if id_pair.shape[0] == 0:
                raise SystemExit(
                    "No event in this listmode has a row in the system matrix. It was "
                    "almost certainly built from a different listmode."
                )

    # ---- sensitivity: precomputed from file, or built here from LOR pairs ----
    panels = load_panel_groups(lut, args.lut, args.panel_ids)
    print(f"Panels: {len(panels)} -> "
          + ", ".join(f"{label} ({ids.size})" for label, ids in panels))

    precomputed_sensitivity = None
    total_cross = None
    if args.sensitivity_map is not None:
        precomputed_sensitivity = load_sensitivity_map(
            args.sensitivity_map, shape, dr, int(n_det), args.psf_fwhm_mm
        )
        # detector_ids_sensitivity is only ever read inside the _backward_full
        # chain, which the subclass replaces wholesale, so a placeholder keeps
        # us from materialising up to 1e9 ids for nothing.
        sens_ids = torch.zeros((1, 2), dtype=torch.long)
        sens_w = torch.ones(1, dtype=torch.float32)
    else:
        sens_ids_np, total_cross = build_sensitivity_pairs(
            panels, args.n_sensitivity_lors, args.sensitivity_seed
        )
        sens_ids = torch.from_numpy(sens_ids_np).long()
        if args.sensitivity_weights is not None:
            sens_w = torch.from_numpy(np.load(args.sensitivity_weights).astype(np.float32))
            if sens_w.shape[0] != sens_ids.shape[0]:
                raise ValueError("sensitivity-weights length must match the sensitivity pairs")
        else:
            sens_w = torch.ones(sens_ids.shape[0], dtype=torch.float32)
        print(f"Sensitivity LORs: {len(sens_ids):,} of {total_cross:,} cross-panel "
              f"({'exact' if len(sens_ids)==total_cross else 'sampled'}), "
              f"{'flood/custom' if args.sensitivity_weights else 'unit'} weights")

    # ---- optional resolution model (kept inside the same system matrix) ----
    transforms = []
    if args.psf_fwhm_mm is not None:
        from pytomography.transforms.shared import GaussianFilter
        transforms.append(GaussianFilter(args.psf_fwhm_mm, n_sigmas=3.0))
        print(f"Image-space Gaussian PSF: {args.psf_fwhm_mm} mm FWHM")

    # ---- system matrix + Poisson likelihood + MLEM ----
    proj_meta = PETLMProjMeta(
        id_pair, scanner_LUT=lut, tof_meta=tof_meta,
        detector_ids_sensitivity=sens_ids, weights_sensitivity=sens_w,
    )
    matrix_kwargs = dict(
        obj2obj_transforms=transforms,
        FOV_scale_enabled=False,
        N_splits=args.n_splits,
    )
    if args.lor_samples:
        system_matrix = make_system_matrix(args, object_meta, proj_meta, **matrix_kwargs)
    elif sparse_matrix is not None:
        system_matrix = SparseForwardSystemMatrix(
            object_meta, proj_meta,
            matrix=sparse_matrix, row_index=row_index,
            precomputed_sensitivity=precomputed_sensitivity,
            **matrix_kwargs,
        )
    elif precomputed_sensitivity is not None:
        system_matrix = PrecomputedSensitivitySystemMatrix(
            object_meta, proj_meta,
            precomputed_sensitivity=precomputed_sensitivity,
            **matrix_kwargs,
        )
    else:
        system_matrix = PETLMSystemMatrix(object_meta, proj_meta, **matrix_kwargs)
    likelihood = PoissonLogLikelihood(system_matrix)

    print(f"Running MLEM for {args.n_iters} iterations"
          + (" (sensitivity loaded from file)" if precomputed_sensitivity is not None
             else " (computing sensitivity image once -- may take a bit on CPU)") + "...")
    init = torch.ones(shape, device=device)
    algorithm = MLEM(likelihood, init)
    save_iters = sorted(set(args.save_iters or []) | {args.n_iters})
    stub = args.output_dir / "reconstruction_mlem_tof"
    out = algorithm(
        n_iters=args.n_iters,
        callback=SaveIterationsCallback(stub, save_iters, affine),
    )

    # ---- save ----
    final = out.detach().cpu().numpy().astype(np.float32)
    np.save(f"{stub}.npy", final)
    nib.save(nib.Nifti1Image(final, affine), f"{stub}.nii")
    save_preview(final, args.output_dir / "reconstruction_mlem_tof_preview.png",
                 args.voxel_size_mm, f"pytomography MLEM ({len(det)} events, {args.n_iters} iters)")
    with open(args.output_dir / "reconstruction_meta.json", "w") as f:
        json.dump({
            "engine": f"pytomography {pytomography.__version__}",
            "finite_crystal_projector": (
                system_matrix.crystal_projector.metadata() if args.lor_samples else None
            ),
            "device": str(device),
            "listmode": str(args.listmode), "lut": str(args.lut),
            "n_events": int(len(det)),
            "grid": list(shape), "voxel_mm": list(dr),
            "n_iters": args.n_iters, "saved_iters": save_iters,
            "tof_num_bins": args.tof_num_bins, "tof_bin_width_mm": bin_width_mm,
            "tof_fwhm_ps": args.tof_fwhm_ps, "tof_fwhm_mm": fwhm_mm,
            "n_panels": len(panels),
            "panels": {label: int(ids.size) for label, ids in panels},
            "sensitivity_map": (
                str(args.sensitivity_map) if args.sensitivity_map else None
            ),
            "sensitivity_lors": (
                None if total_cross is None else int(len(sens_ids))
            ),
            "sensitivity_cross_panel_total": (
                None if total_cross is None else int(total_cross)
            ),
            "sensitivity_exact": (
                None if total_cross is None else bool(len(sens_ids) == total_cross)
            ),
            "psf_fwhm_mm": args.psf_fwhm_mm,
            "image_sum": float(final.sum()), "image_max": float(final.max()),
        }, f, indent=2)
    print(f"Saved: {stub}.nii / .npy, preview PNG, reconstruction_meta.json in {args.output_dir}")


if __name__ == "__main__":
    main()
