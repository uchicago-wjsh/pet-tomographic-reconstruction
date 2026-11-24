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
from pytomography.projectors import SystemMatrix          # CHANGED: was used but never imported
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import MLEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.callbacks import Callback
from supported_likelihood import SupportedEventsPoissonLogLikelihood
from pet_projector.cli import (
    add_projector_arguments, make_system_matrix, validate_projector_arguments,
)

# CHANGED: these were all called in main() but never defined or imported here.
from reconstruct_mlem_pytomography import (
    build_sensitivity_pairs,
    load_panel_groups,
    make_tof_meta,
    object_affine,
    save_preview,
    tof_seconds_to_bins,
)

C_MM_PER_PS = 0.299792458          # speed of light
HALF_C = C_MM_PER_PS / 2.0         # position along LOR = (c/2) * dt


class SaveIterationsCallback(Callback):
    """Save the image (NIfTI) at specific iterations.

    pytomography calls callback.run(object, n_iter, n_subset); n_iter is 0-indexed.
    """

    def __init__(self, save_stub: Path, save_iters: list[int], affine: np.ndarray):
        self.save_stub = str(save_stub)
        self.save_iters = set(save_iters)
        self.affine = affine

    def run(self, object, n_iter, n_subset=None):
        it = n_iter + 1
        # Fail LOUDLY and immediately on a diverged iterate. MLEM's update is
        # multiplicative, so once a voxel reaches inf it poisons its neighbours
        # through the projector and the image is unrecoverable -- but the run
        # would otherwise continue to the last iteration, exit 0, and leave a
        # file full of NaN that looks like a finished reconstruction. That has
        # already happened once: a 1.01e9-event run wrote 81% NaN and reported
        # COMPLETED (see --sensitivity-floor-frac for the cause and the fix).
        bad = ~torch.isfinite(object)
        if bad.any():
            n_bad = int(bad.sum())
            # Report WHERE, not just how many: the fix depends entirely on
            # whether these sit on the grid face, at the object edge, or
            # scattered through the volume, and a bare count cannot tell them
            # apart. Guessing that once already cost a wasted re-run.
            import numpy as _np
            _b = bad.detach().cpu().numpy()
            _idx = _np.argwhere(_b)
            _sh = _np.array(_b.shape)
            print(f"  diverged voxels ({n_bad}), grid {tuple(_sh)}:", flush=True)
            for _i in _idx[:12]:
                _edge = int(_np.minimum(_i, _sh - 1 - _i).min())
                print(f"    index {tuple(int(v) for v in _i)}  "
                      f"{_edge} voxel(s) from the nearest grid face", flush=True)
            _fin = object.detach().cpu().numpy()
            _f = _fin[_np.isfinite(_fin)]
            if _f.size:
                print(f"  finite values: max {_f.max():.4g}, "
                      f"p99.99 {_np.percentile(_f, 99.99):.4g}, "
                      f"median(>0) {_np.median(_f[_f > 0]):.4g}", flush=True)
            raise SystemExit(
                f"MLEM DIVERGED at iteration {it}: {n_bad:,} of "
                f"{object.numel():,} voxels are NaN/Inf. This is almost always "
                f"a handful of near-zero-sensitivity voxels at the edge of the "
                f"grid being amplified without bound; raise "
                f"--sensitivity-floor-frac to mask them out.")
        if it in self.save_iters:
            nib.save(
                nib.Nifti1Image(object.detach().cpu().numpy(), self.affine),
                f"{self.save_stub}_iter{it:03d}.nii",
            )
        return object


class VoxelCorrectedSystemMatrix(PETLMSystemMatrix):    
    """
        PETLMSystemMatrix that applies a voxel-wise correction factor.
        The correction comes from Monte Carlo simulation of the voxel sensitivity. 
        Implements 
            
            H_corr = H @ C

        where C is a diagonal matrix representing the sensitivity correction. 
    """
    
    def __init__(
        self, 
        pet_system_matrix: SystemMatrix,
        correction_map: torch.tensor
    ):
        self.pet_system_matrix = pet_system_matrix
        # CHANGED: was `super().__init__(...)` with a missing comma (SyntaxError) and a
        # `proj2proj_transforms` kwarg PETLMSystemMatrix does not accept. Calling
        # PETLMSystemMatrix.__init__ would also re-run _backward_full(), recomputing the
        # whole sensitivity a second time. SystemMatrix.__init__ just sets the attributes.
        SystemMatrix.__init__(
            self,
            object_meta=pet_system_matrix.object_meta,
            proj_meta=pet_system_matrix.proj_meta,
            obj2obj_transforms=[],
            proj2proj_transforms=[],
        )

        if tuple(correction_map.shape) != tuple(self.object_meta.shape):
            raise ValueError(
                f"correction_map shape {tuple(correction_map.shape)} "
                f"!= object shape {tuple(self.object_meta.shape)}"
            )

        if torch.any(~torch.isfinite(correction_map)):
            raise ValueError("correction_map contains NaN or Inf.")

        if torch.any(correction_map < 0):
            raise ValueError("correction_map must be nonnegative.")

        self.correction_map = correction_map.detach().clone()

    def forward(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,     # CHANGED: pytomography calls forward(obj, subset_idx)
    ) -> torch.Tensor:

        c = self.correction_map.to(device=object.device, dtype=object.dtype)

        return self.pet_system_matrix.forward(c * object, subset_idx)

    # CHANGED: subset bookkeeping lives on the wrapped matrix; delegate so MLEM/OSEM
    # and the likelihood keep working through the wrapper.
    def set_n_subsets(self, n_subsets: int):
        out = self.pet_system_matrix.set_n_subsets(n_subsets)
        self.subset_indices_array = self.pet_system_matrix.subset_indices_array
        return out

    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int):
        return self.pet_system_matrix.get_projection_subset(projections, subset_idx)

    def get_weighting_subset(self, subset_idx: int) -> float:
        return self.pet_system_matrix.get_weighting_subset(subset_idx)
    
    def backward(
        self,
        proj: torch.Tensor,
        subset_idx = None,
        return_norm_constant = False,
    ) -> torch.Tensor:
        
        # CHANGED: subset_idx was accepted but silently dropped, so every subset
        # back-projected the full event list.
        result = self.pet_system_matrix.backward(
            proj, subset_idx=subset_idx, return_norm_constant=return_norm_constant
        )
        
        if return_norm_constant:
            BP, norm_constant = result

            c = self.correction_map.to(device=BP.device, dtype=BP.dtype)

            return (
                c * BP,
                c * norm_constant,
            )

        else:
            c = self.correction_map.to(device=result.device, dtype=result.dtype)

            return c * result
        
    def compute_normalization_factor(
        self,
        subset_idx: int | None = None,
    ) -> torch.Tensor:

        norm = self.pet_system_matrix.compute_normalization_factor(
            subset_idx=subset_idx
        )

        c = self.correction_map.to(device=norm.device, dtype=norm.dtype)

        # H_corr^T 1 = C H^T 1
        return c * norm

class UniformSensitivitySystemMatrix(PETLMSystemMatrix):
    r"""PETLMSystemMatrix whose sensitivity image :math:`\tilde{H}^T 1` is a CONSTANT.

    The MLEM denominator is the sensitivity image -- the back-projection of every
    valid LOR, i.e. the probability that an annihilation at each voxel is detected
    at all. Over a small region at the centre of the scanner that probability is
    very nearly flat, and holding it flat has two consequences worth being explicit
    about:

      * It is exact only to the extent the assumption holds. Check it against the
        Monte-Carlo map (results/voxel_sensitivity.npz) over the region actually
        being reconstructed before trusting it; the four-panel scanner varies 8.8x
        across the full 40 cm FOV, so this is a statement about a SMALL central
        region, not about the scanner.
      * The absolute scale becomes arbitrary. Setting the constant to 1 makes the
        MLEM fixed point satisfy sum(image) = number of events, so the image reads
        as "events per voxel" rather than as an activity. Shape -- which is what a
        line profile shows -- is unaffected, since a constant denominator cancels
        out of every voxel ratio.

    Overriding ``_backward_full`` is what makes this cheap: PETLMSystemMatrix calls
    it once from ``__init__`` and stores the result in ``norm_BP``, and the real one
    back-projects the whole cross-panel LOR set -- 1.0e9 LORs for this scanner, tens
    of minutes and ~26 GB just to enumerate. Here nothing is projected at all, and
    ``detector_ids_sensitivity`` need not be built.
    """

    def _backward_full(self, N_splits: int = 20) -> torch.Tensor:
        # Must stay on CPU: __init__ assigns this to self.norm_BP, which the real
        # implementation returns as .cpu().
        return torch.ones(self.object_meta.shape, dtype=pytomography.dtype)


def load_lut(path: Path) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"LUT must be (n_det, 3); got {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32))


# --------------------------------------------------------------------------- #
# CHANGED: grid-aware loading of the Monte-Carlo sensitivity map.
# --------------------------------------------------------------------------- #
def voxel_centres_mm(shape, dr, origin=None) -> list[np.ndarray]:
    """World coordinates of the voxel CENTRES along each axis, in mm.

    pytomography places voxel [0,0,0] at ``object_origin = (-shape/2 + 0.5)*dr``
    and steps by ``dr``, so centre i sits at ``origin[a] + i*dr[a]``. Passing an
    explicit origin lets this also handle a map that is not centred on zero.
    """
    if origin is None:
        origin = (-np.asarray(shape) / 2.0 + 0.5) * np.asarray(dr)
    return [origin[a] + np.arange(shape[a]) * dr[a] for a in range(3)]


def resample_to_grid(src, src_shape, src_dr, src_origin, dst_shape, dst_dr, dst_origin):
    """Resample ``src`` onto the destination grid, matching PHYSICAL positions.

    Both grids are interpreted in mm, so the result is aligned voxel-for-voxel
    with the reconstruction regardless of differing shape, voxel size or extent.
    Trilinear; voxels outside the source volume come back NaN so the caller can
    decide what to do with them rather than silently getting a 0 or an edge value.
    """
    from scipy.ndimage import map_coordinates

    dst_axes = voxel_centres_mm(dst_shape, dst_dr, dst_origin)
    # world mm -> fractional index in the source array
    idx_axes = [(dst_axes[a] - src_origin[a]) / src_dr[a] for a in range(3)]
    grids = np.meshgrid(*idx_axes, indexing="ij")
    out = map_coordinates(
        np.asarray(src, dtype=np.float64), np.stack(grids),
        order=1, mode="constant", cval=np.nan, prefilter=False,
    )
    return out


def load_mc_sensitivity(path: Path, dst_shape, dst_dr, voxel_mm_override=None):
    """Load the MC sensitivity map and put it on the reconstruction's grid.

    The map's own grid is read from its ``.json`` sidecar when present (that is
    what compute_voxel_sensitivity.py writes); otherwise pass
    --mc-sensitivity-voxel-mm and a zero-centred grid is assumed.

    Returns (resampled_map, n_voxels_outside_source).
    """
    src = np.load(path).astype(np.float64)

    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        src_shape = tuple(int(s) for s in meta["shape"])
        src_dr = [float(v) for v in meta["voxel_size_mm"]]
        src_origin = np.asarray(meta.get("object_origin_mm")
                                or (-np.asarray(src_shape) / 2.0 + 0.5) * np.asarray(src_dr))
        if tuple(src.shape) != src_shape:
            raise ValueError(
                f"{path} has array shape {src.shape} but its sidecar says {src_shape}"
            )
    else:
        if voxel_mm_override is None:
            raise ValueError(
                f"{path} has no .json sidecar, so its voxel size is unknown. "
                f"Pass --mc-sensitivity-voxel-mm."
            )
        src_shape = tuple(src.shape)
        src_dr = list(voxel_mm_override)
        src_origin = (-np.asarray(src_shape) / 2.0 + 0.5) * np.asarray(src_dr)
        print(f"WARNING: no sidecar for {path}; assuming a zero-centred grid with "
              f"voxel size {src_dr} mm.")

    dst_origin = (-np.asarray(dst_shape) / 2.0 + 0.5) * np.asarray(dst_dr)
    same_grid = (tuple(src_shape) == tuple(dst_shape)
                 and np.allclose(src_dr, dst_dr)
                 and np.allclose(src_origin, dst_origin))

    print(f"MC sensitivity: {path}")
    print(f"  map  grid {tuple(src_shape)} @ {tuple(src_dr)} mm  "
          f"FOV {[round(n*d,1) for n,d in zip(src_shape, src_dr)]} mm")
    print(f"  recon grid {tuple(dst_shape)} @ {tuple(dst_dr)} mm  "
          f"FOV {[round(n*d,1) for n,d in zip(dst_shape, dst_dr)]} mm")

    if same_grid:
        print("  grids identical -> no resampling")
        return src, 0

    out = resample_to_grid(src, src_shape, src_dr, src_origin,
                           dst_shape, dst_dr, dst_origin)
    outside = ~np.isfinite(out)
    print(f"  resampled (trilinear); {int(outside.sum()):,} of {out.size:,} voxels "
          f"({100*outside.mean():.2f}%) fall outside the map's FOV")
    return out, int(outside.sum())


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
    p.add_argument("--uniform-sensitivity", action="store_true",
                   help="Hold the MLEM sensitivity image constant instead of back-"
                        "projecting the cross-panel LOR set. Valid only over a region "
                        "small enough for the true sensitivity to be flat -- see "
                        "UniformSensitivitySystemMatrix. Skips the 1e9-LOR enumeration "
                        "entirely, so it also makes --n-sensitivity-lors, "
                        "--mc-sensitivity-map, --panel-ids and --sensitivity-weights "
                        "irrelevant.")
    p.add_argument("--n-sensitivity-lors", type=int, default=0,
                   help="0 = all cross-panel LORs (exact); >0 = random sample of that size, "
                        "spread across panel pairs in proportion to their LOR counts. "
                        "Exact costs ~10 min on 4 CPU threads for the four-panel scanner "
                        "(1.0e9 cross-panel LORs) and is STRONGLY preferred: this map is "
                        "the MLEM denominator, so sampling noise in it is fixed-pattern "
                        "and does not average away over iterations.")
    p.add_argument("--mc-sensitivity-map", type=Path, required=False,
                    help="sensitivity map computed from Monte-Carlo simulation. May be on a "
                         "different grid from the reconstruction; it is resampled onto this "
                         "run's grid by physical position.")
    # CHANGED: needed only when the map has no .json sidecar to describe its grid.
    p.add_argument("--mc-sensitivity-voxel-mm", type=float, nargs="+", default=None,
                   help="Voxel size (mm) of --mc-sensitivity-map: one value for isotropic, or "
                        "three for x y z. Only needed when the map has no .json sidecar; "
                        "otherwise the sidecar's grid is used.")
    # CHANGED: removed --max-exact-sensitivity-lors; the cap no longer exists.
    p.add_argument("--panel-ids", type=Path, default=None,
                   help="Optional .npy of one panel label per detector. Overrides the panel "
                        "grouping otherwise taken from the LUT's .json sidecar or inferred "
                        "geometrically. Only needed for non-planar panels.")
    p.add_argument("--mask-border-voxels", type=int, default=2,
                   help="Freeze at ZERO this many voxels around the outer face of the "
                        "reconstruction grid (default 2). MLEM on this scanner "
                        "diverges from voxels on the grid BOUNDARY: a 5.1e8-event run "
                        "produced three such voxels at 1e7-1e8 while the rest of the "
                        "image was healthy (median scaling 2.53x against an event "
                        "ratio of 2.50x), and by iteration 30 the projector had "
                        "spread the resulting inf/NaN over 75%% of the volume. The "
                        "same geometry at 2.0e8 events is clean, so the failure is "
                        "statistics-dependent and silent. Their sensitivity is NOT "
                        "small -- --sensitivity-floor-frac does not catch them -- so "
                        "this is an edge artefact of the projector, masked "
                        "geometrically. The grid is 320x320x192 mm and the NEMA body "
                        "is 300x230x180 mm, so two voxels (4 mm) of border lie "
                        "outside the phantom on every axis and nothing is lost. "
                        "0 disables.")
    p.add_argument("--sensitivity-floor-frac", type=float, default=1e-3,
                   help="Freeze at ZERO every voxel whose MLEM sensitivity is below "
                        "this fraction of the median sensitivity (default 1e-3). "
                        "MLEM divides by the sensitivity image, so a voxel that no "
                        "LOR meaningfully reaches -- the corners and faces of the "
                        "grid, outside the scanner's real FOV -- has an unbounded "
                        "multiplicative update: one stray back-projected event sends "
                        "it to 1e8, then 1e36, then inf, and the projector spreads "
                        "the NaN over the whole image within ~30 iterations. The "
                        "failure is statistics-dependent, which makes it especially "
                        "nasty: a 2.0e8-event run of this geometry is clean while "
                        "5.1e8 and 1.0e9 event runs of the SAME geometry destroy "
                        "themselves. Because the update is multiplicative, setting "
                        "those voxels to 0 in the initial image freezes them there "
                        "for every iteration, which is exactly what is wanted -- they "
                        "carry no information. 0 disables the mask. Only applies when "
                        "the MLEM sensitivity image is computed, i.e. not with "
                        "--uniform-sensitivity.")
    p.add_argument("--sensitivity-seed", type=int, default=20260723)
    p.add_argument("--sensitivity-weights", type=Path, default=None,
                   help="Optional .npy of per-pair weights aligned with the sensitivity pairs "
                        "(e.g. flood efficiencies). Rarely needed; default is unit (geometric).")
    # optional resolution model + performance
    p.add_argument("--init-image", type=Path, default=None,
                   help="Start MLEM from this .npy image instead of a uniform one "
                        "-- e.g. the TOF backprojection of the same data. Must be on "
                        "the reconstruction grid. See --init-floor-frac: a voxel that "
                        "starts at zero can NEVER become non-zero, because the MLEM "
                        "update is multiplicative.")
    p.add_argument("--init-floor-frac", type=float, default=0.01,
                   help="Floor the initial image at this fraction of its non-zero mean "
                        "(default 0.01). MLEM's update x <- x * (...) is multiplicative, "
                        "so any voxel left at exactly 0 is frozen at 0 for every "
                        "iteration. The TOF backprojection is legitimately 0 outside "
                        "the phantom and in unsampled voxels, so without a floor those "
                        "would be permanently unrecoverable rather than merely dark.")
    p.add_argument("--psf-fwhm-mm", type=float, default=None,
                   help="Optional image-space Gaussian PSF FWHM (mm) to approximate finite-crystal "
                        "resolution lost by dropping sub-LOR sampling.")
    p.add_argument("--n-splits", type=int, default=1,
                   help="Split projections into N chunks to cap memory (raise for large sensitivity sets / GPU).")
    p.add_argument("--max-events", type=int, default=None, help="debug: cap number of events")
    p.add_argument("--exclude-zero-response-events", action="store_true",
                   help="Give zero weight to events with no forward response on the active "
                        "image support; record exclusions. Prevents parallelproj boundary "
                        "backprojection leakage from being amplified by 1/delta.")
    add_projector_arguments(p)
    args = p.parse_args()
    validate_projector_arguments(args)
    # CHANGED: accept one isotropic value or three, normalise to a 3-vector.
    if args.mc_sensitivity_voxel_mm is not None:
        v = args.mc_sensitivity_voxel_mm
        if len(v) == 1:
            v = v * 3
        if len(v) != 3:
            p.error("--mc-sensitivity-voxel-mm takes 1 or 3 values")
        args.mc_sensitivity_voxel_mm = v
    return args


# --------------------------------------------------------------------------- #
def main() -> None:
    # load input arguments
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

    # ---- sensitivity pairs (all cross-panel LORs, or a sample) ----
    if args.uniform_sensitivity:
        if args.mc_sensitivity_map is not None:
            raise SystemExit("--uniform-sensitivity and --mc-sensitivity-map both set "
                             "the MLEM denominator; pick one.")
        panels, sens_ids, sens_w, total_cross = [], None, None, 0
        print("Sensitivity: UNIFORM (constant denominator) -- no LOR set built.")
        print("  Valid only where the true sensitivity is flat over the grid; the "
              "image scale is then arbitrary (normalised so sum = n_events).")
    else:
        panels = load_panel_groups(lut, args.lut, args.panel_ids)
        print(f"Panels: {len(panels)} -> "
              + ", ".join(f"{label} ({ids.size})" for label, ids in panels))
        # CHANGED: build_sensitivity_pairs now takes 3 args; the cap was removed.
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
    matrix_cls = (UniformSensitivitySystemMatrix if args.uniform_sensitivity
                  else PETLMSystemMatrix)
    matrix_kwargs = dict(
        obj2obj_transforms=transforms,
        FOV_scale_enabled=False,
        N_splits=args.n_splits,
    )
    if args.lor_samples:
        pet_system_matrix = make_system_matrix(args, object_meta, proj_meta, **matrix_kwargs)
    else:
        pet_system_matrix = matrix_cls(object_meta, proj_meta, **matrix_kwargs)

    sens_img_np = None          # MLEM sensitivity image, for the floor mask below

    if args.mc_sensitivity_map is not None:
        s_pt = pet_system_matrix.compute_normalization_factor()

        # CHANGED: resample the MC map onto THIS reconstruction's grid so that
        # s_mc[i,j,k] and s_pt[i,j,k] refer to the same physical voxel, whatever
        # grid the map was built on.
        s_mc_np, n_outside = load_mc_sensitivity(
            args.mc_sensitivity_map, shape, dr, args.mc_sensitivity_voxel_mm
        )

        s_pt_np = s_pt.detach().cpu().numpy().astype(np.float64)
        sens_img_np = s_pt_np
        assert s_mc_np.shape == s_pt_np.shape, (s_mc_np.shape, s_pt_np.shape)

        # s_pt carries pytomography's own <1e-7 -> 1e7 substitution for voxels no
        # sensitivity LOR reaches. Dividing there would give c ~ 1e-7 and wipe the
        # voxel out twice over, so leave those uncorrected.
        dead = s_pt_np > 1e6
        with np.errstate(invalid="ignore", divide="ignore"):
            c_np = s_mc_np / (s_pt_np + 1e-10)

        # Voxels where the ratio is meaningful: inside the MC map's FOV, with a
        # real pytomography sensitivity, and with non-zero MC sensitivity.
        valid = np.isfinite(c_np) & ~dead & (s_mc_np > 0)
        if not valid.any():
            raise ValueError("no voxel has a usable MC/pytomography sensitivity ratio")

        # NORMALISE FIRST, then fill. s_mc is an absolute detection probability
        # (~1e-2) while s_pt is an unnormalised backprojection sum, so the raw
        # ratio carries a large arbitrary scale. Dividing by its median makes c a
        # pure SHAPE correction centred on 1.0, which keeps the reconstructed
        # values on the same scale as an uncorrected run.
        c_np /= np.median(c_np[valid])

        # Only now is 1.0 the neutral value, so this is where the fallback goes:
        # outside the MC FOV and in voxels pytomography considers unreachable,
        # apply no correction.
        c_np[~valid] = 1.0
        if (~valid).any():
            print(f"  correction set to 1.0 in {int((~valid).sum()):,} voxels "
                  f"({100*(~valid).mean():.2f}%): "
                  f"{n_outside:,} outside the map, {int(dead.sum()):,} with no "
                  f"pytomography sensitivity")

        c_np = np.clip(c_np, 0.0, None)
        v = c_np[valid]
        print(f"  correction factor (normalised): min {v.min():.3f}  median "
              f"{np.median(v):.3f}  max {v.max():.3f}")

        c = torch.from_numpy(c_np.astype(np.float32)).to(s_pt.device)
        system_matrix = VoxelCorrectedSystemMatrix(pet_system_matrix, c)

    else:
        system_matrix = pet_system_matrix

    likelihood_cls = (SupportedEventsPoissonLogLikelihood
                      if args.exclude_zero_response_events else PoissonLogLikelihood)
    likelihood = likelihood_cls(system_matrix)

    print(f"Running MLEM for {args.n_iters} iterations "
          f"(computing sensitivity image once -- may take a bit on CPU)...")
    if args.init_image is not None:
        init_np = np.load(args.init_image).astype(np.float64)
        if tuple(init_np.shape) != tuple(shape):
            raise SystemExit(f"--init-image {init_np.shape} != reconstruction grid "
                             f"{tuple(shape)}")
        init_np[~np.isfinite(init_np)] = 0.0
        init_np = np.maximum(init_np, 0.0)
        pos = init_np[init_np > 0]
        if pos.size == 0:
            raise SystemExit(f"--init-image {args.init_image} is empty")
        floor = args.init_floor_frac * float(pos.mean())
        n_floored = int((init_np < floor).sum())
        init_np = np.maximum(init_np, floor)
        # Match the uniform start's overall scale (mean 1), so this changes only
        # the SHAPE of the starting point, not its magnitude.
        init_np /= init_np.mean()
        print(f"Initial image: {args.init_image.name}")
        print(f"  floored {n_floored:,} of {init_np.size:,} voxels "
              f"({100*n_floored/init_np.size:.1f}%) at {args.init_floor_frac:g} x the "
              f"non-zero mean; scaled to mean 1 like the uniform start")
        print(f"  range after flooring: {init_np.min():.4g} .. {init_np.max():.4g}")
        init = torch.from_numpy(init_np.astype(np.float32)).to(device)
    else:
        init = torch.ones(shape, device=device)

    # Freeze the unreachable voxels at zero. See --sensitivity-floor-frac.
    # PyTomography's 1e7 placeholders denote uncovered voxels and must not
    # contribute to the median of the actual sensitivity distribution.
    if args.sensitivity_floor_frac > 0 and sens_img_np is not None:
        # PyTomography substitutes 1e7 for zero-sensitivity voxels. Exclude
        # those sentinels from the median, especially on sparsely covered
        # finite-crystal grids where they can outnumber the usable voxels.
        finite = np.isfinite(sens_img_np) & (sens_img_np > 0) & (sens_img_np < 1e6)
        if finite.any():
            med = float(np.median(sens_img_np[finite]))
            thr = args.sensitivity_floor_frac * med
            dead = ~(finite & (sens_img_np >= thr))
            n_dead = int(dead.sum())
            if n_dead:
                init = init * torch.from_numpy((~dead).astype(np.float32)).to(init.device)
            print(f"Sensitivity floor: median {med:.4g}, threshold {thr:.4g} "
                  f"({args.sensitivity_floor_frac:g} x median)")
            print(f"  froze {n_dead:,} of {dead.size:,} voxels "
                  f"({100*n_dead/dead.size:.3f}%) at zero -- unreachable by the "
                  f"sensitivity LOR set")
    elif args.sensitivity_floor_frac > 0:
        print("Sensitivity floor: SKIPPED (no MLEM sensitivity image was computed)")

    # Freeze the outer shell of the grid. See --mask-border-voxels. This is the
    # guard that actually stops the divergence seen on this scanner; the
    # sensitivity floor above catches a different (and here empty) failure mode.
    nb_ = args.mask_border_voxels
    if nb_ > 0:
        keep = np.zeros(tuple(shape), dtype=bool)
        keep[nb_:-nb_ or None, nb_:-nb_ or None, nb_:-nb_ or None] = True
        n_edge = int((~keep).sum())
        init = init * torch.from_numpy(keep.astype(np.float32)).to(init.device)
        print(f"Border mask: froze {n_edge:,} of {keep.size:,} voxels "
              f"({100*n_edge/keep.size:.2f}%) in the outer {nb_}-voxel shell "
              f"({nb_*dr[0]:g} x {nb_*dr[1]:g} x {nb_*dr[2]:g} mm)")

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
                pet_system_matrix.crystal_projector.metadata() if args.lor_samples else None
            ),
            "device": str(device),
            "listmode": str(args.listmode), "lut": str(args.lut),
            "n_events": int(len(det)),
            "grid": list(shape), "voxel_mm": list(dr),
            "n_iters": args.n_iters, "saved_iters": save_iters,
            "tof_num_bins": args.tof_num_bins, "tof_bin_width_mm": bin_width_mm,
            "tof_fwhm_ps": args.tof_fwhm_ps, "tof_fwhm_mm": fwhm_mm,
            "uniform_sensitivity": bool(args.uniform_sensitivity),
            "n_panels": len(panels),
            "panels": {label: int(ids.size) for label, ids in panels},
            "sensitivity_lors": 0 if sens_ids is None else int(len(sens_ids)),
            "sensitivity_cross_panel_total": int(total_cross),
            "sensitivity_exact": (None if args.uniform_sensitivity
                                  else bool(len(sens_ids) == total_cross)),
            "psf_fwhm_mm": args.psf_fwhm_mm,
            "image_sum": float(final.sum()), "image_max": float(final.max()),
            "exclude_zero_response_events": args.exclude_zero_response_events,
            "zero_response_exclusions_by_iteration": getattr(likelihood, "exclusion_history", []),
            "n_modeled_events": int(id_pair.shape[0]) - (
                likelihood.exclusion_history[-1] if args.exclude_zero_response_events else 0),
            "mask_border_voxels": args.mask_border_voxels,
            "sensitivity_floor_frac": args.sensitivity_floor_frac,
            "mc_sensitivity_map": str(args.mc_sensitivity_map) if args.mc_sensitivity_map else None,
        }, f, indent=2)
    print(f"Saved: {stub}.nii / .npy, preview PNG, reconstruction_meta.json in {args.output_dir}")


if __name__ == "__main__":
    main()
