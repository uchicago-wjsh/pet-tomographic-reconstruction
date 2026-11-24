#!/usr/bin/env python3
"""TOF-only image: back-project each event along its LOR using its TOF estimate.

No iterative reconstruction. Two kernels are available (--kernel):

  delta     (default) Each coincidence is placed at the SINGLE point its TOF
            says the annihilation occurred,

                p = midpoint(d1, d2)  +  (c/2) * dt * unit(d1 -> d2)

            and the points are histogrammed. This is the most-likely-position
            image. It is not the correct backprojection operator -- it throws
            away the fact that the TOF estimate has a width -- but it is the
            cheapest thing that uses the timing, and needs only numpy.

  gaussian  Each event is smeared ALONG its LOR by the TOF kernel itself, a
            Gaussian of sigma = CTR_FWHM / 2.3548 * c/2 centred on that same
            point. This is the actual TOF listmode backprojection H^T 1, the
            adjoint of the forward model MLEM uses, and it is what a single
            unregularised backprojection of the data means.

            Computed with parallelproj's joseph3d_back_tof_lm -- the same
            operator pytomography drives inside PETLMSystemMatrix.backward, so
            this image is consistent with the MLEM path by construction.
            parallelproj is imported ONLY for this kernel, so the delta path
            keeps its numpy-and-scipy-only dependency.

Which one you want depends on the question. The delta image shows where the
timing alone localises each event; the Gaussian image is the operator that
actually appears in reconstruction. For a point source they differ sharply:
the delta image is a cusp (density ~ 1/r^2 near the origin), while the Gaussian
image carries the 6 mm FWHM of the kernel.

Two things follow from that, and both matter when comparing it to MLEM:

  * The raw histogram is proportional to activity x sensitivity, so it
    inherits the scanner's sensitivity shape. Pass --sensitivity-map to divide
    it out and get an activity estimate:

        activity(v) = counts(v) / sensitivity(v)

    The map is CROPPED AND RESAMPLED onto this grid by physical position
    first, so it may be on any grid -- the 200^3 @ 2 mm Monte-Carlo map and
    the 112x112x208 A^T 1 map both work. Without it the image carries the
    scanner's sensitivity shape uncorrected (for the four-panel scanner, the
    +27% axial rise diagnosed earlier).

    --sensitivity-smooth-mm Gaussian-smooths that map first. Noise in a
    DENOMINATOR is multiplicative -- a 10% fluctuation there is a 10% error in
    the voxel regardless of how many counts it collected -- and the sensitivity
    is smooth in space, so this is nearly free.
  * Resolution is set by the TOF kernel, not by the geometry: sigma along the
    LOR is CTR/2.355 * c/2, i.e. 2.5 mm at 40 ps but 7.6 mm at 120 ps. The
    histogram is a delta at the most likely position, so the blur comes from
    the CTR alone.

Grid and FOV default to the box sweep's 112 x 112 x 208 @ 2 mm so the output is
voxel-aligned with results/recon and can be compared directly.

    python reconstruction/tof_backproject.py \
        --listmode results/listmode/lm_600s_ctr40ps.npz \
        --lut reconstruction/scanner_lut.npy \
        --out results/tof_only/tof_600s_ctr40ps.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HALF_C_MM_PER_S = 0.299792458 / 2.0 * 1e12      # mm per second


def voxel_centres_mm(shape, dr, origin=None) -> list[np.ndarray]:
    """World coordinates of the voxel CENTRES along each axis, in mm.

    Same convention as the reconstruction: voxel [0,0,0] sits at
    ``(-shape/2 + 0.5)*dr`` and the grid steps by ``dr``.
    """
    if origin is None:
        origin = (-np.asarray(shape) / 2.0 + 0.5) * np.asarray(dr)
    return [origin[a] + np.arange(shape[a]) * dr[a] for a in range(3)]


def resample_to_grid(src, src_dr, src_origin, dst_shape, dst_dr, dst_origin):
    """Crop/resample ``src`` onto the destination grid, matching PHYSICAL positions.

    Trilinear. Voxels outside the source volume come back NaN so the caller can
    decide, rather than silently getting 0 or a clamped edge value.

    Deliberately duplicated from recon_mlem.load_mc_sensitivity rather than
    imported: that module pulls in torch/pytomography/parallelproj, and this
    script otherwise needs only numpy and scipy. The two are checked to agree
    to machine precision.
    """
    from scipy.ndimage import map_coordinates

    dst_axes = voxel_centres_mm(dst_shape, dst_dr, dst_origin)
    idx_axes = [(dst_axes[a] - src_origin[a]) / src_dr[a] for a in range(3)]
    grids = np.meshgrid(*idx_axes, indexing="ij")
    return map_coordinates(np.asarray(src, dtype=np.float64), np.stack(grids),
                           order=1, mode="constant", cval=np.nan, prefilter=False)


def smooth_sensitivity(sens, dr, fwhm_mm):
    """Gaussian-smooth the sensitivity map, ignoring voxels it does not cover.

    Dividing by a NOISY sensitivity map injects that noise straight into the
    activity estimate, and it does so multiplicatively -- a 10% fluctuation in
    the denominator is a 10% error in every voxel, no matter how many counts
    the numerator has. The sensitivity is a smooth function of position, so
    smoothing the denominator costs almost no resolution and removes that.

    Plain gaussian_filter would be wrong here: the resampled map carries NaN
    wherever it does not reach, and NaN spreads across the whole image under
    convolution. This is a NORMALISED convolution -- smooth the values with
    the gaps zeroed, smooth the validity mask, divide -- so voxels near the
    map's edge are averaged only over real data. The original NaN mask is put
    back afterwards, since smoothing must not invent coverage where there was
    none.

    fwhm_mm is the kernel FWHM in mm, matching the --psf-fwhm-mm convention
    used elsewhere in this project; it is converted to a per-axis sigma in
    voxels, so anisotropic grids are handled.
    """
    from scipy.ndimage import gaussian_filter

    sigma_vox = (fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / np.asarray(dr, float)
    valid = np.isfinite(sens)
    num = gaussian_filter(np.where(valid, sens, 0.0), sigma_vox, mode="nearest")
    den = gaussian_filter(valid.astype(np.float64), sigma_vox, mode="nearest")
    out = np.full(sens.shape, np.nan, dtype=np.float64)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    out[~valid] = np.nan
    return out, sigma_vox


def load_sensitivity(path: Path, dst_shape, dst_dr):
    """Load a sensitivity map and put it on this script's grid.

    The map's own grid comes from its ``.json`` sidecar (written by
    compute_voxel_sensitivity.py / unfold_mc_sensitivity.py). Returns
    (resampled_map, n_voxels_outside_the_source).
    """
    src = np.load(path).astype(np.float64)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise SystemExit(
            f"{path} has no .json sidecar, so its grid is unknown and it cannot "
            f"be registered to the reconstruction grid."
        )
    meta = json.loads(sidecar.read_text())
    src_shape = tuple(int(v) for v in meta["shape"])
    src_dr = np.asarray([float(v) for v in meta["voxel_size_mm"]])
    src_origin = np.asarray(meta.get("object_origin_mm")
                            or (-np.asarray(src_shape) / 2.0 + 0.5) * src_dr)
    if tuple(src.shape) != src_shape:
        raise ValueError(f"{path} array shape {src.shape} != sidecar {src_shape}")

    dst_origin = (-np.asarray(dst_shape) / 2.0 + 0.5) * np.asarray(dst_dr)
    print(f"Sensitivity map: {path}")
    fmt = lambda v: "(" + ", ".join(f"{float(x):g}" for x in v) + ")"
    fov = lambda n_, d_: "[" + ", ".join(f"{n*d:g}" for n, d in zip(n_, d_)) + "]"
    print(f"  map  {tuple(int(v) for v in src_shape)} @ {fmt(src_dr)} mm  "
          f"FOV {fov(src_shape, src_dr)} mm")
    print(f"  grid {tuple(int(v) for v in dst_shape)} @ {fmt(dst_dr)} mm  "
          f"FOV {fov(dst_shape, dst_dr)} mm")

    same = (tuple(src_shape) == tuple(dst_shape)
            and np.allclose(src_dr, dst_dr) and np.allclose(src_origin, dst_origin))
    if same:
        print("  grids identical -> no resampling")
        return src, 0
    out = resample_to_grid(src, src_dr, src_origin, dst_shape, dst_dr, dst_origin)
    outside = int((~np.isfinite(out)).sum())
    print(f"  cropped/resampled (trilinear); {outside:,} of {out.size:,} voxels "
          f"({100*outside/out.size:.2f}%) fall outside the map")
    return out, outside


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listmode", type=Path, required=True)
    p.add_argument("--lut", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid-size", type=int, default=112)
    p.add_argument("--grid-size-z", type=int, default=208)
    p.add_argument("--voxel-size-mm", type=float, default=2.0)
    p.add_argument("--voxel-size-z-mm", type=float, default=None)
    p.add_argument("--frontface-basis", type=Path, default=None,
                   help="scanner_frontface.npz from make_frontface_basis.py. When "
                        "given, each LOR endpoint is drawn UNIFORMLY over the "
                        "crystal's 4x4 mm front face instead of being pinned to the "
                        "crystal centre. The centres sit on the crystal pitch, which "
                        "puts every LOR midpoint on a 2 mm lattice and prints a comb "
                        "into the image of a source at the isocentre; sampling the "
                        "face removes it.")
    p.add_argument("--frontface-seed", type=int, default=20260911,
                   help="seed for the front-face sampling")
    p.add_argument("--kernel", choices=("delta", "gaussian"), default="delta",
                   help="delta = histogram the most-likely position (default, numpy "
                        "only); gaussian = smear each event along its LOR with the "
                        "TOF kernel (needs parallelproj).")
    p.add_argument("--ctr-ps", type=float, default=None,
                   help="Coincidence timing resolution FWHM in ps, setting the "
                        "Gaussian kernel width. Default: read from the listmode's "
                        "own 'ctr_ps' field, which is the value actually applied to "
                        "the data -- override only to model a DIFFERENT kernel than "
                        "the one the data carries.")
    p.add_argument("--tof-bin-width-mm", type=float, default=None,
                   help="parallelproj TOF bin width (default: the voxel size). Only "
                        "sets the kernel's normalisation and a negligible amount of "
                        "extra broadening (Gaussian convolved with a rect of this "
                        "width); the image is rescaled so each event contributes 1.")
    p.add_argument("--tof-nsigmas", type=float, default=3.0,
                   help="Truncate the Gaussian at this many sigma (default 3).")
    p.add_argument("--chunk", type=int, default=20_000_000,
                   help="Events per pass; caps peak memory on large listmodes.")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--sensitivity-map", type=Path, default=None,
                   help="Sensitivity map .npy (with its .json sidecar). Cropped and "
                        "resampled onto this grid by physical position, then divided "
                        "out so the output is an ACTIVITY estimate rather than counts. "
                        "The map may be on any grid.")
    p.add_argument("--sensitivity-smooth-mm", type=float, default=0.0,
                   help="FWHM in mm of a Gaussian applied to the sensitivity map "
                        "before dividing. The map is a smooth function of position, "
                        "so this suppresses its Monte-Carlo noise -- which would "
                        "otherwise enter every voxel multiplicatively -- at almost no "
                        "cost in resolution. 0 disables it.")
    p.add_argument("--min-sensitivity-frac", type=float, default=0.01,
                   help="Voxels whose sensitivity is below this fraction of the map's "
                        "maximum are set to 0 rather than divided: counts/sensitivity "
                        "explodes there and the result is noise, not activity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vz = args.voxel_size_z_mm if args.voxel_size_z_mm is not None else args.voxel_size_mm
    shape = (args.grid_size, args.grid_size, args.grid_size_z)
    dr = np.asarray([args.voxel_size_mm, args.voxel_size_mm, vz], dtype=np.float64)
    # Voxel CENTRES sit at (i - (n-1)/2)*d, so the grid's lower EDGE is at
    # -(n/2)*d. Binning has to use edges, not centres, or everything shifts by
    # half a voxel.
    lo = -0.5 * np.asarray(shape) * dr

    lut = np.load(args.lut).astype(np.float64)
    face = None
    if args.frontface_basis is not None:
        fb = np.load(args.frontface_basis)
        face = (fb["front_face_center"].astype(np.float64),
                fb["axis1"].astype(np.float64), fb["axis2"].astype(np.float64),
                float(fb["half1"]), float(fb["half2"]))
        if face[0].shape != lut.shape:
            raise SystemExit(f"{args.frontface_basis} has {face[0].shape[0]} detectors "
                             f"but {args.lut} has {lut.shape[0]}")
        print(f"Endpoints: sampled uniformly over each crystal's "
              f"{2*face[3]:g} x {2*face[4]:g} mm front face (seed {args.frontface_seed})")
    else:
        print("Endpoints: crystal centres from the LUT")
    face_rng = np.random.default_rng(args.frontface_seed)
    lm = np.load(args.listmode, allow_pickle=True)
    det = lm["detector_ids"]
    tof = lm["tof_seconds"]
    if args.max_events is not None:
        det, tof = det[: args.max_events], tof[: args.max_events]
    n_events = len(det)

    print(f"Grid: {shape} @ {tuple(dr)} mm (FOV {[round(s*d,1) for s,d in zip(shape,dr)]} mm)")
    print(f"Events: {n_events:,}")

    sigma_mm = bin_width_mm = None
    if args.kernel == "gaussian":
        import parallelproj                       # only this kernel needs it

        ctr_ps = args.ctr_ps
        if ctr_ps is None:
            if "ctr_ps" not in lm:
                raise SystemExit(f"{args.listmode} has no 'ctr_ps' field; pass --ctr-ps")
            ctr_ps = float(lm["ctr_ps"])
        if ctr_ps <= 0:
            raise SystemExit("a Gaussian kernel needs a non-zero CTR")
        # CTR is the FWHM of the error on dt; the resulting spatial sigma along
        # the LOR is (CTR/2.3548) * c/2, the same factor the listmode used.
        sigma_mm = ctr_ps * 1e-12 * HALF_C_MM_PER_S / 2.3548
        bin_width_mm = args.tof_bin_width_mm or float(dr[0])
        # parallelproj wants the world coordinate of voxel [0,0,0]'s CENTRE,
        # whereas `lo` above is the grid's lower EDGE.
        img_origin = (lo + 0.5 * dr).astype(np.float32)
        print(f"Kernel: Gaussian along the LOR, CTR {ctr_ps:g} ps FWHM "
              f"-> sigma {sigma_mm:.3f} mm ({2.3548*sigma_mm:.3f} mm FWHM)")
        print(f"        tof bin width {bin_width_mm:g} mm, truncated at "
              f"{args.tof_nsigmas:g} sigma")
    else:
        print("Kernel: delta (most-likely position histogram)")

    img = np.zeros(shape, dtype=np.float64)
    n_in = 0
    for start in range(0, n_events, args.chunk):
        stop = min(start + args.chunk, n_events)
        d = det[start:stop].astype(np.int64)
        t = tof[start:stop].astype(np.float64)
        if face is None:
            p1 = lut[d[:, 0]]
            p2 = lut[d[:, 1]]
        else:
            centre, a1, a2, h1, h2 = face
            # One independent draw per ARM, not one per event: the two crystals
            # are hit by different photons and their entry points are unrelated.
            def sample(ids):
                uv = face_rng.uniform(-1.0, 1.0, size=(ids.size, 2))
                return (centre[ids]
                        + a1[ids] * (uv[:, 0:1] * h1)
                        + a2[ids] * (uv[:, 1:2] * h2))
            p1 = sample(d[:, 0])
            p2 = sample(d[:, 1])

        if args.kernel == "gaussian":
            # Offset of the event's most-likely position from the LOR midpoint,
            # measured along p1 -> p2. Same sign convention as the delta branch:
            # verified against parallelproj, offset +x moves the kernel toward p2.
            offset = HALF_C_MM_PER_S * t
            # The offset MUST go in the integer tofbin, not in tofcenter_offset.
            # parallelproj truncates the kernel at nsigmas about the LOR CENTRE
            # plus the bin position, so an offset carried entirely in
            # tofcenter_offset has its far tail cut: measured on a single LOR at
            # nsigmas=3, an 8 mm offset kept only 44% of its weight AND landed
            # its centroid at 5.7 mm instead of 8.0. Carried as a bin index the
            # weight is 1.0000 and the centroid exact at every offset tested.
            # Only the sub-bin remainder goes in tofcenter_offset, where it is
            # far too small to be clipped.
            tofbin = np.rint(offset / bin_width_mm)
            if np.abs(tofbin).max() > np.iinfo(np.int16).max:
                raise SystemExit(
                    f"TOF offset needs bin {np.abs(tofbin).max():.0f}, beyond int16; "
                    f"raise --tof-bin-width-mm above {bin_width_mm:g}")
            remainder = (offset - tofbin * bin_width_mm).astype(np.float32)
            chunk_img = parallelproj.joseph3d_back_tof_lm(
                p1.astype(np.float32), p2.astype(np.float32),
                shape, img_origin, dr.astype(np.float32),
                np.ones(stop - start, dtype=np.float32),
                float(bin_width_mm),
                np.array([sigma_mm], dtype=np.float32),
                remainder,
                float(args.tof_nsigmas),
                tofbin.astype(np.int16),
            )
            # parallelproj returns a kernel whose integral is `bin_width_mm`;
            # dividing makes each event deposit a total weight of exactly 1, so
            # img.sum() is directly comparable to the delta image's event count.
            img += np.asarray(chunk_img, dtype=np.float64) / bin_width_mm
            n_in = int(round(img.sum()))
            print(f"  {stop:,}/{n_events:,} events  "
                  f"({100*img.sum()/max(stop,1):.1f}% of weight inside the FOV)",
                  flush=True)
        else:
            vec = p2 - p1
            length = np.linalg.norm(vec, axis=1)
            pos = 0.5 * (p1 + p2) + (vec / length[:, None]) * (HALF_C_MM_PER_S * t)[:, None]

            idx = np.floor((pos - lo) / dr).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
            idx = idx[inside]
            n_in += int(inside.sum())
            flat = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
            img += np.bincount(flat, minlength=img.size).reshape(shape)
            print(f"  {stop:,}/{n_events:,} events  ({100*n_in/max(stop,1):.1f}% inside the FOV)",
                  flush=True)

    n_outside = 0
    n_masked = 0
    if args.sensitivity_map is not None:
        sens, n_outside = load_sensitivity(args.sensitivity_map, shape, dr)
        if args.sensitivity_smooth_mm > 0:
            before = np.nanstd(sens) / np.nanmean(sens)
            sens, sigma_vox = smooth_sensitivity(sens, dr, args.sensitivity_smooth_mm)
            after = np.nanstd(sens) / np.nanmean(sens)
            print(f"  smoothed: {args.sensitivity_smooth_mm:g} mm FWHM "
                  f"(sigma {np.array2string(sigma_vox, precision=2)} voxels); "
                  f"map CoV {100*before:.1f}% -> {100*after:.1f}%")
        # Dividing by a near-zero sensitivity turns a handful of counts into a
        # huge activity. Anything below the floor is left at 0: with no
        # meaningful sensitivity there is no activity estimate to make.
        finite = np.isfinite(sens)
        floor = args.min_sensitivity_frac * np.nanmax(sens)
        usable = finite & (sens > floor)
        n_masked = int((~usable).sum())
        activity = np.zeros_like(img)
        np.divide(img, sens, out=activity, where=usable)
        print(f"  sensitivity floor {floor:.6g} "
              f"({args.min_sensitivity_frac:g} x max); {n_masked:,} voxels "
              f"({100*n_masked/img.size:.2f}%) set to 0 "
              f"({n_outside:,} outside the map, {n_masked - n_outside:,} below the floor)")
        img = activity

    img = img.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, img)
    args.out.with_suffix(".json").write_text(json.dumps({
        "method": ("TOF backprojection, Gaussian kernel along the LOR"
                   if args.kernel == "gaussian"
                   else "TOF backprojection (most-likely-position histogram)"),
        "kernel": args.kernel,
        "endpoint": ("frontface_random" if face is not None else "crystal_center"),
        "frontface_basis": (str(args.frontface_basis.resolve())
                            if args.frontface_basis else None),
        "frontface_seed": args.frontface_seed if face is not None else None,
        "tof_sigma_mm": sigma_mm,
        "tof_bin_width_mm": bin_width_mm,
        "tof_nsigmas": args.tof_nsigmas if args.kernel == "gaussian" else None,
        "listmode": str(args.listmode.resolve()), "lut": str(args.lut.resolve()),
        "shape": list(shape), "voxel_size_mm": [float(v) for v in dr],
        "n_events": int(n_events), "n_events_in_fov": int(n_in),
        "sensitivity_corrected": args.sensitivity_map is not None,
        "sensitivity_map": (str(args.sensitivity_map.resolve())
                            if args.sensitivity_map else None),
        "min_sensitivity_frac": args.min_sensitivity_frac,
        "sensitivity_smooth_fwhm_mm": args.sensitivity_smooth_mm,
        "n_voxels_outside_sensitivity_map": int(n_outside),
        "n_voxels_zeroed": int(n_masked),
    }, indent=2) + "\n")
    label = "events' weight" if args.kernel == "gaussian" else "events"
    print(f"\n{n_in:,} of {n_events:,} {label} landed in the FOV "
          f"({100*n_in/n_events:.1f}%)")
    print(f"Saved {args.out}  sum {img.sum():.6g}  max {img.max():.6g}")


if __name__ == "__main__":
    main()
