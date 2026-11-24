"""Shared opt-in arguments for the existing reconstruction entry points."""
from pathlib import Path

from .geometry import CrystalGeometry
from .sampling import SamplingConfig


def add_projector_arguments(parser):
    group = parser.add_argument_group("finite-crystal PET projection")
    group.add_argument("--lor-samples", type=int, default=0,
                       help="Paired face/depth rays per LOR; >0 enables the finite-crystal operator.")
    group.add_argument("--crystal-geometry", type=Path,
                       help="Crystal-frame .npz from make_crystal_geometry.py; required with --lor-samples.")
    group.add_argument("--lor-sampling-seed", type=int, default=0,
                       help="Fixed endpoint-sampling seed, independent of event order and subsets.")
    group.add_argument("--lor-batch-rays", type=int, default=65536,
                       help="Maximum sampled rays per projection batch (must be >= --lor-samples).")
    group.add_argument("--crystal-depth-mu-mm", type=float, default=None,
                       help="Optional inverse-mm coefficient for truncated exponential normal depth; default uniform.")


def validate_projector_arguments(args):
    if args.lor_samples < 0:
        raise ValueError("--lor-samples must be >= 0")
    if args.lor_samples == 0:
        if args.crystal_geometry is not None or args.crystal_depth_mu_mm is not None:
            raise ValueError("--crystal-geometry/--crystal-depth-mu-mm require --lor-samples > 0")
        return
    if args.crystal_geometry is None:
        raise ValueError("--lor-samples requires --crystal-geometry (see make_crystal_geometry.py)")
    # Existing cached matrices/maps contain center-ray coefficients. Never
    # silently combine those with the new event operator.
    for name in ("system_matrix_dir", "sensitivity_map", "uniform_sensitivity"):
        if getattr(args, name, None):
            raise ValueError(f"--lor-samples cannot be combined with --{name.replace('_', '-')}; recompute sampled sensitivity")
    sampling_from_args(args)
    if args.lor_batch_rays < args.lor_samples:
        raise ValueError("--lor-batch-rays must be >= --lor-samples")


def sampling_from_args(args):
    return SamplingConfig(samples_per_lor=args.lor_samples,
                          seed=args.lor_sampling_seed,
                          depth_mu_per_mm=args.crystal_depth_mu_mm)


def make_system_matrix(args, object_meta, proj_meta, **kwargs):
    from .pytomography import FiniteCrystalPETLMSystemMatrix

    geometry = CrystalGeometry.load(args.crystal_geometry)
    print(f"Finite-crystal projector: {args.lor_samples} paired rays/LOR, "
          f"seed={args.lor_sampling_seed}, depth="
          f"{'uniform' if args.crystal_depth_mu_mm is None else 'truncated exponential'}", flush=True)
    return FiniteCrystalPETLMSystemMatrix(
        object_meta, proj_meta, crystal_geometry=geometry,
        sampling=sampling_from_args(args), max_rays_per_batch=args.lor_batch_rays,
        **kwargs)
