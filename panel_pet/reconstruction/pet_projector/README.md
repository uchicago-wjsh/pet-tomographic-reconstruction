# Finite-crystal PET projector

This package replaces a single crystal-center ray with a Monte Carlo average
over interaction positions in both detector crystals. It calls parallelproj's
existing Joseph CPU/CUDA kernels; the new part is the finite-crystal system
matrix, endpoint sampling, batching, and matched PyTomography adapter.

For detector pair i and voxel j, the operator is

```text
A[i,j] = (1/K) Σ(k=1..K) Joseph(p1[i,k], p2[i,k], voxel j)
forward(f)[i] = (1/K) Σ(k=1..K) projection_of_ray_k(f)
backward(g)   = Σ(i,k) backprojection_of_ray_k(g[i]/K)
```

Each sample draws two independent entrance-face positions and two independent
depths, one of each per crystal. K means K paired rays, with K sampled points
per crystal, rather than K² combinations. By default face coordinates and
depth are uniform. The optional `depth_mu_per_mm` selects a normalized
truncated exponential along the crystal normal. This is a specified endpoint
distribution, not a full transport, oblique penetration, scatter, or absolute
detector-efficiency model.

Samples are fixed by detector IDs, sample index, and seed. Repeated events for
the same pair reuse the same rays. Forward, backward, subsets, sensitivity,
and later iterations regenerate the same samples without storing all rays.
Increasing K retains the existing sample prefix. Reversing a pair swaps its
endpoints. No measured TOF value is randomized. The TOF kernel uses the
midpoint and direction of each sampled ray, with the original measured bin.

## Use with this repository

Activate the environment in the repository README and run these commands from `panel_pet/`.

```bash
python reconstruction/make_crystal_geometry.py \
  --macro macros/geometry.mac \
  --lut reconstruction/scanner_lut.npy \
  --out reconstruction/crystal_geometry.npz

python reconstruction/reconstruct_mlem_pytomography.py \
  --listmode results/listmode/YOUR_LISTMODE.npz \
  --lut reconstruction/scanner_lut.npy \
  --output-dir results/recon_finite_crystal \
  --grid-size 25 --voxel-size-mm 2 \
  --lor-samples 32 \
  --crystal-geometry reconstruction/crystal_geometry.npz \
  --lor-sampling-seed 1234 --lor-batch-rays 65536 \
  --n-sensitivity-lors 100000 --n-iters 5
```

The example is a small trial with sampled sensitivity. Choose the actual
image grid, TOF settings, and accepted sensitivity pairs for your experiment.
The package does not choose a statistically adequate K or sensitivity sample
size automatically. Compare increasing values before relying on results.

Both `reconstruct_mlem_pytomography.py` and `recon_mlem.py` accept these options.
Omitting `--lor-samples` retains their existing center-ray behavior. The latter
also supports its existing Monte Carlo sensitivity correction; its geometric
denominator is recomputed with the finite-crystal model before that correction.
Endpoint sampling parameters and a crystal-geometry SHA256 are saved in
`reconstruction_meta.json` under `finite_crystal_projector`.

The exporter reads this repository's GATE panel geometry and placement
rotations, including outward depth directions. For another scanner, supply
explicit frames using the API below. A ring scanner's mean-DOI LUT alone is
insufficient to determine crystal face dimensions and orientation.

## Python API

To install the projector package, run this from `panel_pet/`.

```bash
python -m pip install --no-deps --no-build-isolation ./reconstruction
```

Install parallelproj and its compiled libraries through the existing conda
environment first. The tested backend is parallelproj 1.10.2 with PyTomography
3.4.0. GPU use also requires the existing CuPy/CUDA setup. The repository's
GPU sbatch scripts already set `PARALLELPROJ_CUDA_LIB` and
`PARALLELPROJ_CUDA_KERNEL_FILE` to the installed library and kernel paths.

```python
import torch
from pet_projector import (
    CrystalGeometry, CrystalProjector, SamplingConfig, TOFParameters,
)

geometry = CrystalGeometry.load("reconstruction/crystal_geometry.npz")
projector = CrystalProjector(
    geometry, image_shape=(25, 25, 25), voxel_size_mm=(2, 2, 2),
    sampling=SamplingConfig(samples_per_lor=32, seed=1234),
    device="cuda", max_rays_per_batch=65536,
)
pairs = torch.tensor([[0, 20000], [1, 20001]])  # example accepted detector IDs
image = torch.ones(25, 25, 25, device="cuda")
predicted = projector.forward(image, pairs)
adjoint = projector.backward(torch.ones(2), pairs)

# Signed bins: positive along p1 -> p2, zero at each sampled ray's midpoint.
tof = TOFParameters(bin_width_mm=2, sigma_mm=5)
bins = torch.tensor([-1, 2])
predicted_tof = projector.forward(image, pairs, tof_bins=bins, tof=tof)

# Inspect A[i,j] for a single LOR, without a full dense system matrix.
flat_voxel_indices, coefficients = projector.voxel_weights(
    pairs[0], tof_bin=-1, tof=tof,
)
```

Construct arbitrary oriented boxes with
`CrystalGeometry(front_face_centers_mm, axes, size_mm)`. Arrays have shapes
`(n_detectors,3)`, `(n_detectors,3,3)`, and `(n_detectors,3)` respectively;
one size triple may also be broadcast to every detector. Axis columns are
two entrance-face unit vectors and a normal pointing into the crystal.
Dimensions are full face width, face height, and depth in mm.
`CrystalGeometry.from_centers` accepts **geometric** crystal centers.

`FiniteCrystalPETLMSystemMatrix` in `pet_projector.pytomography` plugs into
PyTomography MLEM/OSEM. It supports object transforms, event pair weights,
sensitivity pair weights, subsets, TOF, and non-TOF. Provide the accepted
`detector_ids_sensitivity` explicitly; the adapter does not enumerate them.

## Sensitivity and numerical behavior

Sensitivity must use the same sampled geometry, K, and seed. It is computed
as a non-TOF backprojection over accepted detector pairs, with optional pair
weights. Measured events alone do not define the sensitivity. Non-TOF
normalization assumes integration over the full timing acceptance; a restricted
TOF acquisition window needs its own acceptance model.

The reconstruction options reject old `--sensitivity-map` and
`--system-matrix-dir` inputs when finite-crystal projection is enabled because
those caches contain center-ray coefficients. `--uniform-sensitivity` is also
incompatible. Image-based attenuation is not implemented in the adapter;
externally calculated pair weights are supported. Existing center-ray
attenuation calculations must not silently stand in for a sampled model.

Each LOR's value is an average, so raising K does not multiply its sensitivity
by K. Runtime generally grows with sampled-ray count; batching caps expanded
ray working memory, not the input/output arrays or image. Forward and backward
use float32, inherit parallelproj's discretization and floating-point behavior,
and expose explicit adjoints rather than PyTorch autograd. GPU scatter-adds
can differ slightly in accumulation order between runs.

## Validation

From `panel_pet/reconstruction/`, using the environment interpreter:

```bash
PARALLELPROJ_SILENT_IMPORT=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_pet_projector.py' -v
```

Set `PARALLELPROJ_C_LIB` if the environment's C library is not automatically
found. Tests use the real parallelproj kernels and check explicit ray averages,
TOF/non-TOF adjoint identities, extracted voxel coefficients, the center-ray
limit, face/depth bounds and distributions, pair reversal, chunk/subset
invariance, sampled sensitivity, and a small MLEM reconstruction. A CUDA
comparison runs when a GPU and the parallelproj CUDA backend are available.

References: [parallelproj paper](https://doi.org/10.3389/fnume.2023.1324562)
and [parallelproj source](https://github.com/KUL-recon-lab/parallelproj).
