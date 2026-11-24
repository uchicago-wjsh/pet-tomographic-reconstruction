"""Matrix-free finite-crystal PET forward projection and its matched adjoint."""
from dataclasses import dataclass
from contextlib import nullcontext
from math import ceil, isfinite
from numbers import Integral

import numpy as np
import parallelproj
import torch

from .geometry import CrystalGeometry
from .sampling import SamplingConfig, sample_endpoints


@dataclass(frozen=True)
class TOFParameters:
    """parallelproj TOF parameters in spatial mm; bin indices are SIGNED.

    Zero denotes the midpoint of each sampled ray. Positive bin indices
    point from the first endpoint to the second. No TOF value is redrawn.
    """

    bin_width_mm: float
    sigma_mm: float
    center_offset_mm: float = 0.
    n_sigmas: float = 3.

    def __post_init__(self):
        for name in ("bin_width_mm", "sigma_mm", "n_sigmas"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.center_offset_mm):
            raise ValueError("center_offset_mm must be finite")


class CrystalProjector:
    r"""Implements A[i,j] = (1/K) sum_k Joseph(ray[i,k], voxel[j]).

    parallelproj runs its existing CPU or CUDA kernels on batches of sampled
    rays. Projection averages the K ray values; backprojection distributes
    each event value to those SAME K rays with weight 1/K. The image origin
    is the center of voxel [0,0,0]. This API uses explicit adjoints, not
    PyTorch autograd, and all calculations are float32 like parallelproj.
    """

    def __init__(self, geometry: CrystalGeometry, image_shape, voxel_size_mm,
                 *, image_origin_mm=None, sampling=None, device="cpu",
                 max_rays_per_batch=65536):
        self.geometry = geometry
        self.sampling = SamplingConfig() if sampling is None else sampling
        if len(image_shape) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in image_shape):
            raise ValueError("image_shape must contain three positive integers")
        self.image_shape = tuple(int(n) for n in image_shape)
        self.voxel_size_mm = np.broadcast_to(np.asarray(voxel_size_mm, dtype=np.float32), (3,)).copy()
        if not np.isfinite(self.voxel_size_mm).all() or (self.voxel_size_mm <= 0).any():
            raise ValueError("voxel_size_mm must contain positive finite dimensions")
        origin = ((-np.asarray(self.image_shape) / 2 + .5) * self.voxel_size_mm
                  if image_origin_mm is None else image_origin_mm)
        self.image_origin_mm = np.asarray(origin, dtype=np.float32).copy()
        if self.image_origin_mm.shape != (3,) or not np.isfinite(self.image_origin_mm).all():
            raise ValueError("image_origin_mm must be a finite vector of length 3")
        if (isinstance(max_rays_per_batch, bool) or not isinstance(max_rays_per_batch, Integral)
                or max_rays_per_batch < self.sampling.samples_per_lor):
            raise ValueError("max_rays_per_batch must be an integer >= samples_per_lor")
        self.max_rays_per_batch = int(max_rays_per_batch)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("parallelproj requires a CPU or CUDA device")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._faces = torch.tensor(geometry.front_face_centers_mm, dtype=torch.float32, device=self.device)
        self._axes = torch.tensor(geometry.axes, dtype=torch.float32, device=self.device)
        self._sizes = torch.tensor(geometry.size_mm, dtype=torch.float32, device=self.device)
        self._origin = torch.tensor(self.image_origin_mm, device=self.device)
        self._voxel = torch.tensor(self.voxel_size_mm, device=self.device)

    def metadata(self):
        return {"projector": "parallelproj_finite_crystal_joseph",
                "geometry_sha256": self.geometry.fingerprint,
                "image_shape": list(self.image_shape),
                "voxel_size_mm": self.voxel_size_mm.tolist(),
                "image_origin_mm": self.image_origin_mm.tolist(),
                "sampling": self.sampling.metadata()}

    def _pairs(self, pairs):
        pairs = torch.as_tensor(pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("detector_pairs must have shape (n_lors, 2)")
        if pairs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("detector_pairs must contain integer detector IDs")
        if pairs.numel() and (bool((pairs < 0).any()) or bool((pairs >= self.geometry.n_detectors).any())):
            raise ValueError("detector ID is outside the crystal geometry")
        if bool((pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError("a PET LOR must join two different crystals")
        return pairs.to(dtype=torch.int64)

    def _bins(self, bins, tof, count):
        if (bins is None) != (tof is None):
            raise ValueError("tof_bins and TOFParameters must be provided together")
        if bins is None:
            return None
        bins = torch.as_tensor(bins)
        if bins.shape != (count,) or bins.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("tof_bins must be a signed integer vector with one entry per LOR")
        if bins.numel() and (bool((bins < -32768).any()) or bool((bins > 32767).any())):
            raise ValueError("parallelproj TOF bins must fit in int16")
        return bins

    def _chunks(self, pairs, num_splits):
        if isinstance(num_splits, bool) or not isinstance(num_splits, Integral) or num_splits < 1:
            raise ValueError("num_splits must be a positive integer")
        count = len(pairs)
        chunk = min(self.max_rays_per_batch // self.sampling.samples_per_lor,
                    max(1, ceil(count / num_splits)))
        context = nullcontext()
        if self.device.type == "cuda":
            import cupy
            # parallelproj allocates output arrays through CuPy. Select the
            # same device as the tensors, including for an explicit cuda:1.
            context = cupy.cuda.Device(self.device.index)
        with context:
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                endpoints = sample_endpoints(pairs[start:stop].to(self.device),
                                            self._faces, self._axes, self._sizes, self.sampling)
                yield start, stop, endpoints[0].reshape(-1, 3), endpoints[1].reshape(-1, 3)

    def sample_rays(self, detector_pairs):
        """Inspect sampled endpoints (N,K,3); use small inputs to bound memory."""
        pairs = self._pairs(detector_pairs).to(self.device)
        return sample_endpoints(pairs, self._faces, self._axes, self._sizes, self.sampling)

    def _tof_args(self, tof, bins, start, stop):
        return (tof.bin_width_mm,
                torch.tensor([tof.sigma_mm], dtype=torch.float32, device=self.device),
                torch.tensor([tof.center_offset_mm], dtype=torch.float32, device=self.device),
                tof.n_sigmas,
                bins[start:stop].to(device=self.device, dtype=torch.int32).repeat_interleave(self.sampling.samples_per_lor))

    def forward(self, image, detector_pairs, *, tof_bins=None, tof=None, num_splits=1):
        image = torch.as_tensor(image)
        if image.requires_grad:
            raise ValueError("use backward() for the explicit adjoint; autograd is not implemented")
        if tuple(image.shape) != self.image_shape:
            raise ValueError(f"image must have shape {self.image_shape}")
        image = image.to(device=self.device, dtype=torch.float32).contiguous()
        pairs = self._pairs(detector_pairs)
        bins = self._bins(tof_bins, tof, len(pairs))
        result = torch.empty(len(pairs), dtype=torch.float32, device=self.device)
        for start, stop, p1, p2 in self._chunks(pairs, num_splits):
            if tof is None:
                rays = parallelproj.joseph3d_fwd(p1, p2, image, self._origin, self._voxel)
            else:
                rays = parallelproj.joseph3d_fwd_tof_lm(
                    p1, p2, image, self._origin, self._voxel, *self._tof_args(tof, bins, start, stop))
            result[start:stop] = rays.reshape(stop - start, self.sampling.samples_per_lor).mean(1)
        return result

    def backward(self, values, detector_pairs, *, tof_bins=None, tof=None, num_splits=1):
        pairs = self._pairs(detector_pairs)
        bins = self._bins(tof_bins, tof, len(pairs))
        values = torch.as_tensor(values)
        if values.requires_grad:
            raise ValueError("use forward() for the adjoint of backward(); autograd is not implemented")
        if values.numel() == 1:
            values = values.reshape(1).expand(len(pairs))
        if values.shape != (len(pairs),):
            raise ValueError("values must be scalar or have one entry per LOR")
        result = torch.zeros(self.image_shape, dtype=torch.float32, device=self.device)
        for start, stop, p1, p2 in self._chunks(pairs, num_splits):
            ray_values = values[start:stop].to(device=self.device, dtype=torch.float32)
            ray_values = ray_values.repeat_interleave(self.sampling.samples_per_lor) / self.sampling.samples_per_lor
            if tof is None:
                result += parallelproj.joseph3d_back(
                    p1, p2, self.image_shape, self._origin, self._voxel, ray_values)
            else:
                result += parallelproj.joseph3d_back_tof_lm(
                    p1, p2, self.image_shape, self._origin, self._voxel, ray_values,
                    *self._tof_args(tof, bins, start, stop))
        return result

    def sensitivity(self, detector_pairs, weights=1., *, num_splits=1):
        """Non-TOF A_all^T weights using the same sampled endpoint model.

        Pass ALL physically accepted pairs, or a deliberately weighted sample;
        measured event pairs alone do not define the PET sensitivity.
        """
        return self.backward(weights, detector_pairs, num_splits=num_splits)

    def voxel_weights(self, detector_pair, *, tof_bin=None, tof=None):
        """Nonzero flat voxel indices and coefficients of ONE row of A.

        Obtains the row as A^T e_i; returns tensors on the projector device.
        Intended for diagnostics or building small explicit matrices.
        """
        pair = torch.as_tensor(detector_pair).reshape(1, 2)
        bins = None if tof_bin is None else torch.as_tensor([tof_bin])
        row = self.backward(1., pair, tof_bins=bins, tof=tof).flatten()
        indices = row.nonzero().flatten()
        return indices, row[indices]
