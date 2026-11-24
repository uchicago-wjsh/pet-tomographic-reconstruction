"""Stateless endpoint sampling: a fixed matrix across iterations and subsets."""
from dataclasses import asdict, dataclass
from numbers import Integral

import torch


@dataclass(frozen=True)
class SamplingConfig:
    """K independent paired endpoint samples, not a K-by-K Cartesian product.

    By default each endpoint is uniform over its entrance face and depth.
    ``depth_mu_per_mm`` selects a normalized truncated exponential in depth,
    measured along the crystal normal; it is not an oblique photon transport
    model. Disabling a dimension puts it at its geometric midpoint.
    """

    samples_per_lor: int = 16
    seed: int = 0
    sample_face: bool = True
    sample_depth: bool = True
    depth_mu_per_mm: float | None = None

    def __post_init__(self):
        if isinstance(self.samples_per_lor, bool) or not isinstance(self.samples_per_lor, Integral) or self.samples_per_lor < 1:
            raise ValueError("samples_per_lor must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.depth_mu_per_mm is not None:
            import math
            if not math.isfinite(self.depth_mu_per_mm) or self.depth_mu_per_mm <= 0:
                raise ValueError("depth_mu_per_mm must be finite and positive")
            if not self.sample_depth:
                raise ValueError("depth_mu_per_mm requires sample_depth=True")

    def metadata(self):
        return {"algorithm": "pair_hash32_v1", **asdict(self)}


def _hash32(value):
    # All intermediate products fit in signed int64, on CPU and CUDA.
    value = value & 0xFFFFFFFF
    value = (((value >> 16) ^ value) * 0x45D9F3B) & 0xFFFFFFFF
    value = (((value >> 16) ^ value) * 0x45D9F3B) & 0xFFFFFFFF
    return ((value >> 16) ^ value) & 0xFFFFFFFF


def sample_endpoints(pairs, faces, axes, sizes, config):
    """Return (start, end), each (n_pairs, K, 3), on the geometry device.

    Samples depend only on the unordered detector pair, sample number,
    coordinate, and seed. Reordering, duplicates, chunking, and subsets
    therefore cannot change the matrix. Reversing a pair swaps its endpoints.
    """
    ordered = torch.sort(pairs, dim=1).values
    lo, hi = ordered[:, 0, None], ordered[:, 1, None]
    sample = torch.arange(config.samples_per_lor, device=pairs.device, dtype=torch.int64)[None, :]
    key = (_hash32(lo) ^ _hash32(hi + 0x9E3779B9)
           ^ _hash32(sample + config.seed + 0x85EBCA6B))
    endpoints = []
    for arm in range(2):
        ids = ordered[:, arm]
        uniform = torch.stack([
            (_hash32(key + (arm * 3 + coordinate + 1) * 0x9E3779B9).double() + .5) / 2**32
            for coordinate in range(3)
        ], dim=-1).to(torch.float32)
        if not config.sample_face:
            uniform[:, :, :2] = .5
        if not config.sample_depth:
            uniform[:, :, 2] = .5
        elif config.depth_mu_per_mm is not None:
            mu = config.depth_mu_per_mm
            depth = sizes[ids, 2, None]
            uniform[:, :, 2] = -torch.log1p(
                uniform[:, :, 2] * torch.expm1(-mu * depth)
            ) / (mu * depth)
        local = (uniform - uniform.new_tensor([.5, .5, 0.])) * sizes[ids, None, :]
        endpoints.append(faces[ids, None, :] + torch.einsum("nij,nkj->nki", axes[ids], local))
    swapped = (pairs[:, 0] > pairs[:, 1])[:, None, None]
    return (torch.where(swapped, endpoints[1], endpoints[0]).contiguous(),
            torch.where(swapped, endpoints[0], endpoints[1]).contiguous())
