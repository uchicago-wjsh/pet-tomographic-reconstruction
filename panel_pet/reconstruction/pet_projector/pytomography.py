"""PyTomography list-mode MLEM/OSEM adapter with sampled sensitivity."""
import numpy as np
import torch
import pytomography
from pytomography.projectors.PET import PETLMSystemMatrix

from .projector import CrystalProjector, TOFParameters


class FiniteCrystalPETLMSystemMatrix(PETLMSystemMatrix):
    """Replace center rays in BOTH event projection and all-pair sensitivity.

    Explicit ``detector_ids_sensitivity`` are required, preventing accidental
    enumeration of billions of pairs. Pair weights are supported. Image
    attenuation maps are rejected because the parent's implementation uses
    center rays; supply externally calculated pair weights instead.
    """

    def __init__(self, object_meta, proj_meta, *, crystal_geometry, sampling=None,
                 max_rays_per_batch=65536, **kwargs):
        if proj_meta.detector_ids_sensitivity is None:
            raise ValueError("provide the accepted detector_ids_sensitivity explicitly")
        if kwargs.get("attenuation_map") is not None:
            raise ValueError("finite-crystal attenuation maps are not implemented; use explicit pair weights")
        if len(proj_meta.scanner_lut) != crystal_geometry.n_detectors:
            raise ValueError("crystal geometry and detector LUT have different detector counts")
        if proj_meta.detector_ids.ndim != 2 or proj_meta.detector_ids.shape[1] != (3 if proj_meta.tof_meta is not None else 2):
            raise ValueError("detector_ids must have two columns, plus a TOF-bin column when TOF is enabled")
        # A LUT may contain mean DOI positions, rather than geometric centers.
        # Validate detector ordering by checking that each LUT point is inside
        # its corresponding crystal, without assuming a particular mean DOI.
        lut = proj_meta.scanner_lut.detach().cpu().numpy()
        local = np.einsum("nji,nj->ni", crystal_geometry.axes,
                          lut - crystal_geometry.front_face_centers_mm)
        low = -crystal_geometry.size_mm / 2
        low[:, 2] = 0
        high = crystal_geometry.size_mm / 2
        high[:, 2] = crystal_geometry.size_mm[:, 2]
        if not np.isfinite(local).all() or (local < low - 1e-3).any() or (local > high + 1e-3).any():
            raise ValueError("detector LUT points do not lie inside the corresponding crystals; check detector ordering")
        self.crystal_projector = CrystalProjector(
            crystal_geometry, object_meta.shape, object_meta.dr,
            sampling=sampling, device=pytomography.device,
            max_rays_per_batch=max_rays_per_batch)
        super().__init__(object_meta, proj_meta, **kwargs)

    def _event_args(self, subset_idx):
        ids = self.proj_meta.detector_ids
        if subset_idx is not None:
            ids = ids[self.subset_indices_array[subset_idx].cpu()]
        if not self.TOF:
            return ids[:, :2], {}
        meta = self.proj_meta.tof_meta
        tof = TOFParameters(float(meta.bin_width), float(meta.sigma.item()),
                            float(meta.center_offset.item()), float(meta.n_sigmas))
        bins = ids[:, 2] - (meta.num_bins - 1) // 2
        return ids[:, :2], {"tof_bins": bins, "tof": tof}

    def _event_weights(self, subset_idx):
        if self.proj_meta.weights is None:
            raise ValueError("scale_projection_by_sensitivity requires proj_meta.weights")
        return self.get_projection_subset(self.proj_meta.weights, subset_idx).to(self.output_device)

    def forward(self, object, subset_idx=None):
        obj = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            obj = transform.forward(obj)
        pairs, kwargs = self._event_args(subset_idx)
        result = self.crystal_projector.forward(obj, pairs, num_splits=self.N_splits, **kwargs)
        result = result.to(self.output_device)
        if self.scale_projection_by_sensitivity:
            result = result * self._event_weights(subset_idx)
        return result

    def backward(self, proj, subset_idx=None, return_norm_constant=False):
        pairs, kwargs = self._event_args(subset_idx)
        if self.scale_projection_by_sensitivity:
            proj = proj.to(self.output_device) * self._event_weights(subset_idx)
        result = self.crystal_projector.backward(proj, pairs, num_splits=self.N_splits, **kwargs)
        for transform in self.obj2obj_transforms[::-1]:
            result = transform.backward(result)
        result = result.to(self.output_device)
        if return_norm_constant:
            # norm_BP already includes the object transforms, once.
            return result, self.compute_normalization_factor(subset_idx)
        return result

    def _backward_full(self, N_splits=20):
        weights = self.proj_meta.weights_sensitivity
        if weights is None:
            weights = 1.
        norm = self.crystal_projector.sensitivity(
            self.proj_meta.detector_ids_sensitivity, weights,
            num_splits=max(self.N_splits, N_splits))
        for transform in self.obj2obj_transforms[::-1]:
            norm = transform.backward(norm)
        return norm.cpu()
