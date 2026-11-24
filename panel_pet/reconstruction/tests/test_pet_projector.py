"""Numerical contract tests against the actual parallelproj kernels."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import parallelproj
import torch

from pet_projector import CrystalGeometry, CrystalProjector, SamplingConfig, TOFParameters
from pet_projector.cli import add_projector_arguments, validate_projector_arguments
from pet_projector.pytomography import FiniteCrystalPETLMSystemMatrix
from make_crystal_geometry import geometry_from_panel_macro
import pytomography
from pytomography.algorithms import MLEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta


def geometry():
    centers = np.array([[-20, -3, -1], [-20, 0, 0], [-20, 3, 1],
                        [20, -3, 1], [20, 0, 0], [20, 3, -1]], dtype=float)
    axes = np.repeat(np.eye(3)[None], 6, axis=0)
    axes[:, :, 0] = [0, 1, 0]
    axes[:, :, 1] = [0, 0, 1]
    axes[:, :, 2] = np.array([[-1, 0, 0]] * 3 + [[1, 0, 0]] * 3)
    return CrystalGeometry.from_centers(centers, axes, [4, 4, 5])


def pairs():
    return torch.tensor([[i, j] for i in range(3) for j in range(3, 6)])


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.geom = geometry()
        self.pairs = pairs()
        self.projector = CrystalProjector(
            self.geom, (9, 11, 7), (1.5, 1, 2),
            sampling=SamplingConfig(16, seed=125), max_rays_per_batch=32)
        rng = np.random.default_rng(91)
        self.image = torch.tensor(rng.uniform(.1, 1, self.projector.image_shape), dtype=torch.float32)
        self.values = torch.tensor(rng.normal(size=len(self.pairs)), dtype=torch.float32)
        self.bins = torch.tensor([-2, -1, 0, 1, 2, -1, 1, 0, -2])
        self.tof = TOFParameters(1.7, 2.3, .3, 3.)

    def test_center_limit_matches_parallelproj(self):
        projector = CrystalProjector(self.geom, self.projector.image_shape, (1.5, 1, 2),
                                     sampling=SamplingConfig(5, sample_face=False, sample_depth=False))
        centers = torch.tensor(self.geom.centers_mm, dtype=torch.float32)
        p1, p2 = centers[self.pairs[:, 0]], centers[self.pairs[:, 1]]
        for tof in (None, self.tof):
            with self.subTest(tof=tof):
                kwargs = {} if tof is None else {"tof_bins": self.bins, "tof": tof}
                if tof is None:
                    reference = parallelproj.joseph3d_fwd(p1, p2, self.image, projector._origin, projector._voxel)
                    back = parallelproj.joseph3d_back(p1, p2, projector.image_shape, projector._origin, projector._voxel, self.values)
                else:
                    args = (tof.bin_width_mm, torch.tensor([tof.sigma_mm]),
                            torch.tensor([tof.center_offset_mm]), tof.n_sigmas, self.bins)
                    reference = parallelproj.joseph3d_fwd_tof_lm(p1, p2, self.image, projector._origin, projector._voxel, *args)
                    back = parallelproj.joseph3d_back_tof_lm(p1, p2, projector.image_shape, projector._origin, projector._voxel, self.values, *args)
                torch.testing.assert_close(projector.forward(self.image, self.pairs, **kwargs), reference, rtol=2e-6, atol=1e-6)
                torch.testing.assert_close(projector.backward(self.values, self.pairs, **kwargs), back, rtol=3e-6, atol=2e-6)

    def test_forward_is_explicit_average_of_sampled_rays(self):
        p1, p2 = self.projector.sample_rays(self.pairs)
        for tof in (None, self.tof):
            with self.subTest(tof=tof):
                kwargs = {} if tof is None else {"tof_bins": self.bins, "tof": tof}
                if tof is None:
                    reference = parallelproj.joseph3d_fwd(p1.reshape(-1, 3), p2.reshape(-1, 3),
                                                        self.image, self.projector._origin, self.projector._voxel)
                else:
                    reference = parallelproj.joseph3d_fwd_tof_lm(
                        p1.reshape(-1, 3), p2.reshape(-1, 3), self.image,
                        self.projector._origin, self.projector._voxel,
                        tof.bin_width_mm, torch.tensor([tof.sigma_mm]),
                        torch.tensor([tof.center_offset_mm]), tof.n_sigmas,
                        self.bins.repeat_interleave(16))
                reference = reference.reshape(-1, 16).mean(1)
                torch.testing.assert_close(self.projector.forward(self.image, self.pairs, **kwargs), reference)

    def test_adjoint_identity_non_tof_and_tof(self):
        for tof in (None, self.tof):
            with self.subTest(tof=tof):
                kwargs = {} if tof is None else {"tof_bins": self.bins, "tof": tof}
                lhs = torch.dot(self.projector.forward(self.image, self.pairs, **kwargs), self.values)
                rhs = (self.image * self.projector.backward(self.values, self.pairs, **kwargs)).sum()
                torch.testing.assert_close(lhs, rhs, rtol=5e-6, atol=5e-6)

    def test_reordering_duplicates_reversal_and_chunks(self):
        order = torch.tensor([8, 2, 2, 0, 7, 4, 1, 6, 3, 5])
        p1, p2 = self.projector.sample_rays(self.pairs)
        q1, q2 = self.projector.sample_rays(self.pairs[order])
        torch.testing.assert_close(q1, p1[order], rtol=0, atol=0)
        torch.testing.assert_close(q2, p2[order], rtol=0, atol=0)
        r1, r2 = self.projector.sample_rays(self.pairs.flip(1))
        torch.testing.assert_close(r1, p2, rtol=0, atol=0)
        torch.testing.assert_close(r2, p1, rtol=0, atol=0)
        a = self.projector.forward(self.image, self.pairs, tof_bins=self.bins, tof=self.tof)
        b = self.projector.forward(self.image, self.pairs[order], tof_bins=self.bins[order], tof=self.tof, num_splits=20)
        torch.testing.assert_close(b, a[order], rtol=0, atol=0)
        reverse_tof = TOFParameters(self.tof.bin_width_mm, self.tof.sigma_mm,
                                    -self.tof.center_offset_mm, self.tof.n_sigmas)
        reverse = self.projector.forward(self.image, self.pairs.flip(1), tof_bins=-self.bins, tof=reverse_tof)
        torch.testing.assert_close(a, reverse, rtol=3e-6, atol=2e-6)

    def test_samples_inside_crystals_and_distributed_independently(self):
        projector = CrystalProjector(self.geom, (3, 3, 3), 1, sampling=SamplingConfig(8192), max_rays_per_batch=8192)
        starts, ends = projector.sample_rays([[1, 4]])
        local = []
        for points, detector in ((starts, 1), (ends, 4)):
            coords = (points[0].numpy() - self.geom.front_face_centers_mm[detector]) @ self.geom.axes[detector]
            normalized = coords / self.geom.size_mm[detector]
            normalized[:, :2] += .5
            self.assertTrue(((normalized >= 0) & (normalized <= 1)).all())
            np.testing.assert_allclose(normalized.mean(0), .5, atol=.015)
            np.testing.assert_allclose(normalized.std(0), np.sqrt(1/12), atol=.015)
            local.append(normalized)
        corr = np.corrcoef(np.concatenate(local, axis=1).T) - np.eye(6)
        self.assertLess(np.abs(corr).max(), .06)
        # Increasing K preserves earlier samples, enabling convergence studies.
        small = self.projector.sample_rays([[1, 4]])
        other = CrystalProjector(self.geom, self.projector.image_shape, 1,
                                 sampling=SamplingConfig(32, seed=125))
        large = other.sample_rays([[1, 4]])
        torch.testing.assert_close(small[0], large[0][:, :16], rtol=0, atol=0)

    def test_exponential_depth(self):
        mu = .2
        projector = CrystalProjector(self.geom, (3, 3, 3), 1,
                                     sampling=SamplingConfig(8192, depth_mu_per_mm=mu),
                                     max_rays_per_batch=8192)
        p1, _ = projector.sample_rays([[1, 4]])
        depth = ((p1[0].numpy() - self.geom.front_face_centers_mm[1]) @ self.geom.axes[1])[:, 2]
        expected_mean = 1/mu - 5/np.expm1(mu*5)
        self.assertAlmostEqual(float(depth.mean()), expected_mean, delta=.05)

    def test_voxel_weights_reproduce_forward_and_sensitivity(self):
        rows = []
        for i, pair in enumerate(self.pairs):
            indices, weights = self.projector.voxel_weights(pair, tof_bin=int(self.bins[i]), tof=self.tof)
            row = torch.zeros(self.image.numel())
            row[indices] = weights
            rows.append(row)
        matrix = torch.stack(rows)
        torch.testing.assert_close(matrix @ self.image.flatten(),
                                   self.projector.forward(self.image, self.pairs, tof_bins=self.bins, tof=self.tof),
                                   rtol=4e-6, atol=2e-6)
        torch.testing.assert_close((matrix.T @ self.values).reshape(self.image.shape),
                                   self.projector.backward(self.values, self.pairs, tof_bins=self.bins, tof=self.tof),
                                   rtol=4e-6, atol=2e-6)
        weights = torch.arange(1., len(self.pairs) + 1)
        torch.testing.assert_close(self.projector.sensitivity(self.pairs, weights),
                                   self.projector.backward(weights, self.pairs))

    def test_finite_crystal_changes_voxel_contributions(self):
        indices, weights = self.projector.voxel_weights([1, 4])
        center = CrystalProjector(self.geom, self.projector.image_shape, (1.5, 1, 2),
                                  sampling=SamplingConfig(1, sample_face=False, sample_depth=False))
        center_indices, _ = center.voxel_weights([1, 4])
        self.assertGreater(len(indices), len(center_indices))
        self.assertTrue((weights > 0).all())

    def test_empty_singleton_and_invalid_inputs(self):
        empty = torch.empty((0, 2), dtype=torch.int64)
        self.assertEqual(self.projector.forward(self.image, empty).numel(), 0)
        self.assertEqual(float(self.projector.backward(torch.empty(0), empty).sum()), 0.)
        self.assertEqual(self.projector.forward(self.image, [[1, 4]]).shape, (1,))
        for bad in ([[0, 0]], [[-1, 4]], [[1, 6]], [[1., 4.]], [[0, 1, 2]]):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                self.projector.forward(self.image, bad)
        with self.assertRaises(ValueError):
            self.projector.forward(self.image, self.pairs, tof=self.tof)
        with self.assertRaises(ValueError):
            self.projector.forward(self.image, self.pairs, tof=self.tof, tof_bins=self.bins + 40000)
        with self.assertRaises(ValueError):
            self.projector.forward(self.image.requires_grad_(), self.pairs)

    def test_geometry_export_rotations_and_roundtrip(self):
        root = Path(__file__).resolve().parents[2]
        panel = geometry_from_panel_macro(root / "macros/geometry.mac")
        import scanner_lut
        expected = scanner_lut.build_scanner_lut(scanner_lut.load_geometry(root / "macros/geometry.mac"))
        np.testing.assert_allclose(panel.centers_mm, expected, atol=1e-10)
        self.assertEqual(panel.n_detectors, 51840)
        self.assertTrue((np.einsum("ni,ni->n", panel.axes[:, :, 2], panel.centers_mm) > 0).all())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "geometry.npz"
            panel.save(path)
            self.assertEqual(panel.fingerprint, CrystalGeometry.load(path).fingerprint)
        with self.assertRaises(ValueError):
            CrystalGeometry(self.geom.front_face_centers_mm, self.geom.axes * 2, [4, 4, 5])


class PyTomographyTests(unittest.TestCase):
    def matrix(self, tof=False, **kwargs):
        self.geom = geometry()
        self.accepted = pairs()
        events = self.accepted[torch.tensor([0, 0, 1, 3, 4, 4, 6, 7, 8])]
        tof_meta = PETTOFMeta(11, 22., 5.) if tof else None
        if tof:
            events = torch.cat([events, torch.tensor([3, 4, 5, 6, 7, 4, 5, 6, 3])[:, None]], 1)
        self.events = events
        metadata = PETLMProjMeta(events, scanner_LUT=torch.tensor(self.geom.centers_mm, dtype=torch.float32),
                                 tof_meta=tof_meta, detector_ids_sensitivity=self.accepted)
        return FiniteCrystalPETLMSystemMatrix(
            ObjectMeta(dr=(1.5, 1, 2), shape=(9, 11, 7)), metadata,
            crystal_geometry=self.geom, sampling=SamplingConfig(8), N_splits=12,
            max_rays_per_batch=16, FOV_scale_enabled=False, **kwargs)

    def test_sampled_sensitivity_and_subsets(self):
        for tof in (False, True):
            with self.subTest(tof=tof):
                matrix = self.matrix(tof)
                image = torch.ones(matrix.object_meta.shape, device=pytomography.device)
                expected = matrix.crystal_projector.sensitivity(self.accepted)
                expected[expected < 1e-7] = 1e7
                torch.testing.assert_close(matrix.norm_BP.to(expected.device), expected)
                matrix.set_n_subsets(4)
                full = matrix.forward(image)
                for subset, indices in enumerate(matrix.subset_indices_array):
                    torch.testing.assert_close(matrix.forward(image, subset), full[indices])
                    values = torch.linspace(-1., 1., len(indices), device=pytomography.device)
                    back, norm = matrix.backward(values, subset, return_norm_constant=True)
                    torch.testing.assert_close(torch.dot(full[indices], values), (image * back).sum(),
                                               rtol=5e-6, atol=5e-6)
                    torch.testing.assert_close(norm, matrix.norm_BP.to(norm.device) * len(indices) / len(full))

    def test_mlem_produces_finite_count_normalized_image(self):
        matrix = self.matrix()
        initial = torch.ones(matrix.object_meta.shape, device=pytomography.device)
        final = MLEM(PoissonLogLikelihood(matrix), initial)(n_iters=3)
        self.assertTrue(bool(torch.isfinite(final).all()))
        self.assertTrue(bool((final >= 0).all()))
        expected_detected = matrix.crystal_projector.forward(final, self.accepted).sum()
        self.assertAlmostEqual(float(expected_detected), len(self.events), delta=.003)

    def test_pair_weights_and_psf_preserve_adjoint(self):
        from pytomography.transforms.shared import GaussianFilter
        matrix = self.matrix(True, obj2obj_transforms=[GaussianFilter(1.2)],
                             scale_projection_by_sensitivity=True)
        matrix.proj_meta.weights = torch.linspace(.5, 1.5, len(self.events))
        image = torch.rand(matrix.object_meta.shape, generator=torch.Generator().manual_seed(2)).to(pytomography.device)
        values = torch.linspace(-1., 1., len(self.events), device=pytomography.device)
        lhs = torch.dot(matrix.forward(image), values)
        rhs = (image * matrix.backward(values)).sum()
        torch.testing.assert_close(lhs, rhs, rtol=5e-6, atol=5e-6)

    def test_incompatible_cached_center_ray_options_rejected(self):
        parser = argparse.ArgumentParser()
        add_projector_arguments(parser)
        for name in ("sensitivity_map", "system_matrix_dir", "uniform_sensitivity"):
            args = parser.parse_args(["--lor-samples", "8", "--crystal-geometry", "frames.npz"])
            setattr(args, name, True)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_projector_arguments(args)
        with self.assertRaises(ValueError):
            self.matrix(attenuation_map=torch.ones(9, 11, 7))

    def test_reconstruction_entrypoints_with_sparse_mc_coverage(self):
        import recon_mlem
        import reconstruct_mlem_pytomography

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geom = geometry()
            geom.save(root / "crystals.npz")
            np.save(root / "lut.npy", geom.centers_mm.astype(np.float32))
            np.save(root / "panels.npy", [0, 0, 0, 1, 1, 1])
            np.save(root / "mc.npy", np.full((7, 7, 7), .01, dtype=np.float32))
            np.savez(root / "lm.npz", detector_ids=pairs().numpy(), tof_seconds=np.zeros(9))
            for module in (recon_mlem, reconstruct_mlem_pytomography):
                with self.subTest(entrypoint=module.__name__):
                    destination = root / module.__name__
                    args = [module.__name__, "--listmode", str(root / "lm.npz"),
                            "--lut", str(root / "lut.npy"), "--panel-ids", str(root / "panels.npy"),
                            "--output-dir", str(destination), "--grid-size", "7", "--voxel-size-mm", "2",
                            "--n-iters", "2", "--n-splits", "12", "--tof-num-bins", "11",
                            "--tof-bin-width-ps", "10", "--tof-fwhm-ps", "40", "--lor-samples", "8",
                            "--crystal-geometry", str(root / "crystals.npz"), "--lor-batch-rays", "16"]
                    if module is recon_mlem:
                        args += ["--mc-sensitivity-map", str(root / "mc.npy"),
                                 "--mc-sensitivity-voxel-mm", "2", "--mask-border-voxels", "0"]
                    with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                    image = np.load(destination / "reconstruction_mlem_tof.npy")
                    self.assertTrue(np.isfinite(image).all())
                    self.assertGreater(image.max(), 0)
                    meta = json.loads((destination / "reconstruction_meta.json").read_text())
                    self.assertEqual(meta["finite_crystal_projector"]["sampling"]["samples_per_lor"], 8)
                    self.assertEqual(meta["finite_crystal_projector"]["geometry_sha256"], geom.fingerprint)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is unavailable")
class CUDATests(unittest.TestCase):
    def test_cuda_matches_cpu_and_adjoint(self):
        g = geometry()
        config = SamplingConfig(16, seed=7)
        cpu = CrystalProjector(g, (9, 11, 7), (1.5, 1, 2), sampling=config)
        gpu_device = f"cuda:{torch.cuda.device_count() - 1}"
        gpu = CrystalProjector(g, (9, 11, 7), (1.5, 1, 2), sampling=config, device=gpu_device)
        image = torch.rand(9, 11, 7)
        values = torch.linspace(-1., 1., 9)
        for tof in (None, TOFParameters(2., 2.5)):
            kwargs = {} if tof is None else {"tof": tof, "tof_bins": torch.arange(-4, 5)}
            forward = gpu.forward(image, pairs(), **kwargs)
            backward = gpu.backward(values, pairs(), **kwargs)
            torch.testing.assert_close(forward.cpu(), cpu.forward(image, pairs(), **kwargs), rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(backward.cpu(), cpu.backward(values, pairs(), **kwargs), rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(torch.dot(forward, values.to(gpu_device)),
                                       (image.to(gpu_device) * backward).sum(), rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
