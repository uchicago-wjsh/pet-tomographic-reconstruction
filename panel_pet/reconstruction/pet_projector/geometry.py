"""Explicit crystal entrance faces, local frames, and dimensions (millimeters)."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CrystalGeometry:
    """Oriented crystal boxes, indexed by the existing flat detector IDs.

    ``axes[d, :, 0:2]`` are the two entrance-face unit vectors and
    ``axes[d, :, 2]`` points from that face into the crystal. ``size_mm``
    contains full face width, face height, and depth, in that order.
    The entrance face is centered at ``front_face_centers_mm``.
    """

    front_face_centers_mm: np.ndarray
    axes: np.ndarray
    size_mm: np.ndarray

    def __post_init__(self):
        faces = np.array(self.front_face_centers_mm, dtype=np.float64, copy=True)
        axes = np.array(self.axes, dtype=np.float64, copy=True)
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("front_face_centers_mm must have shape (n_detectors, 3)")
        if axes.shape != (len(faces), 3, 3):
            raise ValueError("axes must have shape (n_detectors, 3, 3), with basis vectors in columns")
        sizes = np.broadcast_to(np.asarray(self.size_mm, dtype=np.float64), faces.shape).copy()
        if not all(np.isfinite(a).all() for a in (faces, axes, sizes)):
            raise ValueError("crystal geometry must be finite")
        if (sizes <= 0).any():
            raise ValueError("all crystal dimensions must be positive")
        gram = np.swapaxes(axes, 1, 2) @ axes
        if not np.allclose(gram, np.eye(3), rtol=0, atol=1e-6):
            raise ValueError("each crystal frame must be orthonormal")
        for name, array in (("front_face_centers_mm", faces), ("axes", axes), ("size_mm", sizes)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_centers(cls, centers_mm, axes, size_mm):
        """Construct from geometric box centers, NOT mean interaction positions."""
        centers = np.asarray(centers_mm, dtype=np.float64)
        basis = np.asarray(axes, dtype=np.float64)
        sizes = np.broadcast_to(np.asarray(size_mm, dtype=np.float64), centers.shape)
        return cls(centers - basis[:, :, 2] * sizes[:, 2, None] / 2, basis, sizes)

    @property
    def n_detectors(self):
        return len(self.front_face_centers_mm)

    @property
    def centers_mm(self):
        return self.front_face_centers_mm + self.axes[:, :, 2] * self.size_mm[:, 2, None] / 2

    @property
    def fingerprint(self):
        digest = sha256()
        for array in (self.front_face_centers_mm, self.axes, self.size_mm):
            digest.update(np.asarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path):
        np.savez_compressed(path, front_face_centers_mm=self.front_face_centers_mm,
                            axes=self.axes, size_mm=self.size_mm)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["front_face_centers_mm"], data["axes"], data["size_mm"])
