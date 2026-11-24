#!/usr/bin/env python3
"""Create the scanner LUT for the four-panel GATE macro.

The output LUT is a NumPy array with shape ``(n_detectors, 3)``:

    scanner_lut[detector_id] = [x_mm, y_mm, z_mm]

Rather than hard-coding numbers, the geometry is read straight from

    macros/geometry.mac          (crystal box + cubicArray, volume names)
    data/module.placements       (the 360 module placements it points at)

so the LUT cannot drift from the simulated geometry. Re-run this script
after editing either file.

GATE stores the attached PETscanner system levels as:

    level1ID = module copy ID   (0 .. 359, the row order of module.placements)
    level2ID = crystal copy ID  (0 .. 143, the 12 x 12 cubicArray inside a module)

For downstream reconstruction, convert GATE IDs to a flat detector ID with:

    detector_id = detector_id_from_gate_ids(level1ID, level2ID)
                = level1ID * n_crystals_per_module + level2ID

Two GATE conventions matter here and both were verified against the module
and crystal IDs in a real Hits tree (``output/hits.root``, 10177 hits, 100%
agreement on all 360 modules); see ``--validate-root``:

1.  cubicArray copy numbering runs x fastest, then y, then z:

        copy_id = ix + nx * (iy + ny * iz)

    With this macro's 1 x 12 x 12 array that is ``copy_id = iy + 12 * iz``.

2.  The rotation in a genericRepeater placements row is the *passive*
    (Geant4 G4PVPlacement) rotation, so the local -> global matrix is its
    transpose. A "90 deg about z" row therefore maps a local offset
    (x, y, z) to (y, -x, z), NOT to (-y, x, z).

Note (2) is the opposite of what scripts/make_placements.py's docstring
assumes. It has no physical consequence -- a module is one crystal thick and
symmetric about its depth axis, so the plates land in the same place either
way -- but it does mirror the in-plane crystal index, which is exactly what
this LUT has to get right.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LENGTH_UNITS_MM = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0}
ANGLE_UNITS_DEG = {"deg": 1.0, "degree": 1.0, "rad": 180.0 / math.pi, "radian": 180.0 / math.pi}


# --------------------------------------------------------------------------- #
# Macro parsing
# --------------------------------------------------------------------------- #
class Macro:
    """A GATE macro reduced to a list of (command, args) pairs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.commands: list[tuple[str, list[str]]] = []
        for raw in self.path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            self.commands.append((parts[0], parts[1:]))

    def get(self, command: str, default: list[str] | None = None) -> list[str] | None:
        """Last occurrence of ``command`` wins, as in GATE."""

        found = default
        for cmd, args in self.commands:
            if cmd == command:
                found = args
        return found

    def require(self, command: str) -> list[str]:
        args = self.get(command)
        if args is None:
            raise ValueError(f"{self.path}: missing command '{command}'")
        return args

    def find(self, suffix: str) -> list[tuple[str, list[str]]]:
        return [(c, a) for c, a in self.commands if c.endswith(suffix)]


def length_mm(args: list[str]) -> float:
    """Parse ``<value> [unit]`` into millimeters."""

    unit = LENGTH_UNITS_MM[args[1]] if len(args) > 1 else 1.0
    return float(args[0]) * unit


def vector_mm(args: list[str]) -> np.ndarray:
    """Parse ``<x> <y> <z> [unit]`` into a millimeter vector."""

    unit = LENGTH_UNITS_MM[args[3]] if len(args) > 3 else 1.0
    return np.asarray([float(v) for v in args[:3]], dtype=np.float64) * unit


def as_bool(args: list[str]) -> bool:
    token = args[0].lower()
    if token in ("1", "true", "t", "yes"):
        return True
    if token in ("0", "false", "f", "no"):
        return False
    raise ValueError(f"cannot read '{args[0]}' as a boolean")


def box_size_mm(macro: Macro, volume: str) -> np.ndarray:
    return np.asarray(
        [
            length_mm(macro.require(f"/gate/{volume}/geometry/set{axis}Length"))
            for axis in ("X", "Y", "Z")
        ],
        dtype=np.float64,
    )


def attached_volumes(macro: Macro) -> tuple[str, dict[int, str]]:
    """Return the system name and the {level: volume} map it attaches."""

    system = None
    levels: dict[int, str] = {}
    for cmd, args in macro.commands:
        parts = cmd.strip("/").split("/")
        if len(parts) == 5 and parts[:2] == ["gate", "systems"] and parts[3].startswith("level"):
            if parts[4] != "attach":
                continue
            system = parts[2]
            levels[int(parts[3][len("level"):])] = args[0]
    if system is None or not levels:
        raise ValueError(f"{macro.path}: no '/gate/systems/<system>/levelN/attach' commands found")
    return system, levels


# --------------------------------------------------------------------------- #
# Placements file
# --------------------------------------------------------------------------- #
@dataclass
class Placements:
    """Module placements read from a GATE genericRepeater placements file."""

    path: Path
    angles_deg: np.ndarray          # (n,)
    axes: np.ndarray                # (n, 3)
    translations_mm: np.ndarray     # (n, 3)
    labels: list[str]               # (n,) plate name per module, from '#' headers

    def __len__(self) -> int:
        return len(self.angles_deg)


def parse_placements(path: Path) -> Placements:
    """Parse ``time angle axis_x axis_y axis_z tx ty tz`` rows.

    Comment lines starting with ``#`` are skipped, except that a header of the
    form ``###### <name> plate`` is kept as the label for the rows that follow
    (used only to group modules for plotting/metadata).
    """

    angle_scale, length_scale = 1.0, 1.0
    angles, axes, translations, labels = [], [], [], []
    label = "unlabelled"

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            words = line.lstrip("#").strip().split()
            if len(words) == 2 and words[1] == "plate":
                label = words[0]
            continue
        words = line.split()
        head = words[0].lower()
        if head == "rotation":
            angle_scale = ANGLE_UNITS_DEG[words[1]]
            continue
        if head == "translation":
            length_scale = LENGTH_UNITS_MM[words[1]]
            continue
        if head == "time":
            continue
        if len(words) < 8:
            raise ValueError(f"{path}: expected 8 columns, got {len(words)}: {line!r}")
        angles.append(float(words[1]) * angle_scale)
        axes.append([float(v) for v in words[2:5]])
        translations.append([float(v) * length_scale for v in words[5:8]])
        labels.append(label)

    if not angles:
        raise ValueError(f"{path}: no placement rows found")
    return Placements(
        path=Path(path),
        angles_deg=np.asarray(angles, dtype=np.float64),
        axes=np.asarray(axes, dtype=np.float64),
        translations_mm=np.asarray(translations, dtype=np.float64),
        labels=labels,
    )


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Active (right-handed) rotation about ``axis`` by ``angle_deg``."""

    norm = float(np.linalg.norm(axis))
    if norm == 0.0 or angle_deg == 0.0:
        return np.eye(3)
    u = np.asarray(axis, dtype=np.float64) / norm
    theta = math.radians(angle_deg)
    K = np.asarray(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]], dtype=np.float64
    )
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class PanelPETGeometry:
    """Everything the LUT needs, read from geometry.mac + module.placements."""

    macro_path: Path
    placements_path: Path
    system: str
    level1_volume: str
    level2_volume: str

    module_size_mm: np.ndarray            # (3,) module box, local frame
    crystal_size_mm: np.ndarray           # (3,) crystal box, local frame
    repeat_number: np.ndarray             # (3,) cubicArray counts (nx, ny, nz)
    repeat_vector_mm: np.ndarray          # (3,) cubicArray pitch
    auto_center: bool

    module_centers_mm: np.ndarray         # (n_modules, 3) world frame
    module_rotations: np.ndarray          # (n_modules, 3, 3) local -> world
    module_labels: list[str] = field(default_factory=list)

    @property
    def n_modules(self) -> int:
        return int(self.module_centers_mm.shape[0])

    @property
    def n_crystals_per_module(self) -> int:
        return int(np.prod(self.repeat_number))

    @property
    def n_detectors(self) -> int:
        return self.n_modules * self.n_crystals_per_module

    @property
    def depth_axis(self) -> int:
        """Local axis along which the module is thinnest (its depth)."""

        return int(np.argmin(self.module_size_mm))

    @property
    def crystal_depth_mm(self) -> float:
        return float(self.crystal_size_mm[self.depth_axis])

    def plate_slices(self) -> dict[str, tuple[int, int]]:
        """{label: (first_module_id, last_module_id)} in placement order."""

        plates: dict[str, tuple[int, int]] = {}
        for module_id, label in enumerate(self.module_labels):
            first, last = plates.get(label, (module_id, module_id))
            plates[label] = (min(first, module_id), max(last, module_id))
        return plates


def resolve_placements_path(macro_path: Path, filename: str) -> Path:
    """GATE resolves the placements path against its working directory.

    Macros here are run from the panel_pet directory, so try the CWD first and
    then walk up from the macro itself.
    """

    candidates = [
        Path(filename),
        macro_path.parent / filename,
        macro_path.parent.parent / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"placements file '{filename}' not found; tried:\n  {tried}")


def load_geometry(macro_path: Path, placements_path: Path | None = None) -> PanelPETGeometry:
    """Build the geometry description from the GATE macro."""

    macro = Macro(macro_path)
    system, levels = attached_volumes(macro)
    if set(levels) != {1, 2}:
        raise ValueError(
            f"{macro_path}: expected exactly level1 and level2 attachments, got {sorted(levels)}"
        )
    module, crystal = levels[1], levels[2]

    filename = macro.require(f"/gate/{module}/genericRepeater/setPlacementsFilename")[0]
    if placements_path is None:
        placements_path = resolve_placements_path(Path(macro_path), filename)
    placements = parse_placements(placements_path)

    relative = macro.get(f"/gate/{module}/genericRepeater/useRelativeTranslation")
    if relative is not None and as_bool(relative):
        raise NotImplementedError(
            "useRelativeTranslation 1 is not supported; this LUT assumes the "
            "placement translations are absolute in the PETscanner frame."
        )

    repeat_number = np.asarray(
        [
            int(float(macro.require(f"/gate/{crystal}/cubicArray/setRepeatNumber{axis}")[0]))
            for axis in ("X", "Y", "Z")
        ],
        dtype=np.int64,
    )
    repeat_vector = vector_mm(macro.require(f"/gate/{crystal}/cubicArray/setRepeatVector"))
    center_args = macro.get(f"/gate/{crystal}/cubicArray/autoCenter") or macro.get(
        f"/gate/{crystal}/cubicArray/setAutoCenter"
    )
    auto_center = True if center_args is None else as_bool(center_args)

    # GATE placements carry the passive (G4PVPlacement) rotation, so the
    # local -> world matrix is the transpose. Verified against hits.root.
    rotations = np.stack(
        [
            rotation_matrix(placements.axes[i], placements.angles_deg[i]).T
            for i in range(len(placements))
        ]
    )

    return PanelPETGeometry(
        macro_path=Path(macro_path).resolve(),
        placements_path=Path(placements_path).resolve(),
        system=system,
        level1_volume=module,
        level2_volume=crystal,
        module_size_mm=box_size_mm(macro, module),
        crystal_size_mm=box_size_mm(macro, crystal),
        repeat_number=repeat_number,
        repeat_vector_mm=repeat_vector,
        auto_center=auto_center,
        module_centers_mm=placements.translations_mm,
        module_rotations=rotations,
        module_labels=placements.labels,
    )


# --------------------------------------------------------------------------- #
# ID <-> position
# --------------------------------------------------------------------------- #
def centered_positions(n: int, pitch_mm: float, auto_center: bool = True) -> np.ndarray:
    """Positions of ``n`` copies along one GATE cubicArray axis."""

    offsets = np.arange(n, dtype=np.float64) * pitch_mm
    if auto_center:
        offsets -= (n - 1) / 2.0 * pitch_mm
    return offsets


def crystal_local_offsets(geom: PanelPETGeometry) -> np.ndarray:
    """Crystal centers in the module frame, indexed by GATE copy ID.

    GATE's cubicArray repeater loops z outermost and x innermost, so

        copy_id = ix + nx * (iy + ny * iz)
    """

    nx, ny, nz = (int(v) for v in geom.repeat_number)
    x = centered_positions(nx, geom.repeat_vector_mm[0], geom.auto_center)
    y = centered_positions(ny, geom.repeat_vector_mm[1], geom.auto_center)
    z = centered_positions(nz, geom.repeat_vector_mm[2], geom.auto_center)

    offsets = np.empty((nx * ny * nz, 3), dtype=np.float64)
    offsets[:, 0] = np.tile(x, ny * nz)
    offsets[:, 1] = np.tile(np.repeat(y, nx), nz)
    offsets[:, 2] = np.repeat(z, nx * ny)
    return offsets


def detector_id_from_gate_ids(
    module_id: np.ndarray | int,
    crystal_id: np.ndarray | int,
    geom: PanelPETGeometry,
) -> np.ndarray:
    """Convert GATE level1ID/level2ID to a flat scanner LUT detector ID."""

    module = np.asarray(module_id, dtype=np.int64)
    crystal = np.asarray(crystal_id, dtype=np.int64)
    return module * geom.n_crystals_per_module + crystal


def gate_ids_from_detector_id(
    detector_id: np.ndarray | int, geom: PanelPETGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`detector_id_from_gate_ids`."""

    ids = np.asarray(detector_id, dtype=np.int64)
    return ids // geom.n_crystals_per_module, ids % geom.n_crystals_per_module


def module_depth_axes(geom: PanelPETGeometry) -> np.ndarray:
    """Unit vector along each module's depth, in the world frame."""

    local = np.zeros(3)
    local[geom.depth_axis] = 1.0
    return geom.module_rotations @ local


def build_scanner_lut(geom: PanelPETGeometry, *, depth_mode: str = "center") -> np.ndarray:
    """Build the detector-center LUT in millimeters.

    Parameters
    ----------
    depth_mode:
        ``center`` uses crystal geometric centers. ``frontface`` moves each
        coordinate half a crystal depth toward the FOV, onto the face the
        module presents to the isocenter.
    """

    if depth_mode not in ("center", "frontface"):
        raise ValueError("depth_mode must be 'center' or 'frontface'")

    offsets = crystal_local_offsets(geom)                       # (n_cry, 3)
    # (n_mod, n_cry, 3) = rotate each local offset, then translate.
    positions = np.einsum("mij,cj->mci", geom.module_rotations, offsets)
    positions += geom.module_centers_mm[:, None, :]

    if depth_mode == "frontface":
        axes = module_depth_axes(geom)                          # (n_mod, 3)
        outward = np.sign(np.einsum("mi,mi->m", axes, geom.module_centers_mm))
        shift = -outward[:, None] * axes * (geom.crystal_depth_mm / 2.0)
        positions += shift[:, None, :]

    return positions.reshape(geom.n_detectors, 3).astype(np.float32)


def build_module_lut(geom: PanelPETGeometry, *, depth_mode: str = "center") -> np.ndarray:
    """Module-center coordinates, indexed by level1ID (for visualization)."""

    centers = geom.module_centers_mm.copy()
    if depth_mode == "frontface":
        axes = module_depth_axes(geom)
        outward = np.sign(np.einsum("mi,mi->m", axes, centers))
        centers -= outward[:, None] * axes * (geom.crystal_depth_mm / 2.0)
    return centers.astype(np.float32)


def detector_plate_labels(geom: PanelPETGeometry) -> np.ndarray:
    """Plate label for every detector ID (shape ``(n_detectors,)``)."""

    return np.repeat(np.asarray(geom.module_labels), geom.n_crystals_per_module)


# --------------------------------------------------------------------------- #
# Validation against a GATE Hits tree
# --------------------------------------------------------------------------- #
def validate_against_root(
    lut: np.ndarray, geom: PanelPETGeometry, root_path: Path, tree: str = "Hits"
) -> float:
    """Check that every hit lands inside the crystal its LUT entry names.

    Returns the largest hit-to-LUT distance in mm. Anything below the crystal
    half-diagonal means the ID conventions above are right.
    """

    import uproot

    with uproot.open(root_path) as handle:
        hits = handle[tree]
        arrays = hits.arrays(
            ["level1ID", "level2ID", "posX", "posY", "posZ"], library="np"
        )
    if arrays["level1ID"].size == 0:
        raise ValueError(f"{root_path}:{tree} has no entries")

    module_id = arrays["level1ID"].astype(np.int64)
    crystal_id = arrays["level2ID"].astype(np.int64)
    if module_id.max() >= geom.n_modules or crystal_id.max() >= geom.n_crystals_per_module:
        raise ValueError(
            f"{root_path} IDs exceed this geometry "
            f"(level1 max {module_id.max()} / {geom.n_modules}, "
            f"level2 max {crystal_id.max()} / {geom.n_crystals_per_module})"
        )

    positions = np.stack([arrays["posX"], arrays["posY"], arrays["posZ"]], axis=1)
    expected = lut[detector_id_from_gate_ids(module_id, crystal_id, geom)]
    distance = np.linalg.norm(positions - expected, axis=1)

    half_diagonal = float(np.linalg.norm(geom.crystal_size_mm) / 2.0)
    inside = float(np.mean(distance <= half_diagonal + 1e-3))
    print(
        f"Validation vs {root_path}:{tree}: {len(distance)} hits, "
        f"max offset {distance.max():.3f} mm, crystal half-diagonal "
        f"{half_diagonal:.3f} mm, inside {100.0 * inside:.2f}%"
    )
    if inside < 1.0:
        raise SystemExit(
            "LUT does NOT reproduce the simulated crystal positions -- the ID "
            "conventions in this script no longer match GATE."
        )
    return float(distance.max())


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def set_axes_equal(ax) -> None:
    """Use an equal scale on all 3D axes."""

    limits = np.asarray([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    radius = 0.5 * float(np.ptp(limits, axis=1).max())
    middles = limits.mean(axis=1)

    ax.set_xlim3d([middles[0] - radius, middles[0] + radius])
    ax.set_ylim3d([middles[1] - radius, middles[1] + radius])
    ax.set_zlim3d([middles[2] - radius, middles[2] + radius])


def draw_plate_outline(ax, points: np.ndarray, color: str) -> None:
    """Draw a wireframe rectangle around one plate's crystals."""

    spans = np.ptp(points, axis=0)
    normal = int(np.argmin(spans))
    u, v = [axis for axis in range(3) if axis != normal]

    corners = np.zeros((5, 3))
    corners[:, normal] = points[:, normal].mean()
    lo_u, hi_u = points[:, u].min(), points[:, u].max()
    lo_v, hi_v = points[:, v].min(), points[:, v].max()
    corners[:, u] = [lo_u, hi_u, hi_u, lo_u, lo_u]
    corners[:, v] = [lo_v, lo_v, hi_v, hi_v, lo_v]
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=2.0)


def plot_detector_setup(
    lut: np.ndarray,
    out_png: Path,
    geom: PanelPETGeometry,
    *,
    depth_mode: str = "center",
    max_crystals: int = 60000,
) -> None:
    """Save a 3D plot of crystals, module centers, and plate outlines."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plate_of_detector = detector_plate_labels(geom)
    module_lut = build_module_lut(geom, depth_mode=depth_mode)
    module_labels = np.asarray(geom.module_labels)
    rng = np.random.default_rng(20260804)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plates = list(geom.plate_slices())
    per_plate = max(1, max_crystals // max(1, len(plates)))
    for index, label in enumerate(plates):
        color = colors[index % len(colors)]
        crystals = lut[plate_of_detector == label]
        draw_plate_outline(ax, crystals, color=color)
        if len(crystals) > per_plate:
            crystals = crystals[rng.choice(len(crystals), per_plate, replace=False)]
        modules = module_lut[module_labels == label]

        ax.scatter(
            crystals[:, 0], crystals[:, 1], crystals[:, 2],
            s=1.5, alpha=0.22, color=color, label=f"crystals: {label}",
        )
        ax.scatter(
            modules[:, 0], modules[:, 1], modules[:, 2],
            s=30, marker="s", edgecolors="black", linewidths=0.4, color=color,
            label=f"module centers: {label}",
        )

    ax.scatter([0.0], [0.0], [0.0], s=45, marker="+", color="black", label="isocenter")
    ax.set_title(f"Four-Panel PET Scanner LUT ({geom.n_detectors} crystals)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=22, azim=-55)
    set_axes_equal(ax)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def save_metadata(
    out_json: Path, out_npy: Path, geom: PanelPETGeometry, *, depth_mode: str
) -> None:
    """Write metadata next to the LUT so reconstruction scripts know the ID map."""

    n_cry = geom.n_crystals_per_module
    plates = {
        label: {
            "n_modules": last - first + 1,
            "module_id_range": [first, last],
            "detector_id_range": [first * n_cry, (last + 1) * n_cry - 1],
        }
        for label, (first, last) in geom.plate_slices().items()
    }

    metadata = {
        "lut_file": str(out_npy),
        "lut_shape": [geom.n_detectors, 3],
        "units": "mm",
        "gate_macro": str(geom.macro_path),
        "placements_file": str(geom.placements_path),
        "gate_system": geom.system,
        "gate_id_hierarchy": {
            "level1ID": f"{geom.level1_volume}_id",
            "level2ID": f"{geom.level2_volume}_id",
        },
        "flat_detector_id_formula": "level1ID * n_crystals_per_module + level2ID",
        "crystal_copy_order": "ix + nx * (iy + ny * iz)  (GATE cubicArray: x fastest, z slowest)",
        "placement_rotation_convention": (
            "passive / G4PVPlacement: local -> world matrix is R(axis, angle).T"
        ),
        "depth_mode": depth_mode,
        "n_modules": geom.n_modules,
        "n_crystals_per_module": n_cry,
        "n_detectors": geom.n_detectors,
        "module_size_mm": geom.module_size_mm.tolist(),
        "crystal_size_mm": geom.crystal_size_mm.tolist(),
        "crystal_repeat_number": geom.repeat_number.tolist(),
        "crystal_repeat_vector_mm": geom.repeat_vector_mm.tolist(),
        "crystal_auto_center": geom.auto_center,
        "plates": plates,
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create a scanner LUT matching macros/geometry.mac."
    )
    parser.add_argument(
        "--macro",
        type=Path,
        default=here.parent / "macros" / "geometry.mac",
        help="GATE geometry macro to read the scanner from.",
    )
    parser.add_argument(
        "--placements",
        type=Path,
        default=None,
        help="Module placements file. Defaults to the one named in the macro.",
    )
    parser.add_argument(
        "--out", type=Path, default=here / "scanner_lut.npy",
        help="Output .npy scanner LUT path.",
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="Output JSON metadata path. Defaults to OUT with .json suffix.",
    )
    parser.add_argument(
        "--depth-mode", choices=("center", "frontface"), default="center",
        help="Use crystal geometric centers or the faces looking at the FOV.",
    )
    parser.add_argument(
        "--validate-root", type=Path, default=None,
        help="GATE ROOT file with a Hits tree; check the LUT reproduces its "
             "level1ID/level2ID positions (requires uproot).",
    )
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="Output 3D geometry PNG. Defaults to OUT with '_geometry.png' suffix.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Only save the LUT/metadata, without creating a visualization.",
    )
    parser.add_argument(
        "--max-plot-crystals", type=int, default=60000,
        help="Maximum number of crystal points to scatter in the 3D plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_npy = args.out.resolve()
    out_json = args.metadata.resolve() if args.metadata else out_npy.with_suffix(".json")
    out_plot = args.plot.resolve() if args.plot else out_npy.with_name(f"{out_npy.stem}_geometry.png")

    geom = load_geometry(args.macro.resolve(), args.placements)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    lut = build_scanner_lut(geom, depth_mode=args.depth_mode)
    np.save(out_npy, lut)
    save_metadata(out_json, out_npy, geom, depth_mode=args.depth_mode)
    if not args.no_plot:
        plot_detector_setup(
            lut, out_plot, geom,
            depth_mode=args.depth_mode, max_crystals=args.max_plot_crystals,
        )

    print(f"Macro: {geom.macro_path}")
    print(f"Placements: {geom.placements_path}")
    print(f"Saved scanner LUT: {out_npy}")
    print(f"Saved metadata: {out_json}")
    if not args.no_plot:
        print(f"Saved geometry plot: {out_plot}")
    print(f"LUT shape: {lut.shape}")
    print(
        f"Detectors: {geom.n_modules} modules x "
        f"{geom.n_crystals_per_module} crystals = {geom.n_detectors}"
    )
    for label, (first, last) in geom.plate_slices().items():
        print(f"  {label:>8s} plate: modules {first}-{last} ({last - first + 1})")
    for axis, name in enumerate("XYZ"):
        print(f"{name} range: {lut[:, axis].min():.3f} to {lut[:, axis].max():.3f} mm")

    if args.validate_root is not None:
        validate_against_root(lut, geom, args.validate_root.resolve())


if __name__ == "__main__":
    main()
