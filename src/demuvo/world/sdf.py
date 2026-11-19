"""Reading a Gazebo world - what a robot would meet, seen from above.

A `.world` or `.sdf` file gives the obstacles of a demuvo world. Only what lies between
`FLOOR` and `REACH` metres above the ground counts, as if the world were cut flat at
the height of a robot: a roof or a ceiling lies above the cut and a floor below it, so
neither is an obstacle. A box, cylinder or sphere that stands in that band becomes one
obstacle. A mesh (`.dae`, `.stl` or `.obj`) is cut at the same heights, and what is
left becomes boxes. A model that the file includes is read too, from the folders that
Gazebo searches. The world has no robot, as in an empty Gazebo world; the sandbox adds
one.

Whatever cannot be read, such as a mesh that is missing, gives a warning that says what
and why. Nothing is skipped without one.
"""

from __future__ import annotations

import math
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from ..verify import WorldError
from .footprint import slice_shapes
from .meshes import Part, read_mesh

SUFFIXES = (".world", ".sdf")
# A shape whose top is below this is floor, and one whose bottom is above the next
# height is overhead; both are left out. Both in metres.
FLOOR = 0.1
REACH = 1.0
# A shape tilted more than this, in radians, is not a flat obstacle.
TILT = 0.2
# Metres of free floor left round the shapes.
MARGIN = 1.0
# The side of a cell, in metres, when a mesh is cut; a wall thinner than this is lost.
MESH_CELL = 0.05
# A shut-in gap of this many square metres or less in a mesh is filled, so a shelf
# drawn as four walls is solid. A room is larger and stays free.
MESH_FILL = 2.5
# A shut-in gap narrower than this many metres is filled too: the space between the
# two faces of a wall, where no robot fits.
MESH_NARROW = 0.6
# The most shapes a mesh may give before the cells are made coarser, and the coarsest.
MAX_SHAPES = 3000
COARSEST_CELL = 0.4
# How deep one included model may include another.
MAX_INCLUDES = 8


def _pose(element, report) -> np.ndarray:
    """The 4 by 4 transform of the `<pose>` child of an element; identity if none."""
    pose = element.find("pose")
    move = np.eye(4)
    if pose is None or not (pose.text or "").strip():
        return move
    numbers = [float(v) for v in pose.text.split()]
    x, y, z, roll, pitch, yaw = [*numbers, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:6]
    if pose.get("degrees", "false").lower() == "true":
        roll, pitch, yaw = (math.radians(a) for a in (roll, pitch, yaw))
    form = pose.get("rotation_format", "euler_rpy")
    if form != "euler_rpy":
        report(f"a pose with rotation format '{form}' was read without its rotation")
        roll = pitch = yaw = 0.0
    if pose.get("relative_to"):
        report("a pose relative_to a frame was read as relative to its parent")
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    move[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    move[:3, 3] = (x, y, z)
    return move


def _extent(obstacle) -> tuple[float, float, float, float]:
    """The lowest and highest x and y a shape reaches."""
    shape = obstacle["shape"]
    if shape["name"] == "polygon":
        xs, ys = zip(*shape["vertices"])
        return min(xs), min(ys), max(xs), max(ys)
    x, y, yaw = obstacle["pose"]
    if shape["name"] == "circle":
        half_x = half_y = shape["radius"]
    else:
        width, depth = shape["size"]
        half_x = (abs(width * math.cos(yaw)) + abs(depth * math.sin(yaw))) / 2
        half_y = (abs(width * math.sin(yaw)) + abs(depth * math.cos(yaw))) / 2
    return x - half_x, y - half_y, x + half_x, y + half_y


class _Reader:
    """One reading of a file: the shapes it found and what it could not read.

    Attributes:
        obstacles: the obstacle tables found from boxes, cylinders and spheres.
        triangles: the triangles of each mesh part found, in world metres, and whether
            the part is closed, so a solid.
        problems: what could not be read, each told once.
        left_out: how many shapes the heights or the tilt left out, by reason.
    """

    def __init__(self, path: Path, kind: str) -> None:
        """Start a reading of `path` that takes the `kind` shapes of each link."""
        self.path, self.kind = path, kind
        self.obstacles: list[dict] = []
        self.triangles: list[tuple[np.ndarray, bool]] = []
        self.problems: dict[str, None] = {}
        self.left_out = {"above": 0, "below": 0, "tilted": 0}
        self._meshes: dict[Path, list[Part] | None] = {}
        self._open: list[Path] = []

    def problem(self, text: str) -> None:
        """Note something that could not be read; the same text is kept once."""
        self.problems[text] = None

    # -- walking the file --
    def visit(
        self,
        element,
        outer: np.ndarray,
        base: Path,
        pose: np.ndarray | None = None,
    ) -> None:
        """Read an element and what it holds, placed by `outer`.

        `pose` replaces the pose of a model, as an `<include>` may ask.
        """
        tag = element.tag
        if tag in ("sdf", "world"):
            here = outer
        elif tag == "model":
            here = outer @ (pose if pose is not None else _pose(element, self.problem))
        elif tag == "link":
            here = outer @ _pose(element, self.problem)
            for shape in element.findall(self.kind):
                self.shape(shape, here @ _pose(shape, self.problem), base)
            return
        else:
            return
        for child in element:
            if child.tag == "include":
                self.include(child, here, base)
            else:
                self.visit(child, here, base, pose if tag == "sdf" else None)

    def include(self, element, outer: np.ndarray, base: Path) -> None:
        """Read the model an `<include>` names, if it can be found."""
        uri = (element.findtext("uri") or "").strip()
        if not uri:
            self.problem("an <include> with no <uri> was left out")
            return
        found = _find(uri, base)
        if isinstance(found, list):
            self.problem(_missing(f"include '{uri}'", found))
            return
        file = _model_file(found)
        if file is None:
            self.problem(f"include '{uri}': {found} holds no model.sdf or model.config")
            return
        if file in self._open or len(self._open) >= MAX_INCLUDES:
            self.problem(f"include '{uri}' includes itself or goes too deep; left out")
            return
        try:
            root = ET.fromstring(file.read_text(encoding="utf-8").lstrip())
        except (OSError, ET.ParseError) as error:
            self.problem(f"include '{uri}': {file} cannot be read: {error}")
            return
        pose = (
            _pose(element, self.problem) if element.find("pose") is not None else None
        )
        self._open.append(file)
        try:
            self.visit(root, outer, file.parent, pose)
        finally:
            self._open.pop()

    # -- shapes --
    def shape(self, element, placed: np.ndarray, base: Path) -> None:
        """Read the geometry of one collision or visual."""
        geometry = element.find("geometry")
        if geometry is None or geometry.find("plane") is not None:
            return  # a ground plane is the floor, which is not an obstacle
        mesh = geometry.find("mesh")
        if mesh is not None:
            self.mesh(mesh, placed, base)
            return
        found = self.primitive(geometry, placed)
        if found is not None:
            self.obstacles.append(found)

    def primitive(self, geometry, placed: np.ndarray):
        """The obstacle of a box, cylinder or sphere at a pose, or None."""
        if geometry.find("box") is not None:
            sx, sy, sz = (float(v) for v in geometry.findtext("box/size").split())
            shape = {"name": "box", "size": [sx, sy]}
        elif geometry.find("cylinder") is not None:
            radius = float(geometry.findtext("cylinder/radius"))
            sz = float(geometry.findtext("cylinder/length"))
            shape = {"name": "circle", "radius": radius}
        elif geometry.find("sphere") is not None:
            radius = float(geometry.findtext("sphere/radius"))
            sz = radius * 2
            shape = {"name": "circle", "radius": radius}
        else:
            kinds = [child.tag for child in geometry] or ["empty"]
            self.problem(f"a '{kinds[0]}' shape cannot be read and was left out")
            return None
        x, y, z = placed[:3, 3]
        yaw = math.atan2(placed[1, 0], placed[0, 0])
        tilt = math.acos(max(-1.0, min(1.0, placed[2, 2])))
        if tilt > TILT:
            self.left_out["tilted"] += 1
        elif z + sz / 2 <= FLOOR:
            self.left_out["below"] += 1
        elif z - sz / 2 >= REACH:
            self.left_out["above"] += 1
        else:
            return {"pose": [float(x), float(y), yaw], "shape": shape}
        return None

    def mesh(self, element, placed: np.ndarray, base: Path) -> None:
        """Add the triangles of a mesh, scaled and placed, to be cut later."""
        uri = (element.findtext("uri") or "").strip()
        if not uri:
            self.problem("a <mesh> with no <uri> was left out")
            return
        found = _find(uri, base)
        if isinstance(found, list):
            self.problem(_missing(f"mesh '{uri}'", found))
            return
        if found not in self._meshes:
            try:
                self._meshes[found] = read_mesh(found)
            except WorldError as error:
                self.problem(f"{error}; mesh '{uri}' was left out")
                self._meshes[found] = None
        parts = self._meshes[found]
        if parts is None:
            return
        name = (element.findtext("submesh/name") or "").strip()
        if name:
            parts = [p for p in parts if name in p.names]
            if not parts:
                self.problem(f"mesh '{uri}' has no part named '{name}'; left out")
                return
        scale = [float(v) for v in (element.findtext("scale") or "1 1 1").split()]
        scale = np.array((scale * 3)[:3] if len(scale) == 1 else scale[:3])
        for part in parts:
            moved = (part.triangles * scale) @ placed[:3, :3].T + placed[:3, 3]
            self.triangles.append((moved, part.closed))

    # -- the result --
    def cut(self) -> list[dict]:
        """The obstacles that the meshes leave between the heights, as tables."""
        if not self.triangles:
            return []
        cell = MESH_CELL
        while True:
            try:
                shapes, specks = slice_shapes(
                    self.triangles,
                    (FLOOR, REACH),
                    cell,
                    largest_hole=MESH_FILL,
                    narrowest=MESH_NARROW,
                )
            except ValueError as error:
                self.problem(f"the meshes were left out: {error}")
                return []
            if len(shapes) <= MAX_SHAPES or cell >= COARSEST_CELL:
                break
            cell *= 2
        if cell != MESH_CELL:
            self.problem(
                f"the meshes need more than {MAX_SHAPES} shapes at {MESH_CELL} m, so "
                f"cells of {cell} m were used and walls thinner than that are lost"
            )
        if len(shapes) > MAX_SHAPES:
            self.problem(
                f"the meshes give {len(shapes)} shapes; the world may run slowly"
            )
        if specks:
            self.problem(
                f"{specks} specks of the meshes, under two cells, were dropped"
            )
        if not shapes:
            self.problem(
                f"the meshes have nothing between {FLOOR} and {REACH} m above the floor"
            )
        return shapes

    def left_out_text(self) -> str | None:
        """What the heights and the tilt left out, in words; None if nothing."""
        reasons = {
            "above": f"above {REACH:g} m (a roof or ceiling)",
            "below": f"lower than {FLOOR:g} m (a floor)",
            "tilted": "tilted",
        }
        parts = [f"{n} {reasons[k]}" for k, n in self.left_out.items() if n]
        if not parts:
            return None
        return f"{sum(self.left_out.values())} shapes were left out: " + ", ".join(
            parts
        )


def _roots(base: Path) -> list[Path]:
    """The folders that hold models, in the order Gazebo looks: near the file first."""
    roots: list[Path] = []
    for folder in (base, base.parent, base.parent.parent):
        roots += [folder, folder / "models"]
    for name in (
        "GZ_SIM_RESOURCE_PATH",
        "IGN_GAZEBO_RESOURCE_PATH",
        "GAZEBO_MODEL_PATH",
        "SDF_PATH",
    ):
        roots += [Path(p) for p in os.environ.get(name, "").split(os.pathsep) if p]
    home = Path.home()
    roots += [home / ".gz" / "models", home / ".gazebo" / "models"]
    roots += [home / ".ignition" / "models"]
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix:
            share = Path(prefix) / "share"
            roots += [share, *sorted(share.glob("*/models"))]
    roots += sorted(Path("/usr/share").glob("gazebo*/models"))
    return list(dict.fromkeys(roots))


def _fuel(uri: str) -> list[Path]:
    """Where a Fuel model of this address sits in the local caches."""
    parts = [unquote(p) for p in uri.split("://", 1)[1].split("/") if p]
    # host/1.0/owner/models/name[/version]
    if len(parts) < 5 or parts[3] != "models":
        return []
    owner, name = parts[2].lower(), parts[4].lower()
    home = Path.home()
    folders = [
        home / ".gz" / "fuel" / parts[0] / owner / "models" / name,
        home / ".ignition" / "fuel" / parts[0] / owner / "models" / name,
    ]
    return [folder / parts[5] if len(parts) > 5 else folder for folder in folders]


def _find(uri: str, base: Path) -> Path | list[Path]:
    """The file or folder a URI names, or the list of places that were searched."""
    if uri.startswith("model://"):
        tried = [root / uri[8:] for root in _roots(base)]
    elif uri.startswith("package://"):
        tried = [root / uri[10:] for root in _roots(base)]
    elif uri.startswith(("http://", "https://")):
        tried = _fuel(uri)
    elif uri.startswith("file://"):
        tried = [Path(uri[7:])]
    else:
        tried = [base / uri]
    for place in tried:
        if place.exists():
            return place
    return tried


def _model_file(place: Path) -> Path | None:
    """The `model.sdf` of a model folder, with its versions, or the file itself."""
    if place.is_file():
        return place
    versions = [p for p in place.iterdir() if p.is_dir() and p.name.isdigit()]
    if not (place / "model.config").exists() and versions:
        place = max(versions, key=lambda p: int(p.name))
    config = place / "model.config"
    if config.exists():
        try:
            names = [e.text.strip() for e in ET.parse(config).getroot().iter("sdf")]
        except (OSError, ET.ParseError):
            names = []
        if names and (place / names[-1]).exists():
            return place / names[-1]
    sdf = place / "model.sdf"
    return sdf if sdf.exists() else None


def _missing(what: str, tried: list[Path]) -> str:
    """The message for something that was not found, with where it was looked for."""
    shown = ", ".join(str(p) for p in tried[:3])
    more = f" and {len(tried) - 3} more places" if len(tried) > 3 else ""
    return (
        f"{what} was not found, so it was left out (looked in {shown}{more}); put "
        f"the folder that holds it in GZ_SIM_RESOURCE_PATH"
    )


def _read(path: Path, kind: str, root) -> _Reader:
    """One reading of the file for the shapes of one kind."""
    reader = _Reader(path, kind)
    reader.visit(root, np.eye(4), path.resolve().parent)
    return reader


def read_sdf(path: str | Path) -> dict:
    """The data of a demuvo world file for a Gazebo world, cut at a robot's height.

    Args:
        path: a `.world` or `.sdf` file.

    Returns:
        The same dictionary that `tomllib` gives for a world file: its version,
        a `world` table sized to hold the shapes and the `obstacle` list.

    Raises:
        WorldError: if the file is not valid XML or holds no shape to stand on.
        OSError: if the file cannot be read.

    Warns:
        UserWarning: for each thing that could not be read or was left out, with the
            reason.
    """
    path = Path(path)
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8").lstrip())
    except ET.ParseError as error:
        raise WorldError(f"file: not valid XML: {error}") from error
    reader = _read(path, "collision", root)
    boxes = reader.cut()
    if not reader.obstacles and not boxes:
        # a file with no solid shape of its own: the world is what it looks like
        seen = _read(path, "visual", root)
        look = seen.cut()
        if seen.obstacles or look:
            seen.problems = {**reader.problems, **seen.problems}
            seen.problem(
                "the file has no collision shape that stands between "
                f"{FLOOR:g} and {REACH:g} m, so its visual shapes were used"
            )
            reader, boxes = seen, look
        else:
            reader.problems.update(seen.problems)
    obstacles = [*reader.obstacles, *boxes]
    note = reader.left_out_text()
    messages = [*reader.problems, *([note] if note else [])]
    if not obstacles:
        extra = f" ({'; '.join(messages)})" if messages else ""
        raise WorldError(
            "file: no box, cylinder, sphere or mesh stands on the floor "
            f"between {FLOOR:g} and {REACH:g} m{extra}"
        )
    for message in messages:
        warnings.warn(f"{path.name}: {message}", stacklevel=2)
    extents = [_extent(o) for o in obstacles]
    # Rounded first, so float noise in a turned box does not add a whole metre.
    x0 = math.floor(round(min(e[0] for e in extents) - MARGIN, 3))
    y0 = math.floor(round(min(e[1] for e in extents) - MARGIN, 3))
    x1 = math.ceil(round(max(e[2] for e in extents) + MARGIN, 3))
    y1 = math.ceil(round(max(e[3] for e in extents) + MARGIN, 3))
    return {
        "version": 1,
        "world": {
            "size": [float(x1 - x0), float(y1 - y0)],
            "origin": [float(x0), float(y0)],
            "seed": 1,
        },
        "obstacle": obstacles,
    }
