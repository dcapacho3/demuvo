"""Reading the triangles of a mesh file - Collada (`.dae`), STL and OBJ.

A mesh is a list of named parts, each a stack of triangles in metres, with z up. A
Collada file's `up_axis` is not read: Gazebo stands the TurtleBot3 meshes upright as
they are written, though they say `Y_UP`, so z is up here too. The Gazebo reader cuts
the parts at the heights a robot meets; see `footprint`.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..verify import WorldError

# The deepest chain of nodes that instance other nodes; a loop in a file stops here.
MAX_DEPTH = 32


@dataclass(frozen=True)
class Part:
    """One piece of a mesh.

    Attributes:
        names: every name the piece goes by in its file, for a `<submesh>` to pick it.
        triangles: the corners of each triangle, shape (N, 3, 3), in metres.
        closed: whether the surface has no gap, so the piece is a solid.
    """

    names: frozenset
    triangles: np.ndarray
    closed: bool = False


def is_closed(triangles: np.ndarray) -> bool:
    """Whether every edge of the triangles is shared by exactly two of them.

    Corners at the same place count as one, so a mesh that repeats a corner for each
    face is still seen as one surface.
    """
    if len(triangles) == 0:
        return False
    _, ids = np.unique(
        np.round(triangles.reshape(-1, 3), 5), axis=0, return_inverse=True
    )
    ids = ids.reshape(-1, 3)
    edges = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def read_mesh(path: str | Path) -> list[Part]:
    """The parts of a `.dae`, `.stl` or `.obj` file.

    Raises:
        WorldError: if the kind is not one of these, or the file cannot be read or
            holds nothing demuvo understands. The message names the file.
    """
    path = Path(path)
    readers = {".dae": _read_dae, ".stl": _read_stl, ".obj": _read_obj}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise WorldError(
            f"{path.name}: demuvo reads .dae, .stl and .obj meshes, not "
            f"'{path.suffix or 'a file with no suffix'}'"
        )
    try:
        parts = [
            Part(p.names, p.triangles, is_closed(p.triangles)) for p in reader(path)
        ]
    except OSError as error:
        raise WorldError(f"{path}: cannot be read: {error}") from error
    except (ValueError, KeyError, IndexError, ET.ParseError) as error:
        raise WorldError(f"{path}: is not a mesh demuvo can read: {error}") from error
    return parts


def _read_stl(path: Path) -> list[Part]:
    """An STL file, binary or text: one part."""
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * count:
            record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            found = np.frombuffer(data, dtype=record, count=count, offset=84)
            return [Part(frozenset(), found["v"].astype(float))]
    text = data.decode("utf-8", errors="replace")
    corners = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not corners or len(corners) % 3:
        raise ValueError("it holds no whole triangle")
    return [Part(frozenset(), np.array(corners, dtype=float).reshape((-1, 3, 3)))]


def _read_obj(path: Path) -> list[Part]:
    """An OBJ file: a part for each `o` or `g` group, or one for the whole."""
    vertices: list[list[float]] = []
    groups: dict[str, list[tuple[int, int, int]]] = {"": []}
    current = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "v":
            vertices.append([float(w) for w in words[1:4]])
        elif words[0] in ("o", "g") and len(words) > 1:
            current = words[1]
            groups.setdefault(current, [])
        elif words[0] == "f":
            ids = [int(w.split("/")[0]) for w in words[1:]]
            ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
            for k in range(1, len(ids) - 1):
                groups[current].append((ids[0], ids[k], ids[k + 1]))
    points = np.array(vertices)
    parts = [
        Part(frozenset({name} if name else ()), points[np.array(faces)])
        for name, faces in groups.items()
        if faces
    ]
    if not parts:
        raise ValueError("it holds no face")
    return parts


def _floats(element) -> np.ndarray:
    """The numbers in the text of an element."""
    return np.array((element.text or "").split(), dtype=float)


def _ints(element) -> np.ndarray:
    """The whole numbers in the text of an element."""
    return np.array((element.text or "").split(), dtype=int)


def _fan(indices: np.ndarray, starts: np.ndarray, size: int) -> np.ndarray:
    """Triangles from the polygons of `size` corners that start at `starts`."""
    return np.concatenate(
        [
            np.stack(
                [indices[starts], indices[starts + k], indices[starts + k + 1]], axis=1
            )
            for k in range(1, size - 1)
        ]
    )


def _triangle_ids(prim, kind: str, ns: str, stride: int, offset: int) -> np.ndarray:
    """The vertex numbers of the triangles of a Collada primitive, three to a row."""
    if kind == "triangles":
        ids = np.concatenate([_ints(p) for p in prim.findall(ns + "p")])
        return ids.reshape(-1, stride)[:, offset].reshape(-1, 3)
    if kind == "polylist":
        counts = _ints(prim.find(ns + "vcount"))
        ids = _ints(prim.find(ns + "p")).reshape(-1, stride)[:, offset]
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return np.concatenate(
            [_fan(ids, first[counts == n], n) for n in np.unique(counts) if n >= 3]
        )
    polygons = (_ints(p).reshape(-1, stride)[:, offset] for p in prim.findall(ns + "p"))
    return np.concatenate(
        [_fan(ids, np.array([0]), len(ids)) for ids in polygons if len(ids) >= 3]
    )


def _positions(mesh, vertex, ns: str) -> np.ndarray:
    """The corner positions of a mesh that a primitive's VERTEX input points at."""
    sources = {s.get("id"): s for s in mesh.findall(ns + "source")}
    vertex_sets = {v.get("id"): v for v in mesh.findall(ns + "vertices")}
    origin = vertex_sets[vertex.get("source").lstrip("#")]
    position = next(
        i for i in origin.findall(ns + "input") if i.get("semantic") == "POSITION"
    )
    source = sources[position.get("source").lstrip("#")]
    accessor = source.find(f"{ns}technique_common/{ns}accessor")
    width = int(accessor.get("stride", "3")) if accessor is not None else 3
    return _floats(source.find(ns + "float_array")).reshape(-1, width)[:, :3]


def _geometry(mesh, ns: str) -> np.ndarray:
    """The triangles of a Collada `<mesh>`, as corner positions in its own units."""
    found = []
    for prim in mesh:
        kind = prim.tag.removeprefix(ns)
        if kind in ("tristrips", "trifans"):
            raise ValueError(f"<{kind}> are not supported")
        if kind not in ("triangles", "polylist", "polygons"):
            continue
        inputs = prim.findall(ns + "input")
        stride = 1 + max(int(i.get("offset", "0")) for i in inputs)
        vertex = next(i for i in inputs if i.get("semantic") == "VERTEX")
        ids = _triangle_ids(prim, kind, ns, stride, int(vertex.get("offset", "0")))
        found.append(_positions(mesh, vertex, ns)[ids])
    if not found:
        raise ValueError("a geometry holds no triangle")
    return np.concatenate(found)


def _matrix(node, ns: str) -> np.ndarray:
    """The 4 by 4 transform of a Collada node, its steps taken in file order."""
    total = np.eye(4)
    for step in node:
        kind = step.tag.removeprefix(ns)
        if kind == "matrix":
            move = _floats(step).reshape(4, 4)
        elif kind == "translate":
            move = np.eye(4)
            move[:3, 3] = _floats(step)
        elif kind == "scale":
            move = np.diag([*_floats(step), 1.0])
        elif kind == "rotate":
            x, y, z, degrees = _floats(step)
            move = np.eye(4)
            move[:3, :3] = _rotation((x, y, z), np.radians(degrees))
        elif kind in ("lookat", "skew"):
            raise ValueError(f"<{kind}> in a node is not supported")
        else:
            continue
        total = total @ move
    return total


def _rotation(axis, angle: float) -> np.ndarray:
    """The rotation of `angle` about a vector (Rodrigues' formula)."""
    x, y, z = np.array(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _read_dae(path: Path) -> list[Part]:
    """A Collada file: a part for each geometry a node of the scene places."""
    root = ET.parse(path).getroot()
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    unit = root.find(f"{ns}asset/{ns}unit")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    meshes = {
        g.get("id"): (g, g.find(ns + "mesh"))
        for g in root.iter(ns + "geometry")
        if g.find(ns + "mesh") is not None
    }
    library = {n.get("id"): n for n in root.iter(ns + "node") if n.get("id")}
    cache: dict[str, np.ndarray] = {}
    parts: list[Part] = []

    def place(node, above: np.ndarray, depth: int) -> None:
        """Add what a node and its children instance, under the transform above."""
        if depth > MAX_DEPTH:
            raise ValueError("nodes instance each other in a loop")
        here = above @ _matrix(node, ns)
        for child in node.findall(ns + "node"):
            place(child, here, depth + 1)
        for instance in node.findall(ns + "instance_node"):
            target = library.get(instance.get("url", "").lstrip("#"))
            if target is None:
                raise ValueError(
                    f"an instanced node, {instance.get('url')}, is missing"
                )
            place(target, here, depth + 1)
        for instance in node.findall(ns + "instance_geometry"):
            ident = instance.get("url", "").lstrip("#")
            if ident not in meshes:
                raise ValueError(f"a geometry, {instance.get('url')}, is missing")
            holder, mesh = meshes[ident]
            if ident not in cache:
                cache[ident] = _geometry(mesh, ns)
            moved = cache[ident] @ here[:3, :3].T + here[:3, 3]
            names = {node.get("name"), node.get("id"), ident, holder.get("name")}
            parts.append(
                Part(
                    frozenset(n for n in names if n),
                    moved * scale,
                )
            )

    scene = root.find(f"{ns}library_visual_scenes/{ns}visual_scene")
    if scene is not None:
        for node in scene.findall(ns + "node"):
            place(node, np.eye(4), 0)
    else:
        for ident, (holder, mesh) in meshes.items():
            names = {ident, holder.get("name")}
            parts.append(
                Part(
                    frozenset(n for n in names if n),
                    _geometry(mesh, ns) * scale,
                )
            )
    if not parts:
        raise ValueError("it places no geometry")
    return parts
