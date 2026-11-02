"""GJK - do two convex parts touch, and how deep.

GJK finds the distance between the two hulls
using only a support function - the vertex furthest along a direction. If that
distance is under the sum of the radii, the parts touch. If the hulls overlap,
GJK has no distance to give, and the separating axis test finds the shallowest
way out instead.
"""

from __future__ import annotations

import math

from ..world.shapes import Part

Vec = tuple[float, float]

_MAX_ROUNDS = 64
_RELATIVE_TOLERANCE = 1e-10


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _support(vertices: tuple[Vec, ...], direction: Vec) -> Vec:
    """The vertex furthest along the direction."""
    return max(vertices, key=lambda v: v[0] * direction[0] + v[1] * direction[1])


def _difference(a: Part, b: Part, direction: Vec) -> Vec:
    """The support point of the difference of the hulls, a minus b."""
    p = _support(a.vertices, direction)
    q = _support(b.vertices, (-direction[0], -direction[1]))
    return p[0] - q[0], p[1] - q[1]


def _on_segment(a: Vec, b: Vec) -> tuple[Vec, list[Vec]]:
    """The point of segment ab nearest the origin, and the corners it needs."""
    ab = (b[0] - a[0], b[1] - a[1])
    length2 = _dot(ab, ab)
    t = 0.0 if length2 == 0 else -_dot(a, ab) / length2
    if t <= 0:
        return a, [a]
    if t >= 1:
        return b, [b]
    return (a[0] + t * ab[0], a[1] + t * ab[1]), [a, b]


def _nearest(simplex: list[Vec]) -> tuple[Vec, list[Vec]] | None:
    """The point of the simplex nearest the origin, and the corners it needs.

    Returns None if the origin is inside a triangle.
    """
    if len(simplex) == 1:
        return simplex[0], simplex
    if len(simplex) == 2:
        return _on_segment(*simplex)
    a, b, c = simplex
    signs = [
        p[0] * q[1] - p[1] * q[0] for p, q in ((a, b), (b, c), (c, a))
    ]  # the origin's side of each edge
    if all(s >= 0 for s in signs) or all(s <= 0 for s in signs):
        return None
    return min(
        (_on_segment(p, q) for p, q in ((a, b), (b, c), (c, a))),
        key=lambda found: _dot(found[0], found[0]),
    )


def hull_gap(a: Part, b: Part) -> Vec | None:
    """The shortest vector from hull b to hull a, ignoring the radii.

    Args:
        a: first part.
        b: second part.

    Returns:
        The vector between the nearest points (zero length means they just
        touch), or None if the hulls overlap.
    """
    v = _difference(a, b, (1.0, 0.0))
    simplex = [v]
    for _ in range(_MAX_ROUNDS):
        n2 = _dot(v, v)
        if n2 < 1e-24:
            return None
        w = _difference(a, b, (-v[0], -v[1]))
        if n2 - _dot(v, w) <= _RELATIVE_TOLERANCE * n2 or w in simplex:
            return v
        found = _nearest([*simplex, w])
        if found is None:
            return None
        v, simplex = found
    return v


def _centre(part: Part) -> Vec:
    n = len(part.vertices)
    return (
        sum(v[0] for v in part.vertices) / n,
        sum(v[1] for v in part.vertices) / n,
    )


def _axes(part: Part) -> list[Vec]:
    """Unit normals of the edges of a hull, none for a single point."""
    vs = part.vertices
    pairs = [(vs[0], vs[1])] if len(vs) == 2 else zip(vs, vs[1:] + vs[:1])
    axes = []
    for p, q in pairs:
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        if length > 0:
            axes.append(((p[1] - q[1]) / length, (q[0] - p[0]) / length))
    return axes


def _ways_out(a: Part, b: Part, axis: Vec) -> list[tuple[float, Vec]]:
    """How far b must move along the axis, and which way, to clear a."""
    pa = [_dot(v, axis) for v in a.vertices]
    pb = [_dot(v, axis) for v in b.vertices]
    return [
        (max(pa) - min(pb), axis),
        (max(pb) - min(pa), (-axis[0], -axis[1])),
    ]


def touch(a: Part, b: Part) -> tuple[Vec, float] | None:
    """Whether two parts overlap, and the shallowest way to part them.

    Args:
        a: first part.
        b: second part.

    Returns:
        None if they do not overlap (touching with no depth counts as not
        overlapping). Otherwise (normal, depth): the unit normal points from a
        toward b. Moving b by `depth` along it, or a by `depth` against it,
        parts them.
    """
    reach = a.radius + b.radius
    gap = hull_gap(a, b)
    if gap is not None:
        distance = math.hypot(*gap)
        if distance >= reach:
            return None
        if distance > 0:
            return (-gap[0] / distance, -gap[1] / distance), reach - distance
    # The hulls overlap or share a point: leave along the axis of least depth.
    ca, cb = _centre(a), _centre(b)
    between = (cb[0] - ca[0], cb[1] - ca[1])
    length = math.hypot(*between)
    axes = _axes(a) + _axes(b)
    axes.append((between[0] / length, between[1] / length) if length else (1.0, 0.0))
    depth, normal = min(
        (way for axis in axes for way in _ways_out(a, b, axis)),
        key=lambda found: found[0],
    )
    total = max(depth, 0.0) + reach
    return (normal, total) if total > 0 else None
