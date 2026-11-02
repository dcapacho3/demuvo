"""Cut a simple polygon into convex pieces.

Ear clipping cuts the polygon into triangles. The Hertel-Mehlhorn step then
removes every cut whose removal leaves a convex piece. The result never has more
than four times the fewest pieces possible. `Polygon` does this
once, the first time its parts are asked for, so the cost is paid before the
first step and not at every step.

"""

from __future__ import annotations

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    """Twice the signed area of the triangle o, a, b."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _area2(points: list[Point]) -> float:
    """Twice the signed area of a polygon; positive if counter-clockwise."""
    n = len(points)
    return sum(
        points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
        for i in range(n)
    )


def _inside(p: Point, a: Point, b: Point, c: Point) -> bool:
    """True if p is in the counter-clockwise triangle abc, edges included."""
    return _cross(a, b, p) >= 0 and _cross(b, c, p) >= 0 and _cross(c, a, p) >= 0


def triangulate(points: list[Point]) -> list[tuple[int, int, int]]:
    """Cut a counter-clockwise simple polygon into triangles by ear clipping.

    Args:
        points: the corners, counter-clockwise.

    Returns:
        Triangles as index triples into `points`, each counter-clockwise.
    """
    ring = list(range(len(points)))
    triangles = []
    while len(ring) > 3:
        n = len(ring)
        for k in range(n):
            i, j, m = ring[k - 1], ring[k], ring[(k + 1) % n]
            a, b, c = points[i], points[j], points[m]
            if _cross(a, b, c) <= 0:
                continue
            if any(_inside(points[o], a, b, c) for o in ring if o not in (i, j, m)):
                continue
            triangles.append((i, j, m))
            del ring[k]
            break
        else:
            straight = [
                k
                for k in range(n)
                if _cross(
                    points[ring[k - 1]], points[ring[k]], points[ring[(k + 1) % n]]
                )
                == 0
            ]
            if not straight:
                raise ValueError("the polygon is not simple")
            del ring[straight[0]]
    if len(ring) == 3 and _cross(*(points[i] for i in ring)) > 0:
        triangles.append(tuple(ring))
    return triangles


def _convex(points: list[Point], ring: list[int]) -> bool:
    """True if the polygon given by the indices turns left (or straight) everywhere."""
    n = len(ring)
    return all(
        _cross(points[ring[k - 1]], points[ring[k]], points[ring[(k + 1) % n]]) >= 0
        for k in range(n)
    )


def _merge(a: list[int], b: list[int]) -> list[int] | None:
    """Join two pieces along a shared edge, or None if they share none."""
    for x, u in enumerate(a):
        v = a[(x + 1) % len(a)]
        for y, w in enumerate(b):
            if w == v and b[(y + 1) % len(b)] == u:
                ra = a[x + 1 :] + a[: x + 1]  # starts at v, ends at u
                rb = b[y + 1 :] + b[: y + 1]  # starts at u, ends at v
                return ra + rb[1:-1]
    return None


def convex_pieces(vertices: tuple[Point, ...]) -> list[tuple[Point, ...]]:
    """Cut a simple polygon into convex pieces.

    Args:
        vertices: the corners in order, either way round.

    Returns:
        Pieces, each a counter-clockwise tuple of corners. A convex polygon
        comes back as one piece.
    """
    points = list(vertices)
    if _area2(points) < 0:
        points.reverse()
    pieces = [list(t) for t in triangulate(points)]
    merged = True
    while merged:
        merged = False
        for x, first in enumerate(pieces):
            for y in range(x + 1, len(pieces)):
                joined = _merge(first, pieces[y])
                if joined is not None and _convex(points, joined):
                    pieces[x] = joined
                    del pieces[y]
                    merged = True
                    break
            if merged:
                break
    return [tuple(points[i] for i in piece) for piece in pieces]
