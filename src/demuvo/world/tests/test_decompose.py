"""Tests for the cut into convex pieces."""

import numpy as np
import pytest

from demuvo.world.decompose import convex_pieces

U = ((0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4))


def area(points):
    """Area."""
    n = len(points)
    return (
        sum(
            points[i][0] * points[(i + 1) % n][1]
            - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
        / 2
    )


def test_a_convex_polygon_stays_one_piece():
    """A convex polygon stays one piece."""
    assert len(convex_pieces(((0, 0), (2, 0), (2, 1), (0, 1)))) == 1


def test_the_u_shape_needs_three_pieces():
    """The u shape needs three pieces."""
    pieces = convex_pieces(U)
    assert len(pieces) == 3
    assert sum(area(p) for p in pieces) == pytest.approx(area(U))
    assert all(area(p) > 0 for p in pieces)


def test_the_corners_may_run_clockwise():
    """The corners may run clockwise."""
    assert len(convex_pieces(U[::-1])) == 3


def test_pieces_cover_the_polygon_and_are_convex():
    """On a random star, the pieces' areas add up and every piece turns left."""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.integers(5, 14))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        radii = rng.uniform(0.5, 3, n)
        star = tuple((r * np.cos(t), r * np.sin(t)) for r, t in zip(radii, angles))
        if not shapely.geometry.Polygon(star).is_valid:
            continue
        pieces = convex_pieces(star)
        assert sum(area(p) for p in pieces) == pytest.approx(area(star))
        for piece in pieces:
            hull = shapely.geometry.MultiPoint(piece).convex_hull
            assert hull.area == pytest.approx(area(piece))
