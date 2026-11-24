"""Tests for the cut into convex pieces."""

import numpy as np
import pytest

from demuvo.world.decompose import convex_pieces, triangulate

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


def test_a_corner_with_another_corner_inside_it_is_not_cut_first():
    """In an arrow shape the tip of the dent blocks the first corner."""
    arrow = [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)]
    triangles = triangulate(arrow)
    assert len(triangles) == 3
    assert sum(area([arrow[i] for i in t]) for t in triangles) == pytest.approx(
        area(arrow)
    )


def test_a_polygon_that_crosses_itself_is_refused():
    """No corner of a self-crossing polygon can be cut off."""
    with pytest.raises(ValueError, match="not simple"):
        triangulate([(0, 2), (1, 0), (0, 1), (1, 3), (3, 0)])


# A traced cave wall from the field style (seed 330, robot radius 0.5 m, 20 by 14 m).
# Once its other ears are cut, four corners are left that lie on one line.
CAVE = (
    (17.1, 6.4), (17.4, 6.5), (17.4, 7.1), (18.2, 7.5), (18.5, 7.5), (18.7, 7.1),
    (19.1, 7.1), (19.3, 6.8), (19.8, 6.8), (19.8, 8.8), (19.0, 9.1), (19.0, 9.6),
    (19.3, 9.8), (19.3, 10.2), (19.0, 10.4), (19.0, 10.8), (19.4, 11.2), (19.8, 11.3),
    (19.8, 13.2), (19.4, 13.0), (19.3, 12.6), (18.9, 12.5), (18.5, 12.1), (18.5, 11.4),
    (18.8, 11.2), (18.6, 10.8), (18.2, 10.8), (17.9, 11.1), (17.1, 11.3), (16.9, 11.6),
    (17.2, 11.8), (17.2, 12.1), (17.0, 12.3), (16.7, 12.3), (16.5, 12.6), (15.5, 12.6),
    (15.3, 12.3), (15.3, 11.2), (15.1, 10.9), (15.2, 10.5), (14.7, 10.3), (14.5, 10.1),
    (14.5, 9.8), (14.8, 9.4), (15.3, 9.4), (15.5, 9.8), (15.5, 10.3), (16.2, 10.9),
    (16.7, 10.8), (16.8, 9.8), (17.2, 9.6), (17.3, 8.8), (16.7, 8.7), (16.3, 9.2),
    (15.9, 9.2), (15.7, 9.0), (15.6, 8.4), (15.3, 8.2), (15.3, 7.7), (15.8, 7.2),
    (16.4, 7.2), (16.6, 6.8),
)  # fmt: skip


def test_corners_left_on_one_line_cut_off_nothing():
    """A polygon whose last four corners are collinear is still cut in full."""
    pieces = convex_pieces(CAVE)
    assert sum(area(piece) for piece in pieces) == pytest.approx(area(CAVE))
