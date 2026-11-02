"""Tests for the touch test between two convex parts."""

import math

import numpy as np
import pytest

from demuvo.sim.gjk import hull_gap, touch
from demuvo.world.shapes import Part


def square(x, y, half=1.0):
    """A square part centred on (x, y)."""
    return Part(
        (
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
        )
    )


def disc(x, y, r):
    """Disc."""
    return Part(((x, y),), r)


def test_far_parts_do_not_touch():
    """Far parts do not touch."""
    assert touch(square(0, 0), square(3, 0)) is None


def test_overlapping_squares_give_the_way_out():
    """Overlapping squares give the way out."""
    normal, depth = touch(square(0, 0), square(1.5, 0))
    assert normal == pytest.approx((1.0, 0.0))
    assert depth == pytest.approx(0.5)


def test_discs_use_the_radii():
    """Discs use the radii."""
    normal, depth = touch(disc(0, 0, 1), disc(1.5, 0, 1))
    assert normal == pytest.approx((1.0, 0.0))
    assert depth == pytest.approx(0.5)
    assert touch(disc(0, 0, 1), disc(2.5, 0, 1)) is None


def test_a_disc_against_a_wall_segment():
    """A disc against a wall segment."""
    wall = Part(((-3.0, 1.0), (3.0, 1.0)))
    normal, depth = touch(disc(0, 0.2, 1), wall)
    assert normal == pytest.approx((0.0, 1.0))
    assert depth == pytest.approx(0.2)


def test_a_wall_through_a_square_gives_the_short_way_out():
    """A wall through a square gives the short way out."""
    wall = Part(((-3.0, 0.3), (3.0, 0.3)))
    normal, depth = touch(square(0, 0), wall)
    assert normal == pytest.approx((0.0, 1.0))
    assert depth == pytest.approx(0.7)


def test_the_normal_points_from_the_first_to_the_second():
    """The normal points from the first to the second."""
    a, b = square(0, 0), square(0, 1.5)
    n1, _ = touch(a, b)
    n2, _ = touch(b, a)  # pylint: disable=arguments-out-of-order
    assert n1 == pytest.approx((0.0, 1.0))
    assert n2 == pytest.approx((0.0, -1.0))


def test_touching_edges_are_not_an_overlap():
    """Touching edges are not an overlap."""
    assert touch(square(0, 0), square(2, 0)) is None


def test_gap_is_none_when_the_hulls_overlap():
    """Gap is none when the hulls overlap."""
    assert hull_gap(square(0, 0), square(1, 0)) is None


def random_part(rng):
    """A random disc, segment or convex polygon with a random radius."""
    centre = rng.uniform(-2, 2, 2)
    kind = rng.integers(3)
    if kind == 0:
        return Part((tuple(centre),), float(rng.uniform(0.1, 1)))
    if kind == 1:
        end = centre + rng.uniform(-2, 2, 2)
        return Part((tuple(centre), tuple(end)), float(rng.uniform(0, 0.5)))
    angles = np.sort(rng.uniform(0, 2 * math.pi, 5))
    r = rng.uniform(0.5, 1.5)
    pts = [(centre[0] + r * math.cos(t), centre[1] + r * math.sin(t)) for t in angles]
    return Part(tuple(pts), float(rng.uniform(0, 0.3)))


def as_shapely(part):
    """As shapely."""
    from shapely.geometry import (
        MultiPoint,  # pylint: disable=import-error,import-outside-toplevel
    )

    return MultiPoint(part.vertices).convex_hull.buffer(part.radius, 64)


def test_it_agrees_with_shapely_on_random_parts():
    """Overlap yes/no matches shapely, and moving by the depth clears the pair."""
    shapely = pytest.importorskip("shapely")
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        a, b = random_part(rng), random_part(rng)
        found = touch(a, b)
        # buffer(64) is a polygon inside the true disc, so allow a thin band
        margin = shapely.distance(as_shapely(a), as_shapely(b))
        if found is None:
            assert margin > -1e-9
            continue
        normal, depth = found
        assert math.hypot(*normal) == pytest.approx(1.0)
        if margin > 0.02:
            pytest.fail(f"touch says overlap but shapely distance is {margin}")
        moved = Part(
            tuple(
                (v[0] + normal[0] * (depth + 0.02), v[1] + normal[1] * (depth + 0.02))
                for v in b.vertices
            ),
            b.radius,
        )
        assert touch(a, moved) is None
        checked += 1
    assert checked > 50
