"""Tests for the bounds of a world."""

from dataclasses import replace

from demuvo.verify import inside

from .small_world import small_world


def test_a_world_starts_at_zero_unless_it_says_otherwise(tmp_path):
    """The small world is 5 m by 4 m; its bounds run from (0, 0) to (5, 4)."""
    assert small_world(tmp_path).bounds == (0.0, 0.0, 5.0, 4.0)


def test_the_bounds_follow_the_origin(tmp_path):
    """A world whose lower-left corner is at (-2, -1)."""
    world = replace(small_world(tmp_path), origin=(-2.0, -1.0))
    assert world.bounds == (-2.0, -1.0, 3.0, 3.0)


def test_a_point_is_inside_when_it_is_between_the_bounds_edges_included():
    """Both ends count, and a point on the far side of an edge does not."""
    bounds = (-2.0, -1.0, 3.0, 3.0)
    assert inside([(-2.0, -1.0), (3.0, 3.0), (0.0, 0.0)], bounds)
    assert not inside([(0.0, 0.0), (3.1, 0.0)], bounds)
    assert not inside([(-2.1, 0.0)], bounds)
