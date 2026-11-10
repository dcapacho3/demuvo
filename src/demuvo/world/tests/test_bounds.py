"""Tests for the bounds of a world."""

from dataclasses import replace
from pathlib import Path

import pytest

from demuvo import load_world
from demuvo.verify import inside
from demuvo.world import WorldError

from .small_world import WORLD, small_world


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


def load_with_origin(tmp_path, origin):
    """The small world with an `origin` line in `[world]`."""
    path = tmp_path / "w.toml"
    path.write_text(
        WORLD.format(extra="").replace("seed = 2", f"origin = {origin}\nseed = 2")
    )
    return load_world(path)


def test_the_world_file_sets_the_origin(tmp_path):
    """A 5 m by 4 m world whose corner is at (-2, -1) ends at (3, 3)."""
    world = load_with_origin(tmp_path, "[-2.0, -1.0]")
    assert world.origin == (-2.0, -1.0)
    assert world.bounds == (-2.0, -1.0, 3.0, 3.0)


def test_a_robot_outside_the_moved_world_is_refused(tmp_path):
    """The robot starts at (1, 1); a world that begins at (2, 2) does not hold it."""
    with pytest.raises(WorldError, match="robot"):
        load_with_origin(tmp_path, "[2.0, 2.0]")


@pytest.mark.parametrize("origin", ["[1.0]", '["a", "b"]', "[1.0, inf]", "3.0"])
def test_the_origin_must_be_two_finite_numbers(tmp_path, origin):
    """The message names the key."""
    with pytest.raises(WorldError, match="origin"):
        load_with_origin(tmp_path, origin)


def test_the_origin_example_loads_and_its_corner_is_at_minus_five_minus_three():
    """The example is the demo world moved; the robot and the goal lie inside it."""
    path = Path(__file__).parents[4] / "examples" / "worlds" / "origin.toml"
    world = load_world(path)
    assert world.bounds == (-5.0, -3.0, 5.0, 3.0)
