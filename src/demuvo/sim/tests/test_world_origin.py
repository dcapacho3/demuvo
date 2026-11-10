"""Tests for a world whose lower-left corner is not at (0, 0)."""

from dataclasses import replace

import numpy as np
import pytest

from demuvo import Simulation, Twist
from demuvo.sensing import cast
from demuvo.world.tests.small_world import small_world


def sim_with_origin(tmp_path, origin=(-1.0, -1.0)):
    """The small world, 5 m by 4 m, moved so that its lower-left corner is at origin."""
    return Simulation(replace(small_world(tmp_path), origin=origin))


def test_the_border_walls_stop_a_robot_at_the_moved_edge(tmp_path):
    """A robot driving west from (1, 1) stops a radius short of the wall at x = -1."""
    sim = sim_with_origin(tmp_path)
    for _ in range(400):
        sim.step({"a": Twist(vx=-1.0)})
    assert sim.poses["a"].xyt[0] == pytest.approx(-0.9, abs=0.01)
    assert [collision.b for collision in sim.collisions] == ["border"]


def test_the_scene_holds_the_border_where_the_world_is(tmp_path):
    """A ray west from the robot at (1, 1) meets the wall 2 m away, at x = -1."""
    scene = sim_with_origin(tmp_path).scene()
    range_west = cast(np.array([1.0, 1.0]), np.array([np.pi]), scene, 20.0, 0)
    assert range_west == pytest.approx([2.0])
