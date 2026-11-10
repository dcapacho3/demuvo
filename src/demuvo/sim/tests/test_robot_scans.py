"""Tests for the scans a robot reports from the lidars it carries."""

import numpy as np
import pytest

from demuvo import Simulation, load_world
from demuvo.sensing import Reading

WORLD = """
version = 1

[world]
size = [8.0, 5.0]
seed = 6

[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.2 }}
pose = [2.0, 2.5, 0.0]
{sensors}
"""


def robot_of(tmp_path, sensors=""):
    """A simulation with one robot, and the robot."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(sensors=sensors))
    sim = Simulation(load_world(path))
    return sim, sim.robots["a"]


def test_a_robot_with_no_lidar_has_no_scans(tmp_path):
    """Nothing carried, nothing to report."""
    sim, robot = robot_of(tmp_path)
    sim.step({})
    assert robot.scans() == {}


def test_a_scan_appears_after_the_first_step(tmp_path):
    """Before it, there is none; after it, one scan whose ranges are the reading."""
    sim, robot = robot_of(tmp_path, "sensors = { lidar = { beams = 4 } }")
    assert robot.scans() == {}
    sim.step({})
    scans = robot.scans()
    assert list(scans) == ["lidar"]
    assert scans["lidar"].ranges == pytest.approx(robot.readings["lidar"].data)
    assert scans["lidar"].max_range == 8.0


def test_each_lidar_gives_its_own_scan(tmp_path):
    """Two named lidars, two scans, each with its own mount."""
    both = (
        'sensors = { front = { type = "lidar", beams = 2, mount = [0.5, 0, 0] }, '
        'rear = { type = "lidar", beams = 2, mount = [-0.5, 0, 3.14] } }'
    )
    sim, robot = robot_of(tmp_path, both)
    sim.step({})
    scans = robot.scans()
    assert set(scans) == {"front", "rear"}
    assert scans["front"].mount == (0.5, 0.0, 0.0)
    assert scans["rear"].mount == (-0.5, 0.0, 3.14)


def test_a_sensor_that_gives_no_scan_is_left_out(tmp_path):
    """A payload sensor with no `scan` method has a reading but no scan."""
    _, robot = robot_of(tmp_path)
    robot.payload["other"] = object()
    robot.readings["other"] = Reading("other", 0.0, np.zeros(1))
    assert robot.scans() == {}
