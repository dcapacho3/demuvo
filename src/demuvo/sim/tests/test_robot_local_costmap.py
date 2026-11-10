"""Tests for the local costmap a robot builds as it drives."""

import numpy as np
import pytest

from demuvo import Simulation, Twist, load_world
from demuvo.costmap import LETHAL

WORLD = """
version = 1

[world]
size = [10.0, 6.0]
seed = 8
{world}
[[obstacle]]
shape = {{ name = "linestring", vertices = [[3.03, 0.5], [3.03, 5.5]] }}

[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.25 }}
pose = [2.0, 2.5, 0.0]
{robot}
"""
LIDAR = "sensors = { lidar = { beams = 90, period = 0.01 } }"


def sim_of(tmp_path, robot="", world=""):
    """A simulation of one robot facing a wall, with the extra lines."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(world=world, robot=robot))
    return Simulation(load_world(path))


def wall_x(costmap):
    """The mean x of the cells marked as obstacles along the wall."""
    rows, cols = np.nonzero(costmap.data == LETHAL)
    return float(costmap.centres(rows, cols)[0].mean())


def test_a_robot_with_a_lidar_builds_a_local_costmap_of_what_it_sees(tmp_path):
    """After a step the wall ahead is marked in the robot's own costmap."""
    sim = sim_of(tmp_path, LIDAR)
    robot = sim.robots["a"]
    assert robot.local_costmap is None  # nothing before the first step
    sim.step({})
    assert robot.local_costmap.cost_at(3.03, 2.5) == LETHAL
    assert robot.local_costmap.cost_at(3.08, 2.5) >= 253  # inflated, next to the wall
    assert robot.local_costmap.cost_at(2.0, 2.5) == 0


def test_a_robot_with_no_lidar_has_no_local_costmap(tmp_path):
    """Nothing to build it from."""
    sim = sim_of(tmp_path)
    sim.step({})
    assert sim.robots["a"].local is None
    assert sim.robots["a"].local_costmap is None


def test_the_local_costmap_can_be_switched_off(tmp_path):
    """`enabled = false` leaves the lidar and drops the costmap."""
    sim = sim_of(tmp_path, LIDAR, "\n[costmap.local]\nenabled = false\n")
    sim.step({})
    assert sim.robots["a"].readings["lidar"] is not None
    assert sim.robots["a"].local_costmap is None


def test_the_settings_of_the_world_shape_the_costmap(tmp_path):
    """A 2 m window of 10 cm cells has 20 cells a side."""
    world = "\n[costmap.local]\nwindow = 2.0\nresolution = 0.1\n"
    sim = sim_of(tmp_path, LIDAR, world)
    sim.step({})
    assert sim.robots["a"].local_costmap.data.shape == (20, 20)


def test_the_hits_are_placed_with_the_estimated_pose_so_a_drift_shows(tmp_path):
    """A drifting estimate marks the wall off its place, and a true one does not."""
    marked = {}
    for estimator, noise in (("ground_truth", 0.0), ("dead_reckoning", 1.5)):
        encoders = f"encoders = {{ noise = {noise} }}, " if noise else ""
        robot = (
            f'estimator = "{estimator}"\n'
            f"sensors = {{ {encoders}lidar = {{ beams = 90, period = 0.01 }} }}"
        )
        sim = sim_of(tmp_path, robot, "\n[costmap.local]\nperiod = 0.01\n")
        for _ in range(60):
            sim.step({"a": Twist(vx=0.5)})
        marked[estimator] = wall_x(sim.robots["a"].local_costmap)
    assert marked["ground_truth"] == pytest.approx(3.05, abs=0.03)
    assert abs(marked["dead_reckoning"] - marked["ground_truth"]) > 0.03
