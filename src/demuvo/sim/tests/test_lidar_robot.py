"""Tests for the lidars a robot carries in a simulation."""

import numpy as np
import pytest

from demuvo import Simulation, Twist, load_world
from demuvo.world import WorldError

WORLD = """
version = 1

[world]
size = [10.0, 6.0]
seed = 4
{world}
[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.2 }}
pose = [2.0, 3.0, 0.0]
{robot}
{other}
"""
OTHER = """
[[robot]]
name = "b"
model = "diff_drive"
shape = { name = "circle", radius = 0.25 }
pose = [6.0, 3.0, 0.0]
"""


def sim_of(tmp_path, robot="", world="", other=""):
    """A simulation of one robot a in an empty room, with the extra lines."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(world=world, robot=robot, other=other))
    return Simulation(load_world(path))


def test_a_robot_with_no_lidar_has_no_payload(tmp_path):
    """Nothing is carried, and nothing is read."""
    sim = sim_of(tmp_path)
    sim.step({})
    assert sim.robots["a"].payload == {}
    assert sim.robots["a"].readings == {}


def test_a_lidar_reads_the_room_from_the_first_step(tmp_path):
    """Four beams round a robot at (2, 3): back 2 m, down 3, ahead 8, up 3."""
    sim = sim_of(tmp_path, "sensors = { lidar = { beams = 4, max_range = 20 } }")
    assert sim.robots["a"].readings == {}
    sim.step({})
    reading = sim.robots["a"].readings["lidar"]
    assert reading.sensor == "lidar"
    assert reading.data == pytest.approx([2.0, 3.0, 8.0, 3.0])


def test_two_lidars_on_one_robot_read_from_their_own_mounts(tmp_path):
    """A lidar 0.5 m ahead looking forward, one 0.5 m behind looking back."""
    robot = (
        'sensors = { front = { type = "lidar", beams = 1, mount = [0.5, 0, 0] },'
        ' rear = { type = "lidar", beams = 1, mount = [-0.5, 0, 3.141592653589793] } }'
    )
    sim = sim_of(tmp_path, robot)
    sim.step({})
    readings = sim.robots["a"].readings
    assert set(readings) == {"front", "rear"}
    assert readings["front"].data == pytest.approx([7.5])
    assert readings["rear"].data == pytest.approx([1.5])


def test_a_lidar_sees_other_robots_but_not_its_own_body(tmp_path):
    """Robot b at (6, 3) with radius 0.25 is 3.75 m ahead of a."""
    sim = sim_of(tmp_path, "sensors = { lidar = { beams = 1 } }", other=OTHER)
    sim.step({})
    assert sim.robots["a"].readings["lidar"].data == pytest.approx([3.75])


def test_the_world_default_applies_and_the_robots_own_setting_wins(tmp_path):
    """A world max_range of 1 m blinds the lidar; the robot's own 20 m restores it."""
    world = "\n[sensors.lidar]\nmax_range = 1.0\n"
    blind = sim_of(tmp_path, "sensors = { lidar = { beams = 1 } }", world)
    blind.step({})
    assert np.isinf(blind.robots["a"].readings["lidar"].data).all()
    own = "sensors = { lidar = { beams = 1, max_range = 20 } }"
    sees = sim_of(tmp_path, own, world)
    sees.step({})
    assert sees.robots["a"].readings["lidar"].data == pytest.approx([8.0])


def test_a_lidar_does_not_change_what_the_estimator_does(tmp_path):
    """A robot that dead-reckons drives the same with or without a lidar."""
    base = 'estimator = "dead_reckoning"\n'
    plain = sim_of(tmp_path, base)
    carrying = sim_of(tmp_path, base + "sensors = { lidar = { beams = 8 } }")
    for sim in (plain, carrying):
        for _ in range(20):
            sim.step({"a": Twist(vx=1.0)})
    assert carrying.estimates["a"].pose.xyt == pytest.approx(
        plain.estimates["a"].pose.xyt
    )


def test_noise_of_one_lidar_is_not_changed_by_adding_a_robot(tmp_path):
    """Each lidar has its own random stream, named by the robot and the sensor."""
    robot = "sensors = { lidar = { beams = 6, noise = 0.05, max_range = 20 } }"
    alone = sim_of(tmp_path, robot)
    crowd = sim_of(tmp_path, robot, other=OTHER.replace("6.0, 3.0", "9.0, 5.0"))
    for sim in (alone, crowd):
        sim.step({})
    assert np.array_equal(
        alone.robots["a"].readings["lidar"].data,
        crowd.robots["a"].readings["lidar"].data,
    )


def test_a_setting_the_lidar_refuses_names_the_robot_and_the_sensor(tmp_path):
    """The loader checks numbers; the lidar checks what they mean."""
    with pytest.raises(
        WorldError, match="robot 'a': sensors.lidar: fov must be above 0"
    ):
        sim_of(tmp_path, "sensors = { lidar = { fov = 9.0 } }")
