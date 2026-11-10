"""Tests for naming lidars in the `sensors` table of a robot."""

import warnings

import pytest

from demuvo.world import WorldError

from .small_world import small_world


def sensors_of(tmp_path, extra):
    """The sensors the small world's robot is given by the extra lines."""
    return small_world(tmp_path, extra).robots[0].sensors


def test_a_lidar_is_named_by_its_kind(tmp_path):
    """`lidar = { ... }` is one lidar, and its settings come back as numbers."""
    got = sensors_of(tmp_path, "sensors = { lidar = { beams = 90, max_range = 12 } }")
    assert got == {"lidar": {"beams": 90.0, "max_range": 12.0}}


def test_a_robot_can_carry_several_lidars_each_with_a_name_and_a_mount(tmp_path):
    """A front and a rear lidar; the mount is a list and may be negative."""
    got = sensors_of(
        tmp_path,
        'sensors = { front = { type = "lidar", mount = [0.3, 0.0, 0.0] }, '
        'rear = { type = "lidar", mount = [-0.3, 0.0, 3.14] } }',
    )
    assert got["front"] == {"type": "lidar", "mount": (0.3, 0.0, 0.0)}
    assert got["rear"] == {"type": "lidar", "mount": (-0.3, 0.0, 3.14)}


def test_a_lidar_is_not_a_setting_the_estimator_ignores(tmp_path):
    """A payload sensor is carried, so there is nothing to warn about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sensors_of(tmp_path, "sensors = { lidar = {} }")
        sensors_of(tmp_path, 'sensors = { top = { type = "lidar" } }')


def test_a_sensor_the_estimator_does_not_read_still_warns(tmp_path):
    """Encoders on a robot that uses the true pose have no effect, and say so."""
    with pytest.warns(UserWarning, match="sensors.encoders is set but estimator"):
        sensors_of(tmp_path, "sensors = { encoders = { noise = 0.1 } }")


def test_the_world_defaults_for_a_lidar_do_not_give_every_robot_one(tmp_path):
    """`[sensors.lidar]` sets settings only; a robot that names no lidar has none."""
    world = small_world(tmp_path, "\n[sensors.lidar]\nbeams = 10\n")
    assert world.sensors == {"lidar": {"beams": 10.0}}
    assert world.robots[0].sensors == {}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("sensors = { front = {} }", "unknown key 'front', or give it a 'type'"),
        ('sensors = { a = { type = "encoders" } }', "'type' must be one of lidar"),
        ('sensors = { a = { type = "radar" } }', "'type' must be one of lidar"),
        ("sensors = { lidar = { colour = 1 } }", "unknown key 'colour'"),
        ("sensors = { lidar = { mount = [1, 2] } }", "'mount' needs 3 finite"),
        ("sensors = { lidar = { mount = 1 } }", "'mount' needs 3 finite"),
        (
            "sensors = { lidar = { max_range = -1 } }",
            "'max_range' must not be negative",
        ),
        ("sensors = { lidar = { beams = 1.0e999 } }", "'beams' needs 1 finite"),
    ],
)
def test_bad_lidar_keys_are_named(tmp_path, extra, message):
    """Each mistake names the robot, the sensor and the key."""
    with pytest.raises(WorldError, match=f"robot 'a' sensors.*{message}"):
        sensors_of(tmp_path, extra)


def test_a_type_is_not_allowed_in_the_world_defaults(tmp_path):
    """Defaults are by kind, so a name and a type have no place there."""
    with pytest.raises(WorldError, match="file sensors.lidar: unknown key 'type'"):
        small_world(tmp_path, '\n[sensors.lidar]\ntype = "lidar"\n')
