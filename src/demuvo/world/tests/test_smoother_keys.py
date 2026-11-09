"""Tests for the `smoother` keys of a robot in a world file."""

import pytest

from demuvo.world import WorldError

from .small_world import small_world


def robot(tmp_path, extra=""):
    """Load a small world whose robot has the extra lines, and return the robot."""
    return small_world(tmp_path, extra).robots[0]


def test_a_robot_has_no_smoother_unless_it_names_one(tmp_path):
    """No key gives None and no parameters."""
    spec = robot(tmp_path)
    assert (spec.smoother, spec.smoother_params) == (None, {})


def test_the_smoother_and_its_params_come_back_as_written(tmp_path):
    """Both keys are read as written."""
    spec = robot(
        tmp_path,
        'smoother = "demuvo.smoothers:Anything"\nsmoother_params = { max_cost = 5 }',
    )
    assert spec.smoother == "demuvo.smoothers:Anything"
    assert spec.smoother_params == {"max_cost": 5}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('smoother = "Anything"', "'smoother' must be 'module:Class'"),
        ("smoother_params = { a = 1 }", "'smoother_params' needs 'smoother'"),
    ],
)
def test_bad_smoother_keys_are_named(tmp_path, extra, message):
    """Each mistake names the robot and the key."""
    with pytest.raises(WorldError, match=f"robot 'a': {message}"):
        robot(tmp_path, extra)
