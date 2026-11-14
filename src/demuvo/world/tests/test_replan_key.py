"""Tests for the `replan` key of a robot."""

import pytest

from demuvo.world import ReplanSpec, WorldError

from .small_world import small_world


def replan_of(tmp_path, line):
    """The replan spec the small world's robot gets from one more line."""
    return small_world(tmp_path, line).robots[0].replan


def test_no_key_or_false_gives_none_and_true_gives_the_defaults(tmp_path):
    """A robot plans once for each task unless the file says otherwise."""
    assert replan_of(tmp_path, "") is None
    assert replan_of(tmp_path, "replan = false") is None
    assert replan_of(tmp_path, "replan = true") == ReplanSpec()


def test_every_setting_is_read_as_written(tmp_path):
    """The period, the two switches and the two distances."""
    spec = replan_of(
        tmp_path,
        "replan = { period = 0.5, timer = true, blocked = false, off_path = 0.3, "
        "stuck = 0, changed = 0.25 }",
    )
    assert spec == ReplanSpec(
        period=0.5, timer=True, blocked=False, off_path=0.3, stuck=0.0, changed=0.25
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("replan = 3", "robot 'a': 'replan' must be true, false or a table"),
        ("replan = { rate = 1.0 }", "robot 'a'.replan: unknown key 'rate'"),
        ("replan = { timer = 1 }", "robot 'a'.replan: 'timer' must be true or false"),
        ("replan = { period = 0 }", "robot 'a'.replan: period must be a positive"),
        ("replan = { off_path = -1 }", "robot 'a'.replan: off_path must be 0 or more"),
        ("replan = { changed = -0.1 }", "robot 'a'.replan: changed must be 0 or more"),
        ('replan = { stuck = "long" }', "'stuck' needs 1 finite number"),
    ],
)
def test_a_wrong_value_is_refused_by_robot_and_key(tmp_path, line, message):
    """The message names the robot and the setting."""
    with pytest.raises(WorldError, match=message):
        replan_of(tmp_path, line)


@pytest.mark.parametrize("field", ["timer", "blocked"])
def test_a_spec_built_in_python_checks_its_switches_itself(field):
    """A switch that is not true or false is refused by the spec, not only the file."""
    with pytest.raises(ValueError, match=f"{field} must be true or false"):
        ReplanSpec(**{field: 1})
