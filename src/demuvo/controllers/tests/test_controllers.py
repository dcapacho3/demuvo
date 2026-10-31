"""Tests for the pure pursuit controller."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Observation, PurePursuit, StraightLine, Twist


def look(pose, goal, tolerance=(0.1, 0.1), path=None):
    """An Observation for a robot at pose with a straight path to goal."""
    pose, goal = Transform2D.from_xyt(pose), Transform2D.from_xyt(goal)
    if path is None:
        path = StraightLine().plan(None, pose, goal)
    return Observation(
        time=0.0,
        dt=0.05,
        pose=pose,
        path=path,
        goal=goal,
        goal_tolerance=tolerance,
    )


def test_a_goal_straight_ahead_gives_the_set_speed_and_no_turn():
    """Far goal, facing it: vx = speed and wz = 0."""
    command = PurePursuit().command(look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)))
    assert (command.vx, command.wz) == pytest.approx((0.5, 0.0))


def test_the_robot_slows_down_near_the_goal():
    """Closer than speed / approach_gain: vx is approach_gain times the distance."""
    command = PurePursuit().command(look((4.6, 1.0, 0.0), (5.0, 1.0, 0.0)))
    assert command.vx == pytest.approx(0.4)


def test_the_turn_follows_the_side_of_the_path():
    """A path that bends left turns left (wz > 0); to the right turns right."""
    left = PurePursuit().command(look((3.0, 3.0, 0.0), (6.0, 4.0, 0.0)))
    right = PurePursuit().command(look((3.0, 3.0, 0.0), (6.0, 2.0, 0.0)))
    assert left.vx > 0.0 > right.wz
    assert left.wz > 0.0


def test_a_path_that_runs_beside_the_robot_curves_toward_it():
    """Path parallel to the heading, 0.2 m to the left: drive and steer left."""
    xs = np.linspace(0.0, 5.0, 100)
    path = np.column_stack([xs, np.full(100, 0.2), np.zeros(100)])
    command = PurePursuit().command(look((0.0, 0.0, 0.0), (5.0, 0.2, 0.0), path=path))
    assert command.vx == 0.5
    assert command.wz > 0.0


def test_a_goal_behind_turns_in_place():
    """Facing away: no forward speed, the turn rate at its limit."""
    command = PurePursuit().command(look((5.0, 1.0, 0.0), (1.0, 1.0, 0.0)))
    assert command.vx == 0.0
    assert abs(command.wz) == 1.0


def test_a_robot_at_its_goal_with_the_right_angle_stops():
    """Inside both tolerances the command is zero."""
    assert PurePursuit().command(look((3.0, 3.0, 0.0), (3.02, 3.0, 0.05))) == Twist()


@pytest.mark.parametrize("angle", [1.0, -1.0])
def test_a_robot_at_its_goal_turns_on_the_spot_to_the_goal_angle(angle):
    """At the goal with the wrong angle: turn toward it, at most rotate_speed."""
    command = PurePursuit().command(look((3.0, 3.0, 0.0), (3.0, 3.0, angle)))
    assert command == Twist(wz=math.copysign(1.0, angle))


def test_the_switch_turns_the_final_rotation_off():
    """With rotate_to_goal off, arriving in position is enough to stop."""
    controller = PurePursuit(rotate_to_goal=False)
    assert controller.command(look((3.0, 3.0, 0.0), (3.0, 3.0, 1.0))) == Twist()


def test_the_final_turn_takes_the_short_way_round():
    """From 3.0 rad to -3.0 rad is a short turn to the left, not a long one right."""
    command = PurePursuit().command(look((3.0, 3.0, 3.0), (3.0, 3.0, -3.0)))
    assert command.wz > 0.0


@pytest.mark.parametrize(
    "obs",
    [
        look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), path=np.empty((0, 3))),
        Observation(
            0.0, 0.05, Transform2D.identity(), np.zeros((2, 3)), None, (0.1, 0.1)
        ),
    ],
)
def test_a_missing_goal_or_path_is_an_error(obs):
    """There is nothing to follow."""
    with pytest.raises(ValueError, match="needs a goal and a path"):
        PurePursuit().command(obs)


@pytest.mark.parametrize(
    "name",
    ["speed", "lookahead", "approach_gain", "turn_in_place_angle", "turn_gain"]
    + ["rotate_speed"],
)
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, True])
def test_a_bad_number_is_refused_when_the_controller_is_built(name, bad):
    """Every tunable number must be positive and finite."""
    with pytest.raises(ValueError, match=name):
        PurePursuit(**{name: bad})
