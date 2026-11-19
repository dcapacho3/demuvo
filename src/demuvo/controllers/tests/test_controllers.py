"""Tests for the pure pursuit controller."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Ackermann, DiffDrive, Observation, PurePursuit, Twist

from .observations import look, moved, see


def test_a_goal_straight_ahead_gives_the_set_speed_and_no_turn():
    """Far goal, facing it: vx = speed and wz = 0."""
    command = PurePursuit().command(look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)))
    assert (command.vx, command.wz) == pytest.approx((0.5, 0.0))


def test_the_robot_slows_down_near_the_goal():
    """Closer than speed / approach_gain: vx is approach_gain times the distance."""
    command = PurePursuit().command(look((4.6, 1.0, 0.0), (5.0, 1.0, 0.0)))
    assert command.vx == pytest.approx(0.4)


def test_a_path_that_ends_under_the_robot_is_a_zero_command():
    """On the path's last point but short of the goal: stand still, not a crash."""
    path = np.array([[1.0, 1.0, 0.0]])
    command = PurePursuit().command(look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), path=path))
    assert (command.vx, command.wz) == (0.0, 0.0)


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


def test_an_empty_path_gives_a_zero_command():
    """There is nothing to follow, so the robot stays still."""
    obs = look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), path=np.empty((0, 3)))
    assert PurePursuit().command(obs) == Twist()


def test_a_path_with_no_goal_is_an_error():
    """A path that leads nowhere is a mistake, and the message says so."""
    obs = Observation(
        0.0, 0.05, Transform2D.identity(), np.zeros((2, 3)), None, (0.1, 0.1)
    )
    with pytest.raises(ValueError, match="needs a goal to follow a path"):
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


# A car's limits: the default Ackermann, which cannot turn in place.
CAR = Ackermann().limits()


@pytest.mark.parametrize(
    ("pose", "goal"),
    [
        ((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)),  # straight ahead
        ((3.0, 3.0, 0.0), (6.0, 4.0, 0.0)),  # a bend to the left
        ((1.0, 1.0, 0.0), (1.0, 4.0, 1.57)),  # 90 degrees off: a turn on the spot
        ((1.0, 1.0, 0.0), (0.3, 1.0, 3.14)),  # behind
        ((5.0, 1.0, 0.0), (5.0, 1.0, 1.0)),  # at the goal, wrong angle
    ],
)
def test_a_diff_drive_is_driven_exactly_as_without_limits(pose, goal):
    """Limits that turn in place change nothing, so a diff drive keeps its behaviour."""
    plain = PurePursuit().command(look(pose, goal))
    with_limits = PurePursuit().command(see(pose, goal, DiffDrive().limits()))
    assert with_limits == plain


def test_a_car_keeps_its_turn_rate_inside_the_steering():
    """A bend too sharp for the steering is cut to the cone at that speed.

    The goal lies at a bearing of 0.7 rad, inside `turn_in_place_angle`, so both
    drive; but the pure pursuit curvature, 2 sin(0.7) / 0.6, is 2.15 /m where
    the car's steering allows 1.82 /m.
    """
    goal = (3.0 + 2.0 * math.cos(0.7), 3.0 + 2.0 * math.sin(0.7), 0.7)
    plain = PurePursuit().command(look((3.0, 3.0, 0.0), goal))
    tight = PurePursuit().command(see((3.0, 3.0, 0.0), goal, CAR))
    assert plain.vx > 0.0
    assert plain.wz > CAR.max_w(plain.vx)  # the diff drive would turn tighter
    assert tight.vx == plain.vx
    assert tight.wz == pytest.approx(CAR.max_w(tight.vx))


def test_a_car_swings_on_full_lock_instead_of_turning_on_the_spot():
    """A goal 90 degrees to the left: drive forward on full lock to the left."""
    command = PurePursuit().command(see((1.0, 1.0, 0.0), (1.0, 4.0, 1.57), CAR))
    assert command.vx > 0.0
    assert command.wz == pytest.approx(CAR.max_w(command.vx))
    assert command == Ackermann().clip(command)  # the model drives exactly this


def test_a_car_with_the_goal_behind_goes_round_to_the_left():
    """Straight behind is bearing pi, so it swings to the left, not stands still."""
    command = PurePursuit().command(see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), CAR))
    assert command.vx > 0.0
    assert command.wz > 0.0


def test_a_car_at_the_goal_stops_and_leaves_its_angle():
    """It cannot turn on the spot to the goal's angle, so it stands still."""
    command = PurePursuit().command(see((5.0, 1.0, 0.0), (5.0, 1.0, 1.0), CAR))
    assert command == Twist()


def test_a_reversing_car_backs_toward_a_target_behind():
    """With allow_reversing, a goal behind is backed to, inside the steering."""
    command = PurePursuit(allow_reversing=True).command(
        see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), CAR)
    )
    assert command.vx < 0.0
    assert abs(command.wz) <= CAR.max_w(command.vx) + 1e-12
    assert command == Ackermann().clip(command)


def test_without_allow_reversing_a_car_never_backs_up():
    """The default swings round forwards instead, as before."""
    command = PurePursuit().command(see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), CAR))
    assert command.vx > 0.0


def test_allow_reversing_changes_nothing_for_a_diff_drive():
    """A robot that turns in place turns toward a goal behind, as always."""
    plain = PurePursuit().command(look((1.0, 1.0, 0.0), (0.3, 1.0, 3.14)))
    flagged = PurePursuit(allow_reversing=True).command(
        see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), DiffDrive().limits())
    )
    assert flagged == plain


def test_a_car_backs_along_a_path_that_backs_without_allow_reversing():
    """A path whose headings disagree with its motion is driven backwards."""
    path = np.array([[1.0, 1.0, 0.0], [0.7, 1.0, 0.0], [0.3, 1.0, 0.0]])
    obs = see((1.0, 1.0, 0.0), (0.3, 1.0, 0.0), CAR, path=path)
    assert PurePursuit().command(obs).vx < 0.0


def test_a_reversing_car_swings_toward_a_target_beside_it():
    """A target to the side, not behind, is swung to on full lock, not backed to."""
    command = PurePursuit(allow_reversing=True).command(
        see((1.0, 1.0, 0.0), (1.0, 4.0, 1.57), CAR)
    )
    assert command.vx > 0.0
    assert command.wz == pytest.approx(CAR.max_w(command.vx))


def test_a_reversing_car_with_a_one_point_path_drives_to_the_point():
    """A path of one point is one forward run."""
    obs = see((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), CAR, path=np.array([[5.0, 1.0, 0.0]]))
    command = PurePursuit(allow_reversing=True).command(obs)
    assert command.vx > 0.0
    assert command.wz == pytest.approx(0.0)


def test_a_reversing_car_on_the_last_path_point_short_of_the_goal_stands_still():
    """With no direction left to steer, it stops instead of dividing by zero."""
    obs = see((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), CAR, path=np.array([[1.0, 1.0, 0.0]]))
    assert PurePursuit(allow_reversing=True).command(obs) == Twist()


def test_a_reversing_car_arrives_without_going_round():
    """Backing 0.7 m beats a 3 m forward loop: the robot gets there directly."""
    model, controller = Ackermann(), PurePursuit(allow_reversing=True)
    obs = see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), CAR)
    pose, driven = obs.pose, 0.0
    for _ in range(60):
        obs = moved(obs, pose)
        command = controller.command(obs)
        pose = model.step(pose, command, obs.dt)
        driven += abs(command.vx) * obs.dt
    assert float(np.hypot(*(pose.translation - obs.goal.translation))) <= 0.15
    assert driven < 1.0
