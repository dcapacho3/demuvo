"""Tests for the mecanum model: the step, the wheels and the limits."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import MODELS, DiffDrive, Mecanum, Twist
from demuvo.models import WHEELS

TURN = 2 * math.pi


def drive(pose, command, dt, model=None):
    """The (x, y, angle) after one step from an (x, y, angle) pose."""
    model = model or Mecanum()
    return tuple(model.step(Transform2D.from_xyt(pose), command, dt).xyt)


def chord(pose, command, dt):
    """The step written out by hand: the chord at the mid-step heading.

    The tests' oracle, independent of robomath.
    """
    x, y, angle = pose
    theta = command.wz * dt
    half = theta / 2
    s = math.sin(half) / half if half else 1.0
    mid = angle + half
    c, n = math.cos(mid), math.sin(mid)
    vx, vy = command.vx, command.vy
    end = angle + theta
    return (
        x + s * dt * (vx * c - vy * n),
        y + s * dt * (vx * n + vy * c),
        math.atan2(math.sin(end), math.cos(end)),
    )


def close(a, b, tol=1e-12):
    """True if two (x, y, angle) agree, the angles round the circle."""
    da = math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2]))
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol and abs(da) <= tol


def test_the_table_names_the_mecanum_model():
    """The world file finds it by the name "mecanum"."""
    assert MODELS["mecanum"] is Mecanum


# The cases of section 5.4, one formula for all of them.


def test_standing_still_keeps_the_pose_bit_for_bit():
    """A zero twist moves nothing, not even by a rounding."""
    pose = (1.234, -5.678, 0.9)
    assert drive(pose, Twist(), 0.1) == tuple(Transform2D.from_xyt(pose).xyt)


@pytest.mark.parametrize(
    ("heading", "vy", "end"),
    [
        (0.0, 1.0, (0.0, 2.0)),  # facing +x, strafe left: +y
        (0.0, -1.0, (0.0, -2.0)),  # strafe right: -y
        (math.pi / 2, 1.0, (-2.0, 0.0)),  # facing +y, its left is -x
    ],
)
def test_a_pure_strafe_moves_sideways_and_keeps_the_heading(heading, vy, end):
    """Vy alone for 2 s moves 2 m to the robot's side, and it does not turn."""
    x, y, angle = drive((0.0, 0.0, heading), Twist(vy=vy), 2.0)
    assert (x, y) == pytest.approx(end)
    assert angle == Transform2D.from_xyt((0.0, 0.0, heading)).xyt[2]


def test_a_diagonal_is_a_straight_line_along_the_commanded_direction():
    """Vx = vy = 0.5 from heading pi/4 goes straight up +y by hypot * t."""
    x, y, angle = drive((1.0, 1.0, math.pi / 4), Twist(vx=0.5, vy=0.5), 2.0)
    assert (x, y) == pytest.approx((1.0, 1.0 + math.sqrt(2.0)))
    assert angle == pytest.approx(math.pi / 4)


def test_a_spin_in_place_keeps_the_position_bit_for_bit():
    """Vx = vy = 0: the centre stays exactly where it is."""
    pose = (1.234, -5.678, 0.9)
    x, y, angle = drive(pose, Twist(wz=0.7), 1.0)
    start = Transform2D.from_xyt(pose).xyt
    assert (x, y) == (start[0], start[1])
    assert angle == pytest.approx(1.6)


@pytest.mark.parametrize(
    "command",
    [
        Twist(vx=0.6, vy=0.4, wz=0.8),
        Twist(vx=-0.3, vy=0.5, wz=-0.6),
        Twist(vy=0.5, wz=1.0),
    ],
)
def test_moving_while_turning_rides_the_circle_about_the_icr(command):
    """The centre stays at hypot(vx, vy) / |wz| from the body point (-vy/wz, vx/wz)."""
    start = Transform2D.from_xyt((1.0, 1.0, 0.5))
    vx, vy, wz = command.vx, command.vy, command.wz
    icr = start.matrix @ np.array([-vy / wz, vx / wz, 1.0])
    radius = math.hypot(vx, vy) / abs(wz)
    for t in np.linspace(0.5, 8.0, 6):
        x, y, _ = Mecanum().step(start, command, t).xyt
        assert math.hypot(x - icr[0], y - icr[1]) == pytest.approx(radius, abs=1e-12)


def test_strafing_while_turning_orbits_a_point_on_its_own_x_axis():
    """Vy = -0.5, wz = 1: the robot circles a point 0.5 m ahead, always facing it."""
    model, start = Mecanum(), Transform2D.from_xyt((0.0, 0.0, 0.0))
    for t in (0.5, 1.0, 2.0, 3.0):
        x, y, angle = model.step(start, Twist(vy=-0.5, wz=1.0), t).xyt
        assert math.hypot(x - 0.5, y) == pytest.approx(0.5)
        bearing = math.atan2(-y, 0.5 - x)  # from the robot to the centre
        assert math.cos(bearing - angle) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "wz", [0.0, 5e-324, 1e-300, -1e-12, 1e-6, 0.3, -1.7, 2.9, math.pi, 6.0]
)
@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_the_step_is_the_closed_form_chord(wz, dt):
    """Tiny, zero, large and negative turns all match the hand-written formula."""
    model = Mecanum(max_w=10.0)
    pose, command = (0.3, -0.2, 2.8), Twist(vx=0.8, vy=-0.5, wz=wz)
    assert close(drive(pose, command, dt, model), chord(pose, command, dt), 1e-14)


@pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-9, 1e-15, 1e-100, 5e-324])
def test_the_step_is_continuous_through_a_zero_turn(eps):
    """No jump at wz = 0: a tiny turn ends within C * eps of the straight line.

    C bounds the derivative of the pose in wz, |v| * dt^2 / 2 for the position
    plus dt for the angle; 1 is ample at dt = 0.5.
    """
    pose, dt = (0.3, -0.2, 1.0), 0.5
    straight = drive(pose, Twist(vx=0.8, vy=-0.5), dt)
    for sign in (1, -1):
        turned = drive(pose, Twist(vx=0.8, vy=-0.5, wz=sign * eps), dt)
        assert close(turned, straight, 1.0 * eps)


def test_one_big_step_equals_many_small_ones():
    """The step is exact, so the step length does not change the result."""
    command = Twist(vx=0.7, vy=-0.3, wz=-0.4)
    pose = Transform2D.from_xyt((1.0, 2.0, 0.3))
    for _ in range(100):
        pose = Mecanum().step(pose, command, 0.02)
    assert close(tuple(pose.xyt), drive((1.0, 2.0, 0.3), command, 2.0))


def test_a_full_turn_in_one_step_comes_back_to_the_start():
    """Wz * dt = 2 pi: the arc closes, so the pose is the start."""
    model = Mecanum(max_w=2.0)
    end = drive((1.0, 2.0, 0.3), Twist(vx=0.5, vy=0.2, wz=1.0), TURN, model)
    assert close(end, (1.0, 2.0, 0.3))


def test_stepping_back_by_the_opposite_twist_undoes_the_step():
    """Exp(-xi) is the inverse of exp(xi)."""
    model, start = Mecanum(), Transform2D.from_xyt((0.4, -0.7, -2.0))
    there = model.step(start, Twist(vx=0.6, vy=0.3, wz=0.9), 1.5)
    back = model.step(there, Twist(vx=-0.6, vy=-0.3, wz=-0.9), 1.5)
    assert close(tuple(back.xyt), tuple(start.xyt))


def test_a_heading_across_pi_has_no_seam():
    """Turning through pi changes only the angle's number, not the motion."""
    command, dt = Twist(vx=0.5, vy=0.3, wz=0.4), 0.5
    for angle in (math.pi - 1e-9, -math.pi + 1e-9):
        assert close(
            drive((0.0, 0.0, angle), command, dt), chord((0.0, 0.0, angle), command, dt)
        )
    a = drive((0.0, 0.0, math.pi - 1e-9), command, dt)
    b = drive((0.0, 0.0, -math.pi + 1e-9), command, dt)
    assert close(a, b, 1e-8)


def test_with_no_sideways_speed_it_moves_as_a_diff_drive_bit_for_bit():
    """At vy = 0 the two models take the same step, to the last bit."""
    pose = Transform2D.from_xyt((1.0, 2.0, 0.3))
    for command in (Twist(vx=0.7, wz=-0.4), Twist(vx=-0.2, wz=0.9), Twist(wz=1.0)):
        assert tuple(Mecanum().step(pose, command, 0.1).xyt) == tuple(
            DiffDrive().step(pose, command, 0.1).xyt
        )


def test_commanding_the_mid_step_heading_goes_straight_while_spinning():
    """Design 5.5: (vx, vy) = R(-(angle + theta/2)) d / (s dt) moves by exactly d.

    The robot goes +x in the world at 0.5 m/s while it spins at 1 rad/s.
    """
    model, dt, wz = Mecanum(), 0.1, 1.0
    theta = wz * dt
    s = math.sin(theta / 2) / (theta / 2)
    pose = Transform2D.from_xyt((0.0, 0.0, 0.0))
    for _ in range(100):
        mid = pose.xyt[2] + theta / 2
        vx = (math.cos(mid) * 0.5) / s
        vy = (-math.sin(mid) * 0.5) / s
        pose = model.step(pose, Twist(vx=vx, vy=vy, wz=wz), dt)
    assert tuple(pose.xyt[:2]) == pytest.approx((5.0, 0.0), abs=1e-12)


# The limits and the clip.


def test_its_limits_are_a_box_that_turns_in_place_and_strafes():
    """Flat in wz, with a sideways bound of its own."""
    limits = Mecanum(max_v=2.0, max_vy=0.5, max_w=0.7).limits()
    assert limits.turns_in_place and limits.strafes
    assert (limits.max_v, limits.max_vy) == (2.0, 0.5)
    assert [limits.max_w(v) for v in (0.0, 1.0, 2.0)] == [0.7, 0.7, 0.7]


def test_a_command_inside_the_box_is_kept():
    """Nothing to cut, so nothing changes."""
    command = Twist(vx=0.5, vy=-0.4, wz=0.3)
    assert Mecanum().clip(command) == command


@pytest.mark.parametrize(
    ("command", "wanted"),
    [
        (Twist(vx=2.0, vy=1.0), Twist(vx=1.0, vy=0.5)),  # vx binds
        (Twist(vx=-0.5, vy=2.0), Twist(vx=-0.25, vy=1.0)),  # vy binds, max_vy = 1
        (Twist(vx=3.0, vy=-3.0, wz=5.0), Twist(vx=1.0, vy=-1.0, wz=1.0)),
    ],
)
def test_an_over_limit_command_keeps_its_direction_of_travel(command, wanted):
    """Vx and vy shrink together; the axis that binds lands on its bound exactly."""
    assert Mecanum().clip(command) == wanted


def test_a_clipped_command_stays_inside_the_box():
    """Over many commands, the clip never leaves the box, even by a rounding."""
    model = Mecanum(max_v=0.7, max_vy=0.3, max_w=1.1)
    rng = np.random.default_rng(3)
    for vx, vy, wz in rng.uniform(-5, 5, (2000, 3)):
        out = model.clip(Twist(vx=vx, vy=vy, wz=wz))
        assert abs(out.vx) <= 0.7 and abs(out.vy) <= 0.3 and abs(out.wz) <= 1.1
        if abs(vx) > 0.7 or abs(vy) > 0.3:
            assert math.atan2(out.vy, out.vx) == pytest.approx(math.atan2(vy, vx))


def test_the_step_clips_first():
    """An over-limit command moves the robot as its clipped command does."""
    command = Twist(vx=4.0, vy=2.0, wz=3.0)
    assert drive((0.0, 0.0, 0.0), command, 0.5) == drive(
        (0.0, 0.0, 0.0), Mecanum().clip(command), 0.5
    )


@pytest.mark.parametrize("name", ["vz", "wx", "wy"])
def test_what_it_cannot_follow_is_ignored_with_a_warning(name):
    """It moves as if the field were 0, and one warning names it."""
    wanted = drive((0.0, 0.0, 0.0), Twist(vx=1.0, vy=0.5), 1.0)
    with pytest.warns(UserWarning, match=name):
        moved = drive((0.0, 0.0, 0.0), Twist(vx=1.0, vy=0.5, **{name: 2.0}), 1.0)
    assert moved == wanted


def test_a_command_it_can_follow_gives_no_warning(recwarn):
    """Vx, vy and wz are all its own."""
    drive((0.0, 0.0, 0.0), Twist(vx=0.5, vy=0.5, wz=0.5), 1.0)
    assert not recwarn


@pytest.mark.parametrize(
    "name", ["max_v", "max_vy", "max_w", "wheel_radius", "wheelbase", "track"]
)
@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf])
def test_a_limit_or_a_size_must_be_a_positive_number(name, bad):
    """Every one is divided by or compared, so above 0, and refused by name."""
    with pytest.raises(ValueError, match=name):
        Mecanum(**{name: bad})


# The wheels.


def test_the_wheels_are_named_in_order():
    """Encoders and from_encoders use front left, front right, rear left, rear right."""
    assert WHEELS == ("fl", "fr", "rl", "rr")


@pytest.mark.parametrize(
    ("command", "angles"),
    [
        (Twist(vx=1.0), [10.0, 10.0, 10.0, 10.0]),  # all four roll forward
        (Twist(vy=1.0), [-10.0, 10.0, 10.0, -10.0]),  # strafe left: an X
        (Twist(wz=1.0), [-3.0, 3.0, -3.0, 3.0]),  # spin left: k = 0.3
    ],
)
def test_the_wheel_increments_worked_by_hand(command, angles):
    """Radius 0.1 m, wheelbase 0.4 m and track 0.2 m, 1 s: k = 0.3 m."""
    model = Mecanum(wheel_radius=0.1, wheelbase=0.4, track=0.2)
    assert model.encoders(command, 1.0) == pytest.approx(angles)
    assert model.from_encoders(np.array(angles)) == pytest.approx(
        [command.vx, command.vy, command.wz]
    )


def test_the_front_pair_turns_by_as_much_as_the_rear_pair():
    """Without slip fl + fr = rl + rr, for every command."""
    model = Mecanum()
    rng = np.random.default_rng(5)
    for vx, vy, wz in rng.uniform(-1, 1, (200, 3)):
        fl, fr, rl, rr = model.encoders(Twist(vx=vx, vy=vy, wz=wz), 0.1)
        assert fl + fr == pytest.approx(rl + rr, abs=1e-12)


def test_odometry_reproduces_the_step():
    """From_encoders(encoders(cmd)) is the body step, so dead reckoning is exact."""
    model, dt = Mecanum(), 0.1
    command = model.clip(Twist(vx=0.8, vy=-0.6, wz=0.9))
    body = model.from_encoders(model.encoders(command, dt))
    assert body == pytest.approx([command.vx * dt, command.vy * dt, command.wz * dt])
    pose = Transform2D.from_xyt((1.0, 2.0, 0.3))
    assert close(tuple(pose.rplus(body).xyt), tuple(model.step(pose, command, dt).xyt))


def test_readings_that_disagree_give_the_least_squares_step():
    """A slip on one wheel: the step is the fit that best explains all four."""
    model = Mecanum()
    readings = model.encoders(Twist(vx=0.5, vy=0.2, wz=0.3), 0.1) + np.array(
        [0.01, 0.0, 0.0, 0.0]
    )
    k, r = model.reach, model.wheel_radius
    jacobian = np.array([[1, -1, -k], [1, 1, k], [1, 1, -k], [1, -1, k]]) / r
    fit, *_ = np.linalg.lstsq(jacobian, readings, rcond=None)
    assert model.from_encoders(readings) == pytest.approx(fit)
