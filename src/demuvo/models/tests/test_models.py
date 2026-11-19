"""Tests for the kinematic models."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import MODELS, Ackermann, DiffDrive, Limits, Mecanum, Twist

TURN = 2 * math.pi


def drive(pose, command, dt, model=None):
    """The (x, y, angle) after one step from an (x, y, angle) pose."""
    model = model or DiffDrive()
    return tuple(model.step(Transform2D.from_xyt(pose), command, dt).xyt)


def test_the_table_names_every_model():
    """The world file finds a model by its name."""
    assert MODELS["diff_drive"] is DiffDrive
    assert MODELS["ackermann"] is Ackermann


def test_a_straight_line_follows_the_heading():
    """Heading +y and wz = 0: the robot moves along +y."""
    x, y, angle = drive((1.0, 1.0, math.pi / 2), Twist(vx=1.0), 2.0)
    assert (x, y, angle) == pytest.approx((1.0, 3.0, math.pi / 2))


def test_a_spin_in_place_keeps_the_position():
    """Vx = 0: the position stays and the angle grows by wz * dt."""
    x, y, angle = drive((3.0, 2.0, 0.0), Twist(wz=1.0), 0.5)
    assert (x, y, angle) == pytest.approx((3.0, 2.0, 0.5))


def test_the_angle_is_wrapped():
    """A turn past pi comes back in (-pi, pi]."""
    _, _, angle = drive((0.0, 0.0, 3.0), Twist(wz=1.0), 0.5)
    assert angle == pytest.approx(3.5 - TURN)


def test_a_half_circle_has_radius_v_over_w():
    """Vx = 1, wz = 0.5: radius 2. One exact step of half a turn ends at (0, 4)."""
    x, y, angle = drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=0.5), TURN)
    assert (x, y) == pytest.approx((0.0, 4.0))
    assert abs(angle) == pytest.approx(math.pi)


def test_one_big_step_equals_many_small_ones():
    """The step is exact, so the step length does not change the result."""
    command = Twist(vx=0.7, wz=-0.4)
    pose = Transform2D.from_xyt((1.0, 2.0, 0.3))
    for _ in range(100):
        pose = DiffDrive().step(pose, command, 0.02)
    assert tuple(pose.xyt) == pytest.approx(drive((1.0, 2.0, 0.3), command, 2.0))


def test_the_default_limits():
    """Without parameters a robot reaches 1 m/s and 1 rad/s."""
    assert DiffDrive().clip(Twist(vx=5.0, wz=-5.0)) == Twist(vx=1.0, wz=-1.0)


def test_a_command_is_clipped_to_the_limits():
    """Speeds above the limits are cut, in both directions."""
    model = DiffDrive(max_v=1.0, max_w=0.5)
    assert model.clip(Twist(vx=5.0, wz=-5.0)) == Twist(vx=1.0, wz=-0.5)
    assert model.clip(Twist(vx=-5.0, wz=5.0)) == Twist(vx=-1.0, wz=0.5)
    assert model.clip(Twist(vx=0.2, wz=0.1)) == Twist(vx=0.2, wz=0.1)
    assert drive((0.0, 0.0, 0.0), Twist(vx=5.0), 1.0, model) == pytest.approx(
        (1.0, 0.0, 0.0)
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["vx", "vy", "vz", "wx", "wy", "wz"])
def test_a_twist_that_is_not_finite_is_refused(bad, name):
    """Nan or infinity in any field raises, naming the field."""
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        Twist(**{name: bad})


@pytest.mark.parametrize("name", ["vy", "vz", "wx", "wy"])
def test_what_the_robot_cannot_follow_is_ignored_with_a_warning(name):
    """Diff drive moves as if the field were 0, and one warning names it."""
    wanted = drive((0.0, 0.0, 0.0), Twist(vx=1.0), 1.0)
    with pytest.warns(UserWarning, match=name):
        pose = drive((0.0, 0.0, 0.0), Twist(vx=1.0, **{name: 2.0}), 1.0)
    assert pose == pytest.approx(wanted)


def test_a_command_the_robot_can_follow_gives_no_warning(recwarn):
    """The warning is only for a nonzero field the model does not use."""
    drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=0.5), 1.0)
    assert not recwarn


def wheels(command, dt=1.0, **params):
    """The (left, right) wheel angles of a diff-drive for a command."""
    return DiffDrive(**params).encoders(command, dt)


def test_a_straight_command_turns_both_wheels_alike():
    """1 m/s for 1 s on wheels of radius 0.1 m is 10 rad on each."""
    assert wheels(Twist(vx=1.0), wheel_radius=0.1) == pytest.approx([10.0, 10.0])


def test_a_spin_turns_the_wheels_opposite_ways():
    """Wz = 1 on a track of 0.4 m: each wheel moves 0.2 m, left back, right on."""
    assert wheels(Twist(wz=1.0), wheel_radius=0.1, track=0.4) == pytest.approx(
        [-2.0, 2.0]
    )


@pytest.mark.parametrize(
    ("readings", "step"),
    [
        ([0.5, 0.5], (0.05, 0.0, 0.0)),  # both wheels 0.5 rad: 5 cm straight
        ([0.5, 0.3], (0.04, 0.0, -0.05)),  # right slower: a turn to the right
    ],
)
def test_wheel_increments_give_the_body_step(readings, step):
    """Wheel radius 0.1 m and track 0.4 m, worked by hand."""
    model = DiffDrive(wheel_radius=0.1, track=0.4)
    assert model.from_encoders(readings) == pytest.approx(step)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_a_wheel_size_must_be_a_positive_number(bad):
    """Radius and track are divided by, so they cannot be zero."""
    with pytest.raises(ValueError, match="wheel_radius"):
        DiffDrive(wheel_radius=bad)
    with pytest.raises(ValueError, match="track"):
        DiffDrive(track=bad)


# The Ackermann model. Its steering cone at the defaults, max_steer 0.5 and
# wheelbase 0.3: |wz| <= |vx| * tan(0.5) / 0.3.
REACH = math.tan(0.5) / 0.3


def test_an_ackermann_cannot_turn_in_place():
    """Vx = 0 forces wz = 0: the robot stands where it is."""
    assert Ackermann().clip(Twist(wz=1.0)) == Twist()
    assert drive((3.0, 2.0, 0.5), Twist(wz=1.0), 1.0, Ackermann()) == pytest.approx(
        (3.0, 2.0, 0.5)
    )


def test_an_ackermann_turn_is_cut_to_the_steering_cone():
    """Wz beyond |vx| * tan(max_steer) / wheelbase is cut, keeping vx and sign."""
    model = Ackermann()
    cut = model.clip(Twist(vx=1.0, wz=10.0))
    assert (cut.vx, cut.wz) == pytest.approx((1.0, REACH))
    cut = model.clip(Twist(vx=-1.0, wz=-10.0))
    assert (cut.vx, cut.wz) == pytest.approx((-1.0, -REACH))
    assert model.clip(Twist(vx=0.5, wz=0.1)) == Twist(vx=0.5, wz=0.1)


def test_an_ackermann_least_turning_radius_is_wheelbase_over_tan_max_steer():
    """At full lock the arc has radius wheelbase / tan(max_steer)."""
    model = Ackermann(max_steer=math.pi / 4, wheelbase=1.0)  # radius 1
    x, y, angle = drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=100.0), math.pi, model)
    assert (x, y) == pytest.approx((0.0, 2.0))
    assert abs(angle) == pytest.approx(math.pi)


def test_an_ackermann_step_is_exact():
    """The step is one arc, so the step length does not change the result."""
    command = Twist(vx=0.7, wz=-0.9)
    pose = Transform2D.from_xyt((1.0, 2.0, 0.3))
    for _ in range(100):
        pose = Ackermann().step(pose, command, 0.02)
    assert tuple(pose.xyt) == pytest.approx(
        drive((1.0, 2.0, 0.3), command, 2.0, Ackermann())
    )


def test_an_ackermann_speed_is_clipped_before_the_cone():
    """Vx above max_v is cut first, and the cone is taken at the cut speed."""
    cut = Ackermann(max_v=0.5).clip(Twist(vx=5.0, wz=10.0))
    assert (cut.vx, cut.wz) == pytest.approx((0.5, 0.5 * REACH))


def test_a_steering_angle_past_max_steer_is_refused_by_the_check():
    """A command that skipped clip and turns too tight raises, naming the limit."""
    with pytest.raises(ValueError, match="max_steer"):
        Ackermann()._check_steerable(Twist(vx=0.1, wz=5.0))
    Ackermann()._check_steerable(Ackermann().clip(Twist(vx=1.0, wz=100.0)))


def test_an_ackermann_rear_wheels_read_like_a_diff_drive():
    """Straight at 1 m/s on 0.1 m wheels is 10 rad each; a turn splits them by track."""
    model = Ackermann(wheel_radius=0.1, track=0.4, max_steer=1.0, wheelbase=0.1)
    assert model.encoders(Twist(vx=1.0), 1.0) == pytest.approx([10.0, 10.0])
    assert model.encoders(Twist(vx=1.0, wz=1.0), 1.0) == pytest.approx([8.0, 12.0])
    assert model.from_encoders([0.5, 0.3]) == pytest.approx((0.04, 0.0, -0.05))


@pytest.mark.parametrize("bad", [0.0, -0.1, math.pi / 2, math.pi, math.nan])
def test_an_ackermann_max_steer_must_be_an_angle_short_of_a_right_angle(bad):
    """Tan(max_steer) is the cone's slope, so the angle must be in (0, pi/2)."""
    with pytest.raises(ValueError, match="max_steer"):
        Ackermann(max_steer=bad)


@pytest.mark.parametrize("name", ["max_v", "wheelbase", "wheel_radius", "track"])
@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf])
def test_an_ackermann_size_must_be_a_positive_number(name, bad):
    """Every length and the speed limit are divided by or compared, so above 0."""
    with pytest.raises(ValueError, match=name):
        Ackermann(**{name: bad})


def test_an_ackermann_straight_line_follows_the_heading():
    """Wz = 0: the robot moves along its own x axis, nothing else changes."""
    x, y, angle = drive((1.0, 1.0, math.pi / 2), Twist(vx=1.0), 2.0, Ackermann())
    assert (x, y, angle) == pytest.approx((1.0, 3.0, math.pi / 2))


def test_an_ackermann_quarter_turn_is_the_exact_arc_of_the_steering_angle():
    """Wheelbase 1 m and steering atan(0.5) give radius vx / wz = 2 m.

    A quarter turn from the origin heading +x ends at (2, 2) heading +y,
    worked by hand.
    """
    model = Ackermann(max_steer=math.pi / 4, wheelbase=1.0)
    quarter = (math.pi / 2) / 0.5  # seconds until wz * dt = pi / 2
    x, y, angle = drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=0.5), quarter, model)
    assert (x, y, angle) == pytest.approx((2.0, 2.0, math.pi / 2))


@pytest.mark.parametrize("name", ["vy", "vz", "wx", "wy"])
def test_what_an_ackermann_cannot_follow_is_ignored_with_a_warning(name):
    """It moves as if the field were 0, and one warning names it."""
    wanted = drive((0.0, 0.0, 0.0), Twist(vx=1.0), 1.0, Ackermann())
    with pytest.warns(UserWarning, match=name):
        moved = drive((0.0, 0.0, 0.0), Twist(vx=1.0, **{name: 2.0}), 1.0, Ackermann())
    assert moved == pytest.approx(wanted)


def test_an_ackermann_command_it_can_follow_gives_no_warning(recwarn):
    """Vx and wz inside the cone are its own; no warning."""
    drive((0.0, 0.0, 0.0), Twist(vx=1.0, wz=0.5), 1.0, Ackermann())
    assert not recwarn


def test_an_ackermann_odometry_reproduces_the_step():
    """From_encoders(encoders(cmd)) is the body step the pose moved by.

    So noiseless dead reckoning tracks the true pose exactly.
    """
    model = Ackermann()
    command = model.clip(Twist(vx=0.8, wz=5.0))
    dt = 0.01
    body = model.from_encoders(model.encoders(command, dt))
    assert body == pytest.approx([command.vx * dt, 0.0, command.wz * dt])


def test_a_diff_drive_limit_is_flat_and_turns_in_place():
    """Its max_w holds at every speed, standing still included."""
    limits = DiffDrive(max_v=2.0, max_w=0.5).limits()
    assert limits.turns_in_place
    assert limits.max_v == 2.0
    assert [limits.max_w(v) for v in (0.0, 0.1, 2.0)] == [0.5, 0.5, 0.5]


def test_an_ackermann_limit_is_a_wedge_and_cannot_turn_in_place():
    """Its turn-rate bound grows with speed from zero at standstill."""
    model = Ackermann(max_steer=math.pi / 4, wheelbase=0.5)
    limits = model.limits()
    assert not limits.turns_in_place
    assert limits.max_w(0.0) == 0.0
    assert limits.max_w(1.0) == pytest.approx(2.0)  # tan(pi/4) / 0.5
    assert limits.max_w(-1.0) == pytest.approx(2.0)


@pytest.mark.filterwarnings("ignore:.* ignores vy")
@pytest.mark.parametrize(
    "model", [DiffDrive(), Ackermann(), DiffDrive(max_w=0.3), Mecanum(max_vy=0.4)]
)
@pytest.mark.parametrize(
    "command",
    [
        Twist(),
        Twist(vx=5.0, wz=-5.0),
        Twist(vx=-0.4, wz=2.0),
        Twist(wz=1.0),
        Twist(vx=0.3, vy=-2.0, wz=0.5),
    ],
)
def test_the_limits_clip_as_the_model_does(model, command):
    """Limits.clip is the model's clip, so there is one source of truth."""
    assert model.limits().clip(command) == model.clip(command)


def test_the_limits_refuse_a_cone_with_no_bound():
    """Cap and slope cannot both be infinite, or every turn rate would pass."""
    with pytest.raises(ValueError, match="cap and slope"):
        Limits(max_v=1.0)


@pytest.mark.parametrize("name", ["max_v", "cap", "slope"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_a_limit_bound_must_be_above_zero(name, bad):
    """A bound of zero or less, or nan, is refused by name."""
    values = {"max_v": 1.0, "cap": 1.0}
    values[name] = bad
    with pytest.raises(ValueError, match=name):
        Limits(**values)


def old_clip(limits, command):
    """Limits.clip as it was before max_vy: vx clamped, then wz at that vx."""
    vx = max(-limits.max_v, min(limits.max_v, command.vx))
    bound = limits.max_w(vx)
    return Twist(vx=vx, wz=max(-bound, min(bound, command.wz)))


@pytest.mark.parametrize(
    "model", [DiffDrive(), DiffDrive(max_v=0.7, max_w=0.3), Ackermann()]
)
def test_a_robot_that_cannot_strafe_clips_as_before_bit_for_bit(model):
    """Max_vy changed nothing for diff drive and Ackermann, to the last bit."""
    limits = model.limits()
    assert limits.max_vy == 0.0
    assert not limits.strafes
    rng = np.random.default_rng(7)
    for vx, vy, wz in rng.uniform(-5, 5, (2000, 3)):
        command = Twist(vx=vx, vy=vy, wz=wz)
        assert limits.clip(command) == old_clip(limits, command)


def test_the_limits_keep_vy_only_for_a_robot_that_strafes():
    """With max_vy above 0, vy is cut to it; with 0, it is dropped."""
    command = Twist(vx=0.2, vy=0.3, wz=0.1)
    assert Limits(max_v=1.0, cap=1.0).clip(command) == Twist(vx=0.2, wz=0.1)
    assert Limits(max_v=1.0, cap=1.0, max_vy=0.5).clip(command) == command


def test_the_limits_shrink_vx_and_vy_by_one_factor():
    """A command toward (2, 2) with bounds (1, 0.5) keeps its 45 degrees."""
    limits = Limits(max_v=1.0, cap=1.0, max_vy=0.5)
    assert limits.clip(Twist(vx=2.0, vy=2.0)) == Twist(vx=0.5, vy=0.5)
    assert limits.clip(Twist(vx=-4.0, vy=1.0)) == Twist(vx=-1.0, vy=0.25)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_a_sideways_bound_must_be_finite_and_not_negative(bad):
    """Max_vy may be 0, a robot that cannot strafe, but not less, nan or infinite."""
    with pytest.raises(ValueError, match="max_vy"):
        Limits(max_v=1.0, cap=1.0, max_vy=bad)
