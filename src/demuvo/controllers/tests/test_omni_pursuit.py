"""Tests for the omni pursuit controller, a path follower for a robot that strafes."""

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import (
    DiffDrive,
    Mecanum,
    Observation,
    OmniPursuit,
    Run,
    StraightLine,
    Twist,
    load_world,
)
from demuvo.controllers.omni_pursuit import body_velocity
from demuvo.options import controllers, settings_of

MECANUM = Mecanum().limits()
EXAMPLE = (
    Path(__file__).resolve().parents[4] / "examples" / "worlds" / "omni_pursuit.toml"
)


def look(pose, goal, limits=MECANUM, tolerance=(0.1, 0.1), path=None, dt=0.05):
    """An Observation for a robot at pose with a straight path to goal."""
    pose, goal = Transform2D.from_xyt(pose), Transform2D.from_xyt(goal)
    if path is None:
        path = StraightLine().plan(pose, goal)
    return Observation(
        time=0.0,
        dt=dt,
        pose=pose,
        path=path,
        goal=goal,
        goal_tolerance=tolerance,
        limits=limits,
    )


def drive(controller, pose, goal, model=None, seconds=20.0, dt=0.05, **kwargs):
    """Follow a straight path to goal on `model`; the poses after each run."""
    model = model or Mecanum()
    start, end = Transform2D.from_xyt(pose), Transform2D.from_xyt(goal)
    path = StraightLine().plan(start, end)
    pose, poses = start, []
    for i in range(round(seconds / dt)):
        obs = Observation(
            i * dt, dt, pose, path, end, (0.1, 0.1), limits=model.limits(), **kwargs
        )
        pose = model.step(pose, controller.command(obs), dt)
        poses.append(pose)
    return poses


def test_a_goal_straight_ahead_gives_the_set_speed_and_nothing_else():
    """Facing a far goal with its angle: vx = speed, no sideways speed, no turn."""
    command = OmniPursuit().command(look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)))
    assert (command.vx, command.vy, command.wz) == pytest.approx((0.5, 0.0, 0.0))


def test_a_goal_to_the_left_is_reached_by_strafing_without_turning():
    """The robot moves sideways at speed and keeps facing as the goal does."""
    command = OmniPursuit().command(look((1.0, 1.0, 0.0), (1.0, 4.0, 0.0)))
    assert (command.vx, command.vy, command.wz) == pytest.approx((0.0, 0.5, 0.0))


def test_a_diagonal_aim_moves_along_the_diagonal_at_speed():
    """Facing +x, a goal up and to the right: the travel points at it, at `speed`."""
    command = OmniPursuit().command(look((0.0, 0.0, 0.0), (3.0, 3.0, 0.0)))
    assert math.atan2(command.vy, command.vx) == pytest.approx(math.pi / 4)
    assert math.hypot(command.vx, command.vy) == pytest.approx(0.5)


def test_the_robot_slows_down_near_the_goal():
    """0.3 m from the goal with approach_gain 1: 0.3 m/s, whatever the direction."""
    command = OmniPursuit().command(look((1.0, 1.0, 0.0), (1.0, 1.3, 0.0)))
    assert math.hypot(command.vx, command.vy) == pytest.approx(0.3)


def test_it_turns_toward_the_goal_angle_while_it_moves():
    """Goal angle 0.4 rad off: wz = turn_gain * error, and it still moves."""
    command = OmniPursuit().command(look((1.0, 1.0, 0.0), (5.0, 1.0, 0.4)))
    assert command.wz == pytest.approx(0.8)
    assert math.hypot(command.vx, command.vy) == pytest.approx(0.5, rel=1e-3)


def test_the_turn_is_limited_by_rotate_speed_and_by_the_robot():
    """A large error turns at rotate_speed, and never above the robot's max_w."""
    obs = look((1.0, 1.0, 0.0), (5.0, 1.0, 3.0))
    assert OmniPursuit(rotate_speed=0.6).command(obs).wz == pytest.approx(0.6)
    slow = look((1.0, 1.0, 0.0), (5.0, 1.0, 3.0), limits=Mecanum(max_w=0.3).limits())
    assert OmniPursuit(rotate_speed=2.0).command(slow).wz == pytest.approx(0.3)


def test_face_path_faces_along_the_path_until_the_goal_is_near():
    """Far off it turns to the path's heading; within lookahead, to the goal's angle."""
    far = OmniPursuit(face_path=True).command(look((0.0, 0.0, 0.0), (0.0, 4.0, 0.0)))
    assert far.wz > 0.0  # the path heads +y, so it turns left to face it
    near = OmniPursuit(face_path=True).command(look((0.0, 0.0, 1.0), (0.0, 0.5, 0.0)))
    assert near.wz < 0.0  # the goal faces +x, so it turns back right


@pytest.mark.parametrize("angle", [0.05, -0.05])
def test_at_the_goal_with_the_right_angle_it_stops(angle):
    """Inside both tolerances there is nothing left to do."""
    command = OmniPursuit().command(look((5.0, 1.0, angle), (5.0, 1.0, 0.0)))
    assert command == Twist()


@pytest.mark.parametrize(("angle", "sign"), [(1.0, 1.0), (-1.0, -1.0), (3.0, 1.0)])
def test_at_the_goal_it_turns_on_the_spot_the_short_way(angle, sign):
    """Inside the distance tolerance it only turns, toward the goal's angle."""
    command = OmniPursuit().command(look((5.0, 1.0, 0.0), (5.0, 1.0, angle)))
    assert command.vx == 0.0 and command.vy == 0.0
    assert math.copysign(1.0, command.wz) == sign


def test_a_command_is_always_one_the_robot_will_follow():
    """Fast settings against a slow robot: the command is inside its limits."""
    model = Mecanum(max_v=0.4, max_vy=0.2, max_w=0.5)
    obs = look((1.0, 1.0, 0.0), (2.0, 4.0, 2.0), limits=model.limits())
    command = OmniPursuit(speed=2.0, rotate_speed=3.0).command(obs)
    assert model.clip(command) == command


def test_a_command_too_fast_sideways_keeps_its_direction():
    """A strafe at 2 m/s on a robot of max_vy 0.3 is cut to 0.3, not bent."""
    obs = look((1.0, 1.0, 0.0), (1.0, 5.0, 0.0), limits=Mecanum(max_vy=0.3).limits())
    command = OmniPursuit(speed=2.0).command(obs)
    assert (command.vx, command.vy) == pytest.approx((0.0, 0.3))


def test_the_body_velocity_moves_the_robot_exactly_on_its_aim():
    """Held for dt while turning, the corrected command moves by world * dt.

    The command is a body twist and sweeps an arc; turning the aim into the frame
    at the mid-step heading and dividing by sin(h) / h puts the chord on it
    (design, section 5.5).
    """
    for yaw, wz, dt in [(0.3, 1.0, 0.1), (-2.0, -1.5, 0.5), (1.0, 0.0, 0.2)]:
        wx, wy = 0.4, -0.25
        vx, vy = body_velocity(wx, wy, yaw, wz, dt)
        start = Transform2D.from_xyt((1.0, 2.0, yaw))
        end = Mecanum(max_w=2.0).step(start, Twist(vx=vx, vy=vy, wz=wz), dt)
        assert end.xyt[:2] == pytest.approx([1.0 + wx * dt, 2.0 + wy * dt], abs=1e-12)


def test_turning_on_the_way_does_not_drift_off_a_straight_path():
    """A quarter turn on the way: the robot stays on the line, and arrives aligned."""
    poses = drive(OmniPursuit(), (0.0, 0.0, 0.0), (3.0, 0.0, math.pi / 2))
    assert max(abs(p.xyt[1]) for p in poses) < 1e-9
    x, y, yaw = poses[-1].xyt
    assert math.hypot(x - 3.0, y) <= 0.1 and abs(yaw - math.pi / 2) <= 0.1
    assert poses[len(poses) // 10].xyt[2] > 0.3  # it turned on the way, not at the end


def test_without_limits_it_still_drives_a_strafe():
    """A caller that does not know the model gets the command it would send."""
    obs = look((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), limits=None)
    assert OmniPursuit().command(obs).vy == pytest.approx(0.5)


def test_a_robot_that_cannot_strafe_stays_still_and_the_log_says_so_once(caplog):
    """A diff drive cannot follow the aim sideways, so it stands; one warning."""
    controller = OmniPursuit()
    obs = look((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), limits=DiffDrive().limits())
    with caplog.at_level(logging.WARNING):
        commands = [controller.command(obs) for _ in range(3)]
    assert commands == [Twist()] * 3
    assert sum("cannot move sideways" in r.message for r in caplog.records) == 1


def test_an_empty_path_gives_a_zero_command():
    """No path, nothing to follow."""
    obs = look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), path=np.zeros((0, 3)))
    assert OmniPursuit().command(obs) == Twist()


def test_on_the_last_path_point_short_of_the_goal_it_only_turns():
    """With no direction to aim at, it turns toward the goal heading and stays put."""
    path = np.array([[1.0, 1.0, 0.0]])
    obs = look((1.0, 1.0, 0.0), (5.0, 1.0, 1.0), path=path)
    command = OmniPursuit().command(obs)
    assert command.vx == command.vy == 0.0 and command.wz > 0.0


def test_a_path_with_no_goal_is_an_error():
    """A path needs a goal to finish at."""
    obs = look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0))
    obs = Observation(obs.time, obs.dt, obs.pose, obs.path, None, obs.goal_tolerance)
    with pytest.raises(ValueError, match="needs a goal to follow a path"):
        OmniPursuit().command(obs)


@pytest.mark.parametrize(
    "name", ["speed", "lookahead", "approach_gain", "turn_gain", "rotate_speed"]
)
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, True])
def test_a_bad_number_is_refused_when_the_controller_is_built(name, bad):
    """Every tunable number must be positive and finite."""
    with pytest.raises(ValueError, match=name):
        OmniPursuit(**{name: bad})


def test_the_window_lists_omni_pursuit_and_its_settings():
    """The inspector offers it, with its public settings only."""
    assert controllers()["Omni pursuit"] == "demuvo.controllers:OmniPursuit"
    names = [s.name for s in settings_of("demuvo.controllers:OmniPursuit")]
    assert names == [
        "speed",
        "lookahead",
        "approach_gain",
        "turn_gain",
        "rotate_speed",
        "face_path",
    ]


def test_the_example_laps_the_box_without_turning():
    """Every goal faces +x, so it strafes and backs round, its heading never moving."""
    run = Run(load_world(EXAMPLE))
    headings = []
    while not run.until_done(run.time + 0.5) and run.time < 60.0:
        headings.append(abs(run.sim.poses["omni"].xyt[2]))
    assert run.finished
    assert run.result()[-1] == "0 collisions"
    assert max(headings) < 1e-9
