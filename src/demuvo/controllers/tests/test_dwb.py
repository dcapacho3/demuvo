"""Tests for the DWB controller."""

# pylint: disable=protected-access  # these tests check the module's own helpers

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import (
    DWB,
    Ackermann,
    DiffDrive,
    Mecanum,
    Observation,
    Run,
    StraightLine,
    Twist,
    edit,
    load_world,
)
from demuvo.controllers import dwb
from demuvo.costmap import LETHAL, Costmap
from demuvo.costmap.layers import inflate
from demuvo.options import controllers, hidden_settings, settings_of

from .observations import look, moved, see

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "dwb.toml"


def window(marks=(), centre=(3.0, 3.0), size=3.0, resolution=0.05):
    """A local costmap round `centre`, with LETHAL boxes (x0, y0, x1, y1), inflated."""
    cells = round(size / resolution)
    data = np.zeros((cells, cells), np.uint8)
    origin = (centre[0] - size / 2, centre[1] - size / 2)
    grid = Costmap(data, resolution, origin)
    for x0, y0, x1, y1 in marks:
        r0, c0 = grid.world_to_cell(x0, y0)
        r1, c1 = grid.world_to_cell(x1, y1)
        data[max(r0, 0) : r1 + 1, max(c0, 0) : c1 + 1] = LETHAL
    return Costmap(inflate(data, resolution, 0.2, 0.55, 10.0), resolution, origin)


def drive(controller, obs, runs):
    """Run the controller `runs` times, moving the robot by each command."""
    model, pose = DiffDrive(), obs.pose
    for _ in range(runs):
        obs = moved(obs, pose)
        pose = model.step(pose, controller.command(obs), obs.dt)
    return pose


def test_a_clear_path_ahead_drives_forward_and_straight():
    """On a free straight path the best command goes along it, not round."""
    controller = DWB()
    obs = look((1.5, 3.0, 0.0), (4.5, 3.0, 0.0), costmap=window())
    pose = drive(controller, obs, 40)
    assert controller.command(obs).vx > 0.0
    x, y, yaw = pose.xyt
    assert x > 2.3 and abs(y - 3.0) < 0.02 and abs(yaw) < 0.05


def test_the_speed_grows_no_faster_than_the_acceleration_limit():
    """From rest the first command is at most acc_lim_x * dt."""
    command = DWB(acc_lim_x=1.0).command(look((1.5, 3.0, 0.0), (4.5, 3.0, 0.0)))
    assert 0.0 < command.vx <= 1.0 * 0.05 + 1e-12


def test_the_robot_leaves_a_path_the_lidar_finds_blocked_and_passes_the_block():
    """A box on the path in the local costmap: drive round it without touching it."""
    box = (3.3, 2.7, 3.6, 3.3)
    controller, model = DWB(), DiffDrive()
    pose = Transform2D.from_xyt((2.0, 3.0, 0.0))
    goal = Transform2D.from_xyt((4.4, 3.0, 0.0))
    path = StraightLine().plan(pose, goal)
    closest = math.inf
    for step in range(400):
        costmap = window([box], centre=(3.0, 3.0), size=4.0)
        obs = Observation(
            step * 0.05, 0.05, pose, path, goal, (0.1, 0.1), 1, costmap=costmap
        )
        pose = model.step(pose, controller.command(obs), 0.05)
        x, y = pose.xyt[:2]
        closest = min(
            closest,
            math.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - y, 0, y - box[3])),
        )
        if math.dist((x, y), (4.4, 3.0)) < 0.1:
            break
    assert math.dist(pose.xyt[:2], (4.4, 3.0)) < 0.1
    assert closest > 0.2  # the robot's radius: it never touched the box


def test_a_robot_shut_in_stops_and_the_log_says_so_once(caplog):
    """No trajectory that moves is free: stand still, and warn once for the path."""
    walls = [(2.4, 2.0, 2.6, 4.0), (3.4, 2.0, 3.6, 4.0)]
    walls += [(2.4, 2.4, 3.6, 2.6), (2.4, 3.4, 3.6, 3.6)]
    controller = DWB()
    obs = look((3.0, 3.0, 0.0), (4.4, 3.0, 0.0), costmap=window(walls))
    with caplog.at_level(logging.WARNING, logger="demuvo.controllers.dwb"):
        assert controller.command(obs) == Twist()
        assert controller.command(obs) == Twist()
    assert len(caplog.records) == 1
    assert "stops until one is" in caplog.records[0].getMessage()


def test_a_robot_with_no_local_costmap_follows_its_path():
    """With no lidar it scores on a blank grid and still gets along the path."""
    pose = drive(DWB(), look((1.0, 1.0, 0.0), (3.0, 1.0, 0.0)), 60)
    assert pose.xyt[0] > 2.0 and abs(pose.xyt[1] - 1.0) < 0.05


def test_an_empty_path_gives_a_zero_command():
    """There is nothing to follow, so the robot stays still."""
    obs = look((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), path=np.empty((0, 3)))
    assert DWB().command(obs) == Twist()


def test_a_path_with_no_goal_is_an_error():
    """A path that leads nowhere is a mistake, and the message says so."""
    obs = Observation(
        0.0, 0.05, Transform2D.identity(), np.zeros((2, 3)), None, (0.1, 0.1)
    )
    with pytest.raises(ValueError, match="needs a goal to follow a path"):
        DWB().command(obs)


def test_at_the_goal_it_turns_to_the_goal_angle_then_stops():
    """Inside the distance tolerance: turn on the spot, then stop when aligned."""
    assert DWB().command(look((3.0, 3.0, 0.0), (3.0, 3.0, 1.0))) == Twist(wz=1.0)
    assert DWB().command(look((3.0, 3.0, 0.0), (3.02, 3.0, 0.05))) == Twist()
    assert (
        DWB(rotate_to_goal=False).command(look((3.0, 3.0, 0.0), (3.0, 3.0, 1.0)))
        == Twist()
    )


def test_a_rollout_lands_where_the_diff_drive_model_moves():
    """Each rolled-out pose is the model's exact step for that time."""
    start = np.array([1.0, 2.0, 0.3])
    vx, wz = np.array([0.4, 0.4, 0.0]), np.array([0.0, -0.7, 0.9])
    poses = dwb.rollout(start, vx, wz, 1.0, 0.25)
    assert poses.shape == (3, 4, 3)
    for k in range(3):
        end = DiffDrive().step(
            Transform2D.from_xyt(start), Twist(vx=vx[k], wz=wz[k]), 1.0
        )
        assert poses[k, -1, :2] == pytest.approx(end.xyt[:2])
        assert poses[k, -1, 2] == pytest.approx(0.3 + wz[k])


def test_the_wavefront_goes_round_a_wall_and_skips_blocked_seeds():
    """Distances flow through the gap; a seed on a blocked cell seeds nothing."""
    free = np.ones((5, 5), bool)
    free[:4, 2] = False  # a wall in column 2 with a gap in the top row
    steps = dwb.wavefront(free, (np.array([0, 1]), np.array([0, 2])))
    assert steps[0, 0] == 0.0 and np.isinf(steps[1, 2])
    # Up beside the wall, over through the gap, down the far side; no diagonal
    # step may cut a wall cell's corner, so the ends of the wall are straight.
    assert steps[0, 4] == pytest.approx(8 + 2 * math.sqrt(2))
    assert np.isinf(dwb.wavefront(free, (np.array([1]), np.array([2])))).all()


def test_a_path_of_one_point_samples_as_that_point():
    """A degenerate path has no polyline to walk; its point comes back alone."""
    path = np.array([[1.0, 2.0, 0.5]])
    points = dwb.along_path(path, 0.025)
    assert points.shape == (1, 2)
    assert points[0] == pytest.approx([1.0, 2.0])
    points[0] = 0.0  # a copy: the path is left untouched
    assert path[0, 0] == 1.0


def test_no_usable_path_point_leaves_infinite_path_to_the_goal():
    """With every path point blocked or outside, the goal critics see inf."""
    grid = window()
    free = ~dwb.grow(grid.data >= dwb.INSCRIBED)
    points = np.array([[3.0, 3.0], [3.1, 3.0]])
    maps = dwb._Maps(
        grid=grid,
        free=free,
        to_path=np.zeros(free.shape),
        to_goal=np.zeros(free.shape),
        points=points,
        arc=np.array([0.0, 0.1]),
        inside=np.array([True, True]),
        blocked=np.array([True, True]),
    )
    left = dwb.left_along(points, maps, usable=np.zeros(2, bool))
    assert np.isinf(left).all()


def test_a_blocked_path_aims_at_its_last_free_point():
    """While the path ahead is blocked, the angle critics aim at the local goal."""
    grid = window()
    free = ~dwb.grow(grid.data >= dwb.INSCRIBED)
    maps = dwb._Maps(
        grid=grid,
        free=free,
        to_path=np.zeros(free.shape),
        to_goal=np.zeros(free.shape),
        points=np.array([[3.0, 3.0], [3.1, 3.0], [3.2, 3.0]]),
        arc=np.array([0.0, 0.1, 0.2]),
        inside=np.array([True, True, True]),
        blocked=np.array([False, True, False]),
    )
    assert DWB()._aim(maps, 0, 0.05) == pytest.approx([3.2, 3.0])


def test_blocked_cells_grow_by_one_cell_all_round():
    """The footprint margin: a lone blocked cell becomes a block of nine."""
    blocked = np.zeros((5, 5), bool)
    blocked[2, 2] = True
    assert dwb.grow(blocked).sum() == 9


@pytest.mark.parametrize(
    "name",
    ["max_vel_x", "max_vel_theta", "acc_lim_x", "acc_lim_theta", "sim_time"]
    + ["time_granularity", "rotate_speed", "turn_gain"],
)
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, True])
def test_a_bad_positive_number_is_refused_when_the_controller_is_built(name, bad):
    """Every speed, limit and time must be positive and finite."""
    with pytest.raises(ValueError, match=name):
        DWB(**{name: bad})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"obstacle_scale": -1.0}, "obstacle_scale"),
        ({"cost_weight": math.nan}, "cost_weight"),
        ({"vx_samples": 0}, "vx_samples"),
        ({"vtheta_samples": 2.5}, "vtheta_samples"),
    ],
)
def test_other_bad_settings_are_refused(kwargs, match):
    """Weights are 0 or more, samples are counts."""
    with pytest.raises(ValueError, match=match):
        DWB(**kwargs)


def test_the_window_lists_dwb_and_its_settings():
    """The inspector offers DWB, and only its public settings."""
    assert controllers()["DWB"] == "demuvo.controllers:DWB"
    names = [s.name for s in settings_of("demuvo.controllers:DWB")]
    assert "max_vel_x" in names and "sim_time" in names
    assert not any(name.startswith("_") for name in names)


def test_the_example_world_goes_round_what_only_the_lidar_sees():
    """The straight path runs through both obstacles; the robot gets there unhurt."""
    run = Run(EXAMPLE)
    assert run.until_done(60.0)
    lines = run.result()
    assert lines[0].startswith("r1 task 1: reached")
    assert lines[-1] == "0 collisions"
    again = Run(EXAMPLE)
    again.until_done(60.0)
    assert again.result() == lines  # the same seed gives the same run


def steer(controller, obs, model, runs):
    """Run the controller `runs` times on `model`, returning the poses and commands."""
    pose, poses, commands = obs.pose, [], []
    for _ in range(runs):
        obs = moved(obs, pose)
        command = controller.command(obs)
        commands.append(command)
        pose = model.step(pose, command, obs.dt)
        poses.append(pose)
    return poses, commands


@pytest.mark.parametrize(
    ("pose", "goal"),
    [
        ((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)),  # straight ahead
        ((1.0, 1.0, 0.0), (1.0, 4.0, 1.57)),  # 90 degrees off
        ((1.0, 1.0, 0.0), (0.3, 1.0, 3.14)),  # behind
        ((5.0, 1.0, 0.0), (5.0, 1.0, 1.0)),  # at the goal, wrong angle
    ],
)
def test_a_diff_drive_is_driven_exactly_as_without_limits(pose, goal):
    """Limits that turn in place change nothing, so a diff drive keeps its behaviour."""
    plain = list(steer(DWB(), look(pose, goal), DiffDrive(), 20)[1])
    limited = steer(DWB(), see(pose, goal, DiffDrive().limits()), DiffDrive(), 20)[1]
    assert limited == plain


def test_a_car_is_only_given_commands_it_will_drive():
    """Every command lies in the steering cone, so the scored arc is the driven arc."""
    model = Ackermann()
    _, commands = steer(
        DWB(), see((1.0, 1.0, 0.0), (1.0, 4.0, 1.57), model.limits()), model, 40
    )
    assert all(model.clip(c) == c for c in commands)
    assert all(c.vx >= 0.0 for c in commands)


def test_a_car_swings_toward_a_path_at_its_side_instead_of_inching():
    """Goal 90 degrees to the left: within 2 s the car has turned well to the left."""
    model = Ackermann()
    poses, commands = steer(
        DWB(), see((1.0, 1.0, 0.0), (1.0, 4.0, 1.57), model.limits()), model, 40
    )
    assert poses[-1].xyt[2] > 0.5
    assert all(c.vx >= DWB().car_min_vel_x - 1e-9 for c in commands[3:])


def test_a_car_with_the_path_behind_turns_rather_than_driving_off():
    """Goal straight behind: the heading error to it shrinks, it does not drive away."""
    model = Ackermann()
    poses, _ = steer(
        DWB(), see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), model.limits()), model, 60
    )
    assert abs(poses[-1].xyt[2]) > 1.0
    assert poses[-1].xyt[0] < 2.0


def test_a_car_at_the_goal_stops_and_leaves_its_angle():
    """It cannot turn on the spot to the goal's angle, so it stands still."""
    command = DWB().command(see((5.0, 1.0, 0.0), (5.0, 1.0, 1.0), Ackermann().limits()))
    assert command == Twist()


def test_a_reversing_car_backs_toward_a_goal_behind():
    """With allow_reversing a car backs up instead of looping round forwards."""
    model = Ackermann()
    poses, commands = steer(
        DWB(allow_reversing=True),
        see((1.0, 1.0, 0.0), (0.3, 1.0, 3.14), model.limits()),
        model,
        60,
    )
    assert any(c.vx < 0.0 for c in commands)
    assert all(model.clip(c) == c for c in commands)
    assert poses[-1].xyt[0] < 1.0  # it went backwards, toward the goal


# A robot that can move sideways: DWB's window gets a third axis, vy.
MECANUM_EXAMPLE = EXAMPLE.parent / "mecanum_dwb.toml"


@pytest.mark.parametrize("model", [DiffDrive(), Ackermann()], ids=["diff", "car"])
@pytest.mark.parametrize(
    ("pose", "goal"),
    [((1.0, 1.0, 0.0), (5.0, 1.0, 0.0)), ((1.0, 1.0, 0.0), (1.0, 4.0, 1.57))],
)
def test_a_robot_that_cannot_strafe_ignores_the_sideways_settings(model, pose, goal):
    """No vy is tried or sent, so the commands are those of DWB without the axis."""
    obs = see(pose, goal, model.limits())
    plain = steer(DWB(), obs, model, 20)[1]
    other = steer(DWB(max_vel_y=0.9, acc_lim_y=9.0, vy_samples=9), obs, model, 20)[1]
    assert other == plain
    assert all(c.vy == 0.0 for c in plain)


def test_a_strafing_robot_slides_toward_a_path_at_its_side():
    """Goal to the left: it strafes from the first run, at acc_lim_y per run."""
    model = Mecanum()
    poses, commands = steer(
        DWB(), see((1.0, 1.0, 0.0), (1.0, 4.0, 1.57), model.limits()), model, 40
    )
    assert [c.vy for c in commands[:4]] == pytest.approx([0.125, 0.25, 0.375, 0.5])
    assert poses[-1].xyt[1] > 2.0


def test_a_strafing_robot_is_only_given_commands_it_will_follow():
    """Every command is inside the robot's limits, so the scored arc is the driven."""
    model = Mecanum(max_vy=0.2)
    _, commands = steer(
        DWB(max_vel_y=0.5),
        see((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), model.limits()),
        model,
        40,
    )
    assert all(model.clip(c) == c for c in commands)
    assert max(abs(c.vy) for c in commands) == pytest.approx(0.2)


def test_the_sideways_speed_grows_no_faster_than_its_acceleration_limit():
    """From rest the first sideways speed is at most acc_lim_y * dt."""
    obs = see((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), Mecanum().limits())
    assert abs(DWB(acc_lim_y=1.0).command(obs).vy) <= 1.0 * obs.dt + 1e-12


def test_a_rollout_with_a_sideways_speed_lands_where_the_mecanum_model_moves():
    """Each rolled-out pose is the mecanum model's exact step for that time."""
    start = np.array([1.0, 2.0, 0.3])
    vx, vy = np.array([0.4, 0.0, -0.2]), np.array([0.0, 0.5, 0.3])
    wz = np.array([0.6, -0.7, 0.0])
    poses = dwb.rollout(start, vx, wz, 1.0, 0.25, vy=vy)
    for k in range(3):
        end = Mecanum().step(
            Transform2D.from_xyt(start), Twist(vx=vx[k], vy=vy[k], wz=wz[k]), 1.0
        )
        assert poses[k, -1] == pytest.approx(end.xyt)


@pytest.mark.parametrize("name", ["max_vel_y", "acc_lim_y"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, True])
def test_a_bad_sideways_number_is_refused_when_the_controller_is_built(name, bad):
    """The sideways speed and acceleration limits must be positive and finite."""
    with pytest.raises(ValueError, match=name):
        DWB(**{name: bad})


@pytest.mark.parametrize("bad", [0, 2.5, True])
def test_the_sideways_samples_are_a_count(bad):
    """Vy_samples is a whole number of 1 or more."""
    with pytest.raises(ValueError, match="vy_samples"):
        DWB(vy_samples=bad)


def test_the_sideways_settings_show_only_for_a_robot_that_strafes():
    """A form hides max_vel_y, acc_lim_y and vy_samples unless the robot strafes."""
    world = edit.set_part(
        load_world(EXAMPLE), "r1", "controller", "demuvo.controllers:DWB"
    )
    sideways = set(DWB.strafe_only)
    assert sideways <= hidden_settings(world.robots[0], "controller")
    for model in ("ackermann", "mecanum"):
        spec = edit.set_model(world, "r1", model).robots[0]
        hidden = hidden_settings(spec, "controller")
        assert (sideways <= hidden) == (model != "mecanum")


def test_the_mecanum_example_slides_round_what_only_the_lidar_sees():
    """It reaches the goal past both obstacles, unhurt, strafing on the way."""
    run = Run(MECANUM_EXAMPLE)
    sideways = []
    while not run.until_done(run.time + 0.5) and run.time < 60.0:
        sideways.append(abs(run.sim.velocity["r1"][1]))
    assert run.finished
    lines = run.result()
    assert lines[0].startswith("r1 task 1: reached")
    assert lines[-1] == "0 collisions"
    assert max(sideways) > 0.3
