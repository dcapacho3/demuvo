"""Tests for the DWB controller."""

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import DWB, DiffDrive, Observation, Run, StraightLine, Twist
from demuvo.controllers import dwb
from demuvo.costmap import LETHAL, Costmap
from demuvo.costmap.layers import inflate
from demuvo.options import controllers, settings_of

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


def look(pose, goal, costmap=None, path=None, tolerance=(0.1, 0.1), changes=1):
    """An Observation for a robot at pose with a straight path to goal."""
    pose, goal = Transform2D.from_xyt(pose), Transform2D.from_xyt(goal)
    if path is None:
        path = StraightLine().plan(pose, goal)
    return Observation(
        time=0.0,
        dt=0.05,
        pose=pose,
        path=path,
        goal=goal,
        goal_tolerance=tolerance,
        path_changes=changes,
        costmap=costmap,
    )


def drive(controller, obs, runs):
    """Run the controller `runs` times, moving the robot by each command."""
    model, pose = DiffDrive(), obs.pose
    for _ in range(runs):
        obs = Observation(
            obs.time + obs.dt,
            obs.dt,
            pose,
            obs.path,
            obs.goal,
            obs.goal_tolerance,
            obs.path_changes,
            costmap=obs.costmap,
        )
        pose = model.step(pose, controller.command(obs), obs.dt)
    return pose


def test_a_clear_path_ahead_drives_forward_and_straight():
    """On a free straight path the best command goes along it, not round."""
    controller = DWB()
    obs = look((1.5, 3.0, 0.0), (4.5, 3.0, 0.0), window())
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
    obs = look((3.0, 3.0, 0.0), (4.4, 3.0, 0.0), window(walls))
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
        ({"min_vel_x": 2.0}, "min_vel_x"),
        ({"vx_samples": 0}, "vx_samples"),
        ({"vtheta_samples": 2.5}, "vtheta_samples"),
    ],
)
def test_other_bad_settings_are_refused(kwargs, match):
    """Weights are 0 or more, min_vel_x is at most max_vel_x, samples are counts."""
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
