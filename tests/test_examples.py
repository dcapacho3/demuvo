"""The example worlds and the example plugins keep working."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Run, Twist
from demuvo.controllers import Observation
from demuvo.costmap import Costmap

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WORLDS = sorted((EXAMPLES / "worlds").glob("*.toml"))


@pytest.fixture(name="plugins")
def plugins_on_the_path(monkeypatch):
    """The example plugins folder on the import path, as the README says to do.

    The fixture gives a function that imports a plugin module by name.
    """
    monkeypatch.syspath_prepend(str(EXAMPLES / "plugins"))
    yield importlib.import_module
    for name in ("breadth_first", "turn_then_drive"):
        sys.modules.pop(name, None)


@pytest.mark.usefixtures("plugins")
@pytest.mark.parametrize("path", WORLDS, ids=lambda p: p.name)
def test_every_example_world_loads_and_steps(path):
    """A world file in examples/worlds opens and runs a few steps."""
    run = Run(path)
    run.step(20)
    assert run.time > 0


@pytest.mark.usefixtures("plugins")
def test_the_plugin_race_ends_with_both_robots_at_their_goals_and_no_collision():
    """The robot driven by the example plugins finishes, beside the built-in one."""
    run = Run(EXAMPLES / "worlds" / "plugins.toml")
    assert run.until_done(120.0)
    lines = run.result()
    assert any(line.startswith("r1 task 1: reached") for line in lines)
    assert any(line.startswith("r2 task 1: reached") for line in lines)
    assert lines[-1] == "0 collisions"


def pose(x, y, yaw=0.0):
    """A pose from numbers."""
    return Transform2D.from_xyt((x, y, yaw))


def test_breadth_first_goes_round_a_wall_in_fewest_steps_and_ends_on_the_goal(plugins):
    """A wall with a gap at the top: the path goes through the gap."""
    data = np.zeros((20, 20), np.uint8)
    data[:15, 10] = 254  # a wall at x = 5 m with a gap above y = 7.5 m
    planner = plugins("breadth_first").BreadthFirst()
    planner.configure(Costmap(data, 0.5))
    path = planner.plan(pose(1.2, 1.2), pose(8.8, 1.2, 0.5))
    assert path.shape[1] == 3 and len(path) > 15
    assert path[0, :2].tolist() == [1.2, 1.2] and path[-1, :2].tolist() == [8.8, 1.2]
    assert path[-1, 2] == 0.5 and path[:, 1].max() > 7.5
    cells = np.floor(path[:, :2] / 0.5).astype(int)
    assert (data[cells[:, 1], cells[:, 0]] < 254).all()


def test_breadth_first_gives_an_empty_path_when_there_is_no_way(plugins):
    """A start or goal in a wall or off the map, and a goal shut in, all give none."""
    data = np.zeros((20, 20), np.uint8)
    data[:, 10] = 254  # a wall right across
    data[5, 5] = 254
    planner = plugins("breadth_first").BreadthFirst()
    planner.configure(Costmap(data, 0.5))
    for start, goal in (
        (pose(1.2, 1.2), pose(8.8, 1.2)),  # shut off by the wall
        (pose(2.75, 2.75), pose(1.2, 1.2)),  # the start is in a blocked cell
        (pose(1.2, 1.2), pose(2.75, 2.75)),  # so is the goal
        (pose(1.2, 1.2), pose(50.0, 1.2)),  # the goal is off the map
    ):
        assert planner.plan(start, goal).shape == (0, 3)


def observation(pose_, goal, path):
    """What a controller sees, with no sensors."""
    return Observation(0.0, 0.05, pose_, path, goal, (0.1, 0.1))


def test_turn_then_drive_stops_turns_or_drives_by_where_the_path_is(plugins):
    """No path or at the goal: stop. Facing away: turn. Facing the path: drive."""
    controller = plugins("turn_then_drive").TurnThenDrive()
    path = np.array([[x, 0.0, 0.0] for x in np.linspace(0.0, 4.0, 41)])
    goal = pose(4.0, 0.0)
    assert (
        controller.command(observation(pose(0, 0), goal, np.empty((0, 3)))) == Twist()
    )
    assert controller.command(observation(pose(3.95, 0.0), goal, path)) == Twist()
    left = controller.command(observation(pose(1.0, 0.0, -1.5), goal, path))
    assert left.vx == 0.0 and left.wz > 0
    right = controller.command(observation(pose(1.0, 0.0, 1.5), goal, path))
    assert right.vx == 0.0 and right.wz < 0
    drive = controller.command(observation(pose(1.0, 0.0, 0.05), goal, path))
    assert drive.vx > 0 and abs(drive.wz) < 0.2
    far = controller.command(observation(pose(3.9, 0.3, 0.0), goal, path[:5]))
    assert far.vx > 0 or far.wz != 0
