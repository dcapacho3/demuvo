"""Tests for writing a world back to a file."""

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from demuvo import load_world
from demuvo.world import RobotSpec, save_world
from demuvo.world.save import world_text

from .small_world import small_world
from .test_map_world import load as load_with_map
from .test_map_world import write_map

EXAMPLES = sorted(
    (Path(__file__).resolve().parents[4] / "examples" / "worlds").glob("*.toml")
)

# Every optional key, every shape and every kind of value a world file takes.
RICH = """
version = 1

[world]
size = [8.0, 8.0]
origin = [-1.0, -2.0]
seed = 7
dt = 0.02

[[obstacle]]
pose = [1.0, 1.0, 0.5]
shape = { name = "box", size = [1.0, 0.4] }

[[obstacle]]
pose = [3.0, 1.0, 0.0]
shape = { name = "circle", radius = 0.3 }

[[obstacle]]
shape = { name = "polygon", vertices = [[4.0, 3.0], [5.0, 3.0], [4.5, 4.0]] }

[[obstacle]]
shape = { name = "linestring", vertices = [[0.0, 4.0], [2.0, 4.5], [2.0, 5.0]] }

[[robot]]
name = "a"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 3.0, 0.7]
params = { max_v = 0.8, max_w = 2.0 }
planner = "demuvo.planners:AStar"
planner_params = { tolerance = 0.2 }
smoother = "demuvo.smoothers.line_of_sight:LineOfSight"
controller = "demuvo.controllers:PurePursuit"
controller_params = { speed = 0.5, lookahead = 0.6 }
control_period = 0.1
estimator = "dead_reckoning"
control_pose = "truth"
sensors = { front = { type = "lidar", beams = 16, mount = [0.1, 0.0, 0.0] } }

[[robot]]
name = 'q"uote\\é'
model = "diff_drive"
shape = { name = "box", size = [0.4, 0.3] }
pose = [6.0, 3.0, -2.0]
controller = "demuvo.controllers:PurePursuit"
controller_params = { "odd key" = 1, flag = true, label = "x" }
estimator = "dead_reckoning"
sensors = { encoders = { noise = 0.01 } }

[[task]]
robot = "a"
goal = [5.0, 5.0, 1.0]
goal_tolerance = [0.2, 0.3]
stop = false
wait = 1.5

[[task]]
robot = "a"
goal = [2.0, 2.0, 0.0]

[sensors]
encoders = { noise = 0.02, ticks_per_rev = 2048.0 }

[costmap.global]
resolution = 0.1
plugins = ["inflation"]
inflation = { inflation_radius = 0.7, cost_scaling_factor = 8.0 }

[costmap.local]
window = 2.0
plugins = ["obstacle"]
obstacle = { obstacle_max_range = 2.0, raytrace_max_range = 2.5 }
"""


def same(a, b):
    """Two worlds are equal in everything but the last bits of their poses."""
    assert (a.size, a.seed, a.dt, a.origin, a.sensors) == (
        b.size,
        b.seed,
        b.dt,
        b.origin,
        b.sensors,
    )
    assert (a.costmap, a.local_costmap) == (b.costmap, b.local_costmap)
    assert len(a.obstacles) == len(b.obstacles)
    for x, y in zip(a.obstacles, b.obstacles):
        assert x.shape == y.shape
        assert x.pose.isclose(y.pose, 1e-9)
    assert len(a.robots) == len(b.robots)
    for x, y in zip(a.robots, b.robots):
        assert replace(x, pose=y.pose) == y
        assert x.pose.isclose(y.pose, 1e-9)
    assert len(a.tasks) == len(b.tasks)
    for x, y in zip(a.tasks, b.tasks):
        assert replace(x, goal=y.goal) == y
        assert x.goal.isclose(y.goal, 1e-9)


def round_trip(world, folder):
    """Save a world, load it again without a warning, and check the text is stable."""
    path = folder / "saved.toml"
    save_world(world, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = load_world(path)
    same(world, again)
    assert world_text(again, folder) == path.read_text(encoding="utf-8")
    return again


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_every_example_world_saves_and_loads_again(example, tmp_path):
    """Each file of the examples folder comes back as the same world."""
    round_trip(load_world(example), tmp_path)


def test_a_world_with_every_key_saves_and_loads_again(tmp_path):
    """All shapes, all robot keys, odd characters in names and keys, both costmaps."""
    source = tmp_path / "rich.toml"
    source.write_text(RICH, encoding="utf-8")
    world = load_world(source)
    again = round_trip(world, tmp_path)
    assert again.robots[1].name == 'q"uote\\é'
    assert again.robots[1].controller_params["odd key"] == 1


def test_a_local_costmap_with_both_layers_saves_and_loads_again(tmp_path):
    """The inflation layer of the local costmap keeps its own settings."""
    extra = "[costmap.local]\ninflation = { inflation_radius = 0.4 }\n"
    world = small_world(tmp_path, extra)
    assert round_trip(world, tmp_path).local_costmap.inflation_radius == 0.4


def test_keys_left_at_their_default_are_not_written(tmp_path):
    """A small world gives a short file."""
    demo = next(p for p in EXAMPLES if p.name == "demo.toml")
    text = world_text(load_world(demo), tmp_path)
    for key in (
        "dt =",
        "origin =",
        "control_period =",
        "estimator =",
        "stop =",
        "[costmap",
    ):
        assert key not in text


def test_a_map_file_is_written_as_a_path_from_the_saved_file(tmp_path):
    """The map stays found when the file moves to another folder next to its map."""
    write_map(tmp_path, [[255] * 5] * 4)
    world = load_with_map(tmp_path, costmap='source = "m.yaml"\nunknown = "blocked"')
    (tmp_path / "more").mkdir()
    save_world(world, tmp_path / "more" / "w.toml")
    assert 'source = "../m.yaml"' in (tmp_path / "more" / "w.toml").read_text()
    round_trip(world, tmp_path)


def test_a_value_the_format_cannot_hold_is_refused(tmp_path):
    """A set in a plugin setting has no TOML form, and the error says so."""
    world = load_world(EXAMPLES[0])
    robot = world.robots[0]
    odd = replace(
        robot, controller="demuvo.controllers:PurePursuit", controller_params={"x": {1}}
    )
    with pytest.raises(TypeError, match="cannot write a set"):
        save_world(replace(world, robots=(odd, *world.robots[1:])), tmp_path / "w.toml")
    assert isinstance(odd, RobotSpec)
