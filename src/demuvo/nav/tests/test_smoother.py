"""Tests for the smoother step of the navigation."""

from pathlib import Path

import numpy as np
import pytest

from demuvo import Navigation, Simulation, WorldError, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"
PLANNER = 'planner = "demuvo.planners:StraightLine"'

MODULE = """
import numpy as np
from demuvo import Twist
costmaps = []
calls = []
class EveryOther:
    def configure(self, costmap):
        costmaps.append(costmap)
    def smooth(self, path):
        calls.append(len(path))
        return path[::2]
class Idle:
    def command(self, obs):
        return Twist()
class NoPath:
    def configure(self, costmap):
        pass
    def plan(self, start, goal):
        return np.empty((0, 3))
"""


def world_with(tmp_path, monkeypatch, smoother=None, planner=None):
    """The demo world with a smoother, and a planner of the user's, if given."""
    (tmp_path / "my_smooth.py").write_text(MODULE)
    monkeypatch.syspath_prepend(tmp_path)
    import my_smooth  # pylint: disable=import-error,import-outside-toplevel

    my_smooth.costmaps.clear()  # the module stays loaded between tests
    my_smooth.calls.clear()
    smoother_line = f'smoother = "{smoother}"\n' if smoother else ""
    planner_line = f'planner = "{planner}"' if planner else PLANNER
    text = DEMO.read_text().replace(PLANNER, smoother_line + planner_line, 1)
    if planner:  # PurePursuit refuses an empty path, so a controller of the user's
        text = text.replace("demuvo.controllers:PurePursuit", "my_smooth:Idle")
        text = text.replace("controller_params = { speed = 0.5, lookahead = 0.6 }", "")
    path = tmp_path / "w.toml"
    path.write_text(text)
    return my_smooth, load_world(path)


def plan_once(world):
    """Run the navigation one step and return its path for r1."""
    sim = Simulation(world)
    commands = Navigation(world)
    commands(sim)
    return commands, sim


def test_a_robot_with_no_smoother_keeps_the_planners_path(tmp_path, monkeypatch):
    """Without the key the path is the planner's, point for point."""
    _, world = world_with(tmp_path, monkeypatch)
    commands, _ = plan_once(world)
    assert len(commands.paths["r1"]) > 100


def test_the_smoother_improves_the_path_after_the_planner(tmp_path, monkeypatch):
    """The path the controller gets is the smoother's, made from the planner's."""
    _, world = world_with(tmp_path, monkeypatch)
    plain, _ = plan_once(world)
    module, world = world_with(tmp_path, monkeypatch, "my_smooth:EveryOther")
    commands, sim = plan_once(world)
    assert module.calls == [len(plain.paths["r1"])]
    assert len(commands.paths["r1"]) == len(plain.paths["r1"][::2])
    assert module.costmaps == [sim.costmaps["r1"]]


def test_the_smoother_is_configured_once(tmp_path, monkeypatch):
    """A second path from the same robot does not configure it again."""
    module, world = world_with(tmp_path, monkeypatch, "my_smooth:EveryOther")
    sim = Simulation(world)
    commands = Navigation(world)
    commands(sim)
    commands.set_goal("r1", world.tasks[0].goal)
    for _ in range(5):  # the controller runs every 5 steps
        sim.step({})
    commands(sim)
    assert len(module.calls) == 2
    assert len(module.costmaps) == 1


def test_an_empty_path_is_not_smoothed(tmp_path, monkeypatch):
    """A planner that finds no path leaves the smoother out."""
    module, world = world_with(
        tmp_path, monkeypatch, "my_smooth:EveryOther", planner="my_smooth:NoPath"
    )
    commands, _ = plan_once(world)
    assert module.calls == []
    assert commands.paths["r1"].shape == (0, 3)


def test_a_path_from_outside_is_not_smoothed(tmp_path, monkeypatch):
    """`set_path` gives a finished path, so neither planner nor smoother runs."""
    module, world = world_with(tmp_path, monkeypatch, "my_smooth:EveryOther")
    sim = Simulation(world)
    commands = Navigation(world)
    commands.set_path("r1", np.zeros((9, 3)))
    commands(sim)
    assert module.calls == []
    assert len(commands.paths["r1"]) == 9


def test_a_class_with_no_smooth_method_is_refused_at_load(tmp_path, monkeypatch):
    """A planner named as a smoother fails before the first step, by name."""
    _, world = world_with(tmp_path, monkeypatch, "demuvo.planners:StraightLine")
    with pytest.raises(WorldError, match="robot 'r1': smoother .* has no 'smooth'"):
        Navigation(world)
