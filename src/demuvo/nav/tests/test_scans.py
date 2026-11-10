"""Tests for the scans a controller is given."""

from pathlib import Path

import numpy as np
import pytest

from demuvo import Navigation, Simulation, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"
PLANNER = 'planner = "demuvo.planners:StraightLine"'

MODULE = """
from demuvo import Twist
seen = []
class Spy:
    def __init__(self, speed=0.5, lookahead=0.6):
        pass
    def command(self, obs):
        seen.append(obs)
        return Twist()
"""


def spy_world(tmp_path, monkeypatch, sensors=""):
    """The demo world with a controller that records what it is given."""
    (tmp_path / "my_scans.py").write_text(MODULE)
    monkeypatch.syspath_prepend(tmp_path)
    import my_scans  # pylint: disable=import-error,import-outside-toplevel

    my_scans.seen.clear()  # the module stays loaded between tests
    text = DEMO.read_text().replace("demuvo.controllers:PurePursuit", "my_scans:Spy")
    text = text.replace(PLANNER, f"{sensors}\n{PLANNER}", 1)
    path = tmp_path / "w.toml"
    path.write_text(text)
    return my_scans, load_world(path)


def test_a_controller_with_no_lidar_gets_an_empty_mapping(tmp_path, monkeypatch):
    """`scans` is there, and empty."""
    module, world = spy_world(tmp_path, monkeypatch)
    sim, commands = Simulation(world), Navigation(world)
    sim.step(commands(sim))
    assert module.seen[0].scans == {}


def test_a_controller_is_given_the_latest_scan_of_its_robot(tmp_path, monkeypatch):
    """After a step the lidar has scanned; the next control run sees it."""
    module, world = spy_world(
        tmp_path, monkeypatch, "sensors = { lidar = { beams = 8, max_range = 20 } }"
    )
    sim = Simulation(world, dt=0.05)
    commands = Navigation(world)
    for _ in range(3):
        sim.step(commands(sim))
    first, last = module.seen[0], module.seen[-1]
    assert first.scans == {}  # no step had been taken yet
    scan = last.scans["lidar"]
    assert len(scan.ranges) == 8
    assert scan.ranges == pytest.approx(sim.robots["r1"].readings["lidar"].data)
    assert np.isfinite(scan.ranges).any()
