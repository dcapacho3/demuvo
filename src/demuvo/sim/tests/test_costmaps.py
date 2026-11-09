"""Tests for the costmap each robot holds in the simulation."""

from pathlib import Path

from demuvo import Simulation, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


def test_the_estimate_carries_the_costmap():
    """Before and after a step, the robot's estimate holds its costmap."""
    sim = Simulation(load_world(DEMO))
    assert sim.costmaps["r1"] is not None
    assert sim.estimates["r1"].map is sim.costmaps["r1"]
    sim.step({})
    assert sim.estimates["r1"].map is sim.costmaps["r1"]


def test_the_costmap_can_be_switched_off(tmp_path):
    """`enabled = false` gives no costmap, and the estimate's map is None."""
    path = tmp_path / "w.toml"
    path.write_text(DEMO.read_text() + "\n[costmap.global]\nenabled = false\n")
    sim = Simulation(load_world(path))
    assert sim.costmaps == {"r1": None}
    sim.step({})
    assert sim.estimates["r1"].map is None
