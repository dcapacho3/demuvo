"""Tests for the simulation loop."""

from pathlib import Path

import pytest

from demuvo import Simulation, load_world

DEMO = Path(__file__).resolve().parents[1] / "examples" / "worlds" / "demo.toml"


@pytest.fixture(name="world")
def demo_world():
    """The example world: one robot, r1, at (1, 1) heading +x."""
    return load_world(DEMO)


def test_robots_start_at_their_pose(world):
    """Before any step, each robot is at its start pose and the time is zero."""
    sim = Simulation(world)
    assert sim.poses == {"r1": (1.0, 1.0, 0.0)}
    assert sim.time == 0.0


def test_a_robot_drives_straight(world):
    """One second at 1 m/s moves the robot 1 m along +x."""
    sim = Simulation(world, dt=0.1)
    for _ in range(10):
        sim.step({"r1": (1.0, 0.0)})
    assert sim.poses["r1"] == pytest.approx((2.0, 1.0, 0.0))
    assert sim.steps == 10
    assert sim.time == pytest.approx(1.0)


def test_a_robot_with_no_command_stops(world):
    """A robot missing from the commands does not move."""
    sim = Simulation(world)
    sim.step({})
    assert sim.poses["r1"] == (1.0, 1.0, 0.0)
    assert sim.steps == 1


def test_an_unknown_robot_name_is_refused(world):
    """A command for a robot that does not exist is an error, not ignored."""
    sim = Simulation(world)
    with pytest.raises(ValueError, match="r2"):
        sim.step({"r2": (1.0, 0.0)})
    assert sim.steps == 0


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan")])
def test_the_step_length_must_be_positive(world, dt):
    """Zero, negative and nan step lengths are refused."""
    with pytest.raises(ValueError, match="dt"):
        Simulation(world, dt=dt)


def test_the_same_seed_gives_the_same_run(world):
    """Two simulations of one world draw the same numbers and move the same."""
    first, second = Simulation(world), Simulation(world)
    for sim in (first, second):
        for _ in range(20):
            sim.step({"r1": (0.5, 0.3)})
    assert first.poses == second.poses
    assert first.rng.random() == second.rng.random()


def test_a_robot_overrides_the_speed_limit(tmp_path):
    """`params` in the file replaces the model default: 0.5 m/s, not 1 m/s."""
    text = DEMO.read_text().replace(
        'controller = "demuvo.controllers:GoToGoal"',
        'controller = "demuvo.controllers:GoToGoal"\nparams = { max_v = 0.5 }',
    )
    path = tmp_path / "slow.toml"
    path.write_text(text)
    sim = Simulation(load_world(path), dt=1.0)
    sim.step({"r1": (3.0, 0.0)})
    assert sim.poses["r1"] == pytest.approx((1.5, 1.0, 0.0))
