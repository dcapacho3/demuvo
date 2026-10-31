"""Tests for the simulation loop."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from demuvo import Simulation, Twist, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


@pytest.fixture(name="world")
def demo_world():
    """The example world: one robot, r1, at (1, 1) heading +x."""
    return load_world(DEMO)


def test_robots_start_at_their_pose(world):
    """Before any step, each robot is at its start pose and the time is zero."""
    sim = Simulation(world)
    assert list(sim.poses) == ["r1"]
    assert tuple(sim.poses["r1"].xyt) == (1.0, 1.0, 0.0)
    assert sim.time == 0.0


def test_a_robot_drives_straight(world):
    """One second at 1 m/s moves the robot 1 m along +x."""
    sim = Simulation(world, dt=0.1)
    for _ in range(10):
        sim.step({"r1": Twist(vx=1.0, wz=0.0)})
    assert tuple(sim.poses["r1"].xyt) == pytest.approx((2.0, 1.0, 0.0))
    assert sim.steps == 10
    assert sim.time == pytest.approx(1.0)


def test_a_robot_with_no_command_stops(world):
    """A robot missing from the commands does not move."""
    sim = Simulation(world)
    sim.step({})
    assert tuple(sim.poses["r1"].xyt) == (1.0, 1.0, 0.0)
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
            sim.step({"r1": Twist(vx=0.5, wz=0.3)})
    assert first.poses == second.poses
    assert first.rng.random() == second.rng.random()


def test_a_robot_overrides_the_speed_limit(tmp_path):
    """`params` in the file replaces the model default: 0.5 m/s, not 1 m/s."""
    text = DEMO.read_text().replace(
        "controller_params = { speed = 0.5, lookahead = 0.6 }",
        "controller_params = { speed = 0.5, lookahead = 0.6 }\n"
        "params = { max_v = 0.5 }",
    )
    path = tmp_path / "slow.toml"
    path.write_text(text)
    sim = Simulation(load_world(path), dt=1.0)
    sim.step({"r1": Twist(vx=3.0, wz=0.0)})
    assert tuple(sim.poses["r1"].xyt) == pytest.approx((1.5, 1.0, 0.0))


def test_a_robot_that_fails_leaves_the_world_as_it_was(tmp_path):
    """Robot 1 does not move if the model of robot 2 raises."""
    text = (
        DEMO.read_text()
        + """
[[robot]]
name = "r2"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [3.0, 3.0, 0.0]
"""
    )
    path = tmp_path / "two.toml"
    path.write_text(text)
    sim = Simulation(load_world(path))

    def fail(*_):
        raise ValueError("broken")

    sim.models["r2"] = SimpleNamespace(step=fail)
    before = dict(sim.poses)
    with pytest.raises(ValueError, match="broken"):
        sim.step({"r1": Twist(vx=1.0)})
    assert sim.poses == before
    assert sim.steps == 0


def test_dt_comes_from_the_world_unless_given(tmp_path):
    """The world's dt is the default; an argument replaces it."""
    path = tmp_path / "w.toml"
    path.write_text(DEMO.read_text().replace("dt = 0.01", "dt = 0.02"))
    world = load_world(path)
    assert Simulation(world).dt == 0.02
    assert Simulation(world, dt=0.1).dt == 0.1


def test_velocity_is_the_motion_of_the_last_step(world):
    """Zero at the start, the command after a step, zero again for no command."""
    sim = Simulation(world, dt=0.1)
    assert sim.velocity["r1"] == pytest.approx([0.0, 0.0, 0.0])
    sim.step({"r1": Twist(vx=0.5, wz=0.2)})
    assert sim.velocity["r1"] == pytest.approx([0.5, 0.0, 0.2])
    sim.step({})
    assert sim.velocity["r1"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_driven_is_the_exact_length_of_the_path(world):
    """A circle of radius 0.5 m for 6 s at 0.5 m/s is 3 m, not the shorter chords."""
    sim = Simulation(world, dt=0.1)
    for _ in range(60):
        sim.step({"r1": Twist(vx=0.5, wz=1.0)})
    assert sim.driven["r1"] == pytest.approx(3.0, abs=1e-9)


def test_turning_on_the_spot_and_standing_still_drive_nothing(world):
    """Only motion of the robot's origin counts."""
    sim = Simulation(world, dt=0.1)
    assert sim.driven["r1"] == 0.0
    for _ in range(10):
        sim.step({"r1": Twist(wz=1.0)})
    assert sim.driven["r1"] == pytest.approx(0.0, abs=1e-12)


def test_a_failed_step_does_not_change_driven(world):
    """The step is refused before anything is added."""
    sim = Simulation(world, dt=0.1)
    sim.step({"r1": Twist(vx=1.0)})
    before = dict(sim.driven)
    with pytest.raises(ValueError, match="no robot named"):
        sim.step({"ghost": Twist(vx=1.0)})
    assert sim.driven == before
