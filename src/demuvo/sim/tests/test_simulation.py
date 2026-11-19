"""Tests for the simulation loop."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from demuvo import Simulation, Twist, edit, load_world

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

    sim.models["r2"] = SimpleNamespace(step=fail, clip=lambda command: command)
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


def test_velocity_is_correct_past_the_scalar_limit():
    """Five robots take the batch route and each gets its own body velocity."""
    world = edit.new_world(size=(14.0, 8.0))
    for x in (2.0, 4.0, 6.0, 8.0):
        world = edit.add_robot(world, f"r{len(world.robots) + 1}", (x, 2.0, 0.0))
    sim = Simulation(world, dt=0.1)
    assert len(sim.robots) == 5
    sim.step({name: Twist(vx=0.5) for name in sim.models})
    for name in sim.models:
        assert sim.velocity[name] == pytest.approx([0.5, 0.0, 0.0])


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


def test_an_ackermann_robot_arcs_and_cannot_spin(world):
    """An ackermann robot arcs under its clipped command and cannot spin in place."""
    sim = Simulation(edit.set_model(world, "r1", "ackermann"), dt=0.1)
    for _ in range(10):
        sim.step({"r1": Twist(wz=1.0)})
    assert tuple(sim.poses["r1"].xyt) == pytest.approx((1.0, 1.0, 0.0))
    for _ in range(10):
        sim.step({"r1": Twist(vx=0.5, wz=0.2)})
    x, y, angle = sim.poses["r1"].xyt
    assert (x, y) != pytest.approx((1.0, 1.0))
    assert angle == pytest.approx(0.2)


def test_a_mecanum_robot_strafes_and_reports_its_sideways_velocity(world):
    """Vy alone moves it to its left; velocity and driven see the sideways motion."""
    sim = Simulation(edit.set_model(world, "r1", "mecanum"), dt=0.1)
    for _ in range(10):
        sim.step({"r1": Twist(vy=0.5)})
    assert tuple(sim.poses["r1"].xyt) == pytest.approx((1.0, 1.5, 0.0))
    assert sim.velocity["r1"] == pytest.approx([0.0, 0.5, 0.0])
    assert sim.driven["r1"] == pytest.approx(0.5)


def test_a_mecanum_robot_spins_in_place_and_drives_nothing(world):
    """It turns about its centre, so the position stays and driven stays 0."""
    sim = Simulation(edit.set_model(world, "r1", "mecanum"), dt=0.1)
    for _ in range(10):
        sim.step({"r1": Twist(wz=1.0)})
    assert tuple(sim.poses["r1"].xyt) == pytest.approx((1.0, 1.0, 1.0))
    assert sim.driven["r1"] == pytest.approx(0.0, abs=1e-12)


def test_a_mecanum_robot_dead_reckons_on_its_four_wheels(world):
    """Noiseless encoders on four wheels give back the true pose."""
    world = edit.set_estimator(
        edit.set_model(world, "r1", "mecanum"), "r1", "dead_reckoning"
    )
    sim = Simulation(world, dt=0.05)
    for _ in range(40):
        sim.step({"r1": Twist(vx=0.4, vy=0.3, wz=0.5)})
    truth, estimate = sim.poses["r1"].xyt, sim.estimates["r1"].pose.xyt
    assert tuple(estimate) == pytest.approx(tuple(truth), abs=1e-12)
    assert tuple(truth) != pytest.approx((1.0, 1.0, 0.0))


def with_params(world, **params):
    """The world with r1's model params replaced."""
    import dataclasses  # pylint: disable=import-outside-toplevel

    robots = tuple(
        dataclasses.replace(r, params=params) if r.name == "r1" else r
        for r in world.robots
    )
    return dataclasses.replace(world, robots=robots)


def test_a_turn_of_half_a_circle_per_step_is_refused(world):
    """Past pi the velocity read back from a step aliases, so the world is refused.

    The message names the robot, the turn it may make and the way out.
    """
    fast = with_params(world, max_w=35.0)
    with pytest.raises(ValueError, match=r"'r1' may turn 3\.5 rad.*half a circle"):
        Simulation(fast, dt=0.1)
    assert Simulation(fast, dt=0.01).dt == 0.01  # the same world, 0.35 rad a step


def test_the_turn_per_step_bound_is_strict_at_pi(world):
    """Exactly pi is refused, the sign of the read-back turn is ambiguous there."""
    import math  # pylint: disable=import-outside-toplevel

    with pytest.raises(ValueError, match="half a circle"):
        Simulation(with_params(world, max_w=math.pi), dt=1.0)
    Simulation(with_params(world, max_w=3.14), dt=1.0)


def test_a_car_turn_per_step_is_bounded_through_its_steering(world):
    """An ackermann robot's largest turn rate is slope * max_v, not a flat cap."""
    car = edit.set_model(world, "r1", "ackermann")
    # tan(1.5) / 0.1 * 3.0 = 423 rad/s at full speed: 4.2 rad in one 0.01 s step.
    fast = with_params(car, max_v=3.0, max_steer=1.5, wheelbase=0.1)
    with pytest.raises(ValueError, match="'r1' may turn"):
        Simulation(fast, dt=0.01)
    assert Simulation(fast, dt=0.001).steps == 0  # a tenth of the step is fine
