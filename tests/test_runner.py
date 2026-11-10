"""Tests for a run: the world in motion that a window or a benchmark steps."""

from pathlib import Path

import numpy as np
import pytest

from demuvo import Navigation, Run, Simulation, edit, load_world
from demuvo.world import WorldError

WORLDS = Path(__file__).resolve().parents[1] / "examples" / "worlds"
DEMO = WORLDS / "demo.toml"
LIDAR = WORLDS / "lidar.toml"


def test_a_run_steps_as_the_simulation_and_navigation_do():
    """The same poses as stepping the two by hand."""
    run = Run(DEMO)
    run.step(300)
    world = load_world(DEMO)
    sim, nav = Simulation(world), Navigation(world)
    for _ in range(300):
        sim.step(nav(sim))
    assert run.time == sim.time
    assert np.array_equal(run.poses["r1"], sim.poses["r1"].xyt)


def test_a_run_takes_a_world_made_in_memory():
    """A world from the edit functions runs without a file."""
    world = edit.set_goal(edit.new_world(), "r1", (8.0, 3.0, 0.0))
    run = Run(world)
    run.step(100)
    assert run.poses["r1"][0] > 5.0


def test_a_step_length_of_its_own_replaces_the_worlds():
    """A step of 0.025 s, a whole number of control periods."""
    assert Run(DEMO, dt=0.025).sim.dt == 0.025


def test_reset_starts_again_and_the_run_repeats():
    """Time returns to zero, and the second run gives the same poses."""
    run = Run(DEMO)
    run.step(200)
    first = run.poses["r1"]
    run.reset()
    assert run.time == 0.0
    run.step(200)
    assert np.array_equal(run.poses["r1"], first)


def test_set_world_runs_the_new_world_from_time_zero():
    """After an edit the new robot is there and the clock is at zero."""
    run = Run(edit.new_world())
    run.step(10)
    run.set_world(edit.add_robot(run.world, "r2", (1.0, 1.0, 0.0)))
    assert run.time == 0.0
    assert set(run.poses) == {"r1", "r2"}


def test_advance_covers_the_time_asked():
    """One second is a hundred steps of 0.01 s, and at least one step is taken."""
    run = Run(DEMO)
    assert run.advance(1.0) == 100
    assert run.time == pytest.approx(1.0)
    assert run.advance(0.0) == 1


def test_advance_stops_when_the_wall_clock_budget_is_spent():
    """A budget of zero allows the one step that is always taken."""
    run = Run(DEMO)
    assert run.advance(1.0, budget=0.0) == 1
    assert run.time == pytest.approx(0.01)


def test_a_run_knows_when_every_task_is_done():
    """The demo robot reaches its goal at 19.26 s."""
    run = Run(DEMO)
    assert not run.finished
    assert run.until_done(60.0)
    assert 19.0 < run.time < 20.0
    assert any("reached at" in line for line in run.result())


def test_until_done_gives_up_at_the_limit():
    """Too little time says False and stops on time."""
    run = Run(DEMO)
    assert not run.until_done(2.0)
    assert run.time == pytest.approx(2.0)
    assert any("not reached" in line for line in run.result())


def test_a_run_with_no_tasks_is_never_finished():
    """Nothing was asked of the robots."""
    assert not Run(edit.new_world()).finished


def test_the_scans_of_a_lidar_robot_are_placed_in_the_world():
    """Two lidars of 90 beams each, with their origin at the robot's pose."""
    run = Run(LIDAR)
    assert not run.scans("r1")
    run.step(30)
    scans = run.scans("r1")
    front, rear = scans[0], scans[1]
    assert len(front.angles) == len(front.ranges) == 90
    assert front.max_range == 4.0
    x, y, yaw = run.poses["r1"]
    assert front.origin == pytest.approx(
        (x + 0.15 * np.cos(yaw), y + 0.15 * np.sin(yaw))
    )
    assert rear.origin != front.origin


def test_a_robot_without_a_lidar_has_no_scans_and_no_local_costmap():
    """The demo has neither."""
    run = Run(DEMO)
    run.step(20)
    assert not run.scans("r1")
    assert run.local_costmap("r1") is None


def test_the_local_costmap_appears_after_the_first_update():
    """The lidar robot builds one from its scans."""
    run = Run(LIDAR)
    run.step(60)
    assert run.local_costmap("r1") is not None


def test_a_robot_has_a_path_once_its_planner_has_run():
    """None before the first step, rows of (x, y, angle) after."""
    run = Run(DEMO)
    assert run.path("r1") is None
    run.step(5)
    assert run.path("r1").shape[1] == 3


@pytest.mark.parametrize("method", ["scans", "local_costmap", "path"])
def test_an_unknown_robot_is_refused(method):
    """The same plain message for every question about a robot."""
    with pytest.raises(WorldError, match="no robot named 'x'"):
        getattr(Run(DEMO), method)("x")


def test_a_run_in_this_process_has_no_news_to_take_and_nothing_to_close():
    """`poll` is False and `close` does nothing, so a window can call both."""
    run = Run(DEMO)
    assert run.poll() is False
    assert run.close() is None
