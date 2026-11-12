"""Tests for a run in which the planner finds no path."""

import logging
from pathlib import Path

from demuvo import Navigation, Simulation, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


def test_a_goal_inside_an_obstacle_leaves_the_robot_still_and_the_run_going(
    tmp_path, caplog
):
    """A* finds no path to a goal inside the box; the run goes on and nothing moves."""
    text = DEMO.read_text()
    text = text.replace("demuvo.planners:StraightLine", "demuvo.planners:AStar")
    text = text.replace("goal = [9.0, 5.0, 0.0]", "goal = [4.0, 4.5, 0.0]")
    path = tmp_path / "w.toml"
    path.write_text(text)
    world = load_world(path)
    sim = Simulation(world, dt=0.05)
    commands = Navigation(world)
    start = sim.poses["r1"].xyt
    with caplog.at_level(logging.WARNING):
        for _ in range(100):
            sim.step(commands(sim))
    commands.judge.update(sim)
    assert commands.paths["r1"].shape == (0, 3)
    assert tuple(sim.poses["r1"].xyt) == tuple(start)
    assert not commands.judge.done
    # the planner says why and the navigation says who, each once for the whole run
    assert len(caplog.messages) == 2
    assert "AStar: the goal cell (90, 80) is blocked (cost 254)" in caplog.messages[0]
    assert caplog.messages[1] == (
        "robot 'r1': no path for task 1, from (1.00, 1.00) to (4.00, 4.50); "
        "it stays still"
    )


def test_a_path_that_ends_short_of_the_goal_names_the_robot_and_task(tmp_path, caplog):
    """The planner moves a blocked goal to a free cell; the navigation says who."""
    text = DEMO.read_text()
    text = text.replace(
        'planner = "demuvo.planners:StraightLine"',
        'planner = "demuvo.planners:AStar"\nplanner_params = { tolerance = 1.0 }',
    )
    text = text.replace("goal = [9.0, 5.0, 0.0]", "goal = [4.0, 4.5, 0.0]")
    path = tmp_path / "w.toml"
    path.write_text(text)
    world = load_world(path)
    sim = Simulation(world, dt=0.05)
    commands = Navigation(world)
    with caplog.at_level(logging.WARNING):
        commands(sim)
    assert any(
        m.startswith("robot 'r1': the path for task 1 ends") for m in caplog.messages
    )
