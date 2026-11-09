"""Tests for a run in which the planner finds no path."""

from pathlib import Path

from demuvo import Navigation, Simulation, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


def test_a_goal_inside_an_obstacle_leaves_the_robot_still_and_the_run_going(tmp_path):
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
    for _ in range(100):
        sim.step(commands(sim))
    commands.judge.update(sim)
    assert commands.paths["r1"].shape == (0, 3)
    assert tuple(sim.poses["r1"].xyt) == tuple(start)
    assert not commands.judge.done
