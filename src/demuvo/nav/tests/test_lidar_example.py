"""Test for the lidar example world."""

from pathlib import Path

from demuvo import Navigation, Simulation, load_world
from demuvo.costmap import LETHAL

LIDAR = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "lidar.toml"


def test_the_robot_sees_the_box_with_two_lidars_and_reaches_its_goal():
    """It goes round the box with no collision and its local costmap marks the box."""
    world = load_world(LIDAR)
    sim = Simulation(world, dt=0.05)
    commands = Navigation(world)
    robot = sim.robots["r1"]
    seen, marked = set(), 0
    for _ in range(300):
        sim.step(commands(sim))
        seen |= set(robot.scans())
        if robot.local_costmap is not None:
            marked = max(marked, int((robot.local_costmap.data == LETHAL).sum()))
    commands.judge.update(sim)
    assert not sim.collisions
    assert set(commands.judge.done) == {("r1", 0)}
    assert seen == {"front", "rear"}
    assert marked > 50  # the box and the circle showed up as obstacles
