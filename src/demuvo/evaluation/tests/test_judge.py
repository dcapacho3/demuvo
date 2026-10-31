"""Tests for the judge: which task a robot is on and when it is done."""

from pathlib import Path

import pytest
from robomath import Transform2D

from demuvo import Simulation, Task, Twist, load_world
from demuvo.evaluation import Judge

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


def goal(x):
    """A task for r1 at (x, 1) facing +x, with the default tolerance."""
    return Task("r1", Transform2D.from_xyt((x, 1.0, 0.0)))


def test_a_robot_that_nothing_drives_is_judged_from_its_motion_alone():
    """Hand-driven to the goal: done only once it stands still, at that time."""
    sim = Simulation(load_world(DEMO), dt=0.1)
    judge = Judge({"r1": [goal(2.0)]})
    for _ in range(10):
        sim.step({"r1": Twist(vx=1.0)})
        judge.update(sim)
    assert judge.done == {}  # at the goal, but still moving
    sim.step({})
    judge.update(sim)
    assert judge.done == pytest.approx({("r1", 0): 1.1})


def test_a_waypoint_is_passed_at_speed_unless_it_asks_to_stop():
    """A middle task needs no stop by default; with `stop` it does."""
    for stop, expected in ((None, 1), (True, 0)):
        sim = Simulation(load_world(DEMO), dt=0.1)
        middle = Task("r1", Transform2D.from_xyt((1.5, 1.0, 0.0)), stop=stop)
        judge = Judge({"r1": [middle, goal(5.0)]})
        for _ in range(5):
            sim.step({"r1": Twist(vx=1.0)})
            judge.update(sim)
        assert judge.current["r1"] == expected


def test_a_wait_must_be_held_without_a_break():
    """Leaving the goal during the wait starts the count again."""
    sim = Simulation(load_world(DEMO), dt=0.1)
    task = Task("r1", Transform2D.from_xyt((1.0, 1.0, 0.0)), wait=0.3)
    judge = Judge({"r1": [task, goal(5.0)]})
    for command in [Twist(), Twist(), Twist(vx=0.5)] + [Twist()] * 5:
        sim.step({"r1": command})
        judge.update(sim)
    assert judge.current["r1"] == 1
    assert judge.done[("r1", 0)] == pytest.approx(0.4)  # not 0.1, the first stop


def test_update_twice_in_one_step_changes_nothing():
    """Two callers may both call `update` in the same step."""
    sim = Simulation(load_world(DEMO), dt=0.1)
    judge = Judge({"r1": [goal(1.0)]})
    sim.step({})
    judge.update(sim)
    first = dict(judge.done), dict(judge.current)
    judge.update(sim)
    assert (judge.done, judge.current) == first


def test_reset_forgets_a_robot_and_a_robot_with_no_task_is_ignored():
    """After `reset` the robot starts again at task 0 with nothing done."""
    sim = Simulation(load_world(DEMO), dt=0.1)
    judge = Judge({"r1": [goal(1.0)]})
    sim.step({})
    judge.update(sim)
    assert judge.done
    judge.reset("r1")
    assert (judge.done, judge.current["r1"]) == ({}, 0)
    empty = Judge({"r1": []})
    empty.update(sim)
    assert empty.task("r1") is None
    assert empty.done == {}


def test_sliding_sideways_is_not_standing_still():
    """A robot that moves along y at the goal, as a mecanum robot can, is not done."""
    sim = Simulation(load_world(DEMO), dt=0.1)
    judge = Judge({"r1": [goal(1.0)]})
    sim.velocity["r1"][:] = [0.0, 0.5, 0.0]
    judge.update(sim)
    assert judge.done == {}
    sim.velocity["r1"][:] = [0.0, 0.0, 0.0]
    judge.update(sim)
    assert judge.done == {("r1", 0): 0.0}
