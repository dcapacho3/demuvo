"""Tests for the result lines of a run."""

from pathlib import Path

from robomath import Transform2D

from demuvo import Simulation, Task, load_world
from demuvo.evaluation import Judge, goal_error, result_lines

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


def goal(x):
    """A task for r1 at (x, 1) facing +x, with the default tolerance."""
    return Task("r1", Transform2D.from_xyt((x, 1.0, 0.0)))


def judged(*tasks):
    """The demo robot, which stays at (1, 1), after one step with a judge."""
    sim = Simulation(load_world(DEMO), dt=0.1)
    judge = Judge({"r1": list(tasks)})
    sim.step({})
    judge.update(sim)
    return sim, judge


def test_goal_error_is_the_distance_and_the_angle_left():
    """The robot at (1, 1, 0) is 2 m and 1 rad from (3, 1, 1)."""
    pose = Transform2D.from_xyt((1.0, 1.0, 0.0))
    task = Task("r1", Transform2D.from_xyt((3.0, 1.0, 1.0)))
    assert goal_error(pose, task) == (2.0, 1.0)


def test_a_reached_task_says_when():
    """Done at the first step, and the robot drove nothing."""
    sim, judge = judged(goal(1.0))
    assert result_lines(sim, judge) == [
        "r1 task 1: reached at 0.10 s",
        "r1 drove 0.00 m",
    ]


def test_the_task_in_progress_says_how_far_the_robot_is():
    """The task it is on shows the distance and angle left; later ones do not."""
    sim, judge = judged(goal(3.0), goal(5.0))
    assert result_lines(sim, judge) == [
        "r1 task 1: not reached, 2.00 m and 0.00 rad away",
        "r1 task 2: not reached",
        "r1 drove 0.00 m",
    ]


def test_a_task_done_earlier_stays_reached_when_the_robot_moves_on():
    """The robot starts on the first goal, so it passes it and works on the second."""
    sim, judge = judged(goal(1.0), goal(5.0))
    assert result_lines(sim, judge) == [
        "r1 task 1: reached at 0.10 s",
        "r1 task 2: not reached, 4.00 m and 0.00 rad away",
        "r1 drove 0.00 m",
    ]


def test_a_robot_with_no_task_gives_no_lines():
    """Nothing to report."""
    sim, judge = judged()
    assert not result_lines(sim, judge)
