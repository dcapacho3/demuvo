"""Tests for the key numbers of a run and their CSV file."""

import csv
from datetime import datetime
from pathlib import Path

import pytest
from robomath import Transform2D

from demuvo import Simulation, Task, load_world
from demuvo.evaluation import Judge, metrics_path, metrics_rows, write_metrics
from demuvo.evaluation import metrics as module
from demuvo.sim.collisions import Collision

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


def test_a_reached_task_fills_the_success_columns():
    """Done at the first step: success 1, its time noted, no goal error left."""
    sim, judge = judged(goal(1.0))
    (row,) = metrics_rows(sim, judge)
    assert row["robot"] == "r1"
    assert (row["tasks_done"], row["tasks_total"], row["success"]) == (1, 1, 1)
    assert row["done_at_s"] == pytest.approx(0.1)
    assert (row["goal_distance_m"], row["goal_angle_rad"]) == ("", "")
    assert (row["time_s"], row["steps"]) == (pytest.approx(0.1), 1)
    assert row["driven_m"] == 0.0 and row["collisions"] == 0


def test_a_task_in_progress_gives_the_goal_error():
    """Not done: success 0, no time, and the distance and angle left."""
    sim, judge = judged(goal(3.0), goal(5.0))
    (row,) = metrics_rows(sim, judge)
    assert (row["tasks_done"], row["tasks_total"], row["success"]) == (0, 2, 0)
    assert row["done_at_s"] == ""
    assert row["goal_distance_m"] == pytest.approx(2.0)
    assert row["goal_angle_rad"] == pytest.approx(0.0)


def test_a_robot_with_no_tasks_leaves_success_empty():
    """Nothing asked, so neither success nor failure."""
    sim, judge = judged()
    (row,) = metrics_rows(sim, judge)
    assert (row["tasks_total"], row["success"], row["done_at_s"]) == (0, "", "")


def test_the_seed_and_the_path_changes_fill_their_columns_when_given():
    """None leaves them empty, as a run driven from outside demuvo does."""
    sim, judge = judged(goal(1.0))
    (row,) = metrics_rows(sim, judge)
    assert (row["seed"], row["path_changes"]) == ("", "")
    (row,) = metrics_rows(sim, judge, {"r1": 3}, seed=7)
    assert (row["seed"], row["path_changes"]) == (7, 3)


def test_collisions_count_the_contacts_the_robot_was_part_of():
    """A contact of another pair does not count against this robot."""
    sim, judge = judged(goal(1.0))
    sim.collisions.append(Collision(0.1, "r1", "border"))
    sim.collisions.append(Collision(0.1, "r2", "obstacle 1"))
    (row,) = metrics_rows(sim, judge)
    assert row["collisions"] == 1


def test_ground_truth_gives_a_pose_error_of_zero():
    """The demo robot believes exactly where it is."""
    sim, judge = judged(goal(1.0))
    (row,) = metrics_rows(sim, judge)
    assert row["estimator"] == "ground_truth"
    assert row["pose_error_m"] == 0.0


def test_the_default_path_names_the_world_and_the_time():
    """The name comes from METRICS_NAME; the directory from METRICS_DIR."""
    when = datetime(2026, 11, 14, 15, 30, 0)
    assert metrics_path("demo", when) == Path("demuvo_metrics_demo_20261114-153000.csv")
    assert metrics_path(when=datetime.now()).name.startswith("demuvo_metrics_world_")


def test_write_metrics_writes_the_rows_as_csv(tmp_path):
    """The file has the header, one line per robot, and empties where none apply."""
    sim, judge = judged(goal(3.0))
    file = write_metrics(metrics_rows(sim, judge, seed=1), tmp_path / "run.csv")
    assert file == tmp_path / "run.csv"
    with open(file, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["robot"] == "r1" and rows[0]["seed"] == "1"
    assert rows[0]["success"] == "0" and rows[0]["done_at_s"] == ""


def test_write_metrics_without_a_destination_takes_the_default_path(
    tmp_path, monkeypatch
):
    """`METRICS_DIR` says where; changing it moves every export."""
    monkeypatch.setattr(module, "METRICS_DIR", tmp_path)
    sim, judge = judged(goal(1.0))
    file = write_metrics(metrics_rows(sim, judge))
    assert file.parent == tmp_path and file.name.startswith("demuvo_metrics_world_")
    assert "robot,seed,time_s" in file.read_text(encoding="utf-8")
