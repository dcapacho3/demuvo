"""The key numbers of a run, as rows for a CSV file.

One row per robot, with the run-level numbers (seed, time, steps) repeated on
every row, so the rows of many runs or seeds concatenate into one flat table
that a spreadsheet or pandas reads without reshaping.

Where the file goes and what it is called live in `METRICS_DIR`, `METRICS_NAME`
and `STAMP` below; change those to change every export. A caller that wants a
particular file passes its path to `write_metrics` instead.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from ..sim.simulation import Simulation
from .judge import Judge, goal_error

# The default place and name of the file; `metrics_path` puts them together.
METRICS_DIR = Path(".")
METRICS_NAME = "demuvo_metrics_{world}_{stamp}.csv"
STAMP = "%Y%m%d-%H%M%S"

# The columns of the file, in order. A number that does not apply to a robot,
# such as the goal error of one whose tasks are all done, is left empty.
COLUMNS = (
    "robot",
    "seed",
    "time_s",
    "steps",
    "tasks_done",
    "tasks_total",
    "success",
    "done_at_s",
    "driven_m",
    "goal_distance_m",
    "goal_angle_rad",
    "path_changes",
    "collisions",
    "estimator",
    "pose_error_m",
)


def metrics_path(world: str = "world", when: datetime | None = None) -> Path:
    """The default file: `METRICS_NAME` under `METRICS_DIR`.

    Args:
        world: the name of the world, usually the stem of its file.
        when: the time in the name; None takes now.
    """
    stamp = (when or datetime.now()).strftime(STAMP)
    return METRICS_DIR / METRICS_NAME.format(world=world, stamp=stamp)


def metrics_rows(
    sim: Simulation,
    judge: Judge,
    path_changes: Mapping[str, int] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """The key numbers of a run, one row per robot; see `COLUMNS`.

    `success` is 1 when the robot has done every task, 0 when not, and empty for
    a robot with no tasks. `done_at_s` is the time its last task was reached.
    `goal_distance_m` and `goal_angle_rad` say how far the task it is still on
    is from done. `path_changes` counts every path the robot was handed, the
    first plan of each task and every replan alike; it is empty when the caller
    does not know it, as a simulation driven from outside demuvo does not.
    `collisions` counts the contacts the robot was part of. `pose_error_m` is
    how far the estimator's pose ended from the true one, 0 under ground truth.

    Args:
        sim: the simulation, at the end of the run.
        judge: the judge that followed the run; call its `update` first so the
            last step counts.
        path_changes: how many paths each robot was handed, by name; None if
            unknown.
        seed: the seed of the world, or None to leave the column empty.
    """
    rows = []
    for name, robot in sim.robots.items():
        tasks = judge.tasks.get(name, [])
        done = sum(1 for key in judge.done if key[0] == name)
        all_done = bool(tasks) and done == len(tasks)
        distance = angle = ""
        if tasks and not all_done:
            distance, angle = goal_error(sim.poses[name], tasks[judge.current[name]])
        true, believed = sim.poses[name].xyt, robot.estimate.pose.xyt
        rows.append(
            {
                "robot": name,
                "seed": "" if seed is None else seed,
                "time_s": sim.time,
                "steps": sim.steps,
                "tasks_done": done,
                "tasks_total": len(tasks),
                "success": int(all_done) if tasks else "",
                "done_at_s": judge.done[(name, len(tasks) - 1)] if all_done else "",
                "driven_m": sim.driven[name],
                "goal_distance_m": distance,
                "goal_angle_rad": angle,
                "path_changes": "" if path_changes is None else path_changes[name],
                "collisions": sum(1 for c in sim.collisions if name in (c.a, c.b)),
                "estimator": robot.spec.estimator,
                "pose_error_m": math.hypot(
                    true[0] - believed[0], true[1] - believed[1]
                ),
            }
        )
    return rows


def write_metrics(
    rows: list[dict], destination: str | os.PathLike | None = None
) -> Path:
    """Write the rows as a CSV file and return its path.

    Args:
        rows: what `metrics_rows` gave.
        destination: the file to write; None takes `metrics_path()`. A file
            that exists is replaced.
    """
    path = Path(destination) if destination is not None else metrics_path()
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path
