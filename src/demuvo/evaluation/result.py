"""The result of a run, as lines of text."""

from __future__ import annotations

from ..sim.simulation import Simulation
from .judge import Judge, goal_error


def result_lines(sim: Simulation, judge: Judge) -> list[str]:
    """What each robot did, one line per task and one for the distance driven.

    A task is "reached at" the time the robot first met all its conditions, even
    if it has moved on since. The task a robot is on, if not done, says how far
    it is from the goal. Robots with no tasks give no lines.

    Args:
        sim: the simulation, at the end of the run.
        judge: the judge that followed the run; call its `update` first so the
            last step counts.
    """
    lines = []
    for name, tasks in judge.tasks.items():
        if not tasks:
            continue
        for i, task in enumerate(tasks):
            label = f"{name} task {i + 1}:"
            if (name, i) in judge.done:
                lines.append(f"{label} reached at {judge.done[(name, i)]:.2f} s")
            elif i == judge.current[name]:
                distance, angle = goal_error(sim.poses[name], task)
                lines.append(
                    f"{label} not reached, {distance:.2f} m and {angle:.2f} rad away"
                )
            else:
                lines.append(f"{label} not reached")
        lines.append(f"{name} drove {sim.driven[name]:.2f} m")
    return lines
