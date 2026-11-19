"""Observations that the controller tests build the same way."""

import dataclasses

from robomath import Transform2D

from demuvo import Observation, StraightLine


def look(
    pose, goal, *, tolerance=(0.1, 0.1), path=None, limits=None, costmap=None, **more
):
    """An Observation for a robot at pose with a straight path to goal.

    Args:
        pose: (x, y, angle) of the robot.
        goal: (x, y, angle) of the goal.
        tolerance: the goal's distance and angle tolerance.
        path: the path; a straight line to the goal if None.
        limits: the limits of the robot's model, or None.
        costmap: the robot's local costmap, or None.
        more: any other field of `Observation`, such as `dt` or `path_changes`.
    """
    pose, goal = Transform2D.from_xyt(pose), Transform2D.from_xyt(goal)
    if path is None:
        path = StraightLine().plan(pose, goal)
    return Observation(
        **{"time": 0.0, "dt": 0.05, "path_changes": 1}
        | more
        | {
            "pose": pose,
            "path": path,
            "goal": goal,
            "goal_tolerance": tolerance,
            "limits": limits,
            "costmap": costmap,
        }
    )


def see(pose, goal, limits, **kwargs):
    """`look`, with the robot's limits in the observation."""
    return look(pose, goal, limits=limits, **kwargs)


def moved(obs, pose):
    """The same observation one step later, with the robot at a new pose."""
    return dataclasses.replace(obs, time=obs.time + obs.dt, pose=pose)
