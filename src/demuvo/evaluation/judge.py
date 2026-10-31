"""The judge: which task each robot is on, and when it is done.

It reads only poses, velocities and time from the simulation, so it judges a
robot the same way whatever drives it: a demuvo controller, `set_path`, or a
program outside demuvo that sends commands to `Simulation.step`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from robomath import Transform2D, wrap_angle

from ..sim.simulation import Simulation
from ..world import Task

# Below these speeds, in m/s and rad/s, a robot counts as standing still.
STOPPED = (0.01, 0.01)


def reached(pose: Transform2D, task: Task) -> bool:
    """True if the pose is within the task's distance and angle of its goal."""
    xy_tolerance, yaw_tolerance = task.goal_tolerance
    x, y, yaw = pose.xyt
    gx, gy, gyaw = task.goal.xyt
    return (
        math.hypot(gx - x, gy - y) <= xy_tolerance
        and abs(wrap_angle(gyaw - yaw)) <= yaw_tolerance
    )


class Judge:
    """Walks each robot through its tasks and notes when each one is done.

    A task is done when the robot is within its tolerance of the goal, stands
    still if the task asks for it (`Task.stop`), and has held that for
    `Task.wait` seconds. A robot then moves on to its next task; the last task
    stays current.

    Call `update` once per step, after the step, from whoever runs the loop;
    calling it twice in one step changes nothing.

    Attributes:
        tasks: the tasks of each robot, in order, by name. The judge reads this
            mapping and does not copy it, so changes to it show at once.
        current: index of the task each robot is on, by name.
        done: for each finished task, (robot name, task index) to the simulated
            time at which the robot first met all its conditions.
        stopped: speeds in m/s and rad/s below which a robot counts as still.
    """

    def __init__(
        self, tasks: Mapping[str, list[Task]], stopped: tuple[float, float] = STOPPED
    ) -> None:
        """Start every robot at its first task."""
        self.tasks = tasks
        self.stopped = stopped
        self.current = dict.fromkeys(tasks, 0)
        self.done: dict[tuple[str, int], float] = {}
        self._met: dict[str, float] = {}

    def task(self, name: str) -> Task | None:
        """The task the robot is on, or None if it has none."""
        tasks = self.tasks[name]
        return tasks[self.current[name]] if tasks else None

    def reset(self, name: str) -> None:
        """Forget what a robot has done, for when its tasks are replaced."""
        self.current[name] = 0
        self._met.pop(name, None)
        self.done = {key: t for key, t in self.done.items() if key[0] != name}

    def update(self, sim: Simulation) -> None:
        """Judge every robot at the current step; move on past tasks that are done."""
        for name, tasks in self.tasks.items():
            i = self.current[name]
            # `_finished` is asked of the last task too, so it notes its time.
            while tasks and self._finished(name, i, sim) and i < len(tasks) - 1:
                i += 1
                self._met.pop(name, None)
            self.current[name] = i

    def _finished(self, name: str, i: int, sim: Simulation) -> bool:
        """True once the robot has done task `i`; the first time notes `done`."""
        tasks = self.tasks[name]
        task = tasks[i]
        stop = task.stop
        if stop is None:
            stop = i == len(tasks) - 1 or task.wait > 0
        vx, vy, wz = sim.velocity[name]
        still = math.hypot(vx, vy) <= self.stopped[0] and abs(wz) <= self.stopped[1]
        if not (reached(sim.poses[name], task) and (still or not stop)):
            self._met.pop(name, None)
            return False
        since = self._met.setdefault(name, sim.time)
        if sim.time - since < task.wait - 1e-9:  # float noise in the clock
            return False
        self.done.setdefault((name, i), since)
        return True
