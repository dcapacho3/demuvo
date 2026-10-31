"""Joining planners and controllers to a simulation."""

from __future__ import annotations

import math
import warnings

import numpy as np
from robomath import Transform2D

from .controllers import Controller, Observation
from .judge import Judge
from .models import Twist
from .planners import Planner
from .plugins import load_class
from .simulation import Simulation
from .world import DEFAULT_GOAL_TOLERANCE, Task, World, WorldError

DEFAULT_PLANNER = "demuvo.planners:StraightLine"


class Navigation:
    """The planner and controller of each robot, and the path each one follows.

    Call it with the simulation at every step; it returns the command of every
    robot that has a controller, which is what the view takes. A robot with no
    controller gets no command and stops, so a program outside demuvo, such as a
    ROS bridge running Nav2, can send its own commands to `Simulation.step`.

    A robot follows its tasks in order. At a robot's first step, when it starts
    a new task, or after `set_goal`, its planner makes a path from where the
    robot is to the goal. The `judge` says when a task is done and moves the
    robot on to the next; the last goal stays, so the controller holds the robot
    there.
    `set_path` gives a robot a path made elsewhere, such as a Nav2 plan, and the
    planner then stays out of it.

    A robot's controller runs once every `control_period` seconds, as the Nav2
    controller does, and its last command stays in force between runs. Run k
    falls on step ceil(k * period / dt), counted from whole numbers so the
    schedule never drifts. A period that is not a whole number of steps gives
    gaps of two lengths, one step apart, and the mean gap is still the period.
    A planner runs when the controller does.

    Attributes:
        tasks: the tasks of each robot, in order, by name. Starts as the tasks
            of the world file.
        paths: the path of each robot that has one, by name, shape (N, 3).
        judge: which task each robot is on and which are done; see `Judge`.
    """

    def __init__(self, world: World) -> None:
        """Build the planner and controller each robot names.

        A robot with no planner uses `DEFAULT_PLANNER`.

        Raises:
            WorldError: if a named module, class or argument does not work.
        """
        self.tasks: dict[str, list[Task]] = {r.name: [] for r in world.robots}
        for task in world.tasks:
            self.tasks[task.robot].append(task)
        self.paths: dict[str, np.ndarray] = {}
        self.judge = Judge(self.tasks)
        self._planned: dict[str, int] = {}
        self._planners: dict[str, Planner] = {}
        self._controllers: dict[str, Controller] = {}
        self._periods = {r.name: r.control_period for r in world.robots}
        self._runs = dict.fromkeys(self._periods, 0)
        self._last_run = dict.fromkeys(self._periods, 0)
        self._held: dict[str, Twist] = {}
        self._checked_dt: float | None = None
        for robot in world.robots:
            if robot.controller is None:
                continue
            self._controllers[robot.name] = load_class(
                robot.controller,
                robot.controller_params,
                robot.name,
                "controller",
                "command",
            )
            self._planners[robot.name] = load_class(
                robot.planner or DEFAULT_PLANNER,
                robot.planner_params,
                robot.name,
                "planner",
                "plan",
            )

    def check(self, dt: float) -> None:
        """Check every control period against the simulation step `dt`.

        `__call__` does this when `dt` is new; call it earlier to fail early.

        Raises:
            WorldError: if a control period is shorter than `dt`, since a
                controller cannot run twice in one step.

        Warns if a period is not a whole number of steps.
        """
        if dt == self._checked_dt:
            return
        for name in self._controllers:
            period = self._periods[name]
            steps = round(period / dt, 9)
            label = f"robot '{name}': control_period {period:g}"
            if steps < 1:
                raise WorldError(f"{label} is below dt {dt:g}")
            if not steps.is_integer():
                warnings.warn(
                    f"{label} is not a whole number of steps of dt {dt:g}, "
                    "so the gap between runs alternates between two lengths"
                )
        self._checked_dt = dt

    def set_goal(
        self,
        name: str,
        goal: Transform2D | None,
        goal_tolerance: tuple[float, float] = DEFAULT_GOAL_TOLERANCE,
    ) -> None:
        """Give a robot one new goal, or None for no task; it is planned again.

        Raises:
            ValueError: if there is no robot of that name.
        """
        self._check(name)
        self.tasks[name] = [] if goal is None else [Task(name, goal, goal_tolerance)]
        self.judge.reset(name)
        self.paths.pop(name, None)

    def set_path(self, name: str, path) -> None:
        """Give a robot a path made elsewhere, as (x, y, angle) rows.

        If the robot has no task, the last pose of the path is its goal.

        Raises:
            ValueError: if there is no robot of that name, or the path is not
                an array of shape (N, 3) of finite numbers with N at least 1.
        """
        self._check(name)
        array = np.array(path, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
            raise ValueError(f"path must have shape (N, 3), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("path must hold finite numbers")
        self.paths[name] = array

    def _check(self, name: str) -> None:
        """Refuse a name that is not a robot of the world."""
        if name not in self.tasks:
            raise ValueError(f"no robot named '{name}'")

    def _due(self, name: str, dt: float) -> int:
        """The step of the robot's next controller run."""
        return math.ceil(round(self._runs[name] * self._periods[name] / dt, 9))

    def __call__(self, sim: Simulation) -> dict[str, Twist]:
        """The command of every robot that has a controller, for this step."""
        self.check(sim.dt)
        self.judge.update(sim)
        commands = {}
        for name, controller in self._controllers.items():
            if sim.steps < self._due(name, sim.dt):
                commands[name] = self._held[name]
                continue
            pose = sim.poses[name]
            task = self.judge.task(name)
            if self._planned.get(name, 0) != self.judge.current[name]:
                self.paths.pop(name, None)  # the robot moved on to its next task
            if name not in self.paths:
                self.paths[name] = np.empty((0, 3))
                self._planned[name] = self.judge.current[name]
                if task is not None:
                    path = self._planners[name].plan(sim.world, pose, task.goal)
                    self.paths[name] = path
            path = self.paths[name]
            goal = task.goal if task else None
            if goal is None and len(path) > 0:
                goal = Transform2D.from_xyt(path[-1])
            tolerance = task.goal_tolerance if task else DEFAULT_GOAL_TOLERANCE
            gap = (
                (sim.steps - self._last_run[name]) * sim.dt
                if self._runs[name]
                else self._periods[name]
            )
            commands[name] = self._held[name] = controller.command(
                Observation(
                    time=sim.time,
                    dt=gap,
                    pose=pose,
                    path=path,
                    goal=goal,
                    goal_tolerance=tolerance,
                )
            )
            self._runs[name] += 1
            self._last_run[name] = sim.steps
        return commands
