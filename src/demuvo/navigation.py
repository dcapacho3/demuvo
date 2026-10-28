"""Joining planners and controllers to a simulation."""

from __future__ import annotations

import math

import numpy as np
from robomath import Transform2D, wrap_angle

from .controllers import Controller, Observation
from .models import Twist
from .planners import Planner
from .plugins import load_class
from .simulation import Simulation
from .world import DEFAULT_GOAL_TOLERANCE, Task, World

DEFAULT_PLANNER = "demuvo.planners:StraightLine"


def reached(pose: Transform2D, task: Task) -> bool:
    """True if the pose is within the task's distance and angle of its goal."""
    xy_tolerance, yaw_tolerance = task.goal_tolerance
    x, y, yaw = pose.xyt
    gx, gy, gyaw = task.goal.xyt
    return (
        math.hypot(gx - x, gy - y) <= xy_tolerance
        and abs(wrap_angle(gyaw - yaw)) <= yaw_tolerance
    )


class Navigation:
    """The planner and controller of each robot, and the path each one follows.

    Call it with the simulation at every step; it returns the command of every
    robot that has a controller, which is what the view takes. A robot with no
    controller gets no command and stops, so a program outside demuvo, such as a
    ROS bridge running Nav2, can send its own commands to `Simulation.step`.

    A robot follows its tasks in order. At a robot's first step, when it starts
    a new task, or after `set_goal`, its planner makes a path from where the
    robot is to the goal. A task is done when the robot is within its tolerance
    of the goal; the last goal stays, so the controller holds the robot there.
    `set_path` gives a robot a path made elsewhere, such as a Nav2 plan, and the
    planner then stays out of it.

    Attributes:
        tasks: the tasks of each robot, in order, by name. Starts as the tasks
            of the world file.
        paths: the path of each robot that has one, by name, shape (N, 3).
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
        self._current = dict.fromkeys(self.tasks, 0)
        self._planners: dict[str, Planner] = {}
        self._controllers: dict[str, Controller] = {}
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
        self._current[name] = 0
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

    def _task(self, name: str, pose: Transform2D) -> Task | None:
        """The task the robot is on; move on past tasks that are done."""
        tasks, i = self.tasks[name], self._current[name]
        while i < len(tasks) - 1 and reached(pose, tasks[i]):
            i += 1
            self.paths.pop(name, None)
        self._current[name] = i
        return tasks[i] if tasks else None

    def __call__(self, sim: Simulation) -> dict[str, Twist]:
        """The command of every robot that has a controller, for this step."""
        commands = {}
        for name, controller in self._controllers.items():
            pose = sim.poses[name]
            task = self._task(name, pose)
            if name not in self.paths:
                self.paths[name] = np.empty((0, 3))
                if task is not None:
                    path = self._planners[name].plan(sim.world, pose, task.goal)
                    self.paths[name] = path
            path = self.paths[name]
            goal = task.goal if task else None
            if goal is None and len(path) > 0:
                goal = Transform2D.from_xyt(path[-1])
            tolerance = task.goal_tolerance if task else DEFAULT_GOAL_TOLERANCE
            commands[name] = controller.command(
                Observation(
                    time=sim.time,
                    dt=sim.dt,
                    pose=pose,
                    path=path,
                    goal=goal,
                    goal_tolerance=tolerance,
                )
            )
        return commands
