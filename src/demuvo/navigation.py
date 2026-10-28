"""Joining planners and controllers to a simulation."""

from __future__ import annotations

import numpy as np

from .controllers import Controller, Observation
from .models import Twist
from .planners import Planner
from .plugins import load_class
from .poses import Pose
from .simulation import Simulation
from .world import World

DEFAULT_PLANNER = "demuvo.planners:StraightLine"


class Navigation:
    """The planner and controller of each robot, and the path each one follows.

    Call it with the simulation at every step; it returns the command of every
    robot that has a controller, which is what the view takes. A robot with no
    controller gets no command and stops, so a program outside demuvo, such as a
    ROS bridge running Nav2, can send its own commands to `Simulation.step`.

    At a robot's first step, or after `set_goal`, its planner makes a path from
    where the robot is to its goal. `set_path` gives a robot a path made
    elsewhere, such as a Nav2 plan, and the planner then stays out of it.

    Attributes:
        goals: the goal of each robot as (x, y, angle), or None, by name. It
            starts as the goal in the world file.
        paths: the path of each robot that has one, by name, shape (N, 3).
    """

    def __init__(self, world: World) -> None:
        """Build the planner and controller each robot names.

        A robot with no planner uses `DEFAULT_PLANNER`.

        Raises:
            WorldError: if a named module, class or argument does not work.
        """
        self.goals: dict[str, Pose | None] = {r.name: r.goal for r in world.robots}
        self.paths: dict[str, np.ndarray] = {}
        self._planners: dict[str, Planner] = {}
        self._controllers: dict[str, Controller] = {}
        for robot in world.robots:
            if robot.controller is None:
                continue
            self._controllers[robot.name] = load_class(
                robot.controller, robot.controller_params, robot.name, "controller"
            )
            self._planners[robot.name] = load_class(
                robot.planner or DEFAULT_PLANNER,
                robot.planner_params,
                robot.name,
                "planner",
            )

    def set_goal(self, name: str, goal: Pose | None) -> None:
        """Give a robot a new goal, or None for no goal; it is planned again.

        Raises:
            ValueError: if there is no robot of that name.
        """
        self._check(name)
        self.goals[name] = None if goal is None else tuple(map(float, goal))
        self.paths.pop(name, None)

    def set_path(self, name: str, path) -> None:
        """Give a robot a path made elsewhere, as (x, y, angle) rows.

        If the robot has no goal, the last pose of the path is its goal.

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
        if name not in self.goals:
            raise ValueError(f"no robot named '{name}'")

    def __call__(self, sim: Simulation) -> dict[str, Twist]:
        """The command of every robot that has a controller, for this step."""
        commands = {}
        for name, controller in self._controllers.items():
            pose = sim.poses[name]
            goal = self.goals[name]
            if name not in self.paths:
                self.paths[name] = np.empty((0, 3))
                if goal is not None:
                    start = tuple(map(float, pose.xyt))
                    self.paths[name] = self._planners[name].plan(sim.world, start, goal)
            path = self.paths[name]
            if goal is None and len(path) > 0:
                goal = tuple(map(float, path[-1]))
            commands[name] = controller.command(
                Observation(
                    time=sim.time,
                    dt=sim.dt,
                    pose=pose,
                    path=path,
                    goal=goal,
                    goal_tolerance=sim.specs[name].goal_tolerance,
                )
            )
        return commands
