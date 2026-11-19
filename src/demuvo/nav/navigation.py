"""Joining planners and controllers to a simulation."""

from __future__ import annotations

import logging
import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from robomath import Transform2D

from ..controllers import Controller, Observation
from ..costmap import Costmap, overlay
from ..evaluation.judge import Judge
from ..models import MODELS, Twist
from ..planners import Planner
from ..plugins import load_class
from ..sim.simulation import Simulation
from ..smoothers import Smoother
from ..world import DEFAULT_GOAL_TOLERANCE, DEFAULT_PLANNER, Task, World, WorldError
from .replan import Replanner, Sight

LOGGER = logging.getLogger(__name__)


@dataclass
class _Plan:
    """What the navigation remembers about the path a robot holds.

    Attributes:
        task: the index of the task the path is for.
        time: when the path was planned, or last tried.
        outside: True for a path from `set_path`, which the planner leaves alone.
        failed: True if the last plan found no path.
    """

    task: int
    time: float
    outside: bool = False
    failed: bool = False


class Navigation:  # pylint: disable=too-many-instance-attributes
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
    A robot's smoother, if it has one, improves each path the planner makes.
    `set_path` gives a robot a path made elsewhere, such as a Nav2 plan, and the
    planner and smoother then stay out of it.

    A robot with a `replan` spec also plans again while it drives, when the
    `Replanner` gives a reason: the path ahead is blocked in its local costmap,
    it is off the path, it stands still, or a timer ran out. Such a robot plans
    on the global costmap with its local costmap laid over it, so the planner
    goes round what the lidar sees, and its planner and smoother are configured
    again with the new costmap before each plan. The plan is made at a
    controller run, from the pose the controller sees, and replaces the path
    before the controller runs, so the controller follows the new path from
    that run on and `Observation.path_changes` grows by one. A plan that finds
    no path leaves the path as it is, says so, and is tried again every period,
    quietly, until one is found.

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
        replanner: decides when a robot plans again; see `Replanner`.
    """

    def __init__(self, world: World) -> None:
        """Build the planner, controller and smoother each robot names.

        A robot with no planner uses `DEFAULT_PLANNER`.

        Raises:
            WorldError: if a named module, class or argument does not work.
        """
        self.tasks: dict[str, list[Task]] = {r.name: [] for r in world.robots}
        for task in world.tasks:
            self.tasks[task.robot].append(task)
        self.paths: dict[str, np.ndarray] = {}
        self._specs = {r.name: r for r in world.robots}
        self.judge = Judge(self.tasks)
        self.replanner = Replanner({r.name: r.replan for r in world.robots})
        self._plans: dict[str, _Plan] = {}
        self._path_changes = dict.fromkeys(self.tasks, 0)
        self._planners: dict[str, Planner] = {}
        self._maps: dict[str, Costmap | None] = {}  # what each planner was given
        self._smoothers: dict[str, Smoother] = {}
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
                ("configure", "plan"),
            )
            limits = MODELS[robot.model](**robot.params).limits()
            if not limits.turns_in_place and not getattr(
                self._planners[robot.name], "curvature_bounded", False
            ):
                LOGGER.warning(
                    "robot '%s': it cannot turn in place, but its planner does "
                    "not bound the path's curvature; it may not track the path "
                    "or reach a goal's angle. The Dubins planner does.",
                    robot.name,
                )
            self._warn_if_tighter_than_steering(robot.name, limits)
            if robot.smoother is not None:
                self._smoothers[robot.name] = load_class(
                    robot.smoother,
                    robot.smoother_params,
                    robot.name,
                    "smoother",
                    ("configure", "smooth"),
                )

    def _warn_if_tighter_than_steering(self, name: str, limits) -> None:
        """Warn if the planner bends its path tighter than the robot can steer."""
        radius = getattr(self._planners[name], "turning_radius", None)
        if radius is None or limits.turns_in_place:
            return
        least = 1.0 / limits.slope
        if radius < least * (1.0 - 1e-9):
            LOGGER.warning(
                "robot '%s': its planner bends the path to a radius of %.2f m, but "
                "the robot cannot steer tighter than %.2f m; it may not track the "
                "path or reach a goal's angle.",
                name,
                radius,
                least,
            )

    @property
    def path_changes(self) -> dict[str, int]:
        """How many paths each robot was handed, by name.

        The first plan of each task, every replan and every `set_path` count
        one each, as `Observation.path_changes` counts them for a controller.
        """
        return dict(self._path_changes)

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
        self._plans.pop(name, None)

    def set_path(self, name: str, path) -> None:
        """Give a robot a path made elsewhere, as (x, y, angle) rows.

        If the robot has no task, the last pose of the path is its goal. The
        robot does not plan again while it follows this path; its next task,
        or `set_goal`, brings its planner back.

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
        self._plans[name] = _Plan(self.judge.current[name], 0.0, outside=True)
        self._path_changes[name] += 1

    def _check(self, name: str) -> None:
        """Refuse a name that is not a robot of the world."""
        if name not in self.tasks:
            raise ValueError(f"no robot named '{name}'")

    def preview(self, sim: Simulation, name: str) -> np.ndarray | None:
        """The path a robot would be given at its first step, made without a step.

        It uses planners of its own, so the run is not touched and its results do
        not change. A robot with no controller or no task has none.
        """
        spec, tasks = self._specs[name], self.tasks[name]
        if spec.controller is None or not tasks:
            return None
        parts = []
        planner = load_class(
            spec.planner or DEFAULT_PLANNER,
            spec.planner_params,
            name,
            "planner",
            ("configure", "plan"),
        )
        parts.append(planner)
        if spec.smoother is not None:
            parts.append(
                load_class(
                    spec.smoother,
                    spec.smoother_params,
                    name,
                    "smoother",
                    ("configure", "smooth"),
                )
            )
        with _quiet():  # a preview does not give the run's warnings
            try:
                for part in parts:
                    part.configure(sim.estimates[name].map)
            except ValueError:  # a planner that needs a costmap the world turns off
                return None
            truth = self._specs[name].control_pose == "truth"
            pose = sim.poses[name] if truth else sim.estimates[name].pose
            path = planner.plan(pose, tasks[0].goal)
            if len(path) and len(parts) > 1:
                path = parts[1].smooth(path)
        return path

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
            pose = (
                sim.poses[name]
                if self._specs[name].control_pose == "truth"
                else sim.estimates[name].pose
            )
            task = self.judge.task(name)
            self._keep_path(name, sim, pose, task)
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
                    path_changes=self._path_changes[name],
                    scans=sim.robots[name].scans(),
                    costmap=sim.robots[name].local_costmap,
                    limits=sim.robots[name].model.limits(),
                )
            )
            self._runs[name] += 1
            self._last_run[name] = sim.steps
        return commands

    def _keep_path(
        self, name: str, sim: Simulation, pose: Transform2D, task: Task | None
    ) -> None:
        """Give the robot a path for its task, and plan again when there is a reason."""
        current = self.judge.current[name]
        plan = self._plans.get(name)
        if plan is not None and plan.task != current:
            self.paths.pop(name, None)  # the robot moved on to its next task
            plan = None
        if plan is None:
            plan = self._plans[name] = _Plan(current, sim.time)
            self.paths[name] = np.empty((0, 3))
            if task is None or not self._plan(name, sim, pose, task, plan, reason=None):
                self._path_changes[name] += 1  # the empty path is a new path too
        elif task is not None and not plan.outside:
            sight = Sight(
                time=sim.time,
                pose=pose,
                path=self.paths[name],
                task=task,
                planned_at=plan.time,
                failed=plan.failed,
                velocity=sim.velocity[name],
                costmap=sim.robots[name].local_costmap,
            )
            reason = self.replanner.reason(name, sight)
            if reason is not None:
                self._plan(name, sim, pose, task, plan, reason=reason)

    def _plan(
        self,
        name: str,
        sim: Simulation,
        pose: Transform2D,
        task: Task,
        plan: _Plan,
        *,
        reason: str | None,
    ) -> bool:
        """Plan from the pose to the task's goal and hand the path to the robot.

        `reason` is None for the first plan of a task. A plan that finds nothing
        leaves the path the robot has, and says so unless the one before failed
        too. A path replaces the old one whole, and `path_changes` grows, so a
        controller that keeps state about the path starts over.

        Returns:
            True if a path was handed over.
        """
        planner, smoother = self._planners[name], self._smoothers.get(name)
        self._configure(name, self._planning_map(name, sim))
        self.replanner.planned(name, sim.robots[name].local_costmap)
        index = self.judge.current[name]
        with _quiet(plan.failed):  # a retry says nothing new
            path = planner.plan(pose, task.goal)
        plan.time = sim.time
        if len(path) == 0:
            if reason is None:
                _warn_no_path(name, index, pose, task.goal, self.replanner.specs[name])
            elif not plan.failed:
                LOGGER.warning(
                    "robot '%s': no new path for task %d, though %s; it keeps the "
                    "path it has and tries again",
                    name,
                    index + 1,
                    reason,
                )
            plan.failed = True
            return False
        if reason is None or plan.failed:
            _warn_short_path(name, index, path, task)
        if smoother:
            path = smoother.smooth(path)
        if plan.failed:
            LOGGER.warning("robot '%s': found a path for task %d", name, index + 1)
        elif reason is not None:
            LOGGER.info(
                "robot '%s': planned task %d again: %s", name, index + 1, reason
            )
        plan.failed = False
        self.paths[name] = path
        self._path_changes[name] += 1
        self.replanner.forget(name)
        return True

    def _planning_map(self, name: str, sim: Simulation) -> Costmap | None:
        """The costmap the robot plans on.

        The global costmap, with the local costmap laid over it for a robot that
        replans, so what the lidar sees is in the plan.
        """
        base = sim.estimates[name].map
        if self.replanner.specs[name] is None:
            return base
        return overlay(base, sim.robots[name].local_costmap)

    def _configure(self, name: str, costmap: Costmap | None) -> None:
        """Give the planner and smoother the costmap, unless they have it already."""
        if name in self._maps and self._maps[name] is costmap:
            return
        self._planners[name].configure(costmap)
        if name in self._smoothers:
            self._smoothers[name].configure(costmap)
        self._maps[name] = costmap


@contextmanager
def _quiet(enabled: bool = True):
    """Silence the log inside the block, if `enabled`."""
    if not enabled:
        yield
        return
    earlier = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(earlier)


def _warn_no_path(name: str, index: int, start: Transform2D, goal: Transform2D, replan):
    """Say that a robot's planner found no path, and what the robot does now."""
    then = "it stays still"
    if replan is not None:
        then = f"it stays still and tries again every {replan.period:g} s"
    LOGGER.warning(
        "robot '%s': no path for task %d, from (%.2f, %.2f) to (%.2f, %.2f); %s",
        name,
        index + 1,
        *start.xyt[:2],
        *goal.xyt[:2],
        then,
    )


def _warn_short_path(name: str, index: int, path: np.ndarray, task: Task) -> None:
    """Say that a robot's path ends too far from its goal to count as reached."""
    gap = math.dist(path[-1][:2], task.goal.xyt[:2])
    reach = task.goal_tolerance[0]
    if gap > reach:
        LOGGER.warning(
            "robot '%s': the path for task %d ends %.2f m from the goal, "
            "more than its tolerance of %.2f m",
            name,
            index + 1,
            gap,
            reach,
        )
