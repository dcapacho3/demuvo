"""A run - a world and its navigation, stepped by whoever owns the clock.

A window steps it a little each frame, a benchmark steps it to the end, and a
test steps it a known number of times. None of them needs more than this class,
and none of them needs a window library: this module never imports one.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .costmap import Costmap
from .evaluation import result_lines
from .nav.navigation import Navigation
from .sim.simulation import Simulation
from .verify import WorldError
from .world import World, load_world


@dataclass(frozen=True)
class Beams:
    """The latest scan of one range sensor, placed in the world.

    Attributes:
        origin: where the sensor is, as (x, y) in metres.
        angles: the angle of each beam in the world, in radians.
        ranges: the distance each beam read, in metres; `inf` where it read nothing.
        max_range: the farthest the sensor reads, in metres.
    """

    origin: tuple[float, float]
    angles: np.ndarray
    ranges: np.ndarray
    max_range: float


class Run:
    """One world in motion.

    Args:
        source: the path of a world file, or a `World` built or edited in memory.
        dt: step length in seconds; None takes the world's.

    Attributes:
        world: the world being run.
        sim: its simulation.
        nav: its navigation, which gives each robot its command.

    Raises:
        WorldError: if the file or a part a robot names is wrong.
        ValueError: if the step length is not positive or a robot starts
            inside another body.
    """

    def __init__(self, source: str | os.PathLike | World, dt: float | None = None):
        """Load the world if it is a path, and start at time zero."""
        self._dt = dt
        self.world = source if isinstance(source, World) else load_world(source)
        self._preview_of: World | None = None
        self._preview_paths: dict[str, np.ndarray | None] = {}
        self.reset()

    def reset(self) -> None:
        """Start again from the world as it is, at time zero."""
        self.sim = Simulation(self.world, self._dt)
        self.nav = Navigation(self.world)
        self.nav.check(self.sim.dt)

    def poll(self) -> bool:
        """Take news from a simulation that runs elsewhere; here there is none."""
        return False

    def take_log(self) -> list[tuple[float, str, str]]:
        """Messages from a simulation that runs elsewhere; here they print as usual."""
        return []

    def close(self) -> None:
        """Release what the run holds; a run in this process holds nothing."""

    def set_world(self, world: World) -> None:
        """Run another world, for instance after an edit, from time zero."""
        self.world = world
        self.reset()

    @property
    def time(self) -> float:
        """Simulated time in seconds."""
        return self.sim.time

    @property
    def poses(self) -> dict[str, np.ndarray]:
        """The true pose of each robot as (x, y, angle), by name."""
        return {name: pose.xyt for name, pose in self.sim.poses.items()}

    def step(self, count: int = 1) -> None:
        """Advance by `count` steps."""
        for _ in range(count):
            self.sim.step(self.nav(self.sim))

    def advance(self, seconds: float, budget: float | None = None) -> int:
        """Advance by `seconds` of simulated time, at least one step.

        Args:
            seconds: how much simulated time to cover.
            budget: wall-clock seconds to spend at most; a window passes a frame's
                worth so a heavy world slows down and does not freeze. None takes
                all the steps.

        Returns:
            The number of steps taken.
        """
        due = max(1, round(seconds / self.sim.dt))
        deadline = None if budget is None else time.perf_counter() + budget
        taken = 0
        while taken < due:
            self.step()
            taken += 1
            if deadline is not None and time.perf_counter() > deadline:
                break
        return taken

    @property
    def finished(self) -> bool:
        """True once every robot that has tasks has done its last one."""
        self.nav.judge.update(self.sim)
        done = self.nav.judge.done
        tasks = [(name, len(t) - 1) for name, t in self.nav.judge.tasks.items() if t]
        return bool(tasks) and all(key in done for key in tasks)

    def until_done(self, limit: float) -> bool:
        """Step until every robot has finished its tasks, or `limit` seconds pass.

        Returns:
            True if the tasks are done, False if the time ran out first.
        """
        while not self.finished and self.time < limit:
            self.step()
        return self.finished

    def result(self) -> list[str]:
        """What each robot did so far, one line per task; see `result_lines`."""
        self.nav.judge.update(self.sim)
        return result_lines(self.sim, self.nav.judge)

    def _robot(self, name: str):
        """The robot of that name in the simulation, or a plain error."""
        if name not in self.sim.robots:
            raise WorldError(f"no robot named '{name}'")
        return self.sim.robots[name]

    def scans(self, name: str) -> list[Beams]:
        """The latest scan of each range sensor of a robot, placed in the world."""
        robot = self._robot(name)
        placed = []
        for scan in robot.scans().values():
            x, y, yaw = scan.sensor_pose(self.sim.poses[name]).xyt
            placed.append(Beams((x, y), scan.angles + yaw, scan.ranges, scan.max_range))
        return placed

    def local_costmap(self, name: str) -> Costmap | None:
        """The local costmap of a robot, or None if it has none or no update yet."""
        return self._robot(name).local_costmap

    def path(self, name: str) -> np.ndarray | None:
        """The path a robot follows, rows of (x, y, angle); None if it has none.

        Before the first step a robot has no path yet, so the one its planner
        would make is shown. It is made once for each world.
        """
        self._robot(name)
        path = self.nav.paths.get(name)
        if path is None and self.sim.steps == 0:
            path = self._previews().get(name)
        return path

    def _previews(self) -> dict[str, np.ndarray | None]:
        """The paths the planners would make now, kept until the world changes."""
        if self._preview_of is not self.world:
            self._preview_of = self.world
            self._preview_paths = {
                name: self.nav.preview(self.sim, name) for name in self.sim.robots
            }
        return self._preview_paths
