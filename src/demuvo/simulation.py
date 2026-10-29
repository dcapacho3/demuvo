"""The fixed-step loop that moves every robot of a world."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from robomath import Transform2D

from .models import MODELS, Twist
from .world import World


class Simulation:
    """A world in motion.

    Time moves in steps of a fixed length, and all robots advance together.
    The random generator comes from the world seed, so a run repeats.

    Attributes:
        world: the world that was loaded.
        dt: length of a step in seconds.
        rng: random generator, seeded with the world seed.
        models: the kinematic model of each robot, by name.
        poses: current pose of each robot, by name, as a Transform2D; its
            `xyt` is (x, y, angle).
        steps: number of steps taken.
    """

    def __init__(self, world: World, dt: float | None = None) -> None:
        """Place every robot at its start pose.

        Args:
            world: the world to run.
            dt: length of a step in seconds, positive; None takes the world's.
        """
        if dt is None:
            dt = world.dt
        if not 0 < dt < math.inf:
            raise ValueError(f"dt must be positive, got {dt}")
        self.world = world
        self.dt = dt
        self.rng = np.random.default_rng(world.seed)
        self.models = {r.name: MODELS[r.model](**r.params) for r in world.robots}
        self.poses: dict[str, Transform2D] = {r.name: r.pose for r in world.robots}
        self.steps = 0

    @property
    def time(self) -> float:
        """Simulated time in seconds, counted in steps, so it does not drift."""
        return self.steps * self.dt

    def step(self, commands: Mapping[str, Twist]) -> None:
        """Advance every robot by one step; a robot with no command stops.

        Nothing changes if this raises.

        Args:
            commands: command of each robot, by name.

        Raises:
            ValueError: if a name is not a robot of this world.
        """
        for name in commands:
            if name not in self.models:
                raise ValueError(f"no robot named '{name}'")
        moved = {
            name: model.step(self.poses[name], commands.get(name, Twist()), self.dt)
            for name, model in self.models.items()
        }
        self.poses.update(moved)
        self.steps += 1
