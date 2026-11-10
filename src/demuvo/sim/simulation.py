"""The fixed-step loop that moves every robot of a world."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from robomath import Transform2D

from ..costmap import build_costmaps
from ..models import MODELS, Twist
from ..sensing import Estimate, Scene, StepTruth
from ..world import World
from .collisions import Checker, Collision, Contact
from .robot import Robot
from .scene import static_scene, with_robots

# How many times a push out of one contact may uncover another before the robots
# involved are put back where they were.
MAX_PASSES = 4
# Extra distance, in metres, so a pushed robot ends clear and not exactly on the edge.
CLEARANCE = 1e-7


class Simulation:
    """A world in motion.

    Time moves in steps of a fixed length, and all robots advance together.
    The random generator comes from the world seed, so a run repeats.

    Attributes:
        world: the world that was loaded.
        dt: length of a step in seconds.
        rng: random generator, seeded with the world seed.
        models: the kinematic model of each robot, by name.
        costmaps: the global costmap of each robot, by name; None for all if the
            world turns the costmap off. Robots of one radius share a costmap.
        robots: the sensors and estimator of each robot, by name.
        estimates: what each robot believes its pose is, by name; the same as
            `poses` for a robot whose estimator is ground truth.
        poses: current pose of each robot, by name, as a Transform2D; its
            `xyt` is (x, y, angle).
        steps: number of steps taken.
        velocity: how each robot moved in the last step, by name, as
            (vx, vy, wz) in its own frame: the constant twist that takes the
            old pose to the new one. Zeros before the first step.
        driven: metres each robot has driven, by name. Each step adds the
            length of that twist's arc, `hypot(vx, vy) * dt`, so for a model
            that steps along exact arcs the sum is the exact path length.
        collisions: every collision so far, in order. A collision is a pair
            that was not in contact in the step before; a robot that keeps
            pressing on a wall counts once.
    """

    def __init__(self, world: World, dt: float | None = None) -> None:
        """Place every robot at its start pose.

        Args:
            world: the world to run.
            dt: length of a step in seconds, positive; None takes the world's.

        Raises:
            ValueError: if dt is not positive, or if a robot starts inside an
                obstacle, the border or another robot.
        """
        if dt is None:
            dt = world.dt
        if not 0 < dt < math.inf:
            raise ValueError(f"dt must be positive, got {dt}")
        self.world = world
        self.dt = dt
        self.rng = np.random.default_rng(world.seed)
        self.models = {r.name: MODELS[r.model](**r.params) for r in world.robots}
        self.costmaps = build_costmaps(world)
        self.robots = {
            r.name: Robot(
                r, self.models[r.name], world.seed, world.sensors, self.costmaps[r.name]
            )
            for r in world.robots
        }
        self.poses: dict[str, Transform2D] = {r.name: r.pose for r in world.robots}
        self.steps = 0
        self.velocity = {r.name: np.zeros(3) for r in world.robots}
        self.driven = {r.name: 0.0 for r in world.robots}
        self.collisions: list[Collision] = []
        self.checker = Checker(world)
        self._static_scene = static_scene(world)
        self._scene: tuple[int, Scene] | None = None  # the step it was built for
        self._touching: set[tuple[str, str]] = set()
        if overlaps := self.checker.check(self.poses):
            raise ValueError(
                f"'{overlaps[0].a}' starts inside '{overlaps[0].b}'; "
                "move one of them in the world file"
            )

    @property
    def estimates(self) -> dict[str, Estimate]:
        """What each robot believes about its pose."""
        return {name: robot.estimate for name, robot in self.robots.items()}

    @property
    def time(self) -> float:
        """Simulated time in seconds, counted in steps, so it does not drift."""
        return self.steps * self.dt

    def step(self, commands: Mapping[str, Twist]) -> None:
        """Advance every robot by one step; a robot with no command stops.

        A robot whose new pose would overlap an obstacle, the border or another
        robot is pushed back out along the contact normal, so it slides along
        what it hit and its motion into it stops. If pushing does not clear
        every contact, the robots still in contact keep their old pose.

        Nothing changes if this raises.

        Args:
            commands: command of each robot, by name.

        Raises:
            ValueError: if a name is not a robot of this world.
        """
        for name in commands:
            if name not in self.models:
                raise ValueError(f"no robot named '{name}'")
        clipped = {
            name: model.clip(commands.get(name, Twist()))
            for name, model in self.models.items()
        }
        moved = {
            name: model.step(self.poses[name], clipped[name], self.dt)
            for name, model in self.models.items()
        }
        moved = self._resolve(moved)
        velocity = {
            name: pose.rminus(self.poses[name]) / self.dt
            for name, pose in moved.items()
        }
        self.driven = {
            name: self.driven[name] + math.hypot(v[0], v[1]) * self.dt
            for name, v in velocity.items()
        }
        self.velocity = velocity
        self.poses.update(moved)
        self.steps += 1
        for index, (name, robot) in enumerate(self.robots.items()):
            robot.observe(
                StepTruth(
                    self.time,
                    self.poses[name],
                    clipped[name],
                    self.dt,
                    self.scene,
                    index,
                )
            )

    def scene(self) -> Scene:
        """What a ray can hit now: the border, the obstacles and every robot's body.

        The body of a robot has its index in `world.robots` as its owner. The
        scene is built on the first call after a step and kept until the next.
        """
        if self._scene is None or self._scene[0] != self.steps:
            bodies = [
                spec.shape.parts(self.poses[spec.name]) for spec in self.world.robots
            ]
            self._scene = (self.steps, with_robots(self._static_scene, bodies))
        return self._scene[1]

    def _resolve(self, moved: dict[str, Transform2D]) -> dict[str, Transform2D]:
        """Push the proposed poses out of every overlap and record collisions."""
        poses = dict(moved)
        contacts = self.checker.check(poses)
        pairs = {(c.a, c.b) for c in contacts}
        time = (self.steps + 1) * self.dt
        self.collisions += [
            Collision(time, a, b) for a, b in sorted(pairs - self._touching)
        ]
        self._touching = pairs
        for _ in range(MAX_PASSES):
            if not contacts:
                return poses
            pushes = _pushes(contacts)
            for name, push in pushes.items():
                x, y, angle = poses[name].xyt
                poses[name] = Transform2D.from_xyt(
                    np.array([x + push[0], y + push[1], angle])
                )
            contacts = self.checker.check(poses, pushes)
        while contacts:
            back = {n for c in contacts for n in (c.a, c.b if c.b_moves else c.a)}
            back = {n for n in back if poses[n] is not self.poses[n]}
            if not back:
                break
            poses.update({n: self.poses[n] for n in back})
            contacts = self.checker.check(poses, back)
        return poses


def _pushes(contacts: list[Contact]) -> dict[str, tuple[float, float]]:
    """The push that clears the deepest contact of each robot.

    A robot against the world takes the whole depth; two robots share it, so the
    result does not depend on which one is listed first.
    """
    best: dict[str, tuple[float, tuple[float, float]]] = {}

    def offer(name: str, depth: float, direction: tuple[float, float]) -> None:
        if name not in best or depth > best[name][0]:
            best[name] = (depth, direction)

    for c in contacts:
        share = c.depth / 2 if c.b_moves else c.depth
        offer(c.a, share, (-c.normal[0], -c.normal[1]))
        if c.b_moves:
            offer(c.b, share, c.normal)
    return {
        name: (d[0] * (depth + CLEARANCE), d[1] * (depth + CLEARANCE))
        for name, (depth, d) in best.items()
    }
