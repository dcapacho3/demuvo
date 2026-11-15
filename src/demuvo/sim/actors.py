"""The actors of a world in motion: each one's motion, and where it is at a time."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robomath import Transform2D

from ..actors import Motion, Still
from ..plugins import load_class
from ..sensing import stream
from ..world import WorldError
from ..world.shapes import Part

if TYPE_CHECKING:
    from ..world import ActorSpec, World

METHODS = ("configure", "pose_at")


class Actors:
    """The actors of a world, each with its motion built and configured.

    An actor's pose is a function of simulated time, so this holds no clock of
    its own: the simulation asks for the poses at the time of each step. An
    actor whose spec names no motion gets `Still`, and stands where it was put.

    Args:
        world: the world; its actors and its seed.

    Attributes:
        specs: the actors, by name, in file order.
        motions: the motion of each actor, by name, configured and ready.

    Raises:
        WorldError: if a motion cannot be built, or refuses its start pose.
    """

    def __init__(self, world: World) -> None:
        """Build and configure the motion of every actor."""
        self.specs: dict[str, ActorSpec] = {a.name: a for a in world.actors}
        self.motions: dict[str, Motion] = {}
        for spec in world.actors:
            motion = (
                Still()
                if spec.motion is None
                else load_class(
                    spec.motion,
                    spec.motion_params,
                    spec.name,
                    "motion",
                    METHODS,
                    "actor",
                )
            )
            try:
                motion.configure(spec.pose, stream(world.seed, "actor", spec.name))
            except ValueError as error:
                raise WorldError(
                    f"actor '{spec.name}': motion '{spec.motion}': {error}"
                ) from error
            self.motions[spec.name] = motion

    def __len__(self) -> int:
        """How many actors there are."""
        return len(self.specs)

    def __bool__(self) -> bool:
        """True if the world has any actor."""
        return bool(self.specs)

    def poses_at(self, time: float) -> dict[str, Transform2D]:
        """Where every actor is at this time, by name, in file order."""
        return {name: motion.pose_at(time) for name, motion in self.motions.items()}

    def parts_at(self, time: float) -> dict[str, list[Part]]:
        """The convex parts of every actor's body at this time, by name."""
        return {
            name: self.specs[name].shape.parts(pose)
            for name, pose in self.poses_at(time).items()
        }
