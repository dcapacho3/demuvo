"""The Ackermann (kinematic bicycle) model.

Why the command is still a Twist
--------------------------------

A real Ackermann vehicle is driven by a speed and a steering angle. demuvo's
contract (`Model`) commands every robot with a `Twist`, and the simulation,
the controllers and the world file all lean on that. The two views are the
same information: at speed vx with steering angle delta, the yaw rate is

    wz = vx * tan(delta) / wheelbase

so the model takes (vx, wz) like `DiffDrive` and treats the steering angle as
derived, delta = atan(wheelbase * wz / vx). Controllers that sample arcs of
radius vx / wz keep working unchanged; what changes is which arcs are legal.

Why there is no extra state
---------------------------

`Model.step` is a pure function of (pose, command, dt); the simulation stores
nothing about a robot but its pose. The kinematic bicycle with instantaneous
steering fits that exactly: for a constant (vx, wz) the robot follows the same
circular arc as a diff drive, so the state stays the pose alone and the step
stays exact. A steering wheel that takes time to turn (a steering-rate limit)
would make the steering angle state the model has to remember between steps;
that breaks the stateless contract and is deliberately left out of this first
model. If it is ever wanted, the contract itself must grow a state object.

What is different from a diff drive
-----------------------------------

Everything Ackermann about this model lives in `clip`:

 *  It cannot turn in place. The steering angle is bounded by `max_steer`
    (< pi/2), so wz is only reachable in proportion to vx:

        |wz| <= |vx| * tan(max_steer) / wheelbase

    i.e. the turning radius is never below wheelbase / tan(max_steer).
    `clip` keeps vx and shrinks wz to that cone; a command with vx == 0 can
    only stand still. A planner or controller that assumes rotation in place
    will visibly fail with this model, which is the point of having it.

 *  The pose is the centre of the rear axle, the point that moves along the
    arc; the front axle is `wheelbase` ahead of it. A world file should place
    the robot's shape accordingly.

Odometry: the rear wheels roll without slip like a diff drive's with the rear
`track`, so `encoders` / `from_encoders` take the same form and dead
reckoning works unchanged.

The step and the constant-steering check
----------------------------------------

`step` moves the pose by the SE(2) exponential of the body twist,
`pose.rplus([vx * dt, 0, wz * dt])`. That update is exact only when the body
twist is constant over the step; for a car that means constant speed and a
steering wheel held still, so the rear axle rides a circle about a fixed
instantaneous centre of rotation at distance vx / wz. The simulation holds a
command constant between controller runs, so constancy in time is given; what
`step` must check is that the constant steering angle the command implies is
one this car can hold:

    |wz| * wheelbase <= |vx| * tan(max_steer)

`clip` establishes that, and `step` re-checks it (`_check_steerable`) because
`step` is public and a command that skipped `clip`, or a clipped command from
a model with other limits, would otherwise sweep an arc below the least
turning radius without a word. The check is written multiplied out, with no
division and no atan, so vx == 0 needs no special case: it then reads
|wz| <= 0, which is exactly "a car that is not rolling cannot turn".
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from robomath import Transform2D

from ..plugins import check_positive
from .base import FIELDS, Limits, Twist

_USES = ("vx", "wz")
_UNUSED = tuple(name for name in FIELDS if name not in _USES)


@dataclass(frozen=True)
class Ackermann:
    """A car-like robot: front wheels steer, rear wheels drive.

    It follows vx and wz, but wz only within the cone the steering allows:
    |wz| <= |vx| * tan(max_steer) / wheelbase. It cannot move sideways and it
    cannot turn in place. The pose is the centre of the rear axle. The step is
    exact for a constant command, the same arc of radius vx / wz as DiffDrive.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        max_steer: largest steering angle in radians, in (0, pi/2). The least
            turning radius is wheelbase / tan(max_steer).
        wheelbase: distance from the rear axle to the front axle in metres.
        wheel_radius: radius of the rear wheels in metres, for the encoders.
        track: distance between the two rear wheels in metres.
    """

    max_v: float = 1.0
    max_steer: float = 0.5
    wheelbase: float = 0.3
    wheel_radius: float = 0.05
    track: float = 0.25

    name: ClassVar[str] = "ackermann"
    uses: ClassVar[tuple[str, ...]] = _USES
    # Settings only the encoders read; a form shows them when the robot has some.
    encoder_only: ClassVar[tuple[str, ...]] = ("wheel_radius", "track")

    def __post_init__(self) -> None:
        """Refuse sizes that are not positive and a steering angle past pi/2."""
        check_positive(self, "max_v", "wheelbase", "wheel_radius", "track")
        if not 0.0 < self.max_steer < math.pi / 2:
            raise ValueError(f"max_steer must be in (0, pi/2), got {self.max_steer}")

    def limits(self) -> Limits:
        """The feasible commands: a wedge, |wz| in proportion to |vx|."""
        return Limits(max_v=self.max_v, slope=math.tan(self.max_steer) / self.wheelbase)

    def clip(self, command: Twist) -> Twist:
        """Limit a command to the speeds and turning radius this robot has.

        vx is clipped to [-max_v, max_v]; wz is then clipped to
        [-k, k] with k = |vx| * tan(max_steer) / wheelbase, so the implied
        steering angle never passes max_steer and vx == 0 forces wz == 0.
        Warns once if the command asks for a Twist field the model ignores.
        """
        ignored = [name for name in _UNUSED if getattr(command, name) != 0.0]
        if ignored:
            warnings.warn(f"{self.name} ignores {', '.join(ignored)}")
        return self.limits().clip(command)

    def _check_steerable(self, command: Twist) -> None:
        """Refuse a command whose constant steering angle this car cannot hold.

        The step is the exact arc of a twist held constant over dt, which for
        a car is a steering wheel held still at delta = atan(wheelbase * wz / vx).
        The arc is only honest if that angle exists and is within max_steer:

            |wz| * wheelbase <= |vx| * tan(max_steer)

        Multiplied out so vx == 0 needs no special case: it then demands
        wz == 0, a car that is not rolling cannot turn. The tolerance forgives
        the rounding of clip itself, not a real violation.

        Raises:
            ValueError: if the command turns tighter than the steering allows.
        """
        bound = self.limits().max_w(command.vx)
        if abs(command.wz) > bound * (1 + 1e-9) + 1e-12:
            raise ValueError(
                f"{self.name} cannot hold wz={command.wz} at vx={command.vx}: "
                f"steering past max_steer={self.max_steer}"
            )

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command: an exact circular arc.

        The command is clipped, checked steerable, and held constant over the
        step, so the pose moves by the SE(2) exponential of the body twist:
        a circle of radius vx / wz about a fixed centre, or a straight line
        when wz is zero. No intermediate integration, so dt costs nothing.
        """
        command = self.clip(command)
        self._check_steerable(command)
        return pose.rplus(np.array([command.vx * dt, 0.0, command.wz * dt]))

    def encoders(self, command: Twist, dt: float) -> np.ndarray:
        """How far the left and right rear wheels turn in dt, in radians.

        Args:
            command: a command already limited by `clip`.
            dt: length of the step in seconds.
        """
        half = self.track / 2
        left = command.vx - command.wz * half
        right = command.vx + command.wz * half
        return np.array([left, right]) * dt / self.wheel_radius

    def from_encoders(self, readings: np.ndarray) -> np.ndarray:
        """The (x, y, angle) step that rear-wheel increments (left, right) imply.

        The same form as DiffDrive.from_encoders with the rear track; it is
        what `Transform2D.rplus` takes.
        """
        left, right = readings
        return np.array(
            [
                self.wheel_radius * (left + right) / 2,
                0.0,
                self.wheel_radius * (right - left) / self.track,
            ]
        )
