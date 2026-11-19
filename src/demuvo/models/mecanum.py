"""The mecanum (omnidirectional) model.

The full design, with the derivations and the numerical checks, is
`docs/mecanum-model-design.md`; this is the short version.

State and command
-----------------

The state is the pose alone, as for the other models: the pose of the
chassis centre, the point the robot turns about when it does not translate.
Rollers that do not slip and wheels that reach their speed at once leave
nothing else to remember between steps.

The command is the `Twist`, now with vy: the model follows vx, vy and wz, and
ignores the rest with one warning. The four wheel rates are the same command
seen at the motors; they appear only in `encoders` and `from_encoders`, as an
array in the order of `WHEELS`.

The feasible set is a box, |vx| <= max_v, |vy| <= max_vy, |wz| <= max_w,
the same simplification the diff drive makes of its wheel limits. `clip`
shrinks vx and vy by one factor, so the direction of travel is kept (see
`Limits.clip`). Every command in the box can be held for any time, so unlike
`Ackermann` the step needs no check after `clip`.

The step
--------

For a body twist held constant over dt, with theta = wz * dt, the pose moves
by the SE(2) exponential, which is one formula with no cases:

    m  = angle + theta / 2                  the heading at mid-step
    s  = sin(theta / 2) / (theta / 2)       1 when theta == 0
    x' = x + s * dt * (vx * cos m - vy * sin m)
    y' = y + s * dt * (vx * sin m + vy * cos m)
    angle' = angle + theta                  wrapped to (-pi, pi]

The robot moves along the chord of its arc: the commanded velocity turned by
half the step's turn, shortened by s. It is the exact solution of the motion,
so dt does not change where the robot ends up, and it is smooth in every
argument, across wz == 0 too: the straight line is the formula at theta = 0,
not a branch. It is what `Transform2D.rplus` computes, so `step` uses it, as
the other models do; the hand-written form above is the tests' oracle.

The wheels
----------

Seen from above, x forward and y left, the rollers form an X. With
k = (wheelbase + track) / 2 and r the wheel radius, the wheel rates are

    fl = (vx - vy - k wz) / r        fr = (vx + vy + k wz) / r
    rl = (vx + vy - k wz) / r        rr = (vx - vy + k wz) / r

The three columns are orthogonal, so the way back is exact and short:

    vx = r/4    (fl + fr + rl + rr)
    vy = r/4    (-fl + fr + rl - rr)
    wz = r/(4k) (-fl + fr - rl + rr)

Four wheels for three speeds leave one combination over: without slip the
front pair turns by as much as the rear pair, fl + fr = rl + rr. With noisy
encoders the way back is the least-squares body step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from robomath import Transform2D

from ..plugins import check_positive
from .base import Limits, Twist, move, warn_ignored

# The order of the wheels in `encoders` and `from_encoders`: front left, front
# right, rear left, rear right.
WHEELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class Mecanum:
    """Four mecanum wheels: it moves along x, along y and turns, all at once.

    It follows vx, vy and wz inside a box: |vx| <= max_v, |vy| <= max_vy and
    |wz| <= max_w. It can turn in place and move sideways. The pose is the
    centre of the four wheels. The step is exact for a constant command, a
    circular arc of radius hypot(vx, vy) / |wz|, or a straight line in any
    direction when wz is zero.

    Its wheels turn as the command says, with no slip. `encoders` gives each
    wheel's angle increment over one step, in the order of `WHEELS`, and
    `from_encoders` gives back the twist that increment came from.

    Attributes:
        max_v: largest forward speed in m/s, in either direction.
        max_vy: largest sideways speed in m/s, in either direction.
        max_w: largest turn rate in rad/s, in either direction.
        wheel_radius: radius of the four wheels in metres.
        wheelbase: distance from the front axle to the rear axle in metres.
        track: distance between the left and right wheels in metres.
    """

    max_v: float = 1.0
    max_vy: float = 1.0
    max_w: float = 1.0
    wheel_radius: float = 0.05
    wheelbase: float = 0.3
    track: float = 0.3

    name: ClassVar[str] = "mecanum"
    uses: ClassVar[tuple[str, ...]] = ("vx", "vy", "wz")
    # Settings only the encoders read; a form shows them when the robot has some.
    encoder_only: ClassVar[tuple[str, ...]] = ("wheel_radius", "wheelbase", "track")

    def __post_init__(self) -> None:
        """Refuse a speed limit or a size that is not a positive number."""
        check_positive(
            self, "max_v", "max_vy", "max_w", "wheel_radius", "wheelbase", "track"
        )

    @property
    def reach(self) -> float:
        """The lever of wz on each wheel, k = (wheelbase + track) / 2, in metres."""
        return (self.wheelbase + self.track) / 2

    def limits(self) -> Limits:
        """The feasible commands: a box, flat in wz, with a sideways bound."""
        return Limits(max_v=self.max_v, cap=self.max_w, max_vy=self.max_vy)

    def clip(self, command: Twist) -> Twist:
        """Limit a command to what the robot can do, keeping its direction.

        Warns once if the command asks for vz, wx or wy, which it ignores.
        """
        warn_ignored(self, command)
        return self.limits().clip(command)

    def step(self, pose: Transform2D, command: Twist, dt: float) -> Transform2D:
        """The pose after dt seconds of command, on the exact arc of the twist."""
        return move(pose, self.clip(command), dt)

    def encoders(self, command: Twist, dt: float) -> np.ndarray:
        """How far each wheel turns in dt, in radians, in the order of `WHEELS`.

        Args:
            command: a command already limited by `clip`.
            dt: length of the step in seconds.
        """
        vx, vy, turn = command.vx, command.vy, command.wz * self.reach
        rates = np.array(
            [vx - vy - turn, vx + vy + turn, vx + vy - turn, vx - vy + turn]
        )
        return rates * dt / self.wheel_radius

    def from_encoders(self, readings: np.ndarray) -> np.ndarray:
        """The (x, y, angle) step that wheel increments (fl, fr, rl, rr) imply.

        It is the constant twist over the step, in the robot's own frame, in the
        form `Transform2D.rplus` takes: the least-squares fit when the four
        readings do not agree.
        """
        fl, fr, rl, rr = readings
        quarter = self.wheel_radius / 4
        return np.array(
            [
                quarter * (fl + fr + rl + rr),
                quarter * (-fl + fr + rl - rr),
                quarter * (-fl + fr - rl + rr) / self.reach,
            ]
        )
