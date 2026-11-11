"""The program's clock - time between frames, and how well a run keeps up.

A run keeps up if it gains as many simulated seconds as the real seconds that pass
times the speed chosen. The share of that is shown as "real time 100%".
"""

from __future__ import annotations

import time

# The share is worked out over this many seconds.
WINDOW = 0.5


class Clock:
    """Frame times and the real-time share.

    Attributes:
        last: the time of the last frame.
    """

    def __init__(self) -> None:
        """Start now."""
        self.last = time.perf_counter()
        self._from: tuple[float, float] | None = None

    def elapsed(self, now: float) -> float:
        """The seconds since the last frame; `now` becomes the last."""
        gap, self.last = now - self.last, now
        return gap

    def restart(self) -> None:
        """Start the share over, as when the speed changes or the run pauses."""
        self._from = None

    def share(self, now: float, simulated: float, speed: float) -> float | None:
        """The per cent of real time kept up with, once a window has passed.

        Args:
            now: the real time.
            simulated: the simulated time of the run.
            speed: the speed chosen, in simulated seconds for each real second.

        Returns:
            The share, at most 100, or None until enough time has passed.
        """
        if self._from is None:
            self._from = (now, simulated)
        start, since = self._from
        if now - start < WINDOW:
            return None
        self._from = (now, simulated)
        return min((simulated - since) / (now - start) / speed * 100, 100.0)
