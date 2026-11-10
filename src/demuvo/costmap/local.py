"""The local costmap - a window round the robot, filled from its scans."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from robomath import Transform2D

from ..plugins import check_positive
from ..sensing import Scan
from .grid import LETHAL, Costmap
from .layers import inflate

LAYERS = ("obstacle", "inflation")


@dataclass(frozen=True)
class LocalSettings:
    """How a local costmap is built.

    Attributes:
        window: side of the square window round the robot, in metres.
        resolution: side of a cell in metres.
        period: seconds between updates.
        obstacle_max_range: the farthest a hit is marked as an obstacle, in metres.
        raytrace_max_range: the farthest a beam clears cells, in metres; a beam
            that hit nothing clears up to it.
        inflation_radius: how far cost spreads from an obstacle, in metres.
        cost_scaling_factor: how fast cost falls with distance; higher is faster.
        plugins: the layers to use, "obstacle" then "inflation".
    """

    window: float = 3.0
    resolution: float = 0.05
    period: float = 0.2
    obstacle_max_range: float = 2.5
    raytrace_max_range: float = 3.0
    inflation_radius: float = 0.55
    cost_scaling_factor: float = 10.0
    plugins: tuple[str, ...] = LAYERS

    def __post_init__(self) -> None:
        """Check the settings."""
        check_positive(
            self,
            "window",
            "resolution",
            "period",
            "obstacle_max_range",
            "raytrace_max_range",
            "inflation_radius",
            "cost_scaling_factor",
        )
        if list(self.plugins) != [p for p in LAYERS if p in self.plugins]:
            raise ValueError(
                f"plugins must list 'obstacle' and 'inflation', in that order, "
                f"each at most once, got {self.plugins!r}"
            )


class LocalCostmap:
    """The costmap a robot builds round itself from what its lidars see.

    The window follows the robot and keeps its cells aligned to the grid, so a
    mark stays in the same cell as the window moves and drops off when it leaves.
    Each update clears every cell along each beam up to its hit, or up to
    `raytrace_max_range` if it hit nothing, then marks each hit within
    `obstacle_max_range`. Clearing comes first, so a hit is not cleared by another
    beam that passes through it in the same scan. Hits are placed with the pose
    the robot is given, which is its estimate, so a drifting estimate shows.

    Args:
        inscribed_radius: the robot's inscribed radius, in metres, for inflation.
        settings: how to build it; the defaults if None.

    Attributes:
        costmap: the latest costmap, or None before the first update.
    """

    def __init__(
        self, inscribed_radius: float, settings: LocalSettings | None = None
    ) -> None:
        """Start with no costmap."""
        self._radius = inscribed_radius
        self._settings = settings or LocalSettings()
        self._cells = round(self._settings.window / self._settings.resolution)
        self._marks = np.zeros((self._cells, self._cells), dtype=np.uint8)
        self._origin: tuple[float, float] | None = None
        self._seen: dict[str, float] = {}
        self._due = 0.0
        self.costmap: Costmap | None = None

    def update(self, pose: Transform2D, scans: Mapping[str, Scan], time: float) -> None:
        """Move the window to the robot and take in the scans that are new.

        Nothing happens before the update is due, which is every `period` seconds.
        """
        settings = self._settings
        if self.costmap is not None and time + 1e-9 < self._due:
            return
        self._due = (math.floor(time / settings.period + 1e-9) + 1) * settings.period
        self._roll(pose.translation)
        window = Costmap(self._marks, settings.resolution, self._origin)
        new = [
            scan for name, scan in scans.items() if scan.time > self._seen.get(name, -1)
        ]
        self._seen.update({name: scan.time for name, scan in scans.items()})
        for scan in new:
            self._clear(window, pose, scan)
        for scan in new:
            self._mark(window, pose, scan)
        data = self._marks.copy()
        if "inflation" in settings.plugins:
            data = inflate(
                data,
                settings.resolution,
                self._radius,
                settings.inflation_radius,
                settings.cost_scaling_factor,
            )
        self.costmap = Costmap(data, settings.resolution, self._origin)

    def _roll(self, position: np.ndarray) -> None:
        """Put the window on the robot, keeping the marks that stay inside it."""
        res, n = self._settings.resolution, self._cells
        corner = np.floor(np.round((position - n * res / 2) / res, 9)) * res
        origin = (float(corner[0]), float(corner[1]))
        if self._origin is not None and origin != self._origin:
            dr = round((origin[1] - self._origin[1]) / res)
            dc = round((origin[0] - self._origin[0]) / res)
            moved = np.zeros_like(self._marks)
            r0, r1 = max(0, -dr), min(n, n - dr)
            c0, c1 = max(0, -dc), min(n, n - dc)
            if r0 < r1 and c0 < c1:
                moved[r0:r1, c0:c1] = self._marks[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            self._marks[:] = moved
        self._origin = origin

    def _inside(self, window: Costmap, world: np.ndarray):
        """The rows and columns of the world points that fall inside the window."""
        rows, cols = window.cells(world)
        keep = (rows >= 0) & (rows < self._cells) & (cols >= 0) & (cols < self._cells)
        return rows[keep], cols[keep]

    def _clear(self, window: Costmap, pose: Transform2D, scan: Scan) -> None:
        """Empty the cells each beam passed through."""
        settings = self._settings
        reach = np.where(
            np.isfinite(scan.ranges),
            np.minimum(scan.ranges, settings.raytrace_max_range),
            settings.raytrace_max_range,
        )
        along = np.arange(0.0, settings.raytrace_max_range, settings.resolution / 2)
        directions = np.stack([np.cos(scan.angles), np.sin(scan.angles)], axis=1)
        points = along[None, :, None] * directions[:, None, :]
        passed = points[along[None, :] <= reach[:, None]]
        rows, cols = self._inside(window, scan.sensor_pose(pose) @ passed)
        self._marks[rows, cols] = 0

    def _mark(self, window: Costmap, pose: Transform2D, scan: Scan) -> None:
        """Fill the cell of each hit near enough to count as an obstacle."""
        hit = np.isfinite(scan.ranges) & (
            scan.ranges <= self._settings.obstacle_max_range
        )
        angles, ranges = scan.angles[hit], scan.ranges[hit]
        points = ranges[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        rows, cols = self._inside(window, scan.sensor_pose(pose) @ points)
        self._marks[rows, cols] = LETHAL
