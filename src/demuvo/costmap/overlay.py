"""Laying one costmap over another, for a planner that uses what the robot sees."""

from __future__ import annotations

import numpy as np

from .grid import Costmap


def _covered(base: Costmap, window: Costmap):
    """The rows and columns of the base cells in the window's box, or None."""
    rows_w, cols_w = window.data.shape
    x0, y0 = window.origin
    x1, y1 = x0 + cols_w * window.resolution, y0 + rows_w * window.resolution
    r0, c0 = base.world_to_cell(x0, y0)
    r1, c1 = base.world_to_cell(x1, y1)
    height, width = base.data.shape
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, height - 1), min(c1, width - 1)
    if r0 > r1 or c0 > c1:
        return None
    rows, cols = np.meshgrid(
        np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij"
    )
    return rows.ravel(), cols.ravel()


def overlay(base: Costmap | None, window: Costmap | None) -> Costmap | None:
    """The base costmap with the window's costs laid over it.

    Each cell of the base whose centre lies in the window takes the higher of
    its own cost and the window's cost there, so an obstacle the window holds
    blocks the base too and nothing the base knows is forgotten. The grids may
    differ in resolution and origin. The result is a new costmap on the base's
    grid, so a planner configured with it is told of the change.

    Returns:
        The base itself when either is None or the window lies off the base,
        since there is nothing to lay over.
    """
    if base is None or window is None:
        return base
    covered = _covered(base, window)
    if covered is None:
        return base
    rows, cols = covered
    rows_w, cols_w = window.data.shape
    xs, ys = base.centres(rows, cols)
    wr, wc = window.cells(np.column_stack([xs, ys]))
    inside = (wr >= 0) & (wr < rows_w) & (wc >= 0) & (wc < cols_w)
    rows, cols = rows[inside], cols[inside]
    data = base.data.copy()
    data[rows, cols] = np.maximum(data[rows, cols], window.data[wr[inside], wc[inside]])
    return Costmap(data, base.resolution, base.origin)
