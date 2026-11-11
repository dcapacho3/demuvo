"""A planner you can read in one go: breadth-first search on the costmap.

Breadth-first search visits the cells in rings round the start, nearest first, until it
reaches the goal. It moves in four directions and counts every step as the same, so it
finds the path with the fewest steps, which has square corners. `AStar` is this idea
with costs and a guess of the distance left, and `LineOfSight` takes the corners out.

Use it in a world file:

    planner = "breadth_first:BreadthFirst"

and start demuvo with this folder on the path:

    PYTHONPATH=examples/plugins demuvo examples/worlds/plugins.toml
"""

from collections import deque

import numpy as np

from demuvo.paths import with_headings

# A cell costing more than this is avoided. The cost of a cell falls as it lies farther
# from an obstacle, so a limit below 253 keeps the path a little way off the walls.
AVOID = 120
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class BreadthFirst:
    """Fewest steps from the start to the goal over cells that are not blocked."""

    def configure(self, costmap):
        """Keep the costmap, which the navigation gives once and never replaces."""
        self.costmap = costmap

    def plan(self, start, goal):
        """The path as the middles of the cells, from the start to the goal.

        Returns an array of x, y, angle for each point, or an empty one if the goal
        cannot be reached.
        """
        grid, res, (x0, y0) = (
            self.costmap.data,
            self.costmap.resolution,
            self.costmap.origin,
        )
        found = self.costmap.cells(np.array([start.translation, goal.translation]))
        (row_a, row_b), (col_a, col_b) = found
        begin, end = (int(row_a), int(col_a)), (int(row_b), int(col_b))
        rows, cols = grid.shape
        for row, col in (begin, end):
            if not (0 <= row < rows and 0 <= col < cols) or grid[row, col] > AVOID:
                return np.empty((0, 3))
        came_from = {begin: None}
        queue = deque([begin])
        while queue and end not in came_from:
            row, col = queue.popleft()
            for dr, dc in STEPS:
                near = (row + dr, col + dc)
                inside = 0 <= near[0] < rows and 0 <= near[1] < cols
                if inside and near not in came_from and grid[near] <= AVOID:
                    came_from[near] = (row, col)
                    queue.append(near)
        if end not in came_from:
            return np.empty((0, 3))
        cells = []
        while end is not None:
            cells.append(end)
            end = came_from[end]
        cells.reverse()
        points = np.array(
            [[x0 + (c + 0.5) * res, y0 + (r + 0.5) * res] for r, c in cells]
        )
        points[0], points[-1] = start.translation, goal.translation
        return with_headings(points, goal.xyt[2])
