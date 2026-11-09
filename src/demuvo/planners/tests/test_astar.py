"""Tests for the A* planner."""

import logging
import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import AStar, Costmap
from demuvo.costmap import INSCRIBED, LETHAL


def pose(x, y, yaw=0.0):
    """A pose from (x, y, angle)."""
    return Transform2D.from_xyt((x, y, yaw))


def grid(rows=30, cols=50):
    """An empty costmap of 0.1 m cells."""
    return np.zeros((rows, cols), np.uint8)


def plan(data, start, goal, **options):
    """The path of an AStar over the grid between two (x, y, angle) poses."""
    planner = AStar(**options)
    planner.configure(Costmap(data, 0.1))
    return planner.plan(pose(*start), pose(*goal))


def cells_of(path):
    """The (row, column) of each point of a path on the test grid."""
    return [(int(y / 0.1), int(x / 0.1)) for x, y, _ in path]


def test_open_ground_gives_the_shortest_path():
    """With nothing in the way the path is the shortest, using diagonals."""
    path = plan(grid(), (0.55, 0.55, 0.0), (3.55, 2.55, 1.0))
    assert path[0, :2] == pytest.approx((0.55, 0.55))
    assert path[-1] == pytest.approx((3.55, 2.55, 1.0))
    length = np.hypot(*np.diff(path[:, :2], axis=0).T).sum()
    assert length == pytest.approx(2 * math.sqrt(2) * 1.0 + 1.0, abs=1e-6)


def test_the_path_goes_round_a_wall():
    """A wall with a gap at one end - the path uses the gap and no blocked cell."""
    data = grid()
    data[:25, 25] = LETHAL
    path = plan(data, (0.55, 0.55), (4.55, 0.55))
    assert len(path) > 0
    assert all(data[r, c] < INSCRIBED for r, c in cells_of(path))
    assert max(r for r, _ in cells_of(path)) >= 25  # it went through the gap
    assert path[-1, 2] == 0.0


def test_four_connectivity_turns_only_by_ninety_degrees():
    """Every step goes along a row or a column."""
    data = grid()
    data[:25, 25] = LETHAL
    path = plan(data, (0.55, 0.55), (4.55, 0.55), connectivity=4)
    middle = path[1:-1, :2]
    steps = np.diff(middle, axis=0)
    assert (np.isclose(steps[:, 0], 0) | np.isclose(steps[:, 1], 0)).all()
    assert len(path) > len(plan(data, (0.55, 0.55), (4.55, 0.55)))


def test_a_diagonal_step_does_not_cut_a_corner():
    """Two blocked cells touching at a corner cannot be slipped between."""
    data = grid(6, 6)
    data[2, 3] = data[3, 2] = LETHAL
    path = plan(data, (0.25, 0.25), (0.35, 0.35))
    assert len(path) > 2  # one diagonal step would give two points


@pytest.mark.parametrize(
    "goal",
    [(2.55, 1.05), (9.0, 1.0), (-1.0, 1.0)],
    ids=["blocked", "off east", "off west"],
)
def test_a_goal_that_cannot_be_reached_gives_no_path(goal):
    """A blocked goal, or one off the map, gives an empty path."""
    data = grid(30, 50)
    data[10, 25] = LETHAL
    assert plan(data, (0.55, 0.55), goal).shape == (0, 3)


def test_a_start_off_the_map_gives_no_path():
    """No cell holds the start."""
    assert plan(grid(), (-1.0, 0.5), (1.0, 1.0)).shape == (0, 3)


def test_a_goal_walled_in_gives_no_path():
    """The goal is free but every way to it is blocked."""
    data = grid()
    data[8:13, 8:13] = LETHAL
    data[10, 10] = 0
    assert plan(data, (0.55, 0.55), (1.05, 1.05)).shape == (0, 3)


def test_a_start_in_a_blocked_cell_still_plans():
    """A robot against a wall can sit in a cell that is blocked."""
    data = grid()
    data[0, 0] = INSCRIBED
    path = plan(data, (0.05, 0.05), (1.55, 0.55))
    assert len(path) > 1 and path[-1, 0] == pytest.approx(1.55)


def test_a_start_in_the_goal_cell_gives_two_points():
    """Nothing to search: the path is the start, then the goal."""
    path = plan(grid(), (1.01, 1.01, 0.3), (1.09, 1.04, 0.7))
    assert path.shape == (2, 3)
    assert path[-1, 2] == 0.7


def test_the_path_keeps_off_costly_cells():
    """The short way round a block crosses a costly strip; a weight avoids it."""
    data = grid(30, 50)
    data[10:25, 20:30] = LETHAL
    data[8:10, 20:30] = 200
    start, goal = (1.55, 1.25), (4.55, 1.25)

    def cost(path):
        return sum(int(data[r, c]) for r, c in cells_of(path))

    assert cost(plan(data, start, goal, cost_weight=2.0)) < cost(
        plan(data, start, goal, cost_weight=0.0)
    )


def test_a_run_repeats():
    """The same inputs give the same path, equal costs and all."""
    data = grid()
    data[:25, 25] = LETHAL
    first = plan(data, (0.55, 0.55), (4.55, 0.55))
    second = plan(data, (0.55, 0.55), (4.55, 0.55))
    assert np.array_equal(first, second)


def test_the_costmap_is_required():
    """A world with no costmap is refused, and so is planning before configure."""
    planner = AStar()
    with pytest.raises(ValueError, match="needs a costmap"):
        planner.configure(None)
    with pytest.raises(ValueError, match="not configured"):
        planner.plan(pose(0, 0), pose(1, 1))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"connectivity": 5}, "connectivity"),
        ({"cost_weight": -1.0}, "cost_weight"),
        ({"cost_weight": math.nan}, "cost_weight"),
        ({"cost_weight": True}, "cost_weight"),
    ],
)
def test_bad_settings_are_refused(options, message):
    """A bad setting is refused when the planner is built."""
    with pytest.raises(ValueError, match=message):
        AStar(**options)


@pytest.mark.parametrize(
    ("start", "goal", "message"),
    [
        (
            (0.55, 0.55),
            (2.55, 1.05),
            "AStar: the goal cell (10, 25) is blocked (cost 254), "
            "so the robot's centre cannot be there",
        ),
        ((0.55, 0.55), (9.0, 1.0), "AStar: the goal (9.00, 1.00) is off the map"),
        ((-1.0, 0.5), (1.0, 1.0), "AStar: the start (-1.00, 0.50) is off the map"),
        (
            (0.55, 0.55),
            (1.05, 1.05),
            "AStar: no way from the start (5, 5) to the goal (10, 10); "
            "one of them is walled in by blocked cells",
        ),
    ],
    ids=["blocked goal", "goal off the map", "start off the map", "walled in"],
)
def test_no_path_says_why(caplog, start, goal, message):
    """Each reason for an empty path is logged once, in words."""
    data = grid()
    data[10, 25] = LETHAL
    data[8:13, 8:13] = LETHAL
    data[10, 10] = 0
    with caplog.at_level(logging.WARNING):
        assert plan(data, start, goal).shape == (0, 3)
    assert caplog.messages == [message]
