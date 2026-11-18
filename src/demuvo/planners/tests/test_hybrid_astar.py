"""Tests for the Hybrid A* planner."""

import logging
import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Costmap
from demuvo.costmap import LETHAL
from demuvo.planners import HybridAStar


def pose(x, y, yaw=0.0):
    """A pose from (x, y, angle)."""
    return Transform2D.from_xyt((x, y, yaw))


def grid(rows=30, cols=50):
    """An empty costmap of 0.1 m cells, 3 m by 5 m."""
    return np.zeros((rows, cols), np.uint8)


def plan(data, start, goal, **options):
    """The path of a HybridAStar over the grid between two (x, y, angle) poses."""
    planner = HybridAStar(**options)
    planner.configure(Costmap(data, 0.1))
    return planner.plan(pose(*start), pose(*goal))


def touches_blocked(data, path):
    """True if a point of the path lies on a blocked cell."""
    rows = (path[:, 1] / 0.1).astype(int)
    cols = (path[:, 0] / 0.1).astype(int)
    return bool((data[rows, cols] >= 253).any())


def length(path):
    """The length of a path."""
    return float(np.hypot(*np.diff(path[:, :2], axis=0).T).sum())


def test_open_ground_gives_a_path_that_ends_on_the_goal_pose():
    """The path starts at the start, ends at the goal's pose and bends gently."""
    path = plan(grid(), (0.5, 0.5, 0.0), (4.0, 2.0, 1.57), turning_radius=0.5)
    assert path[0] == pytest.approx((0.5, 0.5, 0.0))
    assert path[-1] == pytest.approx((4.0, 2.0, 1.57))
    turn = np.abs(np.arctan2(np.sin(np.diff(path[:, 2])), np.cos(np.diff(path[:, 2]))))
    step = np.hypot(*np.diff(path[:, :2], axis=0).T)
    assert (turn <= step / 0.5 * 1.05 + 1e-6).all()  # never tighter than the radius


def test_the_path_goes_round_a_wall_that_the_straight_way_would_hit():
    """A wall across the middle with a gap at the top: the path uses the gap."""
    data = grid()
    data[0:20, 25] = LETHAL
    path = plan(data, (1.0, 0.5, 0.0), (4.0, 0.5, 0.0), turning_radius=0.4)
    assert len(path) > 2 and not touches_blocked(data, path)
    assert path[:, 1].max() > 2.0
    assert path[-1] == pytest.approx((4.0, 0.5, 0.0))


def test_backing_up_gives_a_shorter_path_to_a_goal_behind():
    """Forward only it loops round; with allow_reversing it backs up."""
    forward = plan(grid(), (2.0, 1.5, 0.0), (1.5, 1.5, 0.0), turning_radius=0.5)
    backward = plan(
        grid(),
        (2.0, 1.5, 0.0),
        (1.5, 1.5, 0.0),
        turning_radius=0.5,
        allow_reversing=True,
    )
    assert length(backward) < length(forward) / 2
    step = np.diff(backward[:, :2], axis=0)
    along = (
        np.cos(backward[:-1, 2]) * step[:, 0] + np.sin(backward[:-1, 2]) * step[:, 1]
    )
    assert (along < 0).any()  # some of it is driven backwards


def test_a_search_with_arcs_round_a_corner_uses_reverse_when_it_helps():
    """A dead end only a reversing car gets out of: the path has both directions."""
    data = grid()
    data[10:20, 15] = LETHAL
    data[10, 15:25] = LETHAL
    data[19, 15:25] = LETHAL
    path = plan(
        data,
        (2.0, 1.5, 0.0),
        (1.0, 1.5, 0.0),
        turning_radius=0.4,
        allow_reversing=True,
    )
    assert len(path) > 2 and not touches_blocked(data, path)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"turning_radius": 0.0}, "turning_radius"),
        ({"spacing": -1.0}, "spacing"),
        ({"cost_weight": -1.0}, "cost_weight"),
        ({"reverse_penalty": 0.5}, "reverse_penalty"),
        ({"reverse_penalty": "x"}, "reverse_penalty"),
        ({"max_expansions": 0}, "max_expansions"),
        ({"max_expansions": 2.5}, "max_expansions"),
        ({"max_expansions": True}, "max_expansions"),
    ],
)
def test_bad_settings_are_refused_when_the_planner_is_built(kwargs, match):
    """Each message names the setting."""
    with pytest.raises(ValueError, match=match):
        HybridAStar(**kwargs)


def test_a_planner_without_a_costmap_says_so():
    """It cannot work blind, and it must be configured before it plans."""
    with pytest.raises(ValueError, match="needs a costmap"):
        HybridAStar().configure(None)
    with pytest.raises(ValueError, match="not configured"):
        HybridAStar().plan(pose(1, 1), pose(2, 2))


@pytest.mark.parametrize(
    "start, goal, words",
    [
        ((-1.0, 1.0, 0.0), (2.0, 1.0, 0.0), "start"),
        ((1.0, 1.0, 0.0), (9.0, 1.0, 0.0), "goal"),
    ],
)
def test_a_start_or_goal_off_the_map_gives_no_path(caplog, start, goal, words):
    """The message says which one."""
    with caplog.at_level(logging.WARNING):
        path = plan(grid(), start, goal)
    assert path.shape == (0, 3) and f"the {words}" in caplog.text


def test_a_blocked_goal_gives_no_path(caplog):
    """The car's centre cannot be in a blocked cell."""
    data = grid()
    data[10, 30] = LETHAL
    with caplog.at_level(logging.WARNING):
        path = plan(data, (1.0, 1.0, 0.0), (3.05, 1.05, 0.0))
    assert path.shape == (0, 3) and "is blocked" in caplog.text


def test_a_goal_walled_in_gives_no_path_and_says_how_many_poses_were_opened(caplog):
    """The search ends when it has nothing left to open."""
    data = grid()
    data[8:13, 28:33] = LETHAL
    data[9:12, 29:32] = 0
    with caplog.at_level(logging.WARNING):
        path = plan(data, (1.0, 1.0, 0.0), (3.05, 1.05, 0.0), turning_radius=0.4)
    assert path.shape == (0, 3)
    assert "no path from the start" in caplog.text
    assert "stopped there" not in caplog.text


def test_a_search_that_runs_out_of_expansions_says_so(caplog):
    """A wall to go round and one pose to open gives up at once."""
    data = grid()
    data[0:25, 25] = LETHAL
    with caplog.at_level(logging.WARNING):
        path = plan(
            data, (1.0, 0.5, 0.0), (4.0, 0.5, 0.0), turning_radius=0.4, max_expansions=1
        )
    assert path.shape == (0, 3) and "and stopped there" in caplog.text


def test_a_start_in_a_blocked_cell_is_allowed_but_no_arc_leaves_it(caplog):
    """As for A*, the start may be against a wall; here it is inside one."""
    data = grid()
    data[5:8, 5:8] = LETHAL
    with caplog.at_level(logging.WARNING):
        path = plan(data, (0.65, 0.65, 0.0), (3.0, 2.0, 0.0), turning_radius=0.4)
    assert path.shape == (0, 3)


def test_the_costs_of_a_step_follow_the_costmap_and_a_turn_or_cusp_costs_more():
    """Walls raise the price of an arc; changing direction or steering adds to it."""
    from demuvo.planners import hybrid_astar  # pylint: disable=import-outside-toplevel

    # pylint: disable=protected-access

    assert hybrid_astar._change(None, (1, 0.0)) == 0.0
    assert hybrid_astar._change((1, 0.0), (1, 0.0)) == 0.0
    assert hybrid_astar._change((1, 0.0), (1, 1.0)) == hybrid_astar.TURN
    assert hybrid_astar._change((1, 0.0), (-1, 0.0)) == hybrid_astar.CUSP
    assert math.isinf(hybrid_astar._Far()[(1, 1)])


@pytest.mark.parametrize(
    "start",
    [(0.1, 0.1, math.pi), (4.9, 2.9, 0.0), (0.1, 2.9, 1.57), (4.9, 0.1, -1.57)],
)
def test_arcs_that_leave_the_map_are_dropped_and_a_car_backs_out_of_the_edge(start):
    """A car at an edge facing outwards cannot go on, so it backs onto the map."""
    path = plan(
        grid(), start, (2.5, 1.5, 0.0), turning_radius=0.4, allow_reversing=True
    )
    assert len(path) > 2 and path[-1] == pytest.approx((2.5, 1.5, 0.0))
    assert path[:, 0].min() >= 0.0 and path[:, 0].max() <= 5.0
    assert path[:, 1].min() >= 0.0 and path[:, 1].max() <= 3.0


def test_points_past_any_edge_of_the_map_are_not_clear():
    """Past the low edges, the high edges, or on a blocked cell."""
    planner = HybridAStar()
    data = grid()
    data[10, 10] = LETHAL
    planner.configure(Costmap(data, 0.1))
    inside = np.array([[2.0, 1.0], [2.5, 1.0]])
    assert planner._clear(inside) is not None  # pylint: disable=protected-access
    for points in (
        [[-0.1, 1.0]],
        [[1.0, -0.1]],
        [[5.1, 1.0]],
        [[1.0, 3.1]],
        [[1.05, 1.05]],
    ):
        assert planner._clear(np.array(points)) is None  # pylint: disable=protected-access
