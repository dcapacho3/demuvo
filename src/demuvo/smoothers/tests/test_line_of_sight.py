"""Tests for the line-of-sight smoother."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import AStar, Costmap
from demuvo.costmap import INSCRIBED, LETHAL
from demuvo.smoothers.line_of_sight import LineOfSight


def grid(rows=30, cols=50):
    """An empty costmap of 0.1 m cells."""
    return np.zeros((rows, cols), np.uint8)


def smoother(data, **options):
    """A LineOfSight over the grid."""
    result = LineOfSight(**options)
    result.configure(Costmap(data, 0.1))
    return result


def planned(data, start, goal, **options):
    """The path an AStar finds between two (x, y, angle) poses."""
    planner = AStar(**options)
    planner.configure(Costmap(data, 0.1))
    return planner.plan(Transform2D.from_xyt(start), Transform2D.from_xyt(goal))


def poses(points, yaw=0.0):
    """A path of (x, y) points as poses, all with the same angle."""
    return np.array([(x, y, yaw) for x, y in points], dtype=float)


def length(path):
    """The length of a path in metres."""
    return np.hypot(*np.diff(path[:, :2], axis=0).T).sum()


def free(data, path):
    """True if no point of the path lies in a cell that is blocked."""
    cells = [(int(y / 0.1), int(x / 0.1)) for x, y, _ in path]
    return all(data[r, c] < INSCRIBED for r, c in cells[1:])


def test_a_staircase_in_open_ground_becomes_one_straight_line():
    """A four-neighbour path of right angles in free space shrinks to the diagonal."""
    data = grid()
    path = planned(data, (0.55, 0.55, 0.0), (3.55, 2.55, 0.5), connectivity=4)
    out = smoother(data).smooth(path)
    assert out[0, :2] == pytest.approx((0.55, 0.55))
    assert out[-1] == pytest.approx((3.55, 2.55, 0.5))
    line = np.array([out[-1, 0] - out[0, 0], out[-1, 1] - out[0, 1]])
    off = (out[:, 0] - out[0, 0]) * line[1] - (out[:, 1] - out[0, 1]) * line[0]
    assert np.abs(off).max() == pytest.approx(0.0, abs=1e-9)


def test_the_result_is_spaced_and_points_along_its_steps():
    """Points are at most `spacing` apart; each looks along its step."""
    out = smoother(grid(), spacing=0.25).smooth(poses([(0.5, 0.5), (1.5, 0.5)], 0.3))
    assert len(out) == 5
    assert out[:, 0] == pytest.approx([0.5, 0.75, 1.0, 1.25, 1.5])
    assert out[:-1, 2] == pytest.approx(0.0)
    assert out[-1, 2] == pytest.approx(0.3)


def test_a_path_round_a_wall_keeps_its_corner_and_crosses_nothing_blocked():
    """The straight lines still go through the gap, never through the wall."""
    data = grid()
    data[:25, 25] = LETHAL
    path = planned(data, (0.55, 0.55, 0.0), (4.55, 0.55, 0.0))
    out = smoother(data).smooth(path)
    assert free(data, out)
    assert max(int(y / 0.1) for _, y, _ in out) >= 25  # it still uses the gap
    assert length(out) <= length(path) + 1e-9
    assert out[-1, 0] == pytest.approx(4.55)


def test_it_finds_a_clear_line_beyond_one_that_is_blocked():
    """The line to the third point crosses a wall; the one to the fourth does not."""
    data = grid()
    data[10:15, 20] = LETHAL
    path = poses([(1.05, 1.25), (1.05, 0.55), (3.05, 1.25), (3.05, 2.85)])
    out = smoother(data).smooth(path)
    assert length(out) == pytest.approx(math.hypot(2.0, 1.6), abs=1e-9)


def test_a_shortcut_never_moves_onto_dearer_ground_than_the_stretch_it_replaces():
    """A path above a costly strip stays above it; a path through it may go straight."""
    data = grid()
    data[10:15, 10:40] = 100

    above = poses([(0.55, 0.55), (0.55, 1.55), (4.55, 1.55)])
    kept = smoother(data).smooth(above)
    assert all(data[int(y / 0.1), int(x / 0.1)] == 0 for x, y, _ in kept)
    through = poses(
        [(0.55, 0.55), (0.55, 1.25), (1.55, 1.25), (3.55, 1.25), (4.55, 1.25)]
    )
    straight = smoother(data).smooth(through)
    assert length(straight) < length(through) - 0.5


def test_a_line_through_the_corner_of_two_blocked_cells_is_refused():
    """Two blocked cells that touch at a corner cannot be squeezed between."""
    data = grid(6, 6)
    data[2, 3] = data[3, 2] = LETHAL
    # the line from the first to the last point runs through their shared corner
    out = smoother(data).smooth(poses([(0.25, 0.25), (0.25, 0.45), (0.35, 0.35)]))
    assert any(np.allclose(point[:2], (0.25, 0.45)) for point in out)


def test_a_start_in_a_blocked_cell_still_works():
    """The planner allows it, so the smoother does too."""
    data = grid()
    data[0, 0] = INSCRIBED
    out = smoother(data).smooth(poses([(0.05, 0.05), (0.15, 0.15), (0.55, 0.55)]))
    assert out[0, :2] == pytest.approx((0.05, 0.05))
    assert out[-1, :2] == pytest.approx((0.55, 0.55))


def test_a_path_that_leaves_the_map_is_refused():
    """A point off the costmap is named, not looked up."""
    with pytest.raises(ValueError, match="leaves the costmap at point 1"):
        smoother(grid(5, 5)).smooth(poses([(0.15, 0.15), (0.65, 0.15), (0.45, 0.45)]))


def test_a_run_repeats():
    """The same path gives the same result."""
    data = grid()
    data[:25, 25] = LETHAL
    path = planned(data, (0.55, 0.55, 0.0), (4.55, 0.55, 0.0))
    assert np.array_equal(smoother(data).smooth(path), smoother(data).smooth(path))


def test_the_costmap_is_required():
    """A world with no costmap is refused, and so is smoothing before configure."""
    result = LineOfSight()
    with pytest.raises(ValueError, match="needs a costmap"):
        result.configure(None)
    with pytest.raises(ValueError, match="not configured"):
        result.smooth(poses([(0, 0), (1, 1)]))


@pytest.mark.parametrize("spacing", [0.0, -0.1, math.nan, True, "0.1"])
def test_the_spacing_must_be_a_positive_number(spacing):
    """A bad spacing is refused when the smoother is built."""
    with pytest.raises(ValueError, match="spacing"):
        LineOfSight(spacing=spacing)
