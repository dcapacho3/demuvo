"""Tests for the cells a line crosses, and for the centres of cells."""

import numpy as np
import pytest

from demuvo.costmap import Costmap


def cells(costmap, a, b):
    """The (row, column) pairs a line from a to b crosses, in order."""
    rows, cols = costmap.line_cells(np.array(a, dtype=float), np.array(b, dtype=float))
    return list(zip(rows.tolist(), cols.tolist()))


def unit():
    """A costmap of 10 by 10 cells, each 1 m."""
    return Costmap(np.zeros((10, 10), np.uint8), 1.0)


def test_a_horizontal_line_crosses_one_row_of_cells():
    """From the middle of column 0 to the middle of column 4."""
    assert cells(unit(), (0.5, 2.5), (4.5, 2.5)) == [(2, c) for c in range(5)]


def test_a_line_going_down_lists_its_cells_from_the_start():
    """The order follows the line, from a to b."""
    assert cells(unit(), (1.5, 5.5), (1.5, 1.5)) == [(r, 1) for r in (5, 4, 3, 2, 1)]


def test_a_shallow_line_steps_to_the_next_row_where_it_crosses_the_edge():
    """Cells in order: it leaves the first row at x = 1.75."""
    assert cells(unit(), (0.5, 0.5), (3.5, 1.7)) == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (1, 3),
    ]


def test_a_line_through_a_corner_also_gives_the_cells_that_touch_it_there():
    """A diagonal through cell centres passes the corners and gives their neighbours."""
    got = set(cells(unit(), (0.5, 0.5), (3.5, 3.5)))
    diagonal = {(i, i) for i in range(4)}
    beside = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    assert got == diagonal | beside


def test_a_line_of_no_length_is_one_cell():
    """Both ends are the same point."""
    assert cells(unit(), (2.5, 3.5), (2.5, 3.5)) == [(3, 2)]


def test_a_line_may_leave_the_grid():
    """The cells outside are listed, with negative columns."""
    assert cells(unit(), (0.5, 0.5), (-2.5, 0.5)) == [(0, 0), (0, -1), (0, -2), (0, -3)]


def test_the_origin_and_the_cell_size_are_taken_into_account():
    """Cells of 0.5 m starting at (1, 2): the line runs along the first row."""
    costmap = Costmap(np.zeros((4, 6), np.uint8), 0.5, (1.0, 2.0))
    assert cells(costmap, (1.25, 2.25), (3.25, 2.25)) == [(0, c) for c in range(5)]


def test_the_centres_of_cells_are_found_for_one_cell_or_many():
    """Cell (1, 2) of 0.5 m cells starting at (1, 2) is centred at (2.25, 2.75)."""
    costmap = Costmap(np.zeros((4, 6), np.uint8), 0.5, (1.0, 2.0))
    assert costmap.cell_to_world(1, 2) == pytest.approx((2.25, 2.75))
    x, y = costmap.centres(np.array([0, 1]), np.array([0, 2]))
    assert x == pytest.approx([1.25, 2.25])
    assert y == pytest.approx([2.25, 2.75])
