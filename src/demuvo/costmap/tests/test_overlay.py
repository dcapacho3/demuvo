"""Tests for laying one costmap over another."""

import numpy as np

from demuvo.costmap import LETHAL, Costmap, overlay


def grid(shape, resolution, origin=(0.0, 0.0)):
    """A free costmap."""
    return Costmap(np.zeros(shape, np.uint8), resolution, origin)


def test_nothing_to_lay_over_gives_the_base_itself():
    """None on either side, or a window off the base, changes nothing."""
    base = grid((10, 10), 0.5)
    assert overlay(None, grid((2, 2), 0.5)) is None
    assert overlay(base, None) is base
    assert overlay(base, grid((2, 2), 0.5, (20.0, 20.0))) is base


def test_the_higher_cost_wins_and_the_base_is_left_as_it_was():
    """A mark in the window blocks the base; a dearer base cell keeps its cost."""
    base = grid((10, 10), 0.5)
    base.data[2, 2] = 100
    base.data[2, 3] = 200
    window = grid((4, 4), 0.5, (1.0, 1.0))
    window.data[0, 0] = LETHAL  # the cell at (1.0..1.5, 1.0..1.5): base cell (2, 2)
    window.data[0, 1] = 50  # base cell (2, 3)
    merged = overlay(base, window)
    assert merged is not base
    assert (merged.resolution, merged.origin) == (base.resolution, base.origin)
    assert merged.data[2, 2] == LETHAL and merged.data[2, 3] == 200
    assert merged.data.sum() == LETHAL + 200
    assert base.data[2, 2] == 100  # the base was copied, not written


def test_grids_of_different_resolution_and_origin_meet_cell_by_cell():
    """A coarse window cell covers every fine base cell whose centre lies in it."""
    base = grid((40, 40), 0.25, (-1.0, -1.0))
    window = grid((3, 3), 1.0, (0.3, 0.3))
    window.data[1, 1] = LETHAL  # the square 1.3..2.3 on both axes
    merged = overlay(base, window)
    rows, cols = np.nonzero(merged.data)
    xs, ys = base.centres(rows, cols)
    assert len(rows) == 16
    assert (
        (xs > 1.3).all() and (xs < 2.3).all() and (ys > 1.3).all() and (ys < 2.3).all()
    )


def test_a_window_that_hangs_off_the_base_marks_only_the_part_on_it():
    """The cells outside the base are dropped without an error."""
    base = grid((4, 4), 1.0)
    window = grid((4, 4), 1.0, (2.0, 2.0))
    window.data[:] = LETHAL
    merged = overlay(base, window)
    assert merged.data.tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, LETHAL, LETHAL],
        [0, 0, LETHAL, LETHAL],
    ]
