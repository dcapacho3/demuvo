"""Tests for the grid edits the styles share."""

import numpy as np
import pytest

from demuvo.generate.carve import (
    carve_box,
    carve_lane,
    cell_at,
    cells_of,
    clear_border,
    clear_disc,
    fill_border,
    majority,
)


def blocked(rows=10, cols=10):
    """A grid that is blocked everywhere."""
    return np.ones((rows, cols), dtype=bool)


def test_a_border_is_cleared_and_filled_from_the_outside_in():
    """Two rings of a ten by ten grid leave a six by six middle."""
    mask = blocked()
    clear_border(mask, 2)
    assert mask.sum() == 36 and mask[2:8, 2:8].all()
    free = np.zeros((10, 10), dtype=bool)
    fill_border(free, 2)
    assert free.sum() == 100 - 36


def test_a_disc_is_cleared_round_a_place_in_the_world():
    """The cells whose centres lie within the radius, and no others."""
    mask = blocked(20, 20)
    clear_disc(mask, (0.0, 0.0), 0.1, (1.0, 1.0), 0.25)
    assert not mask[10, 10]  # the cell the point is in
    assert mask[5, 5]  # and a cell well away from it
    cleared = np.argwhere(~mask)
    away = np.hypot(cleared[:, 0] + 0.5 - 10, cleared[:, 1] + 0.5 - 10) * 0.1
    assert away.max() <= 0.25


def test_a_disc_that_misses_the_grid_changes_nothing():
    """A point west of the world clears none of it, and does not wrap around."""
    mask = blocked()
    clear_disc(mask, (0.0, 0.0), 0.1, (-5.0, -5.0), 0.2)
    assert mask.all()


def test_a_box_is_carved_and_clipped_to_what_is_inside():
    """A rectangle that reaches past the edge is cut off at it."""
    mask = blocked()
    carve_box(mask, (-5, -5, 3, 3), (1, 1, 9, 9))
    assert not mask[1:3, 1:3].any() and mask[0, :].all()
    before = mask.copy()
    carve_box(mask, (20, 20, 30, 30), (1, 1, 9, 9))  # wholly outside
    assert np.array_equal(mask, before)


def test_a_lane_turns_its_corner_one_way_or_the_other():
    """The two Ls between the same ends cover the two different corners."""
    one, two = blocked(20, 20), blocked(20, 20)
    inside = (0, 0, 20, 20)
    carve_lane(one, (2, 2), (12, 16), 3, inside, True)
    carve_lane(two, (2, 2), (12, 16), 3, inside, False)
    assert not one[2, 2] and not one[12, 16]  # both ends on either route
    assert not two[2, 2] and not two[12, 16]
    assert not np.array_equal(one, two)
    assert not one[2, 16] and two[2, 16]  # the corner each one turns


def test_a_lane_is_as_wide_as_it_is_asked_to_be():
    """A straight run of width three clears three rows."""
    mask = blocked(20, 20)
    carve_lane(mask, (10, 2), (10, 15), 3, (0, 0, 20, 20), True)
    assert not mask[9:12, 2:15].any()
    assert mask[8, 2:15].all() and mask[12, 2:15].all()


def test_the_majority_rule_fills_a_nick_and_rubs_out_a_speck():
    """Most of the nine cells round it decides, and off the grid counts as free."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False  # a nick in the middle of a block
    mask[6, 6] = True  # a speck in a corner
    after = majority(mask)
    assert after[3, 3]  # the nick is filled
    assert not after[6, 6]  # the speck is gone


@pytest.mark.parametrize(
    "metres, resolution, want",
    [(0.7, 0.1, 7), (0.05, 0.1, 1), (0.0, 0.1, 1), (0.25, 0.1, 3)],
)
def test_a_length_covers_whole_cells_and_never_none(metres, resolution, want):
    """Half a cell is still a cell, and nothing is still one cell."""
    assert cells_of(metres, resolution) == want


def test_a_point_of_the_world_falls_in_the_cell_that_holds_it():
    """Row from y and column from x, counted off the origin."""
    assert cell_at((1.0, 2.0), (0.0, 0.0), 0.1) == (20, 10)
    assert cell_at((-4.95, -2.95), (-5.0, -3.0), 0.1) == (0, 0)
