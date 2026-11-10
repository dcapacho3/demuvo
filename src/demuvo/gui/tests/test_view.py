"""Tests for the mapping between world metres and window pixels."""

import numpy as np
import pytest

from demuvo.gui.view import MARGIN, View

BOUNDS = (0.0, 0.0, 10.0, 6.0)


def test_a_fitted_view_shows_the_whole_board_inside_its_area_and_centred():
    """The board and its margin fill the tighter side; the rest is shared evenly."""
    view = View(BOUNDS).fit((100, 50, 800, 600), bottom_room=60)
    left, top, right, bottom = view.board()
    assert view.scale == pytest.approx(min(800 / 10.7, 540 / 6.7))
    assert right - left == pytest.approx(10 * view.scale)
    assert bottom - top == pytest.approx(6 * view.scale)
    assert left >= 100 and right <= 900 and top >= 50 and bottom <= 650 - 60
    assert view.rect == (100, 50, 800, 600)


def test_the_margin_is_kept_on_every_side_of_the_board():
    """With a tight area the board sits MARGIN metres from the edges."""
    area = (0, 0, 10.7 * 40, 6.7 * 40 + 60)
    view = View(BOUNDS).fit(area)
    left, top, right, bottom = view.board()
    assert view.scale == pytest.approx(40)
    assert left == pytest.approx(MARGIN * 40) and top == pytest.approx(MARGIN * 40)
    assert right == pytest.approx(10.7 * 40 - MARGIN * 40)
    assert bottom == pytest.approx(6.7 * 40 - MARGIN * 40)


def test_world_y_runs_up_and_screen_y_runs_down():
    """The world origin is the board's bottom-left; a larger y is higher up."""
    view = View(BOUNDS).fit((0, 0, 500, 400))
    left, top, right, bottom = view.board()
    assert view.to_screen(0, 0) == (pytest.approx(left), pytest.approx(bottom))
    x, y = view.to_screen(10, 6)
    assert (float(x), float(y)) == pytest.approx((right, top))


def test_a_pixel_maps_back_to_the_world_point_it_came_from():
    """to_world undoes to_screen, for a number and for an array."""
    view = View(BOUNDS).fit((20, 30, 640, 480))
    sx, sy = view.to_screen(3.25, 4.5)
    assert view.to_world(float(sx), float(sy)) == pytest.approx((3.25, 4.5))
    xs, ys = view.to_screen(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert xs.shape == (2,) and ys[0] > ys[1]


def test_a_board_that_does_not_start_at_zero_is_placed_by_its_bounds():
    """Bounds from (-2, -1) to (3, 3): the lower-left corner is bottom-left."""
    view = View((-2.0, -1.0, 3.0, 3.0)).fit((0, 0, 400, 400))
    left, top, right, bottom = view.board()
    x, y = view.to_screen(-2, -1)
    assert (float(x), float(y)) == pytest.approx((left, bottom))
    assert view.to_world(left, bottom) == pytest.approx((-2.0, -1.0))
    assert view.to_world(right, top) == pytest.approx((3.0, 3.0))


def test_a_shifted_view_moves_every_pixel_by_the_same_amount():
    """A cut-out whose corner is (40, 30) uses the view shifted by minus that."""
    view = View(BOUNDS).fit((40, 30, 500, 400))
    cut = view.shifted(-40, -30)
    a = view.to_screen(2.0, 3.0)
    b = cut.to_screen(2.0, 3.0)
    assert (float(a[0]) - 40, float(a[1]) - 30) == pytest.approx(
        (float(b[0]), float(b[1]))
    )
    assert cut.scale == view.scale
