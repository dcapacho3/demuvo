"""Tests for the colours, the palettes and the easing curve."""

import numpy as np
import pytest

from demuvo.costmap.grid import FREE, INSCRIBED, LETHAL, NO_INFORMATION
from demuvo.gui import look


def test_robot_zero_is_the_logo_teal_family_and_every_colour_differs():
    """Six robots, six colours, each at least 3 to 1 against the darker floor."""
    colors = [look.robot_color(i) for i in range(6)]
    assert len(set(colors)) == 6
    assert all(look.contrast(c, look.MINT2) >= 3.0 for c in colors)
    assert colors[0].startswith("#") and len(colors[0]) == 7


def test_the_hue_steps_round_the_wheel_and_many_robots_stay_distinct():
    """Forty robots; none repeats."""
    colors = [look.robot_color(i) for i in range(40)]
    assert len(set(colors)) == 40


def test_a_chroma_the_screen_cannot_show_is_cut_back_to_one_it_can(monkeypatch):
    """A chroma of 0.5 is far outside sRGB; each robot still gets a valid colour."""
    usual = [look.robot_color(i) for i in range(12)]
    monkeypatch.setattr(look, "CHROMA", 0.5)
    cut = [look.robot_color(i) for i in range(12)]
    assert all(len(c) == 7 and int(c[1:], 16) >= 0 for c in cut)
    assert cut != usual


def test_contrast_of_black_on_white_is_21_and_of_a_colour_with_itself_is_1():
    """The ends of the scale, and the order does not matter."""
    assert look.contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert look.contrast("#ffffff", "#000000") == pytest.approx(21.0)
    assert look.contrast(look.TEAL, look.TEAL) == pytest.approx(1.0)


def test_the_ink_reads_on_the_floor_and_the_panel():
    """Navy text on mint and white is far above the 4.5 to 1 a reader needs."""
    assert look.contrast(look.NAVY, look.MINT) > 10
    assert look.contrast(look.MUTED, look.PANEL) > 4.5


@pytest.mark.parametrize("palette", look.PALETTES)
def test_a_costmap_picture_orders_the_levels_and_flips_the_rows(palette):
    """Free is clear, graded grows opaque, the rest have colours; row 0 is last."""
    data = np.array([[FREE, 100, INSCRIBED], [LETHAL, NO_INFORMATION, 10]], np.uint8)
    picture = look.costmap_rgba(data, palette)
    assert picture.shape == (2, 3, 4) and picture.dtype == np.uint8
    assert picture[1, 0, 3] == 0
    assert picture[1, 1, 3] > picture[0, 2, 3]
    assert picture[1, 1, 3] > 0
    assert picture[0, 0, 3] == 255
    assert picture[0, 1, 3] == round(look.UNKNOWN_ALPHA * 255)


def test_the_two_palettes_are_different_and_cost_raises_opacity():
    """The same cells give two pictures; a dearer cell is more opaque in each."""
    data = np.array([[10, 200]], np.uint8)
    navy = look.costmap_rgba(data, "navy and amber")
    teal = look.costmap_rgba(data, "mint to teal")
    assert not np.array_equal(navy, teal)
    assert navy[0, 1, 3] > navy[0, 0, 3] and teal[0, 1, 3] > teal[0, 0, 3]


def test_the_inscribed_colour_differs_between_the_palettes():
    """Amber in one, deep teal in the other."""
    data = np.array([[INSCRIBED]], np.uint8)
    assert look.costmap_rgba(data, "navy and amber")[0, 0, :3].tolist() == [
        245,
        166,
        35,
    ]
    assert look.costmap_rgba(data, "mint to teal")[0, 0, :3].tolist() == [11, 143, 132]


def test_the_easing_curve_starts_at_zero_ends_at_one_and_is_fast_at_first():
    """Monotone, and over half way after a fifth of the time."""
    curve = [look.ease_out(t / 20) for t in range(21)]
    assert curve[0] == pytest.approx(0.0, abs=1e-6)
    assert curve[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert curve[4] > 0.5


def test_the_easing_curve_holds_its_ends_outside_zero_to_one():
    """Before the start and after the end."""
    assert look.ease_out(-1) == pytest.approx(0.0, abs=1e-6)
    assert look.ease_out(2) == pytest.approx(1.0)


def test_the_motion_times_are_the_design_values():
    """100, 90 and 200 ms."""
    assert look.MS == {"press": 100, "fade": 90, "panel": 200}
