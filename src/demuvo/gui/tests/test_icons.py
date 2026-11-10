"""Tests for the icons and the key codes."""

import numpy as np
import pytest

from demuvo.gui import icons, keys
from demuvo.gui.raster import Canvas


def ink(canvas):
    """The darkness drawn on a white canvas."""
    return float((1.0 - canvas.px[..., 0]).sum())


@pytest.mark.parametrize("name", [*icons.ICONS, icons.RESET])
def test_every_icon_draws_inside_its_square(name):
    """Some ink, and none outside the 24 pixel square at the centre of the canvas."""
    canvas = Canvas(60, 60)
    icons.draw_icon(canvas, name, (30, 30), 24, "#0f2742")
    assert ink(canvas) > 5
    outside = np.ones((60, 60), bool)
    outside[16:44, 16:44] = False
    assert canvas.px[outside].min() > 0.95


def test_an_icon_scales_with_its_size():
    """Twice the size, about four times the ink."""
    small, large = Canvas(80, 80), Canvas(80, 80)
    icons.draw_icon(small, "play", (40, 40), 20, "#000000")
    icons.draw_icon(large, "play", (40, 40), 40, "#000000")
    assert ink(large) == pytest.approx(4 * ink(small), rel=0.25)


def test_an_icon_takes_its_colour_and_opacity():
    """Teal at half opacity on white is lighter than at full."""
    full, half = Canvas(40, 40), Canvas(40, 40)
    icons.draw_icon(full, "check", (20, 20), 24, "#1fb5a6")
    icons.draw_icon(half, "check", (20, 20), 24, "#1fb5a6", opacity=0.5)
    assert full.px[..., 0].min() < 0.2
    assert ink(half) < ink(full)


def test_a_part_of_each_kind_is_drawn(monkeypatch):
    """The listed icons have no disc, so one is added to cover that part."""
    monkeypatch.setitem(icons.ICONS, "dot", [("disc", 12, 12, 4)])
    canvas = Canvas(40, 40)
    icons.draw_icon(canvas, "dot", (20, 20), 24, "#000000")
    assert canvas.px[20, 20, 0] == 0.0


def test_an_unknown_icon_is_a_key_error():
    """The name is the key."""
    with pytest.raises(KeyError):
        icons.draw_icon(Canvas(10, 10), "nothing", (5, 5), 8, "#000000")


def test_key_codes_are_glfws():
    """A few the toolkit uses, letters in lower case, digits, and the modifier bits."""
    assert keys.KEY["enter"] == 257 and keys.KEY["a"] == 65 and keys.KEY["9"] == 57
    assert keys.KEY["left"] == 263 and keys.KEY["escape"] == 256
    assert (keys.SHIFT, keys.CTRL) == (1, 2)
