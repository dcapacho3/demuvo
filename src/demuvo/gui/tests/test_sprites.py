"""Tests for sprites: the visible pixels, batching, order and clipping."""

import numpy as np
import pytest

from demuvo.gui.raster import Canvas
from demuvo.gui.sprites import Sprite, draw_sprites


def disc(size=6, color=(1.0, 0.0, 0.0)):
    """A solid square of one colour with a transparent corner."""
    image = np.zeros((size, size, 4), np.float32)
    image[..., :3] = color
    image[..., 3] = 1.0
    image[0, 0, 3] = 0.0
    return Sprite(image)


def test_a_sprite_keeps_only_its_visible_pixels():
    """A 6 by 6 square without one corner has 35 of them."""
    sprite = disc()
    assert len(sprite.alpha) == 35 and sprite.rgb.shape == (35, 3)
    assert (sprite.height, sprite.width) == (6, 6)


def test_the_offsets_of_the_pixels_depend_on_the_canvas_width_and_are_kept():
    """Row times width plus column, worked out once for each width."""
    sprite = disc(3)
    assert sprite.offsets(10)[0] == 1
    assert sprite.offsets(10)[-1] == 2 * 10 + 2
    assert sprite.offsets(10) is sprite.offsets(10)
    assert sprite.offsets(20)[-1] == 2 * 20 + 2


def test_one_sprite_looks_like_blending_its_picture():
    """The same pixels as `blend_rgba`."""
    sprite = disc()
    drawn, blended = Canvas(20, 20), Canvas(20, 20)
    draw_sprites(drawn, [sprite], [5], [7])
    blended.blend_rgba(5, 7, sprite.rgba)
    assert np.array_equal(drawn.px, blended.px)


def test_many_sprites_that_do_not_touch_look_like_one_after_another():
    """Eight sprites in a row, and an overall opacity."""
    sprites = [disc(5, (0.2 * i, 0.5, 1.0 - 0.1 * i)) for i in range(8)]
    xs = [6 * i for i in range(8)]
    drawn, blended = Canvas(60, 12), Canvas(60, 12)
    draw_sprites(drawn, sprites, xs, [3] * 8, opacity=0.5)
    for sprite, x in zip(sprites, xs):
        blended.blend_rgba(x, 3, sprite.rgba, 0.5)
    assert np.allclose(drawn.px, blended.px, atol=1e-6)


def test_the_later_sprite_is_on_top_where_two_overlap():
    """A chain of three, each over the one before, as in the list order."""
    red, green, blue = disc(6, (1, 0, 0)), disc(6, (0, 1, 0)), disc(6, (0, 0, 1))
    drawn, blended = Canvas(30, 12), Canvas(30, 12)
    draw_sprites(drawn, [red, green, blue], [2, 5, 8], [3, 3, 3])
    for sprite, x in zip((red, green, blue), (2, 5, 8)):
        blended.blend_rgba(x, 3, sprite.rgba)
    assert np.array_equal(drawn.px, blended.px)
    assert drawn.px[5, 6].tolist() == [0.0, 1.0, 0.0]
    assert drawn.px[5, 9].tolist() == [0.0, 0.0, 1.0]


def test_a_sprite_partly_outside_the_clip_is_cut_and_one_outside_is_skipped():
    """Over the left edge, past the right edge, wholly away, and one inside."""
    sprite = disc()
    canvas = Canvas(20, 20)
    draw_sprites(canvas, [sprite] * 4, [-3, 18, 500, 8], [5, 5, 5, 12])
    assert canvas.px[5, 0].tolist() == [1.0, 0.0, 0.0]
    assert canvas.px[7, 19].tolist() == [1.0, 0.0, 0.0]
    assert canvas.px[14, 10].tolist() == [1.0, 0.0, 0.0]
    assert (1.0 - canvas.px[..., 1]).sum() == pytest.approx(
        3 * 6 + 2 * 6 + 35 - 3 * 0, abs=40
    )


def test_a_sprite_stays_inside_a_clip_box_smaller_than_the_canvas():
    """Nothing lands outside the box."""
    canvas = Canvas(30, 30)
    canvas.set_clip(10, 10, 20, 20)
    draw_sprites(canvas, [disc(8)] * 2, [8, 12], [12, 12])
    outside = np.ones(canvas.px.shape[:2], bool)
    outside[10:20, 10:20] = False
    assert canvas.px[outside].min() == 1.0
    assert canvas.px[12, 15, 1] == 0.0


def test_no_sprites_or_none_inside_change_nothing():
    """An empty list, and a list where all are outside."""
    canvas = Canvas(10, 10)
    draw_sprites(canvas, [], [], [])
    draw_sprites(canvas, [disc()], [99], [99])
    assert (1.0 - canvas.px).sum() == 0.0
