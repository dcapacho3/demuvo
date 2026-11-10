"""Tests for the canvas: colours, clipping, shapes, text and pictures."""

import math

import numpy as np
import pytest

from demuvo.gui import font
from demuvo.gui.raster import Canvas, is_convex, rgba

BLACK = rgba("#000000")
WHITE = rgba("#ffffff")


def ink(canvas, channel=0):
    """How much of a white canvas the drawing has covered, in pixels."""
    return float((1.0 - canvas.px[..., channel]).sum())


def test_a_colour_is_read_from_hex_with_an_opacity():
    """Each pair of digits is a channel from 0 to 1."""
    assert rgba("#ff8000", 0.5) == (1.0, 128 / 255, 0.0, 0.5)


def test_a_canvas_starts_as_one_colour_and_fill_repaints_it():
    """A hex string or a colour; the picture comes out as uint8."""
    canvas = Canvas(4, 3, "#336699")
    assert canvas.px.shape == (3, 4, 3)
    assert canvas.to_uint8()[0, 0].tolist() == [51, 102, 153]
    canvas.fill((1.0, 0.0, 0.0, 1.0))
    assert canvas.to_uint8()[2, 3].tolist() == [255, 0, 0]


def test_to_uint8_clamps_what_drawing_pushed_out_of_range():
    """Blending never leaves 0 to 1, but a caller may write to `px`."""
    canvas = Canvas(2, 1)
    canvas.px[0, 0] = (1.5, -0.5, 0.5)
    assert canvas.to_uint8()[0, 0].tolist() == [255, 0, 128]


def test_drawing_stays_inside_the_clip_and_the_clip_can_be_lifted():
    """Nothing changes outside the box, and a box beyond the canvas is cut to it."""
    canvas = Canvas(20, 20)
    canvas.set_clip(5, 5, 10, 10)
    canvas.rect((0, 0, 20, 20), BLACK)
    assert ink(canvas) == pytest.approx(25.0)
    canvas.set_clip(-5, -5, 100, 100)
    assert canvas.clip == (0, 0, 20, 20)


def test_a_mask_and_a_picture_wholly_outside_the_clip_change_nothing():
    """Both blends return early."""
    canvas = Canvas(10, 10)
    canvas.set_clip(0, 0, 5, 5)
    canvas.blend(6, 6, np.ones((3, 3), np.float32), BLACK)
    canvas.blend_rgba(6, 6, np.ones((3, 3, 4), np.float32), 1.0)
    assert ink(canvas) == 0.0


def test_blend_mixes_by_coverage_and_opacity():
    """Half covered by half opaque black on white is a quarter dark."""
    canvas = Canvas(2, 2)
    canvas.blend(0, 0, np.full((1, 1), 0.5, np.float32), (0, 0, 0, 0.5))
    assert canvas.px[0, 0, 0] == pytest.approx(0.75)


def test_blend_rgba_uses_straight_alpha_and_an_overall_opacity():
    """A red pixel at half alpha and half opacity, over white."""
    canvas = Canvas(2, 2)
    image = np.zeros((1, 1, 4), np.float32)
    image[0, 0] = (1.0, 0.0, 0.0, 0.5)
    canvas.blend_rgba(1, 1, image, 0.5)
    assert canvas.px[1, 1].tolist() == pytest.approx([1.0, 0.75, 0.75])


def test_a_rectangle_covers_whole_pixels_fully_and_edges_by_their_share():
    """A box from 1.5 to 4.5 has half-covered columns at both ends."""
    canvas = Canvas(8, 3)
    canvas.rect((1.5, 0, 3, 3), BLACK)
    columns = 1.0 - canvas.px[1, :, 0]
    assert columns.tolist() == pytest.approx([0, 0.5, 1, 1, 0.5, 0, 0, 0])


def test_a_rectangle_with_opacity_blends_and_one_off_the_canvas_does_nothing():
    """Half opaque black is mid grey; a rectangle past the edge is skipped."""
    canvas = Canvas(6, 6)
    canvas.rect((1, 1, 4, 4), (0, 0, 0, 0.5))
    assert canvas.px[2, 2, 0] == pytest.approx(0.5)
    canvas.rect((50, 50, 4, 4), BLACK)
    canvas.rect((50, 50, 4, 4), BLACK, radius=2)
    assert ink(canvas) == pytest.approx(16 * 0.5)


def test_rounded_corners_leave_the_corner_pixel_clear():
    """The middle is solid, the corner pixel is not."""
    canvas = Canvas(20, 20)
    canvas.rect((2, 2, 16, 16), BLACK, radius=6)
    assert canvas.px[10, 10, 0] == 0.0
    assert canvas.px[2, 2, 0] > 0.95
    assert canvas.px[3, 3, 0] > 0.8
    plain = Canvas(20, 20)
    plain.rect((2, 2, 16, 16), BLACK)
    assert ink(canvas) < ink(plain)


def test_a_rectangle_outline_is_a_ring_just_inside_its_edge():
    """A 1 pixel line on a 10 pixel box covers the 36 edge pixels and no more."""
    canvas = Canvas(20, 20)
    canvas.rect_outline((5, 5, 10, 10), BLACK, width=1.0)
    assert ink(canvas) == pytest.approx(36, abs=3)
    assert canvas.px[10, 10, 0] == 1.0
    canvas.rect_outline((500, 500, 10, 10), BLACK)


def test_a_disc_covers_its_area_and_a_ring_sits_inside_the_radius():
    """The area is pi r squared, and a stroke does not grow the disc."""
    canvas = Canvas(60, 60)
    canvas.circle((30, 30), 20, fill=BLACK)
    assert ink(canvas) == pytest.approx(math.pi * 400, rel=0.01)
    ring = Canvas(60, 60)
    ring.circle((30, 30), 20, stroke=BLACK, width=2.0)
    assert ring.px[30, 30, 0] == 1.0
    assert ring.px[30, 11, 0] < 0.1 and ring.px[30, 8, 0] == 1.0
    ring.circle((-500, -500), 5, fill=BLACK)


def test_a_convex_polygon_covers_its_area_either_way_round():
    """A triangle of area 200, the points in either order, with and without an edge."""
    points = [(10, 10), (30, 10), (10, 30)]
    areas = []
    for order in (points, points[::-1]):
        canvas = Canvas(40, 40)
        canvas.convex(order, fill=BLACK)
        areas.append(ink(canvas))
    assert areas == pytest.approx([200, 200], rel=0.02)
    edged = Canvas(40, 40)
    edged.convex(points, stroke=BLACK, width=1.0)
    assert edged.px[14, 14, 0] == 1.0
    assert edged.px[10, 10, 0] < 1.0 or edged.px[11, 11, 0] < 1.0
    edged.convex([(-500, -500), (-490, -500), (-490, -490)], fill=BLACK)


def test_a_polygon_that_is_not_convex_is_filled_by_sampling():
    """An L of area 300; the notch stays clear."""
    canvas = Canvas(40, 40)
    canvas.polygon([(5, 5), (15, 5), (15, 25), (35, 25), (35, 35), (5, 35)], BLACK)
    assert ink(canvas) == pytest.approx(10 * 30 + 20 * 10, rel=0.01)
    assert canvas.px[10, 25, 0] == 1.0
    canvas.polygon([(5, 5), (15, 5), (15, 5)], BLACK)
    canvas.polygon([(-500, -500), (-490, -500), (-490, -490)], BLACK)


def test_text_is_drawn_between_the_pen_and_the_returned_x():
    """Dark pixels appear, and the returned x is the width plus the start."""
    face = font.regular()
    canvas = Canvas(200, 30)
    end = canvas.text(face, "Hello, robot 123", (4, 20), size=14, color=BLACK)
    assert end == pytest.approx(4 + face.width("Hello, robot 123", 14))
    assert canvas.px.min() < 0.2
    assert canvas.px[:, int(end) + 3 :].min() == 1.0


def test_text_draws_nothing_for_spaces_and_obeys_the_clip():
    """A string of spaces leaves the canvas white; a clip cuts the letters."""
    face = font.regular()
    canvas = Canvas(60, 30)
    assert canvas.text(face, "   ", (4, 20), size=14, color=BLACK) > 4
    assert ink(canvas) == 0.0
    canvas.set_clip(0, 0, 10, 30)
    canvas.text(face, "WWWW", (4, 20), size=14, color=BLACK)
    assert canvas.px[:, 10:].min() == 1.0


def test_a_small_picture_is_magnified_without_smoothing_and_blends_alpha():
    """Each source pixel becomes a 10 by 10 block."""
    canvas = Canvas(40, 40, "#000000")
    image = np.zeros((2, 2, 4), np.uint8)
    image[0, 0] = (255, 0, 0, 255)
    image[1, 1] = (0, 255, 0, 128)
    canvas.image_scaled(image, (10, 10), 10)
    assert canvas.px[15, 15].tolist() == [1.0, 0.0, 0.0]
    assert canvas.px[25, 25, 1] == pytest.approx(128 / 255, abs=0.01)
    assert canvas.px[15, 25].tolist() == [0.0, 0.0, 0.0]
    assert canvas.px[9, 15, 0] == 0.0 and canvas.px[15, 9, 0] == 0.0


def test_a_picture_can_be_magnified_by_different_amounts_and_cut_by_the_clip():
    """Wider than tall, with one corner clipped away; wholly outside draws nothing."""
    canvas = Canvas(40, 40, "#000000")
    image = np.full((1, 1, 4), 255, np.uint8)
    canvas.set_clip(0, 0, 20, 20)
    canvas.image_scaled(image, (10, 10), (20, 5))
    assert canvas.px[12, 19, 0] == 1.0 and canvas.px[12, 20, 0] == 0.0
    assert canvas.px[16, 15, 0] == 0.0
    before = canvas.px.copy()
    canvas.image_scaled(image, (30, 30), 5)
    assert np.array_equal(canvas.px, before)


def test_a_convex_polygon_is_told_from_a_concave_one():
    """A square is convex either way round; an L is not."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    ell = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    assert is_convex(square) and is_convex(square[::-1]) and not is_convex(ell)
