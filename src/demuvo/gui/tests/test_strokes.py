"""Tests for lines, dashes and rays."""

import numpy as np
import pytest

from demuvo.gui import strokes
from demuvo.gui.raster import Canvas, rgba

BLACK = rgba("#000000")
WHITE = rgba("#ffffff")


def ink(canvas):
    """The total darkness drawn on a white canvas."""
    return float((1.0 - canvas.px[..., 0]).sum())


def test_a_point_on_a_pixel_centre_puts_its_weight_on_that_pixel():
    """No split between pixels."""
    canvas = Canvas(10, 10)
    strokes.stamp_points(canvas, np.array([[4.5, 5.5]]), 0.8, BLACK)
    assert canvas.px[5, 4, 0] == pytest.approx(0.2)
    assert ink(canvas) == pytest.approx(0.8)


def test_a_point_between_pixels_is_shared_by_bilinear_weights():
    """On the corner of four pixels each gets a quarter."""
    canvas = Canvas(10, 10)
    strokes.stamp_points(canvas, np.array([[5.0, 5.0]]), 1.0, BLACK)
    assert canvas.px[4:6, 4:6, 0].ravel().tolist() == pytest.approx([0.75] * 4)


def test_no_points_or_points_outside_the_clip_change_nothing():
    """An empty call, and points on the border where a splat would leave the box."""
    canvas = Canvas(10, 10)
    strokes.stamp_points(canvas, np.empty((0, 2)), 1.0, BLACK)
    strokes.stamp_points(canvas, np.array([[0.2, 5.0], [99.0, 5.0]]), 1.0, BLACK)
    assert ink(canvas) == 0.0


def test_a_splat_never_leaves_the_clip_box():
    """The pixel beyond a point at the edge of the clip is not touched."""
    canvas = Canvas(20, 20)
    canvas.set_clip(5, 5, 15, 15)
    strokes.stamp_points(canvas, np.array([[14.9, 10.0], [5.2, 10.0]]), 1.0, BLACK)
    assert canvas.px[:, 15:, 0].min() == 1.0 and canvas.px[:, :5, 0].min() == 1.0


def test_the_weight_of_one_pixel_is_capped_at_one():
    """Many points on one spot make solid black, not more."""
    canvas = Canvas(10, 10)
    strokes.stamp_points(canvas, np.full((50, 2), 4.5), 1.0, BLACK)
    assert canvas.px[4, 4, 0] == 0.0


def test_offsets_splat_each_point_again_shifted():
    """Two offsets make two spots."""
    canvas = Canvas(20, 10)
    strokes.stamp_points(
        canvas, np.array([[5.5, 5.5]]), 1.0, BLACK, offsets=[(0, 0), (6, 0)]
    )
    assert canvas.px[5, 5, 0] == 0.0 and canvas.px[5, 11, 0] == 0.0


def test_colours_that_cross_mix_by_their_weights():
    """Equal red and blue on one pixel give an even mix."""
    canvas = Canvas(10, 10)
    colours = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    strokes.stamp_points(
        canvas,
        np.array([[4.5, 4.5], [4.5, 4.5]]),
        0.5,
        (0, 0, 0, 1),
        sample_rgb=colours,
    )
    red, green, blue = canvas.px[4, 4]
    assert red == pytest.approx(0.5) and blue == pytest.approx(0.5) and green == 0.0


@pytest.mark.parametrize("width", [1.0, 3.0, 6.0])
def test_a_line_covers_about_its_length_times_its_width(width):
    """Thin, medium and thick lines each put the ink where the width says."""
    canvas = Canvas(80, 30)
    strokes.polyline(canvas, [(10, 15), (60, 15)], width, BLACK)
    assert ink(canvas) == pytest.approx(50 * width, rel=0.15)
    assert (1.0 - canvas.px[10:20, 35, 0]).sum() == pytest.approx(width, rel=0.2)
    assert canvas.px[5, 35, 0] == 1.0


def test_a_dashed_line_has_gaps_and_about_the_share_of_ink():
    """Dashes of 6 with gaps of 4 put 60 per cent of the ink down."""
    full, dashed = Canvas(80, 20), Canvas(80, 20)
    strokes.polyline(full, [(5, 10), (75, 10)], 1.0, BLACK)
    strokes.polyline(dashed, [(5, 10), (75, 10)], 1.0, BLACK, dash=(6, 4))
    assert ink(dashed) == pytest.approx(0.6 * ink(full), rel=0.1)
    assert dashed.px[10, 5:75, 0].max() > 0.9


def test_a_line_that_is_one_point_or_has_no_length_draws_almost_nothing():
    """One point is no line; a line from a point to itself is a dot."""
    canvas = Canvas(20, 20)
    strokes.polyline(canvas, [(5, 5)], 1.0, BLACK)
    assert ink(canvas) == 0.0
    strokes.polyline(canvas, [(8, 8), (8, 8)], 1.0, BLACK)
    assert 0 < ink(canvas) < 1.0


def test_many_lines_draw_in_their_own_colours_in_one_pass():
    """A red and a blue line, a one-point path that is skipped, and a dashed pair."""
    canvas = Canvas(100, 60, "#000000")
    strokes.polylines(
        canvas,
        [
            np.array([[5, 15], [95, 15]]),
            np.array([[5, 45], [95, 45]]),
            np.array([[5, 30]]),
        ],
        [rgba("#ff0000"), rgba("#0000ff"), rgba("#ffffff")],
        3,
    )
    assert canvas.px[15, 50, 0] > 0.5 and canvas.px[15, 50, 2] < 0.1
    assert canvas.px[45, 50, 2] > 0.5 and canvas.px[45, 50, 0] < 0.1
    assert canvas.px[30].max() == 0.0
    dashed = Canvas(100, 20, "#000000")
    strokes.polylines(
        dashed, [np.array([[5, 10], [95, 10]])], [rgba("#ffffff")], 1, dash=(6, 4)
    )
    row = dashed.px[10, 5:95, 0]
    assert row.max() > 0.5 and row.min() < 0.1


def test_nothing_to_draw_is_no_error():
    """No paths, or only paths too short to be lines."""
    canvas = Canvas(20, 20)
    strokes.polylines(canvas, [], [], 1)
    strokes.polylines(canvas, [np.array([[1, 1]])], [BLACK], 1)
    assert ink(canvas) == 0.0


def test_rays_from_one_point_reach_out_in_every_direction():
    """Twelve rays of length 40 from the middle."""
    canvas = Canvas(100, 100, "#000000")
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ends = np.column_stack(
        [
            np.full(12, 50.0),
            np.full(12, 50.0),
            50 + 40 * np.cos(angle),
            50 + 40 * np.sin(angle),
        ]
    )
    strokes.segments(canvas, ends, 1, rgba("#ffffff"))
    assert canvas.px[50, 80, 0] > 0.3 and canvas.px[50, 50, 0] > 0.3
    assert canvas.px[10, 10, 0] == 0.0


def test_a_ray_of_no_length_is_a_dot():
    """It does not divide by zero."""
    canvas = Canvas(20, 20)
    strokes.segments(canvas, np.array([[10.0, 10.0, 10.0, 10.0]]), 1, BLACK)
    assert 0 < ink(canvas) < 2
