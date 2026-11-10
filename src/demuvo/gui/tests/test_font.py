"""Tests for the text: outlines, glyph bitmaps, widths, and where the font is found."""

import numpy as np
import pytest

from demuvo.gui import font


def test_a_glyph_is_a_coverage_bitmap_with_its_place_on_the_line():
    """A capital I at 20 px is a tall thin bar, dark inside and clear at the edge."""
    glyph = font.regular().glyph("I", 20)
    rows, columns = glyph.coverage.shape
    assert 12 < rows < 20 and columns < 8
    assert 0.0 <= glyph.coverage.min() and glyph.coverage.max() == 1.0
    assert glyph.coverage[:, 0].max() == 0.0
    assert glyph.coverage[rows // 2].sum() > 1.5
    assert 12 <= glyph.top <= 17 and glyph.advance > 0


def test_a_space_has_a_width_and_no_pixels():
    """Nothing to draw, but the pen moves on."""
    glyph = font.regular().glyph(" ", 13)
    assert glyph.coverage.size == 0 and glyph.advance > 3


def test_a_glyph_is_made_once_per_size():
    """The second call gives the same object, and another size makes another."""
    face = font.Font(font.font_file("Roboto-Regular.ttf"))
    first = face.glyph("a", 13)
    assert face.glyph("a", 13) is first
    face.glyph("a", 20)
    assert face.glyphs_made == 2


def test_a_character_the_font_lacks_is_drawn_as_a_question_mark():
    """A snowman is not in Roboto."""
    face = font.regular()
    assert np.array_equal(face.glyph("☃", 14).coverage, face.glyph("?", 14).coverage)


def test_widths_follow_the_letters_and_the_weight():
    """Narrow letters are narrow, a string is the sum of its letters, bold is wider."""
    face = font.regular()
    assert face.width("iiii", 13) < face.width("WWWW", 13)
    assert (
        face.width("ab", 13)
        == face.glyph("a", 13).advance + face.glyph("b", 13).advance
    )
    assert font.bold().width("robot", 13) > face.width("robot", 13)
    assert face.width("", 13) == 0


def test_bold_puts_more_ink_on_the_page():
    """The same letter covers more pixels in bold."""
    assert (
        font.bold().glyph("a", 24).coverage.sum()
        > font.regular().glyph("a", 24).coverage.sum()
    )


def test_a_line_is_a_little_taller_than_the_text():
    """The gap from one baseline to the next."""
    assert 13 < font.regular().line_height(13) < 20


def test_a_letter_with_a_hole_keeps_it():
    """The inside of an o stays clear, as the winding rule gives."""
    glyph = font.regular().glyph("o", 40).coverage
    middle = glyph[glyph.shape[0] // 2]
    assert middle[0] == 0.0 and middle[len(middle) // 2] == 0.0 and middle.max() == 1.0


def test_curves_are_cut_into_short_straight_pieces():
    """A curve with two off-curve points passes through their middle."""
    quad = font.flatten(
        [
            ("moveTo", ((0, 0),)),
            ("qCurveTo", ((10, 0), (10, 10), (0, 10))),
            ("closePath", ()),
        ],
        1.0,
    )[0]
    assert quad[0].tolist() == [0, 0] and quad[-1].tolist() == [0, -10]
    assert [10, -5] in np.round(quad, 6).tolist() or len(
        quad
    ) == 1 + 2 * font.QUADRATIC_PIECES
    cubic = font.flatten(
        [
            ("moveTo", ((0, 0),)),
            ("curveTo", ((0, 5), (5, 10), (10, 10))),
            ("endPath", ()),
        ],
        2.0,
    )[0]
    assert len(cubic) == 1 + font.CUBIC_PIECES
    assert cubic[-1].tolist() == [20, -20]


def test_a_contour_of_off_curve_points_only_is_a_closed_loop():
    """It starts and ends at the middle of its last and first points."""
    loop = font.flatten(
        [("qCurveTo", ((0, 0), (10, 0), (10, 10), (0, 10), None)), ("closePath", ())],
        1.0,
    )[0]
    assert loop[0].tolist() == [0, -5] and loop[-1].tolist() == [0, -5]
    assert len(loop) == 1 + 4 * font.QUADRATIC_PIECES


def test_the_question_mark_of_the_bold_face_has_a_dot():
    """The dot of that glyph is such a loop, and it must not stop the drawing."""
    assert font.bold().glyph("☃", 11).coverage.sum() > 5


def test_outline_commands_make_one_polyline_for_each_contour():
    """Lines, a second contour, and one that is never closed."""
    contours = font.flatten(
        [
            ("moveTo", ((0, 0),)),
            ("lineTo", ((4, 0),)),
            ("lineTo", ((4, 4),)),
            ("closePath", ()),
            ("moveTo", ((8, 8),)),
            ("lineTo", ((9, 8),)),
            ("moveTo", ((20, 20),)),
            ("lineTo", ((21, 20),)),
        ],
        0.5,
    )
    assert [len(c) for c in contours] == [3, 2, 2]
    assert contours[0].tolist() == [[0, 0], [2, 0], [2, -2]]


def test_the_system_font_comes_first_and_the_bundled_one_is_the_fallback(
    tmp_path, monkeypatch
):
    """A Roboto in the system folder is used; with none there, the shipped copy is."""
    monkeypatch.setattr(font, "SYSTEM_FONTS", tmp_path)
    assert (
        font.font_file("Roboto-Regular.ttf")
        == font.BUNDLED_FONTS / "Roboto-Regular.ttf"
    )
    (tmp_path / "Roboto-Regular.ttf").write_bytes(b"x")
    assert font.font_file("Roboto-Regular.ttf") == tmp_path / "Roboto-Regular.ttf"


def test_a_font_that_is_nowhere_is_refused_with_both_places_named(
    tmp_path, monkeypatch
):
    """The message says where it looked."""
    monkeypatch.setattr(font, "SYSTEM_FONTS", tmp_path)
    with pytest.raises(FileNotFoundError, match="Roboto-Mono.ttf is neither in"):
        font.font_file("Roboto-Mono.ttf")


def test_the_shipped_fonts_and_their_licence_are_in_the_package():
    """Both weights open, and the licence file sits beside them."""
    for name in ("Roboto-Regular.ttf", "Roboto-Bold.ttf"):
        assert font.Font(font.BUNDLED_FONTS / name).width("robot", 13) > 20
    assert "Apache License" in (font.BUNDLED_FONTS / "LICENSE.txt").read_text()
