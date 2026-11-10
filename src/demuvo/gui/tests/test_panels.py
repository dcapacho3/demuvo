"""Tests for the note, the form row, the title, the badge and the playback pill."""

import pytest

from demuvo.gui import look
from demuvo.gui.inputs import NumberSpec, Spin
from demuvo.gui.panels import SPEEDS, START_CLOCK, Badge, FormRow, Note, Pill, TitleRow
from demuvo.gui.raster import Canvas, rgba

from .fakes import click, hosted, ink


def test_a_note_is_hidden_until_it_has_text_and_hides_again_when_emptied():
    """The text shows it; an empty text hides it."""
    note = Note()
    assert not note.visible
    note.set_text("A run is in progress.")
    assert note.visible
    note.set_text("")
    assert not note.visible


def test_a_note_is_as_tall_as_its_text_and_draws_in_the_warning_colours():
    """More words need more height; the box has the warning fill."""
    note = Note()
    note.set_text(
        "A run is in progress. Press Reset to edit again, and wait for it to end."
    )
    narrow, wide = note.height_for(120), note.height_for(400)
    assert narrow > wide > 16
    note.set_rect(10, 10, 200, note.height_for(200))
    canvas = Canvas(240, 120)
    note.draw(canvas)
    assert canvas.px[12, 100].tolist() == pytest.approx(
        list(rgba(look.WARN_BG)[:3]), abs=0.01
    )
    assert ink(canvas, (20, 15, 200, 60)) > 5


def test_a_form_row_puts_its_control_right_of_the_label_and_fades_the_label_with_it():
    """The control starts 92 pixels in; a disabled control dims the label."""
    field = Spin(1.0, NumberSpec(0, 10, 1))
    row = FormRow("heading", field)
    hosted(row, (10, 10, 250, 36))
    assert (field.x, field.y, field.width, field.height) == (102, 13, 158, 30)
    canvas = Canvas(270, 60)
    row.draw(canvas)
    lit = ink(canvas, (10, 10, 90, 46))
    field.enabled = False
    dim = Canvas(270, 60)
    row.draw(dim)
    assert ink(dim, (10, 10, 90, 46)) < lit


def test_a_title_draws_a_coloured_dot_and_the_name():
    """The dot is in the robot's colour; the name leaves ink to its right."""
    title = TitleRow()
    title.name, title.color = "r1", "#f5a623"
    title.set_rect(10, 10, 200, 34)
    canvas = Canvas(220, 60)
    title.draw(canvas)
    assert canvas.px[27, 21].tolist() == pytest.approx(
        list(rgba("#f5a623")[:3]), abs=0.05
    )
    assert ink(canvas, (40, 10, 120, 44)) > 5


def test_a_badge_is_as_wide_as_its_text_and_draws_a_pill_in_its_kind_colour():
    """Wider text, wider badge; edit is mint, bench is navy."""
    badge = Badge()
    badge.text = "BENCHMARK"
    wide = badge.wanted_width()
    badge.text = "RUN"
    assert wide > badge.wanted_width()
    for kind, fill in (("edit", look.MINT2), ("run", "#fdebc8"), ("bench", look.NAVY)):
        badge.kind = kind
        badge.text = "SANDBOX"
        badge.set_rect(10, 10, badge.wanted_width(), 24)
        canvas = Canvas(150, 44)
        badge.draw(canvas)
        assert canvas.px[13, 20].tolist() == pytest.approx(
            list(rgba(fill)[:3]), abs=0.05
        )


def test_the_pill_calls_back_for_play_step_reset_and_a_new_speed():
    """Each control reaches its callback; the speed arrives as a number."""
    calls = []
    pill = Pill(
        lambda: calls.append("play"),
        lambda: calls.append("step"),
        lambda: calls.append("reset"),
        calls.append,
    )
    host = hosted(pill, (100, 100, 300, 44))
    for button in (pill.play, pill.step, pill.reset):
        click(button)
    click(pill.speed)
    host.popups[-1].items[4].action()
    assert calls == ["play", "step", "reset", 4.0]
    assert [i.text for i in host.popups[-1].items] == [f"{s:g}x" for s in SPEEDS]


def test_the_play_button_turns_into_pause_while_playing():
    """Icon and tooltip both change, and change back."""
    pill = Pill(print, print, print, print)
    pill.set_playing(True)
    assert (pill.play.icon, pill.play.tooltip) == ("pause", "Pause (Space)")
    pill.set_playing(False)
    assert (pill.play.icon, pill.play.tooltip) == ("play", "Play (Space)")


def test_the_clock_shows_minutes_and_tenths_and_says_when_it_changed():
    """65.25 seconds is 1:05.2; the same time again is no change."""
    pill = Pill(print, print, print, print)
    assert pill.clock == START_CLOCK
    assert pill.set_time(65.25) is True and pill.clock == "1:05.2"
    assert pill.set_time(65.25) is False
    assert pill.set_time(0.0) is True and pill.clock == START_CLOCK


def test_the_pill_is_placed_by_its_bottom_centre_and_lays_its_controls_in_a_row():
    """Buttons at 32 pixels each, then the speed choice."""
    pill = Pill(print, print, print, print)
    pill.place(400, 500)
    assert (pill.x + pill.width / 2, pill.y + pill.height) == (400, 500)
    assert pill.play.x == pill.x + 6 and pill.step.x == pill.play.x + 32
    assert pill.speed.x == pill.reset.x + 32 + 6


def test_the_pill_draws_its_card_controls_and_clock_and_keeps_the_card():
    """Ink for all three; the card sprite is made again only when the size changes."""
    pill = Pill(print, print, print, print)
    hosted(pill, (20, 20, pill.width, pill.height))
    canvas = Canvas(400, 100, look.WINDOW)
    pill.draw(canvas)
    assert ink(canvas, (20, 20, 360, 64)) > 20
    first = pill._body  # pylint: disable=protected-access
    pill.draw(canvas)
    assert pill._body is first  # pylint: disable=protected-access
    pill.set_rect(20, 20, pill.width + 40, pill.height)
    pill.draw(canvas)
    assert pill._body is not first  # pylint: disable=protected-access
