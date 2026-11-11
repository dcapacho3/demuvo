"""Tests for the log panel."""

import pytest

from demuvo.gui import look
from demuvo.gui.logbuf import LogBuffer
from demuvo.gui.logview import LogView
from demuvo.gui.raster import Canvas
from demuvo.gui.widgets import Event

from .fakes import click, hosted


def make(messages=0):
    """A view over a buffer with some messages, in a small panel."""
    buffer, cleared = LogBuffer(), []
    for k in range(messages):
        buffer.add("warning", f"message {k}", 1_700_000_000.0 + k)
    view = LogView(buffer, lambda: cleared.append(1))
    hosted(view, (0, 0, 600, 140))
    return view, buffer, cleared


def draw(view):
    """The view drawn on a canvas."""
    canvas = Canvas(620, 160, look.WINDOW)
    view.draw(canvas)
    return canvas


def test_an_empty_log_says_so_and_a_full_one_draws_its_rows():
    """The two pictures differ; the clear button is in the header."""
    empty, buffer, _ = make()
    plain = draw(empty)
    buffer.add("warning", "the planner found no way", 1_700_000_000.0)
    buffer.add("error", "something broke", 1_700_000_001.0)
    full = draw(empty)
    assert (plain.px != full.px).any()


def test_a_long_message_is_cut_to_the_width_and_a_repeat_shows_its_count():
    """The text ends in an ellipsis; the count follows the text."""
    view, buffer, _ = make()
    for _ in range(3):
        buffer.add("warning", "w" * 400, 1_700_000_000.0)
    cut = draw(view)
    short = Canvas(620, 160, look.WINDOW)
    view.buffer.clear()
    view.draw(short)
    assert (cut.px != short.px).any()
    view.buffer.add("warning", "w" * 400, 1_700_000_000.0)
    cut_alone = draw(view)
    assert (cut_alone.px != short.px).any()


def test_news_says_open_once_then_update_and_then_nothing_until_a_change():
    """The panel opens itself for the first message only."""
    view, buffer, _ = make()
    assert view.news() is None
    buffer.add("warning", "one")
    assert view.news() == "open"
    buffer.add("warning", "two")
    assert view.news() == "update"
    assert view.news() is None
    buffer.clear()
    assert view.news() == "update"
    buffer.add("warning", "three")
    assert view.news() == "update"


def test_the_wheel_scrolls_back_through_the_messages_and_news_returns_to_the_newest():
    """Thirty messages do not fit; the wheel goes back and forward within limits."""
    view, buffer, _ = make(30)
    assert view.on_scroll(Event(dy=1))
    assert view.back == 1
    for _ in range(100):
        view.on_scroll(Event(dy=1))
    assert view.back == 30 - view._rows()  # pylint: disable=protected-access
    draw(view)
    for _ in range(100):
        view.on_scroll(Event(dy=-1))
    assert view.back == 0
    view.back = 5
    buffer.add("warning", "new")
    view.news()
    assert view.back == 0


def test_the_clear_button_calls_its_handler():
    """The header has a button that forgets the messages."""
    view, _, cleared = make(2)
    click(view.clear_button)
    assert cleared == [1]
    assert pytest.approx(view.clear_button.x) == view.x + view.width - 92
