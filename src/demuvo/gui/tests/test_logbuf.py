"""Tests for keeping warnings and log lines for the window."""

import logging
import warnings

from demuvo.gui import logbuf
from demuvo.gui.logbuf import LogBuffer, capture


def test_a_message_is_kept_and_the_same_one_again_only_raises_the_count():
    """Equal text in a row is one entry with a count; a change starts a new entry."""
    buffer = LogBuffer()
    buffer.add("warning", "no path", 1.0)
    buffer.add("warning", "no path", 2.0)
    buffer.add("error", "no path", 3.0)
    buffer.add("warning", "other", 4.0)
    assert [(e.text, e.level, e.count) for e in buffer.entries] == [
        ("no path", "warning", 2),
        ("no path", "error", 1),
        ("other", "warning", 1),
    ]
    assert buffer.version == 4


def test_only_the_latest_messages_are_kept(monkeypatch):
    """Older ones are dropped beyond the limit."""
    monkeypatch.setattr(logbuf, "LIMIT", 3)
    buffer = LogBuffer()
    for k in range(6):
        buffer.add("warning", f"m{k}")
    assert [e.text for e in buffer.entries] == ["m3", "m4", "m5"]


def test_drain_hands_over_every_message_once_each_time_it_came_and_extend_takes_them():
    """The triples cross a pipe and fill another buffer."""
    one, other = LogBuffer(), LogBuffer()
    one.add("warning", "a", 1.0)
    one.add("warning", "a", 2.0)
    sent = one.drain()
    assert len(sent) == 2 and not one.entries
    other.extend(sent)
    assert other.entries[0].count == 2
    other.clear()
    assert not other.entries and other.version == 3


def test_capture_sends_warnings_and_log_lines_to_the_buffer_and_then_lets_go():
    """Nothing reaches the terminal while it is on; afterwards things are as before."""
    buffer = LogBuffer()
    before = warnings.showwarning
    release = capture(buffer)
    warnings.warn("a period that is odd")
    logging.getLogger("demuvo.test").warning("no way from %s", "here")
    logging.getLogger("demuvo.test").error("broken")
    logging.getLogger("demuvo.test").info("quiet")
    release()
    assert [(e.level, e.text) for e in buffer.entries] == [
        ("warning", "a period that is odd"),
        ("warning", "no way from here"),
        ("error", "broken"),
    ]
    assert warnings.showwarning is before
    logging.getLogger("demuvo.test").warning("after")
    assert len(buffer.entries) == 3
