"""The messages a run gives - warnings and log lines, kept for the window to show.

Python prints a warning or a log line on the terminal, where a window user never looks.
`capture` sends them to a `LogBuffer` instead. A simulation in another process fills
its own buffer and sends what it holds with each reply.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

# The most messages kept; older ones are dropped.
LIMIT = 500


@dataclass
class Entry:
    """One message.

    Attributes:
        when: the time it came, as `time.time()` gives it.
        level: "warning" or "error".
        text: what it says.
        count: how many times in a row it came.
    """

    when: float
    level: str
    text: str
    count: int = 1


class LogBuffer:
    """The messages so far, with a count of changes so a window knows to redraw.

    Attributes:
        entries: the messages, oldest first.
        version: goes up each time something is added or cleared.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.entries: list[Entry] = []
        self.version = 0

    def add(self, level: str, text: str, when: float | None = None) -> None:
        """Add a message; the same text again just raises the count of the last one."""
        self.version += 1
        last = self.entries[-1] if self.entries else None
        if last is not None and last.text == text and last.level == level:
            last.count += 1
            return
        self.entries.append(Entry(time.time() if when is None else when, level, text))
        del self.entries[:-LIMIT]

    def extend(self, items) -> None:
        """Add messages sent as (when, level, text) triples."""
        for when, level, text in items:
            self.add(level, text, when)

    def drain(self) -> list[tuple[float, str, str]]:
        """Take every message as (when, level, text), once for each time it came."""
        out = [(e.when, e.level, e.text) for e in self.entries for _ in range(e.count)]
        self.entries.clear()
        return out

    def clear(self) -> None:
        """Forget every message."""
        self.entries.clear()
        self.version += 1


class _Handler(logging.Handler):
    """A logging handler that puts the lines in a buffer."""

    def __init__(self, buffer: LogBuffer) -> None:
        """Take the lines of warning level and above."""
        super().__init__(logging.WARNING)
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        """Keep one line."""
        level = "error" if record.levelno >= logging.ERROR else "warning"
        self.buffer.add(level, record.getMessage())


def capture(buffer: LogBuffer):
    """Send warnings and log lines to a buffer, until the returned function is called.

    Nothing is then printed on the terminal for them.

    Returns:
        A function that puts things back as they were.
    """
    handler = _Handler(buffer)
    root = logging.getLogger()
    root.addHandler(handler)
    earlier = warnings.showwarning

    def show(message, *_rest, **_named):
        """Keep a warning as one line of text; the rest is dropped."""
        buffer.add("warning", str(message))

    warnings.showwarning = show

    def release() -> None:
        """Remove the handler and the hook."""
        root.removeHandler(handler)
        warnings.showwarning = earlier

    return release
