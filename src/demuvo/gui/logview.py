"""The log panel - the warnings and log lines of a run, newest at the bottom.

A planner that finds no way, or a control period that is not whole steps, says so
here, in the window, and not on a terminal nobody is watching.
"""

from __future__ import annotations

import time

from . import font, look
from .logbuf import LogBuffer
from .raster import Canvas, rgba
from .widgets import Event, TextButton, Widget, text_baseline

ROW = 24
HEADER = 40
# The colours of an error line; a warning uses the warning text colour.
ERROR_TEXT = "#9b1c1c"


class LogView(Widget):
    """The messages in a list that follows the newest one and scrolls with the wheel.

    Args:
        buffer: the messages to show.
        on_clear: called when Clear is pressed.

    Attributes:
        back: how many rows the list is scrolled back from the newest.
    """

    def __init__(self, buffer: LogBuffer, on_clear) -> None:
        """Make the view with its Clear button."""
        super().__init__()
        self.buffer, self.back = buffer, 0
        self._seen, self._opened = 0, False
        self.clear_button = TextButton("Clear", on_clear)
        self.add(self.clear_button)

    def news(self) -> str | None:
        """What changed in the messages since the last call.

        Returns:
            "open" the first time a message has come, "update" when the messages
            changed, and None when nothing did.
        """
        if self.buffer.version == self._seen:
            return None
        self._seen = self.buffer.version
        self.back = 0
        self.invalidate()
        if self.buffer.entries and not self._opened:
            self._opened = True
            return "open"
        return "update"

    def layout(self) -> None:
        """Put the Clear button at the right of the header."""
        self.clear_button.set_rect(self.x + self.width - 92, self.y + 4, 80, 34)

    def _rows(self) -> int:
        """How many rows fit under the header."""
        return max(int((self.height - HEADER) // ROW), 1)

    def draw(self, canvas: Canvas) -> None:
        """Draw the header, the rows in view, and the button."""
        face = font.regular()
        canvas.rect((self.x, self.y, self.width, self.height), rgba(look.PANEL))
        canvas.rect((self.x, self.y, self.width, 1), rgba(look.LINE))
        entries = self.buffer.entries
        title = f"LOG  ({len(entries)})" if entries else "LOG"
        baseline = text_baseline(self.y, HEADER, 11)
        canvas.text(
            font.bold(), title, (self.x + 12, baseline), size=11, color=rgba(look.MUTED)
        )
        self.clear_button.draw(canvas)
        if not entries:
            at = (self.x + 12, text_baseline(self.y + HEADER, ROW, 13))
            canvas.text(face, "No messages.", at, size=13, color=rgba(look.MUTED))
            return
        rows = self._rows()
        self.back = min(self.back, max(len(entries) - rows, 0))
        last = len(entries) - self.back
        saved = canvas.clip
        left, top = max(saved[0], int(self.x)), max(saved[1], int(self.y + HEADER))
        right = min(saved[2], int(self.x + self.width))
        bottom = min(saved[3], int(self.y + self.height))
        canvas.set_clip(left, top, right, bottom)
        for row, entry in enumerate(entries[max(last - rows, 0) : last]):
            self._draw_entry(canvas, face, entry, self.y + HEADER + row * ROW)
        canvas.clip = saved

    def _draw_entry(self, canvas: Canvas, face, entry, top: float) -> None:
        """One line: the time, then the text cut to fit, then how often it came."""
        colour = look.WARN_TEXT if entry.level == "warning" else ERROR_TEXT
        baseline = text_baseline(top, ROW, 13)
        stamp = time.strftime("%H:%M:%S", time.localtime(entry.when))
        canvas.text(
            face, stamp, (self.x + 12, baseline), size=13, color=rgba(look.MUTED)
        )
        suffix = f"  x{entry.count}" if entry.count > 1 else ""
        room = self.width - 12 - 86 - face.width(suffix, 13) - 12
        text = entry.text
        if face.width(text, 13) > room:
            while text and face.width(text + "…", 13) > room:
                text = text[:-1]
            text += "…"
        canvas.text(
            face, text + suffix, (self.x + 86, baseline), size=13, color=rgba(colour)
        )

    def on_scroll(self, event: Event) -> bool:
        """The wheel moves back through the messages, and forward again."""
        most = max(len(self.buffer.entries) - self._rows(), 0)
        self.back = min(max(self.back + (1 if event.dy > 0 else -1), 0), most)
        self.invalidate()
        return True
