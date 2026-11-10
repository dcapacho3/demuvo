"""A stand-in for the window, so a widget can be tested alone."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from demuvo.gui.widgets import Event

SRC = Path(__file__).resolve().parents[3]


class FakeHost:
    """Records what a widget asks of its window.

    Attributes:
        focus: the widget with the focus.
        keyboard_focus: True if the focus came from the keyboard.
        invalidated: every widget asked to be redrawn, in order.
        popups: the popups opened.
        closed: how many times a popup was closed.
        clipboard: the text on the clipboard.
    """

    def __init__(self):
        """Start with nothing focused and nothing recorded."""
        self.focus = None
        self.keyboard_focus = False
        self.invalidated = []
        self.popups = []
        self.closed = 0
        self.clipboard = ""
        self.groups = []

    def invalidate(self, widget=None):
        """Note a redraw."""
        self.invalidated.append(widget)

    def set_focus(self, widget):
        """Move the focus, telling the widgets as the window does."""
        if self.focus is not None and self.focus is not widget:
            self.focus.on_focus_change(False)
        self.focus = widget
        if widget is not None:
            widget.on_focus_change(True)

    def select_in_group(self, group, button):
        """Note the group and check the button."""
        self.groups.append((group, button))
        button.checked = True

    def open_popup(self, popup):
        """Note the popup."""
        self.popups.append(popup)

    def close_popup(self):
        """Count a close."""
        self.closed += 1

    def clip_get(self):
        """The clipboard."""
        return self.clipboard

    def clip_set(self, text):
        """Set the clipboard."""
        self.clipboard = text


def click(widget, x=None, y=None, outside=False):
    """Press and release the left button on a widget, or just off it."""
    px = widget.x + widget.width / 2 if x is None else x
    py = widget.y + widget.height / 2 if y is None else y
    event = Event(px, py)
    widget.on_press(event)
    if outside:
        event = Event(widget.x - 50, widget.y - 50)
    widget.on_release(event)


def ink(canvas, region=None):
    """The darkness on a white canvas, in a region (x0, y0, x1, y1) if given."""
    x0, y0, x1, y1 = region or (0, 0, canvas.width, canvas.height)
    return float((1.0 - canvas.px[y0:y1, x0:x1, 0]).sum())


def hosted(widget, rect=(10, 10, 120, 32)):
    """Give a widget a fake host and a place."""
    host = FakeHost()
    widget.adopt(host)
    widget.set_rect(*rect)
    return host


def on_virtual_display(snippet):
    """Run a Python snippet on a virtual display and return its finished process.

    The snippet may use `{src}` for the source folder, which is put on its path.
    The test is skipped if no virtual display is installed.
    """
    if shutil.which("xvfb-run") is None:
        pytest.skip("needs a virtual display")
    return subprocess.run(
        ["xvfb-run", "-a", sys.executable, "-I", "-c", snippet.format(src=str(SRC))],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
