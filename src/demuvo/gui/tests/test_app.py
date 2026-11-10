"""Tests for opening the window on a session."""

import pytest

from demuvo.gui import app as module
from demuvo.gui import glfw as binding
from demuvo.gui import window as window_module
from demuvo.gui.session import Session

from .fakes import EXAMPLES, on_virtual_display


class StubWindow:
    """A window that opens nothing: it renders one frame and returns."""

    shown = []

    def __init__(self, title, width, height):
        """Note the title and size."""
        self.title, self.size = title, (width, height)

    def framebuffer_size(self):
        """A small frame."""
        return 640, 480

    def run(self, program, seconds):
        """Render a frame and note the program and the time limit."""
        frame = program.render(*self.framebuffer_size())
        StubWindow.shown.append((self.title, frame.shape, seconds, bool(frame.any())))


def test_a_sandbox_session_opens_a_window_of_the_frame_size(monkeypatch):
    """The title names the mode; the frame has the window's size."""
    monkeypatch.setattr(module, "Window", StubWindow)
    StubWindow.shown = []
    assert module.show(Session(), 2.0) == 0
    assert StubWindow.shown == [("demuvo - sandbox", (480, 640, 3), 2.0, True)]


def test_a_benchmark_session_says_so_in_the_title(monkeypatch):
    """A fixed world, named in the window."""
    monkeypatch.setattr(module, "Window", StubWindow)
    StubWindow.shown = []
    assert module.show(Session(EXAMPLES / "demo.toml", "benchmark")) == 0
    assert StubWindow.shown[0][0] == "demuvo - benchmark"


def test_a_window_that_cannot_open_is_a_plain_message_and_exit_one(monkeypatch, capsys):
    """No display, or no GLFW."""

    def refuse(*_args):
        raise RuntimeError(
            "GLFW is not installed; on Ubuntu run: sudo apt install libglfw3"
        )

    monkeypatch.setattr(module, "Window", refuse)
    assert module.show(Session()) == 1
    assert "sudo apt install libglfw3" in capsys.readouterr().err


SNIPPET = """
import sys
sys.path.insert(0, {src!r})
from demuvo.run.cli import main
sys.exit(main([{world!r}, "--bench", "--seconds", "1.5"]))
"""


def test_a_real_window_opens_runs_and_closes_on_a_virtual_display():
    """The whole chain, from the command to GLFW and OpenGL, for 1.5 s."""
    try:
        binding.load_library()
        window_module.load_gl()
    except RuntimeError:
        pytest.skip("libglfw3 or libgl1 is not installed")
    result = on_virtual_display(
        SNIPPET.replace("{world!r}", repr(str(EXAMPLES / "demo.toml")))
    )
    assert result.returncode == 0, result.stderr
