"""Tests for the `demuvo gui` command."""

import runpy

import numpy as np
import pytest

from demuvo.gui import app as module
from demuvo.run import cli

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
        StubWindow.shown.append((self.title, frame.shape, seconds, bool(np.any(frame))))


def test_the_command_opens_a_window_over_a_sandbox_by_default(monkeypatch):
    """No file: a blank sandbox, a frame of the window's size, the limit passed on."""
    monkeypatch.setattr(module, "Window", StubWindow)
    StubWindow.shown = []
    assert module.main(["--seconds", "2"]) == 0
    assert StubWindow.shown == [("demuvo - sandbox", (480, 640, 3), 2.0, True)]


def test_a_world_file_with_bench_opens_a_fixed_benchmark(monkeypatch):
    """The title names the mode."""
    monkeypatch.setattr(module, "Window", StubWindow)
    StubWindow.shown = []
    assert module.main([str(EXAMPLES / "demo.toml"), "--bench"]) == 0
    assert StubWindow.shown[0][0] == "demuvo - benchmark"


@pytest.mark.parametrize("arguments", [["missing.toml"], ["--bench"]])
def test_a_world_that_cannot_be_opened_is_a_plain_message_and_exit_one(
    arguments, capsys
):
    """A file that is not there, and a benchmark with no file."""
    assert module.main(arguments) == 1
    assert capsys.readouterr().err.startswith("demuvo: ")


def test_a_window_that_cannot_open_is_a_plain_message_and_exit_one(monkeypatch, capsys):
    """No display, or no GLFW."""

    def refuse(*_args):
        raise RuntimeError(
            "GLFW is not installed; on Ubuntu run: sudo apt install libglfw3"
        )

    monkeypatch.setattr(module, "Window", refuse)
    assert module.main([]) == 1
    assert "sudo apt install libglfw3" in capsys.readouterr().err


def test_the_demuvo_command_passes_gui_on_to_the_sandbox(monkeypatch, capsys):
    """`demuvo gui missing.toml` reaches the gui command and fails in its words."""
    assert cli.main(["gui", "missing.toml"]) == 1
    assert capsys.readouterr().err.startswith("demuvo: ")
    monkeypatch.setattr("sys.argv", ["demuvo", "gui", "missing.toml"])
    assert cli.main() == 1


SNIPPET = """
import sys
sys.path.insert(0, {src!r})
from demuvo.gui.app import main
sys.exit(main([{world!r}, "--bench", "--seconds", "1.5"]))
"""


def test_a_real_window_opens_runs_and_closes_on_a_virtual_display():
    """The whole chain: GLFW, OpenGL, the shell and the sandbox, for 1.5 s."""
    result = on_virtual_display(
        SNIPPET.replace("{world!r}", repr(str(EXAMPLES / "demo.toml")))
    )
    assert result.returncode == 0, result.stderr


def test_python_dash_m_runs_the_same_command(monkeypatch, capsys):
    """`python -m demuvo.gui missing.toml` exits with code 1 and the plain message."""
    monkeypatch.setattr("sys.argv", ["demuvo", "missing.toml"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("demuvo.gui", run_name="__main__")
    assert stopped.value.code == 1
    assert capsys.readouterr().err.startswith("demuvo: ")
