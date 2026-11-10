"""Tests for the GLFW binding: against a stand-in library, and against the real one."""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from demuvo.gui import glfw as binding
from demuvo.gui.glfw import Glfw

SRC = Path(__file__).resolve().parents[3]


def require_library():
    """Skip the test if the system's GLFW is not installed."""
    try:
        binding.load_library()
    except RuntimeError:
        pytest.skip("libglfw3 is not installed")


def test_every_call_gets_its_argument_and_result_types():
    """The binding declares the types of all functions it uses."""
    lib = MagicMock()
    Glfw(lib)
    for name, (arguments, result) in binding.FUNCTIONS.items():
        function = getattr(lib, name)
        assert function.argtypes == list(arguments) and function.restype == result


def test_start_and_stop_report_whether_a_display_was_found():
    """The integer from C becomes a bool."""
    lib = MagicMock()
    lib.glfwInit.return_value = 1
    glfw = Glfw(lib)
    assert glfw.init() is True
    lib.glfwInit.return_value = 0
    assert glfw.init() is False
    glfw.terminate()
    lib.glfwTerminate.assert_called_once_with()


def test_a_window_is_opened_with_a_title_and_a_visibility_hint():
    """The title goes as bytes; a null handle is None."""
    lib = MagicMock()
    glfw = Glfw(lib)
    lib.glfwCreateWindow.return_value = 99
    assert glfw.create_window(320, 200, "demuvo", visible=False) == 99
    lib.glfwWindowHint.assert_called_with(binding.VISIBLE, binding.FALSE)
    lib.glfwCreateWindow.assert_called_with(320, 200, b"demuvo", None, None)
    glfw.create_window(1, 1, "x")
    lib.glfwWindowHint.assert_called_with(binding.VISIBLE, binding.TRUE)
    lib.glfwCreateWindow.return_value = 0
    assert glfw.create_window(1, 1, "x") is None


def test_the_context_loop_calls_map_to_glfw():
    """Current context with and without vsync, swap, close flags, events, destroy."""
    lib = MagicMock()
    glfw = Glfw(lib)
    glfw.make_current(7)
    lib.glfwSwapInterval.assert_called_with(1)
    glfw.make_current(7, vsync=False)
    lib.glfwSwapInterval.assert_called_with(0)
    glfw.swap_buffers(7)
    lib.glfwWindowShouldClose.return_value = 1
    assert glfw.should_close(7) is True
    glfw.set_should_close(7)
    lib.glfwSetWindowShouldClose.assert_called_with(7, binding.TRUE)
    glfw.set_should_close(7, False)
    lib.glfwSetWindowShouldClose.assert_called_with(7, binding.FALSE)
    glfw.wait_events(0.25)
    lib.glfwWaitEventsTimeout.assert_called_with(0.25)
    glfw.poll_events()
    glfw.destroy_window(7)
    lib.glfwDestroyWindow.assert_called_with(7)


def test_sizes_and_the_pointer_come_back_as_pairs():
    """Two numbers each, whatever the stand-in leaves in them."""
    glfw = Glfw(MagicMock())
    assert len(glfw.framebuffer_size(1)) == 2
    assert len(glfw.window_size(1)) == 2
    assert len(glfw.cursor_pos(1)) == 2


def test_the_clipboard_text_is_decoded_and_none_is_an_empty_string():
    """Bytes in, str out; and text out goes as bytes."""
    lib = MagicMock()
    glfw = Glfw(lib)
    lib.glfwGetClipboardString.return_value = "1.5 m".encode()
    assert glfw.clipboard(1) == "1.5 m"
    lib.glfwGetClipboardString.return_value = None
    assert glfw.clipboard(1) == ""
    glfw.set_clipboard(1, "é")
    lib.glfwSetClipboardString.assert_called_with(1, "é".encode())


def test_events_reach_the_python_handlers_without_the_window_argument():
    """Each callback is a C function pointer that calls the handler with the rest."""
    lib = MagicMock()
    glfw = Glfw(lib)
    seen = {}
    glfw.set_callbacks(
        5,
        cursor=lambda x, y: seen.update(cursor=(x, y)),
        button=lambda *a: seen.update(button=a),
        scroll=lambda dx, dy: seen.update(scroll=(dx, dy)),
        key=lambda *a: seen.update(key=a),
        char=lambda code: seen.update(char=code),
        refresh=lambda: seen.update(refresh=True),
        resize=lambda w, h: seen.update(resize=(w, h)),
    )
    registered = {}
    for name, (setter, _) in binding.CALLBACKS.items():
        registered[name] = getattr(lib, setter).call_args.args[1]
    registered["cursor"](5, 1.5, 2.5)
    registered["button"](5, 0, binding.PRESS, 2)
    registered["scroll"](5, 0.0, -1.0)
    registered["key"](5, 65, 30, binding.REPEAT, 1)
    registered["char"](5, 0x1F600)
    registered["refresh"](5)
    registered["resize"](5, 800, 600)
    assert seen == {
        "cursor": (1.5, 2.5),
        "button": (0, binding.PRESS, 2),
        "scroll": (0.0, -1.0),
        "key": (65, 30, binding.REPEAT, 1),
        "char": 0x1F600,
        "refresh": True,
        "resize": (800, 600),
    }
    assert len(registered) == 7


def test_a_name_that_is_not_an_event_is_a_key_error():
    """No such callback."""
    with pytest.raises(KeyError):
        Glfw(MagicMock()).set_callbacks(1, mouse=print)


def test_the_loader_says_how_to_install_glfw_when_it_is_missing(monkeypatch):
    """No library anywhere gives a plain message."""

    def fail(name):
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", fail)
    monkeypatch.setenv(binding.ENVIRONMENT, "/nowhere/libglfw.so")
    with pytest.raises(RuntimeError, match="sudo apt install libglfw3"):
        binding.load_library()


def test_the_loader_takes_the_library_named_in_the_environment_first(monkeypatch):
    """A path in the environment variable is tried before the system's."""
    tried = []

    def record(name):
        tried.append(name)
        return MagicMock()

    monkeypatch.setattr(ctypes, "CDLL", record)
    monkeypatch.setenv(binding.ENVIRONMENT, "/opt/libglfw.so.3")
    binding.load_library()
    assert tried == ["/opt/libglfw.so.3"]


SNIPPET = """
import sys
sys.path.insert(0, {src!r})
from demuvo.gui.glfw import Glfw
glfw = Glfw()
assert glfw.init()
window = glfw.create_window(200, 120, "test", visible=False)
assert window
glfw.make_current(window, vsync=False)
assert glfw.framebuffer_size(window) == (200, 120), glfw.framebuffer_size(window)
assert glfw.window_size(window) == (200, 120)
assert len(glfw.cursor_pos(window)) == 2
glfw.set_callbacks(window, refresh=lambda: None, resize=lambda w, h: None)
glfw.poll_events()
glfw.wait_events(0.01)
assert not glfw.should_close(window)
glfw.set_should_close(window)
assert glfw.should_close(window)
glfw.swap_buffers(window)
glfw.destroy_window(window)
glfw.terminate()
print("window ok")
"""


@pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="needs a virtual display")
def test_the_real_library_opens_a_window_on_a_virtual_display():
    """A hidden window of 200 by 120: sizes, events, close flag, destroy."""
    require_library()
    result = subprocess.run(
        ["xvfb-run", "-a", sys.executable, "-I", "-c", SNIPPET.format(src=str(SRC))],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert "window ok" in result.stdout, result.stderr
