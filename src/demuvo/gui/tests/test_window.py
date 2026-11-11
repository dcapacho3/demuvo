"""Tests for the window: events, scaling, uploads and the loop, then a real one."""

import ctypes
from unittest.mock import MagicMock

import numpy as np
import pytest

from demuvo.gui import glfw as binding
from demuvo.gui import window as module
from demuvo.gui.glfw import PRESS, RELEASE, REPEAT, Glfw
from demuvo.gui.window import Window

from .fakes import on_virtual_display


class FakeApp:
    """An application that records what the window tells it."""

    def clip_get(self):
        """The window replaces this with the clipboard."""
        return ""

    def clip_set(self, text):
        """The window replaces this with the clipboard."""

    def __init__(self, frames=1):
        """Start dirty, with `frames` frames to show, then ask to quit."""
        self.dirty, self.dirty_rect, self.quit_requested = True, None, False
        self.events, self.ticks, self.frames, self.shown = [], 0, frames, []

    def mouse_move(self, x, y):
        """Note a move."""
        self.events.append(("move", x, y))

    def mouse_press(self, event):
        """Note a press."""
        self.events.append(("press", event))

    def mouse_release(self, event):
        """Note a release."""
        self.events.append(("release", event))

    def mouse_scroll(self, dy):
        """Note the wheel."""
        self.events.append(("scroll", dy))

    def key(self, key, action, mods):
        """Note a key."""
        self.events.append(("key", key, action, mods))

    def char(self, char):
        """Note a character."""
        self.events.append(("char", char))

    def invalidate_all(self):
        """Note an expose or resize."""
        self.events.append(("all",))

    def busy(self):
        """Busy until the frames are shown."""
        return len(self.shown) < self.frames

    def tick(self, _now):
        """Count a turn, and quit after the frames are shown."""
        self.ticks += 1
        self.quit_requested = len(self.shown) >= self.frames

    def render(self, width, height):
        """A frame of the size asked."""
        self.shown.append((width, height))
        self.dirty = len(self.shown) < self.frames
        return np.zeros((height, width, 3), np.uint8)


def build(scale=1.0):
    """A window over a stand-in GLFW and OpenGL; `scale` is the screen's scaling."""
    glfw = Glfw(MagicMock())
    glfw.lib.glfwInit.return_value = 1
    glfw.lib.glfwCreateWindow.return_value = 11
    glfw.framebuffer_size = lambda handle: (int(200 * scale), int(100 * scale))
    glfw.window_size = lambda handle: (200, 100)
    glfw.cursor_pos = lambda handle: (10.0, 20.0)
    gl = MagicMock()
    return Window("demuvo", 200, 100, glfw=glfw, gl=gl), glfw, gl


def handlers(glfw):
    """The Python handlers the window registered, by event name."""
    found = {
        name: getattr(glfw.lib, setter).call_args.args[1]
        for name, (setter, _) in binding.CALLBACKS.items()
    }
    return {name: lambda *args, cb=cb: cb(11, *args) for name, cb in found.items()}


def test_the_window_opens_current_and_makes_a_texture():
    """GLFW started, a window made current, a texture set to nearest filtering."""
    window, glfw, gl = build()
    glfw.lib.glfwMakeContextCurrent.assert_called_with(11)
    gl.glGenTextures.assert_called_once()
    assert gl.glTexParameteri.call_count == 2
    assert window.texture_size == (0, 0)


def test_no_display_or_no_window_is_a_plain_error():
    """GLFW that will not start, and a window that will not open."""
    glfw = Glfw(MagicMock())
    glfw.lib.glfwInit.return_value = 0
    with pytest.raises(RuntimeError, match="is a display available"):
        Window("x", 10, 10, glfw=glfw, gl=MagicMock())
    glfw.lib.glfwInit.return_value = 1
    glfw.lib.glfwCreateWindow.return_value = 0
    with pytest.raises(RuntimeError, match="could not open a window"):
        Window("x", 10, 10, glfw=glfw, gl=MagicMock())
    glfw.lib.glfwTerminate.assert_called_once()


def test_the_scale_is_the_frame_over_the_screen_and_one_if_the_screen_is_zero():
    """A doubling screen gives 2; a window of zero width gives 1."""
    window, glfw, _ = build(scale=2.0)
    assert window.scale() == 2.0 and window.framebuffer_size() == (400, 200)
    glfw.window_size = lambda handle: (0, 0)
    assert window.scale() == 1.0


def test_pointer_events_reach_the_application_in_frame_pixels():
    """A move at (10, 20) on a doubling screen is (20, 40); a press has modifiers."""
    window, glfw, _ = build(scale=2.0)
    app = FakeApp()
    window.attach(app)
    on = handlers(glfw)
    on["cursor"](10.0, 20.0)
    on["button"](0, PRESS, 3)
    on["button"](1, RELEASE, 0)
    on["scroll"](0.0, -1.0)
    assert app.events[0] == ("move", 20.0, 40.0)
    kind, press = app.events[1]
    assert kind == "press" and (press.x, press.y) == (20.0, 40.0)
    assert press.button == 0 and press.shift and press.ctrl
    kind, release = app.events[2]
    assert kind == "release" and release.button == 1 and not release.shift
    assert app.events[3] == ("scroll", -1.0)


def test_keys_pass_on_press_and_repeat_but_not_release_and_text_is_a_character():
    """Key 65 pressed, repeated, released; a typed code point becomes a character."""
    window, glfw, _ = build()
    app = FakeApp()
    window.attach(app)
    on = handlers(glfw)
    for action in (PRESS, REPEAT, RELEASE):
        on["key"](65, 30, action, 2)
    on["char"](ord("é"))
    assert app.events == [("key", 65, PRESS, 2), ("key", 65, REPEAT, 2), ("char", "é")]


def test_expose_and_resize_ask_the_application_to_redraw_everything():
    """Both events call `invalidate_all`."""
    window, glfw, _ = build()
    app = FakeApp()
    window.attach(app)
    handlers(glfw)["refresh"]()
    handlers(glfw)["resize"](300, 200)
    assert app.events == [("all",), ("all",)]


def test_the_application_gets_the_clipboard_through_the_window():
    """Both functions are set, and go to GLFW."""
    window, glfw, _ = build()
    app = FakeApp()
    glfw.lib.glfwGetClipboardString.return_value = b"2.5"
    window.attach(app)
    assert app.clip_get() == "2.5"
    app.clip_set("7")
    glfw.lib.glfwSetClipboardString.assert_called_with(11, b"7")


def test_the_first_frame_makes_the_texture_and_later_ones_upload_a_box():
    """Same size: a sub-image of the dirty box, with the skips set and cleared."""
    window, _, gl = build()
    frame = np.zeros((100, 200, 3), np.uint8)
    window.upload(frame)
    gl.glTexImage2D.assert_called_once()
    assert window.texture_size == (200, 100)
    window.upload(frame, (20, 10, 60, 50))
    args = gl.glTexSubImage2D.call_args.args
    assert args[2:6] == (20, 10, 40, 40)
    gl.glPixelStorei.assert_any_call(module.GL_UNPACK_SKIP_PIXELS, 20)
    gl.glPixelStorei.assert_any_call(module.GL_UNPACK_SKIP_PIXELS, 0)
    window.upload(frame)
    assert gl.glTexSubImage2D.call_args.args[2:6] == (0, 0, 200, 100)
    window.upload(np.zeros((50, 80, 3), np.uint8), (0, 0, 10, 10))
    assert gl.glTexImage2D.call_count == 2 and window.texture_size == (80, 50)


def test_drawing_covers_the_window_with_one_quad():
    """A viewport of the frame size, a clear, and four textured corners."""
    window, _, gl = build(scale=2.0)
    window.draw()
    gl.glViewport.assert_called_with(0, 0, 400, 200)
    assert gl.glVertex2f.call_count == 4 and gl.glTexCoord2f.call_count == 4


def test_show_frame_uploads_draws_and_swaps():
    """The three steps, in order."""
    window, glfw, gl = build()
    window.show_frame(np.zeros((100, 200, 3), np.uint8))
    gl.glBegin.assert_called_once()
    glfw.lib.glfwSwapBuffers.assert_called_once_with(11)


def test_the_loop_waits_ticks_shows_dirty_frames_and_closes():
    """Three frames, then the application quits and the window is closed."""
    window, glfw, _ = build()
    app = FakeApp(frames=3)
    glfw.lib.glfwWindowShouldClose.side_effect = [0, 0, 0, 0, 0, 1]

    def quit_flag(_handle, _value):
        glfw.lib.glfwWindowShouldClose.side_effect = [1]

    glfw.lib.glfwSetWindowShouldClose.side_effect = quit_flag
    window.run(app)
    assert len(app.shown) == 3 and app.ticks >= 3
    glfw.lib.glfwWaitEventsTimeout.assert_any_call(module.BUSY_WAIT)
    glfw.lib.glfwDestroyWindow.assert_called_once_with(11)
    glfw.lib.glfwTerminate.assert_called()


def test_an_idle_application_waits_long_and_a_time_limit_ends_the_loop():
    """Not busy and not dirty: the long wait, and the loop stops at the limit."""
    window, glfw, _ = build()
    app = FakeApp(frames=0)
    app.dirty = False
    glfw.lib.glfwWindowShouldClose.return_value = 0
    window.run(app, max_seconds=0.0)
    glfw.lib.glfwWaitEventsTimeout.assert_called_with(module.IDLE_WAIT)
    assert not app.shown


def test_the_opengl_loader_names_the_package_when_it_is_missing(monkeypatch):
    """No library anywhere gives a plain message."""

    def fail(name):
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", fail)
    with pytest.raises(RuntimeError, match="sudo apt install libgl1"):
        module.load_gl()


SNIPPET = """
import sys
sys.path.insert(0, {src!r})
import numpy as np
from demuvo.gui.window import Window
window = Window("test", 64, 48, visible=False)
frame = np.zeros((48, 64, 3), np.uint8)
frame[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 4
frame[..., 1] = np.arange(48, dtype=np.uint8)[:, None] * 5
window.upload(frame)
window.draw()
import ctypes
size = 64 * 48 * 3
buffer = (ctypes.c_ubyte * size)()
window.gl.glReadPixels(0, 0, 64, 48, 0x1907, 0x1401, buffer)
read = np.frombuffer(buffer, np.uint8).reshape(48, 64, 3)[::-1]
difference = np.abs(read.astype(int) - frame.astype(int)).max()
assert difference <= 1, difference
frame[10:20, 10:30] = (255, 255, 255)
window.upload(frame, (10, 10, 30, 20))
window.draw()
window.gl.glReadPixels(0, 0, 64, 48, 0x1907, 0x1401, buffer)
read = np.frombuffer(buffer, np.uint8).reshape(48, 64, 3)[::-1]
assert np.abs(read.astype(int) - frame.astype(int)).max() <= 1
window.show_frame(frame)
window.close()
print("frame ok")
"""


def test_a_real_window_shows_exactly_the_frame_it_was_given_and_a_partial_update():
    """Upload, draw, read back; then change a box, upload just it, read again."""
    try:
        binding.load_library()
        module.load_gl()
    except RuntimeError:
        pytest.skip("libglfw3 or libgl1 is not installed")
    result = on_virtual_display(SNIPPET)
    assert "frame ok" in result.stdout, result.stderr


def test_the_loop_uploads_the_box_that_was_damaged_before_the_render_cleared_it():
    """A shell forgets its damage when it renders, so the loop reads it first."""
    window, glfw, _ = build()
    app = FakeApp(frames=1)
    app.dirty_rect = (-8, 20, 50, 40)
    render = app.render

    def render_and_clear(width, height):
        app.dirty_rect = None
        return render(width, height)

    app.render = render_and_clear
    window.show_frame = MagicMock()
    glfw.lib.glfwWindowShouldClose.side_effect = [0, 0, 1, 1]
    window.run(app)
    assert window.show_frame.call_args.args[1] == (-8, 20, 50, 40)


def test_an_upload_box_is_kept_inside_the_frame():
    """A box that spills over the edge is cut to the frame."""
    window, _, gl = build()
    frame = np.zeros((50, 80, 3), np.uint8)
    window.upload(frame)
    window.upload(frame, (-8, -4, 120, 70))
    args = gl.glTexSubImage2D.call_args.args
    assert args[2:6] == (0, 0, 80, 50)
