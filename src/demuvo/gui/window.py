"""The native window - GLFW opens it, and a frame is shown as one OpenGL texture.

The renderer draws the whole window into a numpy array. Showing it needs only one
texture and one quad, through the system's libGL called with ctypes, and the part
that changed is uploaded alone. The events GLFW reports go to the application.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .glfw import PRESS, REPEAT, Glfw, open_library
from .imgio import read_png
from .widgets import Event

GL_TEXTURE_2D = 0x0DE1
GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE, GL_QUADS = 0x1907, 0x8051, 0x1401, 0x0007
GL_NEAREST, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER = 0x2600, 0x2801, 0x2800
GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH = 0x0CF5, 0x0CF2
GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS = 0x0CF4, 0x0CF3
GL_COLOR_BUFFER_BIT = 0x4000
# The corners of the quad: texture x and y, then screen x and y.
QUAD = ((0, 0, -1, 1), (1, 0, 1, 1), (1, 1, 1, -1), (0, 1, -1, -1))
# The icon is shipped as pictures of these sizes, in pixels.
ICON_SIZES = (32, 64, 128, 180)
ASSETS = Path(__file__).parent / "assets"
# Seconds to wait for an event: briefly while something moves, long when idle.
BUSY_WAIT, IDLE_WAIT = 0.004, 0.25
# The modifier bits GLFW reports.
SHIFT_BIT, CTRL_BIT = 1, 2


class Application(Protocol):
    """What the window needs from the program it shows.

    Attributes:
        dirty: True when a new frame is needed.
        dirty_rect: (x0, y0, x1, y1) of what changed, or None for all of it.
        quit_requested: True when the program wants the window closed.
        title: the text for the window's title bar.
        clip_get: a function the window sets, that returns the clipboard text.
        clip_set: a function the window sets, that puts text on the clipboard.
    """

    clip_get: object
    clip_set: object
    dirty: bool
    dirty_rect: tuple[int, int, int, int] | None
    quit_requested: bool
    title: str

    def mouse_move(self, x: float, y: float) -> None:
        """The pointer moved to a pixel of the frame."""

    def mouse_press(self, event: Event) -> None:
        """A mouse button went down."""

    def mouse_release(self, event: Event) -> None:
        """A mouse button came up."""

    def mouse_scroll(self, dy: float) -> None:
        """The wheel turned."""

    def key(self, key: int, action: int, mods: int) -> None:
        """A key went down or repeated."""

    def char(self, char: str) -> None:
        """A character was typed."""

    def invalidate_all(self) -> None:
        """Everything needs a redraw, as the window was exposed or resized."""

    def busy(self) -> bool:
        """True while something is moving, so the loop should not sleep long."""

    def tick(self, now: float) -> None:
        """Advance animations and the run to the time `now`, in seconds."""

    def render(self, width: int, height: int) -> np.ndarray:
        """The frame as uint8 RGB, shape (height, width, 3)."""


def load_gl() -> ctypes.CDLL:
    """Open the system's OpenGL library.

    Raises:
        RuntimeError: if it is not installed, with how to install it.
    """
    names = [ctypes.util.find_library("GL"), "libGL.so.1"]
    return open_library(
        names, "OpenGL is not installed; on Ubuntu run: sudo apt install libgl1"
    )


class Window:
    """A window that shows the frames of an application and gives it the events.

    Args:
        title: the title of the window.
        width: width in screen units.
        height: height in screen units.
        visible: False opens it hidden, for tests.
        glfw: a `Glfw`; None opens the system's.
        gl: an OpenGL library; None opens the system's.

    Raises:
        RuntimeError: if GLFW or OpenGL is missing, there is no display, or the window
            cannot be opened.
    """

    def __init__(
        self,
        title: str,
        width: int,
        height: int,
        visible: bool = True,
        *,
        glfw: Glfw | None = None,
        gl=None,
    ) -> None:
        """Open the window and make a texture for the frame."""
        self.glfw = glfw or Glfw()
        if not self.glfw.init():
            raise RuntimeError("could not start GLFW; is a display available?")
        self.handle = self.glfw.create_window(width, height, title, visible)
        if not self.handle:
            self.glfw.terminate()
            raise RuntimeError("could not open a window")
        self.glfw.make_current(self.handle)
        self._set_icon()
        self.gl = gl or load_gl()
        self.app: Application | None = None
        self.texture = ctypes.c_uint(0)
        self.texture_size = (0, 0)
        self._prepare()

    def _set_icon(self) -> None:
        """Put the demuvo icon on the window; a window without one is still fine."""
        try:
            images = [
                (size, size, read_png(ASSETS / f"icon_{size}.png").tobytes())
                for size in ICON_SIZES
            ]
            self.glfw.set_icon(self.handle, images)
        except (OSError, ValueError, AttributeError):
            pass

    def _prepare(self) -> None:
        """Make the texture that holds the frame and set how it is read."""
        gl = self.gl
        gl.glGenTextures(1, ctypes.byref(self.texture))
        gl.glBindTexture(GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        gl.glEnable(GL_TEXTURE_2D)
        gl.glTexCoord2f.argtypes = [ctypes.c_float] * 2
        gl.glVertex2f.argtypes = [ctypes.c_float] * 2

    def scale(self) -> float:
        """Frame pixels per screen unit: 2 on a screen that doubles everything."""
        frame, _ = self.glfw.framebuffer_size(self.handle)
        screen, _ = self.glfw.window_size(self.handle)
        return frame / screen if screen else 1.0

    def framebuffer_size(self) -> tuple[int, int]:
        """The size of the frame to draw, in pixels."""
        return self.glfw.framebuffer_size(self.handle)

    def attach(self, app: Application) -> None:
        """Send every event of the window to an application, in pixels of the frame."""
        self.app = app
        app.clip_get = lambda: self.glfw.clipboard(self.handle)
        app.clip_set = lambda text: self.glfw.set_clipboard(self.handle, text)
        self.glfw.set_callbacks(
            self.handle,
            cursor=lambda x, y: app.mouse_move(x * self.scale(), y * self.scale()),
            button=self._button,
            scroll=lambda _dx, dy: app.mouse_scroll(dy),
            key=lambda key, _scan, action, mods: self._key(key, action, mods),
            char=lambda code: app.char(chr(code)),
            refresh=app.invalidate_all,
            resize=lambda _w, _h: app.invalidate_all(),
        )

    def _key(self, key: int, action: int, mods: int) -> None:
        """Pass on a key press or repeat; a release is not used."""
        if action in (PRESS, REPEAT):
            self.app.key(key, action, mods)

    def _button(self, button: int, action: int, mods: int) -> None:
        """Pass on a mouse button with the pointer's place in frame pixels."""
        x, y = self.glfw.cursor_pos(self.handle)
        scale = self.scale()
        event = Event(
            x * scale,
            y * scale,
            button=button,
            shift=bool(mods & SHIFT_BIT),
            ctrl=bool(mods & CTRL_BIT),
        )
        if action == PRESS:
            self.app.mouse_press(event)
        else:
            self.app.mouse_release(event)

    def upload(self, frame: np.ndarray, dirty=None) -> None:
        """Put a frame in the texture.

        Args:
            frame: uint8 RGB, shape (height, width, 3).
            dirty: (x0, y0, x1, y1) of what changed. If the frame has the size of the
                last one, only that box is uploaded; None uploads all of it.
        """
        gl = self.gl
        height, width = frame.shape[:2]
        frame = np.ascontiguousarray(frame)
        gl.glBindTexture(GL_TEXTURE_2D, self.texture)
        pointer = frame.ctypes.data_as(ctypes.c_void_p)
        if self.texture_size != (width, height):
            gl.glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB8,
                width,
                height,
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                pointer,
            )
            self.texture_size = (width, height)
            return
        x0, y0, x1, y1 = (0, 0, width, height) if dirty is None else dirty
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        gl.glPixelStorei(GL_UNPACK_ROW_LENGTH, width)
        gl.glPixelStorei(GL_UNPACK_SKIP_PIXELS, x0)
        gl.glPixelStorei(GL_UNPACK_SKIP_ROWS, y0)
        gl.glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            pointer,
        )
        for name in (GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS):
            gl.glPixelStorei(name, 0)

    def draw(self) -> None:
        """Draw the texture over the whole window."""
        gl = self.gl
        width, height = self.framebuffer_size()
        gl.glViewport(0, 0, width, height)
        gl.glClear(GL_COLOR_BUFFER_BIT)
        gl.glBegin(GL_QUADS)
        for tx, ty, vx, vy in QUAD:
            gl.glTexCoord2f(tx, ty)
            gl.glVertex2f(vx, vy)
        gl.glEnd()

    def show_frame(self, frame: np.ndarray, dirty=None) -> None:
        """Show a uint8 RGB frame, uploading only the part that changed."""
        self.upload(frame, dirty)
        self.draw()
        self.glfw.swap_buffers(self.handle)

    def run(self, app: Application, max_seconds: float | None = None) -> None:
        """Run until the window is closed, the application quits, or the time is up.

        Each turn waits for events (briefly while something moves), lets the
        application advance, and shows a new frame if it needs one.
        """
        self.attach(app)
        started = time.perf_counter()
        title = None
        while not self.glfw.should_close(self.handle):
            self.glfw.wait_events(BUSY_WAIT if app.busy() else IDLE_WAIT)
            app.tick(time.perf_counter())
            if app.title != title:
                title = app.title
                self.glfw.set_title(self.handle, title)
            if app.quit_requested:
                self.glfw.set_should_close(self.handle)
            if app.dirty:
                # render clears the damage, so the box is read first
                box = app.dirty_rect
                self.show_frame(app.render(*self.framebuffer_size()), box)
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                break
        self.close()

    def close(self) -> None:
        """Close the window and GLFW."""
        self.glfw.destroy_window(self.handle)
        self.glfw.terminate()
