"""A small binding to GLFW - the window and its events - through ctypes.

GLFW is a C library that opens a window and reports the mouse and keyboard. It is
in Ubuntu and Debian as `libglfw3` and has a permissive licence. Only the dozen
calls the program needs are bound, so no Python package is needed for it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

# Hints and values, as in glfw3.h.
VISIBLE = 0x00020004
# Names a desktop matches to its launcher entry and icon.
X11_CLASS_NAME, X11_INSTANCE_NAME, WAYLAND_APP_ID = 0x00024001, 0x00024002, 0x00026001
APP_NAME = b"demuvo"
# The standard pointer shapes GLFW makes, by the name a widget gives.
CURSORS = {"arrow": 0x00036001, "hresize": 0x00036005, "vresize": 0x00036006}
TRUE, FALSE = 1, 0
RELEASE, PRESS, REPEAT = 0, 1, 2
# Set this to the path of a libglfw to use that one.
ENVIRONMENT = "DEMUVO_GLFW"

_WINDOW = ctypes.c_void_p
_C_INT, _C_DOUBLE, _C_UINT = ctypes.c_int, ctypes.c_double, ctypes.c_uint
# The callback kinds: the C function name that sets it, and the signature it takes.
CALLBACKS = {
    "cursor": ("glfwSetCursorPosCallback", (_WINDOW, _C_DOUBLE, _C_DOUBLE)),
    "button": ("glfwSetMouseButtonCallback", (_WINDOW, _C_INT, _C_INT, _C_INT)),
    "scroll": ("glfwSetScrollCallback", (_WINDOW, _C_DOUBLE, _C_DOUBLE)),
    "key": ("glfwSetKeyCallback", (_WINDOW, _C_INT, _C_INT, _C_INT, _C_INT)),
    "char": ("glfwSetCharCallback", (_WINDOW, _C_UINT)),
    "refresh": ("glfwSetWindowRefreshCallback", (_WINDOW,)),
    "resize": ("glfwSetFramebufferSizeCallback", (_WINDOW, _C_INT, _C_INT)),
}


class _Image(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """GLFW's picture of an icon: a size and a pointer to RGBA rows."""

    _fields_ = [
        ("width", _C_INT),
        ("height", _C_INT),
        ("pixels", ctypes.c_void_p),
    ]


# The other calls: name, argument types and result type.
FUNCTIONS = {
    "glfwInit": ((), _C_INT),
    "glfwTerminate": ((), None),
    "glfwWindowHint": ((_C_INT, _C_INT), None),
    "glfwWindowHintString": ((_C_INT, ctypes.c_char_p), None),
    "glfwCreateWindow": (
        (_C_INT, _C_INT, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p),
        _WINDOW,
    ),
    "glfwDestroyWindow": ((_WINDOW,), None),
    "glfwMakeContextCurrent": ((_WINDOW,), None),
    "glfwSwapInterval": ((_C_INT,), None),
    "glfwSwapBuffers": ((_WINDOW,), None),
    "glfwWindowShouldClose": ((_WINDOW,), _C_INT),
    "glfwSetWindowShouldClose": ((_WINDOW, _C_INT), None),
    "glfwSetWindowTitle": ((_WINDOW, ctypes.c_char_p), None),
    "glfwCreateStandardCursor": ((_C_INT,), ctypes.c_void_p),
    "glfwSetCursor": ((_WINDOW, ctypes.c_void_p), None),
    "glfwSetWindowIcon": ((_WINDOW, _C_INT, ctypes.c_void_p), None),
    "glfwGetFramebufferSize": ((_WINDOW, ctypes.c_void_p, ctypes.c_void_p), None),
    "glfwGetWindowSize": ((_WINDOW, ctypes.c_void_p, ctypes.c_void_p), None),
    "glfwGetCursorPos": ((_WINDOW, ctypes.c_void_p, ctypes.c_void_p), None),
    "glfwWaitEventsTimeout": ((_C_DOUBLE,), None),
    "glfwPollEvents": ((), None),
    "glfwGetClipboardString": ((_WINDOW,), ctypes.c_char_p),
    "glfwSetClipboardString": ((_WINDOW, ctypes.c_char_p), None),
}


def open_library(names, install: str):
    """Open the first of some shared libraries that loads.

    Args:
        names: file names or paths; empty ones are skipped.
        install: what to tell the user if none loads.

    Raises:
        RuntimeError: if none loads.
    """
    for name in filter(None, names):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError(install)


def load_library():
    """Open the GLFW library; see `ENVIRONMENT` to name one.

    Raises:
        RuntimeError: if it cannot be found, with how to install it.
    """
    names = [
        os.environ.get(ENVIRONMENT),
        ctypes.util.find_library("glfw"),
        "libglfw.so.3",
    ]
    return open_library(
        names, "GLFW is not installed; on Ubuntu run: sudo apt install libglfw3"
    )


def _pair(function, window, kind) -> tuple:
    """Call a GLFW function that gives two numbers through pointers."""
    first, second = kind(), kind()
    function(window, ctypes.byref(first), ctypes.byref(second))
    return first.value, second.value


class Glfw:
    """The GLFW calls the window needs, as Python methods.

    Args:
        library: a ctypes library; None opens the system's.
    """

    def __init__(self, library=None) -> None:
        """Declare the types of the calls."""
        self.lib = library or load_library()
        for name, (arguments, result) in FUNCTIONS.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = list(arguments), result
        self._callbacks: list = []
        self._cursors: dict[str, object] = {}

    def init(self) -> bool:
        """Start GLFW; False if there is no display."""
        return bool(self.lib.glfwInit())

    def terminate(self) -> None:
        """Close what GLFW opened."""
        self.lib.glfwTerminate()

    def create_window(self, width: int, height: int, title: str, visible: bool = True):
        """Open a window with an OpenGL context; None if it cannot be opened."""
        self.lib.glfwWindowHint(VISIBLE, TRUE if visible else FALSE)
        # the title changes with the file, so the desktop is told the program's name
        for hint in (X11_CLASS_NAME, X11_INSTANCE_NAME, WAYLAND_APP_ID):
            self.lib.glfwWindowHintString(hint, APP_NAME)
        window = self.lib.glfwCreateWindow(width, height, title.encode(), None, None)
        return window or None

    def destroy_window(self, window) -> None:
        """Close a window."""
        self.lib.glfwDestroyWindow(window)

    def make_current(self, window, vsync: bool = True) -> None:
        """Make a window's OpenGL context current; with vsync, wait for the screen."""
        self.lib.glfwMakeContextCurrent(window)
        self.lib.glfwSwapInterval(1 if vsync else 0)

    def swap_buffers(self, window) -> None:
        """Show what was drawn."""
        self.lib.glfwSwapBuffers(window)

    def should_close(self, window) -> bool:
        """True once the user asked to close the window."""
        return bool(self.lib.glfwWindowShouldClose(window))

    def set_should_close(self, window, value: bool = True) -> None:
        """Ask the loop to end."""
        self.lib.glfwSetWindowShouldClose(window, TRUE if value else FALSE)

    def set_icon(self, window, images) -> None:
        """Give the window an icon, as pictures of several sizes.

        Args:
            window: the window.
            images: (width, height, bytes) for each size, the bytes being RGBA rows.
        """
        array = (_Image * len(images))()
        keep = []  # GLFW copies the pixels in the call, so these must live that long
        for slot, (width, height, pixels) in zip(array, images):
            buffer = ctypes.create_string_buffer(pixels, len(pixels))
            keep.append(buffer)
            slot.width, slot.height = width, height
            slot.pixels = ctypes.cast(buffer, ctypes.c_void_p).value
        self.lib.glfwSetWindowIcon(
            window, len(images), ctypes.cast(array, ctypes.c_void_p)
        )

    def set_cursor(self, window, shape: str) -> None:
        """Show the pointer as a shape - "arrow", "hresize" or "vresize"."""
        if shape not in self._cursors:
            self._cursors[shape] = self.lib.glfwCreateStandardCursor(CURSORS[shape])
        self.lib.glfwSetCursor(window, self._cursors[shape])

    def set_title(self, window, title: str) -> None:
        """Put a title on the window's bar."""
        self.lib.glfwSetWindowTitle(window, title.encode())

    def framebuffer_size(self, window) -> tuple[int, int]:
        """The size of the drawing area in pixels."""
        return _pair(self.lib.glfwGetFramebufferSize, window, _C_INT)

    def window_size(self, window) -> tuple[int, int]:
        """The window size in screen units, which a scaled screen makes smaller."""
        return _pair(self.lib.glfwGetWindowSize, window, _C_INT)

    def cursor_pos(self, window) -> tuple[float, float]:
        """Where the pointer is, in screen units."""
        return _pair(self.lib.glfwGetCursorPos, window, _C_DOUBLE)

    def wait_events(self, timeout: float) -> None:
        """Sleep until an event comes or the timeout passes, then handle events."""
        self.lib.glfwWaitEventsTimeout(timeout)

    def poll_events(self) -> None:
        """Handle the events waiting, without sleeping."""
        self.lib.glfwPollEvents()

    def clipboard(self, window) -> str:
        """The text on the clipboard, or an empty string."""
        text = self.lib.glfwGetClipboardString(window)
        return text.decode(errors="replace") if text else ""

    def set_clipboard(self, window, text: str) -> None:
        """Put text on the clipboard."""
        self.lib.glfwSetClipboardString(window, text.encode())

    def set_callbacks(self, window, **handlers) -> None:
        """Send the window's events to Python functions.

        Args:
            window: the window.
            **handlers: any of `cursor(x, y)`, `button(button, action, mods)`,
                `scroll(dx, dy)`, `key(key, scancode, action, mods)`, `char(code)`,
                `refresh()` and `resize(width, height)`. The window is not passed.

        Raises:
            KeyError: for a name that is not an event.
        """
        for name, handler in handlers.items():
            setter, types = CALLBACKS[name]
            prototype = ctypes.CFUNCTYPE(None, *types)
            callback = prototype(lambda _window, *args, h=handler: h(*args))
            self._callbacks.append(callback)  # kept, or ctypes would free it
            function = getattr(self.lib, setter)
            function.argtypes, function.restype = [_WINDOW, prototype], None
            function(window, callback)
