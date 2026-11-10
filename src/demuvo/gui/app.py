"""Opening the window - show a session in a native window until it is closed."""

from __future__ import annotations

import sys

from .sandbox import Sandbox
from .session import Session
from .shell import Shell
from .window import Window

WIDTH, HEIGHT = 1280, 780


def show(session: Session, seconds: float | None = None) -> int:
    """Open the window on a session and run it until it is closed.

    Args:
        session: the world and its mode.
        seconds: close after this many seconds; None waits for the user.

    Returns:
        0 when the window was shown and closed, 1 if GLFW or OpenGL is missing,
        there is no display, or the window cannot open.
    """
    try:
        window = Window(f"demuvo - {session.mode}", WIDTH, HEIGHT)
    except RuntimeError as error:
        print(f"demuvo: {error}", file=sys.stderr)
        return 1
    width, height = window.framebuffer_size()
    window.run(Shell(Sandbox(session), width, height), seconds)
    return 0
