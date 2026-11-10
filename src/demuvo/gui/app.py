"""The `demuvo gui` command - open the sandbox window.

    demuvo gui [world.toml] [--bench]

With no file the sandbox starts with a blank world to build in. With `--bench` the
world is a fixed benchmark: nothing can be edited, and Reset repeats the run.
"""

from __future__ import annotations

import argparse
import sys

from ..verify import WorldError
from .sandbox import Sandbox
from .session import Session
from .shell import Shell
from .window import Window

WIDTH, HEIGHT = 1280, 780


def main(argv: list[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: the arguments after `gui`; None reads them from the command line.

    Returns:
        0 on success, 1 if the world cannot be read or the window cannot open.
    """
    parser = argparse.ArgumentParser(
        prog="demuvo gui", description=__doc__.split("\n", maxsplit=1)[0]
    )
    parser.add_argument(
        "world", nargs="?", help="a world file; none starts a blank sandbox"
    )
    parser.add_argument(
        "--bench", action="store_true", help="open the world as a fixed benchmark"
    )
    parser.add_argument("--seconds", type=float, help="close after this many seconds")
    args = parser.parse_args(argv)
    mode = "benchmark" if args.bench else "sandbox"
    try:
        session = Session(args.world, mode)
        window = Window(f"demuvo - {mode}", WIDTH, HEIGHT)
    except (WorldError, OSError, ValueError, RuntimeError) as error:
        print(f"demuvo: {error}", file=sys.stderr)
        return 1
    width, height = window.framebuffer_size()
    window.run(Shell(Sandbox(session), width, height), args.seconds)
    return 0
