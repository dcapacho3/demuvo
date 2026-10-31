"""The demuvo command - open a world file and watch it."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from ..nav.navigation import Navigation
from ..sim.simulation import Simulation
from ..world import load_world


def _count(n: int, word: str) -> str:
    """'1 robot', '2 robots'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def main(argv: list[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: the arguments after the command name; None reads them from the
            command line.

    Returns:
        The exit code: 0 on success, 1 if the file or the step length is wrong.
        A missing or malformed option ends the program with code 2, as argparse
        does.
    """
    parser = argparse.ArgumentParser(
        prog="demuvo", description="Open a world file and watch it."
    )
    parser.add_argument("world", help="path of the world file (TOML)")
    parser.add_argument(
        "--check", action="store_true", help="check the file and open nothing"
    )
    parser.add_argument(
        "--dt", type=float, help="step length in s; the default is the file's"
    )
    parser.add_argument(
        "--no-trail", action="store_true", help="do not draw the path of each robot"
    )
    args = parser.parse_args(argv)

    try:
        world = load_world(args.world)
        sim = Simulation(world, args.dt)
        commands = Navigation(world)
        commands.check(sim.dt)
    except (OSError, ValueError) as error:
        print(f"demuvo: {args.world}: {error}", file=sys.stderr)
        return 1
    if args.check:
        print(
            f"{args.world}: ok, {_count(len(world.obstacles), 'obstacle')}, "
            f"{_count(len(world.robots), 'robot')}"
        )
        return 0

    # The view is an optional extra, so it is loaded only when a window opens.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        view = importlib.import_module("demuvo.view")
    except ModuleNotFoundError as error:
        print(
            f"demuvo: the view needs the package '{error.name}': "
            "install the 'view' extra of demuvo",
            file=sys.stderr,
        )
        return 1
    view.play(
        sim,
        commands,
        caption=f"demuvo - {Path(args.world).name}",
        trail=0 if args.no_trail else view.TRAIL,
    )
    return 0
