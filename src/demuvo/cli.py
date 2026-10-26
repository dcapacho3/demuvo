"""The demuvo command - open a world file and watch it."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from .simulation import Simulation
from .world import load_world


def main(argv: list[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: the arguments after the command name; None reads them from the
            command line.

    Returns:
        The exit code: 0 on success, 1 if the file or an option is wrong.
    """
    parser = argparse.ArgumentParser(
        prog="demuvo", description="Open a world file and watch it."
    )
    parser.add_argument("world", help="path of the world file (TOML)")
    parser.add_argument(
        "--check", action="store_true", help="check the file and open nothing"
    )
    parser.add_argument("--dt", type=float, default=0.05, help="step length in s")
    args = parser.parse_args(argv)

    try:
        world = load_world(args.world)
        sim = Simulation(world, args.dt)
    except (OSError, ValueError) as error:
        print(f"demuvo: {args.world}: {error}", file=sys.stderr)
        return 1
    if args.check:
        print(
            f"{args.world}: ok, {len(world.obstacles)} obstacles, "
            f"{len(world.robots)} robots"
        )
        return 0

    # The view is an optional extra, so it is loaded only when a window opens.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        view = importlib.import_module(".view", __package__)
    except ImportError:
        print("demuvo: the view needs pygame: pip install demuvo[view]")
        return 1
    view.play(sim, lambda _: {}, caption=f"demuvo - {Path(args.world).name}")
    return 0
