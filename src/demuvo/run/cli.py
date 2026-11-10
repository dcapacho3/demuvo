"""The demuvo command - open a world file in the sandbox window.

demuvo [world.toml] [--bench] [--check] [--dt S] [--result] [--seconds S]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from ..gui.app import show
from ..gui.session import Session
from ..runner import Run
from ..world import load_world


def _count(n: int, word: str) -> str:
    """'1 robot', '2 robots'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def _parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        prog="demuvo",
        description="Open a world file in the sandbox window, or a blank world.",
    )
    parser.add_argument("world", nargs="?", help="path of the world file (TOML)")
    parser.add_argument(
        "--bench", action="store_true", help="open the world as a fixed benchmark"
    )
    parser.add_argument(
        "--check", action="store_true", help="check the file and open nothing"
    )
    parser.add_argument(
        "--dt", type=float, help="step length in s; the default is the file's"
    )
    parser.add_argument(
        "--result",
        action="store_true",
        help="print what each robot did when the window closes",
    )
    parser.add_argument(
        "--one-process",
        action="store_true",
        help="simulate in the window's process, not in one of its own",
    )
    parser.add_argument(
        "--seconds", type=float, help="close the window after this long"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: the arguments after the command name; None reads them from the
            command line.

    Returns:
        The exit code: 0 on success, 1 if the file, the step length or the window
        is wrong. A malformed option ends the program with code 2, as argparse does.
    """
    args = _parser().parse_args(argv)
    where = f"{args.world}: " if args.world else ""
    try:
        world = load_world(args.world) if args.world else None
        if world is not None and args.dt is not None:
            world = replace(world, dt=args.dt)
        if world is not None:
            Run(world)  # the world must run, whatever happens next
        if args.check:
            return _check(world, args.world)
        mode = "benchmark" if args.bench else "sandbox"
        session = Session(world, mode, remote=not args.one_process)
    except (OSError, ValueError) as error:
        print(f"demuvo: {where}{error}", file=sys.stderr)
        return 1
    try:
        code = show(session, args.seconds)
        if code == 0 and args.result:
            lines = session.run.result()
            if lines:
                print("\n".join(lines))
    finally:
        session.close()
    return code


def _check(world, name: str) -> int:
    """Print a summary of a world that loaded and ran; there is none without a file."""
    if world is None:
        print("demuvo: --check needs a world file", file=sys.stderr)
        return 1
    obstacles, robots = (
        _count(len(world.obstacles), "obstacle"),
        _count(len(world.robots), "robot"),
    )
    print(f"{name}: ok, {obstacles}, {robots}")
    return 0
