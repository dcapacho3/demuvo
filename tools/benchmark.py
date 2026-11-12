"""Measure the step rate of a world, with more robots if asked.

    python tools/benchmark.py
    python tools/benchmark.py examples/worlds/demo.toml --seconds 5
    python tools/benchmark.py examples/worlds/demo.toml --robots 64

Robots added with `--robots` go in at free places, the way the sandbox adds
them, and have no tasks, so they stand still. Use a crowd world to measure
robots that move.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

from demuvo import WorldError, load_world
from demuvo.edit import add_robot
from demuvo.runner import Run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "examples" / "worlds" / "demo.toml"


def _parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "world",
        nargs="?",
        default=str(DEFAULT),
        help="world file (default: the demo world)",
    )
    parser.add_argument(
        "--robots",
        type=int,
        metavar="N",
        help="grow the world to N robots at free places",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        metavar="S",
        help="simulated seconds to run (default: 5)",
    )
    return parser


def _name(world) -> str:
    """A name no robot of the world has."""
    taken = {robot.name for robot in world.robots}
    return next(f"bench{k}" for k in range(1, 100_000) if f"bench{k}" not in taken)


def _grow(world, count: int):
    """Add disc robots on a one-metre grid until the world has `count` robots."""
    if count <= len(world.robots):
        return world
    side = max(world.size[0], world.size[1], math.ceil(math.sqrt(count)) * 1.2 + 2)
    world = replace(world, size=(side, side))
    x = 1.0
    while len(world.robots) < count and x < side:
        y = 1.0
        while len(world.robots) < count and y < side:
            try:
                world = add_robot(world, _name(world), (x, y, 0.0))
            except WorldError:
                pass
            y += 1.0
        x += 1.0
    if len(world.robots) < count:
        raise SystemExit(f"could not place {count} robots in a {side:g} m world")
    return world


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print its lines."""
    args = _parser().parse_args(argv)
    world = load_world(args.world)
    if args.robots is not None:
        world = _grow(world, args.robots)
    run = Run(world)
    steps = max(1, round(args.seconds / run.sim.dt))
    start = time.perf_counter()
    for _ in range(steps):
        run.step()
    elapsed = time.perf_counter() - start
    robots = len(run.sim.robots)
    print(f"{robots} robots, {steps} steps of {run.sim.dt:g} s")
    print(
        f"{steps / elapsed:.0f} steps/s, {steps * robots / elapsed:.0f} robot-steps/s"
    )
    print(f"{1e3 * elapsed / steps:.2f} ms per step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
