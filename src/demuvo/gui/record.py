"""Record a world as an animated GIF or PNG frames, with no window.

The frames are the real window drawn headless, so a GIF in a README shows what the
program shows.

    python -m demuvo.gui.record WORLD OUT.gif [--seconds 8] [--fps 20] [--width 640]
        [--layers rays,local] [--panels]

An output that ends in a slash is a folder, and gets one PNG for each frame.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

from . import imgio
from .sandbox import Sandbox
from .session import Session
from .shell import Shell

# The height of a frame as a share of its width.
ASPECT = 0.62
# Extra width the panels need, and extra height the tool bar and status bar need.
PANEL_WIDTH, BARS_HEIGHT = 560, 112
# Seconds far enough ahead to finish every fade at once.
FADES_DONE = 5.0


@dataclass(frozen=True)
class Framing:
    """What the frames show.

    Attributes:
        width: width of the scene in pixels.
        layers: layers to switch on, such as ("rays", "local").
        panels: True to include the scene list and the inspector.
    """

    width: int = 640
    layers: tuple[str, ...] = ()
    panels: bool = False


def _studio(world, framing: Framing) -> tuple[Session, Sandbox, Shell]:
    """A benchmark window of the right size with the chosen layers on."""
    height = round(framing.width * ASPECT)
    session = Session(world, "benchmark")
    program = Sandbox(session)
    panels = framing.panels
    window = Shell(
        program,
        framing.width + (PANEL_WIDTH if panels else 0),
        height + (BARS_HEIGHT if panels else 0),
    )
    if not panels:
        for dock in window.layout.docks.values():
            dock.width = 0.0
        window.layout.apply()
    for name in framing.layers:
        program.set_layer(name, True)
    window.animator.tick(time.perf_counter() + FADES_DONE)
    return session, program, window


def record(
    world,
    out: str,
    seconds: float = 8.0,
    fps: int = 20,
    framing: Framing = Framing(),
) -> int:
    """Run a world for some simulated seconds and write what the window shows.

    Args:
        world: a world file, or a `World`.
        out: a `.gif` path, or a folder ending in a slash for PNG frames.
        seconds: simulated time to cover.
        fps: frames per second of the output.
        framing: the width, layers and panels; see `Framing`.

    Returns:
        The number of frames written.
    """
    session, program, window = _studio(world, framing)
    per_frame = max(1, round(1 / (fps * session.run.sim.dt)))
    frames = []
    for _ in range(round(seconds * fps)):
        session.run.step(per_frame)
        session.scene.record_trails()
        program.pane.pill.set_time(session.run.time)
        window.invalidate_all()
        frames.append(window.render().copy())
    if out.endswith(("/", "\\")):
        os.makedirs(out, exist_ok=True)
        for number, frame in enumerate(frames):
            imgio.write_png(f"{out}frame_{number:05d}.png", frame)
    else:
        imgio.write_gif(out, frames, fps)
    return len(frames)


def main(argv: list[str] | None = None) -> int:
    """Run the command; returns 0."""
    parser = argparse.ArgumentParser(
        prog="python -m demuvo.gui.record", description=__doc__
    )
    parser.add_argument("world")
    parser.add_argument("out")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--layers", default="", help="layers to switch on, such as rays,local"
    )
    parser.add_argument("--panels", action="store_true", help="include the panels")
    args = parser.parse_args(argv)
    layers = tuple(name for name in args.layers.split(",") if name)
    framing = Framing(args.width, layers, args.panels)
    count = record(args.world, args.out, args.seconds, args.fps, framing)
    print(f"{count} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
