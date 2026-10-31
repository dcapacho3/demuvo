"""Saving a run as a GIF."""

from __future__ import annotations

from ..sim.simulation import Simulation
from .scene import Commands, View
from .style import (
    TRAIL,
)


def record(
    sim: Simulation,
    commands: Commands,
    steps: int,
    path: str,
    *,
    fps: int = 20,
    trail: int = TRAIL,
) -> None:
    """Run `steps` steps and save the run as a GIF, one frame every 1 / fps seconds.

    This needs no display.

    Args:
        sim: the simulation to run.
        commands: called before each step with the simulation; returns the
            command of each robot, by name.
        steps: number of steps to run.
        path: where to save the GIF.
        fps: frames per second of the GIF.
        trail: past positions kept for each robot's trail; 0 draws none.
    """
    view = View(sim, trail=trail)
    skip = max(1, round(1 / (fps * sim.dt)))
    frames = [view.image()]
    for i in range(1, steps + 1):
        view.step(commands)
        if i % skip == 0 or i == steps:
            frames.append(view.image())
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
    )
