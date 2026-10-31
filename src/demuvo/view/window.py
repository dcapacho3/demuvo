"""Showing a run in a window, and the keys that control it."""

from __future__ import annotations

import pygame
from pygame.locals import (
    K_DOWN,
    K_ESCAPE,
    K_RIGHT,
    K_SPACE,
    K_UP,
    KEYDOWN,
    QUIT,
)

from ..sim.simulation import Simulation
from .scene import Commands, View
from .style import (
    SPEEDS,
    TRAIL,
)


def _change_speed(speed: float, up: bool) -> float:
    """The next speed in `SPEEDS` above or below `speed`, or `speed` at the end."""
    if up:
        return min((s for s in SPEEDS if s > speed), default=speed)
    return max((s for s in SPEEDS if s < speed), default=speed)


def _on_key(view: View, key: int) -> None:
    """Apply a key that changes only the view: pause or speed."""
    if key == K_SPACE:
        view.paused = not view.paused
    elif key in (K_UP, K_DOWN):
        view.speed = _change_speed(view.speed, key == K_UP)


def _read_events(view: View) -> tuple[bool, bool]:
    """Handle the waiting events.

    Returns:
        Whether to keep running, and whether the user asked for a single step.
    """
    keep, step_once = True, False
    for event in pygame.event.get():
        if event.type == QUIT:
            keep = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                keep = False
            elif event.key == K_RIGHT:
                view.paused, step_once = True, True
            else:
                _on_key(view, event.key)
    return keep, step_once


def _window(view: View, commands: Commands, steps: int | None, caption: str) -> None:
    """Show the simulation in a window until it is closed."""
    pygame.display.init()
    screen = pygame.display.set_mode(view.size)
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    sim = view.sim
    owed = 0.0  # steps due but not yet taken, so the speed is exact on average
    running = True
    while running:
        running, step_once = _read_events(view)
        if not running:
            break
        elapsed = clock.tick(60) / 1000
        if step_once:
            owed = 1.0
        elif view.paused:
            owed = 0.0
        else:
            owed = min(owed + view.speed * elapsed / sim.dt, 20.0)
        while owed >= 1.0 and (steps is None or sim.steps < steps):
            view.step(commands)
            owed -= 1.0
        view.done = steps is not None and sim.steps >= steps
        # Paused, stepping by hand or finished: draw the pose now. Otherwise draw
        # between the last two steps, as far along as the time left over.
        fraction = 1.0 if view.paused or view.done else min(owed, 1.0)
        screen.blit(view.render(fraction), (0, 0))
        pygame.display.flip()
    pygame.display.quit()


def play(
    sim: Simulation,
    commands: Commands,
    steps: int | None = None,
    caption: str = "demuvo",
    trail: int = TRAIL,
) -> None:
    """Run the simulation in a window until it is closed.

    Args:
        sim: the simulation to run.
        commands: called before each step with the simulation; returns the
            command of each robot, by name.
        steps: number of steps to run before the run counts as done, or None
            to run for as long as the window is open.
        caption: title of the window.
        trail: past positions kept for each robot's trail; 0 draws none.
    """
    _window(View(sim, trail=trail), commands, steps, caption)
