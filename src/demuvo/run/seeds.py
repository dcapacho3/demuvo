"""Run one world under many seeds, each to the end, and tabulate the results.

The seeds are the world's own seed and the ones after it. Each run is alone in
its process, so `jobs` runs go at once and the rows come out the same however
many there are.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, replace

from ..runner import Run
from ..world import World


@dataclass(frozen=True)
class Row:
    """How one seed ended.

    Attributes:
        seed: the seed of the run.
        done: True if every robot finished its tasks before the limit.
        time: simulated seconds when the run ended.
        collisions: contacts counted over the run.
    """

    seed: int
    done: bool
    time: float
    collisions: int


def _one(job: tuple[World, float]) -> Row:
    """Run a world to the end; a module function so a process can receive it."""
    world, limit = job
    run = Run(world)
    done = run.until_done(limit)
    return Row(world.seed, done, run.time, len(run.sim.collisions))


def run_seeds(world: World, count: int, limit: float, jobs: int = 1) -> list[Row]:
    """Run `count` seeds, starting at the world's, one row each in seed order.

    Args:
        world: the world to run.
        count: how many seeds.
        limit: simulated seconds after which a run that is not done stops.
        jobs: how many runs go at once; 1 runs them here, in order.
    """
    work = [(replace(world, seed=world.seed + i), limit) for i in range(count)]
    if jobs <= 1:
        return [_one(item) for item in work]
    with multiprocessing.get_context("spawn").Pool(min(jobs, count)) as pool:
        return pool.map(_one, work)


def table(rows: list[Row]) -> str:
    """The rows as aligned text, with a last line that counts the runs done."""
    lines = ["seed  done  time (s)  collisions"]
    for r in rows:
        done = "yes" if r.done else "no"
        lines.append(f"{r.seed:>4}  {done:>4}  {r.time:>8.2f}  {r.collisions:>10}")
    lines.append(f"{sum(r.done for r in rows)} of {len(rows)} runs done")
    return "\n".join(lines)
