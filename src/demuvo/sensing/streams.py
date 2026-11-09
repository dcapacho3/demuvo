"""One random stream for each named thing, so adding a robot moves no other."""

from __future__ import annotations

import zlib

import numpy as np


def stream(seed: int, *names: str) -> np.random.Generator:
    """A random generator.

    The stream depends only on the seed and the names, never on how many other
    robots or sensors exist, or in which order they were built. `zlib.crc32` is
    used because Python's `hash` of a string changes from run to run.

    Args:
        seed: the seed of the world.
        names: for example a robot name and a sensor name.
    """
    key = [seed, *(zlib.crc32(name.encode()) for name in names)]
    return np.random.default_rng(np.random.SeedSequence(key))
