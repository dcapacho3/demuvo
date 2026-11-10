"""Reading a ROS map file - a `map.yaml` and the PGM image it names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..verify import WorldError


def _read_pgm(path: Path) -> np.ndarray:
    """The brightness of a PGM image (P2 or P5) as floats from 0 to 1, row 0 on top."""
    data = path.read_bytes()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            pos = data.index(b"\n", pos)
            continue
        end = pos
        while end < len(data) and not data[end : end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    magic, width, height, maxval = tokens[0], *(int(t) for t in tokens[1:])
    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else ">u2"
        pixels = np.frombuffer(data, dtype=dtype, count=width * height, offset=pos + 1)
    elif magic == b"P2":
        pixels = np.array(data[pos:].split(), dtype=float)[: width * height]
    else:
        raise WorldError(f"{path}: only PGM images (P2 or P5) can be read")
    return pixels.reshape(height, width) / maxval


@dataclass(frozen=True)
class MapFile:
    """A map read from a file.

    Attributes:
        occupancy: how occupied each cell is, 0 to 1, row 0 at the lowest y.
        resolution: side of a cell in metres.
        origin: world position (x, y) of the lower-left corner of cell (0, 0).
        occupied_thresh: occupancy from which a cell is an obstacle.
        free_thresh: occupancy below which a cell is free; between the two
            thresholds a cell is unknown.
    """

    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]
    occupied_thresh: float
    free_thresh: float

    @property
    def size(self) -> tuple[float, float]:
        """Width and height in metres."""
        rows, cols = self.occupancy.shape
        return cols * self.resolution, rows * self.resolution


def read_map(path: Path) -> MapFile:
    """Read a `map.yaml` and its image.

    Occupancy is `1 - brightness`, or `brightness` if `negate` is set. The `mode`
    key is ignored and the origin's angle must be 0.

    Raises:
        WorldError: if the file or its image cannot be read.
    """
    try:
        keys = {}
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                key, _, value = line.partition(":")
                keys[key.strip()] = value.strip().strip("'\"")
        resolution = float(keys["resolution"])
        origin = [float(v) for v in keys["origin"].strip("[]").split(",")]
        image = _read_pgm(path.parent / keys["image"])
        occupied = float(keys.get("occupied_thresh", 0.65))
        free = float(keys.get("free_thresh", 0.25))
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise WorldError(f"costmap: cannot read map '{path}': {error!r}") from error
    if resolution <= 0 or len(origin) != 3 or origin[2] != 0:
        raise WorldError(
            f"costmap: map '{path}' needs resolution > 0 and an origin with angle 0"
        )
    occupancy = image if keys.get("negate", "0") in ("1", "true") else 1.0 - image
    # image row 0 is the top; cell row 0 is the lowest y
    return MapFile(
        occupancy[::-1].copy(), resolution, (origin[0], origin[1]), occupied, free
    )
