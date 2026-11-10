"""Text - Roboto outlines drawn into small coverage bitmaps, one per glyph and size.

The outlines are read with fontTools. Each glyph is filled by sampling it on a
grid four times finer than the pixels, and the bitmap is kept for the next time.
A canvas blends the bitmap in the colour it wants.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont

# Where Debian and Ubuntu put Roboto, then the copy that ships with demuvo.
SYSTEM_FONTS = Path("/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF")
BUNDLED_FONTS = Path(__file__).resolve().parent / "assets"
# Samples per pixel along each axis when a glyph is filled.
SAMPLES = 4
# Straight pieces per curve segment.
QUADRATIC_PIECES = 6
CUBIC_PIECES = 8


class Glyph(NamedTuple):
    """One character at one size, as a bitmap.

    Attributes:
        coverage: how much of each pixel the character covers, 0 to 1, shape
            (rows, columns); an empty array for a space.
        left: pixels from the pen to the bitmap's left edge.
        top: pixels from the baseline up to the bitmap's top row.
        advance: pixels the pen moves on after this character.
    """

    coverage: np.ndarray
    left: int
    top: int
    advance: float


def _quadratic(p0, control, p1, t: np.ndarray) -> list[tuple[float, float]]:
    """Points of a quadratic curve at the fractions t."""
    u = 1 - t
    x = u * u * p0[0] + 2 * u * t * control[0] + t * t * p1[0]
    y = u * u * p0[1] + 2 * u * t * control[1] + t * t * p1[1]
    return list(zip(x.tolist(), y.tolist()))


def _cubic(p0, c1, c2, p3, t: np.ndarray) -> list[tuple[float, float]]:
    """Points of a cubic curve at the fractions t."""
    u = 1 - t
    x = u**3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t**3 * p3[1]
    return list(zip(x.tolist(), y.tolist()))


def flatten(recording, scale: float) -> list[np.ndarray]:
    """Outline commands as closed polylines in pixels, y down.

    Args:
        recording: the `value` of a fontTools recording pen.
        scale: pixels per font unit.

    Returns:
        One array of (x, y) points for each contour. A TrueType curve is a run of
        off-curve points ending on a point of the curve; between two off-curve
        points the curve passes through their middle. A contour made only of
        off-curve points is a closed loop that starts at the middle of its last
        and first points. A cubic curve is read as well, in case a font has them.
    """
    contours, current, pos = [], [], (0.0, 0.0)
    for op, args in recording:
        if op == "moveTo":
            if current:
                contours.append(np.array(current))
            pos = args[0]
            current = [pos]
        elif op == "lineTo":
            pos = args[0]
            current.append(pos)
        elif op == "qCurveTo":
            *offs, end = args
            if end is None:  # a closed loop of off-curve points, as the dot of a ?
                end = ((offs[-1][0] + offs[0][0]) / 2, (offs[-1][1] + offs[0][1]) / 2)
                pos = end
                current = [end]
            for i, off in enumerate(offs):
                last = i == len(offs) - 1
                middle = (
                    end
                    if last
                    else ((off[0] + offs[i + 1][0]) / 2, (off[1] + offs[i + 1][1]) / 2)
                )
                t = np.linspace(0, 1, QUADRATIC_PIECES + 1)[1:]
                current += _quadratic(pos, off, middle, t)
                pos = middle
        elif op == "curveTo":
            t = np.linspace(0, 1, CUBIC_PIECES + 1)[1:]
            current += _cubic(pos, *args, t)
            pos = args[-1]
        elif op in ("closePath", "endPath") and current:
            contours.append(np.array(current))
            current = []
    if current:
        contours.append(np.array(current))
    return [np.column_stack([c[:, 0] * scale, -c[:, 1] * scale]) for c in contours]


def _add_winding(
    contour: np.ndarray, xs: np.ndarray, ys: np.ndarray, winding: np.ndarray
) -> None:
    """Add to the winding count of every sample a closed polyline winds round."""
    for (ax, ay), (bx, by) in zip(contour, np.roll(contour, -1, axis=0)):
        if ay == by:
            continue
        rising = ay > by
        low, high = (by, ay) if rising else (ay, by)
        rows = np.nonzero((ys >= low) & (ys < high))[0]
        if rows.size == 0:
            continue
        crossing = ax + (bx - ax) * (ys[rows] - ay) / (by - ay)
        right_of_edge = xs[None, :] >= crossing[:, None]
        winding[rows] += right_of_edge.astype(np.int16) * (1 if rising else -1)


def _coverage(contours: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """The coverage bitmap of closed polylines, and its left and top edge in pixels.

    A sample counts as inside where the outline winds round it a non-zero number
    of times, so overlapping parts of a glyph stay filled.
    """
    points = np.vstack(contours)
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 1
    width, height = x1 - x0, y1 - y0
    xs = x0 + (np.arange(width * SAMPLES) + 0.5) / SAMPLES
    ys = y0 + (np.arange(height * SAMPLES) + 0.5) / SAMPLES
    winding = np.zeros((height * SAMPLES, width * SAMPLES), dtype=np.int16)
    for contour in contours:
        _add_winding(contour, xs, ys, winding)
    inside = (winding != 0).reshape(height, SAMPLES, width, SAMPLES)
    return inside.mean(axis=(1, 3)).astype(np.float32), int(x0), int(y0)


class Font:
    """One font file.

    Args:
        path: a TrueType file.

    Attributes:
        glyphs_made: how many glyph bitmaps are in the cache.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the file and read the tables a glyph needs."""
        self._file = TTFont(str(path))
        self._outlines = self._file.getGlyphSet()
        self._names = self._file.getBestCmap()
        self._units = self._file["head"].unitsPerEm
        self._advances = self._file["hmtx"]
        hhea = self._file["hhea"]
        self._height = hhea.ascent - hhea.descent
        self._cache: dict[tuple[str, int], Glyph] = {}

    @property
    def glyphs_made(self) -> int:
        """The number of glyph bitmaps kept."""
        return len(self._cache)

    def glyph(self, char: str, size: int) -> Glyph:
        """The bitmap of a character at a size in pixels.

        A character the font does not have is drawn as a question mark.
        """
        key = (char, size)
        if key not in self._cache:
            self._cache[key] = self._make(char, size)
        return self._cache[key]

    def _make(self, char: str, size: int) -> Glyph:
        """Fill a glyph; see `glyph`."""
        name = self._names.get(ord(char)) or self._names[ord("?")]
        scale = size / self._units
        advance = self._advances[name][0] * scale
        pen = DecomposingRecordingPen(self._outlines)
        self._outlines[name].draw(pen)
        contours = flatten(pen.value, scale)
        if not contours:
            return Glyph(np.zeros((0, 0), np.float32), 0, 0, advance)
        coverage, left, top = _coverage(contours)
        return Glyph(coverage, left, -top, advance)

    def width(self, text: str, size: int) -> float:
        """How many pixels wide a string is at a size."""
        return sum(self.glyph(char, size).advance for char in text)

    def line_height(self, size: int) -> float:
        """Pixels from one baseline to the next, at a size."""
        return self._height * size / self._units


def font_file(name: str) -> Path:
    """The file of a font: the system's if it is there, else the one that ships.

    Args:
        name: the file name, such as "Roboto-Regular.ttf".
    """
    for folder in (SYSTEM_FONTS, BUNDLED_FONTS):
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(
        f"font {name} is neither in {SYSTEM_FONTS} nor in the package"
    )


@cache
def regular() -> Font:
    """Roboto Regular, opened on the first call."""
    return Font(font_file("Roboto-Regular.ttf"))


@cache
def bold() -> Font:
    """Roboto Bold, opened on the first call."""
    return Font(font_file("Roboto-Bold.ttf"))
