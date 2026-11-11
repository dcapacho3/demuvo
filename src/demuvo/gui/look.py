"""How the program looks - colours, a colour for each robot, costmap palettes, motion.

Nothing here draws. The values are the tokens of the design: the mint floor and
the navy ink of the logo, a hue for each robot, two palettes for the costmap and
the curve that eases a motion out.
"""

from __future__ import annotations

import math

import numpy as np

from ..costmap.grid import FREE, INSCRIBED, LETHAL, MAX_NON_OBSTACLE, NO_INFORMATION

# The chess-board floor, the ink, and the accents of the logo.
MINT, MINT2 = "#e6f6f4", "#a9ddd5"
NAVY, NAVY2, TEAL, AMBER, UNKNOWN = (
    "#0f2742",
    "#1b3a5e",
    "#1fb5a6",
    "#f5a623",
    "#aab4bd",
)
# Lines, muted text, the window and the panels.
LINE, MUTED, WINDOW, PANEL = "#d5dde8", "#5b6f86", "#f5f7f8", "#ffffff"
# The note that warns, in a panel.
WARN_BG, WARN_LINE, WARN_TEXT = "#fff4dc", "#f5d28a", "#7a4a00"
# How opaque the graded, inscribed and unknown cells of the costmap are.
GRADED_ALPHA, INSCRIBED_ALPHA, UNKNOWN_ALPHA = 0.50, 0.45, 0.60
PALETTES = ("navy and amber", "mint to teal")
# Milliseconds: a press or hover, a layer fading, a panel sliding.
MS = {"press": 100, "fade": 90, "panel": 200}
# Robot colours: the logo's hue, a step of the golden ratio, a chroma and lightness.
TEAL_HUE, GOLDEN, CHROMA = 184.0, 0.6180339887498949, 0.15
LIGHTNESS = (0.42, 0.52, 0.47)
# A step of lightness when a colour falls outside the screen's range.
CHROMA_STEP = 0.002


def _channels(hex_color: str) -> np.ndarray:
    """The red, green and blue of "#rrggbb" as 0 to 255."""
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _oklab_to_linear(lightness: float, chroma: float, hue: float) -> np.ndarray:
    """Linear RGB of an Oklch colour; outside 0 to 1 means outside sRGB."""
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    long = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    medium = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    short = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return np.array(
        [
            4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short,
            -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short,
            -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short,
        ]
    )


def robot_color(index: int) -> str:
    """The colour of robot number `index`, as "#rrggbb".

    Robot 0 is the teal of the logo. Each next one steps the hue by 0.618 of a
    turn, so any number of robots stay apart, and the lightness cycles so that two
    robots that look alike in hue differ in darkness. A hue the screen cannot show
    at this chroma gets less chroma, with the same hue and lightness.
    """
    lightness = LIGHTNESS[index % len(LIGHTNESS)]
    hue = (TEAL_HUE + 360 * GOLDEN * index) % 360
    chroma = CHROMA
    linear = _oklab_to_linear(lightness, chroma, hue)
    while linear.min() < 0 or linear.max() > 1:
        chroma -= CHROMA_STEP
        linear = _oklab_to_linear(lightness, chroma, hue)
    srgb = np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    red, green, blue = (round(255 * v) for v in srgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


def contrast(first: str, second: str) -> float:
    """The contrast ratio of two colours, 1 to 21; 3 is the least for a shape."""

    def luminance(hex_color: str) -> float:
        parts = _channels(hex_color) / 255
        linear = np.where(
            parts <= 0.04045, parts / 12.92, ((parts + 0.055) / 1.055) ** 2.4
        )
        return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def costmap_rgba(data: np.ndarray, palette: str = PALETTES[0]) -> np.ndarray:
    """A costmap as a uint8 RGBA picture.

    Row 0 of `data` is the lowest y, and the top of a picture is the highest, so
    the rows are flipped. Free cells are clear; graded cost gets more opaque as it
    rises; the inscribed, lethal and unknown cells have colours of their own.

    Args:
        data: uint8 cost cells.
        palette: one of `PALETTES`.
    """
    out = np.zeros(data.shape + (4,), dtype=np.uint8)
    graded = (data > FREE) & (data <= MAX_NON_OBSTACLE)
    cost = data[graded].astype(float) / MAX_NON_OBSTACLE
    if palette == "mint to teal":
        low, high = _channels(MINT2), _channels("#0b8f84")
        out[graded, :3] = (low + (high - low) * cost[:, None]).astype(np.uint8)
        out[graded, 3] = ((0.15 + 0.7 * cost) * 255).astype(np.uint8)
        inscribed, inscribed_alpha = "#0b8f84", 0.75
    else:
        out[graded, :3] = _channels(NAVY).astype(np.uint8)
        out[graded, 3] = (GRADED_ALPHA * cost * 255).astype(np.uint8)
        inscribed, inscribed_alpha = AMBER, INSCRIBED_ALPHA
    for cells, color, alpha in (
        (INSCRIBED, inscribed, inscribed_alpha),
        (LETHAL, NAVY2, 1.0),
        (NO_INFORMATION, UNKNOWN, UNKNOWN_ALPHA),
    ):
        out[data == cells, :3] = _channels(color).astype(np.uint8)
        out[data == cells, 3] = round(alpha * 255)
    return np.ascontiguousarray(np.flipud(out))


def _bezier(u: float, first: float, second: float) -> float:
    """A cubic Bezier from 0 to 1 with these two inner control values, at u."""
    return 3 * (1 - u) ** 2 * u * first + 3 * (1 - u) * u**2 * second + u**3


def ease_out(t: float) -> float:
    """The curve cubic-bezier(0.23, 1, 0.32, 1): fast at first, then settling.

    There is no ease-in. A motion stopped half way and started again begins from
    where it is, so nothing jumps.

    Args:
        t: how far through the time, 0 to 1; outside is held at the ends.
    """
    t = min(max(t, 0.0), 1.0)
    low, high = 0.0, 1.0
    for _ in range(24):  # find the parameter whose x on the curve is t
        middle = (low + high) / 2
        if _bezier(middle, 0.23, 0.32) < t:
            low = middle
        else:
            high = middle
    return _bezier((low + high) / 2, 1.0, 1.0)
