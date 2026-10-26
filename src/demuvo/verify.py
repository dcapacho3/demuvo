"""Small verifications that are shared by the world file parsers.

Each verification returns the value it read, or raises a WorldError with a message
that names the object and the key thats wrong.
"""

from __future__ import annotations

import math


class WorldError(ValueError):
    """The world file is wrong. The message names the object and the key."""


def require_key(data: dict, key: str, label: str):
    """Return data[key], or raise if the key is missing."""
    if key not in data:
        raise WorldError(f"{label}: missing '{key}'")
    return data[key]


def table(value, label: str) -> dict:
    """Check that value is a TOML table."""
    if not isinstance(value, dict):
        raise WorldError(f"{label}: must be a table")
    return value


def check_keys(data: dict, allowed: set[str], label: str) -> None:
    """Refuse a key the format does not define."""
    for key in data:
        if key not in allowed:
            raise WorldError(f"{label}: unknown key '{key}'")


def items(data: dict, key: str) -> list:
    """The entries of a key array, or an empty if the key is absent."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise WorldError(f"file: '{key}' must be written as [[{key}]]")
    return value


def text(data: dict, key: str, label: str) -> str:
    """Return a required non empty string."""
    value = require_key(data, key, label)
    if not isinstance(value, str) or not value:
        raise WorldError(f"{label}: '{key}' must be a non-empty string")
    return value


def numbers(value, count: int, label: str, key: str, positive: bool = False):
    """Check that value is a list of finite numbers"""
    if (
        not isinstance(value, list)
        or len(value) != count
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        raise WorldError(f"{label}: '{key}' needs {count} finite number(s)")
    if positive and any(v <= 0 for v in value):
        raise WorldError(f"{label}: '{key}' must be positive")
    return tuple(float(v) for v in value)


def number(data: dict, key: str, label: str) -> float:
    """Return a required positive number."""
    return numbers([require_key(data, key, label)], 1, label, key, True)[0]


def inside(points, size) -> bool:
    """True if every point lies within the world, which spans (0, 0) to size."""
    width, height = size
    return all(0.0 <= x <= width and 0.0 <= y <= height for x, y in points)
