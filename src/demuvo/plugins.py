"""Building the classes a world file names as "module:Class"."""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import import_module

from .verify import WorldError


def load_class(path: str, params: Mapping, robot: str, kind: str):
    """Build the class `path` ("module:Class") with `params` as keyword arguments.

    Args:
        path: module and class, for example "demuvo.controllers:PurePursuit".
        params: keyword arguments for the class.
        robot: name of the robot that asked for it, for the error message.
        kind: "planner" or "controller", for the error message.

    Raises:
        WorldError: if the module, the class or its arguments do not work.
    """
    module, _, cls = path.partition(":")
    try:
        return getattr(import_module(module), cls)(**params)
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise WorldError(
            f"robot '{robot}': cannot load {kind} '{path}': {error}"
        ) from error


def check_positive(obj, *names: str) -> None:
    """Raise ValueError unless every named attribute of obj is a positive number."""
    for name in names:
        value = getattr(obj, name)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number, got {value!r}")
