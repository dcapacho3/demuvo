"""The choices a program can offer for the parts of a robot.

A drop-down in a window, or a check in a script, needs lists: which models,
planners, smoothers and controllers exist, and which settings each one takes.
This module builds them, so a window does not have to know any class by name.

A package of its own joins a list by naming its class in its entry points, with
the group `demuvo.planners`, `demuvo.smoothers` or `demuvo.controllers`:

    [project.entry-points."demuvo.planners"]
    greedy = "mybench.planners:Greedy"

Only the "module:Class" text is read from the entry point. The class is loaded
when a robot uses it.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points

from .models import MODELS
from .nav.navigation import DEFAULT_PLANNER
from .verify import WorldError

NO_SMOOTHER = "none"
NO_CONTROLLER = "none (driven from outside)"

_PLANNERS = {"Straight line": DEFAULT_PLANNER, "A*": "demuvo.planners:AStar"}
_SMOOTHERS = {
    NO_SMOOTHER: None,
    "Line of sight": "demuvo.smoothers.line_of_sight:LineOfSight",
}
_CONTROLLERS = {
    "Pure pursuit": "demuvo.controllers:PurePursuit",
    NO_CONTROLLER: None,
}

# A setting is a number or a switch. Annotations are text in a module that uses
# `from __future__ import annotations`, so the names are listed too.
_KINDS = {"float": float, "int": int, "bool": bool, float: float, int: int, bool: bool}


@dataclass(frozen=True)
class Setting:
    """One setting of a part, as a form needs it.

    Attributes:
        name: the key in the world file, such as "lookahead".
        kind: `float`, `int` or `bool`.
        default: the value a robot gets if the file does not set it.
    """

    name: str
    kind: type
    default: object


def _with_entry_points(builtin: dict, group: str) -> dict:
    """The built-in choices, then those that installed packages name."""
    choices = dict(builtin)
    for entry in entry_points(group=group):
        choices.setdefault(entry.name, entry.value)
    return choices


def models() -> tuple[str, ...]:
    """The names of the kinematic models."""
    return tuple(MODELS)


def planners() -> dict[str, str | None]:
    """The planners by friendly name, each a "module:Class" path."""
    return _with_entry_points(_PLANNERS, "demuvo.planners")


def smoothers() -> dict[str, str | None]:
    """The smoothers by friendly name; "none" maps to None."""
    return _with_entry_points(_SMOOTHERS, "demuvo.smoothers")


def controllers() -> dict[str, str | None]:
    """The controllers by friendly name; a robot driven from outside maps to None."""
    return _with_entry_points(_CONTROLLERS, "demuvo.controllers")


def label_of(choices: dict[str, str | None], path: str | None) -> str:
    """The friendly name of a path in a table above; the path itself if unknown."""
    for label, value in choices.items():
        if value == path:
            return label
    return path or "none"


def _fields(cls) -> list[Setting]:
    """The settings of a class: a dataclass's fields, else its `__init__` arguments.

    Only numbers and switches that have a default are listed. A setting that
    starts with an underscore, or that is something else (a costmap handed in
    later), is not for a form.
    """
    if dataclasses.is_dataclass(cls):
        raw = [
            (f.name, f.type, f.default)
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
        ]
    else:
        raw = [
            (
                name,
                type(p.default) if p.annotation is p.empty else p.annotation,
                p.default,
            )
            for name, p in inspect.signature(cls).parameters.items()
            if p.default is not p.empty
        ]
    return [
        Setting(name, _KINDS[kind], default)
        for name, kind, default in raw
        if not name.startswith("_") and kind in _KINDS
    ]


def model_settings(name: str) -> list[Setting]:
    """The settings a model takes in its `params` table, such as "max_v"."""
    return _fields(MODELS[name])


def settings_of(path: str | None) -> list[Setting]:
    """The settings of the class a world file names as "module:Class".

    Raises:
        WorldError: if the module or the class cannot be found.
    """
    if not path:
        return []
    module, _, name = path.partition(":")
    try:
        cls = getattr(import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise WorldError(f"cannot load '{path}': {error}") from error
    return _fields(cls)
