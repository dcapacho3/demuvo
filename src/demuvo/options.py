"""The choices a program can offer for the parts of a robot.

A drop-down in a window, or a check in a script, needs lists: which models,
planners, smoothers and controllers exist, and which settings each one takes.
This module builds them, so a window does not have to know any class by name.

A package of its own joins a list by naming its class in its entry points, with
the group `demuvo.planners`, `demuvo.smoothers`, `demuvo.controllers` or
`demuvo.actors`:

    [project.entry-points."demuvo.planners"]
    greedy = "mybench.planners:Greedy"

Only the "module:Class" text is read from the entry point. The class is loaded
when a robot uses it.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points

from .models import MODELS
from .nav.navigation import DEFAULT_PLANNER
from .sensing import ESTIMATORS, Encoders
from .sensing.lidar import LidarSettings
from .verify import WorldError

NO_SMOOTHER = "none"
# The motion of an actor that stands where it was put.
NO_MOTION = "stands still"
NO_CONTROLLER = "none (driven from outside)"

_PLANNERS = {
    "Straight line": DEFAULT_PLANNER,
    "A*": "demuvo.planners:AStar",
    "Dubins": "demuvo.planners:Dubins",
    "Reeds-Shepp": "demuvo.planners:ReedsShepp",
    "Hybrid A*": "demuvo.planners:HybridAStar",
}
_SMOOTHERS = {
    NO_SMOOTHER: None,
    "Line of sight": "demuvo.smoothers.line_of_sight:LineOfSight",
}
_MOTIONS = {
    NO_MOTION: None,
    "Waypoints": "demuvo.actors:Waypoints",
}
_CONTROLLERS = {
    "Pure pursuit": "demuvo.controllers:PurePursuit",
    "DWB": "demuvo.controllers:DWB",
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
        help: what it does, in a sentence, from the class's docstring.
    """

    name: str
    kind: type
    default: object
    help: str = ""


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


def motions() -> dict[str, str | None]:
    """The motions of an actor by friendly name; "stands still" maps to None."""
    return _with_entry_points(_MOTIONS, "demuvo.actors")


def estimators() -> tuple[str, ...]:
    """The names of the estimators, which give a controller its pose."""
    return tuple(ESTIMATORS)


# The sensors a form can add, and the class that holds each one's settings.
_SENSORS = {"encoders": Encoders, "lidar": LidarSettings}


def label_of(choices: dict[str, str | None], path: str | None) -> str:
    """The friendly name of a path in a table above; the path itself if unknown."""
    for label, value in choices.items():
        if value == path:
            return label
    return path or "none"


def _first_sentence(text: str) -> str:
    """The text up to its first full stop, on one line."""
    text = " ".join(text.split())
    stop = text.find(". ")
    return text[: stop + 1] if stop >= 0 else text


def described(cls) -> dict[str, str]:
    """What each setting does, read from the `Attributes:` or `Args:` of a docstring.

    A setting that the docstring does not describe is not in the result.
    """
    lines = (inspect.getdoc(cls) or "").splitlines()
    notes: dict[str, str] = {}
    name = None
    inside = False
    for line in lines:
        if line.rstrip() in ("Attributes:", "Args:"):
            inside = True
        elif inside and line and not line.startswith(" "):
            inside = False
        elif inside and (match := re.match(r"^ {4}(\w+): (.*)", line)):
            name = match.group(1)
            notes[name] = match.group(2)
        elif inside and name and line.strip():
            notes[name] += " " + line.strip()
    return {key: _first_sentence(text) for key, text in notes.items()}


def summary(path: str | None) -> str:
    """The first sentence of the docstring of the class a path names, or ""."""
    if not path:
        return ""
    doc = inspect.getdoc(class_of(path)) or ""
    return _first_sentence(doc.split("\n\n", maxsplit=1)[0])


def summary_of_choice(field: str, spec) -> str:
    """The first sentence of what the class chosen for a part of a robot does."""
    if field == "model":
        return _first_sentence(inspect.getdoc(MODELS[spec.model]) or "")
    path = getattr(spec, field)
    if field == "planner":
        path = path or DEFAULT_PLANNER
    return summary(path)


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
    notes = described(cls)
    return [
        Setting(name, _KINDS[kind], default, notes.get(name, ""))
        for name, kind, default in raw
        if not name.startswith("_") and kind in _KINDS
    ]


def class_of(path: str):
    """The class a "module:Class" path names.

    Raises:
        WorldError: if the module or the class cannot be found.
    """
    module, _, name = path.partition(":")
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise WorldError(f"cannot load '{path}': {error}") from error


def arguments_of(path: str | None) -> tuple[str, ...]:
    """Every keyword a class takes, settings a form cannot show among them.

    A form uses it to see whether a class wants something it must fill in itself,
    such as the points of a path.
    """
    if not path:
        return ()
    cls = class_of(path)
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls))
    return tuple(inspect.signature(cls).parameters)


def choices_of(path: str | None) -> dict[str, tuple[str, ...]]:
    """The settings of a class that are one of a few words, and the words.

    A class offers them as a `CHOICES` class attribute, `{"mode": ("once", ...)}`,
    so a form can show a drop-down for a setting that is not a number.
    """
    if not path:
        return {}
    return {k: tuple(v) for k, v in getattr(class_of(path), "CHOICES", {}).items()}


def sensor_settings(kind: str) -> list[Setting]:
    """The number settings of a sensor kind; a lidar's mount is set apart."""
    return _fields(_SENSORS[kind])


def hidden_settings(spec, part: str) -> frozenset[str]:
    """The settings of one part of a robot that do nothing for it.

    A class lists them in `car_only` (a robot that turns in place has no use for
    them), `spot_only` (a car has none) and `encoder_only` (no use without
    encoders). A setting it lists nowhere always shows.

    Args:
        spec: the `RobotSpec`.
        part: "model", "planner", "smoother" or "controller".
    """
    path = spec.model if part == "model" else getattr(spec, part)
    if part == "planner":
        path = path or DEFAULT_PLANNER
    if not path:
        return frozenset()
    cls = MODELS[path] if part == "model" else class_of(path)
    car = not MODELS[spec.model]().limits().turns_in_place
    hidden = set(getattr(cls, "spot_only" if car else "car_only", ()))
    if "encoders" not in spec.sensors and "encoders" not in (
        ESTIMATORS[spec.estimator].needs
    ):
        hidden |= set(getattr(cls, "encoder_only", ()))
    return frozenset(hidden)


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
    return _fields(class_of(path))
