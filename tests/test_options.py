"""Tests for the lists of parts that a window or a script can offer."""

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint

import pytest

from demuvo import options
from demuvo.options import Setting
from demuvo.world import WorldError


def join(monkeypatch, group, name, value):
    """Make an installed package name a part in an entry point group."""
    entry = EntryPoint(name=name, value=value, group=group)
    monkeypatch.setattr(
        options,
        "entry_points",
        lambda group: [entry] if group == entry.group else [],
    )


class Plain:
    """A class that is not a dataclass, as another package may write it."""

    def __init__(
        self,
        required,
        gain: float = 1.0,
        steps: int = 3,
        flag=True,
        _hidden=1,
        label: str = "x",
        other=None,
    ):
        """Take one setting of each kind, and some that a form leaves out."""


@dataclass
class Mixed:
    """A dataclass with a setting, a required field and a field with a factory."""

    required: float
    gain: float = 2.0
    names: list = field(default_factory=list)
    _private: float = 0.0


def test_the_lists_hold_the_built_in_parts():
    """Each list starts with what demuvo ships."""
    assert options.models() == ("diff_drive",)
    assert options.planners()["A*"] == "demuvo.planners:AStar"
    assert options.planners()["Straight line"] == options.DEFAULT_PLANNER
    assert options.smoothers()[options.NO_SMOOTHER] is None
    assert options.controllers()["Pure pursuit"] == "demuvo.controllers:PurePursuit"
    assert options.controllers()[options.NO_CONTROLLER] is None


def test_an_installed_package_joins_the_list_of_its_group(monkeypatch):
    """A planner named in the entry points appears, and only in the planners."""
    join(monkeypatch, "demuvo.planners", "greedy", "mybench.planners:Greedy")
    assert options.planners()["greedy"] == "mybench.planners:Greedy"
    assert "greedy" not in options.controllers()
    assert "greedy" not in options.smoothers()


def test_a_built_in_name_is_not_replaced_by_a_package(monkeypatch):
    """A package that reuses a built-in name does not change what it means."""
    join(monkeypatch, "demuvo.planners", "A*", "mybench.planners:Other")
    assert options.planners()["A*"] == "demuvo.planners:AStar"


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("demuvo.planners:AStar", "A*"),
        ("mybench.planners:Unknown", "mybench.planners:Unknown"),
        (None, "none"),
    ],
)
def test_label_of_gives_the_friendly_name_or_the_path(path, label):
    """A known path gives its name, an unknown one gives itself, none gives none."""
    assert options.label_of(options.planners(), path) == label


def test_a_controller_driven_from_outside_has_its_own_label():
    """A robot with no controller shows the name that says what that means."""
    assert options.label_of(options.controllers(), None) == options.NO_CONTROLLER


def test_the_settings_of_a_model_come_with_their_defaults():
    """The diff drive takes four numbers."""
    settings = {s.name: s for s in options.model_settings("diff_drive")}
    assert set(settings) == {"max_v", "max_w", "wheel_radius", "track"}
    assert settings["max_v"] == Setting("max_v", float, 1.0)


def test_the_settings_of_a_controller_have_a_kind_each():
    """Numbers and switches are listed; the kind is the Python type."""
    settings = {
        s.name: s for s in options.settings_of("demuvo.controllers:PurePursuit")
    }
    assert settings["lookahead"].kind is float
    assert settings["rotate_to_goal"].kind is bool


def test_a_part_with_no_class_has_no_settings():
    """A robot with no smoother has nothing to set."""
    assert options.settings_of(None) == []


def test_settings_of_a_plain_class_come_from_its_init():
    """Annotated, unannotated and hidden arguments, and one with no default."""
    assert options.settings_of(f"{__name__}:Plain") == [
        Setting("gain", float, 1.0),
        Setting("steps", int, 3),
        Setting("flag", bool, True),
    ]


def test_settings_of_a_dataclass_leave_out_what_a_form_cannot_set():
    """A required field, a field with a factory and a private one are left out."""
    assert options.settings_of(f"{__name__}:Mixed") == [Setting("gain", float, 2.0)]


@pytest.mark.parametrize("path", ["no_such_module:Thing", "demuvo.planners:Nothing"])
def test_a_path_that_cannot_be_loaded_is_refused_in_plain_words(path):
    """The message names the path."""
    with pytest.raises(WorldError, match=f"cannot load '{path}'"):
        options.settings_of(path)
