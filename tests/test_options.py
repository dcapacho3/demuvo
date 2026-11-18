"""Tests for the lists of parts that a window or a script can offer."""

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from demuvo import options
from demuvo.options import Setting
from demuvo.world import WorldError, load_world


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
        *,
        gain: float = 1.0,
        steps: int = 3,
        flag=True,
        _hidden=1,
        label: str = "x",
        other=None,
    ):
        """Take one setting of each kind, and some that a form leaves out."""

    def run(self):
        """Do nothing; a class with no public method is not worth a test."""


@dataclass
class Mixed:
    """A dataclass with a setting, a required field and a field with a factory."""

    required: float
    gain: float = 2.0
    names: list = field(default_factory=list)
    _private: float = 0.0


def test_the_lists_hold_the_built_in_parts():
    """Each list starts with what demuvo ships."""
    assert options.models() == ("diff_drive", "ackermann")
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
    assert settings["max_v"] == Setting(
        "max_v", float, 1.0, "largest forward speed in m/s, in either direction."
    )


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


def test_the_estimators_and_sensors_are_listed_with_their_settings():
    """Names for a drop-down, and the number settings of each sensor kind."""
    assert options.estimators() == ("ground_truth", "dead_reckoning")
    assert options.sensor_kinds() == ("encoders", "lidar")
    encoders = {s.name: s.kind for s in options.sensor_settings("encoders")}
    assert encoders == {"noise": float, "ticks_per_rev": float}
    lidar = {s.name: s.kind for s in options.sensor_settings("lidar")}
    assert lidar["beams"] is int and lidar["max_range"] is float
    assert "mount" not in lidar


def test_the_motions_of_an_actor_are_listed():
    """The table a form shows; "stands still" means no motion at all."""
    table = options.motions()
    assert table["stands still"] is None
    assert table["Waypoints"] == "demuvo.actors:Waypoints"


def test_the_arguments_of_a_class_include_what_a_form_cannot_show():
    """A form asks for them to see what it must fill in itself, such as points."""
    assert "points" in options.arguments_of("demuvo.actors:Waypoints")
    assert options.arguments_of(None) == ()
    # a class that is not a dataclass is read from its __init__
    assert options.arguments_of("demuvo.sensing.lidar:Lidar") == (
        "model",
        "rng",
        "settings",
    )


def test_the_word_settings_of_a_class_are_offered_as_choices():
    """A motion says which of its settings are one of a few words."""
    assert options.choices_of("demuvo.actors:Waypoints")["mode"] == (
        "once",
        "loop",
        "bounce",
    )
    assert options.choices_of(None) == {}
    assert options.choices_of("demuvo.planners:AStar") == {}


def test_field_names_list_every_setting_of_a_motion_lists_included():
    """A form asks this to know whether a motion has waypoints to show."""
    assert options.field_names(None) == ()
    assert "points" in options.field_names("demuvo.actors:Waypoints")
    assert options.field_names("demuvo.gui.tests.test_actor_gui:Plain") == ()
    assert options.field_names("demuvo.controllers:PurePursuit") != ()


class Reaching:
    """A class that is not a dataclass, so its settings are its arguments."""

    def __init__(self, reach: float = 1.0, points=None) -> None:
        """Keep nothing; a test reads the argument names."""


def test_field_names_of_a_plain_class_are_its_arguments():
    """A motion that is not a dataclass is read from its constructor."""
    assert options.field_names("tests.test_options:Reaching") == ("reach", "points")


def test_each_setting_carries_its_first_sentence_from_the_docstring():
    """A form shows it after a pause; a continued line is joined to its setting."""
    notes = {s.name: s.help for s in options.settings_of("demuvo.controllers:DWB")}
    assert notes["max_vel_x"] == "largest forward speed in m/s."
    assert notes["allow_reversing"].startswith("lets it back up")
    assert all(notes.values())


def test_a_class_with_no_documented_settings_describes_none():
    """No Attributes or Args section gives an empty table."""

    class Plain:
        """Nothing documented."""

    assert options.described(Plain) == {}
    assert options.summary(None) == ""
    assert options.summary("demuvo.planners:AStar").startswith("The cheapest path")


def test_the_summary_of_a_choice_is_the_first_sentence_of_the_chosen_class():
    """A model, a planner left at its default, and a part left empty."""
    path = (
        Path(__file__).resolve().parents[1] / "examples" / "worlds" / "ackermann.toml"
    )
    spec = next(iter(load_world(path).robots))
    tables = {}
    assert options.summary_of_choice("model", spec, tables).startswith(
        "A car-like robot"
    )
    assert options.summary_of_choice("planner", spec, tables).startswith("The shortest")
    assert options.summary_of_choice("smoother", spec, tables) == ""
