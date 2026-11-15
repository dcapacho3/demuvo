"""The inspector's form for an actor - its pose, body, flags and motion.

An actor is a movable obstacle the global costmap is not built from, so a robot
meets one only through its lidar. The form shows where it starts, how big it is,
whether a robot is pushed out of it and whether it meets other actors, and which
motion carries it, with that motion's settings. Each field applies its edit
through the session and says why when it is refused.
"""

from __future__ import annotations

import math

from .. import options
from ..world.shapes import Circle
from . import look
from .formbase import FormBase
from .inputs import NumberSpec
from .panels import TitleRow
from .widgets import Checkbox, Label, Section

# The largest body an actor can be given here, in metres.
MAX_SIZE = 200.0
# Largest value a motion setting field accepts.
SETTING_MAX = 1000.0
WHAT = (
    "An actor is a moving obstacle. A planner never sees it: a robot finds it "
    "with its lidar."
)
NO_MOTION_SETTINGS = "This motion has nothing to set here."
# What each flag means, in the order they are shown.
FLAGS = (
    ("solid", "solid", "A robot is pushed out of it; off, the robot drives through."),
    ("collide", "collide", "It meets other actors that also have this on."),
)


class ActorForm(FormBase):
    """Builds the form of one actor into a column.

    Args:
        column: the column to fill.
        session: the session it edits.
        fields: the dict that collects the controls, by key.
        done: called with the result of an edit.
        settings: the dict that collects the motion setting controls, by name.
    """

    def __init__(self, column, session, fields, done, settings: dict) -> None:
        """Keep what the form needs, and where to put the motion settings."""
        super().__init__(column, session, fields, done)
        self.settings = settings

    @property
    def index(self) -> int:
        """The 0-based place of the actor being shown."""
        return self.session.scene.selected_actor

    def actor(self, index: int) -> None:
        """The whole form: title, pose, body, flags and motion."""
        spec = self.session.world.actors[index]
        title = TitleRow()
        title.name, title.color = spec.name, look.AMBER
        self.column.add(title)
        self.column.add(Label(WHAT, 12, look.MUTED, wrap=True))
        self.column.add(Section("Pose"))
        pose = tuple(float(v) for v in self.session.scene.actor_pose(index))
        self.pose_triple(
            ("ax", "ay", "ath"), ("x", "y", "heading"), pose, self._pose_edited
        )
        self.column.add(Section("Body"))
        self._size_rows(spec.shape)
        self.column.add(Section("Meets"))
        for key, label, hint in FLAGS:
            self.switch(
                f"actor_{key}",
                label,
                getattr(spec, key),
                lambda on, k=key: self._flag_edited(k, on),
            )
            self.column.add(Label(hint, 11, look.MUTED, wrap=True))
        self._motion(spec)

    def _size_rows(self, shape) -> None:
        """The radius of a disc, or the length and width of a box."""
        spec = NumberSpec(0.05, MAX_SIZE, 0.05, 2, " m")
        if isinstance(shape, Circle):
            self.number("asize0", "radius", shape.radius, spec, self._size_edited)
        else:
            self.number("asize0", "length", shape.size[0], spec, self._size_edited)
            self.number("asize1", "width", shape.size[1], spec, self._size_edited)

    def _motion(self, spec) -> None:
        """The motion drop-down, then a field for each of its settings."""
        self.column.add(Section("Motion"))
        table = options.motions()
        self.choice(
            "motion",
            "motion",
            list(table),
            options.label_of(table, spec.motion),
            lambda text: self._motion_chosen(table, text),
        )
        if spec.motion is None:
            return
        choices = options.choices_of(spec.motion)
        known = options.settings_of(spec.motion)
        if not known and not choices:
            self.column.add(Label(NO_MOTION_SETTINGS, 12, look.MUTED, wrap=True))
            return
        for name, words in choices.items():
            current = str(spec.motion_params.get(name, words[0]))
            self.choice(
                f"motion.{name}",
                name.replace("_", " "),
                list(words),
                current,
                lambda text, k=name: self._word_edited(k, text),
            )
        for setting in known:
            value = spec.motion_params.get(setting.name, setting.default)
            self._setting_field(setting, value)

    def _setting_field(self, setting, value) -> None:
        """A row for one motion setting: a switch, a whole number or a number."""
        key = f"motion.{setting.name}"
        label = setting.name.replace("_", " ")
        if setting.kind is bool:
            self.switch(key, label, bool(value), lambda _on: self._settings_edited())
        elif setting.kind is int:
            spec = NumberSpec(0, SETTING_MAX, 1, 0, "", True)
            self.number(key, label, int(value), spec, self._settings_edited)
        else:
            step = max(0.01, round(abs(float(setting.default)) / 10, 3))
            spec = NumberSpec(0.0, SETTING_MAX, step, 3)
            self.number(key, label, float(value), spec, self._settings_edited)
        self.settings[setting.name] = self.fields[key]

    # -- edits --
    def _pose_edited(self) -> None:
        """A pose field changed."""
        f = self.fields
        pose = (f["ax"].value, f["ay"].value, math.radians(f["ath"].value))
        self.done(self.session.edits.move_actor(self.index, pose))

    def _size_edited(self) -> None:
        """A body size field changed."""
        f = self.fields
        size = tuple(f[k].value for k in ("asize0", "asize1") if k in f)
        self.done(self.session.edits.resize_actor(self.index, size))

    def _flag_edited(self, key: str, on: bool) -> None:
        """The solid or collide switch changed."""
        self.done(self.session.edits.set_actor_flag(self.index, key, on))

    def _motion_chosen(self, table: dict, text: str) -> None:
        """Another motion was picked; its settings are shown afresh."""
        self.done(self.session.edits.set_actor_motion(self.index, table[text]))

    def _word_edited(self, key: str, text: str) -> None:
        """A motion setting that is one of a few words changed."""
        self.done(self.session.edits.set_actor_motion_params(self.index, {key: text}))

    def _settings_edited(self) -> None:
        """A number or switch of the motion changed."""
        values = {
            name: (w.checked if isinstance(w, Checkbox) else w.value)
            for name, w in self.settings.items()
        }
        self.done(self.session.edits.set_actor_motion_params(self.index, values))
