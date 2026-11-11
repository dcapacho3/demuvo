"""The inspector - the form for the selected robot, in the right-hand column.

With an obstacle selected it shows the obstacle; with nothing, the world.

It shows the robot's pose and goal, its model, planner, smoother and controller, the
settings of each part and the control period; or a hint when nothing is selected.
Each field applies its edit through the session and says why when it is refused.
"""

from __future__ import annotations

import math

from .. import options
from . import look
from .inputs import Combo, NumberSpec, Spin
from .panels import FormRow, Note, TitleRow
from .robotform import RobotForm
from .session import Session
from .widgets import Checkbox, Column, Label, Section, Widget
from .worldform import WorldForm

HINT = "Select a robot or an obstacle to see and edit it."
# Largest value a setting field accepts.
SETTING_MAX = 1000.0


class Inspector:
    """The form for one robot.

    Args:
        column: the column it fills.
        session: the session it edits.
        report: called with a message, to show it to the user.
        refresh: called after every change that worked, so the window can update.
        footer: widgets added at the end, such as the layer switches.

    Attributes:
        note: the box that says why editing is locked.
        fields: the pose, goal, choice and period fields, by name.
        settings: the setting fields of each part, by part and setting name.
    """

    def __init__(
        self, column: Column, session: Session, report, refresh, footer=()
    ) -> None:
        """Make an empty inspector."""
        self.column, self.session = column, session
        self.report, self.refresh, self.footer = report, refresh, list(footer)
        self.note = Note()
        # reading the installed packages for plug-ins takes milliseconds, and the
        # list does not change while the window is open
        self.tables = {
            "planner": options.planners(),
            "smoother": options.smoothers(),
            "controller": options.controllers(),
        }
        self.fields: dict[str, Widget] = {}
        self.settings: dict[str, dict[str, Widget]] = {}

    @property
    def name(self) -> str | None:
        """The selected robot."""
        return self.session.scene.selected

    def show(self, name: str | None) -> None:
        """Rebuild the form for a robot, or for none."""
        col = self.column
        col.clear()
        col.add(Section("Inspector"), self.note)
        self.fields, self.settings = {}, {}
        forms = WorldForm(col, self.session, self.fields, self._done)
        obstacle = self.session.scene.selected_obstacle
        if name is not None:
            self._robot(name)
        elif obstacle is not None:
            forms.obstacle(obstacle)
        else:
            forms.world()
            col.add(Label(HINT, 13, look.MUTED, wrap=True))
        col.add(*self.footer)
        col.layout()
        self.lock()

    def _robot(self, name: str) -> None:
        """The sections for one robot."""
        spec = next(r for r in self.session.world.robots if r.name == name)
        col = self.column
        title = TitleRow()
        title.name, title.color = name, self.session.scene.colors.get(name, look.TEAL)
        col.add(title, Section("Pose"))
        pose = self.session.scene.pose(name)
        self._triple(("x", "y", "th"), pose[0], pose[1], pose[2], self._pose_edited)
        forms = RobotForm(col, self.session, self.fields, self._done)
        forms.goals(name, pose)
        forms.body(spec)
        col.add(Section("Robot"))
        for field, items, current in self._choices(spec):
            box = Combo(
                items, current, on_change=lambda text, f=field: self._chosen(f, text)
            )
            self.fields[field] = box
            col.add(FormRow(field, box))
        self._settings(spec)
        forms.sensing(spec)

    def _triple(self, keys, x, y, angle, callback) -> None:
        """Three fields - x, y and a heading in degrees - added as rows."""
        x0, y0, x1, y1 = self.session.world.bounds
        specs = (
            (x, NumberSpec(x0, x1, 0.05, 2, " m")),
            (y, NumberSpec(y0, y1, 0.05, 2, " m")),
            (math.degrees(angle), NumberSpec(-180, 180, 5, 2, " °")),
        )
        for key, label, (value, spec) in zip(keys, ("x", "y", "heading"), specs):
            field = Spin(value, spec, on_change=lambda _v, cb=callback: cb())
            self.fields[key] = field
            self.column.add(FormRow(label, field))

    def _choices(self, spec):
        """The four choices of a robot: field, items to pick from, the current one."""
        tables = self.tables
        return (
            ("model", list(options.models()), spec.model),
            (
                "planner",
                list(tables["planner"]),
                options.label_of(
                    tables["planner"], spec.planner or options.DEFAULT_PLANNER
                ),
            ),
            (
                "smoother",
                list(tables["smoother"]),
                options.label_of(tables["smoother"], spec.smoother),
            ),
            (
                "controller",
                list(tables["controller"]),
                options.label_of(tables["controller"], spec.controller),
            ),
        )

    def _settings(self, spec) -> None:
        """A section of fields for each part with settings, then the control period."""
        col = self.column
        parts = (
            ("model", options.model_settings(spec.model), spec.params),
            (
                "planner",
                options.settings_of(spec.planner or options.DEFAULT_PLANNER),
                spec.planner_params,
            ),
            ("smoother", options.settings_of(spec.smoother), spec.smoother_params),
            (
                "controller",
                options.settings_of(spec.controller),
                spec.controller_params,
            ),
        )
        for part, known, current in parts:
            if not known:
                continue
            col.add(Section(f"{part} settings"))
            fields = {}
            for setting in known:
                value = current.get(setting.name, setting.default)
                fields[setting.name] = self._setting_field(part, setting, value)
                col.add(FormRow(setting.name.replace("_", " "), fields[setting.name]))
            self.settings[part] = fields
        col.add(Section("Control timing"))
        period = Spin(
            spec.control_period,
            NumberSpec(0.001, 10.0, 0.01, 3, " s"),
            on_change=self._period_edited,
        )
        self.fields["period"] = period
        col.add(FormRow("control period", period))

    def _setting_field(self, part: str, setting, value) -> Widget:
        """A field for one setting: a switch, a whole number or a number."""

        def changed(_value) -> None:
            self._settings_edited(part)

        if setting.kind is bool:
            return Checkbox("on", bool(value), on_change=changed)
        if setting.kind is int:
            spec = NumberSpec(0, SETTING_MAX, 1, 0, "", True)
            return Spin(int(value), spec, on_change=changed)
        step = max(0.01, round(abs(float(setting.default)) / 10, 3))
        return Spin(
            float(value), NumberSpec(0.0, SETTING_MAX, step, 3), on_change=changed
        )

    def lock(self) -> None:
        """Enable the fields while editing is allowed, and say why when it is not."""
        session = self.session
        ok = session.editable
        notes = {
            "bench": (
                "This is a benchmark world. It is fixed, so every run is the same."
            ),
            "run": "A run is in progress. Press Reset (R) to edit again.",
            "edit": "",
        }
        self.note.set_text(notes[session.state])
        for widget in self.fields.values():
            widget.enabled = ok
        for group in self.settings.values():
            for widget in group.values():
                widget.enabled = ok
        self.column.layout()

    def sync_pose(self) -> None:
        """Show the dragged robot's or obstacle's pose in its three pose fields."""
        name = self.name
        place = self.session.scene.selected_obstacle
        if place is not None and "ox" in self.fields:
            x, y, angle = self.session.world.obstacles[place].pose.xyt
            values = (("ox", x), ("oy", y), ("oth", math.degrees(angle)))
        elif name and "x" in self.fields:
            pose = self.session.scene.pose(name)
            values = (("x", pose[0]), ("y", pose[1]), ("th", math.degrees(pose[2])))
        else:
            return
        for key, value in values:
            self.fields[key].value = float(value)
            self.fields[key].invalidate()

    # -- edits --
    def _done(self, error: str | None) -> bool:
        """Report an error, or refresh after a change that worked; True if it worked."""
        if error is not None:
            self.report(error)
            self.show(self.name)
            return False
        self.refresh()
        return True

    def _pose_edited(self) -> None:
        """A pose field changed."""
        f = self.fields
        pose = (f["x"].value, f["y"].value, math.radians(f["th"].value))
        if self.name and self.session.editable:
            self._done(self.session.move_robot(self.name, pose))

    def _chosen(self, field: str, text: str) -> None:
        """A choice changed; the part's settings are shown afresh."""
        if (
            self.name
            and self.session.editable
            and self._done(self.session.choose(self.name, field, text))
        ):
            self.show(self.name)

    def _settings_edited(self, part: str) -> None:
        """A setting of a part changed."""
        if not (self.name and self.session.editable):
            return
        values = {
            key: (w.checked if isinstance(w, Checkbox) else w.value)
            for key, w in self.settings[part].items()
        }
        self._done(self.session.set_settings(self.name, part, values))

    def _period_edited(self, seconds: float) -> None:
        """The control period changed."""
        if self.name and self.session.editable:
            self._done(self.session.set_period(self.name, seconds))
