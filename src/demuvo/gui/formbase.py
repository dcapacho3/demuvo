"""Building blocks for the inspector's forms - number, choice, switch, button, text.

Each adds one labelled row to a column and keeps the control under a key, so the
inspector can lock it and a test can find it.
"""

from __future__ import annotations

import math

from .inputs import Combo, NumberSpec, Spin, TextField
from .panels import FormRow
from .widgets import Checkbox, TextButton, Widget


class FormBase:
    """A form being built into a column.

    Args:
        column: the column to fill.
        session: the session it edits.
        fields: the dict that collects the controls, by key.
        done: called with the result of an edit; it reports an error or refreshes.
    """

    def __init__(self, column, session, fields: dict[str, Widget], done) -> None:
        """Keep what the form needs."""
        self.column, self.session, self.fields, self.done = (
            column,
            session,
            fields,
            done,
        )

    def _only_when_editable(self, action):
        """The action, but a no-op while a run is in progress or the world is fixed."""

        def guarded(*args):
            if self.session.editable:
                action(*args)

        return guarded

    def number(self, key: str, label: str, value: float, spec: NumberSpec, change):
        """A labelled number field; `change` is called with no argument."""
        change = self._only_when_editable(change)
        field = Spin(value, spec, on_change=lambda _v: change())
        self.fields[key] = field
        self.column.add(FormRow(label, field))

    def pose_triple(self, keys, labels, pose, change) -> None:
        """Three number rows for a pose - x and y in metres, the angle in degrees."""
        x0, y0, x1, y1 = self.session.world.bounds
        rows = (
            (pose[0], NumberSpec(x0, x1, 0.05, 2, " m")),
            (pose[1], NumberSpec(y0, y1, 0.05, 2, " m")),
            (math.degrees(pose[2]), NumberSpec(-180, 180, 5, 2, " °")),
        )
        for key, label, (value, spec) in zip(keys, labels, rows):
            self.number(key, label, value, spec, change)

    def choice(self, key: str, label: str, items: list[str], current: str, change):
        """A labelled drop-down; `change` is called with the chosen text."""
        field = Combo(items, current, on_change=self._only_when_editable(change))
        self.fields[key] = field
        self.column.add(FormRow(label, field))

    def switch(self, key: str, label: str, on: bool, change):
        """A labelled check box; `change` is called with the new state."""
        field = Checkbox("on", on, on_change=self._only_when_editable(change))
        self.fields[key] = field
        self.column.add(FormRow(label, field))

    def text(self, key: str, label: str, value: str, submit):
        """A labelled text box; Enter calls `submit` with the text."""
        field = TextField(value, on_submit=self._only_when_editable(submit))
        self.fields[key] = field
        self.column.add(FormRow(label, field))

    def button(self, key: str, text: str, action) -> None:
        """A button on a row of its own."""
        button = TextButton(text, self._only_when_editable(action))
        self.fields[key] = button
        self.column.add(button)
