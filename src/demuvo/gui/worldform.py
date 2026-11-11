"""The inspector's forms for the world and for an obstacle.

With nothing selected the inspector shows the world - its size, seed and step.
With an obstacle selected it shows that obstacle's pose and size. Each field
applies its edit through the session and says why when it is refused.
"""

from __future__ import annotations

import math

from .. import edit
from ..world.shapes import Box, Circle
from . import look
from .costmapform import CostmapForm
from .formbase import FormBase
from .inputs import NumberSpec
from .panels import TitleRow
from .robotform import RobotForm
from .widgets import Label, Section

# The largest world, in metres, and the shortest and longest step, in seconds.
MAX_SIZE = 200.0
STEP_RANGE = (0.0005, 1.0)
NO_SIZE = "This shape is drawn in the world file; here it can be moved or deleted."


class WorldForm(FormBase):
    """Builds the world form and the obstacle form into a column."""

    def _row(self, key: str, label: str, value: float, spec: NumberSpec, change):
        """One labelled number field, kept under a key."""
        self.number(key, label, value, spec, change)

    def world(self) -> None:
        """The size, seed and step of the world."""
        world = self.session.world
        self.column.add(Section("World"))
        for key, label, value in (
            ("width", "width", world.size[0]),
            ("height", "height", world.size[1]),
        ):
            spec = NumberSpec(1.0, MAX_SIZE, 0.5, 2, " m")
            self._row(key, label, value, spec, self._size_edited)
        spec = NumberSpec(0, 2**31 - 1, 1, 0, "", True)
        self._row("seed", "seed", world.seed, spec, self._seed_edited)
        spec = NumberSpec(*STEP_RANGE, 0.001, 4, " s")
        self._row("dt", "step", world.dt, spec, self._step_edited)
        corner = NumberSpec(-1000.0, 1000.0, 0.5, 2, " m")
        self._row("ox0", "corner x", world.origin[0], corner, self._origin_edited)
        self._row("oy0", "corner y", world.origin[1], corner, self._origin_edited)
        self._sensors()
        costmaps = CostmapForm(self.column, self.session, self.fields, self.done)
        costmaps.costmap("global")
        costmaps.costmap("local")

    def _sensors(self) -> None:
        """The sensors every robot starts with, and buttons to add one."""
        robots = RobotForm(self.column, self.session, self.fields, self.done)
        self.column.add(Section("Default sensors"))
        for sensor, settings in self.session.world.sensors.items():
            robots.sensor(None, sensor, settings)
        if "encoders" not in self.session.world.sensors:
            robots.button("add_default_encoders", "Add encoders", self._add_encoders)

    def _add_encoders(self) -> None:
        """Give every robot encoders to start from."""
        self.done(
            self.session.edits.change_world(
                lambda w: edit.set_sensor(w, None, "encoders", {})
            )
        )

    def _origin_edited(self) -> None:
        """A corner field changed."""
        f = self.fields
        corner = (f["ox0"].value, f["oy0"].value)
        self.done(self.session.edits.change_world(lambda w: edit.set_origin(w, corner)))

    def obstacle(self, index: int) -> None:
        """The pose and size of an obstacle."""
        session = self.session
        item = session.world.obstacles[index]
        title = TitleRow()
        title.name, title.color = f"{item.shape.name} {index + 1}", look.NAVY2
        self.column.add(title, Section("Pose"))
        x0, y0, x1, y1 = session.world.bounds
        x, y, angle = (float(v) for v in item.pose.xyt)
        specs = (
            ("ox", "x", x, NumberSpec(x0, x1, 0.05, 2, " m")),
            ("oy", "y", y, NumberSpec(y0, y1, 0.05, 2, " m")),
            ("oth", "heading", math.degrees(angle), NumberSpec(-180, 180, 5, 2, " °")),
        )
        for key, label, value, spec in specs:
            self._row(key, label, value, spec, self._pose_edited)
        self.column.add(Section("Size"))
        shape = item.shape
        if isinstance(shape, Circle):
            spec = NumberSpec(0.05, MAX_SIZE, 0.05, 2, " m")
            self._row("size0", "radius", shape.radius, spec, self._size_of_obstacle)
        elif isinstance(shape, Box):
            spec = NumberSpec(0.05, MAX_SIZE, 0.05, 2, " m")
            self._row("size0", "length", shape.size[0], spec, self._size_of_obstacle)
            self._row("size1", "width", shape.size[1], spec, self._size_of_obstacle)
        else:
            self.column.add(Label(NO_SIZE, 12, look.MUTED, wrap=True))

    # -- edits --
    def _size_edited(self) -> None:
        """A world size field changed."""
        f = self.fields
        self.done(self.session.edits.set_size((f["width"].value, f["height"].value)))

    def _seed_edited(self) -> None:
        """The seed changed."""
        self.done(self.session.edits.set_seed(int(self.fields["seed"].value)))

    def _step_edited(self) -> None:
        """The step length changed."""
        self.done(self.session.edits.set_step(self.fields["dt"].value))

    def _pose_edited(self) -> None:
        """An obstacle pose field changed."""
        f = self.fields
        pose = (f["ox"].value, f["oy"].value, math.radians(f["oth"].value))
        index = self.session.scene.selected_obstacle
        self.done(self.session.edits.move_obstacle(index, pose))

    def _size_of_obstacle(self) -> None:
        """An obstacle size field changed."""
        f = self.fields
        size = tuple(f[k].value for k in ("size0", "size1") if k in f)
        index = self.session.scene.selected_obstacle
        self.done(self.session.edits.resize_obstacle(index, size))
