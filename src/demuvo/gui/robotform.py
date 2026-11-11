"""The inspector's forms for one robot's body, pose source, sensors and goals.

Each field applies its edit through the session and says why when it is refused.
"""

from __future__ import annotations

import functools
import math

from .. import edit, options
from . import look
from .formbase import FormBase
from .inputs import NumberSpec
from .widgets import Label, Section

NO_GOAL = "No goal yet. Pick one with the Goal tool (G), or type it here."
OTHER_BODY = "This body is a {}. Shape it in the world file."
# What the "stop" choice says, in the order of the drop-down.
STOP_CHOICES = {"default": None, "yes": True, "no": False}
# How far, in metres, a new waypoint is put from the last goal.
WAYPOINT_STEP = 0.5
LENGTH = NumberSpec(0.05, 100.0, 0.05, 2, " m")


class RobotForm(FormBase):
    """Builds the body, pose source, sensor and goal sections of one robot."""

    def _robot_edit(self, name: str, make) -> None:
        """Apply an edit that changes a robot."""
        self.done(self.session.edits.change_robot(name, make))

    # -- body --
    def body(self, spec) -> None:
        """The kind and size of the robot's body."""
        self.column.add(Section("Body"))
        shape = spec.shape
        if shape.name not in ("circle", "box"):
            self.column.add(
                Label(OTHER_BODY.format(shape.name), 12, look.MUTED, wrap=True)
            )
            return
        name = spec.name
        self.choice(
            "body",
            "shape",
            ["circle", "box"],
            shape.name,
            lambda kind: self._body_kind(name, kind),
        )
        if shape.name == "circle":
            self.number(
                "radius", "radius", shape.radius, LENGTH, lambda: self._body_size(name)
            )
        else:
            self.number(
                "length", "length", shape.size[0], LENGTH, lambda: self._body_size(name)
            )
            self.number(
                "width", "width", shape.size[1], LENGTH, lambda: self._body_size(name)
            )

    def _body_kind(self, name: str, kind: str) -> None:
        """Change a disc to a box of the same reach, or back."""
        shape = next(r for r in self.session.world.robots if r.name == name).shape
        if kind == "box":
            size = (2 * shape.radius, 2 * shape.radius)
        else:
            size = (max(shape.size) / 2,)
        self._robot_edit(name, lambda w: edit.set_robot_shape(w, name, kind, size))

    def _body_size(self, name: str) -> None:
        """A size field changed."""
        f = self.fields
        if "radius" in f:
            kind, size = "circle", (f["radius"].value,)
        else:
            kind, size = "box", (f["length"].value, f["width"].value)
        self._robot_edit(name, lambda w: edit.set_robot_shape(w, name, kind, size))

    # -- pose source and sensors --
    def sensing(self, spec) -> None:
        """Where the controller's pose comes from, and the sensors."""
        name = spec.name
        self.column.add(Section("Pose source"))
        self.choice(
            "estimator",
            "estimator",
            list(options.estimators()),
            spec.estimator,
            lambda text: self._robot_edit(
                name, lambda w: edit.set_estimator(w, name, text)
            ),
        )
        self.choice(
            "control_pose",
            "controller sees",
            ["estimate", "truth"],
            spec.control_pose,
            lambda text: self._robot_edit(
                name, lambda w: edit.set_control_pose(w, name, text)
            ),
        )
        self.column.add(Section("Sensors"))
        for sensor, settings in spec.sensors.items():
            self.sensor(name, sensor, settings)
        kinds = {s.get("type", n) for n, s in spec.sensors.items()}
        if "encoders" not in kinds:
            self.button(
                "add_encoders",
                "Add encoders",
                lambda: self._add_sensor(spec, "encoders"),
            )
        self.button("add_lidar", "Add lidar", lambda: self._add_sensor(spec, "lidar"))

    def _add_sensor(self, spec, kind: str) -> None:
        """Add a sensor of a kind under a name no other sensor has."""
        taken = set(spec.sensors)
        name = next(
            n for n in (kind, *(f"{kind}_{k}" for k in range(2, 100))) if n not in taken
        )
        settings = {"type": kind} if kind == "lidar" else {}
        self._robot_edit(
            spec.name, lambda w: edit.set_sensor(w, spec.name, name, settings)
        )

    def sensor(self, robot: str | None, sensor: str, settings: dict) -> None:
        """The settings of one sensor; `robot` None is a default of every robot."""
        kind = settings.get("type", sensor)
        title = kind if sensor == kind else f"{sensor} - {kind}"
        self.column.add(Section(title))
        for setting in options.sensor_settings(kind):
            value = settings.get(setting.name, setting.default)
            self.number(
                f"{sensor}.{setting.name}",
                setting.name.replace("_", " "),
                value,
                _sensor_spec(setting),
                lambda s=sensor, k=kind: self._sensor_edited(robot, s, k),
            )
        if kind == "lidar":
            mount = settings.get("mount", (0.0, 0.0, 0.0))
            for i, label in enumerate(("mount x", "mount y", "mount angle")):
                self.number(
                    f"{sensor}.mount{i}",
                    label,
                    float(mount[i]),
                    NumberSpec(-100.0, 100.0, 0.05, 3),
                    lambda s=sensor, k=kind: self._sensor_edited(robot, s, k),
                )
        self.button(
            f"{sensor}.remove",
            f"Remove {sensor}",
            lambda: self._sensor_change(
                robot, lambda w: edit.set_sensor(w, robot, sensor, None)
            ),
        )

    def _sensor_edited(self, robot: str | None, sensor: str, kind: str) -> None:
        """A sensor field changed; every field of it is read back."""
        f = self.fields
        values = {
            s.name: f[f"{sensor}.{s.name}"].value for s in options.sensor_settings(kind)
        }
        if kind == "lidar":
            values["mount"] = [f[f"{sensor}.mount{i}"].value for i in range(3)]
            values["type"] = "lidar"
        self._sensor_change(robot, lambda w: edit.set_sensor(w, robot, sensor, values))

    def _sensor_change(self, robot: str | None, make) -> None:
        """Apply a sensor edit to a robot, or to the world's defaults."""
        if robot is None:
            self.done(self.session.edits.change_world(make))
        else:
            self._robot_edit(robot, make)

    # -- goals --
    def goals(self, name: str, pose) -> None:
        """The goals of a robot, in order, each with its tolerance, stop and wait."""
        world = self.session.world
        places = edit.goals_of(world, name)
        self.column.add(Section("Goal" if len(places) <= 1 else "Goals"))
        if not places:
            self.column.add(Label(NO_GOAL, 12, look.MUTED, wrap=True))
            self._goal_triple(name, 0, None, (float(pose[0]), float(pose[1]), 0.0))
        for k, place in enumerate(places):
            self._goal_block(name, k, place)
        self.button(
            "add_goal", "Add waypoint", lambda: self._add_goal(name, places, pose)
        )

    def _keys(self, k: int) -> tuple[str, str, str]:
        """The field keys of the k-th goal's x, y and heading."""
        return ("gx", "gy", "gth") if k == 0 else (f"g{k}x", f"g{k}y", f"g{k}th")

    def _goal_triple(self, name: str, k: int, place, goal) -> None:
        """Three fields - x, y and a heading in degrees - for one goal."""
        x0, y0, x1, y1 = self.session.world.bounds
        specs = (
            NumberSpec(x0, x1, 0.05, 2, " m"),
            NumberSpec(y0, y1, 0.05, 2, " m"),
            NumberSpec(-180, 180, 5, 2, " °"),
        )
        values = (goal[0], goal[1], math.degrees(goal[2]))
        for key, label, value, spec in zip(
            self._keys(k), ("x", "y", "heading"), values, specs
        ):
            self.number(
                key, label, value, spec, lambda: self._goal_edited(name, k, place)
            )

    def _goal_block(self, name: str, k: int, place: int) -> None:
        """One goal: its place, tolerance, stop rule and wait, and a remove button."""
        task = self.session.world.tasks[place]
        if k > 0:
            self.column.add(Label(f"Waypoint {k + 1}", 12, look.MUTED))
        self._goal_triple(name, k, place, tuple(float(v) for v in task.goal.xyt))
        change = functools.partial(self._goal_options, name, k, place)
        self.number(
            f"tol{k}d",
            "tolerance",
            task.goal_tolerance[0],
            NumberSpec(0.01, 10.0, 0.01, 2, " m"),
            change,
        )
        self.number(
            f"tol{k}a",
            "angle tolerance",
            math.degrees(task.goal_tolerance[1]),
            NumberSpec(0.1, 180.0, 1, 1, " °"),
            change,
        )
        current = next(k_ for k_, v in STOP_CHOICES.items() if v is task.stop)
        self.choice(
            f"stop{k}", "stop at goal", list(STOP_CHOICES), current, lambda _t: change()
        )
        self.number(
            f"wait{k}", "wait", task.wait, NumberSpec(0.0, 3600.0, 0.5, 2, " s"), change
        )
        self.button(
            f"remove{k}",
            "Remove goal",
            lambda: self._robot_edit(name, lambda w: edit.remove_goal(w, place)),
        )

    def _goal_edited(self, name: str, k: int, place) -> None:
        """A goal's place changed: move it, or make the first goal."""
        kx, ky, kth = self._keys(k)
        f = self.fields
        goal = (f[kx].value, f[ky].value, math.radians(f[kth].value))
        if place is None:
            self._robot_edit(name, lambda w: edit.add_goal(w, name, goal))
        else:
            self._robot_edit(name, lambda w: edit.move_goal(w, place, goal))

    def _goal_options(self, name: str, k: int, place: int) -> None:
        """A tolerance, stop or wait field changed."""
        f = self.fields
        options_ = {
            "tolerance": (f[f"tol{k}d"].value, math.radians(f[f"tol{k}a"].value)),
            "stop": STOP_CHOICES[f[f"stop{k}"].current],
            "wait": f[f"wait{k}"].value,
        }
        self._robot_edit(name, lambda w: edit.set_goal_options(w, place, **options_))

    def _add_goal(self, name: str, places: list[int], pose) -> None:
        """Add a goal a little beyond the last one, or beside the robot."""
        world = self.session.world
        base = world.tasks[places[-1]].goal.xyt if places else pose
        x0, y0, x1, y1 = world.bounds
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = base[0] + dx * WAYPOINT_STEP, base[1] + dy * WAYPOINT_STEP
            if x0 + 0.3 <= x <= x1 - 0.3 and y0 + 0.3 <= y <= y1 - 0.3:
                break
        goal = (float(x), float(y), float(base[2]))
        self._robot_edit(name, lambda w: edit.add_goal(w, name, goal))


def _sensor_spec(setting) -> NumberSpec:
    """How to show a sensor setting: whole numbers for a count, else metres or more."""
    if setting.kind is int:
        return NumberSpec(1, 100000, 1, 0, "", True)
    return NumberSpec(0.0, 10000.0, 0.01, 3)
