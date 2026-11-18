"""The scene pane - the world in the window, and what the mouse does to it.

Click a robot or an obstacle to select it; drag it to move it while editing. With
the goal tool, press on the floor to put the selected robot's goal there, drag to
turn its arrow, and let go to set it, as RViz's 2D Goal Pose does; a click alone
sets the heading to 0. Hold Shift to add a waypoint after the goals it has instead.
With the box or circle tool, drag on the floor to draw an obstacle; with the
polygon or line tool, click its corners. The playback pill floats over the bottom
of the pane.
"""

from __future__ import annotations

import math
import time

import numpy as np
from robomath import Transform2D

from ..world.shapes import Circle
from . import font, look
from .look import TEAL
from .panels import Pill
from .raster import Canvas, rgba
from .scenedraw import arrow
from .session import Session
from .strokes import polyline
from .widgets import Event, Widget, text_baseline

# The smallest obstacle a drag makes, in metres; less is taken for a click.
MIN_SIZE = 0.1
# Said while a polygon or line is being drawn.
OUTLINE_HINT = "Double click, right click or press Enter to finish. Esc cancels."
# A click this close, in pixels, to the first corner closes a polygon.
CLOSE_PIXELS = 20.0
# Two clicks within this many seconds and pixels are a double click, which ends a
# polygon or line.
DOUBLE_CLICK = (0.6, 12.0)
# The length in pixels of the arrow shown while a goal is aimed.
AIM_LENGTH = 48.0


class ScenePane(Widget):  # pylint: disable=too-many-instance-attributes
    """The widget that shows the scene and takes the mouse.

    Args:
        session: the session whose scene it shows.
        pill: the playback pill.
        actions: an object that is told what the mouse did; it has
            `select(name)`, `select_obstacle(index)`, `goal_set(name, x, y,
            heading)`, `waypoint_added(name, x, y, heading)`, `dragged()`,
            `dropped(name, x, y, angle)`, `robot_moved(name, x, y, angle)`,
            `obstacle_moved(index, x, y)`, `obstacle_drawn(kind, pose, size)`,
            `outline_drawn(kind, vertices)`, `select_actor(index)`,
            `actor_dropped(index, x, y, angle)` and `sketch_changed()`.
    """

    def __init__(self, session: Session, pill: Pill, actions) -> None:
        """Make the pane, with the pill inside it."""
        super().__init__()
        self.session, self.pill, self.actions = session, pill, actions
        self.add(pill)
        self.dragging: str | None = None
        self.grab = (0.0, 0.0)
        self.moving: int | None = None
        self.moving_actor: int | None = None
        self.moving_point: int | None = None
        self.ghost: tuple[int, float, float] | None = None
        self.sketch: tuple[float, float, float, float] | None = None
        self.outline: list[tuple[float, float]] = []
        self.cursor: tuple[float, float] | None = None
        # The goal being aimed - robot, its place, and whether Shift was held - and
        # the world point the pointer has dragged to.
        self.aiming: tuple[str, float, float, bool] | None = None
        self.aim_to: tuple[float, float] | None = None
        self._last_click: tuple[float, float, float] | None = None

    def on_hover(self, inside: bool) -> None:
        """Note that the pointer is over the scene; it looks the same, so no redraw."""
        self.hover = inside
        if not inside:
            self.down = False

    def rect(self) -> tuple[float, float, float, float]:
        """The pane's rectangle."""
        return (self.x, self.y, self.width, self.height)

    def layout(self) -> None:
        """Fit the board to the pane and put the pill at the bottom."""
        self.session.scene.fit(self.rect())
        self.pill.place(self.x + self.width / 2, self.y + self.height - 16)

    def draw(self, canvas: Canvas) -> None:
        """Draw the scene and the pill over it."""
        self.session.scene.draw(canvas)
        self._draw_sketch(canvas)
        self._draw_outline(canvas)
        self._draw_ghost(canvas)
        self._draw_aim(canvas)
        self.pill.draw(canvas)

    def _draw_sketch(self, canvas: Canvas) -> None:
        """Outline the obstacle being drawn."""
        if self.sketch is None:
            return
        view = self.session.scene.view
        (x0, y0), (x1, y1) = (
            (float(v) for v in view.to_screen(*self.sketch[:2])),
            (float(v) for v in view.to_screen(*self.sketch[2:])),
        )
        color = rgba(TEAL)
        if self.session.tool == "circle":
            canvas.circle((x0, y0), math.hypot(x1 - x0, y1 - y0), stroke=color, width=2)
        else:
            box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            canvas.rect_outline(box, color, width=2, radius=0)

    def _draw_aim(self, canvas: Canvas) -> None:
        """Draw the arrow of the goal being aimed, along the drag."""
        if self.aiming is None:
            return
        _, x, y, _ = self.aiming
        heading = _aimed(x, y, self.aim_to)
        origin = tuple(float(v) for v in self.session.scene.view.to_screen(x, y))
        arrow(canvas, origin, -heading, AIM_LENGTH, rgba(TEAL), outline=rgba("#ffffff"))

    def _draw_ghost(self, canvas: Canvas) -> None:
        """Outline the obstacle where the pointer holds it.

        The world follows a few times a second, so the outline is what moves smoothly.
        """
        if self.ghost is None:
            return
        index, x, y = self.ghost
        item = self.session.world.obstacles[index]
        pose = Transform2D.from_xyt((x, y, float(item.pose.xyt[2])))
        view, color = self.session.scene.view, rgba(TEAL)
        if isinstance(item.shape, Circle):
            cx, cy = (float(v) for v in view.to_screen(x, y))
            canvas.circle(
                (cx, cy), item.shape.radius * view.scale, stroke=color, width=2
            )
            return
        corners = np.array(item.shape.points(pose), dtype=float)
        sx, sy = view.to_screen(corners[:, 0], corners[:, 1])
        ring = np.column_stack([sx, sy])
        if item.shape.closed:
            ring = np.vstack([ring, ring[:1]])
        polyline(canvas, ring, 2, color, dash=(6, 4))

    def _draw_outline(self, canvas: Canvas) -> None:
        """Draw the polygon or line so far, with a dashed segment to the pointer."""
        if not self.outline:
            return
        view = self.session.scene.view
        color = rgba(TEAL)
        canvas.text(
            font.regular(),
            OUTLINE_HINT,
            (self.x + 16, text_baseline(self.y + 8, 24, 12)),
            size=12,
            color=rgba(look.NAVY),
        )
        points = [tuple(float(v) for v in view.to_screen(*p)) for p in self.outline]
        if len(points) > 1:
            polyline(canvas, np.array(points), 2.5, color)
        if self.cursor is not None:
            ahead = tuple(float(v) for v in view.to_screen(*self.cursor))
            polyline(canvas, np.array([points[-1], ahead]), 1.5, color, dash=(6, 4))
        for x, y in points:
            canvas.circle((x, y), 4, fill=color)
        if self.session.tool == "polygon" and len(points) >= 3:
            canvas.circle(points[0], CLOSE_PIXELS, stroke=rgba(TEAL, 0.5), width=1.5)

    def hit(self, px: float, py: float) -> Widget | None:
        """The pill if the point is on it, else the pane."""
        if not self.visible or not self.contains(px, py):
            return None
        found = self.pill.hit(px, py)
        return found if found is not None else self

    def on_press(self, event: Event) -> None:
        """Select a robot, start dragging it, or place a goal."""
        session = self.session
        scene = session.scene
        name = scene.robot_at(event.x, event.y)
        if session.tool in ("polygon", "line") and session.editable:
            self._add_vertex(event)
            return
        if session.tool in ("box", "circle") and session.editable:
            x, y = scene.view.to_world(event.x, event.y)
            self.sketch = (x, y, x, y)
            return
        if (
            session.tool == "goal"
            and scene.selected
            and name is None
            and session.editable
        ):
            x, y = scene.view.to_world(event.x, event.y)
            self.aiming, self.aim_to = (scene.selected, x, y, event.shift), None
            return
        if not name and session.editable and self._route_pressed(event):
            return
        spot = None if name else scene.actor_at(event.x, event.y)
        if spot is not None:
            self.actions.select_actor(spot)
            if session.editable:
                self.moving_actor = spot
                session.history.begin()
                x, y = scene.view.to_world(event.x, event.y)
                px, py, angle = scene.actor_pose(spot)
                self.grab = (float(px) - x, float(py) - y)
                scene.drag_actor(spot, float(px), float(py), float(angle))
            return
        place = None if name else scene.obstacle_at(event.x, event.y)
        if place is not None:
            self.actions.select_obstacle(place)
            if session.editable:
                self.moving = place
                session.history.begin()
                x, y = scene.view.to_world(event.x, event.y)
                px, py = session.world.obstacles[place].pose.translation
                self.grab = (float(px) - x, float(py) - y)
            return
        self.actions.select(name)
        if name and session.editable:
            self.dragging = name
            session.history.begin()
            x, y = scene.view.to_world(event.x, event.y)
            px, py, angle = scene.pose(name)
            self.grab = (px - x, py - y)
            scene.drag(name, px, py, float(angle))

    def _params(self, index: int) -> dict:
        """The motion settings of an actor."""
        return self.session.world.actors[index].motion_params

    def _route_pressed(self, event: Event) -> bool:
        """Grab or make a waypoint of the selected actor; True if it took the press.

        A dot is dragged. The middle of a leg puts a new waypoint there and drags
        it. Shift and a click on the floor puts one at the end.
        """
        scene = self.session.scene
        index = scene.selected_actor
        if (
            index is None
            or "points" not in self.session.world.actors[index].motion_params
        ):
            return False
        x, y = scene.view.to_world(event.x, event.y)
        place = scene.point_at(event.x, event.y)
        if place is None and scene.leg_at(event.x, event.y) is not None:
            place = scene.leg_at(event.x, event.y)
            self.actions.point_inserted(index, place, x, y)
        elif place is None and event.shift and scene.actor_at(event.x, event.y) is None:
            count = len(self._params(index)["points"])
            self.actions.point_inserted(index, count, x, y)
            return True
        if place is None:
            return False
        self.moving_point = place
        self.session.history.begin()
        return True

    def on_move(self, event: Event) -> None:
        """Carry the dragged robot or obstacle, or stretch the sketch."""
        if self.outline:
            self.cursor = self.session.scene.view.to_world(event.x, event.y)
            self.actions.sketch_changed()
        elif self.aiming is not None:
            self.aim_to = self.session.scene.view.to_world(event.x, event.y)
            self.actions.sketch_changed()
        elif self.sketch is not None:
            x, y = self.session.scene.view.to_world(event.x, event.y)
            self.sketch = (*self.sketch[:2], x, y)
            self.actions.dragged()
        elif self.moving is not None:
            x, y = self.session.scene.view.to_world(event.x, event.y)
            self.ghost = (self.moving, x + self.grab[0], y + self.grab[1])
            self.actions.obstacle_moved(*self.ghost)
            self.actions.sketch_changed()
        elif self.moving_point is not None:
            scene = self.session.scene
            x, y = scene.view.to_world(event.x, event.y)
            scene.drag_point = (self.moving_point, x, y)
            self.actions.sketch_changed()
        elif self.moving_actor is not None:
            scene = self.session.scene
            x, y = scene.view.to_world(event.x, event.y)
            angle = float(scene.actor_pose(self.moving_actor)[2])
            scene.drag_actor(
                self.moving_actor, x + self.grab[0], y + self.grab[1], angle
            )
            self.actions.dragged()
        elif self.dragging:
            scene = self.session.scene
            x, y = scene.view.to_world(event.x, event.y)
            angle = float(scene.pose(self.dragging)[2])
            place = (x + self.grab[0], y + self.grab[1], angle)
            scene.drag(self.dragging, *place)
            self.actions.dragged()
            self.actions.robot_moved(self.dragging, *place)

    def on_release(self, event: Event) -> None:
        """Drop the dragged robot or obstacle, or finish the obstacle being drawn."""
        if self.moving is not None and self.ghost is not None:
            self.actions.obstacle_moved(*self.ghost, force=True)
            self.ghost = None
            self.actions.sketch_changed()
        self.moving = None
        if self.aiming is not None:
            self._finish_aim()
        if self.sketch is not None:
            x0, y0, x1, y1 = self.sketch
            self.sketch = None
            self._finish_sketch(x0, y0, x1, y1)
        if self.moving_point is not None:
            scene = self.session.scene
            place, self.moving_point = self.moving_point, None
            drag, scene.drag_point = scene.drag_point, None
            if drag is not None:
                self.actions.point_dropped(scene.selected_actor, place, *drag[1:])
        if self.moving_actor is not None:
            scene = self.session.scene
            index, self.moving_actor = self.moving_actor, None
            x, y, angle = (float(v) for v in scene.actor_pose(index))
            scene.drop_actor()
            self.actions.actor_dropped(index, x, y, angle)
        if self.dragging:
            scene = self.session.scene
            name, self.dragging = self.dragging, None
            x, y, angle = (float(v) for v in scene.pose(name))
            scene.drop()
            self.actions.dropped(name, x, y, angle)
        self.session.history.end()
        self.down = False

    def _finish_aim(self) -> None:
        """Set the goal, or add a waypoint with Shift, where it was pressed.

        It is turned along the drag; a drag shorter than a click's worth leaves the
        heading at 0, as RViz does.
        """
        name, x, y, shift = self.aiming
        end, self.aiming, self.aim_to = self.aim_to, None, None
        heading = _aimed(x, y, end)
        if shift:
            self.actions.waypoint_added(name, x, y, heading)
        else:
            self.actions.goal_set(name, x, y, heading)
        self.actions.sketch_changed()

    def _finish_sketch(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Make the obstacle that was drawn, unless it is smaller than a click."""
        if self.session.tool == "circle":
            radius = math.hypot(x1 - x0, y1 - y0)
            if radius >= MIN_SIZE:
                self.actions.obstacle_drawn("circle", (x0, y0, 0.0), (radius,))
        else:
            size = (abs(x1 - x0), abs(y1 - y0))
            if min(size) >= MIN_SIZE:
                pose = ((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
                self.actions.obstacle_drawn("box", pose, size)
        self.actions.dragged()

    def _add_vertex(self, event: Event) -> None:
        """Add a corner; a click on the first corner, or a double click, ends it."""
        scene = self.session.scene
        point = scene.view.to_world(event.x, event.y)
        now = time.perf_counter()
        last, self._last_click = self._last_click, (now, event.x, event.y)
        seconds, pixels = DOUBLE_CLICK
        if (
            last is not None
            and now - last[0] <= seconds
            and math.hypot(event.x - last[1], event.y - last[2]) <= pixels
        ):
            self._last_click = None
            self.finish_outline()
            return
        if self.session.tool == "polygon" and len(self.outline) >= 3:
            fx, fy = scene.view.to_screen(*self.outline[0])
            if math.hypot(event.x - float(fx), event.y - float(fy)) <= CLOSE_PIXELS:
                self.finish_outline()
                return
        self.outline.append(point)
        self.cursor = point
        self.actions.sketch_changed()

    def on_secondary(self, event: Event) -> None:
        """A right click ends the polygon or line being drawn."""
        if self.outline:
            self.finish_outline()

    def finish_outline(self) -> None:
        """Make the polygon or line from the corners so far, if there are enough."""
        kind = "polygon" if self.session.tool == "polygon" else "linestring"
        need = 3 if kind == "polygon" else 2
        vertices = self.outline
        self.cancel_outline()
        if len(vertices) >= need:
            self.actions.outline_drawn(kind, vertices)

    def cancel_outline(self) -> None:
        """Forget the corners drawn so far."""
        had = bool(self.outline)
        self.outline, self.cursor, self._last_click = [], None, None
        if had:
            self.actions.sketch_changed()


def _aimed(x: float, y: float, end) -> float:
    """The heading of a goal at (x, y) dragged to `end`; 0 if it was not dragged."""
    if end is None or math.hypot(end[0] - x, end[1] - y) < MIN_SIZE:
        return 0.0
    return math.atan2(end[1] - y, end[0] - x)
