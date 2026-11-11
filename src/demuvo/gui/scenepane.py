"""The scene pane - the world in the window, and what the mouse does to it.

Click a robot or an obstacle to select it; drag it to move it while editing. With
the goal tool, click the floor to give the selected robot a goal, or Shift and click
to add a waypoint. With the box or circle tool, drag on the floor to draw an
obstacle; with the polygon or line tool, click its corners. The playback pill floats
over the bottom of the pane.
"""

from __future__ import annotations

import math

import numpy as np

from .look import TEAL
from .panels import Pill
from .raster import Canvas, rgba
from .session import Session
from .strokes import polyline
from .widgets import Event, Widget

# The smallest obstacle a drag makes, in metres; less is taken for a click.
MIN_SIZE = 0.1
# A click this close, in pixels, to the first corner closes a polygon.
CLOSE_PIXELS = 12.0


class ScenePane(Widget):
    """The widget that shows the scene and takes the mouse.

    Args:
        session: the session whose scene it shows.
        pill: the playback pill.
        actions: an object that is told what the mouse did; it has
            `select(name)`, `select_obstacle(index)`, `goal_set(name, x, y,
            heading)`, `waypoint_added(name, x, y, heading)`, `dragged()`,
            `dropped(name, x, y, angle)`,
            `obstacle_moved(index, x, y)`, `obstacle_drawn(kind, pose, size)`,
            `outline_drawn(kind, vertices)` and `sketch_changed()`.
    """

    def __init__(self, session: Session, pill: Pill, actions) -> None:
        """Make the pane, with the pill inside it."""
        super().__init__()
        self.session, self.pill, self.actions = session, pill, actions
        self.add(pill)
        self.dragging: str | None = None
        self.grab = (0.0, 0.0)
        self.moving: int | None = None
        self.sketch: tuple[float, float, float, float] | None = None
        self.outline: list[tuple[float, float]] = []
        self.cursor: tuple[float, float] | None = None

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

    def _draw_outline(self, canvas: Canvas) -> None:
        """Draw the polygon or line so far, with a dashed segment to the pointer."""
        if not self.outline:
            return
        view = self.session.scene.view
        color = rgba(TEAL)
        points = [tuple(float(v) for v in view.to_screen(*p)) for p in self.outline]
        if len(points) > 1:
            polyline(canvas, np.array(points), 2.5, color)
        if self.cursor is not None:
            ahead = tuple(float(v) for v in view.to_screen(*self.cursor))
            polyline(canvas, np.array([points[-1], ahead]), 1.5, color, dash=(6, 4))
        for x, y in points:
            canvas.circle((x, y), 4, fill=color)

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
            heading = scene.heading_to(scene.selected, x, y)
            if event.shift:
                self.actions.waypoint_added(scene.selected, x, y, heading)
            else:
                self.actions.goal_set(scene.selected, x, y, heading)
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

    def on_move(self, event: Event) -> None:
        """Carry the dragged robot or obstacle, or stretch the sketch."""
        if self.outline:
            self.cursor = self.session.scene.view.to_world(event.x, event.y)
            self.actions.sketch_changed()
        elif self.sketch is not None:
            x, y = self.session.scene.view.to_world(event.x, event.y)
            self.sketch = (*self.sketch[:2], x, y)
            self.actions.dragged()
        elif self.moving is not None:
            x, y = self.session.scene.view.to_world(event.x, event.y)
            self.actions.obstacle_moved(self.moving, x + self.grab[0], y + self.grab[1])
        elif self.dragging:
            scene = self.session.scene
            x, y = scene.view.to_world(event.x, event.y)
            angle = float(scene.pose(self.dragging)[2])
            scene.drag(self.dragging, x + self.grab[0], y + self.grab[1], angle)
            self.actions.dragged()

    def on_release(self, event: Event) -> None:
        """Drop the dragged robot, or finish the obstacle being drawn."""
        self.moving = None
        if self.sketch is not None:
            x0, y0, x1, y1 = self.sketch
            self.sketch = None
            self._finish_sketch(x0, y0, x1, y1)
        if self.dragging:
            scene = self.session.scene
            name, self.dragging = self.dragging, None
            x, y, angle = (float(v) for v in scene.pose(name))
            scene.drop()
            self.actions.dropped(name, x, y, angle)
        self.session.history.end()
        self.down = False

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
        """Add a corner; a click on the first corner of a polygon closes it."""
        scene = self.session.scene
        point = scene.view.to_world(event.x, event.y)
        if self.session.tool == "polygon" and len(self.outline) >= 3:
            fx, fy = scene.view.to_screen(*self.outline[0])
            if math.hypot(event.x - float(fx), event.y - float(fy)) <= CLOSE_PIXELS:
                self.finish_outline()
                return
        self.outline.append(point)
        self.cursor = point
        self.actions.sketch_changed()

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
        self.outline, self.cursor = [], None
        if had:
            self.actions.sketch_changed()
