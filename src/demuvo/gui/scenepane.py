"""The scene pane - the world in the window, and what the mouse does to it.

Click a robot to select it; drag it to move it while editing; with the goal tool,
click the floor to give the selected robot a goal. The playback pill floats over
the bottom of the pane.
"""

from __future__ import annotations

from .panels import Pill
from .raster import Canvas
from .session import Session
from .widgets import Event, Widget


class ScenePane(Widget):
    """The widget that shows the scene and takes the mouse.

    Args:
        session: the session whose scene it shows.
        pill: the playback pill.
        actions: an object that is told what the mouse did; it has
            `select(name)`, `goal_set(name, x, y, heading)`,
            `dragged()` and `dropped(name, x, y, angle)`.
    """

    def __init__(self, session: Session, pill: Pill, actions) -> None:
        """Make the pane, with the pill inside it."""
        super().__init__()
        self.session, self.pill, self.actions = session, pill, actions
        self.add(pill)
        self.dragging: str | None = None
        self.grab = (0.0, 0.0)

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
        self.pill.draw(canvas)

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
        if (
            session.tool == "goal"
            and scene.selected
            and name is None
            and session.editable
        ):
            x, y = scene.view.to_world(event.x, event.y)
            heading = session.heading_to(scene.selected, x, y)
            self.actions.goal_set(scene.selected, x, y, heading)
            return
        self.actions.select(name)
        if name and session.editable:
            self.dragging = name
            x, y = scene.view.to_world(event.x, event.y)
            px, py, angle = scene.pose(name)
            self.grab = (px - x, py - y)
            scene.drag(name, px, py, float(angle))

    def on_move(self, event: Event) -> None:
        """Carry the dragged robot with the pointer."""
        if self.dragging:
            scene = self.session.scene
            x, y = scene.view.to_world(event.x, event.y)
            angle = float(scene.pose(self.dragging)[2])
            scene.drag(self.dragging, x + self.grab[0], y + self.grab[1], angle)
            self.actions.dragged()

    def on_release(self, event: Event) -> None:
        """Drop the dragged robot where it is."""
        if self.dragging:
            scene = self.session.scene
            name, self.dragging = self.dragging, None
            x, y, angle = (float(v) for v in scene.pose(name))
            scene.drop()
            self.actions.dropped(name, x, y, angle)
        self.down = False
