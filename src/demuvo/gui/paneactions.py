"""What the scene pane tells the sandbox - the mouse on robots and obstacles.

The pane calls these; each turns a drag, a drop or a drawing into a session edit
and then shows the result or the reason it was refused.
"""

from __future__ import annotations

import time

from .. import edit

# A dragged robot is made part of the world at most this often, in seconds.
FOLLOW_SECONDS = 0.1


class PaneActions:
    """The pane's side of the sandbox.

    Args:
        program: the sandbox, which shows messages and redraws.
    """

    def __init__(self, program) -> None:
        """Work for a sandbox."""
        self.program = program
        self.session = program.session
        self._last_move = 0.0

    def select(self, name: str | None) -> None:
        """Select a robot or none."""
        self.program.select(name)

    def select_obstacle(self, index: int) -> None:
        """Select an obstacle by its 0-based place."""
        if self.session.select_obstacle(index):
            self.program.refresh()

    def dragged(self) -> None:
        """A robot is being dragged: show its pose, redraw only the robots."""
        self.program.inspector.sync_pose()
        self.program.shell.screen.damage.add(self.session.scene.dirty_rect())

    def goal_set(self, name: str, x: float, y: float, heading: float) -> None:
        """A goal was picked with the goal tool."""
        self.program.edited(self.session.set_goal(name, (x, y, heading)))

    def waypoint_added(self, name: str, x: float, y: float, heading: float) -> None:
        """A goal was added after the robot's others, with Shift and the goal tool."""
        self.program.edited(
            self.session.edits.change_robot(
                name, lambda w: edit.add_goal(w, name, (x, y, heading))
            )
        )

    def robot_moved(self, name: str, x: float, y: float, angle: float) -> None:
        """A robot is being dragged: now and then the world and its path follow.

        A place that is refused leaves the world as it was and the robot drawn where
        the pointer is. The drop makes the final edit.
        """
        now = time.perf_counter()
        if now - self._last_move < FOLLOW_SECONDS:
            return
        self._last_move = now
        self.session.move_robot(name, (x, y, angle))
        self.program.scene_changed()

    def dropped(self, name: str, x: float, y: float, angle: float) -> None:
        """A dragged robot was dropped."""
        self.program.edited(self.session.move_robot(name, (x, y, angle)))

    def obstacle_moved(
        self, index: int, x: float, y: float, force: bool = False
    ) -> None:
        """An obstacle is being dragged; the world follows only at the drop.

        Making the world follow costs about a tenth of a second, which would make the
        drag stutter, so the pane draws an outline that moves with the pointer, and
        the drop passes `force` to make the edit.
        """
        if not force:
            return
        angle = float(self.session.world.obstacles[index].pose.xyt[2])
        error = self.session.edits.move_obstacle(index, (x, y, angle))
        if error is not None:
            self.program.message(error)
        self.program.inspector.sync_pose()
        self.program.scene_changed()

    def obstacle_drawn(self, kind: str, pose, size) -> None:
        """An obstacle was drawn with the box or circle tool."""
        self._drawn(self.session.edits.add_obstacle(kind, pose, size))

    def outline_drawn(self, kind: str, vertices) -> None:
        """A polygon or line was drawn with its tool."""
        self._drawn(self.session.edits.add_outline(kind, vertices))

    def _drawn(self, error: str | None) -> None:
        """Show the result of a drawing, then go back to the select tool.

        The next drag then moves the new shape instead of drawing another.
        """
        self.program.edited(error)
        if error is None:
            self.program.set_tool("select")

    def sketch_changed(self) -> None:
        """The outline being drawn changed, so the scene is drawn again."""
        self.program.scene_changed()
