"""What the scene pane tells the sandbox - the mouse on robots and obstacles.

The pane calls these; each turns a drag, a drop or a drawing into a session edit
and then shows the result or the reason it was refused.
"""

from __future__ import annotations


class PaneActions:
    """The pane's side of the sandbox.

    Args:
        program: the sandbox, which shows messages and redraws.
    """

    def __init__(self, program) -> None:
        """Work for a sandbox."""
        self.program = program
        self.session = program.session

    def select(self, name: str | None) -> None:
        """Select a robot or none."""
        self.program.select(name)

    def select_obstacle(self, index: int) -> None:
        """Select an obstacle by its 0-based place."""
        if self.session.select_obstacle(index):
            self.program.refresh()

    def dragged(self) -> None:
        """A robot is being dragged: show its pose in the form and redraw."""
        self.program.inspector.sync_pose()
        self.program.scene_changed()

    def goal_set(self, name: str, x: float, y: float, heading: float) -> None:
        """A goal was picked with the goal tool."""
        self.program.edited(self.session.set_goal(name, (x, y, heading)))

    def dropped(self, name: str, x: float, y: float, angle: float) -> None:
        """A dragged robot was dropped."""
        self.program.edited(self.session.move_robot(name, (x, y, angle)))

    def obstacle_moved(self, index: int, x: float, y: float) -> None:
        """An obstacle is being dragged; keep it where it can be."""
        angle = float(self.session.world.obstacles[index].pose.xyt[2])
        error = self.session.edits.move_obstacle(index, (x, y, angle))
        if error is not None:
            self.program.message(error)
        self.dragged()

    def obstacle_drawn(self, kind: str, pose, size) -> None:
        """An obstacle was drawn with the box or circle tool."""
        self.program.edited(self.session.edits.add_obstacle(kind, pose, size))
