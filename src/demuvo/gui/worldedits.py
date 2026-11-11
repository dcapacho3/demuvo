"""Edits of a session's world that are not about one robot - obstacles, size, files.

Each returns None if it worked, else a plain message; the session is changed only
if it worked.
"""

from __future__ import annotations

from collections.abc import Callable

from .. import edit
from ..verify import WorldError
from ..world import World, load_world, save_world

REFUSED = "That change cannot be made: {}"


class WorldEdits:
    """The obstacle, size and file edits of a session.

    Args:
        session: the session whose world is edited.

    Attributes:
        path: the file the world came from or was last saved to, or None.
    """

    def __init__(self, session) -> None:
        """Work on a session's world."""
        self.session = session
        self.path: str | None = None

    def _edit(
        self,
        make: Callable[[World], World],
        keep: int | None = None,
        *,
        rebuild: bool = False,
        select: str | None = None,
    ):
        """Apply an edit to the world; `keep` is the obstacle left selected.

        `rebuild` is for an edit of the robots, and `select` names the robot to
        leave selected.
        """
        session = self.session
        if (why := session.guard()) is not None:
            return why
        try:
            world = make(session.world)
        except WorldError as error:
            return REFUSED.format(error)
        message = session.apply(world, rebuild=rebuild, select=select)
        if message is None:
            session.scene.selected_obstacle = keep
        return message

    def change_world(self, make: Callable[[World], World]) -> str | None:
        """Apply any edit of the world that is not about one robot."""
        return self._edit(make, self.session.scene.selected_obstacle)

    def change_robot(self, name: str, make: Callable[[World], World]) -> str | None:
        """Apply an edit that changes a robot, and keep that robot selected."""
        return self._edit(make, rebuild=True, select=name)

    def add_obstacle(
        self, kind: str, pose: tuple[float, float, float], size: tuple[float, ...]
    ) -> str | None:
        """Add a box or circle and select it."""
        session = self.session
        place = len(session.world.obstacles)
        message = self._edit(lambda w: edit.add_obstacle(w, kind, pose, size), place)
        if message is None:
            session.scene.selected = None
        return message

    def add_outline(self, kind: str, vertices) -> str | None:
        """Add a polygon or a line through points, and select it."""
        session = self.session
        place = len(session.world.obstacles)
        message = self._edit(lambda w: edit.add_outline(w, kind, vertices), place)
        if message is None:
            session.scene.selected = None
        return message

    def move_obstacle(self, index: int, pose: tuple[float, float, float]) -> str | None:
        """Move an obstacle; one that would leave the world or touch a robot stays."""
        return self._edit(lambda w: edit.move_obstacle(w, index, pose), index)

    def resize_obstacle(self, index: int, size: tuple[float, ...]) -> str | None:
        """Give an obstacle a new size."""
        return self._edit(lambda w: edit.resize_obstacle(w, index, size), index)

    def delete_obstacle(self) -> str | None:
        """Delete the selected obstacle."""
        index = self.session.scene.selected_obstacle
        if index is None:
            return self.session.guard()
        return self._edit(lambda w: edit.remove_obstacle(w, index))

    def set_size(self, size: tuple[float, float]) -> str | None:
        """Give the world a new width and height."""
        return self._edit(
            lambda w: edit.set_size(w, size), self.session.scene.selected_obstacle
        )

    def set_seed(self, seed: int) -> str | None:
        """Give the world a seed."""
        return self._edit(
            lambda w: edit.set_seed(w, seed), self.session.scene.selected_obstacle
        )

    def set_step(self, dt: float) -> str | None:
        """Give the world a step length in seconds."""
        return self._edit(
            lambda w: edit.set_step(w, dt), self.session.scene.selected_obstacle
        )

    def save(self, path: str) -> str | None:
        """Write the world to a file, and remember the file."""
        try:
            save_world(self.session.world, path)
        except OSError as error:
            return f"Could not save: {error}"
        self.path = path
        return None

    def open(self, path: str) -> str | None:
        """Replace the world with the one in a file."""
        if self.session.mode == "benchmark":
            return "A benchmark world is fixed."
        try:
            world = load_world(path)
        except (OSError, ValueError) as error:
            return f"Could not open {path}: {error}"
        return self._replace(world, path)

    def new(self) -> str | None:
        """Replace the world with a blank one, which has no file yet."""
        if self.session.mode == "benchmark":
            return "A benchmark world is fixed."
        return self._replace(edit.new_world(), None)

    def _replace(self, world: World, path: str | None) -> str | None:
        """Make another world the one that runs, ending any run in progress."""
        session = self.session
        session.playing = False
        message = session.apply(world, rebuild=True, select="")
        if message is None:
            session.scene.selected_obstacle = None
            self.path = path
        return message
