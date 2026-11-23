"""Edits of a session's world that are not about one robot - obstacles, size, files.

Each returns None if it worked, else a plain message; the session is changed only
if it worked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .. import edit
from ..generate import STYLES, Clearance, pair_world
from ..verify import WorldError
from ..world import GAZEBO_SUFFIXES as SUFFIXES
from ..world import World, load_world, save_world

REFUSED = "That change cannot be made: {}"


class WorldEdits:  # pylint: disable=too-many-public-methods
    """The obstacle, size and file edits of a session.

    Args:
        session: the session whose world is edited.

    Attributes:
        path: the file the world came from or was last saved to, or None.
        saved: the world as it was then; `modified` compares with it.
    """

    def __init__(self, session) -> None:
        """Work on a session's world."""
        self.session = session
        self.path: str | None = None
        self.saved = session.world
        # What the Generate section of the inspector asks for, kept between forms.
        self.recipe: dict = {
            "style": "caves",
            "seed": None,
            "radius": 0.25,
            "loops": 0.0,
            "pair": False,
        }

    @property
    def modified(self) -> bool:
        """True if the world has changed since it was opened or saved."""
        return self.session.world is not self.saved

    def _edit(
        self,
        make: Callable[[World], World],
        keep: int | None = None,
        *,
        rebuild: bool = False,
        select: str | None = None,
        actor: int | None = None,
    ):
        """Apply an edit to the world; `keep` is the obstacle left selected.

        `rebuild` is for an edit of the robots, `select` names the robot to leave
        selected, and `actor` the 0-based place of the actor to leave selected.
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
            session.scene.selected_actor = actor
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

    def change_actor(self, index: int, make: Callable[[World], World]) -> str | None:
        """Apply an edit of one actor, and keep that actor selected."""
        return self._edit(make, rebuild=True, select="", actor=index)

    def move_actor(self, index: int, pose: tuple[float, float, float]) -> str | None:
        """Move an actor; one that would leave the world stays where it was."""
        return self.change_actor(index, lambda w: edit.move_actor(w, index, pose))

    def resize_actor(self, index: int, size: tuple[float, ...]) -> str | None:
        """Give an actor a new body size."""
        return self.change_actor(index, lambda w: edit.resize_actor(w, index, size))

    def rename_actor(self, index: int, new: str) -> str | None:
        """Give an actor another name."""
        return self.change_actor(index, lambda w: edit.rename_actor(w, index, new))

    def set_actor_flag(self, index: int, key: str, on: bool) -> str | None:
        """Set `solid` or `collide` of an actor."""
        return self.change_actor(
            index, lambda w: edit.set_actor_flag(w, index, key, on)
        )

    def set_actor_motion(self, index: int, path: str | None) -> str | None:
        """Give an actor a motion, or none so it stands still."""
        return self.change_actor(index, lambda w: edit.set_actor_motion(w, index, path))

    def set_actor_motion_params(self, index: int, values: dict) -> str | None:
        """Change the settings of an actor's motion."""
        return self.change_actor(
            index, lambda w: edit.set_actor_motion_params(w, index, values)
        )

    def delete_actor(self) -> str | None:
        """Delete the selected actor."""
        index = self.session.scene.selected_actor
        if index is None:
            return self.session.guard()
        return self._edit(
            lambda w: edit.remove_actor(w, index), rebuild=True, select=""
        )

    def rename_robot(self, old: str, new: str) -> str | None:
        """Give a robot another name, and keep it selected."""
        return self._edit(
            lambda w: edit.rename_robot(w, old, new), rebuild=True, select=new.strip()
        )

    def rename_obstacle(self, index: int, new: str) -> str | None:
        """Give an obstacle a name, or none."""
        return self._edit(lambda w: edit.rename_obstacle(w, index, new), index)

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

    def generate(self) -> str | None:
        """Replace the obstacles with a map drawn from `recipe`.

        The robots, the goals and the actors stay. The map is drawn for the widest
        robot of the world, and for the recipe's radius when there is none. With
        `pair`, the first robot goes to a random start with a random goal, and a
        world with no robot gets one.
        """
        recipe = self.recipe

        def make(world: World) -> World:
            settings, draw = STYLES[recipe["style"]]
            seed = world.seed if recipe["seed"] is None else int(recipe["seed"])
            options = {"loops": recipe["loops"]} if recipe["style"] == "maze" else {}
            try:
                got = draw(
                    world.size,
                    seed,
                    settings(**options),
                    Clearance(radius=None if world.robots else recipe["radius"]),
                    origin=world.origin,
                    base=replace(world, obstacles=()),
                )
                return (
                    pair_world(got, seed, recipe["radius"])
                    if recipe["pair"]
                    else got.world
                )
            except ValueError as error:
                raise WorldError(str(error)) from error

        return self._edit(make, None, rebuild=True, select="")

    def save(self, path: str) -> str | None:
        """Write the world to a file, and remember the file."""
        try:
            save_world(self.session.world, path)
        except OSError as error:
            return f"Could not save: {error}"
        self.path, self.saved = path, self.session.world
        return None

    def open(self, path: str) -> str | None:
        """Replace the world with the one in a file."""
        if self.session.mode == "benchmark":
            return "A benchmark world is fixed."
        try:
            world = load_world(path)
        except (OSError, ValueError) as error:
            return f"Could not open {path}: {error}"
        # A Gazebo file has no TOML to save back to, so Save asks for a name.
        return self._replace(world, None if Path(path).suffix in SUFFIXES else path)

    def new(self) -> str | None:
        """Replace the world with a blank one, which has no file yet."""
        if self.session.mode == "benchmark":
            return "A benchmark world is fixed."
        return self._replace(edit.empty_world(), None)

    def _replace(self, world: World, path: str | None) -> str | None:
        """Make another world the one that runs, ending any run in progress."""
        session = self.session
        session.playing = False
        message = session.apply(world, rebuild=True, select="", remember=False)
        if message is None:
            session.history.clear()
            session.scene.selected_obstacle = None
            self.path, self.saved = path, session.world
        return message
