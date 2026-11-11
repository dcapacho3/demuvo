"""A session - the world being looked at or edited, and the rules for changing it.

There are two modes. In a benchmark the world is a fixed file: nothing can be edited,
and Reset repeats the run exactly. In the sandbox the world is one you change: add
robots, pick their parts, set goals, then run. Edits are allowed only while no run
is in progress; Reset ends the run and unlocks them.

A session has no widgets. Each change returns an error message in plain words, or
None if it worked, and a window shows the message.
"""

from __future__ import annotations

from .. import edit, options
from ..runner import Run
from ..verify import WorldError
from ..world import World
from .history import History
from .remote import RemoteRun
from .scenedraw import SceneDraw
from .worldedits import REFUSED, WorldEdits

MODES = ("sandbox", "benchmark")
# A robot dragged to the edge stops this far from it, in metres.
EDGE = 0.25
# One step of the Step button is about this long, in seconds.
STEP_SECONDS = 0.05
# The most real time one frame may count, in seconds, so a slow frame does not leap.
MAX_FRAME = 0.1
PART_TABLES = {
    "planner": options.planners,
    "smoother": options.smoothers,
    "controller": options.controllers,
}


class Session:
    """The world and its run, in one mode.

    Args:
        source: a world file, or a `World`; None in the sandbox starts a blank world.
        mode: "sandbox" or "benchmark".
        remote: True to simulate in a process of its own, so drawing never waits.

    Attributes:
        mode: "sandbox" or "benchmark".
        run: the run, which the scene reads.
        scene: the drawing of the run.
        tool: "select", "goal", "box" or "circle", the tool the pointer has.
        edits: the edits of obstacles and of the world, and saving and opening.
        history: the undo and redo lists.
        playing: True while the run advances by itself.
        speed: how many seconds of simulated time pass for each real second.

    Raises:
        ValueError: if the mode is unknown, or a benchmark has no world.
    """

    def __init__(
        self,
        source: str | World | None = None,
        mode: str = "sandbox",
        *,
        remote: bool = False,
    ) -> None:
        """Open the world."""
        if mode not in MODES:
            raise ValueError(f"mode must be 'sandbox' or 'benchmark', not {mode!r}")
        if mode == "benchmark" and source is None:
            raise ValueError("a benchmark needs a world file")
        self.mode = mode
        world = edit.new_world() if source is None else source
        self.run = RemoteRun(world) if remote else Run(world)
        self.scene = SceneDraw(self.run)
        self._seen = 0.0
        self._owed = 0.0
        self.tool = "select"
        self.edits = WorldEdits(self)
        self.history = History(self)
        self.playing = False
        self.speed = 1.0

    @property
    def editable(self) -> bool:
        """True if the world may be changed now: sandbox, at time zero, not playing."""
        return self.mode == "sandbox" and self.run.time == 0 and not self.playing

    @property
    def state(self) -> str:
        """The state: "bench", "edit" while editing, or "run" once a run began."""
        if self.mode == "benchmark":
            return "bench"
        return "edit" if self.editable else "run"

    @property
    def world(self) -> World:
        """The world that runs."""
        return self.run.world

    def apply(
        self,
        world: World,
        *,
        rebuild: bool = False,
        select: str | None = None,
        remember: bool = True,
    ) -> str | None:
        """Make a world the one that runs, unless it cannot run.

        Args:
            world: the edited world.
            rebuild: True if the robots changed, which needs a new run and scene.
            select: the robot to select afterwards; "" for none; None keeps the
                selection.
            remember: True to put the world it replaces on the undo list.

        Returns:
            None if it worked, else why not; nothing then changes.
        """
        try:
            edit.validate(world)
        except ValueError as error:
            return REFUSED.format(error)
        keep = self.scene.selected if select is None else (select or None)
        previous = self.run.world
        self.run.set_world(world)
        if remember:
            self.history.remember(previous)
        if rebuild:
            old = self.scene
            self.scene = SceneDraw(self.run)
            self.scene.layers, self.scene.palette = old.layers, old.palette
        else:
            self.scene.set_run(self.run)
        self._seen = 0.0
        self._owed = 0.0
        self.scene.selected = keep
        self.scene.fit(self.scene.view.rect)
        return None

    def select(self, name: str | None) -> bool:
        """Select a robot, or none; True if the selection changed."""
        if name == self.scene.selected and self.scene.selected_obstacle is None:
            return False
        self.scene.selected = name
        self.scene.selected_obstacle = None
        return True

    def select_obstacle(self, index: int | None) -> bool:
        """Select an obstacle by its 0-based place, or none; True if it changed."""
        if index == self.scene.selected_obstacle and self.scene.selected is None:
            return False
        self.scene.selected_obstacle = index
        self.scene.selected = None
        return True

    def set_tool(self, tool: str) -> None:
        """Choose the "select", "goal", "box" or "circle" tool."""
        self.tool = tool

    def guard(self) -> str | None:
        """The reason nothing can be edited now, or None."""
        if self.editable:
            return None
        return (
            "Press Reset (R) to edit again."
            if self.mode == "sandbox"
            else "A benchmark world is fixed."
        )

    def add_robot(self) -> str | None:
        """Add a robot at a free place, and select it."""
        if (why := self.guard()) is not None:
            return why
        taken = {r.name for r in self.world.robots}
        name = next(f"r{k}" for k in range(1, 10_000) if f"r{k}" not in taken)
        try:
            world = edit.add_robot(self.world, name, edit.free_pose(self.world))
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world, rebuild=True, select=name)

    def delete_robot(self) -> str | None:
        """Delete the selected robot."""
        name = self.scene.selected
        if (why := self.guard()) is not None or name is None:
            return why
        try:
            world = edit.remove_robot(self.world, name)
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world, rebuild=True, select="")

    def move_robot(self, name: str, pose: tuple[float, float, float]) -> str | None:
        """Move a robot to a pose, held inside the world's edge."""
        x0, y0, x1, y1 = self.world.bounds
        x = min(max(pose[0], x0 + EDGE), x1 - EDGE)
        y = min(max(pose[1], y0 + EDGE), y1 - EDGE)
        try:
            world = edit.set_pose(self.world, name, (x, y, pose[2]))
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world)

    def set_goal(self, name: str, goal: tuple[float, float, float]) -> str | None:
        """Give a robot a goal."""
        if (why := self.guard()) is not None:
            return why
        try:
            world = edit.set_goal(self.world, name, goal)
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world, rebuild=True, select=name)

    def choose(self, name: str, field: str, text: str) -> str | None:
        """Change the model, planner, smoother or controller by its friendly name."""
        if (why := self.guard()) is not None:
            return why
        try:
            if field == "model":
                world = edit.set_model(self.world, name, text)
            else:
                world = edit.set_part(
                    self.world, name, field, PART_TABLES[field]()[text]
                )
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world)

    def set_settings(self, name: str, part: str, values: dict) -> str | None:
        """Set the settings of one part of a robot."""
        if (why := self.guard()) is not None:
            return why
        try:
            world = edit.set_settings(self.world, name, part, values)
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world)

    def set_period(self, name: str, seconds: float) -> str | None:
        """Set the seconds between runs of a robot's controller."""
        if (why := self.guard()) is not None:
            return why
        try:
            world = edit.set_control_period(self.world, name, seconds)
        except WorldError as error:
            return REFUSED.format(error)
        return self.apply(world)

    def toggle_play(self) -> None:
        """Start or pause the run."""
        self.playing = not self.playing

    def advance(self, seconds: float) -> None:
        """Take the run's news; if playing, advance it by real seconds at the speed.

        Time is counted in whole steps and the rest is carried to the next frame, so
        a fast frame rate does not make the run go faster than real time.
        """
        self.run.poll()
        if self.playing:
            self._owed += min(seconds, MAX_FRAME) * self.speed
            steps = int(self._owed / self.run.sim.dt)
            if steps:
                self._owed -= steps * self.run.sim.dt
                self.run.advance(steps * self.run.sim.dt)
        if self.run.time != self._seen:
            self._seen = self.run.time
            self.scene.record_trails()

    def step_once(self) -> None:
        """Advance by about one control period."""
        self.run.step(max(1, round(STEP_SECONDS / self.run.sim.dt)))
        self._seen = self.run.time
        self.scene.record_trails()

    def reset(self) -> None:
        """End the run, go back to time zero and unlock editing."""
        self.playing = False
        self._owed = 0.0
        self.run.reset()
        self.scene.set_run(self.run)
        self._seen = 0.0

    def close(self) -> None:
        """Stop the simulation process, if there is one."""
        self.run.close()
