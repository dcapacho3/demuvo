"""The simulation in a process of its own, so drawing never waits for a step.

A `RemoteRun` has the calls of a `Run` that the window needs, but the simulation
runs in another process. After each batch of steps the worker sends a snapshot: the
time, the poses, the paths, the lidar scans and the local costmaps. The window draws
the latest snapshot while the worker computes the next one, so a frame costs the
longer of the two and not their sum. The results are the same as in one process: it is
the same code on the same world.

Commands that change the run (load, reset, step) wait for the worker's answer;
`advance` does not, and asks for more only when the worker is idle.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field

from ..costmap import build_costmaps
from ..runner import Beams, Run
from ..world import World, load_world
from .logbuf import LogBuffer, capture

# The most simulated seconds sent to the worker at once; more owed time is dropped, so a
# slow worker falls behind real time and does not spiral.
MAX_REQUEST = 0.5


def snapshot(run: Run) -> dict:
    """What the window reads from a run, as plain data that can cross a pipe."""
    names = list(run.poses)
    return {
        "time": run.time,
        "poses": run.poses,
        "paths": {name: run.path(name) for name in names},
        "scans": {name: run.scans(name) for name in names},
        "local": {name: run.local_costmap(name) for name in names},
    }


def serve(connection) -> None:
    """The worker: obey commands from the window until told to stop.

    Each command is a tuple and has one reply, a snapshot, except `("result",)`,
    which replies with the result lines, and `("stop",)`, which replies with nothing.
    Warnings and log lines the run gives are not printed here; they go in the reply.
    """
    buffer = LogBuffer()
    release = capture(buffer)
    try:
        _obey(connection, buffer)
    finally:
        release()


def _obey(connection, buffer: LogBuffer) -> None:
    """Take commands and answer them; the messages the run gave go with the reply."""
    run: Run | None = None
    while True:
        command, *arguments = connection.recv()
        if command == "stop":
            return
        if command == "load":
            run = Run(*arguments)
        elif command == "reset":
            run.reset()
        elif command == "set_world":
            run.set_world(*arguments)
        elif command == "step":
            run.step(*arguments)
        elif command == "advance":
            run.advance(*arguments)
        elif command == "until_done":
            run.until_done(*arguments)
        if command == "result":
            connection.send((run.result(), run.finished))
        else:
            reply = snapshot(run)
            reply["log"] = buffer.drain()
            connection.send(reply)


@dataclass
class _Sim:
    """The few things the window reads from `Run.sim`."""

    dt: float
    costmaps: dict = field(default_factory=dict)


class RemoteRun:
    """A run whose simulation is in another process.

    Args:
        source: a world file or a `World`.
        dt: the step length in seconds; None takes the world's.
        connection: the pipe to a worker that is running `serve`; None starts a
            worker process.

    Attributes:
        world: the world being run.
        sim: its step length and its global costmaps, which the window reads.
    """

    def __init__(self, source: str | World, dt: float | None = None, connection=None):
        """Start or join a worker and load the world."""
        self._process = None
        if connection is None:
            context = multiprocessing.get_context("spawn")
            connection, child = context.Pipe()
            self._process = context.Process(target=serve, args=(child,), daemon=True)
            self._process.start()
            child.close()
        self._connection = connection
        self._dt = dt
        self._waiting = False
        self._owed = 0.0
        self._snap: dict = {}
        self._news: list[tuple[float, str, str]] = []
        self.world = load_world(source) if isinstance(source, str) else source
        self.sim = _Sim(self._dt or self.world.dt)
        self._load()

    def _load(self) -> None:
        """Send the world to the worker and take its first snapshot."""
        self.sim.dt = self._dt or self.world.dt
        self.sim.costmaps = build_costmaps(self.world)
        self._ask("load", self.world, self._dt)

    # -- talking to the worker --
    def _receive(self):
        """Wait for the reply to the command in flight."""
        reply = self._connection.recv()
        self._waiting = False
        if isinstance(reply, dict):
            self._news.extend(reply.pop("log", []))
        return reply

    def take_log(self) -> list[tuple[float, str, str]]:
        """The messages sent since the last call, as (when, level, text)."""
        news, self._news = self._news, []
        return news

    def _ask(self, *command):
        """Send a command, wait for its reply and keep the snapshot."""
        if self._waiting:
            self._receive()
        self._connection.send(command)
        self._waiting = True
        reply = self._receive()
        if command[0] != "result":
            self._snap = reply
        return reply

    def poll(self) -> bool:
        """Take the worker's answer to `advance` if it has come; True if it did."""
        if self._waiting and self._connection.poll():
            self._snap = self._receive()
            return True
        return False

    # -- what the window reads --
    @property
    def time(self) -> float:
        """Simulated time of the latest snapshot."""
        return self._snap["time"]

    @property
    def poses(self) -> dict:
        """The true pose of each robot in the latest snapshot."""
        return self._snap["poses"]

    def scans(self, name: str) -> list[Beams]:
        """The latest scans of a robot."""
        return self._snap["scans"][name]

    def local_costmap(self, name: str):
        """The local costmap of a robot, or None."""
        return self._snap["local"][name]

    def path(self, name: str):
        """The path a robot follows, or None."""
        return self._snap["paths"][name]

    # -- what the window does --
    def reset(self) -> None:
        """Start again from the world as it is."""
        self._owed = 0.0
        self._ask("reset")

    def set_world(self, world: World) -> None:
        """Run another world from time zero."""
        self.world, self._owed = world, 0.0
        self._ask("set_world", world)
        self.sim.dt = self._dt or world.dt
        self.sim.costmaps = build_costmaps(world)

    def step(self, count: int = 1) -> None:
        """Advance by `count` steps and wait for the answer."""
        self._ask("step", count)

    def advance(self, seconds: float, budget: float | None = None) -> int:
        """Ask for `seconds` of simulated time without waiting.

        If the worker is still busy the time is owed and sent when it is idle.
        `budget` is accepted so a `Run` and a `RemoteRun` are called alike; the
        worker does not stop early.
        """
        del budget  # the worker does not stop early
        self._owed += seconds
        self.poll()
        if not self._waiting and self._owed > 0:
            send, self._owed = min(self._owed, MAX_REQUEST), 0.0
            self._connection.send(("advance", send))
            self._waiting = True
        return 0

    def until_done(self, limit: float) -> bool:
        """Step until every task is done or `limit` seconds pass."""
        self._ask("until_done", limit)
        return self.result_state()[1]

    def result_state(self) -> tuple[list[str], bool]:
        """The result lines so far and whether every task is done."""
        return self._ask("result")

    def result(self) -> list[str]:
        """What each robot did so far; see `Run.result`."""
        return self.result_state()[0]

    @property
    def finished(self) -> bool:
        """True once every robot that has tasks has done its last one."""
        return self.result_state()[1]

    def close(self) -> None:
        """Stop the worker."""
        if self._waiting:
            self._receive()
        self._connection.send(("stop",))
        if self._process is not None:
            self._process.join(timeout=5)
