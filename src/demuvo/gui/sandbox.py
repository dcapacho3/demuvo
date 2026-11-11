"""The sandbox - the program the window shows.

A scene list on the left, the world in the middle with a playback pill over it, the
inspector on the right, a tool bar above and a status bar below. In the sandbox mode
you spawn robots, choose their parts, set goals and run; in the benchmark mode the
world is fixed and only Reset and the layers can change.
"""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path

from .. import edit
from . import look
from .clock import Clock
from .filedialog import Browse, FileDialog
from .inputs import Combo, MenuItem, NameBox, Row, TreeList
from .inspector import Inspector
from .keys import KEY
from .logview import LogView
from .paneactions import PaneActions
from .panels import Badge, FormRow, Pill
from .scenedraw import LAYERS, STATIC
from .scenepane import ScenePane
from .session import Session
from .shell import MENU_H, TOOL_H, Shell
from .widgets import Checkbox, IconButton, Section, Slider

DEFAULT_MESSAGE = "Drag a robot to move it. Ctrl+1 and Ctrl+2 show or hide the panels."
GOAL_MESSAGE = (
    "Click the floor to set the goal of the selected robot; Shift adds a waypoint."
)
# The tools, and what the status bar says when one is chosen.
TOOL_MESSAGES = {
    "select": DEFAULT_MESSAGE,
    "goal": GOAL_MESSAGE,
    "box": "Drag on the floor from one corner to the other to draw a box.",
    "circle": "Drag on the floor from the centre outwards to draw a circle.",
    "polygon": "Click the corners, then the first one or Enter to close. Esc cancels.",
    "line": "Click the points of the line. Press Enter to finish; Esc cancels.",
}
BADGES = {
    "bench": ("BENCHMARK · fixed world", "bench"),
    "edit": ("SANDBOX · editing", "edit"),
    "run": ("SANDBOX · running", "run"),
}
# The floor is part of how demuvo looks, so it has no switch.
ALWAYS_ON = ("floor",)
# The real-time figure shows as slow below this many per cent.
SLOW_BELOW = 90
# Layers whose opacity the slider sets.
FADING = ("costmap", "local")
BUTTON = 32
# Worlds with up to this many robots start with the lidar rays shown.
RAYS_UP_TO = 10
# The key of an obstacle row in the scene list, followed by its 0-based place.
OBSTACLE_KEY = "obstacle:"


class Sandbox:
    """The program: it builds the widgets and turns what the user does into edits.

    Args:
        session: the world, its run and the mode.

    Attributes:
        session: the session.
        menu_titles: the titles of the menu bar.
        groups: the radio groups of tool buttons.
        shell: the shell that shows it; set when the shell is made.
    """

    menu_titles = ("File", "Edit", "View")

    def __init__(self, session: Session) -> None:
        """Make the program; `build` makes the widgets when the shell is made."""
        self.session = session
        self.groups: dict[str, list[IconButton]] = {}
        self.shell: Shell | None = None
        self.buttons: dict[str, IconButton] = {}
        self.badge = Badge()
        self.log_view: LogView | None = None
        self.tree = TreeList(on_select=self.select, on_rename=self._rename)
        self.layer_boxes: dict[str, Checkbox] = {}
        self.layer_on = {name: on for name, (_, on) in LAYERS.items()}
        # a few robots show their rays; a swarm would hide the floor under them
        self.layer_on["rays"] = len(session.world.robots) <= RAYS_UP_TO
        self.inspector: Inspector | None = None
        self.pane: ScenePane | None = None
        self.clock = Clock()
        self.alpha: Slider | None = None
        self.palette: Combo | None = None
        self._warned: str | None = None

    # -- building --
    def build(self, shell: Shell) -> None:
        """Make the tool bar, the lists, the inspector and the pane."""
        self.shell = shell
        chrome = shell.layout.chrome
        self.log_view = LogView(self.session.log, self._clear_log)
        chrome.bottom.add(self.log_view)
        self._toolbar(chrome.toolbar)
        pill = Pill(self.toggle_play, self.step_once, self.reset_run, self._speed)
        self.pane = ScenePane(self.session, pill, PaneActions(self))
        shell.layout.pane = self.pane
        for name, (label, _) in LAYERS.items():
            if name in ALWAYS_ON:
                continue
            self.layer_boxes[name] = Checkbox(
                label,
                self.layer_on[name],
                on_change=lambda on, n=name: self.set_layer(n, on),
            )
        self.alpha = Slider(1.0, 0.1, 1.0, on_change=self._set_opacity)
        self.palette = Combo(
            list(look.PALETTES), look.PALETTES[0], on_change=self._set_palette
        )
        layers = [Section("Layers"), *self.layer_boxes.values()]
        layers += [FormRow("opacity", self.alpha), FormRow("palette", self.palette)]
        chrome.left.add(Section("Scene"), self.tree, *layers)
        self.inspector = Inspector(
            chrome.right, self.session, self.message, self.refresh
        )
        chrome.status.default = chrome.status.message = DEFAULT_MESSAGE
        for name, on in self.layer_on.items():
            self.session.scene.layers.shown[name] = 1.0 if on else 0.0
        self._refresh_lists()
        self.inspector.show(None)
        self._refresh_lock()

    def _toolbar(self, toolbar) -> None:
        """The tool bar's buttons and badge."""
        specs = (
            (
                "left",
                "panel_left",
                lambda on: self.set_panel("left", on),
                "Scene panel (Ctrl+1)",
                True,
                None,
            ),
            (
                "select",
                "select",
                lambda on: self.set_tool("select"),
                "Select (V)",
                True,
                "tool",
            ),
            (
                "goal",
                "goal",
                lambda on: self.set_tool("goal"),
                "Set goal (G)",
                True,
                "tool",
            ),
            (
                "box",
                "box",
                lambda on: self.set_tool("box"),
                "Draw a box (B)",
                True,
                "tool",
            ),
            (
                "circle",
                "circle",
                lambda on: self.set_tool("circle"),
                "Draw a circle (C)",
                True,
                "tool",
            ),
            (
                "polygon",
                "polygon",
                lambda on: self.set_tool("polygon"),
                "Draw a polygon (P)",
                True,
                "tool",
            ),
            (
                "line",
                "line",
                lambda on: self.set_tool("line"),
                "Draw a line (L)",
                True,
                "tool",
            ),
            ("add", "robot_add", self._add_robot, "Add robot (Ctrl+N)", False, None),
            ("delete", "trash", self._delete_selected, "Delete (Del)", False, None),
            (
                "bottom",
                "panel_bottom",
                lambda on: self.set_panel("bottom", on),
                "Log (Ctrl+3)",
                True,
                None,
            ),
            (
                "right",
                "panel_right",
                lambda on: self.set_panel("right", on),
                "Inspector (Ctrl+2)",
                True,
                None,
            ),
        )
        for key, icon, action, tip, checkable, group in specs:
            button = IconButton(
                icon, action, tooltip=tip, checkable=checkable, group=group
            )
            button.checked = key in ("left", "select", "right")
            self.buttons[key] = button
        toolbar.add(*self.buttons.values(), self.badge)
        self.groups["tool"] = [self.buttons[key] for key in TOOL_MESSAGES]

    def layout_toolbar(self, width: float) -> None:
        """Place the tool bar's buttons and badge for a window width."""
        top = MENU_H + 6
        self.buttons["left"].set_rect(8, top, BUTTON, BUTTON)
        x = 8 + 36 + 6
        for key in (*TOOL_MESSAGES, "add", "delete"):
            self.buttons[key].set_rect(x, top, BUTTON, BUTTON)
            x += 34
        self.buttons["right"].set_rect(width - 40, top, BUTTON, BUTTON)
        self.buttons["bottom"].set_rect(width - 76, top, BUTTON, BUTTON)
        wide = self.badge.wanted_width()
        self.badge.set_rect(width - 76 - wide - 8, MENU_H + 10, wide, 24)
        bottom = self.shell.layout.chrome.bottom
        self.log_view.set_rect(bottom.x, bottom.y, bottom.width, bottom.height)

    # -- what the shell asks --
    def menu_items(self, title: str) -> list[MenuItem]:
        """The lines of a menu."""
        ok = self.session.editable
        if title == "File":
            fixed = self.session.mode == "benchmark"
            return [
                MenuItem("New world", enabled=not fixed, action=self._new_world),
                MenuItem(
                    "Open…",
                    shortcut="Ctrl+O",
                    enabled=not fixed,
                    action=self._open_prompt,
                ),
                MenuItem("Save", shortcut="Ctrl+S", action=self._save),
                MenuItem("Save as…", action=self._save_prompt),
                MenuItem(separator=True),
                MenuItem("Quit", shortcut="Ctrl+Q", action=self._quit),
            ]
        if title == "Edit":
            tool = self.session.tool
            history = self.session.history
            return [
                MenuItem(
                    "Undo",
                    shortcut="Ctrl+Z",
                    enabled=ok and history.can_undo,
                    action=self._undo,
                ),
                MenuItem(
                    "Redo",
                    shortcut="Ctrl+Y",
                    enabled=ok and history.can_redo,
                    action=self._redo,
                ),
                MenuItem(separator=True),
                MenuItem(
                    "Add robot", shortcut="Ctrl+N", enabled=ok, action=self._add_robot
                ),
                MenuItem(
                    "Delete selected",
                    shortcut="Del",
                    enabled=ok,
                    action=self._delete_selected,
                ),
                MenuItem(
                    "Set goal",
                    shortcut="G",
                    checked=tool == "goal",
                    enabled=ok and self.session.scene.selected is not None,
                    action=lambda: self.set_tool("goal"),
                ),
                MenuItem(
                    "Draw box",
                    shortcut="B",
                    checked=tool == "box",
                    enabled=ok,
                    action=lambda: self.set_tool("box"),
                ),
                MenuItem(
                    "Draw circle",
                    shortcut="C",
                    checked=tool == "circle",
                    enabled=ok,
                    action=lambda: self.set_tool("circle"),
                ),
                MenuItem(
                    "Draw polygon",
                    shortcut="P",
                    checked=tool == "polygon",
                    enabled=ok,
                    action=lambda: self.set_tool("polygon"),
                ),
                MenuItem(
                    "Draw line",
                    shortcut="L",
                    checked=tool == "line",
                    enabled=ok,
                    action=lambda: self.set_tool("line"),
                ),
                MenuItem(
                    "Select",
                    shortcut="V",
                    checked=tool == "select",
                    action=lambda: self.set_tool("select"),
                ),
                MenuItem(separator=True),
                MenuItem("Reset run", shortcut="R", action=self.reset_run),
            ]
        docks = self.shell.layout.docks
        layers = [
            MenuItem(
                label,
                checked=self.layer_on[name],
                action=lambda n=name: self.set_layer(n, not self.layer_on[n]),
            )
            for name, (label, _) in LAYERS.items()
            if name not in ALWAYS_ON
        ]
        return [
            MenuItem(
                "Scene panel",
                shortcut="Ctrl+1",
                checked=docks["left"].open,
                action=lambda: self.set_panel("left", not docks["left"].open),
            ),
            MenuItem(
                "Inspector",
                shortcut="Ctrl+2",
                checked=docks["right"].open,
                action=lambda: self.set_panel("right", not docks["right"].open),
            ),
            MenuItem(
                "Log",
                shortcut="Ctrl+3",
                checked=docks["bottom"].open,
                action=lambda: self.set_panel("bottom", not docks["bottom"].open),
            ),
            MenuItem(separator=True),
            *layers,
            MenuItem(separator=True),
            MenuItem(
                "Fit to window", shortcut="Ctrl+0", action=self.shell.layout.apply
            ),
        ]

    def command(self, key: int, ctrl: bool) -> bool:
        """A key no widget used; True if it was one of the program's shortcuts."""
        actions = {
            (KEY["1"], True): lambda: self._toggle_panel("left"),
            (KEY["2"], True): lambda: self._toggle_panel("right"),
            (KEY["3"], True): lambda: self._toggle_panel("bottom"),
            (KEY["n"], True): self._add_robot,
            (KEY["f2"], False): self._rename_selected,
            (KEY["z"], True): self._undo,
            (KEY["y"], True): self._redo,
            (KEY["o"], True): self._open_prompt,
            (KEY["s"], True): self._save,
            (KEY["0"], True): self.shell.layout.apply,
            (KEY["delete"], False): self._delete_selected,
            (KEY["space"], False): self.toggle_play,
            (KEY["v"], False): lambda: self.set_tool("select"),
            (KEY["g"], False): lambda: self.set_tool("goal"),
            (KEY["b"], False): lambda: self.set_tool("box"),
            (KEY["c"], False): lambda: self.set_tool("circle"),
            (KEY["p"], False): lambda: self.set_tool("polygon"),
            (KEY["l"], False): lambda: self.set_tool("line"),
            (KEY["enter"], False): self.pane.finish_outline,
            (KEY["escape"], False): self.pane.cancel_outline,
            (KEY["r"], False): self.reset_run,
            (KEY["s"], False): self.step_once,
        }
        action = actions.get((key, ctrl))
        if action:
            action()
        return action is not None

    def world_at(self, x: float, y: float) -> tuple[float, float] | None:
        """The world place under a pixel, or None if it is not over the scene."""
        px, py, width, height = self.pane.rect()
        if px <= x < px + width and py <= y < py + height:
            return self.session.scene.view.to_world(x, y)
        return None

    def busy(self) -> bool:
        """True while the run plays."""
        return self.session.playing

    def after_tick(self, now: float) -> None:
        """Advance a playing run by the real time since the last tick."""
        elapsed = self.clock.elapsed(now)
        if self.session.playing:
            self.session.advance(elapsed)
            self._after_run()
        self._measure_rate(now)
        self._watch_log()

    def _watch_log(self) -> None:
        """Redraw the log when messages came, and open it the first time one does."""
        news = self.log_view.news()
        if news == "open":
            self.set_panel("bottom", True)
        status = self.shell.layout.chrome.status
        count = len(self.session.log.entries)
        notes = (
            f"{count} message{'' if count == 1 else 's'} in the log" if count else ""
        )
        if news is not None and notes != status.notes:
            status.notes = notes
            status.invalidate()

    def _clear_log(self) -> None:
        """Forget the messages."""
        self.session.log.clear()
        self._watch_log()

    def _measure_rate(self, now: float) -> None:
        """Show how much of real time a run keeps up with, as the speed asked for.

        100% means no lag. It is worked out every half second while playing.
        """
        status = self.shell.layout.chrome.status
        if not self.session.playing:
            self.clock.restart()
            if status.rate:
                status.rate, status.slow = "", False
                status.invalidate()
            return
        share = self.clock.share(now, self.session.run.time, self.session.speed)
        if share is not None:
            status.rate, status.slow = f"real time {share:.0f}%", share < SLOW_BELOW
            status.invalidate()

    # -- feedback --
    def message(self, text: str) -> None:
        """Show a message in the status bar."""
        self.shell.layout.chrome.status.show(text)

    def scene_changed(self) -> None:
        """Redraw the whole scene."""
        x, y, width, height = self.pane.rect()
        self.shell.screen.damage.add((int(x), int(y), int(x + width), int(y + height)))

    def _refresh_lists(self) -> None:
        """Rebuild the scene list and the robot count."""
        session = self.session
        robots = session.world.robots
        rows = [("", f"Robots ({len(robots)})", None, 0)]
        rows += [(r.name, r.name, session.scene.colors[r.name], 1) for r in robots]
        rows += [("", "Obstacles", None, 0)]
        for k in range(1, len(session.world.obstacles) + 1):
            label = edit.obstacle_label(session.world, k - 1)
            rows.append((f"{OBSTACLE_KEY}{k - 1}", label, look.NAVY2, 1))
        self.tree.set_rows([Row(*row) for row in rows])
        self.tree.preferred_height = float(max(32 * len(rows), 10))
        place = session.scene.selected_obstacle
        self.tree.selected = (
            session.scene.selected if place is None else f"{OBSTACLE_KEY}{place}"
        )
        status = self.shell.layout.chrome.status
        status.count = f"{len(robots)} robot" + ("" if len(robots) == 1 else "s")
        self.shell.layout.apply()

    def _refresh_lock(self) -> None:
        """Enable what the mode allows, show the badge, and tell the inspector."""
        session = self.session
        ok = session.editable
        for key in ("add", "delete", "box", "circle", "polygon", "line"):
            self.buttons[key].enabled = ok
        self.buttons["goal"].enabled = ok and session.scene.selected is not None
        if session.tool != "select" and not self.buttons[session.tool].enabled:
            self.set_tool("select")
        self.badge.text, self.badge.kind = BADGES[session.state]
        self.inspector.lock()
        self.shell.layout.apply()

    def refresh(self) -> None:
        """Everything after an edit that worked: lists, inspector, lock, scene."""
        self._refresh_lists()
        self.inspector.show(self.session.scene.selected)
        self._refresh_lock()
        self.pane.layout()
        self.scene_changed()
        self.shell.invalidate_all()

    # -- panels, tools, layers --
    def set_panel(self, which: str, on: bool) -> None:
        """Slide a panel open or shut, and show it on its button."""
        self.buttons[which].checked = on
        self.shell.layout.slide(which, on)

    def _toggle_panel(self, which: str) -> None:
        """Slide a panel the other way."""
        self.set_panel(which, not self.shell.layout.docks[which].open)

    def set_tool(self, tool: str) -> None:
        """Choose a tool, and say what it does; the goal tool needs a robot picked."""
        refused = tool == "goal" and self.session.scene.selected is None
        if refused:
            tool = "select"
        self.session.set_tool(tool)
        self.pane.cancel_outline()
        for key in TOOL_MESSAGES:
            self.buttons[key].checked = tool == key
        message = (
            "Select a robot first, then pick its goal."
            if refused
            else TOOL_MESSAGES[tool]
        )
        self.shell.layout.chrome.status.show(message, 6)
        self.shell.invalidate_all()

    def set_layer(self, name: str, on: bool) -> None:
        """Show or hide a layer; one that is drawn every step fades, the rest switch.

        A cached layer would be drawn again in full on every frame of a fade, which
        is slower than the fade is long, so it switches at once.
        """
        self.layer_on[name] = on
        self.layer_boxes[name].checked = on
        layers = self.session.scene.layers
        if name in STATIC:
            layers.shown[name] = 1.0 if on else 0.0
            self.scene_changed()
            return
        self.shell.animator.start(
            f"layer_{name}",
            layers.shown[name],
            1.0 if on else 0.0,
            look.MS["fade"],
            lambda value, n=name: self._fade(n, value),
        )

    def _fade(self, name: str, value: float) -> None:
        """Set how far a layer is faded in."""
        self.session.scene.layers.shown[name] = value
        self.scene_changed()

    def _set_opacity(self, value: float) -> None:
        """The opacity of the costmap layers."""
        for name in FADING:
            self.session.scene.layers.opacity[name] = value
        self.scene_changed()

    def _set_palette(self, name: str) -> None:
        """The costmap palette."""
        self.session.scene.palette = name
        self.scene_changed()

    def _speed(self, speed: float) -> None:
        """The playback speed."""
        self.session.speed = speed
        self.clock.restart()

    # -- files --
    @property
    def title(self) -> str:
        """The window title: the file, a star if it has unsaved changes, the program."""
        edits = self.session.edits
        name = os.path.basename(edits.path) if edits.path else "untitled"
        return f"{name}{' *' if edits.modified else ''} - demuvo"

    def _discard_ok(self, what: str) -> bool:
        """True if nothing is unsaved, or this is the second ask in a row."""
        if not self.session.edits.modified or self._warned == what:
            self._warned = None
            return True
        self._warned = what
        self.message(
            f"Unsaved changes. Choose {what} again to drop them, or save first."
        )
        return False

    def _browse(self, title: str, name: str, action, *, save: bool) -> None:
        """Open the file dialog near the top of the window."""
        screen = self.shell.screen
        width = min(680.0, screen.width - 40)
        height = min(500.0, screen.height - MENU_H - TOOL_H - 40)
        place = ((screen.width - width) / 2, MENU_H + TOOL_H + 16, width, height)
        path = self.session.edits.path
        folder = Path(path).parent if path else Path.cwd()
        self.shell.open_popup(
            FileDialog(Browse(title, action, name, save), folder, place)
        )

    def _open_prompt(self) -> None:
        """Ask which world file to open."""
        if self.session.mode == "sandbox" and self._discard_ok("Open"):
            self._browse("Open a world", "", self._open, save=False)

    def _open(self, path: str) -> None:
        """Open the file at a path, and show the new world."""
        path = os.path.expanduser(path.strip())
        error = self.session.edits.open(path)
        self.pane.pill.set_playing(False)
        self.pane.pill.set_time(0.0)
        if error is not None:
            self.message(error)
            return
        self.refresh()
        self.message(f"Opened {path}.")

    def _new_world(self) -> None:
        """Start again with a blank world."""
        if not self._discard_ok("New world"):
            return
        error = self.session.edits.new()
        self.pane.pill.set_playing(False)
        self.pane.pill.set_time(0.0)
        self.edited(error)

    def _save(self) -> None:
        """Save to the file the world has, or ask for one."""
        path = self.session.edits.path
        if path:
            self._save_to(path)
        else:
            self._save_prompt()

    def _save_prompt(self) -> None:
        """Ask where to save the world."""
        path = self.session.edits.path
        name = os.path.basename(path) if path else "world.toml"
        self._browse("Save the world as", name, self._save_to, save=True)

    def _save_to(self, path: str) -> None:
        """Write the world to a path and say so."""
        path = os.path.expanduser(path.strip())
        error = self.session.edits.save(path)
        self.message(error or f"Saved {path}.")

    def can_quit(self) -> bool:
        """True if the window may close: nothing is unsaved, or it was asked twice."""
        return self._discard_ok("Quit")

    def _quit(self) -> None:
        """Ask the window to close, unless there are unsaved changes."""
        if self._discard_ok("Quit"):
            self.shell.quit_requested = True

    # -- editing: what the pane tells --
    def select(self, name: str | None) -> None:
        """Select a robot, an obstacle (key "obstacle:N") or none."""
        if name is not None and name.startswith(OBSTACLE_KEY):
            changed = self.session.select_obstacle(int(name[len(OBSTACLE_KEY) :]))
        else:
            changed = self.session.select(name)
        if changed:
            self.refresh()

    def _undo(self) -> None:
        """Take back the last edit."""
        self.edited(self.session.history.undo())

    def _redo(self) -> None:
        """Make the last undone edit again."""
        self.edited(self.session.history.redo())

    def _rename(self, key: str, rect) -> None:
        """Open a text box over a row of the scene list, to rename what it names."""
        if (why := self.session.guard()) is not None:
            self.message(why)
            return
        if key.startswith(OBSTACLE_KEY):
            index = int(key[len(OBSTACLE_KEY) :])
            current = self.session.world.obstacles[index].name
            submit = functools.partial(self._renamed, "obstacle", index)
        else:
            current = key
            submit = functools.partial(self._renamed, "robot", key)
        x, y, width, height = rect
        self.shell.open_popup(
            NameBox(current, submit, (x, y, max(width, 160.0), height))
        )

    def _renamed(self, kind: str, which, text: str) -> None:
        """Rename an obstacle (by its place) or a robot (by its name), and show it."""
        edits = self.session.edits
        if kind == "obstacle":
            self.edited(edits.rename_obstacle(which, text))
        else:
            self.edited(edits.rename_robot(which, text))

    def _rename_selected(self) -> None:
        """F2: rename the selected robot or obstacle."""
        place = self.session.scene.selected_obstacle
        key = self.session.scene.selected or (
            None if place is None else f"{OBSTACLE_KEY}{place}"
        )
        if key is not None:
            self._rename(key, self.tree.row_rect(key))

    def _add_robot(self) -> None:
        """Add a robot."""
        self.edited(self.session.add_robot())

    def _delete_selected(self) -> None:
        """Delete the selected obstacle, or else the selected robot."""
        if self.session.scene.selected_obstacle is not None:
            self.edited(self.session.edits.delete_obstacle())
        else:
            self.edited(self.session.delete_robot())

    def edited(self, error: str | None) -> None:
        """Show why an edit was refused, or refresh after one that worked."""
        if error is not None:
            self.message(error)
            self.scene_changed()
            return
        self.refresh()

    # -- running --
    def toggle_play(self) -> None:
        """Start or pause the run."""
        self.session.toggle_play()
        self.pane.pill.set_playing(self.session.playing)
        self.clock.last = time.perf_counter()
        self._refresh_lock()
        self.shell.invalidate_all()

    def step_once(self) -> None:
        """Advance about one control period."""
        self.session.step_once()
        self._after_run()

    def reset_run(self) -> None:
        """End the run and unlock editing."""
        self.session.reset()
        self.pane.pill.set_playing(False)
        self.pane.pill.set_time(0.0)
        self._refresh_lock()
        self.shell.invalidate_all()

    def _after_run(self) -> None:
        """Redraw what a step changed: robots, clock, and the lock at the first step."""
        session, pill = self.session, self.pane.pill
        changed = pill.set_time(session.run.time)
        if self.badge.kind == "edit" and session.run.time > 0:
            self._refresh_lock()
            self.scene_changed()
        else:
            self.shell.screen.damage.add(session.scene.dirty_rect())
        if changed:
            self.shell.screen.damage.add(
                (
                    int(pill.x),
                    int(pill.y),
                    int(pill.x + pill.width),
                    int(pill.y + pill.height),
                )
            )
