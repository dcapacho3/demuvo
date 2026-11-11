"""The sandbox - the program the window shows.

A scene list on the left, the world in the middle with a playback pill over it, the
inspector on the right, a tool bar above and a status bar below. In the sandbox mode
you spawn robots, choose their parts, set goals and run; in the benchmark mode the
world is fixed and only Reset and the layers can change.
"""

from __future__ import annotations

import os
import time

from . import look
from .inputs import Combo, MenuItem, PromptPopup, Row, TreeList
from .inspector import Inspector
from .keys import KEY
from .paneactions import PaneActions
from .panels import Badge, FormRow, Pill
from .scenedraw import LAYERS, STATIC
from .scenepane import ScenePane
from .session import Session
from .shell import MENU_H, TOOL_H, Shell
from .widgets import Checkbox, IconButton, Section, Slider

DEFAULT_MESSAGE = "Drag a robot to move it. Ctrl+1 and Ctrl+2 show or hide the panels."
GOAL_MESSAGE = "Click on the floor to set the goal of the selected robot."
# The tools, and what the status bar says when one is chosen.
TOOL_MESSAGES = {
    "select": DEFAULT_MESSAGE,
    "goal": GOAL_MESSAGE,
    "box": "Drag on the floor from one corner to the other to draw a box.",
    "circle": "Drag on the floor from the centre outwards to draw a circle.",
}
BADGES = {
    "bench": ("BENCHMARK · fixed world", "bench"),
    "edit": ("SANDBOX · editing", "edit"),
    "run": ("SANDBOX · running", "run"),
}
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
        self.tree = TreeList(on_select=self.select)
        self.layer_boxes: dict[str, Checkbox] = {}
        self.layer_on = {name: on for name, (_, on) in LAYERS.items()}
        # a few robots show their rays; a swarm would hide the floor under them
        self.layer_on["rays"] = len(session.world.robots) <= RAYS_UP_TO
        self.inspector: Inspector | None = None
        self.pane: ScenePane | None = None
        self._last_tick = time.perf_counter()
        self.alpha: Slider | None = None
        self.palette: Combo | None = None
        self.scene_heading = Section("Scene")

    # -- building --
    def build(self, shell: Shell) -> None:
        """Make the tool bar, the lists, the inspector and the pane."""
        self.shell = shell
        chrome = shell.layout.chrome
        self._toolbar(chrome.toolbar)
        chrome.left.add(self.scene_heading, self.tree)
        pill = Pill(self.toggle_play, self.step_once, self.reset_run, self._speed)
        self.pane = ScenePane(self.session, pill, PaneActions(self))
        shell.layout.pane = self.pane
        for name, (label, _) in LAYERS.items():
            self.layer_boxes[name] = Checkbox(
                label,
                self.layer_on[name],
                on_change=lambda on, n=name: self.set_layer(n, on),
            )
        self.alpha = Slider(1.0, 0.1, 1.0, on_change=self._set_opacity)
        self.palette = Combo(
            list(look.PALETTES), look.PALETTES[0], on_change=self._set_palette
        )
        footer = [Section("Layers"), *self.layer_boxes.values()]
        footer += [FormRow("opacity", self.alpha), FormRow("palette", self.palette)]
        self.inspector = Inspector(
            chrome.right, self.session, self.message, self.refresh, footer
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
            ("add", "robot_add", self._add_robot, "Add robot (Ctrl+N)", False, None),
            ("delete", "trash", self._delete_selected, "Delete (Del)", False, None),
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
        wide = self.badge.wanted_width()
        self.badge.set_rect(width - 40 - wide - 8, MENU_H + 10, wide, 24)
        self.scene_heading.set_rect(12, MENU_H + TOOL_H + 10, 224, 24)
        self.tree.set_rect(
            12, MENU_H + TOOL_H + 38, 224, max(32 * len(self.tree.rows), 10)
        )

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
            return [
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
                    enabled=ok,
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
            (KEY["n"], True): self._add_robot,
            (KEY["o"], True): self._open_prompt,
            (KEY["s"], True): self._save,
            (KEY["0"], True): self.shell.layout.apply,
            (KEY["delete"], False): self._delete_selected,
            (KEY["space"], False): self.toggle_play,
            (KEY["v"], False): lambda: self.set_tool("select"),
            (KEY["g"], False): lambda: self.set_tool("goal"),
            (KEY["b"], False): lambda: self.set_tool("box"),
            (KEY["c"], False): lambda: self.set_tool("circle"),
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
        elapsed, self._last_tick = now - self._last_tick, now
        if self.session.playing:
            self.session.advance(elapsed)
            self._after_run()

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
        for k, obstacle in enumerate(session.world.obstacles, start=1):
            label = f"{obstacle.shape.name} {k}"
            rows.append((f"{OBSTACLE_KEY}{k - 1}", label, look.NAVY2, 1))
        self.tree.set_rows([Row(*row) for row in rows])
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
        for key in ("add", "delete", "goal", "box", "circle"):
            self.buttons[key].enabled = ok
        if not ok and session.tool != "select":
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
        """Choose a tool, and say what it does."""
        self.session.set_tool(tool)
        for key in TOOL_MESSAGES:
            self.buttons[key].checked = tool == key
        self.shell.layout.chrome.status.show(TOOL_MESSAGES[tool], 6)
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

    # -- files --
    def _prompt(self, title: str, text: str, hint: str, action) -> None:
        """Ask for a line of text in a card near the top of the window."""
        width = min(560.0, self.shell.screen.width - 40)
        x = (self.shell.screen.width - width) / 2
        self.shell.open_popup(
            PromptPopup(
                title, text, action, (x, MENU_H + TOOL_H + 20, width), hint=hint
            )
        )

    def _open_prompt(self) -> None:
        """Ask which world file to open."""
        if self.session.mode == "sandbox":
            self._prompt(
                "Open a world file",
                self.session.edits.path or "",
                "Type the path of a .toml world file, then press Enter. Esc cancels.",
                self._open,
            )

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
        self._prompt(
            "Save the world as",
            self.session.edits.path or "world.toml",
            "Type a path ending in .toml, then press Enter. An old file is replaced.",
            self._save_to,
        )

    def _save_to(self, path: str) -> None:
        """Write the world to a path and say so."""
        path = os.path.expanduser(path.strip())
        error = self.session.edits.save(path)
        self.message(error or f"Saved {path}.")

    def _quit(self) -> None:
        """Ask the window to close."""
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
        self._last_tick = time.perf_counter()
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
