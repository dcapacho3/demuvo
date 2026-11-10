"""The shell - the frame of the window: panels, menus, focus, events and rendering.

It knows nothing about robots. It holds a menu bar, a tool bar, a left panel, a
scene pane, a right column and a status bar; routes the mouse and the keyboard to
the widgets; shows menus, tooltips and messages; animates the panels sliding; and
renders only the rectangle that changed. A program fills it in: it is an object
with these members.

    menu_titles: the titles of the menu bar.
    groups: the radio groups of tool buttons, by name.
    build(shell): add widgets to the panels and set `shell.layout.pane`.
    menu_items(title): the `MenuItem` lines of a menu.
    command(key, ctrl): a key no widget used; True if the program used it.
    layout_toolbar(width): place the tool bar's buttons.
    world_at(x, y): the world place under a pixel, or None.
    busy(): True while the program has something moving.
    after_tick(now): called on every tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import font, look
from .inputs import Popup
from .keys import KEY
from .raster import Canvas, rgba
from .widgets import Column, Event, IconButton, Panel, Widget, text_baseline

MENU_H, TOOL_H, STATUS_H = 34, 44, 26
LEFT_W, RIGHT_W = 248, 312
# The scene keeps at least this many pixels across; the panels give way.
SCENE_MIN = 240
# A tooltip shows after the pointer rests this long, in seconds.
TOOLTIP_DELAY = 0.5
# A drawn widget's dirty box grows by this many pixels each side, for shadows and rings.
DIRTY_PAD = 8
MESSAGE_SECONDS = 8.0


class Damage:
    """The rectangle of the window that needs a redraw.

    Attributes:
        dirty: True if anything needs a redraw.
        rect: (x0, y0, x1, y1) that holds everything that changed, or None.
    """

    def __init__(self, width: int, height: int) -> None:
        """Start with the whole window needing a redraw."""
        self.size = (width, height)
        self.dirty = True
        self.rect: tuple[int, int, int, int] | None = (0, 0, width, height)

    def add(self, rect: tuple[int, int, int, int]) -> None:
        """Include a box in what must be redrawn."""
        self.dirty = True
        old = self.rect
        self.rect = (
            rect
            if old is None
            else (
                min(old[0], rect[0]),
                min(old[1], rect[1]),
                max(old[2], rect[2]),
                max(old[3], rect[3]),
            )
        )

    def everything(self) -> None:
        """Redraw the whole window."""
        self.dirty = True
        self.rect = (0, 0, *self.size)

    def widget(self, widget: Widget) -> None:
        """Redraw a widget and a margin round it."""
        pad = DIRTY_PAD
        self.add(
            (
                int(widget.x) - pad,
                int(widget.y) - pad,
                int(widget.x + widget.width) + pad,
                int(widget.y + widget.height) + pad,
            )
        )

    def take(self) -> tuple[int, int, int, int]:
        """The box to redraw, inside the window; the damage is then cleared."""
        width, height = self.size
        x0, y0, x1, y1 = self.rect or (0, 0, width, height)
        box = (max(x0, 0), max(y0, 0), min(x1, width), min(y1, height))
        self.dirty, self.rect = False, None
        return box


@dataclass
class _Motion:
    """One running animation."""

    start: float
    end: float
    began: float
    ms: float
    apply: object


class Animator:
    """Eases numbers from one value to another over some milliseconds.

    A new animation under the same key replaces the running one and starts from
    where that stood, so nothing jumps.
    """

    def __init__(self) -> None:
        """Start with nothing running."""
        self._running: dict[str, _Motion] = {}

    def start(self, key: str, start: float, end: float, ms: float, apply) -> None:
        """Ease from `start` to `end`, calling `apply(value)` on each tick."""
        self._running[key] = _Motion(start, end, time.perf_counter(), ms, apply)

    def tick(self, now: float) -> None:
        """Move every animation to the time `now`; a finished one is dropped."""
        for key in list(self._running):
            motion = self._running[key]
            done = min((now - motion.began) * 1000 / motion.ms, 1.0)
            span = motion.end - motion.start
            motion.apply(motion.start + span * look.ease_out(done))
            if done >= 1.0:
                motion.apply(motion.end)
                self._running.pop(key, None)

    def busy(self) -> bool:
        """True while any animation runs."""
        return bool(self._running)


class Surface:
    """The picture the window shows: a float canvas and its uint8 copy."""

    def __init__(self, width: int, height: int) -> None:
        """Make a canvas and a frame of this size."""
        self.canvas = Canvas(width, height, look.WINDOW)
        self.frame = np.zeros((height, width, 3), np.uint8)

    def copy(self, box: tuple[int, int, int, int]) -> None:
        """Convert a box of the canvas to uint8 into the frame."""
        x0, y0, x1, y1 = box
        part = np.clip(self.canvas.px[y0:y1, x0:x1], 0.0, 1.0)
        self.frame[y0:y1, x0:x1] = (part * 255.0 + 0.5).astype(np.uint8)


@dataclass
class Pointer:
    """Where the pointer is and what is under it.

    Attributes:
        x: pointer x in pixels.
        y: pointer y in pixels.
        hovered: the widget under the pointer.
        captured: the widget that was pressed, which gets the moves until release.
        tip: (text, x, y) of the tooltip showing, or None.
        since: when the pointer last changed widget.
    """

    x: float = 0.0
    y: float = 0.0
    hovered: Widget | None = None
    captured: Widget | None = None
    tip: tuple[str, float, float] | None = None
    since: float = 0.0


class MenuBar(Panel):
    """The strip of menu titles at the top.

    Attributes:
        titles: the names, left to right.
        places: each title with its x and width.
        open_title: the title whose menu is open, or None.
        hover_title: the title under the pointer, or None.
    """

    def __init__(self, titles: tuple[str, ...]) -> None:
        """Make a bar of titles."""
        super().__init__(look.WINDOW)
        self.titles = titles
        self.places: list[tuple[str, float, float]] = []
        self.open_title: str | None = None
        self.hover_title: str | None = None

    def layout(self) -> None:
        """Work out where each title sits."""
        x, self.places = 14.0, []
        for title in self.titles:
            width = font.regular().width(title, 13) + 24
            self.places.append((title, x - 12, width))
            x += width

    def title_at(self, x: float, y: float) -> str | None:
        """The title under a point, or None."""
        if y >= MENU_H:
            return None
        for title, left, width in self.places:
            if left <= x < left + width:
                return title
        return None

    def draw(self, canvas: Canvas) -> None:
        """Draw the bar, and each title lit if it is open or under the pointer."""
        super().draw(canvas)
        for title, left, width in self.places:
            if title in (self.open_title, self.hover_title):
                canvas.rect((left, 4, width, MENU_H - 8), rgba(look.MINT2), radius=4)
            baseline = text_baseline(0, MENU_H, 13)
            canvas.text(
                font.regular(),
                title,
                (left + 12, baseline),
                size=13,
                color=rgba(look.NAVY),
            )


class StatusBar(Panel):
    """The strip at the bottom: a message, the pointer's place in the world, a count.

    Attributes:
        message: the text on the left.
        until: when a temporary message ends, or 0.
        default: the text to return to.
        coords: the pointer's world place as (x, y), or None.
        count: the text at the right, such as "2 robots".
    """

    def __init__(self) -> None:
        """Make an empty bar."""
        super().__init__(look.PANEL, edge="top")
        self.message = self.default = ""
        self.until = 0.0
        self.coords: tuple[float, float] | None = None
        self.count = ""

    def show(self, text: str, seconds: float = MESSAGE_SECONDS) -> None:
        """Show a temporary message; after `seconds` the default text returns."""
        self.message = text
        self.until = time.perf_counter() + seconds
        self.invalidate()

    def expire(self, now: float) -> None:
        """End a temporary message whose time is up."""
        if self.until and now > self.until:
            self.until, self.message = 0.0, self.default
            self.invalidate()

    def draw(self, canvas: Canvas) -> None:
        """Draw the bar and its three texts."""
        super().draw(canvas)
        face, muted = font.regular(), rgba(look.MUTED)
        baseline = text_baseline(self.y, STATUS_H, 12)
        canvas.text(face, self.message, (12, baseline), size=12, color=muted)
        right = self.width - 12 - face.width(self.count, 12)
        canvas.text(face, self.count, (right, baseline), size=12, color=muted)
        place = "x  -   y  -"
        if self.coords is not None:
            place = f"x {self.coords[0]:5.2f} m   y {self.coords[1]:5.2f} m"
        canvas.text(
            face,
            place,
            (self.width - 170 - face.width(self.count, 12), baseline),
            size=12,
            color=muted,
        )


@dataclass
class Dock:
    """A side panel that can slide shut.

    Attributes:
        full: its width when open.
        width: its width now, which an animation changes.
        open: True if it is open or opening.
    """

    full: float
    width: float = 0.0
    open: bool = True

    def __post_init__(self) -> None:
        """Start open."""
        self.width = self.full


@dataclass
class Chrome:
    """The fixed widgets of the window."""

    root: Panel
    menubar: MenuBar
    toolbar: Panel
    status: StatusBar
    left: Panel
    right: Column


class Layout:
    """Where the fixed widgets sit.

    Args:
        shell: the shell that owns it.

    Attributes:
        chrome: the menu bar, tool bar, status bar and panels.
        pane: the main widget between the panels; the program sets it.
        docks: the open state of the "left" and "right" panels.
    """

    def __init__(self, shell: Shell, menu_titles: tuple[str, ...]) -> None:
        """Make the fixed widgets."""
        self.shell = shell
        self.chrome = Chrome(
            Panel(look.WINDOW),
            MenuBar(menu_titles),
            Panel(look.PANEL, edge="bottom"),
            StatusBar(),
            Panel(look.PANEL, edge="right"),
            Column(pad=12, gap=6, color=look.PANEL, edge="left"),
        )
        self.pane: Widget = Widget()
        self.docks = {"left": Dock(LEFT_W), "right": Dock(RIGHT_W)}

    def assemble(self) -> None:
        """Put the fixed widgets and the pane in one tree, and give it the host."""
        c = self.chrome
        c.root.add(c.menubar, c.toolbar, c.left, self.pane, c.right, c.status)
        c.root.adopt(self.shell)

    def apply(self) -> None:
        """Place every widget for the window size and the docks."""
        screen = self.shell.screen
        w, h = screen.width, screen.height
        left, right = self.docks["left"].width, self.docks["right"].width
        room = max(w - SCENE_MIN, 0)
        if left + right > room:
            share = room / (left + right)
            left, right = left * share, right * share
        top = MENU_H + TOOL_H
        body = h - top - STATUS_H
        c = self.chrome
        c.root.set_rect(0, 0, w, h)
        c.menubar.set_rect(0, 0, w, MENU_H)
        c.toolbar.set_rect(0, MENU_H, w, TOOL_H)
        c.status.set_rect(0, h - STATUS_H, w, STATUS_H)
        c.left.set_rect(0, top, left, body)
        c.right.set_rect(w - right, top, right, body)
        self.pane.set_rect(left, top, w - left - right, body)
        c.left.visible, c.right.visible = left > 1, right > 1
        self.shell.program.layout_toolbar(w)
        c.right.layout()
        screen.damage.everything()

    def slide(self, which: str, on: bool) -> None:
        """Slide a panel open or shut."""
        dock = self.docks[which]
        dock.open = on
        self.shell.animator.start(
            which,
            dock.width,
            dock.full if on else 0.0,
            look.MS["panel"],
            lambda width: self._set_width(which, width),
        )

    def _set_width(self, which: str, width: float) -> None:
        """Set a panel's width and lay the window out again."""
        self.docks[which].width = width
        self.apply()


class Screen:
    """What is drawn: the damage, the canvas and the frame."""

    def __init__(self, width: int, height: int) -> None:
        """Make a frame of this size, all of it needing a draw."""
        self.surface = Surface(width, height)
        self.damage = Damage(width, height)

    @property
    def width(self) -> int:
        """Width in pixels."""
        return self.surface.canvas.width

    @property
    def height(self) -> int:
        """Height in pixels."""
        return self.surface.canvas.height

    def resize(self, width: int, height: int) -> None:
        """Make the frame another size."""
        self.surface = Surface(width, height)
        self.damage.size = (width, height)


class Input:
    """Pointer, keyboard focus, menus and tooltips.

    Args:
        shell: the shell that owns it.

    Attributes:
        pointer: where the pointer is.
        popup: the menu or choice list that is open, or None.
    """

    def __init__(self, shell: Shell) -> None:
        """Start with the pointer nowhere and nothing open."""
        self.shell = shell
        self.pointer = Pointer(since=time.perf_counter())
        self.popup: Popup | None = None

    # -- the mouse --
    def hit(self, x: float, y: float) -> Widget | None:
        """The widget under a point; an open menu is on top."""
        if self.popup is not None and self.popup.contains(x, y):
            return self.popup
        return self.shell.layout.chrome.root.hit(x, y)

    def _hover(self, target: Widget | None) -> None:
        """Tell the old and new widgets under the pointer, and clear the tooltip."""
        pointer = self.pointer
        if pointer.hovered is not None:
            pointer.hovered.on_hover(False)
        pointer.hovered = target
        if target is not None:
            target.on_hover(True)
        pointer.tip = None
        pointer.since = time.perf_counter()
        screen = self.shell.screen
        screen.damage.add((0, 0, screen.width, MENU_H + TOOL_H))

    def move(self, x: float, y: float) -> None:
        """The pointer moved to a pixel."""
        shell, pointer = self.shell, self.pointer
        menubar = shell.layout.chrome.menubar
        damage = shell.screen.damage
        pointer.x, pointer.y = x, y
        target = pointer.captured or self.hit(x, y)
        if target is not pointer.hovered:
            self._hover(target)
        if target is not None:
            target.on_move(Event(x, y))
        title = menubar.title_at(x, y)
        if self.popup is not None and self.popup.owner in menubar.titles and title:
            if (
                title != self.popup.owner
            ):  # with a menu open, another title opens that one
                self.open_menu(title)
        hover = title if target is menubar else None
        if hover != menubar.hover_title:
            menubar.hover_title = hover
            damage.add((0, 0, shell.screen.width, MENU_H))
        status = shell.layout.chrome.status
        status.coords = shell.program.world_at(x, y)
        status.invalidate()
        if pointer.tip:
            pointer.tip = None
            damage.everything()

    def press(self, event: Event) -> None:
        """A mouse button went down."""
        if event.button != 0:
            return
        shell = self.shell
        menubar = shell.layout.chrome.menubar
        if self.popup is not None and not self.popup.contains(event.x, event.y):
            self.close_popup()
            if event.y >= MENU_H:
                return
        target = self.hit(event.x, event.y)
        if target is menubar and event.y < MENU_H:
            title = menubar.title_at(event.x, event.y)
            if title:
                self.open_menu(title)
            return
        shell.keyboard_focus = False
        if not (target and target.takes_focus_on_press):
            self.set_focus(None)
        self.pointer.captured = target
        self.pointer.tip = None
        if target is not None:
            target.on_press(event)

    def release(self, event: Event) -> None:
        """A mouse button came up."""
        target, self.pointer.captured = self.pointer.captured, None
        if target is not None:
            target.on_release(event)

    def scroll(self, dy: float) -> None:
        """The wheel turned: the widget under the pointer, else the first parent."""
        pointer = self.pointer
        widget = self.hit(pointer.x, pointer.y)
        while widget is not None:
            if widget.on_scroll(Event(pointer.x, pointer.y, dy=dy)):
                return
            widget = widget.parent

    # -- the keyboard and the focus --
    def key(self, key: int, mods: int) -> None:
        """A key went down or repeated; Tab moves the focus."""
        shell = self.shell
        event = Event(key=key, shift=bool(mods & 1), ctrl=bool(mods & 2))
        if self.popup is None and key == KEY["tab"]:
            self.move_focus(-1 if event.shift else 1)
        elif shell.focus is not None and shell.focus.on_key(event):
            return
        elif self._activates(key):
            shell.focus.activate()
        elif key == KEY["escape"]:
            self.close_popup()
            if shell.focus is not None:
                self.set_focus(None)
                shell.keyboard_focus = False
        elif event.ctrl and key == KEY["q"]:
            shell.quit_requested = True
        else:
            shell.program.command(key, event.ctrl)

    def _activates(self, key: int) -> bool:
        """True if Space or Enter should click the focused widget."""
        return (
            self.popup is None
            and self.shell.focus is not None
            and key in (KEY["space"], KEY["enter"], KEY["kp_enter"])
        )

    def focusables(self, widget: Widget | None = None) -> list[Widget]:
        """Every widget the Tab key can reach, in drawing order."""
        widget = self.shell.layout.chrome.root if widget is None else widget
        usable = widget.focusable and widget.visible and widget.enabled
        found = [widget] if usable and widget.width > 0 else []
        for child in widget.children:
            if child.visible:
                found += self.focusables(child)
        return found

    def move_focus(self, step: int) -> None:
        """Move the focus to the next (1) or previous (-1) widget, with a ring."""
        order = self.focusables()
        if not order:
            return
        shell = self.shell
        here = (
            order.index(shell.focus)
            if shell.focus in order
            else (-1 if step > 0 else 0)
        )
        shell.keyboard_focus = True
        self.set_focus(order[(here + step) % len(order)])
        shell.screen.damage.everything()

    def set_focus(self, widget: Widget | None) -> None:
        """Give the focus to a widget, or to none, and tell both."""
        shell = self.shell
        if widget is shell.focus:
            return
        old, shell.focus = shell.focus, widget
        if old is not None:
            old.on_focus_change(False)
        if widget is not None:
            widget.on_focus_change(True)

    # -- popups and menus --
    def open_popup(self, popup: Popup) -> None:
        """Show a popup, closing any other."""
        self.close_popup()
        popup.adopt(self.shell)
        self.popup = popup
        self.shell.layout.chrome.menubar.open_title = popup.owner
        self.shell.screen.damage.everything()

    def close_popup(self) -> None:
        """Close the popup that is open."""
        if self.popup is not None:
            self.popup = None
            self.shell.layout.chrome.menubar.open_title = None
            self.shell.screen.damage.everything()

    def open_menu(self, title: str) -> None:
        """Open the menu of a title, under it."""
        places = self.shell.layout.chrome.menubar.places
        left = next(x for name, x, _ in places if name == title)
        items = self.shell.program.menu_items(title)
        self.open_popup(Popup(items, left, MENU_H - 2, owner=title))

    def tick(self, now: float) -> None:
        """Show a tooltip that has waited long enough."""
        pointer = self.pointer
        hovered = pointer.hovered
        waited = now - pointer.since > TOOLTIP_DELAY
        if (
            pointer.tip is None
            and hovered
            and hovered.tooltip
            and waited
            and not self.popup
        ):
            pointer.tip = (
                hovered.tooltip,
                hovered.x + hovered.width / 2,
                hovered.y + hovered.height + 8,
            )
            self.shell.screen.damage.add(
                (
                    int(hovered.x) - 150,
                    int(hovered.y),
                    int(hovered.x + hovered.width) + 150,
                    int(hovered.y + hovered.height) + 44,
                )
            )


class Shell:
    """The frame of a window: layout, input, animation and rendering.

    A program is any object with the methods listed under `Program`; it fills the
    panels, gives the menus and commands, and sets the pane.

    Args:
        program: the program the shell shows.
        width: window width in pixels.
        height: window height in pixels.

    Attributes:
        program: the program.
        screen: the frame and what changed in it.
        layout: the fixed widgets and the panels.
        input: the pointer, focus and menus.
        animator: the running animations.
        focus: the widget with the keyboard focus, or None.
        keyboard_focus: True if the focus came from the keyboard, so a ring shows.
        quit_requested: True when the program wants the window closed.
    """

    def __init__(self, program, width: int, height: int) -> None:
        """Build the frame, let the program fill it, and lay it out."""
        self.program = program
        self.screen = Screen(width, height)
        self.animator = Animator()
        self.layout = Layout(self, program.menu_titles)
        self.input = Input(self)
        self.focus: Widget | None = None
        self.keyboard_focus = False
        self.quit_requested = False
        self._clip = ""
        program.build(self)
        self.layout.assemble()
        self.layout.apply()

    # -- what a widget asks of its window (`Host`) --
    def invalidate(self, widget: Widget | None = None) -> None:
        """Ask for a redraw of a widget, or of everything."""
        if widget is None:
            self.screen.damage.everything()
        else:
            self.screen.damage.widget(widget)

    def invalidate_all(self) -> None:
        """Ask for a redraw of everything."""
        self.screen.damage.everything()

    def set_focus(self, widget: Widget | None) -> None:
        """Give the focus to a widget, or to none."""
        self.input.set_focus(widget)

    def select_in_group(self, group: str, button: IconButton) -> None:
        """Check one button of a group and clear the others."""
        for member in self.program.groups.get(group, []):
            member.checked = member is button
            self.invalidate(member)

    def open_popup(self, popup: Popup) -> None:
        """Show a popup over everything."""
        self.input.open_popup(popup)

    def close_popup(self) -> None:
        """Close the popup that is open."""
        self.input.close_popup()

    def clip_get(self) -> str:
        """The clipboard text; the window replaces this with the system's."""
        return self._clip

    def clip_set(self, text: str) -> None:
        """Put text on the clipboard; the window replaces this with the system's."""
        self._clip = text

    # -- what the window asks of the program (`Application`) --
    def mouse_move(self, x: float, y: float) -> None:
        """The pointer moved to a pixel."""
        self.input.move(x, y)

    def mouse_press(self, event: Event) -> None:
        """A mouse button went down."""
        self.input.press(event)

    def mouse_release(self, event: Event) -> None:
        """A mouse button came up."""
        self.input.release(event)

    def mouse_scroll(self, dy: float) -> None:
        """The wheel turned."""
        self.input.scroll(dy)

    def key(self, key: int, _action: int, mods: int) -> None:
        """A key went down or repeated."""
        self.input.key(key, mods)

    def char(self, char: str) -> None:
        """A character was typed."""
        if self.focus is not None:
            self.focus.on_char(char)

    @property
    def dirty(self) -> bool:
        """True if a new frame is needed."""
        return self.screen.damage.dirty

    @property
    def dirty_rect(self) -> tuple[int, int, int, int] | None:
        """The box that changed, or None."""
        return self.screen.damage.rect

    def busy(self) -> bool:
        """True while something moves, so the loop does not sleep long."""
        return (
            self.animator.busy() or self.input.popup is not None or self.program.busy()
        )

    def tick(self, now: float) -> None:
        """Advance the animations, the program, tooltips and the message."""
        self.animator.tick(now)
        self.program.after_tick(now)
        self.input.tick(now)
        self.layout.chrome.status.expire(now)

    def render(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        """Draw what changed and return the whole frame, uint8 RGB.

        Args:
            width: the window width now; the frame is remade if it changed.
            height: the window height now.
        """
        screen = self.screen
        if width and (width != screen.width or height != screen.height):
            screen.resize(width, height)
            self.layout.apply()
        canvas = screen.surface.canvas
        box = screen.damage.take()
        if box[2] > box[0] and box[3] > box[1]:
            canvas.set_clip(*box)
            self.layout.chrome.root.draw(canvas)
            self._draw_overlays(canvas)
            screen.surface.copy(box)
        canvas.set_clip(0, 0, canvas.width, canvas.height)
        return screen.surface.frame

    def _draw_overlays(self, canvas: Canvas) -> None:
        """Draw the open menu and the tooltip, over everything."""
        popup, tip = self.input.popup, self.input.pointer.tip
        if popup is not None:
            popup.draw(canvas)
        if tip:
            text, center, y = tip
            face = font.regular()
            width = face.width(text, 12) + 16
            x = min(max(center - width / 2, 4), canvas.width - width - 4)
            canvas.rect((x, y, width, 26), rgba(look.NAVY), radius=4)
            canvas.text(
                face,
                text,
                (x + 8, text_baseline(y, 26, 12)),
                size=12,
                color=rgba("#ffffff"),
            )
