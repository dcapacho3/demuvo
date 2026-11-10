"""The widget toolkit - a tree of rectangles that draw themselves and take events.

A widget has a rectangle, children, and hooks for the mouse and keyboard. A tree
hangs under one root, and a host (the application window) routes the events,
runs the animations and redraws what changed. The host is the small contract
`Host`, so these widgets can be tested with a stand-in. Nothing here comes from
another toolkit.

This file holds the base widget, text, buttons, a checkbox, a slider and the
panels that lay widgets out. Number fields, choices and lists are in `inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from . import font, look
from .icons import draw_icon
from .raster import Canvas, rgba

# The height of a row of controls, in pixels.
ROW = 32
BORDER = "#9fb8c4"
DISABLED = 0.45


class Host(Protocol):
    """What a widget needs from the window that holds it.

    Attributes:
        focus: the widget that has the keyboard focus, or None.
        keyboard_focus: True if the focus came from the keyboard, so a ring shows.
    """

    focus: object
    keyboard_focus: bool

    def invalidate(self, widget=None) -> None:
        """Mark a widget, or everything, as needing a redraw."""

    def set_focus(self, widget) -> None:
        """Move the keyboard focus to a widget, or to none."""

    def select_in_group(self, group: str, button) -> None:
        """Check one button of a group of buttons and clear the others."""

    def open_popup(self, popup) -> None:
        """Show a popup over everything."""

    def close_popup(self) -> None:
        """Close the popup that is open."""

    def clip_get(self) -> str:
        """The text on the clipboard."""

    def clip_set(self, text: str) -> None:
        """Put text on the clipboard."""


@dataclass
class Event:
    """A mouse or keyboard event, in window pixels.

    Attributes:
        x: pointer x.
        y: pointer y.
        button: the mouse button, 0 for the left one.
        key: a key code from `keys.KEY`.
        dy: how far the wheel turned; positive is up.
        shift: True if Shift is held.
        ctrl: True if Control is held.
    """

    x: float = 0.0
    y: float = 0.0
    button: int = 0
    key: int = 0
    dy: float = 0.0
    shift: bool = False
    ctrl: bool = False


def text_baseline(top: float, height: float, size: int) -> float:
    """The baseline that centres a line of text of this size in a box."""
    return top + height / 2 + size * 0.36


def fade(enabled: bool) -> float:
    """The opacity of a control: full, or faded when it is disabled."""
    return 1.0 if enabled else DISABLED


def _overlaps(widget: Widget, canvas: Canvas) -> bool:
    """True if the widget overlaps the canvas's clip box."""
    x0, y0, x1, y1 = canvas.clip
    return (
        widget.x < x1
        and widget.x + widget.width > x0
        and widget.y < y1
        and widget.y + widget.height > y0
    )


class Widget:  # pylint: disable=unused-argument
    """A rectangle that draws itself, holds children and takes events.

    The event hooks take the event even where this base class ignores it, so a
    subclass can override one without changing its signature.

    Attributes:
        host: the window that holds the tree; set when the widget is added.
        parent: the widget that holds this one.
        x: left edge.
        y: top edge.
        width: width.
        height: height.
        children: the widgets inside, back to front.
        visible: False hides it and its children.
        enabled: False greys it and makes it ignore input.
        hover: True while the pointer is over it.
        down: True while it is pressed.
        tooltip: text shown after a pause over it, or None.
        preferred_height: the height a column gives it.
        interactive: False lets a click pass through to what is behind.
        focusable: True if the Tab key can stop on it.
    """

    def __init__(self) -> None:
        """Make a widget with no size, no parent and no children."""
        self.host: Host | None = None
        self.parent: Widget | None = None
        self.x = self.y = self.width = self.height = 0.0
        self.children: list[Widget] = []
        self.visible = True
        self.enabled = True
        self.hover = False
        self.down = False
        self.tooltip: str | None = None
        self.preferred_height = float(ROW)
        self.interactive = True
        self.focusable = False

    def add(self, *widgets: Widget) -> Widget:
        """Put widgets inside this one, in front of those already there."""
        for widget in widgets:
            widget.parent = self
            self.children.append(widget)
            widget.adopt(self.host)
        return self

    def adopt(self, host: Host | None) -> None:
        """Give this widget and all below it the same host."""
        self.host = host
        for child in self.children:
            child.adopt(host)

    def clear(self) -> None:
        """Remove every child."""
        self.children = []

    def set_rect(self, x: float, y: float, width: float, height: float) -> None:
        """Place the widget, and its children by `layout`."""
        self.x, self.y, self.width, self.height = x, y, width, height
        self.layout()

    def layout(self) -> None:
        """Place the children; a widget with none has nothing to do."""

    def contains(self, px: float, py: float) -> bool:
        """True if the point is inside the rectangle."""
        return (
            self.x <= px < self.x + self.width and self.y <= py < self.y + self.height
        )

    def hit(self, px: float, py: float) -> Widget | None:
        """The front-most interactive widget at a point, or None."""
        if not self.visible or not self.contains(px, py):
            return None
        for child in reversed(self.children):
            found = child.hit(px, py)
            if found is not None:
                return found
        return self if self.interactive else None

    def invalidate(self) -> None:
        """Ask the host to redraw this widget."""
        if self.host is not None:
            self.host.invalidate(self)

    def ring(self, canvas: Canvas, pad: float = 0.0, radius: float = 4.0) -> None:
        """Draw the focus ring while the keyboard focus is on this widget."""
        host = self.host
        if host is not None and host.focus is self and host.keyboard_focus:
            box = (
                self.x - pad,
                self.y - pad,
                self.width + 2 * pad,
                self.height + 2 * pad,
            )
            canvas.rect_outline(box, rgba(look.TEAL), width=2, radius=radius)

    def activate(self) -> None:
        """What Enter or Space does on a focused widget; most do nothing."""

    def draw(self, canvas: Canvas) -> None:
        """Draw the children that are visible and in the clip box."""
        for child in self.children:
            if child.visible and _overlaps(child, canvas):
                child.draw(canvas)

    def on_hover(self, inside: bool) -> None:
        """The pointer came in (True) or went out (False)."""
        self.hover = inside
        if not inside:
            self.down = False
        self.invalidate()

    def on_press(self, event: Event) -> None:
        """A button went down on the widget."""
        self.down = True
        self.invalidate()

    def on_release(self, event: Event) -> None:
        """A button came up."""
        self.down = False
        self.invalidate()

    def on_move(self, event: Event) -> None:
        """The pointer moved over the widget, or while it is held."""

    def on_scroll(self, event: Event) -> bool:
        """The wheel turned over the widget; True if it used the turn."""
        return False

    def on_key(self, event: Event) -> bool:
        """A key went down while the widget had the focus; True if it used it."""
        return False

    def on_char(self, char: str) -> None:
        """A character was typed while the widget had the focus."""

    def on_focus_change(self, gained: bool) -> None:
        """The widget got the keyboard focus (True) or lost it (False)."""
        self.invalidate()

    def _clicked_up(self, event: Event) -> bool:
        """Release the button and say whether it ended a click on this widget."""
        was_down = self.down
        Widget.on_release(self, event)
        return was_down and self.enabled and self.contains(event.x, event.y)


class Label(Widget):
    """Text, on one line or wrapped to the width.

    Args:
        text: the words.
        size: type size in pixels.
        color: "#rrggbb".
        bold: True for the bold face.
        align: "left", "right" or "center", on one line.
        wrap: True to break the text into lines that fit the width.
    """

    def __init__(
        self,
        text: str = "",
        size: int = 13,
        color: str = look.NAVY,
        *,
        bold: bool = False,
        align: str = "left",
        wrap: bool = False,
    ) -> None:
        """Make a label; it takes no clicks."""
        super().__init__()
        self.text, self.size, self.color = text, size, color
        self.bold, self.align, self.wrap = bold, align, wrap
        self.interactive = False

    def _face(self) -> font.Font:
        """The font of the label."""
        return font.bold() if self.bold else font.regular()

    def lines(self, width: float | None = None) -> list[str]:
        """The text as lines that fit a width; the label's own width if none given."""
        if not self.wrap:
            return [self.text]
        room = (self.width if width is None else width) - 2
        face = self._face()
        out, line = [], ""
        for word in self.text.split():
            trial = (line + " " + word).strip()
            if not line or face.width(trial, self.size) <= room:
                line = trial
            else:
                out.append(line)
                line = word
        return [*out, line]

    def height_for(self, width: float) -> float:
        """The height the text needs at a width."""
        return len(self.lines(width)) * self.size * 1.45 + 4

    def draw(self, canvas: Canvas) -> None:
        """Draw the text."""
        face = self._face()
        color = rgba(self.color, fade(self.enabled))
        if self.wrap:
            for i, line in enumerate(self.lines()):
                baseline = self.y + self.size * 1.1 + i * self.size * 1.45
                canvas.text(face, line, (self.x, baseline), size=self.size, color=color)
            return
        gap = self.width - face.width(self.text, self.size)
        x = self.x + {"left": 0.0, "right": gap}.get(self.align, gap / 2)
        baseline = text_baseline(self.y, self.height, self.size)
        canvas.text(face, self.text, (x, baseline), size=self.size, color=color)


class Section(Label):
    """A small upper-case heading that opens a group in a panel."""

    def __init__(self, text: str) -> None:
        """Make a heading."""
        super().__init__(text.upper(), 11, look.MUTED, bold=True)
        self.preferred_height = 30.0


class IconButton(Widget):
    """A square button with an icon; it can also be a switch, or one of a group.

    Args:
        icon: the name of an icon.
        on_click: called with no argument, or with the checked state if checkable.
        tooltip: the text shown after a pause.
        checkable: True if it stays down after a click.
        group: the name of a group in which one button is checked at a time.
    """

    SIZE = 32

    def __init__(
        self,
        icon: str,
        on_click=None,
        *,
        tooltip: str = "",
        checkable: bool = False,
        group: str | None = None,
    ) -> None:
        """Make a button."""
        super().__init__()
        self.icon, self.on_click, self.tooltip = icon, on_click, tooltip
        self.checkable, self.group, self.checked = checkable, group, False
        self.preferred_height = float(self.SIZE)
        self.focusable = True
        self.set_rect(0, 0, self.SIZE, self.SIZE)

    def draw(self, canvas: Canvas) -> None:
        """Draw the button, lit when it is pressed, checked or under the pointer."""
        box = (self.x, self.y, self.width, self.height)
        if self.enabled and (self.down or self.checked):
            canvas.rect(box, rgba(look.MINT2), radius=4)
        elif self.enabled and self.hover:
            canvas.rect(box, rgba(look.MINT), radius=4)
        center = (self.x + self.width / 2, self.y + self.height / 2)
        opacity = 1.0 if self.enabled else 0.4
        draw_icon(canvas, self.icon, center, 20, look.NAVY, opacity=opacity)
        self.ring(canvas)

    def on_release(self, event: Event) -> None:
        """A click ends here: press and release on the button."""
        if self._clicked_up(event):
            self.click()

    def activate(self) -> None:
        """Enter or Space clicks it."""
        if self.enabled:
            self.click()

    def click(self) -> None:
        """Toggle or select if it can be checked, then call the handler."""
        if self.checkable:
            if self.group and self.host is not None:
                self.host.select_in_group(self.group, self)
            else:
                self.checked = not self.checked
        if self.on_click:
            if self.checkable:
                self.on_click(self.checked)
            else:
                self.on_click()
        self.invalidate()


class Checkbox(Widget):
    """A box with a label that is on or off.

    Args:
        text: the label.
        checked: the starting state.
        on_change: called with the new state.
    """

    def __init__(self, text: str, checked: bool = False, on_change=None) -> None:
        """Make a checkbox."""
        super().__init__()
        self.text, self.checked, self.on_change = text, checked, on_change
        self.focusable = True

    def activate(self) -> None:
        """Enter or Space flips it."""
        if self.enabled:
            self.set(not self.checked)

    def draw(self, canvas: Canvas) -> None:
        """Draw the box and the label."""
        self.ring(canvas)
        if self.hover and self.enabled:
            canvas.rect(
                (self.x, self.y, self.width, self.height), rgba(look.MINT), radius=4
            )
        size = 18
        left, top = self.x + 8, self.y + (self.height - size) / 2
        box = (left, top, size, size)
        if self.checked:
            canvas.rect(box, rgba(look.TEAL, fade(self.enabled)), radius=4)
            draw_icon(
                canvas, "check", (left + size / 2, top + size / 2), size, "#ffffff"
            )
        else:
            canvas.rect(box, rgba("#ffffff"), radius=4)
            canvas.rect_outline(box, rgba(BORDER), width=1.5, radius=4)
        baseline = text_baseline(self.y, self.height, 13)
        color = rgba(look.NAVY, fade(self.enabled))
        canvas.text(
            font.regular(),
            self.text,
            (left + size + 10, baseline),
            size=13,
            color=color,
        )

    def on_release(self, event: Event) -> None:
        """A click flips it."""
        if self._clicked_up(event):
            self.set(not self.checked)

    def set(self, value: bool, notify: bool = True) -> None:
        """Set the state, and call the handler unless `notify` is False."""
        self.checked = value
        self.invalidate()
        if notify and self.on_change:
            self.on_change(value)


class Slider(Widget):
    """A track with a handle that sets a number between two ends.

    Args:
        value: the starting value.
        low: the value at the left end.
        high: the value at the right end.
        on_change: called with the new value.
    """

    def __init__(
        self, value: float = 1.0, low: float = 0.1, high: float = 1.0, on_change=None
    ):
        """Make a slider."""
        super().__init__()
        self.value, self.low, self.high, self.on_change = value, low, high, on_change
        self.preferred_height = 28.0

    def _track(self) -> tuple[float, float]:
        """Where the track starts and ends along x."""
        return self.x + 8, self.x + self.width - 8

    def draw(self, canvas: Canvas) -> None:
        """Draw the track, the filled part and the handle."""
        left, right = self._track()
        middle = self.y + self.height / 2
        accent = rgba(look.TEAL, 1.0 if self.enabled else 0.4)
        canvas.rect((left, middle - 2, right - left, 4), rgba(look.LINE), radius=2)
        handle = left + (right - left) * (self.value - self.low) / (
            self.high - self.low
        )
        canvas.rect((left, middle - 2, handle - left, 4), accent, radius=2)
        canvas.circle((handle, middle), 8, fill=rgba("#ffffff"))
        canvas.circle((handle, middle), 8, stroke=accent, width=2)

    def _set_from(self, x: float) -> None:
        """Set the value from a pointer x, and call the handler."""
        left, right = self._track()
        fraction = min(max((x - left) / (right - left), 0.0), 1.0)
        self.value = self.low + fraction * (self.high - self.low)
        self.invalidate()
        if self.on_change:
            self.on_change(self.value)

    def on_press(self, event: Event) -> None:
        """Jump the handle to the pointer."""
        super().on_press(event)
        if self.enabled:
            self._set_from(event.x)

    def on_move(self, event: Event) -> None:
        """Drag the handle while the button is held."""
        if self.down and self.enabled:
            self._set_from(event.x)


class Panel(Widget):
    """A filled rectangle with an optional edge line; its children are clipped to it.

    Args:
        color: the fill, "#rrggbb".
        edge: "left", "right", "top" or "bottom" for a line along that edge.
    """

    def __init__(self, color: str = look.PANEL, edge: str | None = None) -> None:
        """Make a panel."""
        super().__init__()
        self.color, self.edge = color, edge

    def draw(self, canvas: Canvas) -> None:
        """Fill, draw the edge, then the children inside the panel."""
        canvas.rect((self.x, self.y, self.width, self.height), rgba(self.color))
        line = {
            "right": (self.x + self.width - 1, self.y, 1, self.height),
            "left": (self.x, self.y, 1, self.height),
            "bottom": (self.x, self.y + self.height - 1, self.width, 1),
            "top": (self.x, self.y, self.width, 1),
        }.get(self.edge)
        if line:
            canvas.rect(line, rgba(look.LINE))
        saved = canvas.clip
        canvas.set_clip(
            max(saved[0], self.x),
            max(saved[1], self.y),
            min(saved[2], self.x + self.width),
            min(saved[3], self.y + self.height),
        )
        super().draw(canvas)
        canvas.clip = saved


class Column(Panel):
    """Children stacked top to bottom, scrolled by the wheel when they do not fit.

    Each child gets its `preferred_height`, or the height its text needs if it can
    say (`height_for`).

    Args:
        pad: space round the children.
        gap: space between them.
        color: the fill.
        edge: a line along one edge.
    """

    def __init__(
        self,
        pad: float = 12,
        gap: float = 6,
        color: str = look.PANEL,
        edge: str | None = None,
    ) -> None:
        """Make a column."""
        super().__init__(color, edge)
        self.pad, self.gap, self.scroll = pad, gap, 0.0
        self.content_height = 0.0

    def layout(self) -> None:
        """Stack the children and work out how tall they are all together."""
        y = self.y + self.pad - self.scroll
        room = self.width - 2 * self.pad
        for child in self.children:
            if not child.visible:
                continue
            asks = hasattr(child, "height_for") and (
                not isinstance(child, Label) or child.wrap
            )
            height = child.height_for(room) if asks else child.preferred_height
            child.set_rect(self.x + self.pad, y, room, height)
            y += height + self.gap
        self.content_height = y + self.scroll - self.y + self.pad - self.gap
        self.scroll = min(self.scroll, max(0.0, self.content_height - self.height))

    def on_scroll(self, event: Event) -> bool:
        """Scroll when the content is taller than the column."""
        if self.content_height <= self.height:
            return False
        limit = self.content_height - self.height
        self.scroll = min(max(self.scroll - event.dy * 40, 0.0), limit)
        self.layout()
        self.invalidate()
        return True

    def draw(self, canvas: Canvas) -> None:
        """Draw the panel, and a thin scroll bar if the content overflows."""
        super().draw(canvas)
        if self.content_height > self.height:
            share = self.height / self.content_height
            top = self.y + 2 + (self.height - 4) * (self.scroll / self.content_height)
            thumb = (self.x + self.width - 7, top, 4, (self.height - 4) * share)
            canvas.rect(thumb, rgba(look.LINE), radius=2)
