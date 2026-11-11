"""Widgets that take a choice or a number - a number field, a drop-down, a menu, a list.

They build on the base widgets in `widgets`. A menu or the choices of a drop-down
float over everything as a `Popup`, which the host shows and closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from . import font, look
from .icons import draw_icon
from .keys import KEY
from .raster import Canvas, rgba
from .widgets import (
    BORDER,
    ROW,
    Event,
    Widget,
    fade,
    text_baseline,
)

DIGITS = "0123456789.-"
# Pixels between the field's left edge and its text.
TEXT_PAD = 10


@dataclass(frozen=True)
class NumberSpec:
    """What a number field accepts and how it shows it.

    Attributes:
        low: the smallest value.
        high: the largest value.
        step: how much an arrow key or a turn of the wheel changes it.
        decimals: digits after the point.
        suffix: text after the number, such as " m".
        integer: True for whole numbers only.
    """

    low: float
    high: float
    step: float
    decimals: int = 2
    suffix: str = ""
    integer: bool = False


class LineEdit(Widget):
    """What a text box and a number field share - a buffer, a caret and a selection.

    A click selects everything; Left, Right, Home, End, Delete, Backspace and
    Ctrl+A edit at the caret, and a click while editing puts the caret there.
    """

    takes_focus_on_press = True

    def __init__(self) -> None:
        """Start not editing, with an empty buffer."""
        super().__init__()
        self.editing = False
        self.buffer = ""
        self.cursor = 0
        self.selected = False
        self.focusable = True

    @staticmethod
    def accepts(char: str) -> bool:
        """True if a typed character belongs in the buffer."""
        return char.isprintable()

    def start(self, text: str) -> None:
        """Begin editing a text, all of it selected, so typing replaces it."""
        self.editing = True
        self.buffer = text
        self.cursor, self.selected = len(text), True

    def place_caret(self, x: float) -> None:
        """Put the caret in the gap nearest a pixel x, and drop the selection."""
        face = font.regular()
        offset = x - (self.x + TEXT_PAD)
        self.cursor = min(
            range(len(self.buffer) + 1),
            key=lambda i: abs(face.width(self.buffer[:i], 13) - offset),
        )
        self.selected = False

    def draw_text(self, canvas: Canvas, shown: str) -> None:
        """Draw the text, with the caret or the selection band when editing."""
        face = font.regular()
        left = self.x + TEXT_PAD
        if self.editing and self.selected and shown:
            width = face.width(shown, 13)
            canvas.rect(
                (left, self.y + 5, width, self.height - 10), rgba(look.TEAL, 0.25)
            )
        elif self.editing:
            caret = left + face.width(shown[: self.cursor], 13)
            canvas.rect((caret, self.y + 6, 1.5, self.height - 12), rgba(look.TEAL))
        color = rgba(look.NAVY, fade(self.enabled))
        baseline = text_baseline(self.y, self.height, 13)
        canvas.text(face, shown, (left, baseline), size=13, color=color)

    def draw_frame(self, canvas: Canvas) -> None:
        """Draw the white box, with a teal edge while editing."""
        box = (self.x, self.y, self.width, self.height)
        canvas.rect(box, rgba("#ffffff"), radius=4)
        if self.editing:
            canvas.rect_outline(box, rgba(look.TEAL), width=2, radius=4)
        else:
            edge = BORDER if self.hover and self.enabled else look.LINE
            canvas.rect_outline(box, rgba(edge), width=1, radius=4)

    def on_char(self, char: str) -> None:
        """Add a typed character the field accepts."""
        if self.editing and self.accepts(char):
            self._insert(char)
            self.invalidate()

    def _insert(self, text: str) -> None:
        """Put text at the caret, in place of the selection if there is one."""
        if self.selected:
            self.buffer, self.cursor, self.selected = "", 0, False
        self.buffer = self.buffer[: self.cursor] + text + self.buffer[self.cursor :]
        self.cursor += len(text)

    def _delete(self, backwards: bool) -> None:
        """Remove the selection, else the character before or after the caret."""
        if self.selected:
            self.buffer, self.cursor, self.selected = "", 0, False
        elif backwards and self.cursor > 0:
            self.buffer = self.buffer[: self.cursor - 1] + self.buffer[self.cursor :]
            self.cursor -= 1
        elif not backwards:
            self.buffer = self.buffer[: self.cursor] + self.buffer[self.cursor + 1 :]

    def _move(self, to: int) -> None:
        """Put the caret at a place, clear the selection."""
        self.cursor, self.selected = min(max(to, 0), len(self.buffer)), False

    def _caret_key(self, event: Event) -> bool:
        """Select all, delete, or move the caret; True if the key was one of those."""
        key = event.key
        if event.ctrl and key == KEY["a"]:
            self.selected = True
        elif key in (KEY["backspace"], KEY["delete"]):
            self._delete(key == KEY["backspace"])
        elif key in (KEY["left"], KEY["right"]):
            if self.selected:
                self._move(0 if key == KEY["left"] else len(self.buffer))
            else:
                self._move(self.cursor + (-1 if key == KEY["left"] else 1))
        elif key in (KEY["home"], KEY["end"]):
            self._move(0 if key == KEY["home"] else len(self.buffer))
        else:
            return False
        return True


class Spin(LineEdit):
    """A number field: click to type, Enter to accept, Esc to cancel.

    Clicking selects the whole number, so typing replaces it. Left, Right, Home,
    End, Delete and Backspace edit it where the caret is, and a click puts the
    caret there. Up, Down and the wheel change the value by one step; Ctrl+A,
    Ctrl+C and Ctrl+V select, copy and paste.

    Args:
        value: the starting value.
        spec: the range, step and look; see `NumberSpec`.
        on_change: called with the new value.
    """

    takes_focus_on_press = True

    def __init__(self, value: float, spec: NumberSpec, on_change=None) -> None:
        """Make a number field."""
        super().__init__()
        self.value, self.spec, self.on_change = value, spec, on_change

    @staticmethod
    def accepts(char: str) -> bool:
        """A digit, a point or a minus sign."""
        return char in DIGITS

    def _digits(self) -> str:
        """The value as typed: whole, or to the number of decimals."""
        if self.spec.integer:
            return f"{int(self.value)}"
        return f"{self.value:.{self.spec.decimals}f}"

    def text(self) -> str:
        """The value as shown when not editing, with its suffix."""
        return self._digits() + self.spec.suffix

    def draw(self, canvas: Canvas) -> None:
        """Draw the field; a teal edge and a caret while editing."""
        self.draw_frame(canvas)
        self.draw_text(canvas, self.buffer if self.editing else self.text())

    def set(self, value: float, notify: bool = True) -> None:
        """Set the value, held between the ends and rounded to the field's kind."""
        value = min(max(value, self.spec.low), self.spec.high)
        if self.spec.integer:
            value = int(round(value))
        else:
            value = round(value, max(self.spec.decimals, 3))
        changed = value != self.value
        self.value = value
        self.invalidate()
        if changed and notify and self.on_change:
            self.on_change(value)

    def on_press(self, event: Event) -> None:
        """A click takes the focus, which starts editing; another puts the caret."""
        super().on_press(event)
        if not self.enabled:
            return
        if self.editing:
            self.place_caret(event.x)
        else:
            self.host.set_focus(self)

    def on_focus_change(self, gained: bool) -> None:
        """Start editing with the number selected, or accept what was typed."""
        if gained and self.enabled:
            self.start(self._digits())
        elif not gained and self.editing:
            self._commit()
        self.invalidate()

    def _commit(self) -> None:
        """Take the typed text as the value if it is a number; else keep the old one."""
        self.editing = False
        try:
            self.set(float(self.buffer))
        except ValueError:
            pass

    def _paste(self) -> None:
        """Put the digits of the clipboard at the caret."""
        pasted = "".join(c for c in self.host.clip_get().strip() if c in DIGITS)
        if pasted:
            self.editing = True
            self._insert(pasted)

    def _step(self, up: bool) -> None:
        """Change the value by one step and show it in the buffer."""
        self.editing = False
        self.set(self.value + (self.spec.step if up else -self.spec.step))
        self.start(self._digits())

    def on_key(self, event: Event) -> bool:
        """Copy, paste, delete, accept, cancel and step; True if the key was used."""
        if not self.enabled:
            return False
        key = event.key
        if event.ctrl and key == KEY["c"]:
            self.host.clip_set(self.buffer if self.editing else f"{self.value}")
        elif event.ctrl and key == KEY["v"]:
            self._paste()
        elif self.editing and self._caret_key(event):
            pass
        elif key in (KEY["enter"], KEY["kp_enter"]):
            self._commit()
            self.host.set_focus(None)
        elif key == KEY["escape"]:
            self.editing = False
            self.host.set_focus(None)
        elif key in (KEY["up"], KEY["down"]):
            self._step(key == KEY["up"])
        else:
            return False
        self.invalidate()
        return True

    def on_scroll(self, event: Event) -> bool:
        """A turn of the wheel over the field changes it by a step."""
        if self.enabled and self.hover:
            self.set(self.value + (self.spec.step if event.dy > 0 else -self.spec.step))
            return True
        return False


class TextField(LineEdit):
    """A box for one line of text: Enter hands it over, Esc gives up.

    Args:
        text: the starting text.
        on_submit: called with the text when Enter is pressed.
        on_cancel: called when Esc is pressed.
    """

    def __init__(self, text: str = "", on_submit=None, on_cancel=None) -> None:
        """Make a text box."""
        super().__init__()
        self.buffer = text
        self.on_submit, self.on_cancel = on_submit, on_cancel

    def draw(self, canvas: Canvas) -> None:
        """Draw the box and its text."""
        self.draw_frame(canvas)
        self.draw_text(canvas, self.buffer)

    def on_press(self, event: Event) -> None:
        """A click takes the focus; a click while editing puts the caret."""
        super().on_press(event)
        if not self.enabled:
            return
        if self.editing:
            self.place_caret(event.x)
        else:
            self.host.set_focus(self)

    def on_focus_change(self, gained: bool) -> None:
        """Start editing with the text selected, or stop."""
        if gained and self.enabled:
            self.start(self.buffer)
        else:
            self.editing = False
        self.invalidate()

    def on_key(self, event: Event) -> bool:
        """Copy, paste, edit, hand over or give up; True if the key was used."""
        if not self.enabled:
            return False
        key = event.key
        if event.ctrl and key == KEY["c"]:
            self.host.clip_set(self.buffer)
        elif event.ctrl and key == KEY["v"]:
            pasted = "".join(c for c in self.host.clip_get() if self.accepts(c))
            self._insert(pasted)
        elif self.editing and self._caret_key(event):
            pass
        elif key in (KEY["enter"], KEY["kp_enter"]) and self.on_submit:
            self.on_submit(self.buffer)
        elif key == KEY["escape"] and self.on_cancel:
            self.on_cancel()
        else:
            return False
        self.invalidate()
        return True


@dataclass
class MenuItem:
    """One line of a menu.

    Attributes:
        text: the label.
        checked: None for no check column; True shows a tick, False leaves room.
        shortcut: the keys, shown at the right.
        enabled: False greys it and ignores a click.
        action: called with no argument when it is chosen.
        separator: True for a thin line instead of a line of text.
    """

    text: str = ""
    checked: bool | None = None
    shortcut: str = ""
    enabled: bool = True
    action: object = None
    separator: bool = False


class Popup(Widget):
    """A list that floats over everything: a menu, or the choices of a drop-down.

    A click outside closes it; that is the host's work.

    Args:
        items: the lines.
        x: left edge.
        y: top edge.
        min_width: the least width.
        owner: the name of what opened it, such as a menu title, or None.
    """

    ROW_HEIGHT = 32
    SEPARATOR_HEIGHT = 10

    def __init__(
        self,
        items: list[MenuItem],
        x: float,
        y: float,
        min_width: float = 160,
        owner: str | None = None,
    ) -> None:
        """Size the popup to its widest line and place it."""
        super().__init__()
        self.items, self.owner = items, owner
        self.flip_to: float | None = None
        self.has_check = any(item.checked is not None for item in items)
        self.hover_row = -1
        self.set_rect(x, y, max(min_width, self._widest()), self._height())

    def _widest(self) -> float:
        """The width the lines need, including the check column and the shortcut."""
        face = font.regular()
        lines = [item for item in self.items if not item.separator]
        return max(
            (
                face.width(item.text, 13)
                + (face.width(item.shortcut, 12) + 28 if item.shortcut else 0)
                + (40 if self.has_check else 20)
                + 24
                for item in lines
            ),
            default=0,
        )

    def _height(self) -> float:
        """The height of all the lines and the space above and below."""
        return sum(self._row_height(item) for item in self.items) + 12

    def _row_height(self, item: MenuItem) -> int:
        """The height of one line."""
        return self.SEPARATOR_HEIGHT if item.separator else self.ROW_HEIGHT

    def _rows(self):
        """Each line with its index, top edge and height."""
        top = self.y + 6
        for index, item in enumerate(self.items):
            height = self._row_height(item)
            yield index, item, top, height
            top += height

    def draw(self, canvas: Canvas) -> None:
        """Draw a soft shadow, the card, and each line."""
        canvas.rect(
            (self.x + 1, self.y + 5, self.width, self.height),
            rgba(look.NAVY, 0.10),
            radius=10,
        )
        card = (self.x, self.y, self.width, self.height)
        canvas.rect(card, rgba(look.PANEL), radius=8)
        canvas.rect_outline(card, rgba(look.LINE), width=1, radius=8)
        for index, item, top, height in self._rows():
            self._draw_row(canvas, index, item, top, height)

    def _draw_row(
        self, canvas: Canvas, index: int, item: MenuItem, top, height
    ) -> None:
        """Draw one line: a separator, or a highlight, a tick, the text and the keys."""
        if item.separator:
            canvas.rect((self.x + 8, top + 4, self.width - 16, 1), rgba(look.LINE))
            return
        if index == self.hover_row and item.enabled:
            canvas.rect(
                (self.x + 6, top, self.width - 12, height), rgba(look.MINT), radius=4
            )
        face = font.regular()
        if item.checked:
            draw_icon(canvas, "check", (self.x + 22, top + height / 2), 18, look.TEAL)
        left = self.x + (38 if self.has_check else 16)
        color = rgba(look.NAVY, 1.0 if item.enabled else 0.4)
        canvas.text(
            face,
            item.text,
            (left, text_baseline(top, height, 13)),
            size=13,
            color=color,
        )
        if item.shortcut:
            right = self.x + self.width - 14 - face.width(item.shortcut, 12)
            baseline = text_baseline(top, height, 12)
            canvas.text(
                face, item.shortcut, (right, baseline), size=12, color=rgba(look.MUTED)
            )

    def row_at(self, py: float) -> int:
        """The index of the selectable line at a y, or -1."""
        for index, item, top, height in self._rows():
            if top <= py < top + height and not item.separator:
                return index
        return -1

    def on_move(self, event: Event) -> None:
        """Light the line under the pointer."""
        row = self.row_at(event.y)
        if row != self.hover_row:
            self.hover_row = row
            self.host.invalidate(self)

    def on_release(self, event: Event) -> None:
        """Close the popup and run the action of the line that was clicked."""
        row = self.row_at(event.y)
        if row >= 0 and self.items[row].enabled:
            action = self.items[row].action
            self.host.close_popup()
            if action:
                action()


class PromptPopup(Popup):
    """A small card over the window that asks for one line of text.

    Args:
        title: what is asked, for instance "Open a world file".
        text: the text the box starts with.
        on_submit: called with the text when Enter is pressed; the card closes first.
        place: (x, y, width) of the card - its left and top edges and its width.
        hint: a line of help under the box.
    """

    HEIGHT = 124

    def __init__(
        self,
        title: str,
        text: str,
        on_submit,
        place: tuple[float, float, float],
        *,
        hint: str = "",
    ) -> None:
        """Make the card with its box."""
        x, y, width = place
        self.field = TextField(text, self._submit, self._close)
        super().__init__([], x, y)
        self.title, self.hint, self.on_submit = title, hint, on_submit
        self.add(self.field)
        self.set_rect(x, y, width, self.HEIGHT)

    def layout(self) -> None:
        """Put the box inside the card."""
        self.field.set_rect(self.x + 16, self.y + 42, self.width - 32, 36)

    def adopt(self, host) -> None:
        """Take the host and the keyboard focus, so typing goes to the box."""
        super().adopt(host)
        host.set_focus(self.field)

    def _close(self) -> None:
        """Close the card and give the focus up."""
        self.host.set_focus(None)
        self.host.close_popup()

    def _submit(self, text: str) -> None:
        """Close the card, then hand the text over."""
        self._close()
        self.on_submit(text)

    def draw(self, canvas: Canvas) -> None:
        """Draw a shadow, the card, the title, the box and the hint."""
        card = (self.x, self.y, self.width, self.height)
        canvas.rect(
            (self.x + 1, self.y + 5, self.width, self.height),
            rgba(look.NAVY, 0.10),
            radius=12,
        )
        canvas.rect(card, rgba(look.PANEL), radius=10)
        canvas.rect_outline(card, rgba(look.LINE), width=1, radius=10)
        canvas.text(
            font.bold(),
            self.title,
            (self.x + 16, text_baseline(self.y + 8, 28, 14)),
            size=14,
            color=rgba(look.NAVY),
        )
        self.field.draw(canvas)
        canvas.text(
            font.regular(),
            self.hint,
            (self.x + 16, text_baseline(self.y + 84, 28, 12)),
            size=12,
            color=rgba(look.MUTED),
        )

    def on_press(self, event: Event) -> None:
        """A click on the box goes to the box."""
        if self.field.contains(event.x, event.y):
            self.field.on_press(event)

    def on_move(self, event: Event) -> None:
        """Nothing to light."""

    def on_release(self, event: Event) -> None:
        """Nothing to run."""


class Combo(Widget):
    """A drop-down that shows the current choice and opens a list of the others.

    Args:
        items: the choices.
        current: the one shown.
        on_change: called with the text of a new choice.
    """

    def __init__(self, items: list[str], current: str = "", on_change=None) -> None:
        """Make a drop-down."""
        super().__init__()
        self.items, self.current, self.on_change = items, current, on_change
        self.focusable = True

    def activate(self) -> None:
        """Enter or Space opens the list."""
        if self.enabled:
            self._open()

    def _open(self) -> None:
        """Show the choices under the field, as wide as it is."""
        lines = [
            MenuItem(item, checked=item == self.current, action=self._chooser(item))
            for item in self.items
        ]
        popup = Popup(lines, self.x, self.y + self.height + 4, self.width)
        popup.flip_to = self.y - 4  # where its bottom goes if there is no room below
        self.host.open_popup(popup)

    def _chooser(self, text: str):
        """A function that makes `text` the choice."""
        return lambda: self.choose(text)

    def draw(self, canvas: Canvas) -> None:
        """Draw the field, the choice and the arrow."""
        box = (self.x, self.y, self.width, self.height)
        canvas.rect(box, rgba("#ffffff"), radius=4)
        edge = BORDER if self.hover and self.enabled else look.LINE
        canvas.rect_outline(box, rgba(edge), width=1, radius=4)
        baseline = text_baseline(self.y, self.height, 13)
        color = rgba(look.NAVY, fade(self.enabled))
        canvas.text(
            font.regular(), self.current, (self.x + 10, baseline), size=13, color=color
        )
        arrow = (self.x + self.width - 16, self.y + self.height / 2)
        draw_icon(
            canvas,
            "chevron_down",
            arrow,
            16,
            look.MUTED,
            opacity=1.0 if self.enabled else 0.4,
        )
        self.ring(canvas)

    def on_release(self, event: Event) -> None:
        """A click opens the list."""
        if self._clicked_up(event):
            self._open()

    def choose(self, text: str, notify: bool = True) -> None:
        """Make a choice, and call the handler if it is a new one."""
        changed = text != self.current
        self.current = text
        self.invalidate()
        if changed and notify and self.on_change:
            self.on_change(text)


class Row(NamedTuple):
    """One row of a list.

    Attributes:
        key: what the list reports when the row is chosen; empty for a heading.
        text: the label.
        dot: a colour for a dot before the text, or None.
        indent: levels of indent.
    """

    key: str
    text: str
    dot: str | None = None
    indent: int = 0


class TreeList(Widget):
    """A list of rows: headings, and entries with an optional colour dot.

    One entry at a time can be selected.

    Args:
        on_select: called with the key of a row that was clicked.
    """

    def __init__(self, on_select=None) -> None:
        """Make an empty list."""
        super().__init__()
        self.rows: list[Row] = []
        self.selected: str | None = None
        self.hover_key: str | None = None
        self.on_select = on_select

    def set_rows(self, rows) -> None:
        """Replace the rows."""
        self.rows = list(rows)
        self.invalidate()

    def _row_at(self, py: float) -> Row | None:
        """The row at a y, or None."""
        index = int((py - self.y) // ROW)
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def draw(self, canvas: Canvas) -> None:
        """Draw each row, lit if selected or under the pointer."""
        for index, row in enumerate(self.rows):
            top = self.y + index * ROW
            if row.key and row.key == self.selected:
                canvas.rect((self.x, top, self.width, ROW), rgba(look.MINT2), radius=4)
            elif row.key and row.key == self.hover_key:
                canvas.rect((self.x, top, self.width, ROW), rgba(look.MINT), radius=4)
            left = self.x + 10 + row.indent * 14
            if row.dot:
                center = (left + 9, top + ROW / 2)
                canvas.circle(
                    center, 8, fill=rgba(row.dot), stroke=rgba("#ffffff"), width=1.5
                )
                left += 26
            baseline = text_baseline(top, ROW, 13)
            canvas.text(
                font.regular(),
                row.text,
                (left, baseline),
                size=13,
                color=rgba(look.NAVY),
            )

    def on_move(self, event: Event) -> None:
        """Track which entry the pointer is over."""
        row = self._row_at(event.y)
        key = row.key if row and row.key else None
        if key != self.hover_key:
            self.hover_key = key
            self.invalidate()

    def on_hover(self, inside: bool) -> None:
        """Forget the hover row when the pointer leaves."""
        if not inside:
            self.hover_key = None
        super().on_hover(inside)

    def on_release(self, event: Event) -> None:
        """A click on an entry selects it through the handler."""
        clicked = self._clicked_up(event)
        row = self._row_at(event.y)
        if clicked and row and row.key and self.on_select:
            self.on_select(row.key)
