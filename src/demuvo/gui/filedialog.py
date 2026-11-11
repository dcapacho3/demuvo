"""A file dialog drawn by demuvo - browse folders, pick a file or name a new one.

It floats over the window like a menu. The folder is a line of text above a list;
a click selects, a double click enters a folder or takes a file, and the name box
takes a name, or a whole path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import font, look
from .icons import draw_icon
from .inputs import Popup, TextField
from .raster import Canvas, rgba
from .widgets import Event, TextButton, text_baseline

ROW = 30
# Two clicks within this many seconds are a double click.
DOUBLE_CLICK_SECONDS = 0.45


def list_folder(folder: Path, suffix: str) -> tuple[list[str], list[str]]:
    """The folders and the files ending in `suffix` of a folder, each sorted.

    Names that start with a dot are left out.

    Raises:
        OSError: if the folder cannot be read.
    """
    folders, files = [], []
    for item in folder.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir():
            folders.append(item.name)
        elif item.name.endswith(suffix):
            files.append(item.name)
    return sorted(folders, key=str.lower), sorted(files, key=str.lower)


@dataclass
class Browse:
    """What a dialog is for.

    Attributes:
        title: what is asked, for instance "Open a world".
        on_choose: called with the path chosen, after the card closes.
        name: the file name to start with.
        save: True to name a file that may not exist yet; False to pick one that does.
        suffix: the ending of the files shown, and added to a name that lacks it.
    """

    title: str
    on_choose: object
    name: str = ""
    save: bool = False
    suffix: str = ".toml"


@dataclass
class _Listing:
    """What the list shows and where the pointer and the selection are."""

    folders: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    scroll: int = 0
    hover: int | None = None
    selected: int | None = None


class FileDialog(Popup):
    """A card to open a file, or to save under a name.

    Args:
        ask: what the dialog is for; see `Browse`.
        folder: the folder to start in.
        place: (x, y, width, height) of the card.
    """

    def __init__(
        self,
        ask: Browse,
        folder: str | Path,
        place: tuple[float, float, float, float],
    ) -> None:
        """Make the card and read the folder."""
        self.path_field = TextField(str(folder), self._go, self._close)
        self.name_field = TextField(ask.name, self._accept, self._close)
        self.buttons = {
            "up": TextButton("Up", self._up),
            "home": TextButton("Home", self._home),
            "ok": TextButton("Save" if ask.save else "Open", self._accept_name),
            "cancel": TextButton("Cancel", self._close),
        }
        super().__init__([], place[0], place[1])
        self.ask = ask
        self.folder = Path(folder).expanduser()
        self.view = _Listing()
        self.note, self._confirm, self._last_click = "", None, (0.0, None)
        self._pressed = None
        self.add(self.path_field, self.name_field, *self.buttons.values())
        self.set_rect(*place)
        self._read(self.folder)

    # -- the folder --
    def _read(self, folder: Path) -> None:
        """Show a folder, or say why it cannot be shown."""
        try:
            self.view.folders, self.view.files = list_folder(folder, self.ask.suffix)
        except OSError as error:
            self.note = f"Cannot read that folder: {error.strerror or error}"
            return
        self.folder = folder
        self.view.scroll, self.view.selected = 0, None
        self.note, self._confirm = "", None
        self.path_field.buffer = str(folder)
        self.invalidate()

    def _go(self, text: str) -> None:
        """Go to the folder typed in the path box."""
        folder = Path(text).expanduser()
        if folder.is_dir():
            self._read(folder)
        else:
            self.note = "That folder does not exist."
            self.invalidate()

    def _up(self) -> None:
        """Go to the folder above."""
        self._read(self.folder.parent)

    def _home(self) -> None:
        """Go to the home folder."""
        self._read(Path.home())

    # -- choosing --
    def _accept_name(self) -> None:
        """Take what the name box holds."""
        self._accept(self.name_field.buffer)

    def _accept(self, text: str) -> None:
        """Take a name or a path: enter a folder, ask before replacing, or finish."""
        target = Path(text.strip()).expanduser()
        if not target.is_absolute():
            target = self.folder / target
        if target.is_dir():
            self._read(target)
            return
        if self.ask.save and not target.name.endswith(self.ask.suffix):
            target = target.with_name(target.name + self.ask.suffix)
        if not self.ask.save and not target.is_file():
            self.note = "There is no such file."
            self.invalidate()
            return
        if self.ask.save and target.exists() and self._confirm != target:
            self._confirm = target
            self.note = f"{target.name} exists. Choose Save again to replace it."
            self.invalidate()
            return
        self._close()
        self.ask.on_choose(str(target))

    def _close(self) -> None:
        """Close the card and give the focus up."""
        self.host.set_focus(None)
        self.host.close_popup()

    # -- the list --
    def _list_rect(self) -> tuple[float, float, float, float]:
        """The rectangle of the list."""
        return (self.x + 16, self.y + 92, self.width - 32, self.height - 92 - 128)

    def _visible_rows(self) -> int:
        """How many rows fit."""
        return max(int(self._list_rect()[3] // ROW), 1)

    def _entries(self) -> list[tuple[str, bool]]:
        """Every row: a name and whether it is a folder."""
        return [(n, True) for n in self.view.folders] + [
            (n, False) for n in self.view.files
        ]

    def _entry_at(self, px: float, py: float) -> int | None:
        """The place of the row under a pixel, or None."""
        x, y, width, height = self._list_rect()
        if not (x <= px < x + width and y <= py < y + height):
            return None
        index = self.view.scroll + int((py - y) // ROW)
        return index if index < len(self._entries()) else None

    def layout(self) -> None:
        """Place the buttons, the boxes and the card."""
        x, y, w, h = self.x, self.y, self.width, self.height
        self.buttons["up"].set_rect(x + 16, y + 44, 56, 36)
        self.buttons["home"].set_rect(x + 78, y + 44, 72, 36)
        self.path_field.set_rect(x + 158, y + 44, w - 174, 36)
        self.name_field.set_rect(x + 84, y + h - 98, w - 100, 36)
        self.buttons["cancel"].set_rect(x + w - 232, y + h - 52, 100, 36)
        self.buttons["ok"].set_rect(x + w - 116, y + h - 52, 100, 36)

    def adopt(self, host) -> None:
        """Take the host and the keyboard focus, so typing goes to the name box."""
        super().adopt(host)
        host.set_focus(self.name_field)

    # -- drawing --
    def draw(self, canvas: Canvas) -> None:
        """Draw a shadow, the card, the controls and the list."""
        card = (self.x, self.y, self.width, self.height)
        shadow = (self.x + 1, self.y + 5, self.width, self.height)
        canvas.rect(shadow, rgba(look.NAVY, 0.10), radius=12)
        canvas.rect(card, rgba(look.PANEL), radius=10)
        canvas.rect_outline(card, rgba(look.LINE), width=1, radius=10)
        face = font.regular()
        title_at = (self.x + 16, text_baseline(self.y + 8, 28, 14))
        canvas.text(
            font.bold(), self.ask.title, title_at, size=14, color=rgba(look.NAVY)
        )
        for child in (self.path_field, self.name_field, *self.buttons.values()):
            child.draw(canvas)
        label_at = (self.x + 16, text_baseline(self.y + self.height - 98, 36, 13))
        canvas.text(face, "Name", label_at, size=13, color=rgba(look.NAVY))
        note_at = (self.x + 16, text_baseline(self.y + self.height - 58, 20, 12))
        canvas.text(face, self.note, note_at, size=12, color=rgba(look.WARN_TEXT))
        self._draw_list(canvas)

    def _draw_list(self, canvas: Canvas) -> None:
        """Draw the rows that are in view."""
        x, y, width, height = self._list_rect()
        canvas.rect((x, y, width, height), rgba("#ffffff"), radius=4)
        canvas.rect_outline((x, y, width, height), rgba(look.LINE), width=1, radius=4)
        face = font.regular()
        saved = canvas.clip
        canvas.set_clip(
            max(saved[0], int(x)), max(saved[1], int(y)),
            min(saved[2], int(x + width)), min(saved[3], int(y + height)),
        )  # fmt: skip
        entries = self._entries()
        for row in range(self._visible_rows() + 1):
            index = self.view.scroll + row
            if index >= len(entries):
                break
            name, folder = entries[index]
            top = y + row * ROW
            if index == self.view.selected:
                canvas.rect((x + 2, top, width - 4, ROW), rgba(look.MINT2), radius=4)
            elif index == self.view.hover:
                canvas.rect((x + 2, top, width - 4, ROW), rgba(look.MINT), radius=4)
            icon = "folder" if folder else "file"
            draw_icon(canvas, icon, (x + 22, top + ROW / 2), 18, look.NAVY)
            canvas.text(
                face, name, (x + 40, text_baseline(top, ROW, 13)),
                size=13, color=rgba(look.NAVY),
            )  # fmt: skip
        if not entries:
            at = (x + 14, text_baseline(y, ROW, 13))
            canvas.text(
                face,
                "This folder has no world files.",
                at,
                size=13,
                color=rgba(look.MUTED),
            )
        canvas.clip = saved

    # -- the pointer --
    def on_press(self, event: Event) -> None:
        """Send a click to the control under it, or select a row."""
        self._pressed = None
        for child in (self.path_field, self.name_field, *self.buttons.values()):
            if child.contains(event.x, event.y):
                self._pressed = child
                child.on_press(event)
                return
        index = self._entry_at(event.x, event.y)
        if index is not None:
            self._click_row(index)

    def _click_row(self, index: int) -> None:
        """Select a row; a second click on it enters a folder or takes a file."""
        now = time.perf_counter()
        before, was = self._last_click
        self._last_click = (now, index)
        name, folder = self._entries()[index]
        self.view.selected = index
        self.note, self._confirm = "", None
        if not folder:
            self.name_field.buffer = name
            self.name_field.cursor = len(name)
        if was == index and now - before <= DOUBLE_CLICK_SECONDS:
            if folder:
                self._read(self.folder / name)
            else:
                self._accept(name)
        self.invalidate()

    def on_release(self, event: Event) -> None:
        """Finish a click on a control."""
        child, self._pressed = self._pressed, None
        if child is not None:
            child.on_release(event)

    def on_move(self, event: Event) -> None:
        """Light the row and the button under the pointer."""
        row = self._entry_at(event.x, event.y)
        if row != self.view.hover:
            self.view.hover = row
            self.invalidate()
        for button in self.buttons.values():
            inside = button.contains(event.x, event.y)
            if inside != button.hover:
                button.on_hover(inside)

    def on_scroll(self, event: Event) -> bool:
        """Scroll the list with the wheel."""
        most = max(len(self._entries()) - self._visible_rows(), 0)
        self.view.scroll = min(
            max(self.view.scroll + (-1 if event.dy > 0 else 1), 0), most
        )
        self.invalidate()
        return True
