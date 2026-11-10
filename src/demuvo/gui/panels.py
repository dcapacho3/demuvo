"""The small pieces of the window that are not controls.

A warning note, a form row, a title, a badge and the playback pill. They take
callbacks, so none of them knows the application.
"""

from __future__ import annotations

from . import font, look
from .inputs import Combo
from .raster import Canvas, rgba
from .sprites import render_sprite
from .widgets import IconButton, Label, Widget, fade, text_baseline

# Playback speeds, as multiples of real time.
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# Width of the label of a form row, in pixels.
LABEL_WIDTH = 92
# What the clock shows before a run starts.
START_CLOCK = "0:00.0"


class Note(Widget):
    """A warning box that says why something is locked; it hides when it has no text."""

    def __init__(self) -> None:
        """Make an empty note, which is hidden."""
        super().__init__()
        self.label = Label("", 12, look.WARN_TEXT, wrap=True)
        self.interactive = False
        self.visible = False

    def set_text(self, text: str) -> None:
        """Show a text, or hide the note if it is empty."""
        self.label.text = text
        self.visible = bool(text)

    def height_for(self, width: float) -> float:
        """The height the text needs inside the box."""
        return self.label.height_for(width - 20) + 16

    def layout(self) -> None:
        """Put the text inside the box."""
        self.label.set_rect(self.x + 10, self.y + 6, self.width - 20, self.height - 12)

    def draw(self, canvas: Canvas) -> None:
        """Draw the box and its text."""
        box = (self.x, self.y, self.width, self.height)
        canvas.rect(box, rgba(look.WARN_BG), radius=4)
        canvas.rect_outline(box, rgba(look.WARN_LINE), width=1, radius=4)
        self.label.draw(canvas)


class FormRow(Widget):
    """A label on the left and a control on the right.

    Args:
        label: the words.
        field: the control.
    """

    def __init__(self, label: str, field: Widget) -> None:
        """Make a row holding one control."""
        super().__init__()
        self.label, self.field = label, field
        self.add(field)
        self.preferred_height = 36.0
        self.interactive = False

    def layout(self) -> None:
        """Place the control to the right of the label."""
        self.field.set_rect(
            self.x + LABEL_WIDTH, self.y + 3, self.width - LABEL_WIDTH, self.height - 6
        )

    def draw(self, canvas: Canvas) -> None:
        """Draw the label, faded if the control is off, then the control."""
        color = rgba(look.NAVY, 1.0 if self.field.enabled else 0.55)
        baseline = text_baseline(self.y, self.height, 13)
        canvas.text(
            font.regular(), self.label, (self.x, baseline), size=13, color=color
        )
        self.field.draw(canvas)


class TitleRow(Widget):
    """A coloured dot and a name in large bold type: the title of the selected robot."""

    def __init__(self) -> None:
        """Make a title with no name."""
        super().__init__()
        self.name, self.color = "", look.TEAL
        self.preferred_height = 34.0
        self.interactive = False

    def draw(self, canvas: Canvas) -> None:
        """Draw the dot and the name."""
        middle = self.y + self.height / 2
        canvas.circle(
            (self.x + 11, middle),
            10,
            fill=rgba(self.color),
            stroke=rgba("#ffffff"),
            width=2,
        )
        baseline = text_baseline(self.y, self.height, 16)
        canvas.text(
            font.bold(),
            self.name,
            (self.x + 30, baseline),
            size=16,
            color=rgba(look.NAVY),
        )


class Badge(Widget):
    """A rounded label for the mode: editing, running or a fixed benchmark."""

    COLORS = {
        "edit": (look.MINT2, look.NAVY),
        "run": ("#fdebc8", look.WARN_TEXT),
        "bench": (look.NAVY, "#ffffff"),
    }

    def __init__(self) -> None:
        """Make an empty editing badge."""
        super().__init__()
        self.text, self.kind = "", "edit"
        self.interactive = False

    def wanted_width(self) -> float:
        """The width the text needs, with room at both ends."""
        return font.bold().width(self.text, 11) + 28

    def draw(self, canvas: Canvas) -> None:
        """Draw the pill and the text."""
        background, ink = self.COLORS[self.kind]
        box = (self.x, self.y, self.width, self.height)
        canvas.rect(box, rgba(background), radius=self.height / 2)
        baseline = text_baseline(self.y, self.height, 11)
        canvas.text(
            font.bold(), self.text, (self.x + 14, baseline), size=11, color=rgba(ink)
        )


class Pill(Widget):
    """The playback pill over the scene: play or pause, step, reset, speed, the time.

    Args:
        on_play: called when play or pause is clicked.
        on_step: called for one step.
        on_reset: called for a reset.
        on_speed: called with the new speed, a multiple of real time.
    """

    PAD = 8

    def __init__(self, on_play, on_step, on_reset, on_speed) -> None:
        """Make the pill with its three buttons and its speed choice."""
        super().__init__()
        self.play = IconButton("play", on_play, tooltip="Play (Space)")
        self.step = IconButton("step", on_step, tooltip="Step one control period (S)")
        self.reset = IconButton(
            "reset", on_reset, tooltip="Reset: end the run and edit again (R)"
        )
        self.speed = Combo(
            [f"{s:g}x" for s in SPEEDS],
            "1x",
            on_change=lambda text: on_speed(float(text[:-1])),
        )
        self.clock = START_CLOCK
        self._body_key: tuple[int, int] | None = None
        self._body = None
        self.add(self.play, self.step, self.reset, self.speed)
        self.set_rect(0, 0, 4 + 32 * 3 + 6 + 66 + 8 + 66 + 10, 44)

    def set_playing(self, playing: bool) -> None:
        """Show a pause button while playing, and a play button when not."""
        self.play.icon = "pause" if playing else "play"
        self.play.tooltip = "Pause (Space)" if playing else "Play (Space)"

    def set_time(self, seconds: float) -> bool:
        """Show a time as minutes and seconds; True if the text changed."""
        minutes, rest = divmod(seconds, 60)
        text = f"{int(minutes)}:{rest:04.1f}"
        changed = text != self.clock
        self.clock = text
        return changed

    def place(self, center_x: float, bottom: float) -> None:
        """Put the pill's bottom centre at a point."""
        self.set_rect(
            center_x - self.width / 2, bottom - self.height, self.width, self.height
        )

    def layout(self) -> None:
        """Place the buttons and the speed choice in a row."""
        x = self.x + 6
        for button in (self.play, self.step, self.reset):
            button.set_rect(x, self.y + 6, 32, 32)
            x += 32
        self.speed.set_rect(x + 6, self.y + 7, 66, 30)

    def _shadow(self) -> object:
        """The soft shadow, card and edge, drawn once for the size and kept."""
        key = (round(self.width), round(self.height))
        if self._body_key != key:
            self._body_key = key

            def draw(canvas: Canvas, cx: float, cy: float) -> None:
                x, y = cx - self.width / 2, cy - self.height / 2
                for spread in (4, 3, 2, 1):
                    box = (
                        x - spread,
                        y + 2 + spread,
                        self.width + 2 * spread,
                        self.height,
                    )
                    canvas.rect(
                        box, rgba(look.NAVY, 0.018), radius=self.height / 2 + spread
                    )
                card = (x, y, self.width, self.height)
                canvas.rect(card, rgba(look.PANEL), radius=self.height / 2)
                canvas.rect_outline(
                    card, rgba(look.LINE), width=1, radius=self.height / 2
                )

            self._body = render_sprite(
                (int(self.width) + 2 * self.PAD, int(self.height) + 2 * self.PAD), draw
            )
        return self._body

    def draw(self, canvas: Canvas) -> None:
        """Draw the card, the buttons and the clock."""
        body = self._shadow()
        left = round(self.x + self.width / 2 - body.width / 2)
        top = round(self.y + self.height / 2 - body.height / 2)
        canvas.blend_rgba(left, top, body.rgba)
        for child in self.children:
            child.draw(canvas)
        baseline = text_baseline(self.y, self.height, 13)
        color = rgba(look.NAVY, fade(self.enabled))
        canvas.text(
            font.regular(),
            self.clock,
            (self.x + self.width - 62, baseline),
            size=13,
            color=color,
        )
