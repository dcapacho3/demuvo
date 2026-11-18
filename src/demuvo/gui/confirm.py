"""A card that asks a yes or no question before something that cannot be undone.

It floats over the window like the file dialog. Cancel is the answer Enter and Space
give, so a stray key press never throws work away; Esc and a click outside cancel too.
"""

from __future__ import annotations

from . import font, look
from .inputs import Popup
from .keys import KEY
from .raster import Canvas, rgba
from .widgets import Event, TextButton, text_baseline


class ConfirmBox(Popup):
    """A card with a question, a button that does it and a button that does not.

    Args:
        title: what the card is about, such as "Unsaved changes".
        question: the question, on one line.
        yes: the words on the button that does it, such as "Discard".
        on_yes: called with no argument, after the card closes.
        place: (x, y, width, height) of the card.
    """

    takes_focus_on_press = True

    def __init__(
        self,
        title: str,
        question: str,
        yes: str,
        on_yes,
        place: tuple[float, float, float, float],
    ) -> None:
        """Make the card."""
        self.buttons = {
            "no": TextButton("Cancel", self._close),
            "yes": TextButton(yes, self._yes),
        }
        super().__init__([], place[0], place[1])
        self.title, self.question, self.on_yes = title, question, on_yes
        self.focusable = True
        self.add(*self.buttons.values())
        self.set_rect(*place)
        self._pressed = None

    def layout(self) -> None:
        """Put the buttons on the right of the bottom edge."""
        x, y, w, h = self.x, self.y, self.width, self.height
        self.buttons["no"].set_rect(x + w - 232, y + h - 52, 100, 36)
        self.buttons["yes"].set_rect(x + w - 116, y + h - 52, 100, 36)

    def adopt(self, host) -> None:
        """Take the host and the keyboard focus, so Enter and Space reach the card."""
        super().adopt(host)
        host.set_focus(self)

    def on_close(self) -> None:
        """Give the keyboard focus back."""
        self.host.set_focus(None)

    def on_key(self, event: Event) -> bool:
        """Enter and Space cancel, the safe answer; Esc is the host's."""
        if event.key in (KEY["enter"], KEY["kp_enter"], KEY["space"]):
            self._close()
            return True
        return False

    def _close(self) -> None:
        """Close the card and do nothing."""
        self.host.close_popup()

    def _yes(self) -> None:
        """Close the card, then do what it asked about."""
        self._close()
        self.on_yes()

    def draw(self, canvas: Canvas) -> None:
        """Draw a shadow, the card, the question and the buttons."""
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
        canvas.text(
            font.regular(),
            self.question,
            (self.x + 16, text_baseline(self.y + 48, 28, 13)),
            size=13,
            color=rgba(look.NAVY),
        )
        for button in self.buttons.values():
            button.draw(canvas)

    def on_press(self, event: Event) -> None:
        """Send a click to the button under it."""
        self._pressed = next(
            (b for b in self.buttons.values() if b.contains(event.x, event.y)), None
        )
        if self._pressed is not None:
            self._pressed.on_press(event)

    def on_release(self, event: Event) -> None:
        """Finish a click on a button."""
        button, self._pressed = self._pressed, None
        if button is not None:
            button.on_release(event)

    def on_move(self, event: Event) -> None:
        """Light the button under the pointer."""
        for button in self.buttons.values():
            inside = button.contains(event.x, event.y)
            if inside != button.hover:
                button.on_hover(inside)
