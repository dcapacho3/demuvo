"""The New map card - four small maps to choose from, and a button to take one.

Every style draws a map of the world's own size, for the world's own robot, and the
card shows them side by side with the same seed, so a student sees what each style
makes before choosing. Shuffle map draws the next seed of all four, and Shuffle
start and goal places the pair again on the same maps. Nothing is changed
in the world until Use this map is pressed.
"""

from __future__ import annotations

import numpy as np

from ..generate import STYLES
from ..verify import WorldError
from . import font, look
from .inputs import Popup
from .keys import KEY
from .raster import Canvas, rgba
from .widgets import Checkbox, Event, Slider, TextButton, draw_card, text_baseline

# What each style makes, in a few words a student can use to choose.
KINDS = {
    "field": "scattered obstacles",
    "rooms": "a building of rooms",
    "caves": "caverns",
    "maze": "corridors with dead ends",
}
HINT = "Pick a style and shuffle until the map suits you. It replaces the obstacles."
MARGIN = 16
CAPTION = 30
# The robot radius the slider offers, in metres.
RADII = (0.1, 0.6)


def _pixels(mask: np.ndarray) -> np.ndarray:
    """A picture of a grid of blocked cells, row 0 at the bottom, as RGBA bytes."""
    free = np.array([round(v * 255) for v in rgba(look.MINT)], dtype=np.uint8)
    wall = np.array([round(v * 255) for v in rgba(look.NAVY2)], dtype=np.uint8)
    return np.where(mask[::-1, :, None], wall, free)


class NewMapBox(Popup):
    """A card of four drawn maps, one per style, to choose from.

    Args:
        edits: what draws the maps and takes the chosen one; see `WorldEdits`.
        on_use: called with no argument after the map is applied.
        place: (x, y, width, height) of the card.
    """

    takes_focus_on_press = True

    def __init__(self, edits, on_use, place: tuple[float, float, float, float]):
        """Make the card and draw its first four maps."""
        self.edits, self.on_use = edits, on_use
        self.recipe = dict(edits.recipe)
        if self.recipe["seed"] is None:
            self.recipe["seed"] = edits.session.world.seed
        if self.recipe["pair_seed"] is None:
            self.recipe["pair_seed"] = self.recipe["seed"]
        self.styles = list(STYLES)
        self.drafts: dict[str, tuple] = {}
        self.tiles: dict[str, tuple[float, float, float, float]] = {}
        robots = bool(edits.session.world.robots)
        self.pair = Checkbox(
            "Place a start (teal) and a goal (amber)",
            self.recipe["pair"],
            on_change=self._paired,
        )
        self.radius = Slider(self.recipe["radius"], *RADII)
        self.radius.visible = not robots
        self.loops = Slider(self.recipe["loops"], 0.0, 1.0)
        self.apart = Slider(self.recipe["apart"], 0.1, 0.9)
        self.buttons = {
            "shuffle": TextButton("Shuffle map", self._shuffle),
            "pair": TextButton("Shuffle start and goal", self._shuffle_pair),
            "no": TextButton("Cancel", self._close),
            "yes": TextButton("Use this map", self._use),
        }
        super().__init__([], place[0], place[1])
        self.focusable = True
        self.add(self.pair, self.radius, self.loops, self.apart, *self.buttons.values())
        self.set_rect(*place)
        self._pressed = None
        self._refresh()

    # -- drawing the maps --
    def _refresh(self) -> None:
        """Draw the map of every style that has no map for these settings yet."""
        recipe = self.recipe
        for name in self.styles:
            key = (
                recipe["seed"],
                recipe["radius"],
                recipe["pair"],
                (recipe["pair_seed"], recipe["apart"]) if recipe["pair"] else None,
                recipe["loops"] if name == "maze" else 0.0,
            )
            if self.drafts.get(name, (None,))[0] != key:
                self.drafts[name] = (key, *self._draft(name))
        self.loops.visible = recipe["style"] == "maze"
        self.buttons["pair"].enabled = self.apart.visible = recipe["pair"]
        self.buttons["yes"].enabled = self.drafts[recipe["style"]][1] is not None
        self.invalidate()

    def _draft(self, name: str):
        """The world and the mask a style draws now, or None and what went wrong."""
        try:
            world, mask = self.edits.draft({**self.recipe, "style": name})
        except WorldError as error:
            return None, str(error)
        return world, mask

    # -- the controls --
    def _paired(self, on: bool) -> None:
        """The start and goal box changed."""
        self.recipe["pair"] = on
        self._refresh()

    def _shuffle(self) -> None:
        """Draw the next seed of every style."""
        self.recipe["seed"] += 1
        self._refresh()

    def _shuffle_pair(self) -> None:
        """Place the start and the goal again, on the same maps."""
        self.recipe["pair_seed"] += 1
        self._refresh()

    def _use(self) -> None:
        """Close the card and draw the chosen map into the world."""
        self._close()
        self.edits.recipe.update(self.recipe)
        self.on_use()

    def _close(self) -> None:
        """Close the card and do nothing."""
        self.host.close_popup()

    def adopt(self, host) -> None:
        """Take the host and the keyboard focus, so Enter reaches the card."""
        super().adopt(host)
        host.set_focus(self)

    def on_close(self) -> None:
        """Give the keyboard focus back."""
        self.host.set_focus(None)

    def on_key(self, event: Event) -> bool:
        """Enter takes the chosen map, and the arrows move from style to style."""
        if event.key in (KEY["enter"], KEY["kp_enter"]):
            if self.buttons["yes"].enabled:
                self._use()
            return True
        step = {KEY["right"]: 1, KEY["down"]: 2, KEY["left"]: -1, KEY["up"]: -2}
        if event.key in step:
            at = self.styles.index(self.recipe["style"]) + step[event.key]
            self.recipe["style"] = self.styles[at % len(self.styles)]
            self._refresh()
            return True
        return False

    # -- the layout --
    def layout(self) -> None:
        """Place the four tiles, the controls and the buttons."""
        x, y, w, h = self.x, self.y, self.width, self.height
        tile_w = (w - 3 * MARGIN) / 2
        controls, buttons = 72, 52
        tile_h = (h - 72 - controls - buttons - MARGIN) / 2
        for i, name in enumerate(self.styles):
            left = x + MARGIN + (i % 2) * (tile_w + MARGIN)
            top = y + 68 + (i // 2) * (tile_h + MARGIN)
            self.tiles[name] = (left, top, tile_w, tile_h)
        row = y + 68 + 2 * tile_h + MARGIN + 4
        self.pair.set_rect(x + 8, row, 310, 32)
        self.radius.set_rect(x + 450, row, 100, 32)
        self.loops.set_rect(x + 640, row, 100, 32)
        self.apart.set_rect(x + 450, row + 32, 100, 32)
        top = y + h - buttons + 8
        self.buttons["shuffle"].set_rect(x + MARGIN, top, 112, 36)
        self.buttons["pair"].set_rect(x + MARGIN + 120, top, 176, 36)
        self.buttons["no"].set_rect(x + w - 246, top, 100, 36)
        self.buttons["yes"].set_rect(x + w - 130, top, 114, 36)

    # -- drawing the card --
    def draw(self, canvas: Canvas) -> None:
        """Draw the card, the four tiles, the controls and the buttons."""
        draw_card(canvas, (self.x, self.y, self.width, self.height))
        canvas.text(
            font.bold(),
            "New map",
            (self.x + MARGIN, text_baseline(self.y + 10, 28, 15)),
            size=15,
            color=rgba(look.NAVY),
        )
        canvas.text(
            font.regular(),
            HINT,
            (self.x + MARGIN, text_baseline(self.y + 38, 22, 12)),
            size=12,
            color=rgba(look.MUTED),
        )
        for name in self.styles:
            self._tile(canvas, name)
        self._labels(canvas)
        for child in self.children:
            if child.visible:
                child.draw(canvas)

    def _tile(self, canvas: Canvas, name: str) -> None:
        """One style: its map, or why it has none, and its name below."""
        left, top, width, height = self.tiles[name]
        chosen = name == self.recipe["style"]
        box = (left, top, width, height)
        canvas.rect(box, rgba(look.WINDOW), radius=8)
        _, world, mask = self.drafts[name]
        area = (left + 8, top + 8, width - 16, height - 8 - CAPTION)
        if world is None:
            self._reason(canvas, area, mask)
        else:
            self._picture(canvas, area, world, mask)
        canvas.rect_outline(
            box,
            rgba(look.TEAL if chosen else look.LINE),
            width=3 if chosen else 1,
            radius=8,
        )
        face = font.bold()
        canvas.text(
            face,
            name,
            (left + 12, text_baseline(top + height - CAPTION, CAPTION, 13)),
            size=13,
            color=rgba(look.NAVY),
        )
        canvas.text(
            font.regular(),
            KINDS[name],
            (
                left + 12 + face.width(name, 13) + 10,
                text_baseline(top + height - CAPTION, CAPTION, 12),
            ),
            size=12,
            color=rgba(look.MUTED),
        )

    @staticmethod
    def _reason(canvas: Canvas, area, why: str) -> None:
        """Say in the tile why no map could be drawn, in as many lines as fit."""
        left, top, width, height = area
        canvas.rect(area, rgba(look.WARN_BG), radius=6)
        face, lines = font.regular(), [""]
        for word in why.split():
            if face.width(f"{lines[-1]} {word}", 12) > width - 16 and lines[-1]:
                lines.append(word)
            else:
                lines[-1] = f"{lines[-1]} {word}".strip()
        for i, line in enumerate(lines[: int((height - 16) // 18)]):
            canvas.text(
                face,
                line,
                (left + 8, text_baseline(top + 8 + 18 * i, 18, 12)),
                size=12,
                color=rgba(look.WARN_TEXT),
            )

    @staticmethod
    def _picture(canvas: Canvas, area, world, mask: np.ndarray) -> None:
        """The map, as large as fits the area at the world's own shape."""
        left, top, width, height = area
        rows, cols = mask.shape
        scale = min(width / cols, height / rows)
        size = (cols * scale, rows * scale)
        origin = (left + (width - size[0]) / 2, top + (height - size[1]) / 2)
        canvas.image_scaled(_pixels(mask), origin, scale)
        canvas.rect_outline((*origin, *size), rgba(look.NAVY2), width=1, radius=0)
        sx, sy = size[0] / world.size[0], size[1] / world.size[1]
        for robot in world.robots[:1]:
            x, y = robot.pose.translation - np.array(world.origin)
            canvas.circle(
                (origin[0] + x * sx, origin[1] + size[1] - y * sy),
                5,
                fill=rgba(look.TEAL),
            )
        for task in world.tasks[:1]:
            x, y = task.goal.translation - np.array(world.origin)
            canvas.circle(
                (origin[0] + x * sx, origin[1] + size[1] - y * sy),
                5,
                fill=rgba(look.AMBER),
            )

    def _labels(self, canvas: Canvas) -> None:
        """The words beside the sliders and the seed beside Shuffle."""
        face = font.regular()
        row = self.pair.y
        seed = f"map {self.recipe['seed']}"
        if self.recipe["pair"]:
            seed += f", start and goal {self.recipe['pair_seed']}"
        canvas.text(
            face,
            seed,
            (self.x + MARGIN + 308, text_baseline(self.buttons["shuffle"].y, 36, 12)),
            size=12,
            color=rgba(look.MUTED),
        )
        if self.radius.visible:
            canvas.text(
                face,
                f"robot radius {self.radius.value:.2f} m",
                (self.x + 330, text_baseline(row, 32, 12)),
                size=12,
                color=rgba(look.NAVY),
            )
        if self.apart.visible:
            canvas.text(
                face,
                f"start and goal at least {self.apart.value:.0%} apart",
                (self.x + 210, text_baseline(row + 32, 32, 12)),
                size=12,
                color=rgba(look.NAVY),
            )
        if self.loops.visible:
            canvas.text(
                face,
                f"loops {self.loops.value:.2f}",
                (self.x + 566, text_baseline(row, 32, 12)),
                size=12,
                color=rgba(look.NAVY),
            )

    # -- the pointer --
    def _tile_at(self, x: float, y: float) -> str | None:
        """The style whose tile is under a point."""
        for name, (left, top, width, height) in self.tiles.items():
            if left <= x < left + width and top <= y < top + height:
                return name
        return None

    def on_press(self, event: Event) -> None:
        """Choose a style, or send the press to the control under it."""
        name = self._tile_at(event.x, event.y)
        if name is not None:
            self.recipe["style"] = name
            self._refresh()
            return
        self._pressed = next(
            (c for c in self.children if c.visible and c.contains(event.x, event.y)),
            None,
        )
        if self._pressed is not None:
            self._pressed.on_press(event)

    def on_release(self, event: Event) -> None:
        """Finish a click; a slider is read when it is let go, so a drag draws once."""
        control, self._pressed = self._pressed, None
        if control is None:
            return
        control.on_release(event)
        if control is self.radius:
            self.recipe["radius"] = round(self.radius.value, 2)
        elif control is self.apart:
            self.recipe["apart"] = round(self.apart.value, 2)
        elif control is self.loops:
            self.recipe["loops"] = round(self.loops.value, 2)
        self._refresh()

    def on_move(self, event: Event) -> None:
        """Drag a slider that is held, and light the button under the pointer."""
        if self._pressed is not None:
            self._pressed.on_move(event)
            self.invalidate()
            return
        for child in self.children:
            inside = child.visible and child.contains(event.x, event.y)
            if inside != child.hover:
                child.on_hover(inside)
