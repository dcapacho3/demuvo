"""Undo and redo - the worlds the session has had, most recent last.

Every edit that works puts the world it replaced on the undo list. A drag makes
many edits; between `begin` and `end` only the first is kept, so one undo takes the
whole drag back.
"""

from __future__ import annotations

from ..world import World

# The most worlds kept; older ones are dropped.
LIMIT = 200


class History:
    """The undo and redo lists of a session.

    Args:
        session: the session whose edits are kept.
    """

    def __init__(self, session) -> None:
        """Start with nothing to undo."""
        self.session = session
        self._past: list[World] = []
        self._future: list[World] = []
        self._grouped = False
        self._kept = False

    @property
    def can_undo(self) -> bool:
        """True if there is an edit to take back."""
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        """True if an undone edit can be made again."""
        return bool(self._future)

    def remember(self, world: World) -> None:
        """Keep the world an edit replaced, and forget what could be redone."""
        if self._grouped and self._kept:
            return
        self._past.append(world)
        del self._past[:-LIMIT]
        self._future.clear()
        self._kept = self._grouped

    def begin(self) -> None:
        """Start a group of edits that one undo takes back."""
        self._grouped, self._kept = True, False

    def end(self) -> None:
        """End the group."""
        self._grouped = False

    def clear(self) -> None:
        """Forget everything, for instance after another world was opened."""
        self._past.clear()
        self._future.clear()

    def undo(self) -> str | None:
        """Go back to the world before the last edit; None if it worked."""
        return self._go(self._past, self._future, "undo")

    def redo(self) -> str | None:
        """Make the last undone edit again; None if it worked."""
        return self._go(self._future, self._past, "redo")

    def _go(self, take: list[World], give: list[World], word: str) -> str | None:
        """Move one world from one list to the other and make it the world."""
        session = self.session
        if (why := session.guard()) is not None:
            return why
        if not take:
            return f"Nothing to {word}."
        current = session.world
        message = session.apply(take[-1], rebuild=True, select="", remember=False)
        if message is not None:
            return message
        take.pop()
        give.append(current)
        session.scene.selected_obstacle = None
        return None
