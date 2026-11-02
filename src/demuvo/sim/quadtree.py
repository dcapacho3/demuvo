"""A quadtree of boxes - the broad phase for static objects.

The tree splits a fixed area into four squares, again and again where boxes
crowd.
Three calls: `insert`, `remove` and `query`.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator

Box = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def _overlap(a: Box, b: Box) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


class _Node:
    """One square of the tree and the boxes kept in it."""

    __slots__ = ("box", "depth", "items", "kids")

    def __init__(self, box: Box, depth: int) -> None:
        self.box = box
        self.depth = depth
        self.items: dict[Hashable, Box] = {}
        self.kids: list[_Node] | None = None

    def kid_for(self, box: Box) -> _Node | None:
        """The child square that wholly contains the box, if there is one."""
        if self.kids is None:
            return None
        for kid in self.kids:
            k = kid.box
            if k[0] <= box[0] and k[1] <= box[1] and box[2] <= k[2] and box[3] <= k[3]:
                return kid
        return None

    def split(self) -> None:
        """Make the four child squares."""
        x0, y0, x1, y1 = self.box
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        d = self.depth + 1
        self.kids = [
            _Node((x0, y0, mx, my), d),
            _Node((mx, y0, x1, my), d),
            _Node((x0, my, mx, y1), d),
            _Node((mx, my, x1, y1), d),
        ]


class Quadtree:
    """Boxes in a fixed area, found by the box that overlaps them.

    A box that lies partly or wholly outside the area is kept in the root, so
    nothing is lost.

    Args:
        bounds: the area as (xmin, ymin, xmax, ymax).
        capacity: how many boxes a node holds before it splits.
        max_depth: how many times the area may be halved.
    """

    def __init__(self, bounds: Box, capacity: int = 8, max_depth: int = 8) -> None:
        """Make an empty tree over the area."""
        self._root = _Node(bounds, 0)
        self._capacity = capacity
        self._max_depth = max_depth
        self._home: dict[Hashable, _Node] = {}

    def __len__(self) -> int:
        """The number of boxes in the tree."""
        return len(self._home)

    def insert(self, name: Hashable, box: Box) -> None:
        """Add a box under a name; a name already in the tree is replaced."""
        if name in self._home:
            self.remove(name)
        node = self._root
        while (kid := node.kid_for(box)) is not None:
            node = kid
        node.items[name] = box
        self._home[name] = node
        if len(node.items) > self._capacity and node.depth < self._max_depth:
            self._split(node)

    def remove(self, name: Hashable) -> None:
        """Take a box out; a name not in the tree raises KeyError."""
        del self._home.pop(name).items[name]

    def query(self, box: Box) -> Iterator[Hashable]:
        """Names of the boxes that overlap the box, edges included."""
        stack = [self._root]
        while stack:
            node = stack.pop()
            for name, held in node.items.items():
                if _overlap(held, box):
                    yield name
            if node.kids:
                stack.extend(kid for kid in node.kids if _overlap(kid.box, box))

    def _split(self, node: _Node) -> None:
        """Halve a crowded node and push down the boxes that fit in one half."""
        if node.kids is None:
            node.split()
        for name, held in list(node.items.items()):
            kid = node.kid_for(held)
            if kid is not None:
                del node.items[name]
                kid.items[name] = held
                self._home[name] = kid
        for kid in node.kids:
            if len(kid.items) > self._capacity and kid.depth < self._max_depth:
                self._split(kid)
