"""Tests for the quadtree."""

import numpy as np
import pytest

from demuvo.sim.quadtree import Quadtree


def test_query_finds_only_overlapping_boxes():
    """Query finds only overlapping boxes."""
    tree = Quadtree((0, 0, 10, 10))
    tree.insert("a", (1, 1, 2, 2))
    tree.insert("b", (8, 8, 9, 9))
    assert set(tree.query((0, 0, 3, 3))) == {"a"}
    assert set(tree.query((0, 0, 10, 10))) == {"a", "b"}


def test_remove_and_replace():
    """Remove and replace."""
    tree = Quadtree((0, 0, 10, 10))
    tree.insert("a", (1, 1, 2, 2))
    tree.insert("a", (8, 8, 9, 9))
    assert len(tree) == 1
    assert set(tree.query((0, 0, 3, 3))) == set()
    tree.remove("a")
    assert len(tree) == 0
    with pytest.raises(KeyError):
        tree.remove("a")


def test_a_box_outside_the_area_is_kept():
    """A box outside the area is kept."""
    tree = Quadtree((0, 0, 10, 10))
    tree.insert("far", (20, 20, 21, 21))
    assert set(tree.query((19, 19, 22, 22))) == {"far"}


def test_it_matches_a_plain_scan_on_random_boxes():
    """It matches a plain scan on random boxes."""
    rng = np.random.default_rng(1)
    tree = Quadtree((0, 0, 100, 100), capacity=4)
    boxes = {}
    for i in range(300):
        x, y = rng.uniform(0, 95, 2)
        w, h = rng.choice([0.5, 3, 40], 2)
        boxes[i] = (x, y, x + w, y + h)
        tree.insert(i, boxes[i])
    for i in range(0, 300, 3):
        tree.remove(i)
        del boxes[i]
    for _ in range(100):
        x, y = rng.uniform(0, 90, 2)
        q = (x, y, x + rng.uniform(0, 20), y + rng.uniform(0, 20))
        expect = {
            k
            for k, b in boxes.items()
            if b[0] <= q[2] and q[0] <= b[2] and b[1] <= q[3] and q[1] <= b[3]
        }
        assert set(tree.query(q)) == expect
