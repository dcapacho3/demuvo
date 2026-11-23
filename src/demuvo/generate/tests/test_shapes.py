"""Tests for turning a grid of blocked cells into named obstacles."""

import numpy as np

from demuvo.generate import grid_obstacles


def one_block():
    """A grid with one rectangle of cells, three by two, away from the edges."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 3:6] = True
    return mask


def test_a_rectangle_of_cells_becomes_one_box_in_the_world():
    """The commonest region there is, and it keeps its size and its place."""
    found = grid_obstacles(one_block(), 0.1, (0.0, 0.0))
    assert len(found) == 1
    assert found[0].name == "field 1"
    assert found[0].shape.name == "box"
    assert found[0].shape.size == (0.3, 0.2)
    assert tuple(round(v, 3) for v in found[0].pose.xyt) == (0.45, 0.5, 0.0)


def test_the_origin_moves_the_obstacles_with_the_world():
    """Cell (0, 0) has its corner at the origin, wherever that is."""
    found = grid_obstacles(one_block(), 0.1, (-5.0, -3.0))
    assert tuple(round(v, 3) for v in found[0].pose.xyt) == (-4.55, -2.5, 0.0)


def test_an_empty_grid_gives_no_obstacles():
    """Nothing blocked is nothing to draw."""
    assert not grid_obstacles(np.zeros((10, 10), dtype=bool), 0.1, (0.0, 0.0))


def test_the_regions_are_named_in_order_from_the_number_given():
    """A caller that adds to a world starts the numbering where it likes."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 6:9] = True
    found = grid_obstacles(mask, 0.1, (0.0, 0.0), prefix="rock", first=4)
    assert [o.name for o in found] == ["rock 4", "rock 5"]


def test_nothing_is_dropped_as_a_speck():
    """Each style cleans its own grid, so a single cell is still an obstacle."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    found = grid_obstacles(mask, 0.1, (0.0, 0.0))
    assert len(found) == 1 and found[0].shape.size == (0.1, 0.1)


def test_a_region_whose_outline_would_leave_the_world_is_tiled_with_boxes():
    """Straightening a staircase moves a corner out; the bounds keep it in.

    The region is a wedge cut off flat along the bottom of the grid, which is
    what a blob the border of a world cut off looks like. Its flanks are fitted
    as straight lines and extended until they meet, and at this angle they meet
    1.3 m past the edge. With no bounds it becomes that polygon; with the
    world's bounds it becomes boxes of whole cells, which cannot leave.
    """
    mask = np.zeros((30, 40), dtype=bool)
    for row in range(10):
        mask[row, max(0, 16 - row * 3) : min(40, 24 + row * 6)] = True
    bounds = (0.0, 0.0, 4.0, 3.0)
    loose = grid_obstacles(mask, 0.1, (0.0, 0.0))
    kept = grid_obstacles(mask, 0.1, (0.0, 0.0), bounds=bounds)
    assert [o.shape.name for o in loose] == ["polygon"]
    assert any(
        not (0.0 <= x <= 4.0 and 0.0 <= y <= 3.0)
        for o in loose
        for x, y in o.shape.points(o.pose)
    )
    assert all(
        0.0 <= x <= 4.0 and 0.0 <= y <= 3.0
        for o in kept
        for x, y in o.shape.points(o.pose)
    )
    assert all(o.shape.name == "box" for o in kept)


def test_asking_for_boxes_traces_no_outline_at_all():
    """A round region that would be one polygon becomes a stack of boxes."""
    rows, cols = np.mgrid[0:20, 0:20]
    mask = (rows - 10) ** 2 + (cols - 10) ** 2 <= 36
    polygons = grid_obstacles(mask, 0.1, (0.0, 0.0))
    boxes = grid_obstacles(mask, 0.1, (0.0, 0.0), pieces="boxes")
    assert len(polygons) == 1 and polygons[0].shape.name == "polygon"
    assert len(boxes) > 1 and all(o.shape.name == "box" for o in boxes)
