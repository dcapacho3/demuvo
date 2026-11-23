"""Tests for the grid helpers that every style and the mesh cut share."""

import numpy as np

from demuvo.world.regions import (
    boxes,
    components,
    drop_small,
    eroded,
    fill_holes,
    fits,
    label_regions,
    runs,
    vectorize,
)


def test_the_runs_of_a_row_are_its_stretches_of_marked_cells():
    """The ends are the first and the one past the last, as a slice has them."""
    row = np.array([False, True, True, False, True, False])
    assert runs(row) == [(1, 3), (4, 5)]
    assert not runs(np.zeros(4, dtype=bool))
    assert runs(np.ones(4, dtype=bool)) == [(0, 4)]


def test_cells_that_touch_along_an_edge_are_one_region_and_at_a_corner_are_two():
    """A polygon pinched to a point is no polygon, so corners do not join."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = mask[2, 2] = True
    found, region = label_regions(mask)
    assert len({region[i] for row in found for _, _, i in row}) == 2
    mask[1, 2] = True  # now they are joined along an edge
    found, region = label_regions(mask)
    assert len({region[i] for row in found for _, _, i in row}) == 1


def test_a_region_comes_back_cropped_with_the_corner_it_sits_at():
    """So a caller works on the few cells of a region and not the whole grid."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 7:9] = True
    (crop, row, col) = components(mask)[0]
    assert crop.shape == (2, 2) and crop.all() and (row, col) == (4, 7)


def test_erosion_takes_a_step_off_every_side():
    """A cell is kept while it is more than the steps from outside the mask."""
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    assert eroded(mask, 1).sum() == 9  # a five by five becomes a three by three
    assert eroded(mask, 2).sum() == 1
    assert not eroded(mask, 3).any()


def test_a_shut_in_gap_is_filled_and_a_room_is_left_alone():
    """The inside of a shelf is solid; a room is still a room."""
    grid = np.zeros((20, 20), dtype=bool)
    grid[2:6, 2:6] = True
    grid[3:5, 3:5] = False  # four free cells, walled in
    fill_holes(grid, 10, 0)
    assert grid[2:6, 2:6].all()
    big = np.zeros((20, 20), dtype=bool)
    big[2:18, 2:18] = True
    big[4:16, 4:16] = False  # a room of 144 cells, too large to fill
    fill_holes(big, 10, 0)
    assert not big[4:16, 4:16].any()


def test_a_region_of_too_few_cells_is_cleared():
    """The mirror of filling a gap: the specks themselves go."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    mask[5:8, 5:8] = True
    assert drop_small(mask, 2) == 1
    assert not mask[1, 1] and mask[5:8, 5:8].all()
    assert drop_small(mask, 0) == 0  # nothing is too small for nothing


def test_a_grid_is_covered_by_rectangles_that_join_up_the_rows():
    """Rows with the same run become one box, so a rectangle is one box."""
    grid = np.zeros((6, 6), dtype=bool)
    grid[1:4, 2:5] = True
    assert boxes(grid) == [(2, 5, 1, 4)]


def test_a_table_a_world_file_would_refuse_does_not_fit():
    """A region so thin its box rounds to a side of zero is no shape at all."""
    flat = {"pose": [1.0, 1.0, 0.0], "shape": {"name": "box", "size": [0.5, 0.0]}}
    assert not fits(flat, None)
    assert not fits(flat, (0.0, 0.0, 4.0, 4.0))


def test_a_shape_outside_the_bounds_does_not_fit_and_one_inside_does():
    """The question the loader asks of a saved file, asked before it is saved."""
    table = {"pose": [1.0, 1.0, 0.0], "shape": {"name": "box", "size": [0.5, 0.5]}}
    assert fits(table, (0.0, 0.0, 4.0, 4.0))
    assert not fits(table, (0.0, 0.0, 1.1, 4.0))
    polygon = {"shape": {"name": "polygon", "vertices": [[1, 1], [2, 1], [2, 2]]}}
    assert fits(polygon, (0.0, 0.0, 4.0, 4.0))
    assert not fits(polygon, (0.0, 0.0, 1.5, 4.0))
    assert fits(polygon, None)


def test_a_speck_is_dropped_and_counted_unless_the_caller_wants_it():
    """The mesh cut drops specks; a style that has cleaned its grid keeps them."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    shapes, dropped = vectorize(mask, (0.0, 0.0), 0.1)
    assert shapes == [] and dropped == 1
    shapes, dropped = vectorize(mask, (0.0, 0.0), 0.1, min_cells=1)
    assert len(shapes) == 1 and dropped == 0


def test_a_box_that_would_leave_the_bounds_gives_way_to_the_polygon():
    """The smallest box round a diagonal band reaches past the corner of the grid.

    Without bounds the band is that one turned box, which is the tidier shape.
    With them the box is refused and the band is traced as the polygon it is,
    which stays where its cells are.
    """
    mask = np.zeros((30, 30), dtype=bool)
    for i in range(12):
        mask[1 + i // 2 : 3 + i // 2, 1 + i * 2 : 3 + i * 2] = True
    loose, _ = vectorize(mask, (0.0, 0.0), 0.1, min_cells=1)
    kept, _ = vectorize(mask, (0.0, 0.0), 0.1, min_cells=1, bounds=(0.0, 0.0, 3.0, 3.0))
    assert [s["shape"]["name"] for s in loose] == ["box"]
    assert [s["shape"]["name"] for s in kept] == ["polygon"]
    assert all(
        0.0 <= v <= 3.0 for s in kept for xy in s["shape"]["vertices"] for v in xy
    )
