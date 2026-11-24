"""Tests for the grid helpers that every style and the mesh cut share."""

# pylint: disable=protected-access  # the tests reach the steps a trace is made of

import math

import numpy as np
import pytest

from demuvo.world.regions import (
    _blunt,
    _exact,
    _loops,
    _outline,
    _shape,
    _square_up,
    _strays,
    _weld,
    boxes,
    components,
    drop_small,
    drop_thin,
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
    """The smallest box round a diagonal band reaches a cell past the edge it keeps.

    Without bounds the band is that one turned box, which is the tidier shape.
    With them the box is refused and the band is traced as the polygon it is,
    which stays where its cells are.
    """
    mask = np.zeros((30, 30), dtype=bool)
    for i in range(12):
        mask[1 + i // 2 : 3 + i // 2, 1 + i * 2 : 3 + i * 2] = True
    loose, _ = vectorize(mask, (0.0, 0.0), 0.1, min_cells=1)
    kept, _ = vectorize(
        mask, (0.0, 0.0), 0.1, min_cells=1, bounds=(0.08, 0.0, 3.0, 3.0)
    )
    assert [s["shape"]["name"] for s in loose] == ["box"]
    assert [s["shape"]["name"] for s in kept] == ["polygon"]
    assert all(
        0.08 <= xy[0] <= 3.0 and 0.0 <= xy[1] <= 3.0
        for s in kept
        for xy in s["shape"]["vertices"]
    )


def _band(angle: float, length: float = 3.0, thick: float = 0.3):
    """A grid of 0.05 m cells with a band of the given turn, and its four corners."""
    rows, cols = np.mgrid[0:140, 0:140]
    x, y = (cols + 0.5) * 0.05 - 3.5, (rows + 0.5) * 0.05 - 3.5
    along = x * math.cos(angle) + y * math.sin(angle)
    across = -x * math.sin(angle) + y * math.cos(angle)
    mask = (abs(along) <= length / 2) & (abs(across) <= thick / 2)
    turn = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    around = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length / 2, thick / 2]
    return mask, around @ turn.T + 3.5


@pytest.mark.parametrize("degrees", [0, 15, 31, 60, 100, 135, 170])
def test_a_band_at_any_turn_is_one_box_that_lies_along_it(degrees):
    """The box follows the band: its corners are the band's, whatever the turn.

    The long side is the one along the band, not the one across it, and a turn
    past a quarter or a half circle must not swap the two.
    """
    mask, corners = _band(math.radians(degrees))
    (found,), _ = vectorize(mask, (0.0, 0.0), 0.05, min_cells=1)
    assert found["shape"]["name"] == "box"
    got = _outline(found)
    # every corner of the band is near a corner of the box
    for corner in corners:
        assert np.hypot(*(got - corner).T).min() < 0.12


def test_a_traced_outline_closing_on_its_first_corner_does_not_repeat_it():
    """The polygon is open: its last corner is not its first again."""
    shape = np.array([[0, 0], [10, 0], [10, 5], [5, 8], [0, 5], [0, 0]], dtype=float)
    found = _shape(shape, (0.0, 0.0), 0.1)
    assert found is not None and len(found["shape"]["vertices"]) == 5


def test_a_near_level_wall_is_made_level_and_put_on_the_cell_boundary():
    """Edges within a few percent of an axis lie on it; a clear slope stays."""
    pts = [(0, 0.1), (10, 0.3), (10, 5), (0, 5.05)]
    out = _square_up(pts)
    assert out[0][1] == out[1][1] == 0.0
    assert out[2][1] == out[3][1] == 5.0
    assert out[0][0] == out[3][0] == 0.0 and out[1][0] == out[2][0] == 10.0
    slanted = _square_up([(0, 0), (10, 3), (10, 5), (0, 5)])
    assert slanted[1][1] == 3.0


def test_a_cut_line_that_is_straight_already_keeps_its_place():
    """The cut that splits a region round a hole is the same on both sides of it."""
    out = _square_up([(5, 0), (5, 4), (5.2, 8), (0, 8), (0, 0)])
    assert out[0][0] == out[1][0] == 5.0


def test_a_thin_tip_is_cut_off_and_a_square_corner_is_not():
    """A corner under thirty degrees loses its point; a square one is kept."""
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert len(_blunt(square)) == 4
    sliver = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 1.0]])
    cut = _blunt(sliver)
    assert len(cut) == 4
    assert cut[:, 0].min() > 0.0  # the tip at the origin is gone


def test_a_notch_in_a_free_gap_is_not_cut():
    """A reflex corner points into free space; cutting it would add wall there."""
    notch = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [5.5, 10.0],
            [5.0, 0.5],
            [4.5, 10.0],
            [0.0, 10.0],
        ]
    )
    assert len(_blunt(notch)) == 7


def test_a_degenerate_corner_with_no_length_is_left_alone():
    """Two corners on one point make no angle, so none is cut."""
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert len(_blunt(pts)) == 4


def _loop_of(mask):
    return _loops(mask)[0]


def test_a_wall_with_a_lone_nick_is_traced_cell_for_cell():
    """Long straight edges stay as the cells have them, a step of one cell and all."""
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 5:35] = True
    mask[8:15, 4] = True
    corners = _exact(_loop_of(mask))
    assert corners is not None and len(corners) == 6


def test_a_staircase_is_not_a_wall():
    """Steps in a row are a slope; fitting a line to them is what they need."""
    mask = np.zeros((30, 30), dtype=bool)
    for i in range(12):
        mask[i : i + 3, 2 * i : 2 * i + 4] = True
    assert _exact(_loop_of(mask)) is None


def test_the_cells_an_outline_covers_beyond_its_region_are_counted():
    """Only a cell two or more from the region counts; the edge cell is its own."""
    crop = np.zeros((10, 10), dtype=bool)
    crop[3:7, 3:7] = True
    snug = np.array([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]])
    assert _strays(snug, crop, 0, 0) == 0
    loose = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert _strays(loose, crop, 0, 0) == 100 - 6 * 6


def _table(vertices):
    return {"shape": {"name": "polygon", "vertices": vertices}}


def test_a_corner_a_little_off_its_sister_piece_is_put_on_it():
    """The two sides of a cut meet along one line."""
    left = _table([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    right = _table([[1.04, 0.0], [2.0, 0.0], [2.0, 1.0], [1.04, 1.0]])
    tables = [left, right]
    _weld(tables, [0, 0], 0.08)
    assert tables[0]["shape"]["vertices"][1][0] == tables[1]["shape"]["vertices"][0][0]


def test_pieces_of_different_regions_are_not_welded():
    """Only pieces cut from one region belong together."""
    left = _table([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    right = _table([[1.04, 0.0], [2.0, 0.0], [2.0, 1.0], [1.04, 1.0]])
    _weld([left, right], [0, 1], 0.08)
    assert right["shape"]["vertices"][0][0] == 1.04


def test_a_move_that_would_spoil_the_polygon_is_not_made():
    """A corner that would land on its neighbour stays where it is."""
    thin = _table([[0.0, 0.0], [0.05, 0.0], [0.0, 1.0]])
    wall = _table([[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]])
    _weld([thin, wall], [0, 0], 0.1)
    assert thin["shape"]["vertices"] == [[0.0, 0.0], [0.05, 0.0], [0.0, 1.0]]


def test_a_box_piece_can_be_a_sister_but_is_not_moved():
    """Boxes are cells; a polygon next to one moves to it and the box stays."""
    box = {"pose": [0.5, 0.5, 0.0], "shape": {"name": "box", "size": [1.0, 1.0]}}
    poly = _table([[1.04, 0.0], [2.0, 0.0], [2.0, 1.0], [1.04, 1.0]])
    _weld([box, poly], [0, 0], 0.08)
    assert box["pose"] == [0.5, 0.5, 0.0]
    assert poly["shape"]["vertices"][0][0] == pytest.approx(1.0)


def test_a_line_of_cells_one_wide_is_no_wall_and_a_block_is():
    """The cleaning erases traces of the grid and leaves what stands wide enough."""
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:6, 2:6] = True
    mask[8, 1:11] = True
    mask[0:12, 11] = True
    drop_thin(mask, 2)
    assert mask[2:6, 2:6].all()
    assert not mask[8].any() and not mask[:, 11].any()


def test_a_width_of_one_changes_nothing():
    """A style that draws walls one cell thick asks for 1 and keeps them."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[3, :] = True
    drop_thin(mask, 1)
    assert mask[3].all()


def test_a_box_too_thin_to_read_back_is_refused_and_not_raised():
    """A sliver whose box side rounds to zero gives None, so the caller uses boxes."""
    sliver = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 1e-4], [0.0, 1e-4]])
    assert _shape(sliver, (0.0, 0.0), 1.0, (0.0, 0.0, 10.0, 10.0)) is None
