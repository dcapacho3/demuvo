"""Tests for cutting meshes at a robot's height and turning the cut into obstacles."""

# pylint: disable=protected-access  # some tests check the module's own steps

import math

import numpy as np
import pytest

from demuvo.world import footprint
from demuvo.world.footprint import slice_shapes
from demuvo.world.meshes import is_closed

LOW, HIGH, CELL = 0.1, 1.0, 0.05


def quad(a, b, c, d):
    """Two triangles for four corners."""
    return [[a, b, c], [a, c, d]]


def wall(p, q, z0=0.0, z1=2.0):
    """A vertical wall of two triangles between two points of the floor."""
    return quad((*p, z0), (*q, z0), (*q, z1), (*p, z1))


def walls(points, z0=0.0, z1=2.0):
    """The walls round a closed outline of points: an open surface, no top or floor."""
    return np.array(
        [
            t
            for i in range(len(points))
            for t in wall(points[i], points[(i + 1) % len(points)], z0, z1)
        ],
        dtype=float,
    )


def prism(points, z0=0.0, z1=2.0):
    """A closed solid: walls, a floor and a ceiling over a convex outline."""
    flat = [(*p, z0) for p in points]
    top = [(*p, z1) for p in points]
    caps = [[flat[0], flat[i], flat[i + 1]] for i in range(1, len(points) - 1)]
    caps += [[top[0], top[i + 1], top[i]] for i in range(1, len(points) - 1)]
    return np.concatenate([walls(points, z0, z1), np.array(caps, dtype=float)])


def cut(*shapes, hole=2.5, narrow=0.6):
    """The obstacles of the pieces, each a (triangles, closed) pair."""
    return slice_shapes(
        list(shapes), (LOW, HIGH), CELL, largest_hole=hole, narrowest=narrow
    )


def box_of(shape):
    """The (x0, y0, x1, y1) a shape reaches, for a box or a polygon."""
    if "vertices" in shape["shape"]:
        xs, ys = zip(*shape["shape"]["vertices"])
        return min(xs), min(ys), max(xs), max(ys)
    x, y, yaw = shape["pose"]
    w, d = shape["shape"]["size"]
    half = (abs(w * math.cos(yaw)) + abs(d * math.sin(yaw))) / 2
    other = (abs(w * math.sin(yaw)) + abs(d * math.cos(yaw))) / 2
    return x - half, y - other, x + half, y + other


def test_nothing_between_the_heights_gives_nothing():
    """A roof high up and a floor at the bottom are not cut."""
    roof = np.array(quad((0, 0, 3), (4, 0, 3), (4, 4, 3), (0, 4, 3)), dtype=float)
    floor = np.array(quad((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)), dtype=float)
    assert cut((roof, False), (floor, False)) == ([], 0)
    assert cut() == ([], 0)


def test_a_closed_solid_becomes_one_box_with_the_size_of_its_footprint():
    """A 2 by 1 slab standing on the floor gives a box there, none for its top."""
    solid = prism([(1, 1), (3, 1), (3, 2), (1, 2)])
    assert is_closed(solid)
    shapes, specks = cut((solid, True))
    (box,) = shapes
    assert specks == 0
    assert box["shape"]["name"] == "box"
    assert box["pose"][:2] == pytest.approx([2.0, 1.5], abs=0.03)
    assert box["pose"][2] == 0.0
    assert box["shape"]["size"] == pytest.approx([2.0, 1.0], abs=0.06)


def test_a_turned_slab_is_a_box_with_a_turn_and_not_a_wobbly_polygon():
    """The corners of a square turned by 30 degrees are found, as a box."""
    angle = math.radians(30)
    corners = [
        (3 + 1.2 * math.cos(angle + k * math.pi / 2) * math.sqrt(2) / 1,
         3 + 1.2 * math.sin(angle + k * math.pi / 2) * math.sqrt(2) / 1)
        for k in range(4)
    ]  # fmt: skip
    ((box),) = cut((prism(corners), True))[0]
    assert box["shape"]["name"] == "box"
    assert box["shape"]["size"] == pytest.approx([2.4, 2.4], abs=0.1)
    assert abs(math.sin(2 * box["pose"][2] - 2 * (angle + math.pi / 4))) < 0.08
    assert box["pose"][:2] == pytest.approx([3.0, 3.0], abs=0.05)


def test_a_turned_slab_whose_sides_swap_keeps_its_length_and_width():
    """A 3 by 1 slab turned past a quarter still reports 3 along its own x axis."""
    angle = math.radians(100)
    c, s = math.cos(angle), math.sin(angle)
    corners = [
        (5 + c * x - s * y, 5 + s * x + c * y)
        for x, y in ((-1.5, -0.5), (1.5, -0.5), (1.5, 0.5), (-1.5, 0.5))
    ]
    ((box),) = cut((prism(corners), True))[0]
    assert sorted(box["shape"]["size"]) == pytest.approx([1.0, 3.0], abs=0.1)
    x, y, yaw = box["pose"]
    long_side = box["shape"]["size"][0] > box["shape"]["size"][1]
    # the long side points along the slab's direction, 100 degrees, up to a half turn
    along = yaw if long_side else yaw + math.pi / 2
    assert abs(math.sin(along - angle)) < 0.08


def test_a_hexagon_is_a_polygon_with_six_corners():
    """A shape that is not near a rectangle stays a polygon."""
    corners = [
        (4 + math.cos(k * math.pi / 3), 4 + math.sin(k * math.pi / 3)) for k in range(6)
    ]
    ((shape),) = cut((prism(corners), True))[0]
    assert shape["shape"]["name"] == "polygon"
    assert len(shape["shape"]["vertices"]) == 6
    assert "pose" not in shape


def test_a_diagonal_wall_is_one_slanted_box_and_not_a_staircase():
    """A thin band along a diagonal is one box turned to follow it."""
    band = [(0.0, 0.0), (5.0, 3.0), (5.0, 3.4), (0.0, 0.4)]
    (shape,), _ = cut((prism(band), True))
    assert shape["shape"]["name"] == "box"
    assert shape["pose"][2] == pytest.approx(math.atan2(3, 5), abs=0.03)
    assert 5.8 <= max(shape["shape"]["size"]) <= 6.3


def test_a_bent_wall_is_one_polygon_with_few_corners():
    """Two slanted runs make a chevron with a handful of corners, not one a cell."""
    chevron = [(0, 0), (3, 2), (6, 0), (6, 0.4), (3, 2.4), (0, 0.4)]
    (shape,), _ = cut((prism(chevron), True))
    assert shape["shape"]["name"] == "polygon"
    assert 5 <= len(shape["shape"]["vertices"]) <= 7


def test_a_closed_ring_keeps_its_hole_open_and_fills_between_the_skins():
    """The cut is filled between the two outlines of a ring and not inside it."""
    outer = [(0, 0), (6, 0), (6, 6), (0, 6)]
    inner = [(2, 2), (2, 4), (4, 4), (4, 2)]
    ring = np.concatenate([walls(outer), walls(inner)])
    shapes, _ = cut((ring, True))
    assert covers_centre(shapes, 1.0, 3.0) and covers_centre(shapes, 5.0, 3.0)
    assert not covers_centre(shapes, 3.0, 3.0)


def test_a_cut_across_a_ring_gives_the_segments_of_both_outlines():
    """A plane at 0.5 m crosses each of the 16 wall triangles that reach it once."""
    outer = [(0, 0), (6, 0), (6, 6), (0, 6)]
    segments = footprint._cut(walls(outer), 0.5)
    assert segments.shape == (8, 2, 2)
    grid = np.zeros((130, 130), dtype=bool)
    footprint._inside(grid, np.array([-0.5, -0.5]), segments, 0.05)
    assert grid[60, 60] and not grid[0, 0]


def test_four_walls_round_a_shelf_are_filled_and_four_walls_round_a_room_are_not():
    """A gap under 2.5 square metres is solid; a large one is a room and stays free."""
    shelf, _ = cut((walls([(0, 0), (1, 0), (1, 2), (0, 2)]), False))
    assert len(shelf) == 1 and shelf[0]["shape"]["name"] == "box"
    room, _ = cut((walls([(0, 0), (6, 0), (6, 6), (0, 6)]), False))
    assert not covers_centre(room, 3, 3)
    assert len(room) >= 2


def covers_centre(shapes, x, y):
    """Whether a point lies within some shape, by its real outline."""
    for shape in shapes:
        x0, y0, x1, y1 = box_of(shape)
        if shape["shape"]["name"] == "box":
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        else:
            pts = shape["shape"]["vertices"]
            inside = False
            for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
                if (ay > y) != (by > y) and x < ax + (y - ay) / (by - ay) * (bx - ax):
                    inside = not inside
            if inside:
                return True
    return False


def test_the_gap_between_the_two_faces_of_a_thick_wall_is_filled():
    """Two skins 0.3 m apart are one wall, though the gap between them is long."""
    outer = [(0, 0), (8, 0), (8, 8), (0, 8)]
    inner = [(0.3, 0.3), (7.7, 0.3), (7.7, 7.7), (0.3, 7.7)]
    shapes, _ = cut((np.concatenate([walls(outer), walls(inner)]), False))
    assert covers_centre(shapes, 0.15, 4.0) and covers_centre(shapes, 4.0, 0.15)
    assert not covers_centre(shapes, 4.0, 4.0)


def test_a_speck_of_fewer_than_four_cells_is_dropped_and_counted():
    """A tiny wall adds nothing, and the count says it was dropped."""
    tiny = walls([(0, 0), (0.02, 0), (0.02, 0.02), (0, 0.02)])
    big = prism([(3, 3), (4, 3), (4, 4), (3, 4)])
    shapes, specks = cut((tiny, False), (big, True))
    assert specks >= 1 and len(shapes) == 1


def test_a_mesh_too_wide_for_the_grid_is_refused(monkeypatch):
    """The message gives its size and the cell."""
    monkeypatch.setattr(footprint, "MAX_CELLS", 100)
    with pytest.raises(ValueError, match="too large for cells of 0.05 m"):
        cut((prism([(0, 0), (9, 0), (9, 9), (0, 9)]), True))


def test_a_region_a_polygon_cannot_hold_becomes_boxes(monkeypatch):
    """With room for only two corners, a polygon is refused and boxes are made."""
    monkeypatch.setattr(footprint, "MAX_CORNERS", 2)
    corners = [
        (4 + math.cos(k * math.pi / 3), 4 + math.sin(k * math.pi / 3)) for k in range(6)
    ]
    shapes, _ = cut((prism(corners), True))
    assert len(shapes) > 1 and all(s["shape"]["name"] == "box" for s in shapes)


def test_a_region_with_a_hole_that_cannot_be_cut_apart_becomes_boxes(monkeypatch):
    """Without cuts to spend, a thick ring falls back to boxes that leave the hole."""
    monkeypatch.setattr(footprint, "MAX_CUTS", 0)
    outer = [(0, 0), (6, 0), (6, 6), (0, 6)]
    inner = [(1.5, 1.5), (4.5, 1.5), (4.5, 4.5), (1.5, 4.5)]
    shapes, _ = cut((np.concatenate([walls(outer), walls(inner)]), True))
    assert all(s["shape"]["name"] == "box" for s in shapes)
    assert not covers_centre(shapes, 3.0, 3.0)
    assert covers_centre(shapes, 0.5, 3.0)


def test_a_ring_that_is_cut_apart_leaves_no_hole_in_any_piece():
    """The pieces together still leave the middle free."""
    outer = [(0, 0), (6, 0), (6, 6), (0, 6)]
    inner = [(1.5, 1.5), (4.5, 1.5), (4.5, 4.5), (1.5, 4.5)]
    shapes, _ = cut((np.concatenate([walls(outer), walls(inner)]), True))
    assert not covers_centre(shapes, 3.0, 3.0)
    assert covers_centre(shapes, 0.5, 3.0) and covers_centre(shapes, 5.5, 3.0)
    assert covers_centre(shapes, 3.0, 0.5) and covers_centre(shapes, 3.0, 5.5)


def test_the_lines_fitted_to_a_run_find_a_straight_chain_and_a_closed_one():
    """Douglas-Peucker keeps the ends of a straight chain, and copes with a loop."""
    line = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0], [3.0, 0.0]])
    assert footprint._chain(line, 0.5) == [0, 3]
    assert footprint._chain(line, 0.01) == [0, 1, 2, 3]
    loop = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    assert footprint._chain(loop, 0.1) == [0, 1, 2]


def test_a_polygon_with_fewer_than_three_corners_is_not_a_shape():
    """A degenerate outline gives None."""
    assert (
        footprint._shape(np.array([[0.0, 0.0], [1.0, 0.0]]), (0.0, 0.0), 0.05) is None
    )


def test_a_polygon_that_touches_itself_is_refused():
    """A bow tie is not simple, so the shape is not made."""
    bow = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]) * 4
    assert footprint._shape(bow, (0.0, 0.0), 0.05) is None


def test_the_hull_of_few_points_is_the_points_themselves():
    """Two points have no hull to speak of."""
    assert len(footprint._hull(np.array([[0.0, 0.0], [1.0, 1.0]]))) == 2


def test_a_box_of_no_area_and_a_ragged_shape_are_not_turned_boxes():
    """A line has no box; a shape with a deep notch does not fill the box."""
    assert (
        footprint._turned_box(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 0.1)
        is None
    )
    notch = np.array([[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]], dtype=float)
    assert footprint._turned_box(notch, 0.1) is None
    assert footprint._turned_box(np.array([[0.0, 0.0], [1.0, 1.0]]), 0.1) is None
    # fills 92.5 percent of its box, but one corner is 1.5 m inside it
    dent = np.array([[0, 0], [10, 0], [10, 10], [5, 8.5], [0, 10]], dtype=float)
    assert footprint._turned_box(dent, 0.1) is None
