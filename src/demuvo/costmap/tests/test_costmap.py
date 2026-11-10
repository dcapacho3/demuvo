"""Tests for the costmap - cells, layers and building."""

import dataclasses
import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import load_world
from demuvo.costmap import (
    FREE,
    INSCRIBED,
    LETHAL,
    NO_INFORMATION,
    Costmap,
    build_costmaps,
    inscribed_radius,
)
from demuvo.costmap.layers import generate_static, inflate, load_static
from demuvo.world import WorldError
from demuvo.world.shapes import Box, Circle, Polygon

WORLD = """
version = 1

[world]
size = [4.0, 3.0]
seed = 1

[[obstacle]]
pose = [2.0, 1.5, 0.0]
shape = {{ name = "box", size = [0.5, 0.5] }}

[[obstacle]]
shape = {{ name = "linestring", vertices = [[0.5, 0.5], [1.5, 0.5]] }}

[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.1 }}
pose = [1.0, 2.5, 0.0]

[[robot]]
name = "b"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.1 }}
pose = [3.0, 2.5, 0.0]

[[robot]]
name = "c"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.2 }}
pose = [3.0, 0.5, 0.0]
{extra}
"""


def world(tmp_path, extra=""):
    """Load the test world with extra text at the end of the file."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(extra=extra))
    return load_world(path)


def test_cells_and_world_positions_match():
    """A point is in the cell that holds it; the cell centre maps back to it."""
    costmap = Costmap(np.zeros((4, 6), np.uint8), 0.5, (1.0, 2.0))
    assert costmap.world_to_cell(1.2, 2.2) == (0, 0)
    assert costmap.world_to_cell(3.9, 3.9) == (3, 5)
    assert costmap.cell_to_world(0, 0) == (1.25, 2.25)
    assert costmap.world_to_cell(*costmap.cell_to_world(2, 3)) == (2, 3)


def test_many_points_are_looked_up_at_once():
    """The array lookup agrees with the single-point one, edges and outside included."""
    costmap = Costmap(np.zeros((4, 6), np.uint8), 0.5, (1.0, 2.0))
    points = np.array([[1.2, 2.2], [3.9, 3.9], [1.5, 2.5], [0.0, 9.0]])
    rows, cols = costmap.cells(points)
    assert [(int(r), int(c)) for r, c in zip(rows, cols)] == [
        costmap.world_to_cell(*point) for point in points
    ]
    assert (rows.tolist(), cols.tolist()) == ([0, 3, 1, 14], [0, 5, 1, -2])


def test_outside_the_grid_has_no_information():
    """A point off the grid is out of bounds and its cost is unknown."""
    data = np.full((2, 2), 7, np.uint8)
    costmap = Costmap(data, 1.0)
    assert costmap.in_bounds(0.5, 1.5)
    assert not costmap.in_bounds(2.5, 0.5)
    assert not costmap.in_bounds(-0.1, 0.5)
    assert costmap.cost_at(0.5, 0.5) == 7
    assert costmap.cost_at(5.0, 5.0) == NO_INFORMATION


def test_the_static_layer_marks_the_obstacles_and_the_border(tmp_path):
    """Inside the box, along the wall and on the border is lethal; the open is free."""
    grid = generate_static(world(tmp_path), 0.1)
    assert grid.shape == (30, 40)
    costmap = Costmap(grid, 0.1)
    assert costmap.cost_at(2.0, 1.5) == LETHAL  # inside the box
    assert costmap.cost_at(1.0, 0.5) == LETHAL  # on the thin wall
    assert costmap.cost_at(0.05, 1.5) == LETHAL  # the border
    assert costmap.cost_at(3.95, 1.5) == LETHAL
    assert costmap.cost_at(1.0, 1.5) == FREE
    assert costmap.cost_at(1.0, 0.8) == FREE


def test_a_moved_world_gives_the_same_cells_at_the_moved_origin(tmp_path):
    """Moving the origin and the obstacles by the same offset moves the map."""
    base = world(tmp_path)
    shift = Transform2D.from_xyt(np.array([-2.0, 1.0, 0.0]))
    moved = dataclasses.replace(
        base,
        origin=(-2.0, 1.0),
        obstacles=[dataclasses.replace(o, pose=shift @ o.pose) for o in base.obstacles],
    )
    assert (generate_static(moved, 0.1) == generate_static(base, 0.1)).all()
    costmap = build_costmaps(moved)["a"]
    assert costmap.origin == (-2.0, 1.0)
    assert costmap.cost_at(0.0, 2.5) == LETHAL  # the box, moved from (2.0, 1.5)


def test_a_thin_wall_leaves_no_gap(tmp_path):
    """A wall of zero thickness still marks a connected line of cells."""
    grid = generate_static(world(tmp_path), 0.1)
    row = Costmap(grid, 0.1).world_to_cell(1.0, 0.5)[0]
    assert (grid[row, 5:15] == LETHAL).all()


def test_a_shape_outside_the_grid_is_skipped():
    """The marking window is clipped to the grid."""
    from demuvo.costmap.layers import _mark  # pylint: disable=import-outside-toplevel
    from demuvo.world.shapes import Part  # pylint: disable=import-outside-toplevel

    grid = np.zeros((5, 5), np.uint8)
    _mark(grid, Part(((10.0, 10.0),), 0.2), 0.1)
    assert not grid.any()


def test_a_cell_next_to_a_wall_costs_more_than_a_cell_in_the_open(tmp_path):
    """The test of the plan: inflation raises the cost near an obstacle."""
    costmap = build_costmaps(world(tmp_path))["a"]
    near = costmap.cost_at(2.0, 1.5 + 0.25 + 0.2)  # 0.2 m from the box edge
    far = costmap.cost_at(1.0, 1.5)
    assert far == FREE
    assert near > far


def test_inflation_follows_the_cost_formula():
    """Lethal 254, then 253 up to the inscribed radius, then 252 * exp(-k d)."""
    grid = np.zeros((1, 21), np.uint8)
    grid[0, 0] = LETHAL
    out = inflate(grid, 0.1, 0.25, 0.8, 5.0)[0]
    assert out[0] == LETHAL
    assert out[1] == out[2] == INSCRIBED  # 0.1 m, 0.2 m
    assert out[3] == int(252 * math.exp(-5.0 * (0.3 - 0.25)))
    assert out[8] == int(252 * math.exp(-5.0 * (0.8 - 0.25)))
    assert out[9:].max() == FREE  # beyond the inflation radius


def test_inflation_reaches_at_least_the_inscribed_radius():
    """A robot bigger than the inflation radius still gets its inscribed zone."""
    grid = np.zeros((1, 11), np.uint8)
    grid[0, 0] = LETHAL
    out = inflate(grid, 0.1, 0.5, 0.2, 10.0)[0]
    assert (out[1:6] == INSCRIBED).all()


def test_inflation_does_not_change_the_input():
    """The static array is shared, so inflation works on a copy."""
    grid = np.zeros((3, 3), np.uint8)
    grid[1, 1] = LETHAL
    inflate(grid, 0.1, 0.1, 0.3, 10.0)
    assert grid.sum() == LETHAL


def test_the_inscribed_radius_of_a_body():
    """A circle gives its radius, a box half its short side."""
    assert inscribed_radius(Circle(0.3)) == pytest.approx(0.3)
    assert inscribed_radius(Box((0.8, 0.4))) == pytest.approx(0.2)


def test_the_inscribed_radius_of_a_concave_body_uses_its_outline():
    """An L-shape: the notch corner at (0.3, 0.3) is nearest, not a cut inside it."""
    notched = Polygon(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 0.3), (0.3, 0.3), (0.3, 1.0), (-1.0, 1.0))
    )
    assert inscribed_radius(notched) == pytest.approx(0.3 * math.sqrt(2))


def test_a_body_that_does_not_hold_its_origin_has_no_inscribed_radius():
    """A triangle away from the origin: no circle about the origin fits in it."""
    away = Polygon(((1.0, 1.0), (2.0, 1.0), (1.5, 2.0)))
    assert inscribed_radius(away) == 0.0


def test_robots_of_one_radius_share_a_costmap(tmp_path):
    """Two robots of radius 0.1 get one object; the bigger robot gets its own."""
    costmaps = build_costmaps(world(tmp_path))
    assert costmaps["a"] is costmaps["b"]
    assert costmaps["a"] is not costmaps["c"]
    assert costmaps["c"].data.sum() >= costmaps["a"].data.sum()


def test_the_costmap_is_shared_and_read_only(tmp_path):
    """Robots read one array, so nothing may write to it."""
    costmap = build_costmaps(world(tmp_path))["a"]
    with pytest.raises(ValueError, match="read-only"):
        costmap.data[0, 0] = 1


def test_the_layers_can_be_chosen(tmp_path):
    """Static only gives obstacles with no spread; no static gives an empty map."""
    static_only = build_costmaps(
        world(tmp_path, '[costmap.global]\nplugins = ["static"]')
    )["a"]
    assert set(np.unique(static_only.data)) == {FREE, LETHAL}
    empty = build_costmaps(world(tmp_path, '[costmap.global]\nplugins = ["inflation"]'))
    assert not empty["a"].data.any()


def write_map(tmp_path, pixels, *, header="P5", extra="", image="m.pgm", maxval=255):
    """Write a small PGM and its yaml; return the yaml path."""
    rows, cols = len(pixels), len(pixels[0])
    if header == "P5":
        body = bytes(v for row in pixels for v in row)
        (tmp_path / image).write_bytes(
            f"P5\n# note\n{cols} {rows}\n{maxval}\n".encode() + body
        )
    else:
        text = " ".join(str(v) for row in pixels for v in row)
        (tmp_path / image).write_text(f"{header}\n{cols} {rows}\n{maxval}\n{text}\n")
    path = tmp_path / "m.yaml"
    path.write_text(
        f"image: {image}\nresolution: 0.5\norigin: [1.0, 2.0, 0.0]\n"
        f"occupied_thresh: 0.65  # comment\n{extra}\n"
    )
    return path


def test_a_map_file_becomes_cells(tmp_path):
    """Dark pixels are lethal; the image's top row is the highest y."""
    path = write_map(tmp_path, [[0, 255], [255, 255], [255, 255]])
    grid, resolution, origin = load_static(path)
    assert (resolution, origin) == (0.5, (1.0, 2.0))
    assert grid.shape == (3, 2)
    assert grid[2, 0] == LETHAL  # the top-left pixel is the last row
    assert grid.sum() == LETHAL


def test_a_text_pgm_and_negate_and_sixteen_bits(tmp_path):
    """P2 text works; `negate` makes bright pixels the obstacles."""
    path = write_map(tmp_path, [[0, 255]], header="P2", extra="negate: 1")
    grid, *_ = load_static(path)
    assert grid.tolist() == [[FREE, LETHAL]]
    wide = tmp_path / "wide"
    wide.mkdir()
    (wide / "m.pgm").write_bytes(b"P5\n2 1\n65535\n" + bytes([0, 0, 255, 255]))
    (wide / "m.yaml").write_text("image: m.pgm\nresolution: 1\norigin: [0, 0, 0]\n")
    assert load_static(wide / "m.yaml")[0].tolist() == [[LETHAL, FREE]]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.write_text("resolution: 0.5\norigin: [0, 0, 0]\n"),  # no image
        lambda p: p.write_text("image: gone.pgm\nresolution: 1\norigin: [0, 0, 0]\n"),
        lambda p: p.write_text("image: m.pgm\nresolution: 0\norigin: [0, 0, 0]\n"),
        lambda p: p.write_text("image: m.pgm\nresolution: 1\norigin: [0, 0, 0.5]\n"),
        lambda p: (p.parent / "m.pgm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00"),
    ],
)
def test_a_bad_map_file_is_refused(tmp_path, mutate):
    """A missing key or image, a bad resolution, a turned origin, a colour image."""
    path = write_map(tmp_path, [[0]])
    mutate(path)
    with pytest.raises(WorldError, match="costmap"):
        load_static(path)


def test_a_world_uses_a_map_file_instead_of_its_obstacles(tmp_path):
    """The path is relative to the world file; the map's own resolution wins."""
    write_map(tmp_path, [[255] * 8] * 6)
    loaded = world(tmp_path, '[costmap.global.static]\nsource = "m.yaml"')
    assert loaded.costmap.source == str(tmp_path / "m.yaml")
    costmap = build_costmaps(loaded)["a"]
    assert (costmap.resolution, costmap.origin) == (0.5, (1.0, 2.0))
    assert costmap.data.shape == (6, 8)


def test_a_world_that_turns_the_costmap_off_builds_none(tmp_path):
    """`enabled = false` gives every robot None."""
    off = world(tmp_path, "[costmap.global]\nenabled = false")
    assert build_costmaps(off) == {"a": None, "b": None, "c": None}
