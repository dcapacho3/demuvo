"""Tests for inflation, and for keeping the global costmap between edits."""

import dataclasses
import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import edit
from demuvo.costmap import build
from demuvo.costmap.build import build_costmaps
from demuvo.costmap.grid import INSCRIBED, LETHAL, MAX_NON_OBSTACLE
from demuvo.costmap.layers import inflate
from demuvo.world import Obstacle, WorldError
from demuvo.world.shapes import Box

from ...world.tests.small_world import small_world


def slow_inflate(grid, resolution, inscribed, inflation, scaling):
    """Inflation the plain way, one pass for each cell of the reach: the oracle."""
    lethal = grid == LETHAL
    reach = math.ceil(round(max(inflation, inscribed) / resolution, 9))
    out = grid.copy()
    rows, cols = grid.shape
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if (dy == 0 and dx == 0) or math.hypot(dy, dx) > reach:
                continue
            distance = math.hypot(dy, dx) * resolution
            if distance <= inscribed:
                cost = INSCRIBED
            else:
                cost = int(
                    MAX_NON_OBSTACLE * math.exp(-scaling * (distance - inscribed))
                )
            for r, c in zip(*np.nonzero(lethal)):
                if 0 <= r + dy < rows and 0 <= c + dx < cols:
                    out[r + dy, c + dx] = max(out[r + dy, c + dx], cost)
    return out


@pytest.mark.parametrize("seed", range(12))
def test_inflation_gives_what_one_pass_for_each_cell_of_the_reach_gives(seed):
    """Random grids, sizes and radii: the two-pass distance changes no cell."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((int(rng.integers(12, 30)), int(rng.integers(12, 30))), np.uint8)
    grid[rng.random(grid.shape) < rng.uniform(0.01, 0.15)] = LETHAL
    grid[rng.random(grid.shape) < 0.05] = 255 if seed % 3 == 0 else 7
    args = (
        float(rng.choice([0.05, 0.1, 0.07])),
        float(rng.uniform(0.1, 0.4)),
        float(rng.uniform(0.1, 0.8)),
        float(rng.uniform(1.0, 12.0)),
    )
    assert np.array_equal(inflate(grid, *args), slow_inflate(grid, *args))


def test_a_grid_with_no_obstacle_is_copied_unchanged():
    """No cell is LETHAL, so nothing spreads."""
    grid = np.full((5, 6), 9, np.uint8)
    out = inflate(grid, 0.1, 0.2, 0.5, 10.0)
    assert np.array_equal(out, grid) and out is not grid


def world_with_a_box(tmp_path):
    """The small world with a box in it."""
    return edit.add_obstacle(small_world(tmp_path), "box", (2.0, 2.0, 0.0), (1.0, 1.0))


def test_the_same_obstacles_give_the_same_costmaps_after_a_robot_moves(tmp_path):
    """A drag changes only a pose, so the map is built once and shared."""
    world = world_with_a_box(tmp_path)
    first = build_costmaps(world)
    moved = edit.set_pose(world, "a", (3.5, 3.0, 0.0))
    assert build_costmaps(moved)["a"] is first["a"]


def test_an_edit_of_the_obstacles_or_the_settings_builds_the_map_again(tmp_path):
    """A moved obstacle and a new cell size each give a map of their own."""
    world = world_with_a_box(tmp_path)
    first = build_costmaps(world)["a"]
    moved = edit.move_obstacle(world, 0, (3.0, 2.0, 0.0))
    assert build_costmaps(moved)["a"] is not first
    finer = edit.set_costmap(world, "global", {"resolution": 0.1})
    assert build_costmaps(finer)["a"] is not first


def test_a_map_file_that_changes_or_goes_is_not_taken_from_the_memory(tmp_path):
    """The file's time and size are part of the key, and a deleted file is an error."""
    pixels = tmp_path / "m.pgm"
    pixels.write_bytes(b"P5\n4 4\n255\n" + bytes([255] * 16))
    yaml = tmp_path / "m.yaml"
    yaml.write_text("image: m.pgm\nresolution: 0.5\norigin: [0, 0, 0]\n")
    extra = '[costmap.global.static]\nsource = "m.yaml"\n'
    with pytest.warns(UserWarning):
        world = small_world(tmp_path, extra)
    first = build_costmaps(world)["a"]
    assert build_costmaps(world)["a"] is first
    pixels.write_bytes(b"P5\n4 4\n255\n" + bytes([0] * 16))
    yaml.write_text("image: m.pgm\nresolution: 0.5\norigin: [0, 0, 0]\n# changed\n")
    assert build_costmaps(world)["a"] is not first
    yaml.unlink()
    with pytest.raises(WorldError, match="cannot read map"):
        build_costmaps(world)


# -- patching the map for a changed shape --


def same_as_a_full_build(world):
    """Whether the costmaps given now equal those built from nothing."""
    got = build.build_costmaps(world)
    build.forget()
    want = build.build_costmaps(world)
    return all(np.array_equal(got[n].data, want[n].data) for n in want)


def crowded_world(tmp_path):
    """A world with several shapes and two sizes of robot."""
    world = small_world(tmp_path)
    world = edit.add_robot(world, "big", (4.4, 3.4, 0.0), radius=0.4)
    for x, y in ((1.5, 2.0), (3.0, 1.0), (4.0, 2.5)):
        world = edit.add_obstacle(world, "box", (x, y, 0.0), (0.6, 0.4))
    return edit.add_outline(world, "polygon", [(2, 3), (2.8, 3.2), (2.4, 3.8)])


def test_a_patched_map_is_the_map_a_full_build_gives(tmp_path):
    """Move, resize, add and remove shapes one after another: no cell differs."""
    world = crowded_world(tmp_path)
    build.forget()
    build.build_costmaps(world)
    edits = [
        lambda w: edit.move_obstacle(w, 0, (1.6, 2.1, 0.5)),
        lambda w: edit.move_obstacle(w, 1, (3.1, 1.2, 0.0)),
        lambda w: edit.resize_obstacle(w, 2, (1.0, 0.8)),
        lambda w: edit.add_obstacle(w, "circle", (4.4, 1.0, 0.0), (0.3,)),
        lambda w: edit.remove_obstacle(w, 0),
        lambda w: edit.move_obstacle(w, 1, (2.2, 0.7, 0.0)),
    ]
    for make in edits:
        world = make(world)
        assert same_as_a_full_build(world)


def test_the_patch_is_used_for_a_small_change_and_not_for_a_big_one(
    tmp_path, monkeypatch
):
    """One moved shape patches; with room for none, the map is built again."""
    world = crowded_world(tmp_path)
    calls = []
    real = build.generate_static
    monkeypatch.setattr(
        build,
        "generate_static",
        lambda w, r, window=None: calls.append(window) or real(w, r, window),
    )
    build.forget()
    build.build_costmaps(world)
    build.build_costmaps(edit.move_obstacle(world, 0, (1.6, 2.1, 0.0)))
    assert calls[0] is None and calls[1] is not None
    monkeypatch.setattr(build, "MAX_PATCH", 0)
    build.build_costmaps(edit.move_obstacle(world, 1, (3.1, 1.1, 0.0)))
    assert calls[2] is None


def test_a_change_that_is_not_one_of_shapes_is_not_patched(tmp_path):
    """The same shapes in a new tuple, a map file and a map with no static layer."""
    world = crowded_world(tmp_path)
    build.forget()
    first = build.build_costmaps(world)["a"]
    rebuilt = dataclasses.replace(world, obstacles=(*world.obstacles,))
    assert build.build_costmaps(rebuilt)["a"] is not first
    assert same_as_a_full_build(rebuilt)
    bare = dataclasses.replace(
        world, costmap=dataclasses.replace(world.costmap, plugins=("inflation",))
    )
    build.build_costmaps(bare)
    moved = edit.move_obstacle(bare, 0, (1.6, 2.1, 0.0))
    assert same_as_a_full_build(moved)


def test_a_patch_without_inflation_changes_the_static_layer_only(tmp_path):
    """With no inflation plugin the map is the static layer itself."""
    world = crowded_world(tmp_path)
    plain = dataclasses.replace(
        world, costmap=dataclasses.replace(world.costmap, plugins=("static",))
    )
    build.forget()
    build.build_costmaps(plain)
    moved = edit.move_obstacle(plain, 0, (1.6, 2.1, 0.0))
    assert same_as_a_full_build(moved)
    assert build.build_costmaps(moved)["a"].data.max() == LETHAL


def test_a_shape_off_the_grid_leaves_the_cells_as_they_were(tmp_path):
    """Added beyond the edge of the world, it marks nothing."""
    world = crowded_world(tmp_path)
    build.forget()
    before = build.build_costmaps(world)["a"].data
    far = Obstacle(Transform2D.from_xyt((50.0, 50.0, 0.0)), Box((1.0, 1.0)))
    plain = dataclasses.replace(world, obstacles=(*world.obstacles, far))
    after = build.build_costmaps(plain)["a"].data
    assert np.array_equal(after, before)
    nothing = dataclasses.replace(
        plain, costmap=dataclasses.replace(plain.costmap, plugins=("static",))
    )
    build.forget()
    build.build_costmaps(dataclasses.replace(world, costmap=nothing.costmap))
    again = dataclasses.replace(nothing, obstacles=(*nothing.obstacles[:-1],))
    far_again = dataclasses.replace(again, obstacles=(*again.obstacles, far))
    assert build.build_costmaps(far_again)["a"].data.max() == LETHAL
