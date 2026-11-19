"""Tests for inflation, and for keeping the global costmap between edits."""

import math

import numpy as np
import pytest

from demuvo import edit
from demuvo.costmap.build import build_costmaps
from demuvo.costmap.grid import INSCRIBED, LETHAL, MAX_NON_OBSTACLE
from demuvo.costmap.layers import inflate
from demuvo.world import WorldError

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
