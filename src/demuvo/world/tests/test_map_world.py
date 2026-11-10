"""Tests for a world that reads its costmap from a map file."""

import numpy as np
import pytest

from demuvo import load_world
from demuvo.costmap.grid import FREE, LETHAL, NO_INFORMATION
from demuvo.costmap.layers import load_static
from demuvo.world import WorldError

from .small_world import WORLD


def write_map(tmp_path, pixels, origin="[-2.0, -1.0, 0.0]", extra=""):
    """A map of one metre per cell; the pixels are rows from the top down."""
    rows, cols = len(pixels), len(pixels[0])
    body = bytes(v for row in pixels for v in row)
    (tmp_path / "m.pgm").write_bytes(f"P5\n{cols} {rows}\n255\n".encode() + body)
    (tmp_path / "m.yaml").write_text(
        f"image: m.pgm\nresolution: 1\norigin: {origin}\n{extra}\n"
    )
    return tmp_path / "m.yaml"


def load(tmp_path, world_lines="", costmap='source = "m.yaml"'):
    """The small world with `[world]` lines of its own, no `size` unless given."""
    text = WORLD.format(extra=f"[costmap.global.static]\n{costmap}\n")
    text = text.replace("size = [5.0, 4.0]\n", world_lines)
    path = tmp_path / "w.toml"
    path.write_text(text)
    return load_world(path)


def test_a_world_without_a_size_takes_its_size_and_origin_from_the_map(tmp_path):
    """A map of 5 by 4 cells at 1 m, corner at (-2, -1)."""
    write_map(tmp_path, [[255] * 5] * 4)
    world = load(tmp_path)
    assert world.size == (5.0, 4.0)
    assert world.bounds == (-2.0, -1.0, 3.0, 3.0)


def test_a_world_file_origin_wins_over_the_maps(tmp_path):
    """The size still comes from the map; the origin is the file's."""
    write_map(tmp_path, [[255] * 5] * 4)
    with pytest.warns(UserWarning, match=r"origin \[0.0, 0.0\] differs"):
        world = load(tmp_path, "origin = [0.0, 0.0]\n")
    assert world.bounds == (0.0, 0.0, 5.0, 4.0)


def test_a_given_size_keeps_the_world_at_zero_and_a_mismatch_warns(tmp_path):
    """Adding a map to a working world file does not move the world."""
    write_map(tmp_path, [[255] * 5] * 4)
    with pytest.warns(UserWarning, match="origin") as caught:
        world = load(tmp_path, "size = [5.0, 4.0]\n")
    assert world.bounds == (0.0, 0.0, 5.0, 4.0)
    assert len(caught) == 1


def test_a_matching_size_and_origin_do_not_warn(tmp_path, recwarn):
    """The file repeats what the map says."""
    write_map(tmp_path, [[255] * 5] * 4)
    load(tmp_path, "size = [5.0, 4.0]\norigin = [-2.0, -1.0]\n")
    assert not recwarn.list


def test_a_world_without_size_or_map_still_needs_a_size(tmp_path):
    """With the generated map, `size` is required."""
    with pytest.raises(WorldError, match="size"):
        load(tmp_path, costmap='source = "generate"')


def test_a_map_that_is_off_gives_no_size(tmp_path):
    """A disabled costmap reads no map, so the file needs its own size."""
    write_map(tmp_path, [[255] * 5] * 4)
    text = WORLD.format(extra="[costmap.global]\nenabled = false\n")
    path = tmp_path / "w.toml"
    path.write_text(text.replace("size = [5.0, 4.0]\n", ""))
    with pytest.raises(WorldError, match="size"):
        load_world(path)


def test_unknown_cells_are_free_or_blocked_as_the_world_says(tmp_path):
    """Pixels 0, 128 and 255: obstacle, unknown (occupancy 0.5), free."""
    path = write_map(tmp_path, [[0, 128, 255]])
    free, *_ = load_static(path)
    blocked, *_ = load_static(path, "blocked")
    assert free.tolist() == [[LETHAL, FREE, FREE]]
    assert blocked.tolist() == [[LETHAL, NO_INFORMATION, FREE]]


def test_the_unknown_setting_reaches_the_costmap_spec(tmp_path):
    """`unknown = "blocked"` is kept; the default is free."""
    write_map(tmp_path, [[255] * 5] * 4)
    assert load(tmp_path).costmap.unknown == "free"
    spec = load(tmp_path, costmap='source = "m.yaml"\nunknown = "blocked"').costmap
    assert spec.unknown == "blocked"


def test_a_bad_unknown_setting_is_refused(tmp_path):
    """Only "free" and "blocked" exist."""
    write_map(tmp_path, [[255] * 5] * 4)
    with pytest.raises(WorldError, match="'unknown' must be"):
        load(tmp_path, costmap='source = "m.yaml"\nunknown = "maybe"')


def test_the_unknown_setting_without_a_map_warns(tmp_path):
    """The generated map has no unknown cells."""
    with pytest.warns(UserWarning, match="no effect without a map file"):
        load(
            tmp_path,
            "size = [5.0, 4.0]\n",
            'source = "generate"\nunknown = "blocked"',
        )


def test_the_map_is_read_into_a_float_occupancy(tmp_path):
    """Row 0 is the lowest y, so the image's last row comes first."""
    from demuvo.world.mapfile import read_map  # pylint: disable=import-outside-toplevel

    found = read_map(write_map(tmp_path, [[0, 0], [255, 255]]))
    assert np.allclose(found.occupancy, [[0.0, 0.0], [1.0, 1.0]])
    assert (found.free_thresh, found.occupied_thresh) == (0.25, 0.65)
