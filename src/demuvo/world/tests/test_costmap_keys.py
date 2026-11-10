"""Tests for the `[costmap.global]` table of a world file."""

import warnings

import pytest

from demuvo.world import WorldError

from .small_world import small_world as world


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("[costmap]\nregional = 1", "unknown key 'regional'"),
        ("[costmap.global]\nenabled = 1", "'enabled' must be true or false"),
        ("[costmap.global]\nresolution = -1", "'resolution' must be positive"),
        ('[costmap.global]\nplugins = ["obstacle"]', "'plugins' must list"),
        ('[costmap.global]\nplugins = ["static", "static"]', "twice"),
        ('[costmap.global]\nplugins = ["inflation", "static"]', "before static"),
        ('[costmap.global.static]\nsource = "nope.yaml"', "no map file"),
        ("[costmap.global.inflation]\nradius = 1", "unknown key 'radius'"),
    ],
)
def test_bad_costmap_keys_are_named(tmp_path, extra, message):
    """Each mistake names the table and the key."""
    with pytest.raises(WorldError, match=message):
        world(tmp_path, extra)


def test_a_setting_for_a_layer_that_is_not_used_warns(tmp_path):
    """Inflation settings with no inflation layer have no effect, and say so."""
    extra = (
        '[costmap.global]\nplugins = ["static"]\n'
        "[costmap.global.inflation]\ninflation_radius = 1"
    )
    with pytest.warns(UserWarning, match="inflation is set but not in 'plugins'"):
        world(tmp_path, extra)


def test_the_settings_come_back_as_written(tmp_path):
    """Resolution and inflation settings reach the spec; no table gives the defaults."""
    default = world(tmp_path).costmap
    assert (default.resolution, default.inflation_radius) == (0.05, 0.55)
    assert default.cost_scaling_factor == 10.0 and default.enabled
    extra = (
        "[costmap.global]\nresolution = 0.2\n[costmap.global.inflation]\n"
        "inflation_radius = 1.5\ncost_scaling_factor = 3.0"
    )
    spec = world(tmp_path, extra).costmap
    assert (spec.resolution, spec.inflation_radius, spec.cost_scaling_factor) == (
        0.2,
        1.5,
        3.0,
    )


def test_no_warning_for_a_good_costmap_table(tmp_path):
    """A table that uses what it sets raises nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        world(tmp_path, "[costmap.global.inflation]\ninflation_radius = 1")


def test_a_map_path_is_taken_relative_to_the_world_file(tmp_path):
    """The loader finds the file next to the world file; it does not read it."""
    (tmp_path / "m.yaml").write_text("")
    spec = world(tmp_path, '[costmap.global.static]\nsource = "m.yaml"').costmap
    assert spec.source == str(tmp_path / "m.yaml")
