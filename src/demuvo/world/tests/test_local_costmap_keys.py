"""Tests for the `[costmap.local]` table of a world file."""

import warnings

import pytest

from demuvo.world import LocalCostmapSpec, WorldError

from .small_world import small_world


def local_of(tmp_path, extra=""):
    """The local costmap spec the small world gets from the extra lines."""
    return small_world(tmp_path, extra).local_costmap


def test_no_table_gives_the_defaults(tmp_path):
    """A world file needs no local costmap table."""
    assert local_of(tmp_path) == LocalCostmapSpec()


def test_every_setting_is_read_as_written(tmp_path):
    """The window, the layers and the two layer tables."""
    spec = local_of(
        tmp_path,
        "[costmap.local]\nenabled = false\nwindow = 4.0\nresolution = 0.1\n"
        "period = 0.5\n[costmap.local.obstacle]\nobstacle_max_range = 2.0\n"
        "raytrace_max_range = 2.5\n[costmap.local.inflation]\ninflation_radius = 0.8\n"
        "cost_scaling_factor = 4.0\n",
    )
    assert spec == LocalCostmapSpec(
        enabled=False,
        window=4.0,
        resolution=0.1,
        period=0.5,
        obstacle_max_range=2.0,
        raytrace_max_range=2.5,
        inflation_radius=0.8,
        cost_scaling_factor=4.0,
    )


@pytest.mark.parametrize("layers", [["obstacle"], ["inflation"], []])
def test_the_layers_can_be_chosen(tmp_path, layers):
    """Any of the layers in their fixed order."""
    spec = local_of(
        tmp_path, f"[costmap.local]\nplugins = {layers!r}".replace("'", '"')
    )
    assert spec.plugins == tuple(layers)


def test_the_global_and_local_tables_do_not_disturb_each_other(tmp_path):
    """Each is read from its own table."""
    world = small_world(
        tmp_path,
        "[costmap.global]\nresolution = 0.2\n[costmap.local]\nresolution = 0.1\n",
    )
    assert (world.costmap.resolution, world.local_costmap.resolution) == (0.2, 0.1)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("[costmap.local]\nenabled = 1", "'enabled' must be true or false"),
        ("[costmap.local]\nwindow = -1", "'window' must be positive"),
        ("[costmap.local]\nbogus = 1", "unknown key 'bogus'"),
        ('[costmap.local]\nplugins = ["static"]', "'plugins' must list 'obstacle'"),
        (
            '[costmap.local]\nplugins = ["inflation", "obstacle"]',
            "puts inflation before obstacle",
        ),
        ("[costmap.local.obstacle]\nrange = 1", "unknown key 'range'"),
        (
            "[costmap.local.obstacle]\nobstacle_max_range = 0",
            "'obstacle_max_range' must be positive",
        ),
        ("[costmap.local.inflation]\nradius = 1", "unknown key 'radius'"),
    ],
)
def test_bad_local_costmap_keys_are_named(tmp_path, extra, message):
    """Each mistake names the table and the key."""
    with pytest.raises(WorldError, match=f"costmap.local.*{message}"):
        local_of(tmp_path, extra)


def test_a_setting_for_a_layer_that_is_not_used_warns(tmp_path):
    """Inflation settings with no inflation layer have no effect, and say so."""
    extra = (
        '[costmap.local]\nplugins = ["obstacle"]\n'
        "[costmap.local.inflation]\ninflation_radius = 1\n"
    )
    with pytest.warns(UserWarning, match="inflation is set but not in 'plugins'"):
        local_of(tmp_path, extra)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        local_of(tmp_path, "[costmap.local.inflation]\ninflation_radius = 1\n")
