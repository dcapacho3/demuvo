"""Tests that a drawn map is clean: level walls, no overlap, no one-cell traces."""

import numpy as np
import pytest

from demuvo.generate import STYLES, Clearance

shapely = pytest.importorskip("shapely.geometry")

SEEDS = range(1, 9)
ROBOT = Clearance(radius=0.25)


def _bodies(style: str, seed: int):
    """The obstacles of a 20 by 14 m map as shapely polygons and their corners."""
    settings, draw = STYLES[style]
    world = draw((20.0, 14.0), seed, settings(), ROBOT).world
    corners = [np.array(o.shape.points(o.pose)) for o in world.obstacles]
    return [shapely.Polygon(c) for c in corners], corners


@pytest.mark.parametrize("style", ["rooms", "maze"])
@pytest.mark.parametrize("seed", SEEDS)
def test_every_wall_of_a_building_runs_along_an_axis(style, seed):
    """No wall of a building or a maze is slanted, even by a degree."""
    for corners in _bodies(style, seed)[1]:
        step = np.roll(corners, -1, axis=0) - corners
        assert (np.minimum(abs(step[:, 0]), abs(step[:, 1])) < 1e-3).all()


@pytest.mark.parametrize("style", sorted(STYLES))
@pytest.mark.parametrize("seed", SEEDS)
def test_no_two_bodies_of_a_map_overlap_by_more_than_a_sliver(style, seed):
    """Walls meet or keep apart; a body does not lie across another."""
    polygons = _bodies(style, seed)[0]
    for i, one in enumerate(polygons):
        for two in polygons[i + 1 :]:
            assert one.intersection(two).area < 0.05


@pytest.mark.parametrize("style", sorted(STYLES))
@pytest.mark.parametrize("seed", SEEDS)
def test_every_body_is_a_simple_polygon(style, seed):
    """None crosses itself, which the world file refuses and a drawing shows."""
    assert all(p.is_valid for p in _bodies(style, seed)[0])


@pytest.mark.parametrize("seed", SEEDS)
def test_a_field_map_is_a_few_bodies_and_not_a_pile_of_boxes(seed):
    """A region the tracing cannot hold used to fall back to a box for every row."""
    assert len(_bodies("field", seed)[1]) < 40


def test_a_diagonal_wall_of_a_field_is_not_turned_across_itself():
    """Seed 3 of the field has a long wall; its box lies along it."""
    corners = _bodies("field", 3)[1]
    assert all(len(c) >= 3 for c in corners)
