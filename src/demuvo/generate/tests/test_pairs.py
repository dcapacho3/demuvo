"""Tests for picking a start and a goal out of measured free space."""

import math

import numpy as np
import pytest

from demuvo.generate import (
    Clearance,
    Connectivity,
    GenerateError,
    check_grid,
    generate_caves,
    place_pair,
)

ROBOT = Clearance(radius=0.25)


def open_report(rows=40, cols=60, resolution=0.1, origin=(0.0, 0.0)):
    """A report of a world that is free but for its border."""
    blocked = np.zeros((rows, cols), dtype=bool)
    blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True
    return check_grid(blocked, resolution, Clearance(radius=0.0), origin=origin)


def test_a_pair_is_the_same_every_time_and_moves_with_the_seed():
    """The numbers come from the seed and nothing else."""
    report = open_report()
    one = place_pair(report, 3, min_distance=1.0)
    assert one == place_pair(report, 3, min_distance=1.0)
    assert one != place_pair(report, 4, min_distance=1.0)


def test_both_of_a_pair_lie_where_the_robot_may_stand():
    """A start in a wall is no start, so both come out of the largest region."""
    got = generate_caves((20.0, 14.0), 5, clearance=ROBOT)
    report = got.connectivity
    for seed in range(8):
        start, goal = place_pair(report, seed, min_distance=5.0)
        for x, y in (start, goal):
            row = math.floor(y / report.resolution)
            col = math.floor(x / report.resolution)
            assert report.labels[row, col] == 0


def test_a_pair_is_at_least_as_far_apart_as_was_asked():
    """Straight-line distance, which is what a caller can reason about."""
    report = open_report()
    for seed in range(10):
        start, goal = place_pair(report, seed, min_distance=3.0)
        assert math.dist(start, goal) >= 3.0


def test_a_pair_is_no_further_apart_than_was_asked():
    """Both ends of the range hold at once."""
    report = open_report()
    for seed in range(10):
        start, goal = place_pair(report, seed, min_distance=1.0, max_distance=2.0)
        assert 1.0 <= math.dist(start, goal) <= 2.0


def test_a_pair_that_the_free_space_cannot_hold_is_refused():
    """The message says how wide the free space is, so the asking can be fixed."""
    with pytest.raises(GenerateError, match="about 6.8[0-9] m across"):
        place_pair(open_report(), 1, min_distance=99.0)


def test_a_map_with_nowhere_to_stand_has_no_pair():
    """A report of solid rock gives a plain answer and not an index error."""
    report = check_grid(np.ones((20, 20), dtype=bool), 0.1, Clearance(radius=0.0))
    with pytest.raises(GenerateError, match="nowhere for the robot to stand"):
        place_pair(report, 1, min_distance=0.0)


def test_a_pair_follows_the_origin_of_the_world():
    """A world that reaches west and south gives places that do too."""
    report = open_report(origin=(-5.0, -3.0))
    start, goal = place_pair(report, 1, min_distance=1.0)
    for x, y in (start, goal):
        assert -5.0 <= x <= 1.0 and -3.0 <= y <= 1.0
    assert min(start[0], goal[0]) < 0.0


def test_trying_harder_finds_a_pair_a_single_try_may_miss():
    """One try from one start is enough here, which is what the search is for."""
    report = open_report()
    start, goal = place_pair(report, 2, min_distance=4.0, tries=1)
    assert math.dist(start, goal) >= 4.0


@pytest.mark.parametrize(
    "named, message",
    [
        ({"min_distance": -1.0}, "min_distance must be 0 or more"),
        ({"min_distance": 2.0, "max_distance": 1.0}, "max_distance must be at least"),
        ({"min_distance": 1.0, "tries": 0}, "tries must be 1 or more"),
        ({"min_distance": 1.0, "tries": 1.5}, "tries must be 1 or more"),
    ],
)
def test_an_asking_that_cannot_be_is_refused(named, message):
    """Each argument is checked before any place is picked."""
    with pytest.raises(ValueError, match=message):
        place_pair(open_report(), 1, **named)


def test_a_report_made_by_hand_still_gives_a_pair():
    """Nothing here needs a world, only the labels a check left behind."""
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0, :] = -1
    report = Connectivity(True, (), 0.9, 1, 1.0, (), labels, 0.5, (1.0, 1.0))
    start, goal = place_pair(report, 1, min_distance=1.0)
    assert start[0] >= 1.0 and goal[0] >= 1.0  # the origin is honoured
    assert math.dist(start, goal) >= 1.0
