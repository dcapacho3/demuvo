"""Tests for the local costmap."""

import math
from unittest import mock

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Twist
from demuvo.costmap import (
    INSCRIBED,
    LETHAL,
    NO_INFORMATION,
    LocalCostmap,
)
from demuvo.sensing import Lidar, Scene, StepTruth
from demuvo.world import LocalCostmapSpec


def walls(*xs):
    """A scene of vertical walls at the given x positions, owned by no robot."""
    segments = np.array([[x, -50.0, x, 50.0] for x in xs], dtype=float).reshape(-1, 4)
    return Scene(
        segments,
        np.full(len(segments), Scene.FIXED),
        np.zeros((0, 3)),
        np.zeros(0, dtype=int),
    )


def scan_at(pose, scene, time=0.01, **settings):
    """The scan a lidar at the pose takes of the scene."""
    sensor = Lidar(None, np.random.default_rng(1), **settings)
    truth = StepTruth(time, pose, Twist(), 0.01, lambda: scene, 0)
    return sensor.scan(sensor.read(truth))


def at(x, y=0.0, yaw=0.0):
    """A pose from (x, y, angle)."""
    return Transform2D.from_xyt((x, y, yaw))


def lethal(costmap):
    """The world positions of the centres of the cells marked as obstacles."""
    rows, cols = np.nonzero(costmap.data == LETHAL)
    x, y = costmap.centres(rows, cols)
    return np.column_stack([x, y])


def local(**settings):
    """A local costmap of a robot of inscribed radius 0.2; no inflation unless asked."""
    settings = {"window": 6.0, "resolution": 0.1, "plugins": ("obstacle",), **settings}
    return LocalCostmap(0.2, LocalCostmapSpec(**settings))


def test_the_defaults_are_the_usual_local_costmap_and_none_exists_before_an_update():
    """3 m, 5 cm cells, 5 Hz, hits to 2.5 m, clearing to 3 m, the two layers."""
    settings = LocalCostmapSpec()
    assert (settings.window, settings.resolution, settings.period) == (3.0, 0.05, 0.2)
    assert (settings.obstacle_max_range, settings.raytrace_max_range) == (2.5, 3.0)
    assert (settings.inflation_radius, settings.cost_scaling_factor) == (0.55, 10.0)
    assert settings.plugins == ("obstacle", "inflation")
    assert LocalCostmap(0.2).costmap is None


def test_a_wall_is_marked_where_it_is_and_the_way_to_it_is_free():
    """The cells at the wall are obstacles; those in front and behind are not."""
    costmap = local()
    pose = at(0.5)
    costmap.update(pose, {"lidar": scan_at(pose, walls(2.03))}, 0.2)
    marked = lethal(costmap.costmap)
    assert marked[:, 0] == pytest.approx(2.05)
    assert costmap.costmap.cost_at(2.05, 0.0) == LETHAL
    assert costmap.costmap.cost_at(2.05, 1.0) == LETHAL
    assert costmap.costmap.cost_at(1.0, 0.0) == 0
    assert costmap.costmap.cost_at(2.55, 0.0) == 0


def test_a_wall_seen_from_two_poses_marks_the_same_cells():
    """The mark does not depend on where the robot stood."""
    first, second = local(), local()
    for costmap, pose in ((first, at(0.5)), (second, at(1.0, -0.3, 0.1))):
        costmap.update(pose, {"lidar": scan_at(pose, walls(2.03))}, 0.2)
    for costmap in (first, second):
        assert lethal(costmap.costmap)[:, 0] == pytest.approx(2.05)


def test_a_later_scan_that_passes_through_a_mark_clears_it():
    """The wall is gone, so the next scan sees nothing and empties its cells."""
    costmap = local()
    pose = at(0.5)
    costmap.update(pose, {"lidar": scan_at(pose, walls(2.03))}, 0.2)
    assert len(lethal(costmap.costmap)) > 0
    costmap.update(pose, {"lidar": scan_at(pose, walls(), time=0.3)}, 0.4)
    assert len(lethal(costmap.costmap)) == 0


def test_a_mark_behind_a_nearer_hit_stays():
    """A beam stops at its hit, so it says nothing about what lies beyond."""
    costmap = local()
    pose = at(0.5)
    costmap.update(pose, {"lidar": scan_at(pose, walls(2.03))}, 0.2)
    costmap.update(pose, {"lidar": scan_at(pose, walls(1.03), time=0.3)}, 0.4)
    assert costmap.costmap.cost_at(1.05, 0.0) == LETHAL
    assert costmap.costmap.cost_at(2.05, 0.0) == LETHAL


def test_a_hit_beyond_the_marking_range_clears_but_does_not_mark():
    """A wall at 2.9 m is not marked at 2.5 m, but the cells before it are cleared."""
    costmap = local()
    pose = at(0.0)
    costmap.update(pose, {"lidar": scan_at(pose, walls(1.03))}, 0.2)
    assert len(lethal(costmap.costmap)) > 0
    costmap.update(pose, {"lidar": scan_at(pose, walls(2.93), time=0.3)}, 0.4)
    assert len(lethal(costmap.costmap)) == 0


def test_a_beam_that_hit_nothing_clears_only_as_far_as_the_clearing_range():
    """A mark at 3.2 m survives a scan that saw nothing, which clears only to 3 m."""
    costmap = local(window=8.0, obstacle_max_range=3.5, raytrace_max_range=3.0)
    pose = at(0.0)
    for time, scene in ((0.01, walls(3.23)), (0.3, walls())):
        scan = scan_at(pose, scene, time=time, max_range=3.5)
        costmap.update(pose, {"lidar": scan}, time + 0.2)
    assert costmap.costmap.cost_at(3.25, 0.0) == LETHAL


def test_the_window_follows_the_robot_and_marks_leave_it():
    """A mark stays put while the window moves, and goes when the window leaves it."""
    costmap = local(window=2.0)
    pose = at(0.0)
    costmap.update(pose, {"lidar": scan_at(pose, walls(0.83))}, 0.2)
    assert costmap.costmap.cost_at(0.85, 0.0) == LETHAL
    costmap.update(at(0.3), {}, 0.4)
    assert costmap.costmap.cost_at(0.85, 0.0) == LETHAL
    costmap.update(at(5.0), {}, 0.6)
    assert not costmap.costmap.data.any()
    assert costmap.costmap.cost_at(0.85, 0.0) == NO_INFORMATION
    x, y = costmap.costmap.origin
    assert x == pytest.approx(round(x / 0.1) * 0.1) and y == pytest.approx(
        round(y / 0.1) * 0.1
    )
    assert costmap.costmap.data.shape == (20, 20)


def test_a_hit_outside_the_window_is_ignored():
    """The wall at 1.5 m is outside a window of 2 m, so nothing is marked."""
    costmap = local(window=2.0)
    pose = at(0.0)
    costmap.update(pose, {"lidar": scan_at(pose, walls(1.53))}, 0.2)
    assert not costmap.costmap.data.any()


def test_an_update_happens_once_a_period():
    """The first call updates; a call 0.1 s later does not; one 0.2 s later does."""
    costmap = local()
    costmap.update(at(0.0), {}, 0.2)
    first = costmap.costmap
    costmap.update(at(0.0), {}, 0.3)
    assert costmap.costmap is first
    costmap.update(at(0.0), {}, 0.4)
    assert costmap.costmap is not first


def test_a_scan_is_taken_in_once():
    """The same scan offered at the next update is not cleared and marked again."""
    costmap = local()
    pose = at(0.5)
    scan = scan_at(pose, walls(2.03))
    costmap.update(pose, {"lidar": scan}, 0.2)
    with mock.patch.object(LocalCostmap, "_clear") as clear:
        costmap.update(pose, {"lidar": scan}, 0.4)
    clear.assert_not_called()


def test_several_lidars_are_all_taken_in():
    """A front lidar sees the wall ahead, a rear one the wall behind."""
    costmap = local()
    pose = at(0.0)
    scene = walls(1.03, -1.03)
    front = scan_at(pose, scene, beams=40, fov=math.pi / 2)
    rear = scan_at(pose, scene, beams=40, fov=math.pi / 2, mount=(0.0, 0.0, math.pi))
    costmap.update(pose, {"front": front, "rear": rear}, 0.2)
    xs = lethal(costmap.costmap)[:, 0]
    assert set(np.round(xs, 2)) == {1.05, -1.05}


def test_the_obstacles_are_inflated_by_the_robot_radius():
    """A cell within the inscribed radius of the wall cannot hold the robot's centre."""
    costmap = local(plugins=("obstacle", "inflation"))
    pose = at(0.5)
    costmap.update(pose, {"lidar": scan_at(pose, walls(2.03))}, 0.2)
    assert costmap.costmap.cost_at(1.95, 0.0) >= INSCRIBED
    assert costmap.costmap.cost_at(1.0, 0.0) == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"window": 0.0},
        {"resolution": -0.1},
        {"period": 0.0},
        {"obstacle_max_range": 0.0},
        {"raytrace_max_range": math.nan},
        {"inflation_radius": 0.0},
        {"cost_scaling_factor": -1.0},
        {"plugins": ("inflation", "obstacle")},
        {"plugins": ("obstacle", "obstacle")},
        {"plugins": ("static",)},
    ],
)
def test_bad_settings_are_refused(settings):
    """A bad number or a bad list of layers is refused when the settings are made."""
    with pytest.raises(ValueError):
        LocalCostmapSpec(**settings)
