"""Tests for the scan a range sensor reports."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Twist
from demuvo.sensing import Lidar, Scan, Scene, StepTruth


def test_a_lidar_gives_its_reading_as_a_scan():
    """The scan carries the beams, the ranges, the limits, the mount and the time."""
    sensor = Lidar(
        None,
        np.random.default_rng(1),
        beams=3,
        fov=1.0,
        max_range=9.0,
        min_range=0.1,
        mount=(0.2, 0.0, 0.5),
    )
    wall = Scene(
        np.array([[2.0, -50.0, 2.0, 50.0]]),
        np.array([Scene.FIXED]),
        np.zeros((0, 3)),
        np.zeros(0, dtype=int),
    )
    truth = StepTruth(0.01, Transform2D.identity(), Twist(), 0.01, lambda: wall, 0)
    reading = sensor.read(truth)
    scan = sensor.scan(reading)
    assert isinstance(scan, Scan)
    assert scan.time == 0.01
    assert scan.ranges is reading.data
    assert scan.angles == pytest.approx(sensor.angles)
    assert (scan.min_range, scan.max_range, scan.mount) == (0.1, 9.0, (0.2, 0.0, 0.5))


def test_the_sensor_pose_is_the_robot_pose_moved_by_the_mount():
    """A sensor 1 m to the left of a robot facing +y sits 1 m west of it."""
    scan = Scan(0.0, np.zeros(1), np.zeros(1), 0.0, 1.0, (0.0, 1.0, 0.3))
    pose = scan.sensor_pose(Transform2D.from_xyt((1.0, 1.0, math.pi / 2)))
    assert pose.xyt == pytest.approx([0.0, 1.0, math.pi / 2 + 0.3])


def test_the_hits_of_a_scan_lie_on_the_wall_that_was_seen():
    """Placing each range from the scan's own sensor pose lands on the wall x = 2."""
    sensor = Lidar(
        None,
        np.random.default_rng(1),
        beams=7,
        fov=math.pi / 2,
        max_range=50.0,
        mount=(0.2, -0.1, 0.2),
    )
    wall = Scene(
        np.array([[2.0, -50.0, 2.0, 50.0]]),
        np.array([Scene.FIXED]),
        np.zeros((0, 3)),
        np.zeros(0, dtype=int),
    )
    pose = Transform2D.from_xyt((1.0, 1.0, 0.5))
    reading = sensor.read(StepTruth(0.01, pose, Twist(), 0.01, lambda: wall, 0))
    scan = sensor.scan(reading)
    x, _, yaw = scan.sensor_pose(pose).xyt
    assert np.isfinite(scan.ranges).all()
    assert x + scan.ranges * np.cos(yaw + scan.angles) == pytest.approx(2.0)
