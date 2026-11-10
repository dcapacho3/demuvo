"""Tests for the lidar sensor."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Twist
from demuvo.sensing import Lidar, Scene, StepTruth


def wall(x0, y0, x1, y1, circles=()):
    """A scene of one fixed wall, and circles owned by robot 0."""
    circles = np.array(circles, dtype=float).reshape(-1, 3)
    return Scene(
        np.array([[x0, y0, x1, y1]], dtype=float),
        np.array([Scene.FIXED]),
        circles,
        np.zeros(len(circles), dtype=int),
    )


AHEAD = wall(2.0, -100.0, 2.0, 100.0)


def step(scene=AHEAD, pose=(0.0, 0.0, 0.0), time=0.01, owner=0):
    """The truth of one step, with the scene a lidar can read."""
    return StepTruth(
        time, Transform2D.from_xyt(pose), Twist(), 0.01, lambda: scene, owner
    )


def lidar(seed=1, **settings):
    """A lidar with its own random generator."""
    return Lidar(None, np.random.default_rng(seed), **settings)


def test_a_wall_ahead_is_read_at_every_beam_angle():
    """The beam straight ahead reads 2 m; the others read 2 / cos(angle)."""
    sensor = lidar()
    ranges = sensor.read(step()).data
    assert len(ranges) == 360 and len(sensor.angles) == 360
    ahead = np.flatnonzero(np.isclose(sensor.angles, 0.0))[0]
    assert ranges[ahead] == pytest.approx(2.0)
    front = np.abs(sensor.angles) < math.radians(70)
    assert ranges[front] == pytest.approx(2.0 / np.cos(sensor.angles[front]))
    assert np.isinf(ranges[np.abs(sensor.angles) > math.radians(100)]).all()


def test_the_beams_turn_with_the_robot():
    """A robot facing +y sees a wall along y = 2 straight ahead."""
    sensor = lidar(beams=1)
    ranges = sensor.read(
        step(wall(-100.0, 2.0, 100.0, 2.0), pose=(0.0, 0.0, math.pi / 2))
    )
    assert ranges.data == pytest.approx([2.0])


def test_a_mounted_lidar_reads_from_its_own_position_and_direction():
    """A lidar 0.5 m ahead reads 1.5 m; turned round it looks back from there."""
    forward = lidar(beams=1, mount=(0.5, 0.0, 0.0))
    assert forward.read(step()).data == pytest.approx([1.5])
    backward = lidar(beams=1, mount=(0.5, 0.0, math.pi))
    behind = step(wall(-2.0, -100.0, -2.0, 100.0))
    assert backward.read(behind).data == pytest.approx([2.5])


def test_the_mount_turns_with_the_robot():
    """A lidar 1 m to the left of a robot facing +y sits 1 m to its west."""
    sensor = lidar(beams=1, mount=(0.0, 1.0, 0.0))
    north = step(wall(-100.0, 5.0, 100.0, 5.0), pose=(1.0, 1.0, math.pi / 2))
    assert sensor.read(north).data == pytest.approx([4.0])  # from (0, 1) to y = 5


def test_a_narrow_field_of_view_runs_from_edge_to_edge():
    """Five beams over a quarter turn, centred on the heading, ends included."""
    sensor = lidar(beams=5, fov=math.pi / 2)
    assert sensor.angles == pytest.approx(np.linspace(-math.pi / 4, math.pi / 4, 5))
    assert lidar(beams=1, fov=math.pi / 2).angles.tolist() == [0.0]


def test_readings_outside_the_range_are_inf():
    """Past the farthest range, and nearer than the nearest."""
    assert np.isinf(lidar(beams=1, max_range=1.5).read(step()).data).all()
    assert np.isinf(
        lidar(beams=1, min_range=2.5, max_range=9.0).read(step()).data
    ).all()


def test_noise_is_added_to_the_ranges_and_repeats_with_the_seed():
    """The error has the set spread; a miss stays a miss; the same seed repeats."""
    far = {"beams": 3600, "max_range": 1000.0}
    clean = lidar(**far).read(step()).data
    noisy = lidar(noise=0.05, seed=7, **far).read(step()).data
    again = lidar(noise=0.05, seed=7, **far).read(step()).data
    assert (np.isinf(noisy) == np.isinf(clean)).all()
    seen = np.isfinite(clean)
    error = noisy[seen] - clean[seen]
    assert error.std() == pytest.approx(0.05, rel=0.1)
    assert abs(error.mean()) < 0.01
    assert np.array_equal(noisy, again)


def test_it_scans_once_a_period_and_reports_the_same_scan_between():
    """In 100 steps of 0.01 s a 0.1 s period gives scans at 0.01 s, 0.1 s, 0.2 s."""
    sensor = lidar()
    readings = [sensor.read(step(time=round(k * 0.01, 10))) for k in range(1, 101)]
    scans = []
    for reading in readings:
        if not scans or reading is not scans[-1]:
            scans.append(reading)
    assert [round(r.time, 10) for r in scans] == [0.01] + [
        round(0.1 * k, 10) for k in range(1, 11)
    ]
    assert readings[1] is readings[0]


def test_a_step_longer_than_the_period_scans_every_step():
    """The lidar cannot scan twice in a step, so it scans each one."""
    sensor = lidar(beams=1, period=0.1)
    first = sensor.read(step(time=0.5))
    assert sensor.read(step(time=1.0)) is not first


def test_the_robots_own_body_is_left_out():
    """A circle round the sensor that belongs to the robot does not stop the rays."""
    body = wall(2.0, -100.0, 2.0, 100.0, circles=[(0.0, 0.0, 0.3)])
    assert lidar(beams=1).read(step(body, owner=0)).data == pytest.approx([2.0])
    assert lidar(beams=1).read(step(body, owner=None)).data == pytest.approx([0.0])


def test_a_lidar_needs_the_scene():
    """A step with no scene is refused, and the message says so."""
    truth = StepTruth(0.01, Transform2D.identity(), Twist(), 0.01)
    with pytest.raises(ValueError, match="needs the scene"):
        lidar().read(truth)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"beams": 0}, "beams"),
        ({"beams": 2.5}, "beams"),
        ({"fov": 0.0}, "fov"),
        ({"fov": 7.0}, "fov"),
        ({"min_range": 9.0}, "min_range"),
        ({"noise": -0.1}, "noise"),
        ({"period": 0.0}, "period"),
        ({"mount": (0.0, 0.0)}, "mount"),
        ({"mount": (0.0, math.nan, 0.0)}, "mount"),
    ],
)
def test_a_bad_setting_is_refused_and_named(settings, message):
    """Each mistake names the setting."""
    with pytest.raises(ValueError, match=message):
        lidar(**settings)


def test_the_settings_are_the_ones_the_world_file_may_name():
    """An unknown setting is refused; the table of options lists the real ones."""
    with pytest.raises(TypeError):
        lidar(colour="red")
    assert Lidar.options == {
        "beams",
        "fov",
        "max_range",
        "min_range",
        "noise",
        "period",
        "mount",
    }
    sensor = lidar(mount=[1, 2, 3])
    assert sensor.mount == (1.0, 2.0, 3.0)
