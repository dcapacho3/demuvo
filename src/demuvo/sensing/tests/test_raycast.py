"""Tests for the ray caster."""

import math

import numpy as np
import pytest

from demuvo.sensing.raycast import Scene, cast

FIXED = Scene.FIXED


def scene(segments=(), circles=(), segment_owner=None, circle_owner=None):
    """A scene from lists of segments (x0, y0, x1, y1) and circles (x, y, r)."""
    segments = np.array(segments, dtype=float).reshape(-1, 4)
    circles = np.array(circles, dtype=float).reshape(-1, 3)
    return Scene(
        segments,
        np.full(len(segments), FIXED)
        if segment_owner is None
        else np.array(segment_owner),
        circles,
        np.full(len(circles), FIXED)
        if circle_owner is None
        else np.array(circle_owner),
    )


def rays(*degrees):
    """Angles in radians from angles in degrees."""
    return np.radians(np.array(degrees, dtype=float))


def test_a_wall_at_a_known_distance_gives_that_range_at_every_beam_angle():
    """A wall 2 m ahead is 2 / cos(angle) away along a ray at that angle."""
    wall = scene(segments=[(2.0, -100.0, 2.0, 100.0)])
    angles = np.radians(np.arange(-80, 81, 10))
    ranges = cast(np.array([0.0, 0.0]), angles, wall, 50.0)
    assert ranges == pytest.approx(2.0 / np.cos(angles))


def test_a_ray_stops_at_the_nearest_of_two_walls():
    """The wall behind the first one is hidden."""
    two = scene(segments=[(5.0, -9.0, 5.0, 9.0), (3.0, -9.0, 3.0, 9.0)])
    assert cast(np.array([0.0, 0.0]), rays(0), two, 50.0) == pytest.approx([3.0])


def test_a_ray_that_meets_nothing_gives_inf():
    """Pointing away, parallel to the wall, or past its end."""
    wall = scene(segments=[(2.0, -1.0, 2.0, 1.0)])
    away, parallel, past = rays(180), rays(90), rays(45)
    ranges = cast(
        np.array([0.0, 0.0]), np.concatenate([away, parallel, past]), wall, 50
    )
    assert np.isinf(ranges).all()


def test_a_hit_beyond_the_range_gives_inf():
    """The ray reaches only `max_range`."""
    wall = scene(segments=[(4.0, -9.0, 4.0, 9.0)])
    assert np.isinf(cast(np.array([0.0, 0.0]), rays(0), wall, 3.9)).all()
    assert cast(np.array([0.0, 0.0]), rays(0), wall, 4.0) == pytest.approx([4.0])


def test_the_end_of_a_segment_is_a_hit():
    """A ray that reaches either end point still hits."""
    seg = scene(segments=[(2.0, 0.0, 2.0, 1.0)])
    assert cast(np.array([0.0, 0.0]), rays(0), seg, 9.0) == pytest.approx([2.0])
    assert cast(np.array([0.0, 1.0]), rays(0), seg, 9.0) == pytest.approx([2.0])


def test_a_circle_is_hit_on_its_near_side():
    """A circle of radius 1 centred 3 m ahead is 2 m away; a miss gives inf."""
    ball = scene(circles=[(3.0, 0.0, 1.0)])
    ranges = cast(np.array([0.0, 0.0]), rays(0, 30, 180), ball, 50.0)
    assert ranges[0] == pytest.approx(2.0)
    assert np.isinf(ranges[1:]).all()


def test_a_ray_grazing_a_circle_hits_it():
    """A ray that just touches the circle counts as a hit."""
    ball = scene(circles=[(3.0, 1.0, 1.0)])
    assert cast(np.array([0.0, 0.0]), rays(0), ball, 50.0) == pytest.approx([3.0])


def test_an_origin_inside_a_circle_is_a_hit_at_zero():
    """The sensor is inside the obstacle, so every ray stops at once."""
    ball = scene(circles=[(0.0, 0.0, 1.0)])
    assert cast(np.array([0.2, 0.0]), rays(0, 90, 200), ball, 5.0) == pytest.approx(0.0)


def test_the_owner_of_a_body_can_be_left_out():
    """A robot's own segment and circle are passed through; others still stop it."""
    both = scene(
        segments=[(1.0, -1.0, 1.0, 1.0), (4.0, -1.0, 4.0, 1.0)],
        circles=[(2.0, 0.0, 0.5)],
        segment_owner=[3, FIXED],
        circle_owner=[3],
    )
    origin = np.array([0.0, 0.0])
    assert cast(origin, rays(0), both, 50.0) == pytest.approx([1.0])
    assert cast(origin, rays(0), both, 50.0, skip=3) == pytest.approx([4.0])


def test_the_fixed_world_is_not_skipped_by_default():
    """With no owner to skip, walls owned by no robot still stop the ray."""
    wall = scene(segments=[(2.0, -1.0, 2.0, 1.0)])
    assert cast(np.array([0.0, 0.0]), rays(0), wall, 9.0, skip=None) == pytest.approx(
        [2.0]
    )


def test_an_empty_scene_gives_inf_everywhere():
    """Nothing to hit."""
    assert np.isinf(cast(np.array([0.0, 0.0]), rays(0, 90), scene(), 9.0)).all()


def reference_segment(origin, direction, segments):
    """The nearest segment hit by one ray, worked out the plain way."""
    best = math.inf
    for x0, y0, x1, y1 in segments:
        edge = np.array([x1 - x0, y1 - y0])
        start = np.array([x0, y0]) - origin
        den = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(den) < 1e-12:
            continue
        t = (start[0] * edge[1] - start[1] * edge[0]) / den
        u = (start[0] * direction[1] - start[1] * direction[0]) / den
        if t >= 0 and 0 <= u <= 1:
            best = min(best, t)
    return best


def reference_circle(origin, direction, circles):
    """The nearest circle hit by one ray, worked out the plain way."""
    best = math.inf
    for x, y, r in circles:
        offset = origin - np.array([x, y])
        half = direction @ offset
        inside = offset @ offset - r * r
        disc = half * half - inside
        if inside < 0:
            best = 0.0
        elif disc >= 0 and -half - math.sqrt(disc) >= 0:
            best = min(best, -half - math.sqrt(disc))
    return best


def test_it_agrees_with_a_plain_loop_on_random_scenes():
    """The array version and a one-ray-at-a-time version give the same ranges."""
    rng = np.random.default_rng(5)
    segments = rng.uniform(-8, 8, size=(25, 4))
    circles = np.column_stack([rng.uniform(-8, 8, (6, 2)), rng.uniform(0.2, 1.5, 6)])
    origin = rng.uniform(-3, 3, 2)
    angles = np.linspace(0, math.tau, 90, endpoint=False)
    fast = cast(origin, angles, scene(segments=segments, circles=circles), 12.0)
    for angle, value in zip(angles, fast):
        direction = np.array([math.cos(angle), math.sin(angle)])
        best = min(
            reference_segment(origin, direction, segments),
            reference_circle(origin, direction, circles),
        )
        assert value == pytest.approx(best if best <= 12.0 else math.inf)
