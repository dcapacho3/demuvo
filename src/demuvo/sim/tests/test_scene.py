"""Tests for the scene that rays are cast into."""

import math

import numpy as np
import pytest

from demuvo import Simulation, Twist, load_world
from demuvo.costmap.layers import distance_to_part
from demuvo.sensing import Scene, cast
from demuvo.sim.scene import parts_scene, static_scene
from demuvo.world.shapes import Part

WORLD = """
version = 1

[world]
size = [10.0, 6.0]
seed = 1
{obstacle}
[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.2 }}
pose = [1.0, 3.0, 0.0]

[[robot]]
name = "b"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.2 }}
pose = [5.0, 3.0, 0.0]
"""
BOX = """
[[obstacle]]
pose = [3.0, 4.5, 0.0]
shape = { name = "box", size = [2.0, 1.0] }
"""
ORIGIN = np.array([0.0, 0.0])


def sim_of(tmp_path, obstacle=""):
    """A simulation of the small world, with the obstacle text added."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(obstacle=obstacle))
    return Simulation(load_world(path))


def ranges(scene, origin, *degrees, skip=None):
    """The ranges of rays at angles in degrees, up to 50 m."""
    angles = np.radians(np.array(degrees, dtype=float))
    return cast(np.array(origin, dtype=float), angles, scene, 50.0, skip)


def test_a_circle_part_is_one_circle():
    """One corner and a radius is a disc."""
    scene = parts_scene([(4, [Part(((1.0, 2.0),), 0.5)])])
    assert scene.circles.tolist() == [[1.0, 2.0, 0.5]]
    assert scene.circle_owner.tolist() == [4]
    assert len(scene.segments) == 0


def test_a_point_with_no_radius_is_nothing():
    """A ray cannot hit it."""
    scene = parts_scene([(0, [Part(((1.0, 2.0),))])])
    assert len(scene.segments) == len(scene.circles) == 0


def test_a_wall_part_is_one_segment():
    """Two corners and no radius give one edge, and a zero-length edge is dropped."""
    scene = parts_scene(
        [(0, [Part(((0.0, 0.0), (2.0, 0.0))), Part(((1.0, 1.0),) * 2)])]
    )
    assert scene.segments.tolist() == [[0.0, 0.0, 2.0, 0.0]]


def test_a_swelled_wall_is_hit_on_both_sides_and_at_its_ends():
    """A wall of half-thickness 0.1 along the x axis from 0 to 2."""
    scene = parts_scene([(0, [Part(((0.0, 0.0), (2.0, 0.0)), 0.1)])])
    assert ranges(scene, (1.0, 2.0), -90)[0] == pytest.approx(1.9)
    assert ranges(scene, (1.0, -2.0), 90)[0] == pytest.approx(1.9)
    assert ranges(scene, (-2.0, 0.0), 0)[0] == pytest.approx(1.9)
    assert ranges(scene, (4.0, 0.0), 180)[0] == pytest.approx(1.9)


def test_a_box_part_gives_its_four_edges():
    """A box seen from each side is as far as that face."""
    box = Part(((1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0)))
    scene = parts_scene([(0, [box])])
    assert len(scene.segments) == 4
    assert ranges(scene, (0.0, 1.5), 0)[0] == pytest.approx(1.0)
    assert ranges(scene, (2.0, 5.0), -90)[0] == pytest.approx(3.0)
    assert ranges(scene, (5.0, 1.5), 180)[0] == pytest.approx(2.0)
    assert ranges(scene, (2.0, -3.0), 90)[0] == pytest.approx(4.0)


def outside_by(point, part):
    """How far a point is from the swelled part; 0 on its surface or inside."""
    return distance_to_part(point[None], part)[0]


def test_a_ray_meets_a_swelled_box_exactly_at_its_surface():
    """A hit lies on the surface of the box swelled by 0.3; a shorter ray is outside."""
    box = Part(((1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0)), 0.3)
    scene = parts_scene([(0, [box])])
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(200):
        origin = np.array([2.0, 1.5]) + rng.uniform(-6, 6, 2)
        aim = np.array([2.0, 1.5]) + rng.uniform(-1.5, 1.5, 2) - origin
        angle = np.array([math.atan2(aim[1], aim[0])])
        t = cast(origin, angle, scene, 50.0)[0]
        if outside_by(origin, box) == 0 or math.isinf(t):
            continue
        direction = np.array([math.cos(angle[0]), math.sin(angle[0])])
        assert outside_by(origin + t * direction, box) == pytest.approx(0, abs=1e-9)
        assert outside_by(origin + 0.999 * t * direction, box) > 0
        checked += 1
    assert checked > 100


def test_the_static_scene_is_the_border_and_the_obstacles(tmp_path):
    """Four border walls and the four edges of the box, all owned by no robot."""
    scene = static_scene(sim_of(tmp_path, BOX).world)
    assert len(scene.segments) == 8
    assert (scene.segment_owner == Scene.FIXED).all()
    assert ranges(scene, (1.0, 4.5), 0)[0] == pytest.approx(1.0)
    assert ranges(scene, (1.0, 4.5), 180)[0] == pytest.approx(1.0)


def test_the_scene_holds_every_robots_body_owned_by_its_index(tmp_path):
    """A can see B's body, and its own unless it leaves it out."""
    sim = sim_of(tmp_path)
    scene = sim.scene()
    assert scene.circle_owner.tolist() == [0, 1]
    assert ranges(scene, (1.0, 3.0), 0, skip=0)[0] == pytest.approx(3.8)
    assert ranges(scene, (1.0, 3.0), 0)[0] == pytest.approx(0.0)  # inside its own body


def test_the_scene_follows_the_robots_and_is_built_once_a_step(tmp_path):
    """The same scene within a step, a new one after the robots move."""
    sim = sim_of(tmp_path)
    before = sim.scene()
    assert sim.scene() is before
    sim.step({"a": Twist(vx=1.0)})
    after = sim.scene()
    assert after is not before
    assert before.circles[0, 0] == 1.0
    assert after.circles[0, 0] > 1.0


def test_a_sensor_is_given_the_scene_and_its_robots_index(tmp_path):
    """The truth of each step carries the call that builds the scene, and the owner."""
    sim = sim_of(tmp_path)
    seen = {}
    for name, robot in sim.robots.items():
        observe = robot.observe

        def spy(truth, name=name, observe=observe):
            seen[name] = truth
            return observe(truth)

        robot.observe = spy
    sim.step({})
    assert {name: truth.owner for name, truth in seen.items()} == {"a": 0, "b": 1}
    assert seen["b"].scene() is sim.scene()
