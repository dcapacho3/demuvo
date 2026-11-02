"""Tests for the collision checker, and for what Simulation does with contacts."""

import math

import pytest
from robomath import Transform2D

from demuvo import Simulation, Twist, load_world
from demuvo.sim.collisions import ALL, Checker

HEAD = """
version = 1
[world]
size = [10.0, 6.0]
seed = 1
dt = 0.1
"""
WALL = """
[[obstacle]]
shape = { name = "linestring", vertices = [[5.0, 0.5], [5.0, 5.5]] }
"""


def robot(name, x, y, angle=0.0, radius=0.5):
    """Robot."""
    return f"""
[[robot]]
name = "{name}"
model = "diff_drive"
shape = {{ name = "circle", radius = {radius} }}
pose = [{x}, {y}, {angle}]
"""


def world(tmp_path, *parts):
    """World."""
    path = tmp_path / "w.toml"
    path.write_text(HEAD + "".join(parts))
    return load_world(path)


def pose(x, y, angle=0.0):
    """Pose."""
    return Transform2D.from_xyt((x, y, angle))


def test_a_free_robot_has_no_contact(tmp_path):
    """A free robot has no contact."""
    checker = Checker(world(tmp_path, WALL, robot("a", 2, 3)))
    assert checker.check({"a": pose(2, 3)}) == []


def test_a_robot_in_a_wall_is_reported_with_the_way_out(tmp_path):
    """A robot in a wall is reported with the way out."""
    checker = Checker(world(tmp_path, WALL, robot("a", 2, 3)))
    (contact,) = checker.check({"a": pose(4.8, 3)})
    assert (contact.a, contact.b, contact.b_moves) == ("a", "obstacle 1", False)
    assert contact.normal == pytest.approx((1.0, 0.0))
    assert contact.depth == pytest.approx(0.3)


def test_the_border_is_a_wall(tmp_path):
    """The border is a wall."""
    checker = Checker(world(tmp_path, robot("a", 2, 3)))
    (contact,) = checker.check({"a": pose(0.3, 3)})
    assert contact.b == "border"
    assert contact.normal == pytest.approx((-1.0, 0.0))


def test_two_robots_that_overlap_make_one_contact(tmp_path):
    """Two robots that overlap make one contact."""
    checker = Checker(world(tmp_path, robot("a", 1, 1), robot("b", 3, 1)))
    (contact,) = checker.check({"a": pose(2, 3), "b": pose(2.6, 3)})
    assert (contact.a, contact.b, contact.b_moves) == ("a", "b", True)
    assert contact.depth == pytest.approx(0.4)


def test_a_concave_obstacle_is_hollow(tmp_path):
    """A concave obstacle is hollow."""
    u = """
[[obstacle]]
shape = { name = "polygon", vertices = [
  [2,1], [6,1], [6,5], [5,5], [5,2], [3,2], [3,5], [2,5]] }
"""
    checker = Checker(world(tmp_path, u, robot("a", 4, 4, radius=0.4)))
    assert checker.check({"a": pose(4, 4)}) == []
    assert checker.check({"a": pose(4, 2.2)})


def test_layers_can_let_bodies_pass(tmp_path):
    """Layers can let bodies pass."""
    w = world(tmp_path, WALL, robot("a", 2, 3))
    checker = Checker(w, layers={"a": (2, ~1)})
    assert checker.check({"a": pose(4.8, 3)}) == []
    assert Checker(w, layers={"a": (2, ALL)}).check({"a": pose(4.8, 3)})


def test_a_robot_driving_at_a_wall_stops_at_it_and_counts_once(tmp_path):
    """A robot driving at a wall stops at it and counts once."""
    sim = Simulation(world(tmp_path, WALL, robot("a", 2, 3)))
    for _ in range(60):
        sim.step({"a": Twist(vx=1.0)})
    x, y, _ = sim.poses["a"].xyt
    assert x == pytest.approx(4.5, abs=1e-5)
    assert y == pytest.approx(3.0)
    assert [(c.a, c.b) for c in sim.collisions] == [("a", "obstacle 1")]


def test_a_robot_at_an_angle_slides_along_the_wall(tmp_path):
    """A robot at an angle slides along the wall."""
    sim = Simulation(world(tmp_path, WALL, robot("a", 2, 2, math.pi / 4)))
    for _ in range(45):
        sim.step({"a": Twist(vx=1.0)})
    x, y, _ = sim.poses["a"].xyt
    assert x == pytest.approx(4.5, abs=1e-5)
    assert y > 4.5
    assert len(sim.collisions) == 1


def test_two_robots_driving_head_on_both_stop(tmp_path):
    """Two robots driving head on both stop."""
    sim = Simulation(world(tmp_path, robot("a", 2, 3), robot("b", 8, 3, math.pi)))
    for _ in range(80):
        sim.step({"a": Twist(vx=1.0), "b": Twist(vx=1.0)})
    (xa, ya, _), (xb, yb, _) = sim.poses["a"].xyt, sim.poses["b"].xyt
    assert xb - xa == pytest.approx(1.0, abs=1e-5)
    assert xa + xb == pytest.approx(10.0, abs=1e-5)
    assert ya == pytest.approx(3.0) and yb == pytest.approx(3.0)
    assert [(c.a, c.b) for c in sim.collisions] == [("a", "b")]


def test_the_result_does_not_depend_on_the_order_of_the_robots(tmp_path):
    """The result does not depend on the order of the robots."""

    def run(first, second):
        sim = Simulation(world(tmp_path, first, second))
        for _ in range(80):
            sim.step({"a": Twist(vx=1.0), "b": Twist(vx=1.0)})
        return {n: tuple(p.xyt) for n, p in sim.poses.items()}

    a, b = robot("a", 2, 3), robot("b", 8, 3, math.pi)
    one, two = run(a, b), run(b, a)
    for name in one:
        assert one[name] == pytest.approx(two[name])


def test_a_robot_cannot_leave_through_the_border(tmp_path):
    """A robot cannot leave through the border."""
    sim = Simulation(world(tmp_path, robot("a", 2, 3, math.pi)))
    for _ in range(60):
        sim.step({"a": Twist(vx=1.0)})
    assert sim.poses["a"].xyt[0] == pytest.approx(0.5, abs=1e-5)
    assert sim.collisions[0].b == "border"


def test_a_robot_sent_back_is_not_landed_on_by_another(tmp_path):
    """A robot that cannot be pushed clear returns to where it stood.

    A drives into the wall and goes back. B jumps onto the spot A left. Both
    must end at their old poses, with no overlap, in the same step.
    """
    sim = Simulation(
        world(
            tmp_path,
            WALL,
            robot("A", 4.5, 3.0, radius=0.3),
            robot("B", 3.5, 3.0, radius=0.3),
        ),
        dt=1.0,
    )
    sim.step({"A": Twist(0.3), "B": Twist(1.0)})
    assert sim.poses["A"].xyt[:2] == pytest.approx((4.5, 3.0))
    assert sim.poses["B"].xyt[:2] == pytest.approx((3.5, 3.0))
    assert sim.checker.check(sim.poses) == []
