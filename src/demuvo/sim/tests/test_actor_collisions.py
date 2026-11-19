"""Tests for how robots and actors meet: solid, collide, and what is only recorded."""

import pytest
from robomath import Transform2D

from demuvo import Simulation, Twist, load_world
from demuvo.sim.collisions import Checker

from .worlds import HEAD, robot


def actor(name, x, y, radius=0.5, motion="", **flags):
    """An `[[actor]]` block: a disc, standing still unless a motion is given."""
    lines = "".join(f"{k} = {str(v).lower()}\n" for k, v in flags.items())
    return f"""
[[actor]]
name = "{name}"
shape = {{ name = "circle", radius = {radius} }}
pose = [{x}, {y}, 0.0]
{lines}{motion}"""


def world(tmp_path, *parts):
    """The world of these blocks."""
    path = tmp_path / "w.toml"
    path.write_text(HEAD + "".join(parts))
    return load_world(path)


def pose(x, y, angle=0.0):
    """A pose."""
    return Transform2D.from_xyt((x, y, angle))


def driving(tmp_path, *parts, speed=1.0, steps=20):
    """Run a world with "r" driving along +x, and give back the simulation."""
    sim = Simulation(load_world_text(tmp_path, parts))
    for _ in range(steps):
        sim.step({"r": Twist(speed, 0.0, 0.0)})
    return sim


def load_world_text(tmp_path, parts):
    """The world of these blocks, from a file of its own."""
    path = tmp_path / "drive.toml"
    path.write_text(HEAD + "".join(parts))
    return load_world(path)


def test_a_robot_meets_an_actor_and_the_actor_is_never_moved(tmp_path):
    """The contact names the robot first and says b does not move."""
    checker = Checker(world(tmp_path, robot("r", 2, 3), actor("w", 2.6, 3)))
    (contact,) = checker.check({"r": pose(2, 3)}, actors={"w": pose(2.6, 3)})
    assert (contact.a, contact.b) == ("r", "w")
    assert contact.b_moves is False
    assert contact.pushes is True
    assert contact.depth == pytest.approx(0.4)


def test_an_actor_the_checker_is_not_given_is_not_there(tmp_path):
    """An actor is tested only when its pose is passed in."""
    checker = Checker(world(tmp_path, robot("r", 2, 3), actor("w", 2.6, 3)))
    assert checker.check({"r": pose(2, 3)}) == []


def test_a_robot_passes_through_an_actor_that_is_not_solid(tmp_path):
    """The contact is still reported, with `pushes` false so nothing is moved."""
    checker = Checker(
        world(tmp_path, robot("r", 2, 3), actor("w", 2.6, 3, solid=False))
    )
    (contact,) = checker.check({"r": pose(2, 3)}, actors={"w": pose(2.6, 3)})
    assert contact.pushes is False
    assert contact.depth == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("first", "second", "tested"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
def test_two_actors_meet_only_when_both_carry_collide(tmp_path, first, second, tested):
    """One of the two saying no is enough to keep them apart in the checker."""
    checker = Checker(
        world(
            tmp_path,
            robot("r", 8, 5),
            actor("a", 2, 3, collide=first),
            actor("b", 2.6, 3, collide=second),
        )
    )
    contacts = checker.check(
        {"r": pose(8, 5)}, actors={"a": pose(2, 3), "b": pose(2.6, 3)}
    )
    assert bool(contacts) is tested
    if tested:
        (contact,) = contacts
        assert (contact.a, contact.b) == ("a", "b")
        assert contact.b_moves is False
        assert contact.pushes is False  # recorded only; neither is moved


def test_an_actor_is_not_tested_against_the_obstacles_or_the_border(tmp_path):
    """Where a motion takes an actor is the world file's business."""
    box = """
[[obstacle]]
pose = [3.0, 3.0, 0.0]
shape = { name = "box", size = [1.0, 1.0] }
"""
    checker = Checker(world(tmp_path, robot("r", 8, 5), actor("w", 7, 3), box))
    assert checker.check({"r": pose(8, 5)}, actors={"w": pose(3.0, 3.0)}) == []
    assert checker.check({"r": pose(8, 5)}, actors={"w": pose(0.1, 0.1)}) == []


STILL = ""
# Drives along +x fast enough to be in the robot's way from the start.
CROSS = (
    'motion = "demuvo.actors:Waypoints"\n'
    'motion_params = { points = [[9.0, 3.0]], speed = 0.2, mode = "once" }\n'
)


def test_a_robot_is_stopped_by_a_solid_actor(tmp_path):
    """It drives into the actor and is pushed back out, so it ends short of it."""
    sim = driving(tmp_path, robot("r", 2, 3), actor("w", 5, 3), steps=40)
    assert sim.poses["r"].xyt[0] == pytest.approx(4.0, abs=0.01)  # 5 - 0.5 - 0.5
    assert [(c.a, c.b) for c in sim.collisions] == [("r", "w")]


def test_a_robot_drives_through_an_actor_that_is_not_solid(tmp_path):
    """It keeps its speed, and the collision is counted once."""
    sim = driving(tmp_path, robot("r", 2, 3), actor("w", 5, 3, solid=False), steps=40)
    assert sim.poses["r"].xyt[0] == pytest.approx(6.0, abs=0.01)  # nothing slowed it
    assert [(c.a, c.b) for c in sim.collisions] == [("r", "w")]


def test_a_collision_with_an_actor_is_counted_once_while_it_lasts(tmp_path):
    """The robot presses on the actor for many steps and it stays one collision."""
    sim = driving(tmp_path, robot("r", 2, 3), actor("w", 5, 3), steps=60)
    assert len(sim.collisions) == 1
    assert sim.collisions[0].b == "w"


def test_a_moving_actor_pushes_nothing_but_the_robot_is_kept_out_of_it(tmp_path):
    """The robot is resolved against where the actor is at the end of the step."""
    sim = driving(tmp_path, robot("r", 2, 3), actor("w", 5, 3, motion=CROSS), steps=40)
    gap = sim.actor_poses()["w"].xyt[0] - sim.poses["r"].xyt[0]
    assert gap == pytest.approx(1.0, abs=0.02)  # the two radii, never overlapping


def test_two_actors_that_collide_are_recorded_but_keep_their_motions(tmp_path):
    """Neither is moved: an actor's pose comes from its motion alone."""
    sim = driving(
        tmp_path,
        robot("r", 8, 5),
        actor("a", 2, 3, motion=CROSS, collide=True),
        actor("b", 3, 3, collide=True),
        speed=0.0,
        steps=60,
    )
    assert sim.actor_poses()["a"].xyt[0] == pytest.approx(2.0 + 0.2 * 6.0, abs=1e-9)
    assert sim.actor_poses()["b"].xyt[:2] == pytest.approx([3.0, 3.0])
    assert [(c.a, c.b) for c in sim.collisions] == [("a", "b")]


def test_a_robot_that_starts_inside_an_actor_is_refused(tmp_path):
    """A world file should not start in a collision."""
    with pytest.raises(ValueError, match="'r' starts inside 'w'"):
        Simulation(world(tmp_path, robot("r", 3, 3), actor("w", 3.2, 3)))


def test_a_robot_may_start_inside_an_actor_that_is_not_solid(tmp_path):
    """Nothing is pushed, so there is nothing to refuse."""
    sim = Simulation(world(tmp_path, robot("r", 3, 3), actor("w", 3.2, 3, solid=False)))
    assert sim.poses["r"].xyt[:2] == pytest.approx([3.0, 3.0])


def test_two_actors_may_start_on_top_of_each_other(tmp_path):
    """Their contact is recorded while the run goes on, never refused."""
    sim = Simulation(
        world(
            tmp_path,
            robot("r", 8, 5),
            actor("a", 3, 3, collide=True),
            actor("b", 3.2, 3, collide=True),
        )
    )
    sim.step({})
    assert [(c.a, c.b) for c in sim.collisions] == [("a", "b")]
