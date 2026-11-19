"""Tests for the `[[actor]]` tables and the `[actors]` defaults."""

import pytest

from demuvo.world import ActorDefaults, ActorSpec, WorldError

from .small_world import small_world

ONE = """
[[actor]]
name = "walker"
shape = { name = "circle", radius = 0.25 }
pose = [2.0, 2.0, 0.0]
"""


def actors_of(tmp_path, extra):
    """The actors the small world gets from the extra text."""
    return small_world(tmp_path, extra).actors


def test_a_world_with_no_actor_table_has_none(tmp_path):
    """Actors are optional, and every world written before them still loads."""
    world = small_world(tmp_path, "")
    assert not world.actors
    assert world.actor_defaults == ActorDefaults()


def test_an_actor_is_read_with_its_shape_pose_and_motion(tmp_path):
    """The four keys that describe the body and how it moves."""
    (actor,) = actors_of(
        tmp_path,
        ONE + 'motion = "demuvo.actors:Waypoints"\nmotion_params = { speed = 0.6 }\n',
    )
    assert isinstance(actor, ActorSpec)
    assert actor.name == "walker"
    assert actor.shape.radius == 0.25
    assert actor.pose.xyt == pytest.approx([2.0, 2.0, 0.0])
    assert actor.motion == "demuvo.actors:Waypoints"
    assert actor.motion_params == {"speed": 0.6}


def test_an_actor_with_no_motion_stands_still(tmp_path):
    """A motion is optional: without one the actor is an obstacle off the costmap."""
    (actor,) = actors_of(tmp_path, ONE)
    assert actor.motion is None
    assert actor.motion_params == {}


def test_solid_is_true_and_collide_is_false_by_default(tmp_path):
    """A robot is pushed out of an actor, and actors ignore each other."""
    (actor,) = actors_of(tmp_path, ONE)
    assert actor.solid is True
    assert actor.collide is False


def test_an_actor_sets_its_own_flags(tmp_path):
    """Both flags can be changed on the actor itself."""
    (actor,) = actors_of(tmp_path, ONE + "solid = false\ncollide = true\n")
    assert actor.solid is False
    assert actor.collide is True


def test_the_actors_table_sets_the_default_for_every_actor(tmp_path):
    """`[actors]` is the world's answer for an actor that does not give one."""
    first, second = actors_of(
        tmp_path,
        "[actors]\nsolid = false\ncollide = true\n"
        + ONE
        + ONE.replace("walker", "other")
        + "solid = true\n",
    )
    assert (first.solid, first.collide) == (False, True)
    assert (second.solid, second.collide) == (True, True)


def test_the_world_keeps_the_defaults_it_was_given(tmp_path):
    """So a world written back says what the file said."""
    world = small_world(tmp_path, "[actors]\ncollide = true\n" + ONE)
    assert world.actor_defaults == ActorDefaults(solid=True, collide=True)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (ONE + ONE, "actor 'walker': name used twice"),
        (ONE.replace("walker", "a"), "actor 'a': name used twice"),
        (ONE.replace('name = "walker"', ""), "actor 1: missing 'name'"),
        (ONE.replace("pose = [2.0, 2.0, 0.0]", ""), "actor 'walker': missing 'pose'"),
        (ONE + "speed = 1.0\n", "actor 'walker': unknown key 'speed'"),
        (ONE + "solid = 1\n", "actor 'walker': 'solid' must be true or false"),
        (ONE + "collide = 1\n", "actor 'walker': 'collide' must be true or false"),
        (ONE + 'motion = "Waypoints"\n', "actor 'walker': 'motion' must be"),
        (
            ONE + "motion_params = { speed = 1.0 }\n",
            "actor 'walker': 'motion_params' needs 'motion'",
        ),
        (
            ONE.replace("pose = [2.0, 2.0, 0.0]", "pose = [9.0, 2.0, 0.0]"),
            "actor 'walker': 'pose' must lie inside the world",
        ),
        (
            ONE.replace(
                '{ name = "circle", radius = 0.25 }',
                '{ name = "linestring", vertices = [[1.0, 1.0], [2.0, 1.0]] }',
            ),
            "actor 'walker'",
        ),
        ("[actors]\nsolid = 1\n", "actors: 'solid' must be true or false"),
        ("[actors]\npush = true\n", "actors: unknown key 'push'"),
    ],
)
def test_a_wrong_actor_is_refused_by_name_and_key(tmp_path, extra, message):
    """The message names the actor and the key."""
    with pytest.raises(WorldError, match=message):
        small_world(tmp_path, extra)
