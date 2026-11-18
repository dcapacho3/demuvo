"""Tests for editing the actors of a world."""

import pytest

from demuvo import edit
from demuvo.verify import WorldError


@pytest.fixture(name="world")
def one_actor():
    """A blank world with one actor, "w", standing still in the middle."""
    return edit.add_actor(edit.new_world(), "w", (3.0, 3.0, 0.0))


def test_an_actor_is_added_with_a_disc_body_that_stands_still():
    """The defaults: a disc, solid, not meeting other actors, no motion."""
    world = edit.add_actor(edit.new_world(), "w", (3.0, 3.0, 0.0))
    (actor,) = world.actors
    assert actor.name == "w"
    assert actor.shape.radius == edit.ACTOR_RADIUS
    assert (actor.motion, actor.solid, actor.collide) == (None, True, False)


def test_an_actor_can_be_added_with_a_motion():
    """A script can give the motion and its settings at once."""
    world = edit.add_actor(
        edit.new_world(),
        "w",
        (3.0, 3.0, 0.0),
        motion="demuvo.actors:Waypoints",
        motion_params={"points": [[4.0, 3.0]], "speed": 0.4},
    )
    assert world.actors[0].motion == "demuvo.actors:Waypoints"
    assert world.actors[0].motion_params["speed"] == 0.4


def test_an_actor_added_with_a_motion_that_cannot_run_is_refused():
    """The motion is built then and there, so a bad setting never reaches a world."""
    with pytest.raises(WorldError, match="speed must be a positive"):
        edit.add_actor(
            edit.new_world(),
            "w",
            (3.0, 3.0, 0.0),
            motion="demuvo.actors:Waypoints",
            motion_params={"points": [[4.0, 3.0]], "speed": 0.0},
        )


def test_a_name_a_robot_already_has_is_refused():
    """Robots and actors share one set of names."""
    with pytest.raises(WorldError, match="name used twice"):
        edit.add_actor(edit.new_world(), "r1", (3.0, 3.0, 0.0))


def test_an_actor_that_would_leave_the_world_is_refused(world):
    """Both when it is added and when it is moved."""
    with pytest.raises(WorldError, match="must lie inside the world"):
        edit.add_actor(world, "x", (-5.0, 3.0, 0.0))
    with pytest.raises(WorldError, match="must lie inside the world"):
        edit.move_actor(world, 0, (-5.0, 3.0, 0.0))


def test_an_actor_may_be_moved_resized_and_renamed(world):
    """The three plain edits, each keeping everything else as it was."""
    moved = edit.move_actor(world, 0, (4.0, 2.0, 1.0))
    assert moved.actors[0].pose.xyt == pytest.approx([4.0, 2.0, 1.0])
    assert edit.resize_actor(moved, 0, (0.6,)).actors[0].shape.radius == 0.6
    boxed = edit.resize_actor(moved, 0, (0.6, 0.4))
    assert boxed.actors[0].shape.size == (0.6, 0.4)
    assert edit.rename_actor(moved, 0, " bob ").actors[0].name == "bob"


def test_an_empty_name_is_refused(world):
    """An actor must have a name; it is what a collision reports."""
    with pytest.raises(WorldError, match="a name cannot be empty"):
        edit.rename_actor(world, 0, "   ")


def test_the_two_flags_are_set_by_name(world):
    """`solid` and `collide`, and nothing else."""
    assert edit.set_actor_flag(world, 0, "solid", False).actors[0].solid is False
    assert edit.set_actor_flag(world, 0, "collide", True).actors[0].collide is True
    with pytest.raises(WorldError, match="is not solid or collide"):
        edit.set_actor_flag(world, 0, "speed", True)


def test_a_motion_is_started_off_with_what_a_form_cannot_fill_in(world):
    """Waypoints needs points, so it is given one, a metre on from where it stands."""
    given = edit.set_actor_motion(world, 0, "demuvo.actors:Waypoints")
    assert given.actors[0].motion_params["points"] == [[4.0, 3.0]]
    kept = edit.set_actor_motion_params(given, 0, {"speed": 0.8})
    assert kept.actors[0].motion_params == {"points": [[4.0, 3.0]], "speed": 0.8}
    again = edit.set_actor_motion(kept, 0, "demuvo.actors:Waypoints")
    assert again.actors[0].motion_params["speed"] == 0.8  # its settings are kept


def test_a_motion_can_be_taken_away(world):
    """None leaves an actor that stands where it was put."""
    given = edit.set_actor_motion(world, 0, "demuvo.actors:Waypoints")
    assert edit.set_actor_motion(given, 0, None).actors[0].motion is None
    assert edit.set_actor_motion(given, 0, None).actors[0].motion_params == {}


def test_settings_of_an_actor_with_no_motion_are_refused(world):
    """There is nothing to set."""
    with pytest.raises(WorldError, match="it has no motion to set"):
        edit.set_actor_motion_params(world, 0, {"speed": 1.0})


def test_an_actor_is_removed_by_its_place(world):
    """Unlike a robot, the last actor may go: a world needs none."""
    assert edit.remove_actor(world, 0).actors == ()


def test_a_place_that_is_not_an_actor_is_refused(world):
    """Every edit says which actor it could not find."""
    for call in (
        lambda: edit.remove_actor(world, 3),
        lambda: edit.move_actor(world, 3, (1.0, 1.0, 0.0)),
    ):
        with pytest.raises(WorldError, match="there is no actor"):
            call()


def test_every_name_in_use_is_listed(world):
    """A window asks for it to pick a free name."""
    assert edit.actor_names(world) == {"r1", "w"}


def test_a_free_place_is_never_inside_a_solid_actor():
    """Actors and robots added one after another all get a place of their own."""
    world = edit.new_world()
    for k in range(3):
        pose = edit.free_pose(world, radius=edit.ACTOR_RADIUS)
        world = edit.add_actor(world, f"a{k}", pose)
    for k in range(4):
        world = edit.add_robot(world, f"n{k}", edit.free_pose(world))
    assert len({tuple(a.pose.xyt[:2]) for a in world.actors}) == 3


def test_a_robot_cannot_be_put_inside_a_solid_actor(world):
    """The refusal names the actor."""
    with pytest.raises(WorldError, match="overlap w"):
        edit.add_robot(world, "n", (3.0, 3.0, 0.0))


def test_a_robot_can_be_put_inside_an_actor_that_is_not_solid(world):
    """A robot drives through it, so the overlap is no reason to refuse."""
    world = edit.set_actor_flag(world, 0, "solid", False)
    assert edit.add_robot(world, "n", (3.0, 3.0, 0.0)).robots[-1].name == "n"
