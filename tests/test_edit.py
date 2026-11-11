"""Tests for editing a world."""

import dataclasses
from pathlib import Path

import pytest

from demuvo import edit, load_world
from demuvo.world import WorldError, save_world

DEMO = Path(__file__).resolve().parents[1] / "examples" / "worlds" / "demo.toml"
PURE_PURSUIT = "demuvo.controllers:PurePursuit"


@pytest.fixture(name="world")
def starter():
    """A new world: 10 by 6 m, one robot, r1, in the middle."""
    return edit.new_world()


def names(world):
    """The robot names, in order."""
    return [r.name for r in world.robots]


def test_a_new_world_has_one_robot_and_can_run():
    """The starter robot is in the middle with A* and pure pursuit."""
    world = edit.new_world((8.0, 4.0), seed=3)
    assert (world.size, world.seed, names(world)) == ((8.0, 4.0), 3, ["r1"])
    assert world.robots[0].pose.xyt[:2].tolist() == [4.0, 2.0]
    edit.validate(world)


def test_an_edit_returns_a_new_world_and_leaves_the_old_one(world):
    """The first world still has one robot."""
    more = edit.add_robot(world, "r2", (1.0, 1.0, 0.0))
    assert names(world) == ["r1"]
    assert names(more) == ["r1", "r2"]


def test_a_robot_is_added_with_its_radius_and_parts(world):
    """The radius and the plug-ins come from the call, with defaults."""
    world = edit.add_robot(
        world, "r2", (1.0, 1.0, 0.0), radius=0.3, controller=PURE_PURSUIT
    )
    robot = world.robots[1]
    assert robot.shape.radius == 0.3
    assert robot.planner == "demuvo.planners:AStar"
    assert robot.controller == PURE_PURSUIT


@pytest.mark.parametrize(
    ("name", "pose", "message"),
    [
        ("r1", (1.0, 1.0, 0.0), "name used twice"),
        ("r2", (20.0, 1.0, 0.0), "must lie inside the world"),
        ("r2", (5.0, 3.0, 0.0), "robot 'r2' would overlap robot 'r1'"),
        ("r2", (1.0, 1.0), "'pose' needs 3 finite number"),
    ],
)
def test_a_robot_that_cannot_be_added_is_refused(world, name, pose, message):
    """The message names the robot and what is wrong."""
    with pytest.raises(WorldError, match=message):
        edit.add_robot(world, name, pose)


def test_a_robot_cannot_start_inside_an_obstacle():
    """The message says which obstacle."""
    world = load_world(DEMO)
    x, y, _ = world.obstacles[0].pose.xyt
    with pytest.raises(WorldError, match="would overlap obstacle 1"):
        edit.add_robot(world, "r2", (x, y, 0.0))


def test_free_pose_is_a_place_where_a_robot_fits(world):
    """The place can take a new robot, and it is not the middle, which is taken."""
    pose = edit.free_pose(world)
    assert pose[:2] != (5.0, 3.0)
    assert len(edit.add_robot(world, "r2", pose).robots) == 2


def test_free_pose_says_so_when_the_world_is_full():
    """A one-metre world holds the starter robot and nothing else."""
    with pytest.raises(WorldError, match="no free place"):
        edit.free_pose(edit.new_world((1.0, 1.0)))


def test_a_robot_is_removed_with_its_goals(world):
    """The other robot's goal stays."""
    world = edit.add_robot(world, "r2", (1.0, 1.0, 0.0))
    world = edit.set_goal(edit.set_goal(world, "r1", (8.0, 1.0, 0.0)), "r2", (2, 2, 0))
    world = edit.remove_robot(world, "r1")
    assert names(world) == ["r2"]
    assert [t.robot for t in world.tasks] == ["r2"]


def test_the_last_robot_cannot_be_removed(world):
    """A world needs one."""
    with pytest.raises(WorldError, match="only robot"):
        edit.remove_robot(world, "r1")


def test_a_robot_is_moved(world):
    """The pose changes to the given one."""
    moved = edit.set_pose(world, "r1", (2.0, 2.0, 1.0))
    assert moved.robots[0].pose.xyt.tolist() == pytest.approx([2.0, 2.0, 1.0])


@pytest.mark.parametrize(
    ("pose", "message"),
    [((20.0, 2.0, 0.0), "must lie inside the world"), ((2.0, 2.0), "needs 3")],
)
def test_a_bad_move_is_refused(world, pose, message):
    """Outside the world, or not three numbers."""
    with pytest.raises(WorldError, match=message):
        edit.set_pose(world, "r1", pose)


def test_a_robot_cannot_be_moved_onto_another(world):
    """Dragging over a robot is refused, and the message names both."""
    world = edit.add_robot(world, "r2", (1.0, 1.0, 0.0))
    with pytest.raises(WorldError, match="robot 'r2' would overlap robot 'r1'"):
        edit.set_pose(world, "r2", (5.1, 3.0, 0.0))


def test_a_goal_replaces_the_earlier_one(world):
    """One goal per robot, and goal_of gives it back."""
    assert edit.goal_of(world, "r1") is None
    world = edit.set_goal(world, "r1", (8.0, 1.0, 0.5), tolerance=(0.2, 0.3))
    world = edit.set_goal(world, "r1", (7.0, 2.0, 0.0))
    assert len(world.tasks) == 1
    assert edit.goal_of(world, "r1") == pytest.approx((7.0, 2.0, 0.0))


def test_a_goal_outside_the_world_is_refused(world):
    """The message names the goal."""
    with pytest.raises(WorldError, match="'goal' must lie inside the world"):
        edit.set_goal(world, "r1", (30.0, 1.0, 0.0))


def test_an_unknown_robot_is_refused_by_every_edit(world):
    """The same plain message each time."""
    for call in (
        lambda: edit.remove_robot(world, "x"),
        lambda: edit.set_pose(world, "x", (1, 1, 0)),
        lambda: edit.set_goal(world, "x", (1, 1, 0)),
        lambda: edit.set_model(world, "x", "diff_drive"),
        lambda: edit.set_part(world, "x", "planner", None),
        lambda: edit.set_settings(world, "x", "planner", {}),
        lambda: edit.settings_of_robot(world, "x", "planner"),
        lambda: edit.set_control_period(world, "x", 0.1),
    ):
        with pytest.raises(WorldError, match="no robot named 'x'"):
            call()


def test_a_model_change_resets_the_model_settings(world):
    """Settings of the old model do not carry over."""
    world = edit.set_settings(world, "r1", "model", {"max_v": 0.5})
    assert edit.settings_of_robot(world, "r1", "model") == {"max_v": 0.5}
    world = edit.set_model(world, "r1", "diff_drive")
    assert not edit.settings_of_robot(world, "r1", "model")


def test_an_unknown_model_is_refused(world):
    """The message lists the models."""
    with pytest.raises(WorldError, match="unknown model 'car', expected one of"):
        edit.set_model(world, "r1", "car")


def test_a_part_is_chosen_and_starts_with_default_settings(world):
    """A new smoother, no controller, and the default planner."""
    world = edit.set_settings(world, "r1", "controller", {"lookahead": 0.9})
    smoother = "demuvo.smoothers.line_of_sight:LineOfSight"
    world = edit.set_part(world, "r1", "smoother", smoother)
    world = edit.set_part(world, "r1", "controller", PURE_PURSUIT)
    world = edit.set_part(world, "r1", "planner", None)
    robot = world.robots[0]
    assert (robot.smoother, robot.planner) == (smoother, None)
    assert not edit.settings_of_robot(world, "r1", "controller")
    assert edit.set_part(world, "r1", "controller", None).robots[0].controller is None


@pytest.mark.parametrize(
    ("part", "path", "message"),
    [
        ("planner", "AStar", "planner must be 'module:Class'"),
        ("planner", "no.such:Thing", "cannot load planner 'no.such:Thing'"),
        (
            "smoother",
            "demuvo.planners:AStar",
            "smoother 'demuvo.planners:AStar' has no",
        ),
        ("wheel", "a:B", "'wheel' is not a part"),
    ],
)
def test_a_part_that_cannot_be_chosen_is_refused(world, part, path, message):
    """A bad path, a class that is not a smoother, and a part that does not exist."""
    with pytest.raises(WorldError, match=message):
        edit.set_part(world, "r1", part, path)


def test_settings_are_set_and_read_back(world):
    """An int is accepted for a number, and a number is stored as a float."""
    world = edit.set_settings(world, "r1", "controller", {"lookahead": 1})
    got = edit.settings_of_robot(world, "r1", "controller")
    assert got == {"lookahead": 1.0}
    assert isinstance(got["lookahead"], float)
    edit.validate(world)


def test_the_planner_settings_are_those_of_the_default_when_none_is_named(world):
    """A robot with no planner still shows the settings of the default one."""
    world = edit.set_part(world, "r1", "planner", None)
    world = edit.set_settings(world, "r1", "planner", {"spacing": 0.2})
    assert edit.settings_of_robot(world, "r1", "planner") == {"spacing": 0.2}


@pytest.mark.parametrize(
    ("part", "settings", "message"),
    [
        ("controller", {"speeed": 1.0}, r"controller has no setting 'speeed'; it has"),
        ("controller", {"lookahead": "far"}, "'lookahead' must be a number, got 'far'"),
        ("controller", {"lookahead": True}, "'lookahead' must be a number, got True"),
        ("controller", {"lookahead": float("inf")}, "must be a number, got inf"),
        ("controller", {"rotate_to_goal": 1}, "'rotate_to_goal' must be true or false"),
        ("controller", {"lookahead": -1.0}, "cannot load controller"),
        ("planner", {"connectivity": 8.5}, "'connectivity' must be a whole number"),
        ("planner", {"connectivity": True}, "must be a whole number, got True"),
        ("model", {"max_v": -1.0}, "robot 'r1' params: 'max_v' must be positive"),
        ("model", {"max_v": 1.0, "track": -0.5}, "must be positive"),
        ("model", {"speed": 1.0}, "model has no setting 'speed'"),
        ("wheel", {}, "'wheel' is not a part"),
    ],
)
def test_a_bad_setting_is_refused_in_plain_words(world, part, settings, message):
    """The message names the robot, the part and the setting."""
    with pytest.raises(WorldError, match=message):
        edit.set_settings(world, "r1", part, settings)


def test_a_part_that_is_not_there_has_no_settings(world):
    """A robot driven from outside has no controller to set."""
    world = edit.set_part(world, "r1", "controller", None)
    with pytest.raises(WorldError, match="has no controller, so it has nothing"):
        edit.set_settings(world, "r1", "controller", {})


def test_the_control_period_is_set(world):
    """A positive number is stored; zero is refused."""
    assert edit.set_control_period(world, "r1", 0.2).robots[0].control_period == 0.2
    with pytest.raises(WorldError, match="'control_period' must be positive"):
        edit.set_control_period(world, "r1", 0.0)


def test_validate_names_a_control_period_below_the_step(world):
    """A controller cannot run twice in one step."""
    world = edit.set_control_period(world, "r1", 0.001)
    with pytest.raises(WorldError, match="control_period 0.001 is below dt"):
        edit.validate(world)


def test_validate_refuses_robots_that_overlap(world):
    """A world built by hand with two robots on the same spot."""
    world = edit.add_robot(world, "r2", (1.0, 1.0, 0.0))
    stacked = dataclasses.replace(world.robots[1], pose=world.robots[0].pose)
    world = dataclasses.replace(world, robots=(world.robots[0], stacked))
    with pytest.raises(ValueError, match="starts inside"):
        edit.validate(world)


def test_an_edited_world_can_be_saved_and_loaded(world, tmp_path):
    """The edits come back from a file."""
    world = edit.add_robot(world, "r2", edit.free_pose(world))
    world = edit.set_goal(world, "r2", (8.0, 1.0, 0.0))
    world = edit.set_settings(world, "r2", "controller", {"lookahead": 0.9})
    path = tmp_path / "w.toml"
    save_world(world, path)
    again = load_world(path)
    assert names(again) == ["r1", "r2"]
    assert edit.goal_of(again, "r2") == pytest.approx((8.0, 1.0, 0.0))
    assert edit.settings_of_robot(again, "r2", "controller") == {"lookahead": 0.9}


def test_obstacles_are_added_moved_resized_and_removed(world):
    """Each edit returns a new world; the old one stays as it was."""
    box = edit.add_obstacle(world, "box", (2.0, 2.0, 0.0), (1.0, 0.5))
    both = edit.add_obstacle(box, "circle", (7.0, 4.0, 0.0), (0.4,))
    assert [o.shape.name for o in both.obstacles] == ["box", "circle"]
    assert not world.obstacles
    moved = edit.move_obstacle(both, 0, (3.0, 2.0, 0.5))
    assert [float(v) for v in moved.obstacles[0].pose.xyt] == [3.0, 2.0, 0.5]
    wider = edit.resize_obstacle(moved, 0, (2.0, 0.5))
    assert wider.obstacles[0].shape.size == (2.0, 0.5)
    rounder = edit.resize_obstacle(wider, 1, (0.6,))
    assert rounder.obstacles[1].shape.radius == 0.6
    assert [o.shape.name for o in edit.remove_obstacle(rounder, 0).obstacles] == [
        "circle"
    ]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda w: edit.add_obstacle(w, "star", (1, 1, 0), (1,)), "unknown obstacle"),
        (lambda w: edit.add_obstacle(w, "box", (99, 1, 0), (1, 1)), "inside the world"),
        (lambda w: edit.add_obstacle(w, "box", (5, 3, 0), (1, 1)), "overlap"),
        (lambda w: edit.move_obstacle(w, 5, (1, 1, 0)), "no obstacle number 6"),
        (lambda w: edit.move_obstacle(w, 0, (99, 1, 0)), "inside the world"),
        (lambda w: edit.move_obstacle(w, 0, (5, 3, 0)), "overlap"),
        (lambda w: edit.remove_obstacle(w, 3), "no obstacle number 4"),
    ],
)
def test_a_bad_obstacle_edit_is_refused_in_plain_words(world, call, message):
    """Outside the world, onto a robot, or a missing number gives a WorldError."""
    with_one = edit.add_obstacle(world, "box", (2.0, 2.0, 0.0), (1.0, 0.5))
    with pytest.raises(WorldError, match=message):
        call(with_one)


def test_the_world_size_changes_if_everything_still_fits(world):
    """Growing always works; shrinking past a robot, obstacle or goal is refused."""
    assert edit.set_size(world, (20.0, 8.0)).size == (20.0, 8.0)
    with pytest.raises(WorldError, match="robot 'r1'"):
        edit.set_size(world, (2.0, 2.0))
    boxed = edit.add_obstacle(world, "box", (9.0, 1.0, 0.0), (1.0, 1.0))
    with pytest.raises(WorldError, match="an obstacle"):
        edit.set_size(boxed, (8.0, 6.0))
    goal = edit.set_goal(world, "r1", (9.0, 3.0, 0.0))
    with pytest.raises(WorldError, match="goal of 'r1'"):
        edit.set_size(goal, (8.0, 6.0))
