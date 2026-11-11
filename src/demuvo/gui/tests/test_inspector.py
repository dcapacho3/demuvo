"""Tests for the inspector: its sections, its edits and its lock."""

import dataclasses
import math

import pytest

from demuvo import edit
from demuvo.gui.inputs import Combo, Spin
from demuvo.gui.inspector import HINT, Inspector
from demuvo.gui.robotform import NO_GOAL
from demuvo.gui.session import Session
from demuvo.gui.widgets import Checkbox, Column, Label
from demuvo.gui.worldform import NO_SIZE
from demuvo.world.shapes import Polygon

from .fakes import EXAMPLES, FakeHost


def build(session=None, name="r1"):
    """An inspector over a blank sandbox, showing a robot, with a message log."""
    session = session or Session()
    column = Column()
    column.set_rect(0, 0, 312, 600)
    host = FakeHost()
    column.adopt(host)
    reports, refreshed = [], []
    footer = [Checkbox("rays")]

    def refresh():
        """Note it, and show the form again as the window does."""
        refreshed.append(1)
        inspector.show(session.scene.selected)

    inspector = Inspector(column, session, reports.append, refresh, footer)
    session.select(name)
    inspector.show(name)
    return inspector, session, reports, refreshed


def texts(column):
    """The text of every label in the column."""
    return [c.text for c in column.children if isinstance(c, Label)]


def test_with_nothing_selected_the_inspector_shows_a_hint_and_the_footer():
    """The world form and the hint; the footer widgets come last."""
    inspector, session, _, _ = build(name=None)
    session.select(None)
    inspector.show(None)
    assert HINT in texts(inspector.column)
    assert {"width", "height", "seed", "dt", "ox0", "oy0"} <= set(inspector.fields)
    assert inspector.column.children[-1] is inspector.footer[0]
    assert inspector.name is None


def test_a_robot_gets_pose_goal_choices_settings_and_a_control_period():
    """Every field of the form exists, with the robot's values."""
    inspector, session, _, _ = build()
    assert set(inspector.fields) == {
        "x", "y", "th", "gx", "gy", "gth",
        "model", "planner", "smoother", "controller", "period",
        "body", "radius", "estimator", "control_pose", "add_goal",
        "add_encoders", "add_lidar",
    }  # fmt: skip
    f = inspector.fields
    assert (f["x"].value, f["y"].value, f["th"].value) == (5.0, 3.0, 0.0)
    assert f["planner"].current == "A*" and f["controller"].current == "Pure pursuit"
    assert f["smoother"].current == "none" and f["model"].current == "diff_drive"
    assert f["period"].value == session.world.robots[0].control_period
    assert set(inspector.settings) >= {"model", "planner", "controller"}
    assert NO_GOAL in texts(inspector.column)
    assert inspector.column.children[-1] is inspector.footer[0]


def test_the_goal_fields_start_at_the_robots_place_until_it_has_a_goal():
    """After a goal is set the hint is gone and the fields show the goal."""
    inspector, session, _, _ = build()
    assert (inspector.fields["gx"].value, inspector.fields["gy"].value) == (5.0, 3.0)
    session.set_goal("r1", (8.0, 1.0, math.pi / 2))
    inspector.show("r1")
    assert NO_GOAL not in texts(inspector.column)
    f = inspector.fields
    assert (f["gx"].value, f["gy"].value, f["gth"].value) == (8.0, 1.0, 90.0)


def test_the_settings_fields_have_the_kind_of_each_setting():
    """A switch for a bool, a whole number for an int, a number for a float."""
    inspector, _, _, _ = build()
    controller, planner = (
        inspector.settings["controller"],
        inspector.settings["planner"],
    )
    assert isinstance(controller["rotate_to_goal"], Checkbox)
    assert (
        isinstance(planner["connectivity"], Spin)
        and planner["connectivity"].spec.integer
    )
    assert (
        isinstance(controller["lookahead"], Spin)
        and not controller["lookahead"].spec.integer
    )


def test_editing_the_pose_moves_the_robot_and_refreshes():
    """A new x is applied through the session, and the window is told."""
    inspector, session, reports, refreshed = build()
    inspector.fields["x"].set(2.5)
    assert session.run.poses["r1"][0] == pytest.approx(2.5)
    inspector.fields["th"].set(90.0)
    assert session.run.poses["r1"][2] == pytest.approx(math.pi / 2)
    assert refreshed and not reports


def test_a_pose_that_cannot_be_used_is_reported_and_the_fields_show_the_truth_again():
    """Onto another robot: the message names both, and x goes back."""
    inspector, session, reports, _ = build()
    session.add_robot()
    x2, y2 = session.run.poses["r2"][:2]
    session.select("r1")
    inspector.show("r1")
    inspector.fields["x"].set(x2)
    inspector.fields["y"].set(y2)
    assert reports and "would overlap" in reports[-1]
    assert inspector.fields["x"].value == pytest.approx(session.run.poses["r1"][0])


def test_editing_the_goal_sets_it_through_the_session():
    """Three fields make one goal; the heading is in degrees."""
    inspector, session, _, refreshed = build()
    inspector.fields["gx"].set(8.0)
    inspector.show("r1")
    inspector.fields["gth"].set(45.0)
    goal = edit.goal_of(session.world, "r1")
    assert goal[0] == pytest.approx(8.0) and goal[2] == pytest.approx(math.pi / 4)
    assert refreshed


def test_choosing_a_part_changes_the_world_and_rebuilds_the_settings():
    """A straight-line planner has one setting, spacing."""
    inspector, session, _, refreshed = build()
    inspector.fields["planner"].choose("Straight line")
    assert session.world.robots[0].planner == "demuvo.planners:StraightLine"
    assert set(inspector.settings["planner"]) == {"spacing"}
    assert refreshed


def test_editing_a_setting_applies_it_and_a_bad_one_is_reported():
    """A switch, a whole number and a number; then a value the part refuses."""
    inspector, session, reports, _ = build()
    inspector.settings["controller"]["lookahead"].set(0.9)
    inspector.settings["controller"]["rotate_to_goal"].set(False)
    inspector.settings["planner"]["connectivity"].set(4)
    robot = session.world.robots[0]
    assert robot.controller_params["lookahead"] == pytest.approx(0.9)
    assert robot.controller_params["rotate_to_goal"] is False
    assert robot.planner_params["connectivity"] == 4
    inspector.settings["controller"]["lookahead"].set(0.0)
    assert reports and "cannot load controller" in reports[-1]
    assert inspector.settings["controller"]["lookahead"].value == pytest.approx(0.9)


def test_editing_the_control_period_applies_it_and_a_bad_one_is_reported():
    """0.1 s is good; 0.001 s is below the step."""
    inspector, session, reports, _ = build()
    inspector.fields["period"].set(0.1)
    assert session.world.robots[0].control_period == 0.1
    inspector.fields["period"].set(0.002)
    assert reports and "below dt" in reports[-1]


def test_while_a_run_is_in_progress_the_fields_are_off_and_the_note_says_why():
    """After a step nothing can be edited, and a typed value changes nothing."""
    inspector, session, reports, refreshed = build()
    session.step_once()
    inspector.lock()
    assert inspector.note.visible and "run is in progress" in inspector.note.label.text
    assert not any(w.enabled for w in inspector.fields.values())
    assert not any(w.enabled for g in inspector.settings.values() for w in g.values())
    before = session.world
    inspector.fields["x"].set(2.0)
    inspector.fields["gx"].set(3.0)
    inspector.settings["controller"]["lookahead"].set(0.9)
    inspector.fields["period"].set(0.2)
    inspector.fields["planner"].choose("Straight line")
    assert session.world is before and not reports and not refreshed


def test_a_benchmark_shows_the_fixed_world_note_and_an_editing_sandbox_shows_none():
    """Three states, three notes."""
    bench, _, _, _ = build(Session(EXAMPLES / "demo.toml", "benchmark"))
    assert "benchmark world" in bench.note.label.text
    edit_mode, _, _, _ = build()
    assert not edit_mode.note.visible


def test_the_pose_fields_follow_a_drag():
    """While a robot is dragged the fields show where it is."""
    inspector, session, _, _ = build()
    session.scene.drag("r1", 2.0, 1.5, math.pi)
    inspector.sync_pose()
    f = inspector.fields
    assert (f["x"].value, f["y"].value, f["th"].value) == (2.0, 1.5, 180.0)
    inspector.show(None)
    inspector.sync_pose()


def test_a_combo_that_is_a_choice_lists_the_friendly_names():
    """The planner box lists the names, not the paths."""
    inspector, _, _, _ = build()
    box = inspector.fields["planner"]
    assert isinstance(box, Combo) and box.items == ["Straight line", "A*"]


def test_the_world_form_changes_the_size_seed_and_step():
    """Each field applies its edit through the session and refreshes."""
    inspector, session, reports, refreshed = build(name=None)
    session.select(None)
    inspector.show(None)
    inspector.fields["width"].set(12.0)
    inspector.fields["height"].set(8.0)
    assert session.world.size == (12.0, 8.0)
    inspector.fields["seed"].set(9)
    inspector.fields["dt"].set(0.025)
    assert (session.world.seed, session.world.dt) == (9, 0.025)
    assert refreshed and not reports
    inspector.fields["width"].set(1.0)
    assert "robot 'r1'" in reports[-1]
    assert inspector.fields["width"].value == 12.0


def test_an_obstacle_form_moves_and_resizes_the_selected_obstacle():
    """Box and circle show their sizes; a polygon says it is drawn in the file."""
    inspector, session, reports, _ = build(name=None)
    session.edits.add_obstacle("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    inspector.show(None)
    assert set(inspector.fields) == {"ox", "oy", "oth", "size0", "size1"}
    inspector.fields["ox"].set(3.0)
    inspector.fields["oth"].set(90.0)
    inspector.fields["size0"].set(2.0)
    box = session.world.obstacles[0]
    assert box.pose.xyt[0] == 3.0 and box.shape.size == (2.0, 0.5)
    inspector.fields["ox"].set(99.0)
    assert reports and "inside the world" in reports[-1]
    session.edits.delete_obstacle()
    session.edits.add_obstacle("circle", (2.0, 2.0, 0.0), (0.3,))
    inspector.show(None)
    assert set(inspector.fields) == {"ox", "oy", "oth", "size0"}
    inspector.fields["size0"].set(0.5)
    assert session.world.obstacles[0].shape.radius == 0.5


def test_a_polygon_has_no_size_form_but_turns_about_its_middle():
    """The note says where corners are set; the pose fields are the middle."""
    world = edit.add_outline(edit.new_world(), "polygon", [(1, 1), (3, 1), (2, 3)])
    inspector, session, _, _ = build(Session(world), name=None)
    session.select_obstacle(0)
    inspector.show(None)
    assert NO_SIZE in texts(inspector.column)
    assert set(inspector.fields) == {"ox", "oy", "oth"}
    assert inspector.fields["ox"].value == pytest.approx(2.0)
    inspector.fields["oth"].set(90.0)
    assert edit.outline_center(session.world, 0)[:2] == pytest.approx((2.0, 1.6666667))
    assert session.world.obstacles[0].pose.xyt[2] == pytest.approx(math.pi / 2)
    inspector.fields["ox"].set(5.0)
    assert edit.outline_center(session.world, 0)[0] == pytest.approx(5.0)
    session.scene.selected_obstacle = 0
    inspector.sync_pose()
    assert inspector.fields["ox"].value == pytest.approx(5.0)


def test_a_line_has_a_length_and_a_rotation_about_its_middle():
    """Length scales about the middle; rotation turns about it."""
    world = edit.add_outline(edit.new_world(), "linestring", [(2, 2), (4, 2)])
    inspector, session, _, _ = build(Session(world), name=None)
    session.select_obstacle(0)
    inspector.show(None)
    assert set(inspector.fields) == {"ox", "oy", "oth", "size0"}
    assert inspector.fields["size0"].value == pytest.approx(2.0)
    inspector.fields["size0"].set(4.0)
    inspector.fields["oth"].set(90.0)
    shape = session.world.obstacles[0]
    ends = shape.shape.points(shape.pose)
    assert [v for p in ends for v in p] == pytest.approx([3.0, 0.0, 3.0, 4.0], abs=1e-9)


def test_a_robot_body_changes_kind_and_size_through_the_form():
    """Disc to box keeps the reach; the size fields apply; a refusal is reported."""
    inspector, session, reports, _ = build()
    inspector.fields["radius"].set(0.3)
    assert session.world.robots[0].shape.radius == 0.3
    inspector.fields["body"].choose("box")
    assert session.world.robots[0].shape.size == (0.6, 0.6)
    inspector.fields["length"].set(1.0)
    inspector.fields["width"].set(0.5)
    assert session.world.robots[0].shape.size == (1.0, 0.5)
    inspector.fields["body"].choose("circle")
    assert session.world.robots[0].shape.radius == 0.5
    inspector.fields["body"].choose("circle")
    inspector.fields["radius"].set(50.0)
    assert reports and "leave the world" in reports[-1]


def test_a_polygon_body_is_left_to_the_file():
    """The form says so instead of showing fields that cannot apply."""
    world = edit.new_world()
    spec = world.robots[0]
    shape = Polygon(((-0.2, -0.2), (0.2, -0.2), (0.0, 0.3)))
    polygon = dataclasses.replace(
        world, robots=(dataclasses.replace(spec, shape=shape),)
    )
    inspector, _, _, _ = build(Session(polygon))
    assert any("polygon" in t for t in texts(inspector.column))
    assert "body" not in inspector.fields


def test_the_estimator_and_the_pose_the_controller_sees_are_chosen():
    """Both are drop-downs; a bad edit would be reported."""
    inspector, session, _, _ = build()
    inspector.fields["estimator"].choose("dead_reckoning")
    assert session.world.robots[0].estimator == "dead_reckoning"
    inspector.fields["control_pose"].choose("truth")
    assert session.world.robots[0].control_pose == "truth"


def test_sensors_are_added_edited_and_removed_in_the_form():
    """Encoders and a lidar get their fields; each has a remove button."""
    inspector, session, reports, _ = build()
    inspector.fields["add_encoders"].click()
    assert "encoders" in session.world.robots[0].sensors
    assert "add_encoders" not in inspector.fields
    inspector.fields["encoders.noise"].set(0.02)
    assert session.world.robots[0].sensors["encoders"]["noise"] == 0.02
    inspector.fields["add_lidar"].click()
    inspector.fields["add_lidar"].click()
    assert {"lidar", "lidar_2"} <= set(session.world.robots[0].sensors)
    inspector.fields["lidar.beams"].set(90)
    inspector.fields["lidar.mount0"].set(0.1)
    lidar = session.world.robots[0].sensors["lidar"]
    assert lidar["beams"] == 90 and lidar["mount"][0] == 0.1
    inspector.fields["lidar.max_range"].set(0.0)
    assert reports
    inspector.fields["lidar_2.remove"].click()
    inspector.fields["encoders.remove"].click()
    assert set(session.world.robots[0].sensors) == {"lidar"}


def test_goals_are_typed_added_tuned_and_removed_in_the_form():
    """The first goal keeps its keys; waypoints get numbered ones."""
    inspector, session, _, _ = build()
    inspector.fields["gx"].set(8.0)
    assert edit.goals_of(session.world, "r1") == [0]
    inspector.fields["add_goal"].click()
    assert edit.goals_of(session.world, "r1") == [0, 1]
    assert {"g1x", "g1y", "g1th", "tol1d", "tol1a", "stop1", "wait1"} <= set(
        inspector.fields
    )
    inspector.fields["g1y"].set(5.0)
    inspector.fields["tol1d"].set(0.3)
    inspector.fields["tol1a"].set(10.0)
    inspector.fields["stop1"].choose("yes")
    inspector.fields["wait1"].set(2.0)
    task = session.world.tasks[1]
    assert float(task.goal.xyt[1]) == 5.0
    assert (task.stop, task.wait, task.goal_tolerance[0]) == (True, 2.0, 0.3)
    assert task.goal_tolerance[1] == pytest.approx(math.radians(10.0))
    inspector.fields["stop1"].choose("default")
    inspector.fields["remove0"].click()
    assert edit.goals_of(session.world, "r1") == [0]
    inspector.fields["remove0"].click()
    assert not session.world.tasks and "gx" in inspector.fields


def test_a_waypoint_is_put_beside_the_last_goal_inside_the_world():
    """Near the right wall the next goal goes to the left instead."""
    inspector, session, _, _ = build()
    session.set_goal("r1", (9.8, 3.0, 0.0))
    inspector.show("r1")
    inspector.fields["add_goal"].click()
    assert float(session.world.tasks[1].goal.xyt[0]) == pytest.approx(9.3)


def test_the_world_form_edits_the_corner_the_costmaps_and_the_default_sensors():
    """Switches, numbers, layers, a map file box and the sensors every robot gets."""
    inspector, session, reports, _ = build(name=None)
    session.select(None)
    inspector.show(None)
    f = inspector.fields
    f["ox0"].set(-1.0)
    assert session.world.origin == (-1.0, 0.0)
    f["ox0"].set(0.0)
    f["cm.resolution"].set(0.1)
    f["cm.inflation_radius"].set(0.4)
    f["cm.enabled"].on_change(False)
    assert session.world.costmap.resolution == 0.1 and not session.world.costmap.enabled
    f["cm.inflation"].checked = False
    f["cm.inflation"].on_change(False)
    assert session.world.costmap.plugins == ("static",)
    f["cm.unknown"].choose("blocked")
    assert session.world.costmap.unknown == "blocked"
    f["cm.source"].on_submit("generate")
    f["cm.source"].on_submit("/no/such/map.yaml")
    assert "no map file" in reports[-1]
    f["lc.window"].set(4.0)
    f["lc.obstacle"].checked = False
    f["lc.obstacle"].on_change(False)
    assert session.world.local_costmap.window == 4.0
    assert session.world.local_costmap.plugins == ("inflation",)
    f["add_default_encoders"].click()
    assert session.world.sensors == {"encoders": {}}
    assert "add_default_encoders" not in inspector.fields
    inspector.fields["encoders.noise"].set(0.05)
    assert session.world.sensors["encoders"]["noise"] == 0.05
    inspector.fields["encoders.remove"].click()
    assert session.world.sensors == {}
