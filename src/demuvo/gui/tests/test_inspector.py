"""Tests for the inspector: its sections, its edits and its lock."""

import math

import pytest

from demuvo import edit
from demuvo.gui.inputs import Combo, Spin
from demuvo.gui.inspector import HINT, NO_GOAL, Inspector
from demuvo.gui.session import Session
from demuvo.gui.widgets import Checkbox, Column, Label

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
    inspector = Inspector(
        column, session, reports.append, lambda: refreshed.append(1), footer
    )
    session.select(name)
    inspector.show(name)
    return inspector, session, reports, refreshed


def texts(column):
    """The text of every label in the column."""
    return [c.text for c in column.children if isinstance(c, Label)]


def test_with_nothing_selected_the_inspector_shows_a_hint_and_the_footer():
    """No fields; the footer widgets follow the hint."""
    inspector, session, _, _ = build(name=None)
    session.select(None)
    inspector.show(None)
    assert HINT in texts(inspector.column) and not inspector.fields
    assert inspector.column.children[-1] is inspector.footer[0]
    assert inspector.name is None


def test_a_robot_gets_pose_goal_choices_settings_and_a_control_period():
    """Every field of the form exists, with the robot's values."""
    inspector, session, _, _ = build()
    assert set(inspector.fields) == {
        "x", "y", "th", "gx", "gy", "gth",
        "model", "planner", "smoother", "controller", "period",
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
