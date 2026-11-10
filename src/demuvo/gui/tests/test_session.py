"""Tests for a session: modes, locks, edits and the run, with no widgets."""

import dataclasses
from pathlib import Path

import pytest

from demuvo import edit
from demuvo.gui import look
from demuvo.gui import session as module
from demuvo.gui.session import Session

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"


@pytest.fixture(name="sandbox")
def blank():
    """A blank sandbox: one robot, r1, in the middle."""
    return Session()


def test_the_sandbox_starts_blank_and_a_benchmark_needs_a_world():
    """The modes, and what each needs."""
    fresh = Session()
    assert fresh.mode == "sandbox" and [r.name for r in fresh.world.robots] == ["r1"]
    assert Session(DEMO, "benchmark").mode == "benchmark"
    with pytest.raises(ValueError, match="needs a world file"):
        Session(None, "benchmark")
    with pytest.raises(ValueError, match="mode must be"):
        Session(DEMO, "game")


def test_a_session_can_open_a_world_in_memory():
    """An edited world is as good as a file."""
    world = edit.set_goal(edit.new_world(), "r1", (8.0, 1.0, 0.0))
    assert edit.goal_of(Session(world).world, "r1") == pytest.approx((8.0, 1.0, 0.0))


def test_the_state_says_edit_run_or_bench_and_editing_ends_with_the_first_step(sandbox):
    """Editable at time zero; after a step or while playing, locked."""
    assert sandbox.state == "edit" and sandbox.editable
    sandbox.step_once()
    assert sandbox.state == "run" and not sandbox.editable
    sandbox.reset()
    assert sandbox.editable
    sandbox.toggle_play()
    assert not sandbox.editable and sandbox.state == "run"
    assert Session(DEMO, "benchmark").state == "bench"
    assert not Session(DEMO, "benchmark").editable


def test_a_selection_reports_whether_it_changed(sandbox):
    """The same name again is no change; none clears it."""
    assert sandbox.select("r1") is True and sandbox.scene.selected == "r1"
    assert sandbox.select("r1") is False
    assert sandbox.select(None) is True


def test_the_tool_is_chosen(sandbox):
    """Select or goal."""
    sandbox.set_tool("goal")
    assert sandbox.tool == "goal"


def test_a_robot_is_added_at_a_free_place_with_the_next_free_name_and_selected(sandbox):
    """r2, then r3; the new one is selected; the scene shows it."""
    assert sandbox.add_robot() is None
    assert sandbox.add_robot() is None
    assert [r.name for r in sandbox.world.robots] == ["r1", "r2", "r3"]
    assert sandbox.scene.selected == "r3" and "r3" in sandbox.scene.colors


def test_a_gap_in_the_names_is_filled(sandbox):
    """With r2 deleted the next robot is r2 again."""
    sandbox.add_robot()
    sandbox.select("r2")
    assert sandbox.delete_robot() is None
    sandbox.add_robot()
    assert [r.name for r in sandbox.world.robots] == ["r1", "r2"]


def test_a_full_world_refuses_another_robot():
    """No free place gives the reason."""
    tiny = Session(edit.new_world((1.0, 1.0)))
    assert "no free place" in tiny.add_robot()


def test_nothing_can_be_edited_once_a_run_has_begun_or_in_a_benchmark(sandbox):
    """Every edit says why not, and changes nothing."""
    sandbox.step_once()
    for call in (
        sandbox.add_robot,
        lambda: sandbox.set_goal("r1", (8, 1, 0)),
        lambda: sandbox.choose("r1", "planner", "A*"),
        lambda: sandbox.set_settings("r1", "controller", {"lookahead": 0.9}),
        lambda: sandbox.set_period("r1", 0.1),
    ):
        assert call() == "Press Reset (R) to edit again."
    bench = Session(DEMO, "benchmark")
    assert bench.add_robot() == "A benchmark world is fixed."
    assert [r.name for r in bench.world.robots] == ["r1"]


def test_a_selected_robot_is_deleted_unless_it_is_the_last(sandbox):
    """Nothing selected is nothing to do; the last robot stays; one of two can go."""
    assert sandbox.delete_robot() is None
    sandbox.select("r1")
    assert "only robot" in sandbox.delete_robot()
    sandbox.add_robot()
    sandbox.select("r1")
    assert sandbox.delete_robot() is None
    assert [r.name for r in sandbox.world.robots] == [
        "r2"
    ] and sandbox.scene.selected is None
    sandbox.step_once()
    sandbox.select("r2")
    assert sandbox.delete_robot() == "Press Reset (R) to edit again."


def test_a_robot_is_moved_inside_the_edge_and_refused_onto_another(sandbox):
    """A drop beyond the wall is held a quarter metre in; onto a robot is refused."""
    assert sandbox.move_robot("r1", (50.0, -3.0, 0.5)) is None
    x, y, angle = sandbox.run.poses["r1"]
    assert (x, y) == pytest.approx((10.0 - 0.25, 0.25)) and angle == pytest.approx(0.5)
    sandbox.add_robot()
    other = sandbox.run.poses["r2"]
    error = sandbox.move_robot("r1", tuple(other))
    assert error.startswith("That change cannot be made: robot 'r1' would overlap")


def test_a_move_can_happen_while_a_run_is_paused_at_a_new_world(sandbox):
    """Moves go through `apply`, which keeps the scene and its selection."""
    sandbox.select("r1")
    scene = sandbox.scene
    sandbox.move_robot("r1", (2.0, 2.0, 0.0))
    assert sandbox.scene is scene and scene.selected == "r1"


def test_a_goal_is_set_and_selects_the_robot(sandbox):
    """A goal inside the world; one outside is refused in plain words."""
    assert sandbox.set_goal("r1", (8.0, 1.0, 0.5)) is None
    assert edit.goal_of(sandbox.world, "r1") == pytest.approx((8.0, 1.0, 0.5))
    assert sandbox.scene.selected == "r1"
    assert "must lie inside the world" in sandbox.set_goal("r1", (80.0, 1.0, 0.0))


def test_the_model_and_the_parts_are_chosen_by_friendly_name(sandbox):
    """A* for the planner, no smoother, a controller driven from outside."""
    assert sandbox.choose("r1", "planner", "Straight line") is None
    assert sandbox.world.robots[0].planner == "demuvo.planners:StraightLine"
    assert sandbox.choose("r1", "smoother", "Line of sight") is None
    assert sandbox.choose("r1", "controller", "none (driven from outside)") is None
    assert sandbox.world.robots[0].controller is None
    assert sandbox.choose("r1", "model", "diff_drive") is None
    assert "unknown model" in sandbox.choose("r1", "model", "car")


def test_settings_and_the_control_period_are_set_or_refused_in_plain_words(sandbox):
    """A good value goes in; a bad one comes back as a sentence."""
    assert sandbox.set_settings("r1", "controller", {"lookahead": 0.9}) is None
    assert edit.settings_of_robot(sandbox.world, "r1", "controller") == {
        "lookahead": 0.9
    }
    assert "no setting 'speeed'" in sandbox.set_settings(
        "r1", "controller", {"speeed": 1}
    )
    assert sandbox.set_period("r1", 0.1) is None
    assert sandbox.world.robots[0].control_period == 0.1
    assert "positive" in sandbox.set_period("r1", 0.0)
    assert "below dt" in sandbox.set_period("r1", 0.001)
    assert sandbox.world.robots[0].control_period == 0.1


def test_a_world_that_cannot_run_is_refused_and_nothing_changes(sandbox):
    """Two robots on the same spot, built by hand."""
    world = edit.add_robot(sandbox.world, "r2", (1.0, 1.0, 0.0))
    stacked = dataclasses.replace(world.robots[1], pose=world.robots[0].pose)
    bad = dataclasses.replace(world, robots=(world.robots[0], stacked))
    before = sandbox.world
    error = sandbox.apply(bad)
    assert error.startswith("That change cannot be made:") and "starts inside" in error
    assert sandbox.world is before


def test_a_rebuilt_scene_keeps_the_layers_and_the_palette(sandbox):
    """Adding a robot makes a new scene that looks the same."""
    sandbox.scene.layers.shown["rays"] = 1.0
    sandbox.scene.palette = look.PALETTES[1]
    old = sandbox.scene
    sandbox.add_robot()
    assert sandbox.scene is not old
    assert (
        sandbox.scene.layers is old.layers and sandbox.scene.palette == look.PALETTES[1]
    )


def test_playing_advances_the_run_at_the_speed_and_a_slow_frame_is_capped(sandbox):
    """A 0.5 s frame counts as 0.1 s; at 2x it covers 0.2 s; paused, nothing moves."""
    sandbox.advance(0.5)
    assert sandbox.run.time == 0.0
    sandbox.toggle_play()
    sandbox.speed = 2.0
    sandbox.advance(0.5)
    assert sandbox.run.time == pytest.approx(module.MAX_FRAME * 2.0, abs=0.02)
    assert len(sandbox.scene.trails["r1"]) > 1
    sandbox.toggle_play()
    assert not sandbox.playing


def test_a_step_is_about_a_control_period_and_reset_goes_back_to_zero(sandbox):
    """5 steps of 0.01 s; reset empties the trail and unlocks editing."""
    sandbox.step_once()
    assert sandbox.run.time == pytest.approx(0.05)
    sandbox.toggle_play()
    sandbox.reset()
    assert sandbox.run.time == 0.0 and not sandbox.playing and sandbox.editable
    assert len(sandbox.scene.trails["r1"]) == 1


def test_the_heading_to_a_point_is_the_angle_from_the_robot(sandbox):
    """r1 is in the middle at (5, 3); a point straight above is 90 degrees."""
    assert sandbox.heading_to("r1", 5.0, 5.0) == pytest.approx(1.5707963)
    assert sandbox.heading_to("r1", 9.0, 3.0) == pytest.approx(0.0)
