"""Tests for a session: modes, locks, edits and the run, with no widgets."""

import dataclasses
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from demuvo import edit
from demuvo.gui import look, remote
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


def test_a_fast_frame_rate_does_not_make_the_run_faster_than_real_time(sandbox):
    """Twenty frames of 4 ms are 0.08 s of time, not one step a frame."""
    sandbox.toggle_play()
    for _ in range(20):
        sandbox.advance(0.004)
    assert sandbox.run.time == pytest.approx(0.08, abs=0.011)


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
    assert sandbox.scene.heading_to("r1", 5.0, 5.0) == pytest.approx(1.5707963)
    assert sandbox.scene.heading_to("r1", 9.0, 3.0) == pytest.approx(0.0)


@pytest.fixture(name="remote_session")
def thread_served(monkeypatch):
    """A session whose simulation is a worker served from a thread."""
    ends = []

    def make(world):
        parent, child = multiprocessing.Pipe()
        threading.Thread(target=remote.serve, args=(child,), daemon=True).start()
        ends.append(parent)
        return remote.RemoteRun(world, connection=parent)

    monkeypatch.setattr(module, "RemoteRun", make)
    session = Session(remote=True)
    yield session
    session.close()


def test_a_remote_session_edits_and_resets_like_a_local_one(remote_session):
    """Adding a robot and a goal go through the worker; reset returns to zero."""
    assert remote_session.add_robot() is None
    assert [r.name for r in remote_session.world.robots] == ["r1", "r2"]
    assert set(remote_session.run.poses) == {"r1", "r2"}
    assert remote_session.set_goal("r1", (8.0, 1.0, 0.0)) is None
    remote_session.step_once()
    assert (
        remote_session.run.time == pytest.approx(0.05) and not remote_session.editable
    )
    remote_session.reset()
    assert remote_session.run.time == 0.0 and remote_session.editable


def test_playing_a_remote_session_takes_news_and_records_each_time_once(
    remote_session,
):
    """Advance asks the worker; later calls bring the poses and one trail point."""
    remote_session.toggle_play()
    end = time.perf_counter() + 10
    while remote_session.run.time == 0.0:
        remote_session.advance(0.05)
        assert time.perf_counter() < end
        time.sleep(0.002)
    points = len(remote_session.scene.trails["r1"])
    remote_session.toggle_play()
    remote_session.advance(0.05)
    remote_session.advance(0.05)
    assert len(remote_session.scene.trails["r1"]) in (points, points + 1)


def test_an_obstacle_is_added_selected_moved_resized_and_deleted(sandbox):
    """Each obstacle edit changes the world and keeps the obstacle selected."""
    assert sandbox.edits.add_obstacle("box", (2.0, 2.0, 0.0), (1.0, 0.5)) is None
    assert sandbox.scene.selected_obstacle == 0 and sandbox.scene.selected is None
    assert sandbox.edits.move_obstacle(0, (3.0, 2.0, 0.0)) is None
    assert sandbox.edits.resize_obstacle(0, (2.0, 1.0)) is None
    assert sandbox.world.obstacles[0].shape.size == (2.0, 1.0)
    assert sandbox.scene.selected_obstacle == 0
    assert sandbox.edits.delete_obstacle() is None
    assert not sandbox.world.obstacles and sandbox.scene.selected_obstacle is None


def test_a_bad_obstacle_edit_says_why_and_changes_nothing(sandbox):
    """An obstacle on the robot is refused in plain words."""
    message = sandbox.edits.add_obstacle("box", (5.0, 3.0, 0.0), (1.0, 1.0))
    assert message.startswith("That change cannot be made:") and "overlap" in message
    assert not sandbox.world.obstacles
    assert sandbox.edits.delete_obstacle() is None


def test_obstacles_are_locked_while_a_run_is_in_progress(sandbox):
    """Edits wait for Reset, as for robots."""
    sandbox.step_once()
    assert sandbox.edits.add_obstacle("circle", (2.0, 2.0, 0.0), (0.3,)) == (
        "Press Reset (R) to edit again."
    )
    assert sandbox.edits.set_size((12.0, 7.0)) == "Press Reset (R) to edit again."


def test_selecting_a_robot_or_an_obstacle_drops_the_other(sandbox):
    """Only one thing is selected at a time."""
    sandbox.edits.add_obstacle("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    assert sandbox.select("r1") and sandbox.scene.selected_obstacle is None
    assert not sandbox.select("r1")
    assert sandbox.select_obstacle(0) and sandbox.scene.selected is None
    assert not sandbox.select_obstacle(0)
    assert sandbox.select(None) and sandbox.scene.selected_obstacle is None


def test_the_world_size_changes_and_the_scene_refits(sandbox):
    """A larger world keeps its robot; a smaller one that cuts it off is refused."""
    assert sandbox.edits.set_size((20.0, 8.0)) is None
    assert sandbox.world.size == (20.0, 8.0)
    assert "robot 'r1'" in sandbox.edits.set_size((1.0, 1.0))


def test_a_world_is_saved_and_opened_again(sandbox, tmp_path):
    """Save writes a file that Open reads back, and both remember the path."""
    sandbox.edits.add_obstacle("circle", (2.0, 2.0, 0.0), (0.3,))
    path = str(tmp_path / "mine.toml")
    assert sandbox.edits.save(path) is None and sandbox.edits.path == path
    other = Session()
    assert other.edits.open(path) is None and other.edits.path == path
    assert other.world.obstacles[0].shape.radius == 0.3
    assert other.scene.selected is None


def test_saving_and_opening_report_a_problem_in_plain_words(sandbox, tmp_path):
    """A folder that is not there, or a file that is wrong, is a message."""
    assert sandbox.edits.save(str(tmp_path / "no" / "w.toml")).startswith(
        "Could not save"
    )
    assert sandbox.edits.open(str(tmp_path / "missing.toml")).startswith(
        "Could not open"
    )
    bad = tmp_path / "bad.toml"
    bad.write_text("version = 2\n")
    assert "version" in sandbox.edits.open(str(bad))
    assert (
        Session(DEMO, "benchmark").edits.open(str(bad)) == "A benchmark world is fixed."
    )


def test_opening_a_world_ends_a_run_in_progress(sandbox):
    """Open stops play and starts the new world at time zero."""
    sandbox.toggle_play()
    sandbox.advance(0.1)
    assert sandbox.edits.open(str(DEMO)) is None
    assert not sandbox.playing and sandbox.run.time == 0
