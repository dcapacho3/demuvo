"""Tests for the sandbox program, driven through the shell as a user would."""

import time

import numpy as np
import pytest

from demuvo import edit
from demuvo.gui import look
from demuvo.gui.keys import CTRL, KEY
from demuvo.gui.sandbox import BADGES, DEFAULT_MESSAGE, GOAL_MESSAGE, Sandbox
from demuvo.gui.session import Session
from demuvo.gui.shell import Shell
from demuvo.gui.widgets import Event

from .fakes import EXAMPLES


def start(session=None, width=1280, height=780):
    """A shell over a sandbox program, rendered once."""
    program = Sandbox(session or Session())
    window = Shell(program, width, height)
    window.render()
    return window, program


@pytest.fixture(name="app")
def make_app():
    """A blank sandbox window."""
    return start()


def centre(widget):
    """The centre of a widget."""
    return widget.x + widget.width / 2, widget.y + widget.height / 2


def click(window, widget):
    """Move to a widget's centre and click it."""
    x, y = centre(widget)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))


def at(program, x, y):
    """The pixel of a world point."""
    sx, sy = program.session.scene.view.to_screen(x, y)
    return float(sx), float(sy)


def press_key(window, name, ctrl=False):
    """A key press of a named key."""
    window.key(KEY[name], 1, CTRL if ctrl else 0)


def test_the_window_opens_with_one_robot_a_hint_and_an_editing_badge(app):
    """The scene list, the count, the badge and the inspector hint at the start."""
    window, program = app
    chrome = window.layout.chrome
    assert chrome.status.count == "1 robot" and chrome.status.message == DEFAULT_MESSAGE
    assert (program.badge.text, program.badge.kind) == BADGES["edit"]
    assert [row.text for row in program.tree.rows][:2] == ["Robots (1)", "r1"]
    assert set(program.inspector.fields) == {"width", "height", "seed", "dt"}
    assert window.render().shape == (780, 1280, 3)


def test_the_tool_bar_buttons_are_laid_out_in_a_row_without_overlap(app):
    """Left panel, tools, then the right panel at the far end."""
    _, program = app
    keys = ("left", "select", "goal", "add", "delete")
    xs = [program.buttons[k].x for k in keys]
    assert xs == sorted(xs) and all(b - a >= 32 for a, b in zip(xs, xs[1:]))
    assert program.buttons["right"].x == 1280 - 40
    assert program.badge.x + program.badge.width < program.buttons["right"].x


def test_a_click_on_a_robot_selects_it_and_shows_its_form(app):
    """The tree highlights it, the inspector fills in, the scene marks it."""
    window, program = app
    x, y = at(program, 5.0, 3.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))
    assert program.session.scene.selected == "r1" and program.tree.selected == "r1"
    assert "x" in program.inspector.fields and window.render() is not None


def test_a_click_on_the_scene_list_selects_a_robot(app):
    """The list is another way to select."""
    window, program = app
    rows = program.tree
    window.mouse_move(rows.x + 40, rows.y + 32 + 16)
    window.mouse_press(Event(rows.x + 40, rows.y + 32 + 16))
    window.mouse_release(Event(rows.x + 40, rows.y + 32 + 16))
    assert program.session.scene.selected == "r1"


def test_the_add_button_adds_a_robot_and_selects_it_and_the_count_follows(app):
    """Two robots, the second selected, the list and the status bar updated."""
    window, program = app
    click(window, program.buttons["add"])
    assert [r.name for r in program.session.world.robots] == ["r1", "r2"]
    assert program.session.scene.selected == "r2" and program.tree.selected == "r2"
    assert window.layout.chrome.status.count == "2 robots"
    assert [row.text for row in program.tree.rows][:3] == ["Robots (2)", "r1", "r2"]


def test_delete_removes_the_selected_robot_and_the_last_one_is_kept_with_a_message(app):
    """Delete key on the only robot says why not; with two, one goes."""
    window, program = app
    program.select("r1")
    press_key(window, "delete")
    assert window.layout.chrome.status.message.startswith("That change cannot be made")
    click(window, program.buttons["add"])
    program.select("r1")
    click(window, program.buttons["delete"])
    assert [r.name for r in program.session.world.robots] == ["r2"]


def test_the_goal_tool_is_picked_by_key_and_a_floor_click_sets_the_goal(
    app,
):
    """G selects the tool and says what it does; a floor click gives r1 a goal there."""
    window, program = app
    program.select("r1")
    press_key(window, "g")
    assert program.session.tool == "goal" and program.buttons["goal"].checked
    assert not program.buttons["select"].checked
    assert window.layout.chrome.status.message == GOAL_MESSAGE
    x, y = at(program, 8.0, 1.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))
    assert edit.goal_of(program.session.world, "r1")[:2] == pytest.approx(
        (8.0, 1.0), abs=0.02
    )
    press_key(window, "v")
    assert program.session.tool == "select" and program.buttons["select"].checked


def test_a_goal_click_beyond_the_world_is_refused_in_the_status_bar(app):
    """Inside the pane but past the border."""
    window, program = app
    program.select("r1")
    program.set_tool("goal")
    x, y = at(program, 10.2, 3.0)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))
    assert "must lie inside the world" in window.layout.chrome.status.message
    assert edit.goal_of(program.session.world, "r1") is None


def test_dragging_a_robot_moves_it_and_a_drop_onto_another_is_refused(app):
    """The first drag lands; the second is refused and the robot stays."""
    window, program = app
    click(window, program.buttons["add"])
    x, y = at(program, 5.0, 3.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    nx, ny = at(program, 2.0, 2.0)
    window.mouse_move(nx, ny)
    assert program.inspector.fields["x"].value == pytest.approx(2.0, abs=0.05)
    window.mouse_release(Event(nx, ny))
    assert program.session.run.poses["r1"][:2] == pytest.approx([2.0, 2.0], abs=0.05)
    r2 = program.session.run.poses["r2"]
    sx, sy = at(program, 2.0, 2.0)
    window.mouse_move(sx, sy)
    window.mouse_press(Event(sx, sy))
    ox, oy = at(program, r2[0], r2[1])
    window.mouse_move(ox, oy)
    window.mouse_release(Event(ox, oy))
    assert "would overlap" in window.layout.chrome.status.message
    assert program.session.run.poses["r1"][:2] == pytest.approx([2.0, 2.0], abs=0.05)


def test_play_runs_the_world_locks_editing_and_changes_the_badge_and_the_button(app):
    """Space starts it; time moves on tick; the add button goes off."""
    window, program = app
    program.set_tool("goal")
    press_key(window, "space")
    assert program.session.playing and program.pane.pill.play.icon == "pause"
    assert (program.badge.text, program.badge.kind) == BADGES["run"]
    assert not program.buttons["add"].enabled and program.session.tool == "select"
    assert window.busy()
    window.tick(time.perf_counter() + 0.5)
    assert program.session.run.time > 0 and program.pane.pill.clock != "0:00.0"
    window.render()
    press_key(window, "space")
    assert not program.session.playing and program.pane.pill.play.icon == "play"


def test_step_advances_a_control_period_and_reset_goes_back_and_unlocks(app):
    """S takes a step and locks; R resets and edits are allowed again."""
    window, program = app
    press_key(window, "s")
    assert (
        program.session.run.time == pytest.approx(0.05) and not program.session.editable
    )
    assert program.badge.kind == "run" and not program.buttons["add"].enabled
    press_key(window, "r")
    assert program.session.run.time == 0.0 and program.session.editable
    assert program.badge.kind == "edit" and program.pane.pill.clock == "0:00.0"
    assert program.buttons["add"].enabled


def test_a_second_step_does_not_lock_again_and_only_redraws_the_changed_boxes(app):
    """After the first step the badge is run; later steps use the dirty rectangle."""
    window, program = app
    press_key(window, "s")
    window.render()
    press_key(window, "s")
    assert window.dirty and window.dirty_rect is not None
    assert program.session.run.time == pytest.approx(0.10)


def test_a_layer_fades_out_through_an_animation_and_its_box_and_menu_follow(app):
    """Unchecking the floor box animates its fade to zero."""
    window, program = app
    box = program.layer_boxes["floor"]
    click(window, box)
    assert program.layer_on["floor"] is False and not box.checked
    window.animator.tick(time.perf_counter() + 1.0)
    assert program.session.scene.layers.shown["floor"] == 0.0
    program.set_layer("floor", True)
    window.animator.tick(time.perf_counter() + 1.0)
    assert program.session.scene.layers.shown["floor"] == 1.0
    assert window.render().shape == (780, 1280, 3)


def test_the_opacity_slider_and_the_palette_choice_change_the_costmap_look(app):
    """The slider sets the costmap layers' opacity; the combo sets the palette."""
    _, program = app
    program.alpha.on_press(
        Event(program.alpha.x + program.alpha.width / 2, program.alpha.y + 14)
    )
    assert 0.3 < program.session.scene.layers.opacity["costmap"] < 0.7
    assert (
        program.session.scene.layers.opacity["local"]
        == program.session.scene.layers.opacity["costmap"]
    )
    program.palette.choose(look.PALETTES[1])
    assert program.session.scene.palette == look.PALETTES[1]


def test_the_speed_choice_sets_the_playback_speed(app):
    """The pill's speed combo reaches the session."""
    _, program = app
    program.pane.pill.speed.choose("4x")
    assert program.session.speed == 4.0


def test_the_menus_list_the_commands_and_run_them(app):
    """File has Quit; Edit has the tools; View has the panels and layers."""
    window, program = app
    assert [i.text for i in program.menu_items("File")] == ["Quit"]
    edit_items = program.menu_items("Edit")
    assert [i.text for i in edit_items if not i.separator] == [
        "Add robot", "Delete robot", "Set goal", "Select", "Reset run",
    ]  # fmt: skip
    edit_items[0].action()
    assert len(program.session.world.robots) == 2
    view = program.menu_items("View")
    assert [i.text for i in view[:2]] == ["Scene panel", "Inspector"]
    assert "lidar rays" in [i.text for i in view]
    view[0].action()
    assert window.layout.docks["left"].open is False
    view[[i.text for i in view].index("trails")].action()
    assert program.layer_on["trails"] is False
    view[-1].action()
    program.menu_items("File")[0].action()
    assert window.quit_requested


def test_the_menu_state_follows_the_mode_and_the_tool(app):
    """A locked world greys the edit lines; the tool is ticked."""
    _, program = app
    program.set_tool("goal")
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert lines["Set goal"].checked and not lines["Select"].checked
    program.step_once()
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert not lines["Add robot"].enabled and not lines["Delete robot"].enabled


def test_the_shortcuts_do_their_work_and_an_unknown_key_is_not_used(app):
    """Ctrl+N adds, Ctrl+0 fits, and a letter with no meaning goes unused."""
    window, program = app
    press_key(window, "n", ctrl=True)
    assert len(program.session.world.robots) == 2
    press_key(window, "0", ctrl=True)
    assert program.command(KEY["x"], False) is False


def test_the_panels_slide_by_button_and_by_key(app):
    """The buttons follow the docks."""
    window, program = app
    click(window, program.buttons["left"])
    assert (
        window.layout.docks["left"].open is False
        and not program.buttons["left"].checked
    )
    press_key(window, "2", ctrl=True)
    assert (
        window.layout.docks["right"].open is False
        and not program.buttons["right"].checked
    )
    window.animator.tick(time.perf_counter() + 1.0)
    assert window.layout.pane.x == 0


def test_the_status_bar_shows_the_world_place_under_the_pointer(app):
    """Over the scene the place is a metre value; over a panel there is none."""
    window, program = app
    x, y = at(program, 3.0, 2.0)
    window.mouse_move(x, y)
    assert window.layout.chrome.status.coords == pytest.approx((3.0, 2.0))
    window.mouse_move(20, 400)
    assert window.layout.chrome.status.coords is None


def test_a_benchmark_is_fixed_and_says_so_and_reset_still_works():
    """No edits, a bench badge, a note in the inspector, and Reset repeats the run."""
    window, program = start(Session(EXAMPLES / "demo.toml", "benchmark"))
    assert (program.badge.text, program.badge.kind) == BADGES["bench"]
    assert not program.buttons["add"].enabled
    program.select("r1")
    assert "benchmark world" in program.inspector.note.label.text
    press_key(window, "n", ctrl=True)
    assert window.layout.chrome.status.message.startswith("A benchmark world is fixed")
    press_key(window, "s")
    press_key(window, "r")
    assert program.session.run.time == 0.0


def test_a_selected_robots_form_applies_an_edit_end_to_end(app):
    """Change the planner in the inspector and the world follows."""
    window, program = app
    program.select("r1")
    program.inspector.fields["planner"].choose("Straight line")
    assert program.session.world.robots[0].planner == "demuvo.planners:StraightLine"
    assert window.render().shape == (780, 1280, 3)


def test_a_refused_edit_from_the_inspector_shows_its_reason(app):
    """A control period below the step is explained in the status bar."""
    window, program = app
    program.select("r1")
    program.inspector.fields["period"].set(0.002)
    assert "below dt" in window.layout.chrome.status.message


def test_the_window_draws_a_frame_that_differs_after_an_edit(app):
    """The frame changes when a robot is added."""
    window, program = app
    before = window.render().copy()
    click(window, program.buttons["add"])
    assert not np.array_equal(window.render(), before)
