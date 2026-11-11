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
from demuvo.world import save_world

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
    assert {"width", "height", "seed", "dt"} <= set(program.inspector.fields)
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


def test_a_cached_layer_switches_at_once_and_a_live_one_fades(app):
    """The costmap goes with the click; the trails fade through an animation."""
    window, program = app
    shown = program.session.scene.layers.shown
    box = program.layer_boxes["costmap"]
    click(window, box)
    assert program.layer_on["costmap"] is False and not box.checked
    assert shown["costmap"] == 0.0
    program.set_layer("costmap", True)
    assert shown["costmap"] == 1.0
    click(window, program.layer_boxes["trails"])
    assert shown["trails"] == 1.0 and window.animator.busy()
    window.animator.tick(time.perf_counter() + 1.0)
    assert shown["trails"] == 0.0
    program.set_layer("trails", True)
    window.animator.tick(time.perf_counter() + 1.0)
    assert shown["trails"] == 1.0
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
    assert [i.text for i in program.menu_items("File") if not i.separator] == [
        "New world", "Open…", "Save", "Save as…", "Quit",
    ]  # fmt: skip
    edit_items = program.menu_items("Edit")
    assert [i.text for i in edit_items if not i.separator] == [
        "Undo", "Redo", "Add robot", "Delete selected", "Set goal", "Draw box",
        "Draw circle", "Draw polygon", "Draw line", "Select", "Reset run",
    ]  # fmt: skip
    next(i for i in edit_items if i.text == "Add robot").action()
    assert len(program.session.world.robots) == 2
    view = program.menu_items("View")
    assert [i.text for i in view[:2]] == ["Scene panel", "Inspector"]
    assert "lidar rays" in [i.text for i in view]
    view[0].action()
    assert window.layout.docks["left"].open is False
    view[[i.text for i in view].index("trails")].action()
    assert program.layer_on["trails"] is False
    view[-1].action()
    program.menu_items("File")[-1].action()
    assert not window.quit_requested
    assert "Unsaved changes" in window.layout.chrome.status.message
    program.menu_items("File")[-1].action()
    assert window.quit_requested


def test_the_menu_state_follows_the_mode_and_the_tool(app):
    """A locked world greys the edit lines; the tool is ticked."""
    _, program = app
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert not lines["Set goal"].enabled
    program.select("r1")
    program.set_tool("goal")
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert (
        lines["Set goal"].enabled
        and lines["Set goal"].checked
        and not lines["Select"].checked
    )
    program.step_once()
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert not lines["Add robot"].enabled and not lines["Delete selected"].enabled


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


def test_b_and_c_pick_the_drawing_tools_and_say_what_they_do(app):
    """The buttons follow the keys; the status bar explains each tool."""
    window, program = app
    status = window.layout.chrome.status
    press_key(window, "b")
    assert program.session.tool == "box" and program.buttons["box"].checked
    assert "corner" in status.message
    press_key(window, "c")
    assert program.session.tool == "circle" and not program.buttons["box"].checked
    assert "centre" in status.message


def test_a_drawn_obstacle_is_listed_selected_and_shown_in_the_inspector(app):
    """Draw a box: the list has a row, it is selected and its form is up."""
    window, program = app
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    rows = [row.text for row in program.tree.rows]
    assert "box 1" in rows and program.tree.selected == "obstacle:0"
    assert set(program.inspector.fields) == {"ox", "oy", "oth", "size0", "size1"}
    program.pane.actions.obstacle_drawn("box", (5.0, 3.0, 0.0), (1.0, 1.0))
    assert "overlap" in window.layout.chrome.status.message
    assert len(program.session.world.obstacles) == 1


def test_clicking_an_obstacle_row_selects_that_obstacle(app):
    """The scene list and the pane select the same way."""
    _, program = app
    program.pane.actions.obstacle_drawn("circle", (2.0, 2.0, 0.0), (0.4,))
    program.select("r1")
    assert program.session.scene.selected_obstacle is None
    program.select("obstacle:0")
    assert program.session.scene.selected_obstacle == 0
    program.pane.actions.select_obstacle(0)
    assert program.tree.selected == "obstacle:0"


def test_dragging_an_obstacle_shows_its_place_and_a_bad_place_is_refused(app):
    """The pose fields follow the drag; into the robot it stays where it was."""
    window, program = app
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 1.0))
    program.pane.actions.obstacle_moved(0, 3.0, 2.0, force=True)
    assert program.inspector.fields["ox"].value == pytest.approx(3.0)
    program.pane.actions.obstacle_moved(0, 5.0, 3.0, force=True)
    assert "overlap" in window.layout.chrome.status.message
    assert program.session.world.obstacles[0].pose.xyt[0] == pytest.approx(3.0)


def test_delete_removes_the_selected_obstacle_before_any_robot(app):
    """With an obstacle selected the key removes it and the robot stays."""
    window, program = app
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 1.0))
    press_key(window, "delete")
    assert not program.session.world.obstacles
    assert [r.name for r in program.session.world.robots] == ["r1"]


def test_the_drawing_tools_go_off_when_a_run_starts(app):
    """A run locks editing, so the tool goes back to select."""
    _, program = app
    program.set_tool("box")
    program.step_once()
    assert program.session.tool == "select"
    assert not program.buttons["box"].enabled


def test_a_click_on_a_robot_in_the_scene_selects_it_through_the_pane(app):
    """The pane's click reaches the sandbox, which shows the robot's form."""
    window, program = app
    x, y = at(program, 5.0, 3.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))
    assert program.session.scene.selected == "r1"
    assert "planner" in program.inspector.fields


def test_a_click_on_an_obstacle_in_the_scene_selects_it_and_shows_its_form(app):
    """The pane's click selects the obstacle and the inspector follows."""
    window, program = app
    program.session.edits.add_obstacle("box", (2.0, 2.0, 0.0), (1.0, 1.0))
    program.session.select(None)
    x, y = at(program, 2.0, 2.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))
    assert program.session.scene.selected_obstacle == 0
    assert "ox" in program.inspector.fields


def test_the_lidar_rays_start_on_for_a_few_robots_and_off_for_a_swarm():
    """Ten robots or fewer show their rays; more would cover the floor."""
    _, few = start(Session(EXAMPLES / "lidar.toml"))
    assert few.layer_on["rays"] and few.session.scene.layers.shown["rays"] == 1.0
    many = edit.new_world()
    for k in range(2, 13):
        many = edit.add_robot(many, f"r{k}", edit.free_pose(many))
    _, crowd = start(Session(many))
    assert not crowd.layer_on["rays"]


def type_text(window, text):
    """Type a line into whatever has the focus."""
    for char in text:
        window.char(char)


def test_save_as_asks_for_a_path_then_writes_the_world_and_later_saves_go_there(
    app, tmp_path
):
    """Ctrl+S with no file opens the prompt; Enter writes; the next Ctrl+S reuses it."""
    window, program = app
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    press_key(window, "s", ctrl=True)
    assert window.input.popup is not None
    path = tmp_path / "mine.toml"
    type_text(window, str(path))
    press_key(window, "enter")
    assert window.input.popup is None and path.exists()
    assert f"Saved {path}" in window.layout.chrome.status.message
    program.pane.actions.obstacle_drawn("circle", (7.0, 4.0, 0.0), (0.3,))
    press_key(window, "s", ctrl=True)
    assert window.input.popup is None
    assert "circle" in path.read_text()


def test_open_asks_for_a_path_and_shows_the_world_in_the_file(app, tmp_path):
    """Ctrl+O, a path, Enter: the world is replaced and the lists follow."""
    window, program = app
    path = tmp_path / "other.toml"
    other = edit.add_obstacle(edit.new_world(), "box", (2.0, 2.0, 0.0), (1.0, 1.0))
    save_world(other, path)
    press_key(window, "o", ctrl=True)
    assert window.input.popup is not None
    type_text(window, str(path))
    press_key(window, "enter")
    assert "Opened" in window.layout.chrome.status.message
    assert len(program.session.world.obstacles) == 1
    assert "box 1" in [row.text for row in program.tree.rows]


def test_a_missing_file_keeps_the_dialog_open_with_a_note_and_changes_nothing(
    app, tmp_path
):
    """The dialog says there is no such file and stays, so the name can be fixed."""
    window, program = app
    press_key(window, "o", ctrl=True)
    type_text(window, str(tmp_path / "missing.toml"))
    press_key(window, "enter")
    dialog = window.input.popup
    assert dialog is not None and dialog.note == "There is no such file."
    assert not program.session.world.obstacles
    press_key(window, "escape")
    assert window.input.popup is None


def test_a_file_that_will_not_load_is_reported_in_the_status_bar(app, tmp_path):
    """The dialog takes the file; the world says what is wrong with it."""
    window, _ = app
    bad = tmp_path / "bad.toml"
    bad.write_text("version = 2\n")
    press_key(window, "o", ctrl=True)
    type_text(window, str(bad))
    press_key(window, "enter")
    assert "Could not open" in window.layout.chrome.status.message


def test_escape_closes_the_prompt_and_new_world_starts_blank(app):
    """Esc gives up; New world drops the obstacles."""
    window, program = app
    press_key(window, "o", ctrl=True)
    press_key(window, "escape")
    assert window.input.popup is None
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    program.menu_items("File")[0].action()
    assert program.session.world.obstacles
    assert "Choose New world again" in window.layout.chrome.status.message
    program.menu_items("File")[0].action()
    assert not program.session.world.obstacles


def test_a_benchmark_world_cannot_be_replaced_but_can_be_saved(tmp_path):
    """Open and New are greyed out; Save as still writes the fixed world."""
    window, program = start(Session(EXAMPLES / "demo.toml", "benchmark"))
    items = {i.text: i for i in program.menu_items("File") if not i.separator}
    assert not items["Open…"].enabled and not items["New world"].enabled
    program.command(KEY["o"], True)
    assert window.input.popup is None
    program.menu_items("File")[3].action()
    type_text(window, str(tmp_path / "copy.toml"))
    press_key(window, "enter")
    assert (tmp_path / "copy.toml").exists()


def test_p_and_l_pick_the_outline_tools_and_enter_and_escape_end_a_drawing(app):
    """Click corners, press Enter: a line is made; Esc or a tool change drops one."""
    window, program = app
    press_key(window, "p")
    assert program.session.tool == "polygon" and program.buttons["polygon"].checked
    press_key(window, "l")
    assert program.session.tool == "line" and not program.buttons["polygon"].checked
    for x, y in ((1.0, 1.0), (3.0, 1.0)):
        px, py = at(program, x, y)
        window.mouse_move(px, py)
        window.mouse_press(Event(px, py))
        window.mouse_release(Event(px, py))
    assert len(program.pane.outline) == 2
    press_key(window, "enter")
    assert program.session.world.obstacles[0].shape.name == "linestring"
    assert "linestring 1" in [row.text for row in program.tree.rows]
    px, py = at(program, 5.0, 5.0)
    window.mouse_press(Event(px, py))
    window.mouse_release(Event(px, py))
    press_key(window, "escape")
    assert not program.pane.outline
    window.mouse_press(Event(px, py))
    window.mouse_release(Event(px, py))
    program.set_tool("select")
    assert not program.pane.outline


def test_the_title_names_the_file_and_stars_unsaved_changes(app, tmp_path):
    """Untitled at first; a star after an edit; the file name after a save."""
    window, program = app
    assert (
        program.title == "untitled - demuvo" and window.appearance[0] == program.title
    )
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    assert program.title == "untitled * - demuvo"
    path = tmp_path / "mine.toml"
    program._save_to(str(path))  # pylint: disable=protected-access
    assert program.title == "mine.toml - demuvo"
    program.pane.actions.obstacle_drawn("circle", (7.0, 4.0, 0.0), (0.3,))
    assert program.title == "mine.toml * - demuvo"


def test_ctrl_q_asks_twice_when_there_is_unsaved_work_and_open_and_new_ask_too(app):
    """A saved world quits at once; an edited one needs the second ask."""
    window, program = app
    press_key(window, "q", ctrl=True)
    assert window.quit_requested
    window.quit_requested = False
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    press_key(window, "q", ctrl=True)
    assert not window.quit_requested
    press_key(window, "o", ctrl=True)
    assert window.input.popup is None
    press_key(window, "o", ctrl=True)
    assert window.input.popup is not None
    window.input.close_popup()
    press_key(window, "q", ctrl=True)
    assert not window.quit_requested


def test_shift_click_with_the_goal_tool_adds_a_waypoint_through_the_window(app):
    """A click sets the goal, a Shift click adds another, and the form lists both."""
    window, program = app
    program.select("r1")
    program.set_tool("goal")
    for y, shift in ((1.0, False), (5.0, True)):
        x, py = at(program, 8.0, y)
        window.mouse_move(x, py)
        window.mouse_press(Event(x, py, shift=shift))
        window.mouse_release(Event(x, py))
    assert edit.goals_of(program.session.world, "r1") == [0, 1]
    assert "g1x" in program.inspector.fields


def test_ctrl_z_and_ctrl_y_undo_and_redo_and_the_menu_follows(app):
    """The keys and the Edit menu agree; Undo is grey until there is something."""
    window, program = app
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert not lines["Undo"].enabled and not lines["Redo"].enabled
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    lines = {i.text: i for i in program.menu_items("Edit") if not i.separator}
    assert lines["Undo"].enabled
    press_key(window, "z", ctrl=True)
    assert not program.session.world.obstacles
    assert {i.text: i for i in program.menu_items("Edit") if not i.separator}[
        "Redo"
    ].enabled
    press_key(window, "y", ctrl=True)
    assert len(program.session.world.obstacles) == 1
    press_key(window, "z", ctrl=True)
    press_key(window, "z", ctrl=True)
    assert "Nothing to undo" in window.layout.chrome.status.message


def test_the_floor_has_no_switch_in_the_panel_or_the_menu(app):
    """It is part of how demuvo looks; every other layer can be turned off."""
    _, program = app
    assert "floor" not in program.layer_boxes and "costmap" in program.layer_boxes
    assert "floor" not in [i.text for i in program.menu_items("View")]
    assert program.session.scene.layers.shown["floor"] == 1.0


def test_the_goal_tool_is_grey_until_a_robot_is_selected_and_g_says_why(app):
    """No robot, no goal tool; selecting one turns it on; deselecting turns it off."""
    window, program = app
    assert not program.buttons["goal"].enabled
    press_key(window, "g")
    assert program.session.tool == "select"
    assert "Select a robot first" in window.layout.chrome.status.message
    program.select("r1")
    assert program.buttons["goal"].enabled
    press_key(window, "g")
    assert program.session.tool == "goal"
    program.select(None)
    assert not program.buttons["goal"].enabled and program.session.tool == "select"


def test_a_drawn_shape_returns_to_the_select_tool_so_the_next_drag_moves_it(app):
    """After a box is drawn the pointer selects again."""
    _, program = app
    program.set_tool("box")
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    assert program.session.tool == "select"
    program.set_tool("polygon")
    program.pane.actions.outline_drawn("polygon", [(5, 1), (6, 1), (5.5, 2)])
    assert program.session.tool == "select"
    program.set_tool("box")
    program.pane.actions.obstacle_drawn("box", (5.0, 3.0, 0.0), (1.0, 1.0))
    assert program.session.tool == "box"


def test_a_dragged_robot_takes_its_path_with_it_before_it_is_dropped(app, monkeypatch):
    """While the pointer moves the world follows now and then, path included."""
    window, program = app
    clock = [100.0]
    monkeypatch.setattr("demuvo.gui.paneactions.time.perf_counter", lambda: clock[0])
    session = program.session
    session.set_goal("r1", (9.0, 5.0, 0.0))
    program.refresh()
    x, y = at(program, 5.0, 3.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    tx, ty = at(program, 2.0, 2.0)
    clock[0] += 1.0
    window.mouse_move(tx, ty)
    assert session.run.poses["r1"][:2] == pytest.approx([2.0, 2.0], abs=0.05)
    assert session.run.path("r1")[0][:2] == pytest.approx([2.0, 2.0], abs=0.05)
    nx, ny = at(program, 1.0, 1.0)
    window.mouse_move(nx, ny)  # too soon: the world waits
    assert session.run.poses["r1"][:2] == pytest.approx([2.0, 2.0], abs=0.05)
    window.mouse_release(Event(nx, ny))
    assert session.run.poses["r1"][:2] == pytest.approx([1.0, 1.0], abs=0.05)
    assert session.history.undo() is None
    assert session.run.poses["r1"][:2] == pytest.approx([5.0, 3.0], abs=0.05)


def test_a_dragged_robot_over_an_obstacle_leaves_the_world_as_it_was(app, monkeypatch):
    """A refused place is skipped; the drop reports it."""
    window, program = app
    clock = [100.0]
    monkeypatch.setattr("demuvo.gui.paneactions.time.perf_counter", lambda: clock[0])
    program.pane.actions.obstacle_drawn("box", (8.0, 3.0, 0.0), (1.0, 1.0))
    x, y = at(program, 5.0, 3.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    clock[0] += 1.0
    tx, ty = at(program, 8.0, 3.0)
    window.mouse_move(tx, ty)
    assert session_pose(program) == pytest.approx([5.0, 3.0], abs=0.05)
    window.mouse_release(Event(tx, ty))
    assert "overlap" in window.layout.chrome.status.message


def session_pose(program):
    """Where r1 is in the run."""
    return program.session.run.poses["r1"][:2]


def test_the_status_bar_shows_how_much_of_real_time_a_run_keeps_up_with(app):
    """Frames at the pace of the clock read 100%; a stalled one reads low and warns."""
    window, program = app
    status = window.layout.chrome.status
    assert status.rate == ""
    program.toggle_play()
    now = program.clock.last
    for _ in range(7):
        now += 0.1
        program.after_tick(now)
    assert status.rate == "real time 100%" and not status.slow
    now += 2.0  # one frame took two seconds, and the run only counts a tenth of it
    program.after_tick(now)
    now += 0.6
    program.after_tick(now)
    assert status.slow and int(status.rate.split()[-1].rstrip("%")) < 90
    canvas_before = window.render()
    assert canvas_before.shape[2] == 3
    program.toggle_play()
    program.after_tick(now + 0.1)
    assert status.rate == "" and not status.slow


def test_the_percentage_is_of_the_speed_chosen_and_starts_over_when_it_changes(app):
    """At 4x the run is on time if it gains four seconds for each real second."""
    _, program = app
    program.pane.pill.speed.choose("4x")
    program.toggle_play()
    status = program.shell.layout.chrome.status
    now = program.clock.last
    for _ in range(14):
        now += 0.1
        program.after_tick(now)
    assert status.rate == "real time 100%"
    program.pane.pill.speed.choose("1x")
    assert program.clock.share(program.clock.last, 0.0, 1.0) is None


def test_a_dragged_obstacle_shows_an_outline_and_the_world_follows_at_the_drop(app):
    """The outline moves with the pointer; the world is edited once, on release."""
    window, program = app
    program.pane.actions.obstacle_drawn("circle", (2.0, 2.0, 0.0), (0.4,))
    program.pane.actions.obstacle_drawn("box", (7.0, 4.0, 0.0), (1.0, 1.0))
    sx, sy = at(program, 2.0, 2.0)
    window.mouse_move(sx, sy)
    window.mouse_press(Event(sx, sy))
    before = program.session.world
    for x in (3.0, 4.0):
        mx, my = at(program, x, 2.0)
        window.mouse_move(mx, my)
        window.render()
        assert program.pane.ghost is not None
        assert program.session.world is before
    window.mouse_release(Event(mx, my))
    assert program.session.world.obstacles[0].pose.xyt[0] == pytest.approx(
        4.0, abs=0.03
    )
    assert program.pane.ghost is None
    px, py = at(program, 7.0, 4.0)
    window.mouse_press(Event(px, py))
    qx, qy = at(program, 8.0, 4.0)
    window.mouse_move(qx, qy)
    window.render()
    window.mouse_release(Event(qx, qy))
    assert program.session.world.obstacles[1].pose.xyt[0] == pytest.approx(
        8.0, abs=0.03
    )
    assert program.session.history.undo() is None
    assert program.session.world.obstacles[1].pose.xyt[0] == pytest.approx(
        7.0, abs=0.03
    )


def test_the_log_panel_opens_itself_for_the_first_message_and_the_button_toggles_it(
    app,
):
    """A warning opens the bottom panel and puts a note in the status bar."""
    window, program = app
    docks = window.layout.docks
    assert not docks["bottom"].open
    program.session.log.add("warning", "AStar: no way from the start to the goal")
    program.after_tick(program.clock.last + 0.1)
    window.animator.tick(time.perf_counter() + 5.0)
    assert docks["bottom"].open and docks["bottom"].width == docks["bottom"].full
    assert program.buttons["bottom"].checked
    assert window.layout.chrome.status.notes == "1 message in the log"
    window.render()
    click(window, program.buttons["bottom"])
    window.animator.tick(time.perf_counter() + 5.0)
    assert not docks["bottom"].open
    program.session.log.add("warning", "another")
    program.after_tick(program.clock.last + 0.2)
    assert not docks["bottom"].open
    assert window.layout.chrome.status.notes == "2 messages in the log"
    press_key(window, "3", ctrl=True)
    window.animator.tick(time.perf_counter() + 5.0)
    assert docks["bottom"].open
    view = program.menu_items("View")
    next(i for i in view if i.text == "Log").action()
    window.animator.tick(time.perf_counter() + 5.0)
    assert not docks["bottom"].open


def test_clearing_the_log_empties_it_and_the_note(app):
    """The Clear button forgets the messages and the status note goes."""
    window, program = app
    program.session.log.add("warning", "one")
    program.after_tick(program.clock.last + 0.1)
    program.log_view.clear_button.click()
    assert not program.session.log.entries
    assert window.layout.chrome.status.notes == ""
