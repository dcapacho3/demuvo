"""Tests for renaming a robot or an obstacle from the scene list."""

import pytest

from demuvo.gui.keys import KEY
from demuvo.gui.widgets import Event

from .fakes import press_key, start, type_text


@pytest.fixture(name="app")
def make_app():
    """A blank sandbox window."""
    return start()


def double_click(window, program, key, clock):
    """Two quick clicks on a row of the scene list."""
    x, y, width, height = program.tree.row_rect(key)
    point = (x + width / 2, y + height / 2)
    for _ in range(2):
        window.mouse_move(*point)
        window.mouse_press(Event(*point))
        window.mouse_release(Event(*point))
        clock[0] += 0.1


def test_a_double_click_on_a_robot_row_renames_it_with_its_goal_and_undo_restores_it(
    app, monkeypatch
):
    """The text box opens over the row; Enter renames; the goal and selection follow."""
    window, program = app
    clock = [10.0]
    monkeypatch.setattr("demuvo.gui.inputs.time.perf_counter", lambda: clock[0])
    program.session.set_goal("r1", (8.0, 1.0, 0.0))
    program.refresh()
    double_click(window, program, "r1", clock)
    box = window.input.popup
    assert box is not None and box.field.buffer == "r1"
    type_text(window, "scout")
    press_key(window, "enter")
    assert window.input.popup is None
    world = program.session.world
    assert [r.name for r in world.robots] == ["scout"] and world.tasks[
        0
    ].robot == "scout"
    assert program.session.scene.selected == "scout"
    assert [row.text for row in program.tree.rows][:2] == ["Robots (1)", "scout"]
    assert program.session.history.undo() is None
    assert program.session.world.robots[0].name == "r1"


def test_a_double_click_on_an_obstacle_row_names_it_and_a_name_shows_in_the_list(
    app, monkeypatch
):
    """The name replaces "box 1" in the list and the inspector; a repeat is refused."""
    window, program = app
    clock = [10.0]
    monkeypatch.setattr("demuvo.gui.inputs.time.perf_counter", lambda: clock[0])
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    program.pane.actions.obstacle_drawn("circle", (7.0, 4.0, 0.0), (0.3,))
    double_click(window, program, "obstacle:0", clock)
    type_text(window, "table")
    press_key(window, "enter")
    assert "table" in [row.text for row in program.tree.rows]
    assert program.session.world.obstacles[0].name == "table"
    clock[0] += 5.0
    double_click(window, program, "obstacle:1", clock)
    type_text(window, "table")
    press_key(window, "enter")
    assert "already exists" in window.layout.chrome.status.message
    program.session.select_obstacle(0)
    program.refresh()
    titles = [c.name for c in program.inspector.column.children if hasattr(c, "name")]
    assert titles == ["table"]


def test_f2_renames_the_selected_item_and_a_locked_world_refuses(app):
    """F2 opens the box for the selection; nothing is selected or locked: no box."""
    window, program = app
    window.key(KEY["f2"], 1, 0)
    assert window.input.popup is None
    program.select("r1")
    window.key(KEY["f2"], 1, 0)
    assert window.input.popup is not None
    press_key(window, "escape")
    assert window.input.popup is None and window.focus is None
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    window.key(KEY["f2"], 1, 0)
    assert window.input.popup.field.buffer == ""
    press_key(window, "escape")
    program.step_once()
    program._rename("r1", (0, 0, 100, 30))  # pylint: disable=protected-access
    assert window.input.popup is None
    assert "Reset" in window.layout.chrome.status.message


def test_clicking_away_from_the_name_box_closes_it_and_typing_no_longer_goes_to_it(
    app,
):
    """The focus is given back, so the Delete key deletes in the scene again."""
    window, program = app
    program.pane.actions.obstacle_drawn("box", (2.0, 2.0, 0.0), (1.0, 0.5))
    window.key(KEY["f2"], 1, 0)
    assert window.input.popup is not None
    window.mouse_press(Event(700, 500))
    window.mouse_release(Event(700, 500))
    assert window.input.popup is None and window.focus is None
