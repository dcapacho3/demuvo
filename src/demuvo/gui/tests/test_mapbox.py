"""Tests for the New map card - four maps to choose from, and the one that is used."""

import numpy as np
import pytest

from demuvo import edit
from demuvo.gui import look
from demuvo.gui.keys import KEY
from demuvo.gui.mapbox import NewMapBox
from demuvo.gui.raster import Canvas, rgba
from demuvo.gui.session import Session
from demuvo.gui.widgets import Event

from .fakes import click_window, press_key, start

STYLES = ["field", "rooms", "caves", "maze"]


def card(program):
    """The card that is open."""
    return program.shell.input.popup


@pytest.fixture(name="app")
def make_app():
    """A blank sandbox with the card open."""
    window, program = start()
    program._new_map()  # pylint: disable=protected-access
    return window, program


def press_on(box, widget):
    """Press and let go on a widget of the card."""
    x, y = widget.x + widget.width / 2, widget.y + widget.height / 2
    box.on_press(Event(x, y))
    box.on_release(Event(x, y))


def test_the_card_shows_a_map_of_every_style_for_the_same_seed(app):
    """A student sees what each style makes before choosing."""
    window, program = app
    box = card(program)
    assert isinstance(box, NewMapBox)
    assert list(box.drafts) == STYLES
    assert all(draft[1] is not None for draft in box.drafts.values())
    assert {draft[0][0] for draft in box.drafts.values()} == {box.recipe["seed"]}
    assert window.render() is not None


def test_the_card_opens_from_the_file_menu_the_tool_bar_and_ctrl_g():
    """Three ways in, and no way in while a run is in progress."""
    window, program = start()
    items = {i.text: i for i in program.menu_items("File")}
    assert items["New map…"].enabled and items["New map…"].shortcut == "Ctrl+G"
    click_window(window, program.buttons["map"])
    assert isinstance(card(program), NewMapBox)
    window.close_popup()
    press_key(window, "g", ctrl=True)
    assert isinstance(card(program), NewMapBox)
    window.close_popup()
    program.session.step_once()
    assert not {i.text: i for i in program.menu_items("File")}["New map…"].enabled
    program._new_map()  # pylint: disable=protected-access
    assert card(program) is None


def test_a_click_on_a_tile_chooses_its_style(app):
    """The chosen tile is the one that is drawn, and Use takes."""
    _, program = app
    box = card(program)
    left, top, _, _ = box.tiles["rooms"]
    box.on_press(Event(left + 5, top + 5))
    assert box.recipe["style"] == "rooms"


def test_the_arrow_keys_move_from_style_to_style(app):
    """Right and left by one, down and up by a row, round the four."""
    _, program = app
    box = card(program)
    box.recipe["style"] = "field"
    for name, style in (("right", "rooms"), ("down", "maze"), ("left", "caves")):
        box.on_key(Event(key=KEY[name]))
        assert box.recipe["style"] == style
    assert not box.on_key(Event(key=KEY["a"]))


def test_shuffle_draws_the_next_seed_of_every_style(app):
    """The seed shows, so a student can say which map it was."""
    _, program = app
    box = card(program)
    seed, before = box.recipe["seed"], dict(box.drafts)
    press_on(box, box.buttons["shuffle"])
    assert box.recipe["seed"] == seed + 1
    assert all(box.drafts[name][1:] != before[name][1:] for name in STYLES)


def test_using_a_map_replaces_the_obstacles_keeps_the_robot_and_can_be_undone(app):
    """The old map comes back with Undo, and the status bar says which was drawn."""
    window, program = app
    box = card(program)
    session = program.session
    before = session.world
    box.recipe["style"] = "rooms"
    press_on(box, box.buttons["yes"])
    assert card(program) is None
    assert session.world is not before
    assert [r.name for r in session.world.robots] == ["r1"]
    assert all(o.name.startswith("wall ") for o in session.world.obstacles)
    assert "rooms map" in window.layout.chrome.status.message
    assert session.history.undo() is None
    assert session.world == before


def test_the_map_that_is_used_is_the_map_that_was_shown(app):
    """The card draws what it showed: same style, same seed, same walls."""
    _, program = app
    box = card(program)
    box.recipe["style"] = "caves"
    shown = box.drafts["caves"][1]
    press_on(box, box.buttons["yes"])
    assert program.session.world.obstacles == shown.obstacles


def test_enter_uses_the_chosen_map_and_cancel_changes_nothing(app):
    """Enter is the quick way; Cancel leaves the world alone."""
    window, program = app
    before = program.session.world
    press_on(card(program), card(program).buttons["no"])
    assert card(program) is None and program.session.world is before
    program._new_map()  # pylint: disable=protected-access
    assert card(program).on_key(Event(key=KEY["enter"]))
    assert program.session.world is not before
    del window


def test_a_start_and_a_goal_are_placed_when_asked_for(app):
    """The first robot moves to the start and gets a goal."""
    _, program = app
    box = card(program)
    pose = program.session.world.robots[0].pose.xyt.tolist()
    box.pair.set(True)
    assert box.recipe["pair"] is True
    press_on(box, box.buttons["yes"])
    world = program.session.world
    assert world.robots[0].pose.xyt.tolist() != pose and len(world.tasks) == 1
    assert program.session.edits.recipe["pair"] is True


def test_the_loops_slider_is_for_the_maze_and_is_read_when_it_is_let_go(app):
    """A drag draws one maze at the end, not one for every pixel."""
    _, program = app
    box = card(program)
    assert not box.loops.visible
    box.recipe["style"] = "maze"
    box.on_key(Event(key=KEY["right"]))
    box.on_key(Event(key=KEY["left"]))
    assert box.loops.visible
    x, y = box.loops.x + box.loops.width - 8, box.loops.y + 16
    box.on_press(Event(x, y))
    box.on_move(Event(x, y))
    drawn = box.drafts["maze"]
    assert box.recipe["loops"] == 0.0
    box.on_release(Event(x, y))
    assert box.recipe["loops"] == pytest.approx(1.0)
    assert box.drafts["maze"] is not drawn


def test_a_world_with_no_robot_asks_for_the_radius_and_gets_a_robot_with_a_pair():
    """The map needs a robot size, so the card asks; a pair then adds the robot."""
    window, program = start(Session(edit.empty_world()))
    program._new_map()  # pylint: disable=protected-access
    box = card(program)
    assert box.radius.visible
    x, y = box.radius.x + box.radius.width * 0.5, box.radius.y + 16
    box.on_press(Event(x, y))
    box.on_release(Event(x, y))
    assert box.recipe["radius"] == pytest.approx(0.35, abs=0.03)
    box.pair.set(True)
    press_on(box, box.buttons["yes"])
    assert len(program.session.world.robots) == 1
    assert window.render() is not None


def test_a_style_that_cannot_be_drawn_says_why_and_cannot_be_used():
    """No map is better than a broken one, and the reason is in the tile."""
    session = Session(edit.set_size(edit.empty_world(), (1.0, 1.0)))
    window, program = start(session)
    program._new_map()  # pylint: disable=protected-access
    box = card(program)
    failed = [n for n in STYLES if box.drafts[n][1] is None]
    assert failed
    box.recipe["style"] = failed[0]
    box._refresh()  # pylint: disable=protected-access
    assert not box.buttons["yes"].enabled
    before = session.world
    assert box.on_key(Event(key=KEY["enter"]))
    assert session.world is before
    assert window.render() is not None


def test_the_tiles_and_the_chosen_one_are_drawn(app):
    """The chosen tile has a teal frame and the others a grey one."""
    _, program = app
    box = card(program)
    canvas = Canvas(1280, 780, look.WINDOW)
    box.draw(canvas)
    left, top, _, height = box.tiles[box.recipe["style"]]
    teal = canvas.px[round(top + height / 2), round(left)]
    other = box.tiles["field" if box.recipe["style"] != "field" else "rooms"]
    grey = canvas.px[round(other[1] + other[3] / 2), round(other[0])]
    assert teal.tolist() != grey.tolist()


def test_the_pointer_lights_a_button_and_a_press_off_the_controls_is_ignored(app):
    """Moving over a button lights it; the card's own empty space does nothing."""
    _, program = app
    box = card(program)
    yes = box.buttons["yes"]
    box.on_move(Event(yes.x + 5, yes.y + 5))
    assert yes.hover
    box.on_move(Event(box.x + 2, box.y + 2))
    assert not yes.hover
    box.on_press(Event(box.x + 2, box.y + 2))
    box.on_release(Event(box.x + 2, box.y + 2))
    assert card(program) is box


def has(canvas, hue, region=None, near=0.02):
    """True if some pixel of the canvas, or of a region of it, is close to a colour."""
    pixels = canvas.px if region is None else canvas.px[region]
    return bool((abs(pixels - np.array(rgba(hue)[:3])).max(axis=-1) < near).any())


def test_a_start_a_goal_and_the_loops_are_drawn_when_they_are_on(app):
    """The teal start and the amber goal sit on the picture; the maze has loops."""
    _, program = app
    box = card(program)
    box.pair.set(True)
    box.recipe["style"] = "maze"
    box._refresh()  # pylint: disable=protected-access
    canvas = Canvas(1280, 780, look.WINDOW)
    box.draw(canvas)
    picture = (slice(round(box.y) + 60, round(box.y + box.height) - 100),) * 1
    assert has(canvas, look.TEAL, picture) and has(canvas, look.AMBER, picture)
    assert box.loops.visible


def test_the_reason_a_style_cannot_be_drawn_is_written_in_its_tile():
    """The words wrap inside the tile and stop where it ends."""
    session = Session(edit.set_size(edit.empty_world(), (1.0, 1.0)))
    _, program = start(session)
    program._new_map()  # pylint: disable=protected-access
    box = card(program)
    canvas = Canvas(1280, 780, look.WINDOW)
    box.draw(canvas)
    failed = next(n for n in STYLES if box.drafts[n][1] is None)
    left, top, width, height = box.tiles[failed]
    tile = (
        slice(round(top), round(top + height)),
        slice(round(left), round(left + width)),
    )
    assert has(canvas, look.WARN_BG, tile)
    assert has(canvas, look.WARN_TEXT, tile, near=0.25)


def test_shuffling_the_start_and_goal_keeps_the_maps():
    """The pair has its own seed, so the same map is tried from another place."""
    session = Session(edit.set_size(edit.new_world(), (20.0, 14.0)))
    _, program = start(session)
    program._new_map()  # pylint: disable=protected-access
    box = card(program)
    assert not box.buttons["pair"].enabled
    box.pair.set(True)
    assert box.buttons["pair"].enabled
    first = box.drafts["rooms"][1]
    press_on(box, box.buttons["pair"])
    second = box.drafts["rooms"][1]
    assert box.recipe["seed"] == box.recipe["pair_seed"] - 1
    assert second.obstacles == first.obstacles
    assert second.robots[0].pose != first.robots[0].pose
    assert second.tasks != first.tasks
    box.recipe["style"] = "rooms"
    press_on(box, box.buttons["yes"])
    assert program.session.world.robots[0].pose == second.robots[0].pose


def test_the_distance_slider_moves_the_start_and_goal_apart():
    """The pair keeps at least the asked share of the world's diagonal between it."""
    session = Session(edit.set_size(edit.new_world(), (20.0, 14.0)))
    _, program = start(session)
    program._new_map()  # pylint: disable=protected-access
    box = card(program)
    assert not box.apart.visible
    box.pair.set(True)
    assert box.apart.visible
    diagonal = float(np.hypot(20.0, 14.0))
    for at in (0.15, 0.85):
        x, y = box.apart.x + at * box.apart.width, box.apart.y + 16
        box.on_press(Event(x, y))
        box.on_release(Event(x, y))
        assert box.recipe["apart"] == pytest.approx(box.apart.value, abs=0.01)
        world = box.drafts["rooms"][1]
        start_, goal = world.robots[0].pose, world.tasks[-1].goal
        gap = float(np.linalg.norm(start_.translation - goal.translation))
        assert gap >= box.recipe["apart"] * diagonal - 1e-6
