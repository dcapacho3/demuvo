"""Tests for actors in the window: the button, the list, the scene and the inspector."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from demuvo import edit
from demuvo.gui import look, scenedraw
from demuvo.gui.keys import CTRL, KEY
from demuvo.gui.sandbox import ACTOR_KEY, Sandbox
from demuvo.gui.session import Session
from demuvo.gui.shell import Shell
from demuvo.gui.widgets import Event, Section
from demuvo.verify import WorldError

# A motion named by a test, to see a form built for a class demuvo does not ship.
PLAIN = "demuvo.gui.tests.test_actor_gui:Plain"


@dataclass
class Plain:
    """A motion with nothing a form can show, so the form says so."""

    def configure(self, start, rng) -> None:
        """Keep where it starts."""
        self._start = start

    def pose_at(self, time: float):
        """Always the start pose."""
        return self._start


@dataclass
class Counted:
    """A motion with a whole-number setting, which a form shows as one.

    Attributes:
        repeats: how many times it would go round; nothing reads it.
    """

    repeats: int = 2

    def configure(self, start, rng) -> None:
        """Keep where it starts."""
        self._start = start

    def pose_at(self, time: float):
        """Always the start pose."""
        return self._start


COUNTED = "demuvo.gui.tests.test_actor_gui:Counted"


def start(session=None, width=1280, height=780):
    """A shell over a sandbox program, rendered once."""
    program = Sandbox(session or Session(edit.new_world()))
    window = Shell(program, width, height)
    window.render()
    return window, program


@pytest.fixture(name="app")
def make_app():
    """A blank sandbox window."""
    return start()


@pytest.fixture(name="with_actor")
def make_app_with_actor():
    """A sandbox window whose world has one actor, selected."""
    window, program = start()
    program._add_actor()
    program.select(f"{ACTOR_KEY}0")
    window.render()
    return window, program


def at(program, x, y):
    """The pixel of a world point."""
    sx, sy = program.session.scene.view.to_screen(x, y)
    return float(sx), float(sy)


def click(window, x, y):
    """Press and release at a pixel."""
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y))
    window.mouse_release(Event(x, y))


def test_the_button_adds_an_actor_and_selects_it(app):
    """It lands at a free place, standing still, and the inspector shows it."""
    window, program = app
    click(window, *_centre(program.buttons["add_actor"]))
    world = program.session.world
    assert [a.name for a in world.actors] == ["a1"]
    assert world.actors[0].motion is None
    assert program.session.scene.selected_actor == 0
    assert "ax" in program.inspector.fields


def test_the_shortcut_and_the_menu_add_an_actor(app):
    """Ctrl+M, and the Edit menu item."""
    window, program = app
    window.key(KEY["m"], 1, CTRL)
    assert len(program.session.world.actors) == 1
    item = next(i for i in program.menu_items("Edit") if i.text == "Add actor")
    item.action()
    assert len(program.session.world.actors) == 2


def test_the_scene_list_shows_the_actors_under_a_heading_of_their_own(with_actor):
    """With a count, in amber, and the selected row is the actor's."""
    _, program = with_actor
    rows = {row.key: row for row in program.tree.rows}
    assert f"{ACTOR_KEY}0" in rows
    assert rows[f"{ACTOR_KEY}0"].dot == look.AMBER
    assert any(row.text == "Actors (1)" for row in program.tree.rows)
    assert program.tree.selected == f"{ACTOR_KEY}0"


def test_a_world_with_no_actor_has_no_actor_heading(app):
    """The list is as it was before actors existed."""
    _, program = app
    assert all("Actor" not in row.text for row in program.tree.rows)


def test_clicking_an_actor_in_the_scene_selects_it(app):
    """And clicking the empty floor lets it go."""
    window, program = app
    program.edited(program.session.add_actor())
    window.render()
    x, y, _ = program.session.world.actors[0].pose.xyt
    click(window, *at(program, float(x), float(y)))
    assert program.session.scene.selected_actor == 0
    assert program.session.scene.selected is None


def test_dragging_an_actor_moves_it(with_actor):
    """The drop makes the edit, and the actor is still selected."""
    window, program = with_actor
    x, y, _ = program.session.world.actors[0].pose.xyt
    start_px = at(program, float(x), float(y))
    target = at(program, float(x) + 1.0, float(y))
    window.mouse_move(*start_px)
    window.mouse_press(Event(*start_px))
    window.mouse_move(*target)
    window.mouse_release(Event(*target))
    moved = program.session.world.actors[0].pose.xyt
    assert float(moved[0]) == pytest.approx(float(x) + 1.0, abs=0.05)
    assert program.session.scene.selected_actor == 0


def test_delete_removes_the_selected_actor(with_actor):
    """The Delete key, with an actor selected, deletes the actor."""
    window, program = with_actor
    window.key(KEY["delete"], 1, 0)
    assert program.session.world.actors == ()
    assert program.session.world.robots  # the robot is untouched


def test_the_inspector_edits_the_pose_the_body_and_the_flags(with_actor):
    """Every field of an actor that is not its motion."""
    window, program = with_actor
    fields = program.inspector.fields
    fields["ax"].set(2.0)
    fields["asize0"].set(0.4)
    assert program.session.world.actors[0].pose.xyt[0] == pytest.approx(2.0)
    assert program.session.world.actors[0].shape.radius == pytest.approx(0.4)
    program.inspector.fields["actor_solid"].activate()
    assert program.session.world.actors[0].solid is False
    program.inspector.fields["actor_collide"].activate()
    assert program.session.world.actors[0].collide is True
    window.render()


def test_the_inspector_gives_a_motion_and_shows_its_settings(with_actor):
    """Picking Waypoints starts it off with a path, so it runs at once."""
    window, program = with_actor
    program.inspector.fields["motion"].choose("Waypoints")
    spec = program.session.world.actors[0]
    assert spec.motion == "demuvo.actors:Waypoints"
    assert spec.motion_params["points"]  # started off for us
    window.render()
    fields = program.inspector.fields
    assert "motion.speed" in fields and "motion.mode" in fields
    fields["motion.speed"].set(0.8)
    assert program.session.world.actors[0].motion_params["speed"] == pytest.approx(0.8)
    fields["motion.mode"].choose("bounce")
    assert program.session.world.actors[0].motion_params["mode"] == "bounce"
    fields["motion.face"].activate()
    assert program.session.world.actors[0].motion_params["face"] is False


def test_a_motion_can_be_taken_away_again(with_actor):
    """Choosing "stands still" leaves an actor that stands and has no settings."""
    window, program = with_actor
    program.inspector.fields["motion"].choose("Waypoints")
    window.render()
    program.inspector.fields["motion"].choose("stands still")
    window.render()
    assert program.session.world.actors[0].motion is None
    assert "motion.speed" not in program.inspector.fields


def test_a_refused_setting_is_reported_and_nothing_changes(with_actor):
    """A speed of zero is no speed at all."""
    window, program = with_actor
    program.inspector.fields["motion"].choose("Waypoints")
    window.render()
    before = program.session.world
    program.inspector.fields["motion.speed"].set(0.0)
    assert program.session.world is before


def test_the_actor_is_drawn_and_found_under_the_pointer(with_actor):
    """The scene draws it where it is, and a pixel on it finds its place."""
    _, program = with_actor
    scene = program.session.scene
    x, y, _ = program.session.world.actors[0].pose.xyt
    sx, sy = scene.view.to_screen(float(x), float(y))
    assert scene.actor_at(float(sx), float(sy)) == 0
    far = scene.view.to_screen(float(x) + 2.0, float(y))
    assert scene.actor_at(float(far[0]), float(far[1])) is None


def test_the_actors_layer_can_be_switched_off(with_actor):
    """It is a layer like the obstacles, with its own switch."""
    _, program = with_actor
    assert program.layer_on["actors"] is True
    program.set_layer("actors", False)
    assert program.layer_on["actors"] is False


def test_an_actor_is_renamed_from_the_scene_list(with_actor):
    """F2 opens the box over its row, and the name follows."""
    window, program = with_actor
    program._renamed("actor", 0, "walker")
    assert program.session.world.actors[0].name == "walker"
    window.render()
    assert any(row.text == "walker" for row in program.tree.rows)


def test_the_window_follows_an_actor_while_the_run_plays(with_actor):
    """The drawn pose comes from the run, so a moving actor is drawn moving."""
    window, program = with_actor
    program.inspector.fields["motion"].choose("Waypoints")
    scene = program.session.scene
    before = np.array(scene.actor_pose(0))
    program.session.run.advance(1.0)
    scene.refresh_poses()
    assert not np.allclose(before, np.array(scene.actor_pose(0)))
    window.render()


def _centre(widget):
    """The centre of a widget."""
    return widget.x + widget.width / 2, widget.y + widget.height / 2


def test_a_box_actor_is_drawn_sized_and_found_like_a_disc(with_actor):
    """Every branch that asks whether an actor's body is a circle."""
    window, program = with_actor
    program.edited(program.session.edits.resize_actor(0, (0.8, 0.5)))
    window.render()
    scene = program.session.scene
    x, y, _ = program.session.world.actors[0].pose.xyt
    sx, sy = scene.view.to_screen(float(x), float(y))
    assert scene.actor_at(float(sx), float(sy)) == 0
    assert scene.actor_boxes()["a1"][0] < float(sx)
    fields = program.inspector.fields
    assert fields["asize0"].value == pytest.approx(0.8)
    fields["asize1"].set(0.6)
    assert program.session.world.actors[0].shape.size == pytest.approx((0.8, 0.6))


def test_a_motion_with_nothing_to_set_says_so(with_actor):
    """A class of one's own with no settings and no choices."""
    window, program = with_actor
    program.edited(program.session.edits.set_actor_motion(0, PLAIN))
    window.render()
    assert program.session.world.actors[0].motion == PLAIN
    assert not program.inspector.motion


def test_a_whole_number_setting_is_shown_as_one(with_actor):
    """An int field, which steps by one."""
    window, program = with_actor
    program.edited(program.session.edits.set_actor_motion(0, COUNTED))
    window.render()
    field = program.inspector.fields["motion.repeats"]
    assert field.value == 2
    field.set(5)
    assert program.session.world.actors[0].motion_params["repeats"] == 5


def test_clicking_an_actor_while_a_robot_is_selected_refreshes(app):
    """The selection moves from the robot to the actor."""
    window, program = app
    program._add_actor()
    program.select("r1")
    window.render()
    x, y, _ = program.session.world.actors[0].pose.xyt
    click(window, *at(program, float(x), float(y)))
    assert program.session.scene.selected_actor == 0
    assert program.session.scene.selected is None


def test_selecting_the_actor_that_is_already_selected_changes_nothing(with_actor):
    """So the window does not rebuild the form on every click."""
    _, program = with_actor
    assert program.session.select_actor(0) is False
    assert program.session.select_obstacle(None) is True


def test_f2_renames_the_selected_actor(with_actor):
    """The box opens over the actor's row with its name in it."""
    window, program = with_actor
    program._rename_selected()
    popup = window.input.popup
    assert popup is not None
    popup.field.text = "walker"
    popup.field.on_submit(popup.field.text)
    assert program.session.world.actors[0].name == "walker"


def test_an_actor_cannot_be_added_or_edited_while_a_run_is_on(with_actor):
    """Everything is locked once the run has begun."""
    _, program = with_actor
    program.session.step_once()
    assert "Reset" in program.session.add_actor()
    assert "Reset" in program.session.edits.delete_actor()
    assert "Reset" in program.session.edits.move_actor(0, (1.0, 1.0, 0.0))


def test_delete_with_no_actor_selected_is_no_actor_edit(app):
    """`delete_actor` is asked only when an actor is selected."""
    _, program = app
    assert program.session.edits.delete_actor() is None


def test_an_actor_that_will_not_fit_is_reported(app, monkeypatch):
    """There is nowhere free, so the window says so and adds nothing."""
    _, program = app

    def no_room(*_args, **_kwargs):
        raise WorldError("there is no free place for another robot")

    monkeypatch.setattr(edit, "free_pose", no_room)
    assert "no free place" in program.session.add_actor()
    assert program.session.world.actors == ()


def test_an_actor_route_is_a_layer_and_is_drawn_numbered():
    """The route shows with the actor, and turning its layer off hides it."""
    from demuvo.gui.raster import Canvas

    session = Session(edit.new_world())
    session.add_actor()
    session.edits.set_actor_motion(0, "demuvo.actors:Waypoints")
    session.edits.set_actor_motion_params(0, {"points": [[7.0, 1.0], [7.0, 5.0]]})
    scene = session.scene
    scene.fit((0, 0, 800, 480))
    assert scenedraw.route_of(session.world.actors[0]).shape == (3, 2)
    shown, hidden = Canvas(800, 480, "#ffffff"), Canvas(800, 480, "#ffffff")
    scene.draw_overlays(shown)
    scene.layers.shown["routes"] = 0.0
    scene.draw_overlays(hidden)
    assert (shown.px != hidden.px).any()


def _selected_actor_with_route():
    """A window with one actor on a two-point route, selected."""
    window, program = start()
    program._add_actor()
    program.select(f"{ACTOR_KEY}0")
    program.inspector.fields["motion"].choose("Waypoints")
    program.session.edits.set_actor_motion_params(
        0, {"points": [[7.0, 1.0], [7.0, 5.0]]}
    )
    program.refresh()
    return program, window, 0


def _open_waypoints(program):
    """Open the Waypoints section of the inspector, as a click on its heading does."""
    head = next(
        c
        for c in program.inspector.column.children
        if isinstance(c, Section) and c.title == "Waypoints"
    )
    head.activate()
    return head


def test_the_waypoints_section_shows_a_row_per_point_when_opened():
    """Closed its rows are hidden; opened, each point has x and y."""
    program, _, _ = _selected_actor_with_route()
    fields = program.inspector.fields
    assert not fields["wp0x"].parent.visible
    head = _open_waypoints(program)
    assert head.title == "Waypoints" and not head.collapsed
    assert fields["wp0x"].parent.visible
    assert (fields["wp1x"].value, fields["wp1y"].value) == (7.0, 5.0)


def test_waypoints_are_added_edited_and_removed_from_the_fold():
    """Each button and field changes the points of the actor, and undo restores them."""
    program, _, index = _selected_actor_with_route()
    _open_waypoints(program)
    program.inspector.fields["wp_add"].on_click()
    points = program.session.world.actors[index].motion_params["points"]
    assert points == [[7.0, 1.0], [7.0, 5.0], [7.5, 5.0]]
    program.inspector.fields["wp0x"].set(2.0)
    assert program.session.world.actors[index].motion_params["points"][0] == [2.0, 1.0]
    program.inspector.fields["wp0remove"].on_click()
    assert len(program.session.world.actors[index].motion_params["points"]) == 2


def test_the_last_waypoint_has_no_remove_button():
    """A route needs one point, so the button is not offered for it."""
    program, _, index = _selected_actor_with_route()
    _open_waypoints(program)
    program.inspector.fields["wp1remove"].on_click()
    assert "wp0remove" not in program.inspector.fields
    assert len(program.session.world.actors[index].motion_params["points"]) == 1


def _points(program):
    """The waypoints of the first actor."""
    return program.session.world.actors[0].motion_params["points"]


def _drag(window, program, start, target):
    """Press at one world point and release at another."""
    first, last = at(program, *start), at(program, *target)
    window.mouse_move(*first)
    window.mouse_press(Event(*first))
    window.mouse_move(*last)
    window.mouse_release(Event(*last))


def test_dragging_a_waypoint_dot_moves_that_waypoint():
    """The drop sets the point; the others stay, and the actor does not move."""
    program, window, _ = _selected_actor_with_route()
    before = program.session.world.actors[0].pose.xyt.copy()
    first, last = at(program, 7.0, 5.0), at(program, 8.0, 4.0)
    window.mouse_move(*first)
    window.mouse_press(Event(*first))
    window.mouse_move(*last)
    live = program.session.scene.route(0)
    window.mouse_release(Event(*last))
    assert live[2] == pytest.approx([8.0, 4.0], abs=0.05)
    assert _points(program)[0] == [7.0, 1.0]
    assert _points(program)[1] == pytest.approx([8.0, 4.0], abs=0.05)
    assert (program.session.world.actors[0].pose.xyt == before).all()


def test_dragging_the_middle_of_a_leg_puts_a_new_waypoint_there():
    """The leg from the first to the second point is cut at its middle."""
    program, window, _ = _selected_actor_with_route()
    _drag(window, program, (7.0, 3.0), (6.0, 3.0))
    assert len(_points(program)) == 3
    assert _points(program)[1] == pytest.approx([6.0, 3.0], abs=0.05)
    assert _points(program)[2] == [7.0, 5.0]


def test_shift_click_on_the_floor_adds_a_waypoint_at_the_end():
    """Only with an actor that has a route selected, and not on a body."""
    program, window, _ = _selected_actor_with_route()
    x, y = at(program, 2.0, 5.0)
    window.mouse_move(x, y)
    window.mouse_press(Event(x, y, shift=True))
    window.mouse_release(Event(x, y))
    assert _points(program)[-1] == pytest.approx([2.0, 5.0], abs=0.05)
    assert len(_points(program)) == 3


def test_a_press_away_from_the_route_does_nothing_to_it():
    """A plain click on the floor leaves the points as they were."""
    program, window, _ = _selected_actor_with_route()
    click(window, *at(program, 2.0, 5.0))
    assert len(_points(program)) == 2


def test_an_actor_with_no_route_is_not_given_one_by_a_click():
    """The route code only acts on an actor whose motion has points."""
    window, program = start()
    program._add_actor()
    program.select(f"{ACTOR_KEY}0")
    window.mouse_move(100, 100)
    window.mouse_press(Event(100, 100, shift=True))
    window.mouse_release(Event(100, 100))
    assert program.session.world.actors[0].motion is None


def test_no_waypoint_is_under_a_pixel_without_an_actor_or_a_route():
    """Nothing selected, or an actor that stands still, has no point to pick."""
    window, program = start()
    assert program.session.scene.point_at(100, 100) is None
    program._add_actor()
    program.select(f"{ACTOR_KEY}0")
    assert program.session.scene.selected_actor == 0
    assert program.session.scene.point_at(100, 100) is None


def test_a_waypoint_gets_its_own_wait_and_heading_from_the_fold():
    """The wait makes it a three-number point, the heading switch a four-number one."""
    program, window, _ = _selected_actor_with_route()
    _open_waypoints(program)
    program.inspector.fields["wp1wait"].set(3.0)
    assert _points(program)[1] == [7.0, 5.0, 3.0]
    program.inspector.fields["wp1turn"].activate()
    assert _points(program)[1] == [7.0, 5.0, 3.0, 0.0]
    program.inspector.fields["wp1th"].set(90.0)
    assert _points(program)[1][3] == pytest.approx(math.pi / 2)
    program.inspector.fields["wp1turn"].activate()
    assert len(_points(program)[1]) == 3
    program.inspector.fields["wp1wait"].set(0.0)
    assert _points(program)[1] == [7.0, 5.0]


def test_dragging_a_waypoint_keeps_its_wait_and_heading():
    """Only the place changes."""
    program, window, _ = _selected_actor_with_route()
    program.session.edits.set_actor_motion_params(
        0, {"points": [[7.0, 1.0], [7.0, 5.0, 2.0, 1.0]]}
    )
    program.refresh()
    _drag(window, program, (7.0, 5.0), (8.0, 4.0))
    assert _points(program)[1][2:] == [2.0, 1.0]
