"""Tests for zooming and moving the scene: camera, pane, keys and scale bar."""

# pylint: disable=protected-access  # some tests read the scene's cache and clock

import numpy as np
import pytest

from demuvo import edit, load_world
from demuvo.gui import look
from demuvo.gui.keys import KEY
from demuvo.gui.raster import Canvas, rgba
from demuvo.gui.robots import MIN_RADIUS, Footprint, display_scale
from demuvo.gui.sandbox import ZOOM_KEY, Sandbox
from demuvo.gui.scenedraw import MIN_ACTOR_RADIUS
from demuvo.gui.scenepane import ZOOM_STEP
from demuvo.gui.session import Session
from demuvo.gui.shell import Shell
from demuvo.gui.view import KEEP, MAX_SCALE, MIN_ZOOM, Camera, View
from demuvo.gui.widgets import Event
from demuvo.gui.zoombar import draw_scale_bar, nice_length

ROOM = 60.0
RECT = (0.0, 0.0, 1000.0, 700.0)


def base_view():
    """The fitted view of a 10 by 6 metre board in a 1000 by 700 pixel pane."""
    return View((0.0, 0.0, 10.0, 6.0)).fit(RECT, ROOM)


def test_the_fitted_camera_gives_the_fitted_view_and_100_percent():
    """No zoom and no shift change nothing."""
    base, camera = base_view(), Camera()
    assert camera.view(base, ROOM) == base and camera.percent == 100


def test_a_zoom_keeps_the_world_point_under_the_pointer_where_it_was():
    """Zooming in at a pixel leaves the same place under it, and doubles the scale."""
    base, camera = base_view(), Camera()
    about = (300.0, 200.0)
    before = base.to_world(*about)
    camera.zoom_by(2.0, about, base, ROOM)
    view = camera.view(base, ROOM)
    assert view.scale == pytest.approx(2 * base.scale)
    assert view.to_world(*about) == pytest.approx(before)
    assert camera.percent == 200


def test_a_zoom_with_no_pointer_is_about_the_middle_of_the_free_area():
    """Keys zoom about the middle, so the centre of the board stays at the centre."""
    base, camera = base_view(), Camera()
    camera.zoom_by(1.5, None, base, ROOM)
    centre = Camera.centre(RECT, ROOM)
    view = camera.view(base, ROOM)
    assert view.to_world(*centre) == pytest.approx(base.to_world(*centre))


def test_the_zoom_stops_at_the_limits():
    """Out to 40 percent, in to 400 pixels a metre."""
    base, camera = base_view(), Camera()
    camera.zoom_by(0.001, None, base, ROOM)
    assert camera.zoom == MIN_ZOOM
    camera.zoom_by(1e6, None, base, ROOM)
    assert camera.view(base, ROOM).scale == pytest.approx(MAX_SCALE)
    small = View((0.0, 0.0, 10.0, 6.0)).fit((0.0, 0.0, 100.0, 100.0), 0.0)
    huge = Camera()
    huge.zoom = 0.01
    assert huge.view(small, 0.0).scale > 0  # an out of range zoom is brought back


def test_a_pan_moves_the_picture_and_cannot_lose_the_board():
    """The board keeps `KEEP` pixels in the window, whichever way it is pushed."""
    base, camera = base_view(), Camera()
    camera.pan_by(5000, 5000)
    left, top, _, _ = camera.view(base, ROOM).board()
    assert left == pytest.approx(RECT[2] - KEEP) and top == pytest.approx(
        RECT[3] - KEEP
    )
    camera.reset()
    camera.pan_by(-9000, -9000)
    _, _, right, bottom = camera.view(base, ROOM).board()
    assert right == pytest.approx(KEEP) and bottom == pytest.approx(KEEP)
    camera.reset()
    camera.pan_by(30, -20)
    assert camera.view(base, ROOM).ox == pytest.approx(base.ox + 30)


def test_a_box_is_framed_in_the_middle_and_the_whole_board_is_framed_again_by_reset():
    """Framing centres the box; reset gives back the fitted view."""
    base, camera = base_view(), Camera()
    camera.frame((4.0, 2.0, 6.0, 4.0), base, ROOM)
    view = camera.view(base, ROOM)
    sx, sy = view.to_screen(5.0, 3.0)
    assert (float(sx), float(sy)) == pytest.approx(Camera.centre(RECT, ROOM))
    camera.reset()
    assert camera.view(base, ROOM) == base


def test_a_scale_bar_length_is_a_round_number_that_fits():
    """The longest of the round lengths that is under 150 pixels."""
    assert nice_length(100.0) == 1.0
    assert nice_length(10.0) == 10.0
    assert nice_length(10000.0) == 0.05  # nothing fits: the shortest
    assert nice_length(0.001) == 500.0  # all fit: the longest


def test_the_scale_bar_draws_and_names_the_zoom_only_when_it_is_not_100():
    """Something is drawn, and a zoomed view differs from a fitted one."""
    view = base_view()
    plain, zoomed = Canvas(1000, 700), Canvas(1000, 700)
    draw_scale_bar(plain, view, 100)
    draw_scale_bar(zoomed, view, 250)
    assert plain.px.any() and not np.array_equal(plain.px, zoomed.px)


def start(world=None):
    """A shell over a sandbox, rendered once."""
    session = Session(world or edit.new_world())
    program = Sandbox(session)
    window = Shell(program, 1280, 780)
    window.render()
    return window, program


def scene_middle(program):
    """A pixel inside the scene."""
    x, y, width, height = program.pane.rect()
    return x + width / 2, y + height / 2 - 40


def test_the_wheel_zooms_the_scene_about_the_pointer():
    """A notch up zooms in by a step, a notch down out again, over the scene."""
    window, program = start()
    scene = program.session.scene
    x, y = scene_middle(program)
    before = scene.view.to_world(x, y)
    window.mouse_move(x, y)
    window.mouse_scroll(1.0)
    assert scene.camera.zoom == pytest.approx(ZOOM_STEP)
    assert scene.view.to_world(x, y) == pytest.approx(before)
    window.mouse_scroll(-1.0)
    assert scene.camera.zoom == pytest.approx(1.0)


def test_the_middle_button_moves_the_view_and_a_release_stops_it():
    """Press, drag and release the middle button: the picture follows the pointer."""
    window, program = start()
    scene = program.session.scene
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    start_ox = scene.view.ox
    window.mouse_press(Event(x, y, button=2))
    window.mouse_move(x + 40, y + 10)
    assert scene.view.ox == pytest.approx(start_ox + 40)
    window.mouse_release(Event(x + 40, y + 10, button=2))
    window.mouse_move(x + 90, y + 10)
    assert scene.view.ox == pytest.approx(start_ox + 40)


def test_a_middle_press_off_any_widget_does_nothing():
    """Over no widget at all, or with a menu open, there is nothing to move."""
    window, program = start()
    window.mouse_press(Event(-5, -5, button=2))  # off the window
    assert program.pane.panning is None
    program.shell.input.open_menu("View")
    x, y = scene_middle(program)
    window.mouse_press(Event(x, y, button=2))
    assert program.pane.panning is None


def test_another_mouse_button_does_nothing_to_the_view():
    """A fourth button, say, is not a pan."""
    window, program = start()
    x, y = scene_middle(program)
    window.mouse_press(Event(x, y, button=3))
    assert program.pane.panning is None


def test_the_hand_tool_moves_the_view_with_the_left_button():
    """H picks it, a left drag moves the scene, and nothing is edited."""
    window, program = start()
    program.command(KEY["h"], False)
    assert program.session.tool == "hand" and program.buttons["hand"].checked
    scene = program.session.scene
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    start_ox = scene.view.ox
    window.mouse_press(Event(x, y))
    window.mouse_move(x - 30, y)
    window.mouse_release(Event(x - 30, y))
    assert scene.view.ox == pytest.approx(start_ox - 30)
    assert not program.session.edits.modified


def test_the_keys_zoom_in_out_and_fit_and_the_menu_does_the_same():
    """Plus, minus and 0, with Ctrl or on the keypad, and the View menu's lines."""
    _, program = start()
    scene = program.session.scene
    program.command(KEY["equal"], False)
    assert scene.camera.zoom == pytest.approx(ZOOM_KEY)
    program.command(KEY["kp_add"], False)
    program.command(KEY["equal"], True)
    assert scene.camera.zoom == pytest.approx(ZOOM_KEY**3)
    for key, ctrl in (
        (KEY["minus"], False),
        (KEY["kp_subtract"], False),
        (KEY["minus"], True),
    ):
        program.command(key, ctrl)
    assert scene.camera.zoom == pytest.approx(1.0)
    program.command(KEY["equal"], False)
    program.command(KEY["0"], False)
    assert scene.camera.zoom == 1.0
    program.command(KEY["equal"], False)
    program.command(KEY["home"], False)
    assert scene.camera.zoom == 1.0
    items = {m.text: m for m in program.menu_items("View") if m.text}
    items["Zoom in"].action()
    items["Zoom out"].action()
    items["Fit the world"].action()
    assert scene.camera.zoom == 1.0 and not items["Zoom to selection"].enabled


def test_f_zooms_to_the_selected_item_or_says_to_select_one():
    """A robot, an obstacle and an actor each frame; nothing selected is a message."""
    world = edit.add_obstacle(edit.new_world(), "box", (2.0, 2.0, 0.0), (0.5, 0.5))
    world = edit.add_actor(world, "w", (7.0, 4.0, 0.0))
    window, program = start(world)
    scene = program.session.scene
    program.command(KEY["f"], False)
    assert scene.camera.zoom == 1.0
    assert "Select a robot" in window.layout.chrome.status.message
    scene.selected = "r1"
    zoom_line = next(
        m for m in program.menu_items("View") if m.text == "Zoom to selection"
    )
    assert zoom_line.enabled
    program.command(KEY["f"], False)
    robot_zoom = scene.camera.zoom
    assert robot_zoom > 1.5
    scene.selected, scene.selected_obstacle = None, 0
    program.command(KEY["f"], False)
    assert scene.camera.zoom > 1.5
    scene.selected_obstacle, scene.selected_actor = None, 0
    program.command(KEY["f"], False)
    assert scene.camera.zoom > 1.5


def test_a_zoom_survives_an_edit_and_a_world_of_another_size_is_shown_whole():
    """Moving a robot keeps the zoom; a new world starts from the whole board."""
    window, program = start()
    scene = program.session.scene
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    window.mouse_scroll(2.0)
    zoomed = scene.camera.zoom
    program.session.move_robot("r1", (3.0, 2.0, 0.0))
    assert scene.camera.zoom == pytest.approx(zoomed)
    program.session.edits.change_world(lambda w: edit.set_size(w, (12.0, 8.0)))
    assert scene.camera.zoom == 1.0


def test_a_moved_view_keeps_its_scale_and_its_middle_when_the_window_changes_size():
    """What the user looks at stays as large, and its middle stays in the middle."""
    window, program = start()
    scene = program.session.scene
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    window.mouse_scroll(3.0)
    scene.pan_by(40, -25)
    window.render()
    before = scene.view
    middle = before.to_world(*Camera.centre(before.rect, 60.0))
    window.render(1000, 700)
    after = scene.view
    assert after.rect != before.rect and after.scale == pytest.approx(before.scale)
    assert after.to_world(*Camera.centre(after.rect, 60.0)) == pytest.approx(middle)


def test_a_view_that_was_not_moved_is_fitted_again_when_the_window_changes_size():
    """The whole board stays in view, and the zoom stays at 100 percent."""
    window, program = start()
    scene = program.session.scene
    window.render(1000, 700)
    assert scene.camera.zoom == 1.0 and not scene.camera.moved
    left, top, right, bottom = scene.view.board()
    assert left > 0 and top > 0 and right < 1000 and bottom < 700


def test_the_camera_holds_a_scale_and_a_world_point_at_the_middle():
    """The shift that puts a point in the middle at a chosen scale."""
    base, camera = base_view(), Camera()
    assert not camera.moved
    camera.hold(base, ROOM, 2 * base.scale, (3.0, 4.0))
    view = camera.view(base, ROOM)
    assert camera.moved and view.scale == pytest.approx(2 * base.scale)
    sx, sy = view.to_screen(3.0, 4.0)
    assert (float(sx), float(sy)) == pytest.approx(Camera.centre(RECT, ROOM))


def test_a_view_nobody_touched_is_not_shifted_by_a_window_that_hides_the_board():
    """Only the picture is held in sight; the camera is not marked as moved."""
    base = View((0.0, 0.0, 10.0, 6.0)).fit((0.0, 0.0, 100.0, 100.0), 60.0)
    camera = Camera()
    camera.view(base, 60.0)
    assert not camera.moved


def test_edits_keep_the_zoom_and_the_place_you_look_at():
    """A goal, another robot, a rename, a setting, undo and redo keep the view."""
    world = edit.add_obstacle(edit.new_world(), "box", (2.0, 2.0, 0.0), (0.5, 0.5))
    window, program = start(world)
    session = program.session
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    window.mouse_scroll(4.0)
    session.scene.pan_by(30, 12)
    window.render()
    seen = (session.scene.view, session.scene.camera.zoom)

    def kept():
        window.render()
        assert (session.scene.view, session.scene.camera.zoom) == seen

    assert session.set_goal("r1", (5.0, 2.0, 0.0)) is None
    kept()
    assert session.set_settings("r1", "controller", {"lookahead": 0.7}) is None
    kept()
    program._add_robot()
    kept()
    program._add_actor()
    kept()
    assert session.edits.rename_robot("r1", "robo") is None
    kept()
    program._undo()
    kept()
    program._redo()
    kept()
    program.reset_run()
    kept()


def test_a_rebuilt_scene_of_another_size_is_shown_whole():
    """The zoom is kept only while the world is the same size."""
    window, program = start()
    session = program.session
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    window.mouse_scroll(3.0)
    bigger = edit.set_size(session.world, (14.0, 9.0))
    assert session.apply(bigger, rebuild=True) is None
    window.render()
    assert session.scene.camera.zoom == 1.0


def test_the_scene_is_drawn_roughly_while_the_view_moves_and_in_full_after():
    """Half size and doubled while zooming, drawn again in full once it has settled."""
    window, program = start()
    scene = program.session.scene
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    window.mouse_scroll(1.0)
    assert scene.rough and program.busy()
    window.render()
    rough = scene._cutout.pixels.copy()
    assert scene.settle(0.0) is False  # not yet still long enough
    program.after_tick(scene._moved_at + 1.0)  # the loop's tick, a second later
    assert not scene.rough and not program.busy()
    window.render()
    sharp = scene._cutout.pixels
    assert sharp.shape == rough.shape and not np.array_equal(sharp, rough)
    assert scene.settle(scene._moved_at + 5.0) is False  # nothing left to end


def test_a_jump_to_the_whole_world_or_the_selection_is_drawn_in_full_at_once():
    """Only a gesture is drawn roughly; a key that jumps is not."""
    window, program = start()
    scene = program.session.scene
    scene.selected = "r1"
    program.command(KEY["f"], False)
    window.render()
    assert not scene.rough
    program.command(KEY["0"], False)
    assert not scene.rough


# -- robots and actors stay readable when the world is zoomed out --

SMALL_BODIES = """
version = 1
[world]
size = [60.0, 40.0]
seed = 1
[[robot]]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [10.0, 10.0, 0.0]
[[robot]]
name = "r2"
model = "diff_drive"
shape = { name = "box", size = [0.5, 0.3] }
pose = [20.0, 10.0, 0.0]
[[actor]]
name = "a1"
shape = { name = "circle", radius = 0.25 }
pose = [10.0, 20.0, 0.0]
[[actor]]
name = "a2"
shape = { name = "box", size = [0.6, 0.4] }
pose = [20.0, 20.0, 0.0]
"""


def small_bodies(tmp_path):
    """A big world with robots and actors that would be a few pixels across."""
    path = tmp_path / "small.toml"
    path.write_text(SMALL_BODIES)
    window, program = start(load_world(path))
    return window, program, program.session.scene


def test_a_robot_keeps_a_minimum_size_on_the_screen_and_a_big_one_is_not_changed():
    """The drawing scale is raised for a tiny robot and left alone when it is large."""
    foot = Footprint(True, None, 0.2, 0.2)
    assert display_scale(foot, 5.0) == pytest.approx(MIN_RADIUS / 0.2)
    assert display_scale(foot, 500.0) == 500.0


def pixel_distance_to_ink(window, scene, point, colour_of):
    """How far, in pixels, from a world point the nearest pixel of a colour lies."""
    frame = window.render()
    cx, cy = (int(round(float(v))) for v in scene.view.to_screen(*point))
    near = frame[cy - 30 : cy + 30, cx - 30 : cx + 30].astype(int)
    rows, cols = np.nonzero(np.abs(near - colour_of).sum(axis=2) < 40)
    return float(np.hypot(rows - 30, cols - 30).max()) if len(rows) else 0.0


def test_zoomed_out_a_robot_and_an_actor_are_still_drawn_at_a_readable_size(tmp_path):
    """The disc robot reaches the minimum radius, and the actor does too."""
    window, program, scene = small_bodies(tmp_path)
    program.fit_world()
    window.render()
    assert scene.view.scale < 20  # a body of 0.2 metres is under 4 pixels
    teal = np.array([round(v * 255) for v in rgba(look.robot_color(0))[:3]])
    robot_reach = pixel_distance_to_ink(window, scene, (10.0, 10.0), teal)
    assert robot_reach >= MIN_RADIUS - 2
    amber = np.array([round(v * 255) for v in rgba(look.AMBER)[:3]])
    actor_reach = pixel_distance_to_ink(window, scene, (10.0, 20.0), amber)
    assert actor_reach >= MIN_ACTOR_RADIUS - 2


def test_a_robot_or_actor_drawn_larger_than_it_is_can_be_clicked_where_it_is_drawn(
    tmp_path,
):
    """The click reaches the symbol, not only the true body; far off it misses."""
    window, program, scene = small_bodies(tmp_path)
    program.fit_world()
    window.render()
    rx, ry = (float(v) for v in scene.view.to_screen(10.0, 10.0))
    assert scene.robot_at(rx + 8, ry) == "r1" and scene.robot_at(rx + 40, ry) is None
    ax, ay = (float(v) for v in scene.view.to_screen(10.0, 20.0))
    assert scene.actor_at(ax + 6, ay) == 0 and scene.actor_at(ax + 30, ay) is None
    bx, by = (float(v) for v in scene.view.to_screen(20.0, 20.0))
    assert scene.actor_at(bx + 6, by) == 1


def test_a_box_actor_grows_about_its_middle_and_its_selection_ring_follows(tmp_path):
    """The corners are scaled about the centre, and the box round it has room."""
    window, program, scene = small_bodies(tmp_path)
    program.fit_world()
    grow = scene._grow(scene.world.actors[1].shape)
    assert grow > 1.0
    ring = np.array([[10.0, 10.0], [14.0, 10.0], [14.0, 12.0]])
    assert scene._grown(ring, (0.0, 0.0), 1.0) is ring
    centre = np.array([float(v) for v in scene.view.to_screen(20.0, 20.0)])
    corners = centre + np.array([[3.0, 2.0], [-3.0, 2.0]])
    assert scene._grown(corners, (20.0, 20.0), 2.0) == pytest.approx(
        centre + (corners - centre) * 2
    )
    scene.selected_actor = 1
    window.render()  # draws the ring round the grown box
    left, top, right, bottom = scene.actor_boxes()["a2"]
    assert right - left > 2 * MIN_ACTOR_RADIUS and bottom - top > 2 * MIN_ACTOR_RADIUS
