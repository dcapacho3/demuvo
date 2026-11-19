"""Tests for zooming and moving the scene: camera, pane, keys and scale bar."""

import numpy as np
import pytest

from demuvo import edit
from demuvo.gui.keys import KEY
from demuvo.gui.raster import Canvas
from demuvo.gui.sandbox import ZOOM_KEY, Sandbox
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


def test_the_zoom_is_kept_when_the_window_is_resized():
    """The pane changes size and the zoom does not."""
    window, program = start()
    scene = program.session.scene
    x, y = scene_middle(program)
    window.mouse_move(x, y)
    window.mouse_scroll(2.0)
    zoomed = scene.camera.zoom
    window.render(1000, 700)
    assert scene.camera.zoom == pytest.approx(zoomed)
