"""Tests for drawing a world: the cached layers, trails, rays and goals."""

# The tests look at the cache the scene keeps, which is private to it.
# pylint: disable=protected-access

from pathlib import Path

import numpy as np
import pytest

from demuvo import Run
from demuvo.gui import look, scenedraw
from demuvo.gui.raster import Canvas, rgba
from demuvo.gui.scenedraw import SceneDraw, arrow

from .fakes import ink

EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "worlds"

# A world with every kind of obstacle, two robots and a goal for each.
WORLD = """
version = 1
[world]
seed = 4
size = [10.0, 6.0]
[[obstacle]]
pose = [3.0, 3.0, 0.0]
shape = { name = "box", size = [1.0, 1.0] }
[[obstacle]]
pose = [5.0, 1.5, 0.0]
shape = { name = "circle", radius = 0.5 }
[[obstacle]]
shape = { name = "polygon", vertices = [[6, 4], [7, 4], [7, 5], [6.4, 4.4], [6, 5]] }
[[obstacle]]
shape = { name = "polygon", vertices = [[8.0, 1.0], [9.0, 1.0], [8.5, 2.0]] }
[[obstacle]]
shape = { name = "linestring", vertices = [[1.0, 5.0], [2.5, 5.5]] }
[[robot]]
pose = [1.0, 1.0, 0.0]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
planner = "demuvo.planners:AStar"
controller = "demuvo.controllers:PurePursuit"
[[robot]]
pose = [1.0, 3.0, 0.0]
name = "r2"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
[[robot]]
pose = [9.0, 3.0, 0.0]
name = "r3"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
[[task]]
robot = "r1"
goal = [9.0, 5.0, 1.0]
[[task]]
robot = "r2"
goal = [4.0, 3.5, 0.0]
"""


@pytest.fixture(name="run")
def world_run(tmp_path):
    """A run of the world above."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD)
    return Run(path)


@pytest.fixture(name="scene")
def fitted(run):
    """A scene fitted into a 700 by 460 area at (20, 30)."""
    scene = SceneDraw(run)
    scene.fit((20, 30, 700, 460), bottom_room=40)
    return scene


def paint(scene, with_overlays=False):
    """The scene drawn on a canvas the size of the window."""
    canvas = Canvas(760, 520, look.WINDOW)
    scene.draw_cutout(canvas)
    if with_overlays:
        left, top, right, bottom = (round(v) for v in scene.view.board())
        canvas.set_clip(left, top, right, bottom)
        scene.draw_overlays(canvas)
        canvas.set_clip(0, 0, 760, 520)
    return canvas


def at(scene, x, y):
    """The pixel of a world point."""
    sx, sy = scene.view.to_screen(x, y)
    return round(float(sx)), round(float(sy))


def test_the_floor_is_a_chess_board_on_the_window_colour(scene):
    """Neighbouring metres have the two floor colours; outside is the window."""
    canvas = paint(scene)
    a = canvas.px[at(scene, 0.5, 0.5)[::-1]]
    b = canvas.px[at(scene, 1.5, 0.5)[::-1]]
    assert a.tolist() != b.tolist()
    outside = canvas.px[40, 40]
    assert outside.tolist() == pytest.approx(list(rgba(look.WINDOW)[:3]), abs=0.01)


def test_a_layer_fades_out_and_back_in_through_its_opacity(scene):
    """With the floor off the board shows the window colour."""
    scene.layers.shown["floor"] = 0.0
    scene.layers.shown["costmap"] = 0.0
    canvas = paint(scene)
    pixel = canvas.px[at(scene, 0.5, 0.5)[::-1]]
    assert pixel.tolist() == pytest.approx(list(rgba(look.WINDOW)[:3]), abs=0.01)
    scene.layers.opacity["obstacles"] = 0.5
    assert scene.layers.alpha("obstacles") == 0.5


def test_obstacles_of_every_kind_are_drawn_in_navy(scene):
    """A box, a disc, a concave polygon, a triangle and a wall are drawn in navy."""
    canvas = paint(scene)
    for x, y in ((3.0, 3.0), (5.0, 1.5), (6.5, 4.15), (8.5, 1.4), (1.7, 5.25)):
        red, _, blue = canvas.px[at(scene, x, y)[::-1]]
        assert red < 0.3 and blue < 0.4, (x, y)


def test_the_board_has_a_navy_outline_and_the_cut_out_is_clipped_to_the_board(scene):
    """The outline is dark; the planned path never leaves the board."""
    canvas = paint(scene)
    left, top, _, _ = scene.view.board()
    assert canvas.px[round(top) - 1, round(left) + 40, 0] < 0.4
    assert canvas.px[round(top) - 8, round(left) + 40, 0] > 0.8


def test_the_global_costmap_darkens_the_floor_near_an_obstacle(scene):
    """The same place looks different with the costmap layer than without it."""
    near = at(scene, 3.0 + 0.5 + 0.1, 3.0)
    with_costmap = paint(scene).px[near[::-1]].copy()
    scene.layers.shown["costmap"] = 0.0
    without = paint(scene).px[near[::-1]].copy()
    assert with_costmap.tolist() != pytest.approx(without.tolist(), abs=0.05)


def test_a_world_with_no_costmap_draws_without_one(tmp_path):
    """The layer has nothing to show and nothing breaks."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD + "\n[costmap.global]\nenabled = false\n")
    scene = SceneDraw(Run(path))
    scene.fit((20, 30, 700, 460), bottom_room=40)
    assert ink(paint(scene)) > 0


def test_each_goal_is_an_arrow_in_its_robots_colour_inside_a_ring(scene):
    """The marker at the goal is in the robot's colour, and r2's differs from r1's."""
    canvas = paint(scene)
    gx, gy = at(scene, 9.0, 5.0)
    patch = canvas.px[gy - 30 : gy + 30, gx - 30 : gx + 30].reshape(-1, 3)
    teal = np.array(rgba(scene.colors["r1"])[:3])
    assert np.abs(patch - teal).sum(axis=1).min() < 0.15
    assert scene.colors["r1"] != scene.colors["r2"]


def test_a_dashed_line_leads_from_each_robot_towards_its_goal(scene):
    """With no path yet it is straight from the start to the goal, in dashes."""
    canvas = paint(scene)
    (x0, y0), (x1, y1) = at(scene, 1.0, 3.0), at(scene, 4.0, 3.5)
    color = np.array(rgba(scene.colors["r2"])[:3])
    along = [
        canvas.px[round(y0 + (y1 - y0) * t), round(x0 + (x1 - x0) * t)]
        for t in np.linspace(0.15, 0.85, 60)
    ]
    near = sum(np.abs(p - color).sum() < 0.3 for p in along)
    assert 10 < near < 55


def test_a_planned_path_replaces_the_straight_line(scene, run, monkeypatch):
    """After the planner runs the paths change and the cut-out goes stale."""
    monkeypatch.setattr(scenedraw, "PATH_CHECK", 0.0)
    _, before = scene.cutout_key()
    run.step(5)
    _, after = scene.cutout_key()
    assert before != after


def test_the_cut_out_is_drawn_once_and_again_only_when_something_changes(
    scene, monkeypatch
):
    """Steady frames reuse it; a new palette, a fade or a new fit redraws it."""
    calls = []
    original = scene._draw_cutout
    monkeypatch.setattr(
        scene, "_draw_cutout", lambda a, k: (calls.append(1), original(a, k))
    )
    paint(scene)
    paint(scene)
    assert len(calls) == 1
    scene.palette = look.PALETTES[1]
    paint(scene)
    scene.layers.shown["goals"] = 0.4
    paint(scene)
    scene.fit((20, 30, 600, 400), bottom_room=40)
    paint(scene)
    assert len(calls) == 4


def test_a_scene_too_small_to_draw_is_skipped(scene):
    """A rectangle under two pixels wide draws nothing and does not fail."""
    scene.fit((0, 0, 1, 1), bottom_room=0)
    canvas = Canvas(20, 20, "#ff0000")
    scene.draw_cutout(canvas)
    assert canvas.px[10, 10].tolist() == [1.0, 0.0, 0.0]


def test_the_floor_off_the_canvas_is_skipped(scene):
    """A view whose board lies beyond a small canvas draws nothing."""
    canvas = Canvas(10, 10, "#ff0000")
    scene._floor(canvas, scene.view.shifted(5000, 5000), 1.0)
    assert canvas.px[5, 5].tolist() == [1.0, 0.0, 0.0]


def test_trails_follow_the_robots_and_can_be_cleared(scene, run):
    """After steps the trail is a coloured line; clearing keeps one point."""
    for _ in range(20):
        run.step(10)
        scene.record_trails()
    assert len(scene.trails["r1"]) > 10
    canvas = paint(scene, with_overlays=True)
    x0, y0 = at(scene, 1.0, 1.0)
    x1, y1 = at(scene, *run.poses["r1"][:2])
    mid = canvas.px[
        (y0 + y1) // 2 - 2 : (y0 + y1) // 2 + 3, (x0 + x1) // 2 - 3 : (x0 + x1) // 2 + 4
    ]
    assert mid.min() < 0.9
    scene.reset_history()
    assert len(scene.trails["r1"]) == 1


def test_a_dragged_robot_is_drawn_where_it_is_dragged(scene):
    """The pose comes from the drag while it lasts, then from the run."""
    assert scene.pose("r1").tolist() == pytest.approx([1.0, 1.0, 0.0])
    scene.drag("r1", 4.0, 2.0, 0.5)
    assert scene.pose("r1").tolist() == [4.0, 2.0, 0.5]
    assert scene.pose("r2").tolist() == pytest.approx([1.0, 3.0, 0.0])
    scene.drop()
    assert scene.pose("r1").tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_a_scene_can_draw_another_run(scene):
    """Replacing the run empties the caches and takes the new world."""
    other = Run(EXAMPLES / "demo.toml")
    scene.set_run(other)
    assert set(scene.trails) == {"r1"} and scene.world is other.world
    scene.fit((20, 30, 700, 460), bottom_room=40)
    assert ink(paint(scene)) > 0


def lidar_scene():
    """The lidar example, after four seconds of running."""
    run = Run(EXAMPLES / "lidar.toml")
    scene = SceneDraw(run)
    scene.fit((20, 30, 700, 460), bottom_room=40)
    run.step(400)
    scene.record_trails()
    return scene


def test_lidar_rays_are_drawn_in_the_robots_colour_when_the_layer_is_on():
    """Off at first; on, the overlays carry ink from the robot outwards."""
    scene = lidar_scene()
    base = ink(paint(scene, with_overlays=True))
    scene.layers.shown["rays"] = 1.0
    assert ink(paint(scene, with_overlays=True)) > base + 50


def test_the_local_costmap_is_drawn_when_the_layer_is_on_and_kept_between_frames():
    """A picture is made once per costmap and palette, and again if one changes."""
    scene = lidar_scene()
    base = paint(scene, with_overlays=True).px.copy()
    scene.layers.shown["local"] = 1.0
    with_local = paint(scene, with_overlays=True).px
    assert np.abs(with_local - base).sum() > 20
    kept = scene._local["r1"]
    paint(scene, with_overlays=True)
    assert scene._local["r1"] is kept
    scene.palette = look.PALETTES[1]
    paint(scene, with_overlays=True)
    assert scene._local["r1"] is not kept


def test_a_robot_with_no_local_costmap_adds_nothing_to_that_layer(scene):
    """The world has no lidar, so there is nothing to draw."""
    scene.layers.shown["local"] = 1.0
    scene.layers.shown["rays"] = 1.0
    assert paint(scene, with_overlays=True) is not None
    assert scene._local_picture("r1") is None


def test_an_arrow_points_along_its_heading_and_has_an_outline():
    """Heading 0 puts the head on the right, 90 degrees up; the outline is white."""
    right, up = Canvas(80, 80, "#000000"), Canvas(80, 80, "#000000")
    arrow(right, (40, 40), 0.0, 30, rgba("#ff0000"), outline=rgba("#ffffff"))
    arrow(up, (40, 40), -np.pi / 2, 30, rgba("#ff0000"))
    assert right.px[40, 66, 0] > 0.5 and right.px[40, 10, 0] == 0.0
    assert up.px[14, 40, 0] > 0.5 and up.px[66, 40, 0] == 0.0
    assert right.px[..., 1].max() > 0.2 and up.px[..., 1].max() == 0.0


def test_the_layer_table_lists_the_layers_bottom_first():
    """Robots are on top; the floor is below everything."""
    names = list(scenedraw.LAYERS)
    assert names[0] == "floor" and names[-1] == "robots"
    assert scenedraw.LAYERS["rays"][1] is False and scenedraw.LAYERS["floor"][1] is True


def test_the_obstacle_under_a_pixel_is_found_by_its_real_shape(scene):
    """Box, circle and polygon hit inside; a notch, a line and the floor do not."""
    assert scene.obstacle_at(*at(scene, 3.2, 3.2)) == 0
    assert scene.obstacle_at(*at(scene, 5.3, 1.5)) == 1
    assert scene.obstacle_at(*at(scene, 5.6, 1.5)) is None
    assert scene.obstacle_at(*at(scene, 6.8, 4.1)) == 2
    assert scene.obstacle_at(*at(scene, 6.5, 4.9)) is None
    assert scene.obstacle_at(*at(scene, 1.75, 5.25)) is None
    assert scene.obstacle_at(*at(scene, 0.5, 0.5)) is None


def test_the_selected_obstacle_gets_an_outline(scene):
    """A box, a circle and a polygon each show ink round them; none changes nothing."""
    plain = paint(scene)
    for index in (0, 1, 3):
        scene.selected_obstacle = index
        canvas = paint(scene)
        scene._draw_selected_obstacle(canvas)
        assert not np.array_equal(canvas.px, plain.px)
    for index in (None, 99):
        scene.selected_obstacle = index
        canvas = paint(scene)
        scene._draw_selected_obstacle(canvas)
        assert np.array_equal(canvas.px, plain.px)
