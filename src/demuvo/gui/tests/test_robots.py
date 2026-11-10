"""Tests for drawing robots: footprints, sprites, labels, hits and dirty rectangles."""

# The tests look at the caches the scene keeps, which are private to it.
# pylint: disable=protected-access

import numpy as np
import pytest

from demuvo import Run, edit
from demuvo.gui import look, robots
from demuvo.gui.raster import Canvas, rgba
from demuvo.gui.scenedraw import SceneDraw
from demuvo.world.shapes import Box, Circle, Polygon

from .fakes import ink

# One robot of each kind of body: a disc, a box, a convex and a concave polygon.
WORLD = """
version = 1
[world]
seed = 6
size = [10.0, 6.0]
[[robot]]
pose = [2.0, 1.5, 0.0]
name = "disc"
model = "diff_drive"
shape = { name = "circle", radius = 0.3 }
[[robot]]
pose = [5.0, 1.5, 1.0]
name = "box"
model = "diff_drive"
shape = { name = "box", size = [0.8, 0.4] }
[[robot]]
pose = [8.0, 1.5, 0.0]
name = "hex"
model = "diff_drive"
shape = { name = "polygon", vertices = [
  [0.3, 0], [0.15, 0.26], [-0.15, 0.26],
  [-0.3, 0], [-0.15, -0.26], [0.15, -0.26]] }
[[robot]]
pose = [3.0, 4.5, 0.5]
name = "ell"
model = "diff_drive"
shape = { name = "polygon", vertices = [
  [-0.4, -0.4], [0.4, -0.4], [0.4, 0.0],
  [0.0, 0.0], [0.0, 0.4], [-0.4, 0.4]] }
"""


@pytest.fixture(name="scene")
def kinds(tmp_path):
    """A scene fitted to a 700 by 460 area with a robot of each kind of body."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD)
    scene = SceneDraw(Run(path))
    scene.fit((20, 30, 700, 460), bottom_room=40)
    return scene


def frame(scene, opacity=1.0, clip=None):
    """The scene's robots drawn on a window-coloured canvas."""
    canvas = Canvas(760, 520, look.WINDOW)
    if clip:
        canvas.set_clip(*clip)
    robots.draw_robots(scene, canvas, opacity)
    return canvas


def at(scene, name):
    """The pixel of a robot's origin."""
    x, y = scene.view.to_screen(*scene.pose(name)[:2])
    return round(float(x)), round(float(y))


def test_the_footprint_of_a_disc_is_its_radius_both_ways():
    """Inscribed and bound radii are the same, and there is no outline."""
    foot = robots.footprint(Circle(0.3))
    assert foot.circular and foot.outline is None
    assert foot.inscribed == 0.3 and foot.bound == 0.3


def test_the_footprint_of_a_box_has_the_corners_and_two_radii():
    """A 0.8 by 0.4 box: half the short side inside, the half diagonal outside."""
    foot = robots.footprint(Box((0.8, 0.4)))
    assert not foot.circular and foot.outline.shape == (4, 2)
    assert foot.inscribed == pytest.approx(0.2)
    assert foot.bound == pytest.approx(np.hypot(0.4, 0.2))


def test_a_body_that_does_not_surround_its_origin_still_gets_a_size_for_the_arrow():
    """The nearest corner coordinate stands in; with a corner on an axis, 0.1."""
    away = robots.footprint(Polygon(((0.2, 0.2), (0.6, 0.2), (0.4, 0.6))))
    assert away.inscribed == pytest.approx(0.2)
    on_axis = robots.footprint(Polygon(((0.0, 0.5), (0.5, 0.5), (0.5, 1.0))))
    assert on_axis.inscribed == 0.1


def test_a_disc_is_drawn_in_its_colour_with_a_white_arrow_and_a_label(scene):
    """The ring is the robot's colour, the arrow is white, and a label is above."""
    canvas = frame(scene)
    x, y = at(scene, "disc")
    radius = 0.3 * scene.view.scale
    ring = canvas.px[y, round(x - radius + 3)]
    assert ring.tolist() == pytest.approx(
        list(rgba(scene.colors["disc"])[:3]), abs=0.05
    )
    assert canvas.px[y - 3 : y + 4, x - 3 : x + 8].mean() > ring.mean() + 0.05
    label = canvas.px[round(y - radius - 22) : round(y - radius - 2), x - 20 : x + 20]
    assert label.min() < 0.9


def test_the_arrow_turns_with_the_heading_and_headings_share_sprites(scene):
    """Two discs of one size at one heading make one arrow sprite."""
    scene.drag("disc", 2.0, 1.5, 0.0)
    right = frame(scene)
    scene.drag("disc", 2.0, 1.5, np.pi / 2)
    up = frame(scene)
    x, y = at(scene, "disc")
    ahead = (slice(y - 3, y + 4), slice(x + 4, x + 12))
    above = (slice(y - 12, y - 4), slice(x - 3, x + 4))
    assert right.px[ahead].mean() > up.px[ahead].mean() + 0.05
    assert up.px[above].mean() > right.px[above].mean() + 0.05
    arrows = [k for k in scene.sprites._sprites if k[0] == "arrow"]
    assert len(arrows) == 2


@pytest.mark.parametrize("name", ["box", "hex", "ell"])
def test_a_polygon_robot_is_drawn_with_its_outline_in_its_colour_and_an_arrow(
    scene, name
):
    """The body has the robot's colour and the arrow is white near the origin."""
    canvas = frame(scene)
    x, y = at(scene, name)
    patch = canvas.px[y - 25 : y + 25, x - 25 : x + 25].reshape(-1, 3)
    color = np.array(rgba(scene.colors[name])[:3])
    assert np.abs(patch - color).sum(axis=1).min() < 0.1
    assert canvas.px[y - 3 : y + 4, x - 3 : x + 8].mean() > color.mean() + 0.03


def test_a_box_robot_is_drawn_again_when_its_heading_changes(scene):
    """The body sprite is cached by heading; a new heading makes a new body."""
    frame(scene)
    before = len(scene.sprites)
    scene.drag("box", 5.0, 1.5, 1.0 + 0.5)
    frame(scene)
    assert len(scene.sprites) > before
    scene.drag("box", 5.0, 1.5, 1.0 + 0.01)
    kept = len(scene.sprites)
    frame(scene)
    assert len(scene.sprites) == kept


def test_the_selected_robot_has_a_halo_and_its_label_moves_up(scene):
    """A pale disc round it, larger than the robot, with a navy edge."""
    plain = frame(scene)
    scene.selected = "disc"
    lit = frame(scene)
    x, y = at(scene, "disc")
    reach = robots._halo_radius(scene.feet["disc"], scene.view.scale)
    assert reach > 0.3 * scene.view.scale
    assert not np.array_equal(plain.px, lit.px)
    edge = lit.px[y, round(x + reach) - 3 : round(x + reach) + 2, 0]
    assert edge.min() < 0.6


def test_robots_fade_with_the_layer_opacity(scene):
    """Half opacity leaves less ink than full."""
    assert ink(frame(scene, opacity=0.4)) < ink(frame(scene))


def test_a_crowd_has_no_labels_except_the_selected_robot():
    """Past 40 robots the names are left out to keep drawing cheap."""
    world = edit.new_world()
    for i in range(45):
        world = edit.add_robot(world, f"r{i + 2}", (0.5 + i % 9, 0.5 + i // 9, 0.0))
    scene = SceneDraw(Run(world))
    scene.fit((20, 30, 700, 460), bottom_room=40)
    frame(scene)
    assert not [k for k in scene.sprites._sprites if k[0] == "label"]
    scene.selected = "r10"
    frame(scene)
    assert [k[1] for k in scene.sprites._sprites if k[0] == "label"] == ["r10"]


def test_a_click_finds_the_robot_under_it_and_a_small_robot_is_easy_to_hit(scene):
    """On the body, near it, and nowhere."""
    x, y = at(scene, "box")
    assert scene.robot_at(x, y) == "box"
    assert scene.robot_at(x + 2, y + 2) == "box"
    assert scene.robot_at(5, 5) is None
    tiny = SceneDraw(Run(edit.set_pose(edit.new_world(), "r1", (5.0, 3.0, 0.0))))
    tiny.fit((20, 30, 700, 460), bottom_room=40)
    cx, cy = at(tiny, "r1")
    assert tiny.robot_at(cx + 11, cy) == "r1" and tiny.robot_at(cx + 40, cy) is None


def test_the_nearest_of_two_robots_that_both_reach_the_click_wins():
    """Two robots close together: the click goes to the nearer."""
    world = edit.add_robot(edit.new_world(), "r2", (5.5, 3.0, 0.0))
    scene = SceneDraw(Run(world))
    scene.fit((20, 30, 700, 460), bottom_room=40)
    x1, y1 = at(scene, "r1")
    x2, _ = at(scene, "r2")
    assert scene.robot_at(x1 + 3, y1) == "r1" and scene.robot_at(x2 - 3, y1) == "r2"


def full_scene(scene):
    """The whole scene drawn on a canvas."""
    canvas = Canvas(760, 520, look.WINDOW)
    scene.draw(canvas)
    return canvas


def settled(scene):
    """Draw once and take the dirty rectangles that follow, as the window does."""
    scene.dirty_rect()
    canvas = full_scene(scene)
    scene.dirty_rect()
    return canvas


def test_the_whole_scene_draws_robots_over_the_layers(scene):
    """The floor, then the robots; a hidden robots layer leaves them out."""
    shown = full_scene(scene)
    scene.layers.shown["robots"] = 0.0
    hidden = full_scene(scene)
    x, y = at(scene, "disc")
    assert shown.px[y, x].tolist() != hidden.px[y, x].tolist()
    assert ink(shown) > ink(hidden)


def test_a_scene_asked_to_draw_outside_its_rectangle_draws_nothing(scene):
    """The clip does not meet the scene."""
    canvas = Canvas(760, 520, "#ff0000")
    canvas.set_clip(0, 0, 10, 10)
    scene.draw(canvas)
    assert canvas.px[5, 5].tolist() == [1.0, 0.0, 0.0]
    assert canvas.clip == (0, 0, 10, 10)


def test_the_first_dirty_rectangle_is_the_whole_scene_and_a_quiet_one_is_small(scene):
    """With nothing moving the box holds only the robots' own places."""
    x, y, width, height = scene.view.rect
    full = (int(x), int(y), int(x + width), int(y + height))
    assert scene.dirty_rect() == full
    full_scene(scene)
    quiet = scene.dirty_rect()
    assert quiet != full
    x0, y0, x1, y1 = quiet
    assert (x1 - x0) * (y1 - y0) <= 700 * 400


def test_the_dirty_rectangle_grows_to_hold_where_a_robot_was_and_is(scene):
    """A drag from one corner to another covers both."""
    settled(scene)
    scene.drag("disc", 2.0, 4.5, 0.0)
    moved = scene.dirty_rect()
    first = robots.robot_boxes(scene)["disc"]
    assert moved[1] <= first[1] and moved[3] >= first[3]
    old_top = at(scene, "box")[1] - 100
    assert moved[3] - moved[1] > 100 and old_top < moved[3]


def test_a_global_change_makes_the_dirty_rectangle_the_whole_scene(scene):
    """Rays or the local costmap showing, a new palette, or a new set of robots."""
    settled(scene)
    scene.layers.shown["rays"] = 1.0
    x, y, width, height = scene.view.rect
    full = (int(x), int(y), int(x + width), int(y + height))
    assert scene.dirty_rect() == full
    scene.layers.shown["rays"] = 0.0
    scene.dirty_rect()
    scene.palette = look.PALETTES[1]
    assert scene.dirty_rect() == full
    scene.dirty_rect()
    scene.layers.shown["local"] = 1.0
    assert scene.dirty_rect() == full
    scene.layers.shown["local"] = 0.0
    scene.dirty_rect()
    scene._last_boxes = {"other": (0, 0, 1, 1)}
    assert scene.dirty_rect() == full


def test_drawing_only_the_dirty_rectangle_gives_the_same_pixels_as_a_full_redraw(scene):
    """Redraw only the box of a moved robot, and compare with a full redraw."""
    canvas = settled(scene)
    scene.drag("box", 6.0, 2.5, 1.2)
    x0, y0, x1, y1 = scene.dirty_rect()
    canvas.set_clip(x0, y0, x1, y1)
    scene.draw(canvas)
    canvas.set_clip(0, 0, 760, 520)
    assert np.abs(canvas.px - full_scene(scene).px).max() < 1e-6


def test_a_selected_polygon_robot_has_the_halo_too(scene):
    """The halo is drawn behind a box as behind a disc."""
    plain = frame(scene)
    scene.selected = "box"
    lit = frame(scene)
    x, y = at(scene, "box")
    reach = robots._halo_radius(scene.feet["box"], scene.view.scale)
    assert not np.array_equal(plain.px, lit.px)
    assert lit.px[y, round(x + reach) - 3 : round(x + reach) + 2, 0].min() < 0.6
