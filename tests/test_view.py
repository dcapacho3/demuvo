"""Tests for the pygame view."""

from pathlib import Path

import numpy as np
import pygame
import pytest
from PIL import Image
from pygame.locals import (
    K_DOWN,
    K_ESCAPE,
    K_RIGHT,
    K_SPACE,
    K_UP,
    KEYDOWN,
    MOUSEMOTION,
    QUIT,
    K_a,
    K_s,
)

from demuvo import Box, Circle, Simulation, load_world
from demuvo.view import (
    HUD,
    OBSTACLE,
    ROBOT_COLORS,
    SPEEDS,
    View,
    _change_speed,
    outline,
    play,
    record,
)

DEMO = str(Path(__file__).resolve().parents[1] / "examples" / "worlds" / "demo.toml")


@pytest.fixture(autouse=True)
def fonts():
    """The view draws text, so pygame's font module must be ready."""
    pygame.font.init()


def pixel(view, frame, x, y):
    """The colour of the frame at the world point (x, y)."""
    px, py = view.to_pixel(x, y)
    return tuple(frame.get_at((round(px), round(py) + HUD)))[:3]


def close(colour, expected, tolerance=12):
    """True if two colours differ by less than `tolerance` on every channel."""
    expected = pygame.Color(expected)
    return all(abs(a - b) < tolerance for a, b in zip(colour, expected[:3]))


def test_a_box_outline_goes_around_the_box():
    """Consecutive corners are one side apart, never a diagonal."""
    points = outline(Box((4.0, 2.0)), (0.0, 0.0, 0.0))
    sides = [np.hypot(*(points[i] - points[i - 1])) for i in range(4)]
    assert sorted(sides) == pytest.approx([2.0, 2.0, 4.0, 4.0])


def test_a_circle_outline_stays_at_the_radius():
    """Every point of the outline is one radius from the centre."""
    points = outline(Circle(0.5), (1.0, 2.0, 0.3))
    assert np.hypot(points[:, 0] - 1.0, points[:, 1] - 2.0) == pytest.approx(0.5)


def test_the_frame_has_the_size_the_view_reports():
    """The picture is as wide as the view says and has the top bar above it."""
    view = View(Simulation(load_world(DEMO)))
    assert view.render().get_size() == view.size
    assert view.size[1] > HUD


def test_the_floor_has_no_gaps():
    """Every pixel of the world is one of the two floor shades or a thing on it."""
    view = View(Simulation(load_world(DEMO)))
    frame = view.render()
    x0, y0 = view.to_pixel(0.0, 6.0)
    x1, y1 = view.to_pixel(10.0, 0.0)
    colours = {
        tuple(frame.get_at((x, y + HUD)))[:3]
        for x in range(round(x0) + 6, round(x1) - 6)
        for y in range(round(y0) + 6, round(y1) - 6)
    }
    outside = tuple(pygame.Color("#90a4ae"))[:3]
    assert outside not in colours


def test_things_are_drawn_where_they_are_in_the_world():
    """The circle obstacle and the robot appear at their metres, with y up."""
    sim = Simulation(load_world(DEMO))
    view = View(sim)
    frame = view.render()
    assert close(pixel(view, frame, 7.0, 2.0), OBSTACLE)
    # 0.1 m below the robot centre; the heading line points right, not down.
    assert close(pixel(view, frame, 1.0, 0.9), ROBOT_COLORS[0])
    # Above the circle obstacle, a floor tile of one of the two shades.
    assert pixel(view, frame, 7.5, 4.5) in {(236, 239, 241), (221, 226, 230)}


def test_the_robot_moves_in_the_picture():
    """After a step the frame changes and the trail grows."""
    sim = Simulation(load_world(DEMO), dt=1.0)
    view = View(sim)
    before = pygame.image.tobytes(view.render(), "RGB")
    view.step(lambda _: {"r1": (1.0, 0.0)})
    assert pygame.image.tobytes(view.render(), "RGB") != before
    assert list(view.trails["r1"]) == pytest.approx([(1.0, 1.0), (2.0, 1.0)])
    assert close(pixel(view, view.render(), 2.0, 0.9), ROBOT_COLORS[0])


def test_the_trail_keeps_only_the_last_positions():
    """A trail longer than the limit drops its oldest points."""
    sim = Simulation(load_world(DEMO))
    view = View(sim, trail=3)
    for _ in range(10):
        view.step(lambda _: {"r1": (0.5, 0.0)})
    assert len(view.trails["r1"]) == 3


def test_the_bar_shows_the_state():
    """Paused and done change the top bar."""
    view = View(Simulation(load_world(DEMO)))
    plain = pygame.image.tobytes(view.render(), "RGB")
    view.paused = True
    paused = pygame.image.tobytes(view.render(), "RGB")
    view.done = True
    done = pygame.image.tobytes(view.render(), "RGB")
    assert len({plain, paused, done}) == 3


def test_a_world_without_goals_or_trails_still_draws(tmp_path):
    """A robot with no goal has no star."""
    lines = Path(DEMO).read_text(encoding="utf-8").splitlines()
    text = "\n".join(line for line in lines if not line.startswith("goal"))
    path = tmp_path / "w.toml"
    path.write_text(text)
    view = View(Simulation(load_world(path)))
    frame = view.render()
    assert not close(pixel(view, frame, 9.0, 5.0), ROBOT_COLORS[0], 40)


def test_record_saves_a_gif(tmp_path):
    """The GIF has the first frame and one more for each step."""
    path = tmp_path / "run.gif"
    sim = Simulation(load_world(DEMO))
    record(sim, lambda _: {"r1": (0.5, 0.5)}, steps=5, path=str(path))
    assert sim.steps == 5
    with Image.open(path) as gif:
        assert gif.n_frames == 6


def test_a_gif_frame_is_every_skip_steps(tmp_path):
    """At 10 frames a second and dt = 0.05, a frame comes every 2 steps."""
    path = tmp_path / "run.gif"
    sim = Simulation(load_world(DEMO))
    record(sim, lambda _: {"r1": (0.5, 0.5)}, steps=6, path=str(path), fps=10)
    with Image.open(path) as gif:
        assert gif.n_frames == 4


def test_the_speed_steps_through_the_list():
    """Up and down move one place, and stop at the ends."""
    assert _change_speed(1.0, up=True) == 2.0
    assert _change_speed(1.0, up=False) == 0.5
    assert _change_speed(SPEEDS[-1], up=True) == SPEEDS[-1]
    assert _change_speed(SPEEDS[0], up=False) == SPEEDS[0]


@pytest.fixture(name="window")
def run_window(monkeypatch):
    """Run the window with no display; the events come from a list.

    Call the returned function with the events of each frame. After the last
    frame the window gets a QUIT, so the loop always ends.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    def run(frames, steps=None):
        queue = list(frames)
        monkeypatch.setattr(
            pygame.event,
            "get",
            lambda: queue.pop(0) if queue else [pygame.event.Event(QUIT)],
        )
        sim = Simulation(load_world(DEMO))
        play(sim, lambda _: {"r1": (0.5, 0.0)}, steps=steps, caption="a title")
        return sim

    return run


def key(code):
    """A key press."""
    return pygame.event.Event(KEYDOWN, key=code)


def test_the_window_runs_the_steps_then_waits(window):
    """The run stops at `steps` and the window stays until it is closed."""
    sim = window([[]] * 40, steps=3)
    assert sim.steps == 3


def test_the_window_runs_until_closed(window):
    """With no limit the simulation keeps stepping while the window is open."""
    sim = window([[]] * 10)
    assert sim.steps > 0


def test_space_pauses(window):
    """A paused window takes no steps."""
    assert window([[key(K_SPACE)]] + [[]] * 10).steps == 0


def test_space_again_resumes(window):
    """Two presses pause and resume."""
    frames = [[key(K_SPACE)], [key(K_SPACE)]] + [[]] * 40
    assert window(frames).steps > 0


def test_right_arrow_steps_once(window):
    """One press takes exactly one step and leaves the window paused."""
    assert window([[key(K_RIGHT)]] + [[]] * 10).steps == 1


def test_escape_closes_the_window(window):
    """Escape ends the loop at once."""
    assert window([[key(K_ESCAPE)]] + [[]] * 40).steps == 0


def test_the_speed_keys_do_not_stop_the_run(window):
    """Faster and slower keys are accepted, and other keys are ignored."""
    mouse = pygame.event.Event(MOUSEMOTION, pos=(0, 0), rel=(0, 0), buttons=(0, 0, 0))
    frames = [[key(K_UP)], [key(K_DOWN)], [key(K_a)], [mouse]]
    assert window(frames + [[]] * 20).steps > 0


def test_s_saves_a_screenshot(window, tmp_path, monkeypatch):
    """The S key writes a PNG named after the simulated time."""
    monkeypatch.chdir(tmp_path)
    window([[key(K_s)]])
    assert [p.name for p in tmp_path.glob("demuvo-*.png")] == ["demuvo-0.00s.png"]


def test_the_window_gets_the_caption(window, monkeypatch):
    """The caption passed to `play` becomes the window title."""
    captions = []
    monkeypatch.setattr(pygame.display, "set_caption", captions.append)
    window([[]])
    assert captions == ["a title"]
