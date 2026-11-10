"""Tests for the scene pane: select, drag, drop, and the goal tool."""

import pytest

from demuvo.gui.panels import Pill
from demuvo.gui.raster import Canvas
from demuvo.gui.scenepane import ScenePane
from demuvo.gui.session import Session
from demuvo.gui.widgets import Event

from .fakes import EXAMPLES, hosted, ink


class Actions:
    """Records what the pane tells its owner."""

    def __init__(self, session):
        """Keep the session, so a goal or a drop can be applied as the window does."""
        self.session, self.log = session, []

    def select(self, name):
        """Note a selection and make it."""
        self.log.append(("selected", name))
        self.session.select(name)

    def goal_set(self, name, x, y, heading):
        """Note a goal and set it."""
        self.log.append(("goal", name, round(x, 2), round(y, 2), round(heading, 2)))
        self.session.set_goal(name, (x, y, heading))

    def dropped(self, name, x, y, angle):
        """Note a drop and apply it."""
        self.log.append(("dropped", name, round(x, 2), round(y, 2)))
        self.session.move_robot(name, (x, y, angle))

    def dragged(self):
        """Note a drag."""
        self.log.append(("dragged",))


def build(session=None):
    """A pane over a blank sandbox, placed in a 700 by 500 area."""
    session = session or Session()
    actions = Actions(session)
    pill = Pill(print, print, print, print)
    pane = ScenePane(session, pill, actions)
    hosted(pane, (100, 60, 700, 500))
    return pane, session, actions


def at(pane, x, y):
    """The pixel of a world point."""
    sx, sy = pane.session.scene.view.to_screen(x, y)
    return float(sx), float(sy)


def test_the_pane_fits_the_board_and_puts_the_pill_at_the_bottom_centre():
    """The scene takes the pane's rectangle; the pill sits 16 pixels from the foot."""
    pane, session, _ = build()
    assert session.scene.view.rect == (100, 60, 700, 500)
    assert pane.pill.y + pane.pill.height == pytest.approx(60 + 500 - 16)
    assert pane.pill.x + pane.pill.width / 2 == pytest.approx(100 + 350)


def test_the_pane_draws_the_scene_and_the_pill():
    """Ink from the robot and the pill."""
    pane, _, _ = build()
    canvas = Canvas(820, 600)
    pane.draw(canvas)
    assert ink(canvas, (100, 60, 800, 540)) > 100
    assert ink(canvas, (300, 500, 700, 550)) > 20


def test_the_pill_takes_clicks_over_it_and_the_pane_the_rest():
    """A point on the pill hits its button; a point in the world hits the pane."""
    pane, _, _ = build()
    centre = (pane.pill.play.x + 10, pane.pill.play.y + 10)
    assert pane.hit(*centre) is pane.pill.play
    assert pane.hit(120, 100) is pane
    assert pane.hit(5, 5) is None
    pane.visible = False
    assert pane.hit(120, 100) is None


def test_a_click_on_a_robot_selects_and_drags_and_a_click_on_the_floor_clears():
    """Select r1; a click away from it selects none."""
    pane, session, actions = build()
    pane.on_press(Event(*at(pane, 5.0, 3.0)))
    assert session.scene.selected == "r1" and pane.dragging == "r1"
    pane.on_release(Event(*at(pane, 5.0, 3.0)))
    pane.on_press(Event(*at(pane, 1.0, 1.0)))
    assert session.scene.selected is None and pane.dragging is None
    assert ("selected", None) in actions.log


def test_dragging_a_robot_moves_it_and_dropping_applies_the_place():
    """Grab a little off its middle and drag: it keeps that offset on the drop."""
    pane, session, actions = build()
    gx, gy = at(pane, 5.1, 3.0)
    pane.on_press(Event(gx, gy))
    tx, ty = at(pane, 2.1, 2.0)
    pane.on_move(Event(tx, ty))
    assert tuple(session.scene.pose("r1")[:2]) == pytest.approx((2.0, 2.0), abs=0.02)
    assert ("dragged",) in actions.log
    pane.on_release(Event(tx, ty))
    assert actions.log[-1][:2] == ("dropped", "r1")
    assert tuple(session.run.poses["r1"][:2]) == pytest.approx((2.0, 2.0), abs=0.02)
    assert pane.dragging is None


def test_a_move_without_a_drag_and_a_release_without_one_do_nothing():
    """Hovering over the pane changes nothing."""
    pane, _, actions = build()
    pane.on_move(Event(*at(pane, 5.0, 3.0)))
    pane.on_release(Event(0, 0))
    assert not actions.log


def test_a_benchmark_world_can_be_selected_but_not_dragged():
    """Selecting works; no drag starts."""
    pane, session, _ = build(Session(EXAMPLES / "demo.toml", "benchmark"))
    x, y = session.scene.pose("r1")[:2]
    pane.on_press(Event(*at(pane, x, y)))
    assert session.scene.selected == "r1" and pane.dragging is None


def test_with_the_goal_tool_a_click_on_the_floor_sets_the_goal_towards_it():
    """The heading points from the robot to the click, and the goal is there."""
    pane, session, actions = build()
    session.select("r1")
    session.set_tool("goal")
    pane.on_press(Event(*at(pane, 8.0, 3.0)))
    kind, name, x, y, heading = actions.log[-1]
    assert (kind, name, x, y, heading) == ("goal", "r1", 8.0, 3.0, 0.0)
    assert pane.dragging is None


def test_the_goal_tool_on_a_robot_selects_as_usual_and_needs_a_selection():
    """A robot click still selects it; with none selected a floor click selects none."""
    pane, session, actions = build()
    session.set_tool("goal")
    pane.on_press(Event(*at(pane, 1.0, 1.0)))
    assert actions.log[-1] == ("selected", None)
    pane.on_press(Event(*at(pane, 5.0, 3.0)))
    assert session.scene.selected == "r1"


def test_a_run_in_progress_blocks_the_goal_tool_and_the_drag():
    """After a step the session is locked, so a click only selects."""
    pane, session, actions = build()
    session.select("r1")
    session.set_tool("goal")
    session.step_once()
    x, y = session.scene.pose("r1")[:2]
    pane.on_press(Event(*at(pane, 8.0, 5.0)))
    assert not any(item[0] == "goal" for item in actions.log)
    pane.on_press(Event(*at(pane, x, y)))
    assert pane.dragging is None
