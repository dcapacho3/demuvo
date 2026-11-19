"""Tests that a controller is told what its robot's model can follow."""

from pathlib import Path

import pytest

from demuvo import Navigation, Simulation, edit, load_world

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"

MODULE = """
from demuvo import Twist
seen = []
class Spy:
    def __init__(self, speed=0.5, lookahead=0.6):
        pass
    def command(self, obs):
        seen.append(obs)
        return Twist()
"""


def spy_world(tmp_path, monkeypatch, model):
    """The demo world on the given model, with a controller that records."""
    (tmp_path / "my_limits.py").write_text(MODULE)
    monkeypatch.syspath_prepend(tmp_path)
    import my_limits  # pylint: disable=import-error,import-outside-toplevel

    my_limits.seen.clear()  # the module stays loaded between tests
    text = DEMO.read_text().replace("demuvo.controllers:PurePursuit", "my_limits:Spy")
    path = tmp_path / "w.toml"
    path.write_text(text)
    return my_limits, edit.set_model(load_world(path), "r1", model)


def observe(world):
    """Run one controller step and return what the spy saw."""
    sim = Simulation(world)
    nav = Navigation(world)
    sim.step(nav(sim))


def test_a_diff_drive_controller_gets_flat_limits(tmp_path, monkeypatch):
    """The observation carries the model's limits, which turn in place."""
    spy, world = spy_world(tmp_path, monkeypatch, "diff_drive")
    observe(world)
    limits = spy.seen[-1].limits
    assert limits is not None
    assert limits.turns_in_place
    assert limits.max_v == 1.0
    assert limits.max_w(0.0) == 1.0


def test_an_ackermann_controller_gets_the_wedge(tmp_path, monkeypatch):
    """The observation says an ackermann robot cannot turn in place."""
    spy, world = spy_world(tmp_path, monkeypatch, "ackermann")
    observe(world)
    limits = spy.seen[-1].limits
    assert limits is not None
    assert not limits.turns_in_place
    assert limits.max_w(0.0) == 0.0
    assert limits.max_w(1.0) > 0.0


GOALS = {
    "left": (1.0, 4.0, 1.57),
    "behind": (0.3, 1.0, 3.14),
    "ahead": (5.0, 1.0, 0.0),
}


def drive_to(world, goal, seconds=15.0):
    """Run the world's navigation toward a goal; the time it got within 0.1 m."""
    import dataclasses  # pylint: disable=import-outside-toplevel

    from robomath import Transform2D  # pylint: disable=import-outside-toplevel

    from demuvo.world import Task  # pylint: disable=import-outside-toplevel

    task = Task("r1", Transform2D.from_xyt(goal), (0.1, 3.2))
    world = dataclasses.replace(world, tasks=(task,))
    sim, nav = Simulation(world), Navigation(world)
    while sim.time < seconds:
        sim.step(nav(sim))
        x, y, _ = sim.poses["r1"].xyt
        if ((x - goal[0]) ** 2 + (y - goal[1]) ** 2) ** 0.5 <= 0.1:
            return sim.time
    return None


@pytest.mark.parametrize(
    "controller", ["demuvo.controllers:PurePursuit", "demuvo.controllers:DWB"]
)
@pytest.mark.parametrize("goal", list(GOALS), ids=list(GOALS))
def test_an_ackermann_robot_reaches_goals_a_diff_drive_reaches(controller, goal):
    """A car reaches the same goals as a diff drive, without stalling at the start.

    The goals need a turn on the spot for a diff drive; the car swings instead.
    """
    world = edit.set_part(load_world(DEMO), "r1", "controller", controller)
    assert drive_to(edit.set_model(world, "r1", "diff_drive"), GOALS[goal]) is not None
    assert drive_to(edit.set_model(world, "r1", "ackermann"), GOALS[goal]) is not None


EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "ackermann.toml"


def test_the_ackermann_example_arrives_at_every_goal_angle():
    """Dubins paths end at the goal's angle, so the car finishes each task aligned.

    The example's three waypoints all use the default tolerance, 0.1 m and
    0.1 rad, which the car can only meet by arriving along the goal's heading.
    """
    from demuvo import Run  # pylint: disable=import-outside-toplevel

    run = Run(load_world(EXAMPLE))
    assert run.until_done(90.0)


def test_a_car_on_a_corner_making_planner_warns_once(caplog):
    """An ackermann robot with the straight-line planner logs a warning.

    With Dubins, and for a diff drive on any planner, it does not.
    """
    import logging  # pylint: disable=import-outside-toplevel

    def warnings_for(model, planner):
        world = edit.set_model(load_world(DEMO), "r1", model)
        world = edit.set_part(world, "r1", "planner", planner)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="demuvo.nav.navigation"):
            Navigation(world)
        return [r for r in caplog.records if "curvature" in r.message]

    assert len(warnings_for("ackermann", "demuvo.planners:StraightLine")) == 1
    assert warnings_for("ackermann", "demuvo.planners:Dubins") == []
    assert warnings_for("diff_drive", "demuvo.planners:StraightLine") == []


def test_a_planner_tighter_than_the_steering_warns(caplog):
    """The default car cannot steer under 0.55 m; a planner at 0.3 m is a mistake."""
    import logging  # pylint: disable=import-outside-toplevel

    def warnings_for(radius):
        world = edit.set_model(load_world(DEMO), "r1", "ackermann")
        world = edit.set_part(world, "r1", "planner", "demuvo.planners:Dubins")
        world = edit.set_settings(world, "r1", "planner", {"turning_radius": radius})
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="demuvo.nav.navigation"):
            Navigation(world)
        return [r for r in caplog.records if "cannot steer tighter" in r.message]

    assert len(warnings_for(0.3)) == 1
    assert warnings_for(0.6) == []
    assert warnings_for(0.5493) == []  # about the least radius itself


PARKING = (
    Path(__file__).resolve().parents[4] / "examples" / "worlds" / "reeds_shepp.toml"
)


def test_the_reeds_shepp_example_parks_the_car():
    """Each goal needs a change of direction, which the car follows cusp by cusp.

    All three tasks use the default tolerance, 0.1 m and 0.1 rad.
    """
    from demuvo import Run  # pylint: disable=import-outside-toplevel

    run = Run(load_world(PARKING))
    assert run.until_done(90.0)


def test_dwb_follows_the_same_parking_world_cusp_by_cusp():
    """DWB drives each run in its own direction and lands on every goal angle."""
    from demuvo import Run  # pylint: disable=import-outside-toplevel

    path = PARKING.with_name("reeds_shepp_dwb.toml")
    run = Run(load_world(path))
    assert run.until_done(120.0)
    assert run.result()[-1] == "0 collisions"
    assert all("reached" in line for line in run.result()[:3])


def test_the_parking_world_is_parked_with_the_default_lookahead_too():
    """A long lookahead cuts the last arc; the car shortens it near the end instead."""
    from demuvo import Run  # pylint: disable=import-outside-toplevel

    world = edit.set_settings(load_world(PARKING), "car", "controller", {})
    run = Run(world)
    assert run.until_done(90.0)
    assert all("reached" in line for line in run.result()[:3])


def test_a_mecanum_controller_gets_the_sideways_bound(tmp_path, monkeypatch):
    """The observation says a mecanum robot turns in place and strafes."""
    spy, world = spy_world(tmp_path, monkeypatch, "mecanum")
    observe(world)
    limits = spy.seen[-1].limits
    assert limits is not None
    assert limits.turns_in_place
    assert limits.strafes
    assert (limits.max_v, limits.max_vy, limits.max_w(0.0)) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "controller",
    [
        "demuvo.controllers:PurePursuit",
        "demuvo.controllers:DWB",
        "demuvo.controllers:OmniPursuit",
    ],
)
@pytest.mark.parametrize("goal", list(GOALS), ids=list(GOALS))
def test_a_mecanum_robot_reaches_the_goals_a_diff_drive_reaches(controller, goal):
    """Pure pursuit drives it as a diff drive; DWB and omni pursuit strafe it."""
    world = edit.set_part(load_world(DEMO), "r1", "controller", controller)
    assert drive_to(edit.set_model(world, "r1", "mecanum"), GOALS[goal]) is not None


MECANUM = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "mecanum.toml"


def test_the_mecanum_example_finishes_its_lap():
    """Four waypoints round the box, on dead reckoning, with no collision."""
    from demuvo import Run  # pylint: disable=import-outside-toplevel

    run = Run(load_world(MECANUM))
    assert run.until_done(90.0)
