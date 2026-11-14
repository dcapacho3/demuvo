"""Tests for planning again while the robot drives."""

import logging
from pathlib import Path

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import Navigation, Run, Simulation, Twist, load_world
from demuvo.costmap import LETHAL, Costmap
from demuvo.nav.replan import Replanner, Sight, changed_area, path_blocked
from demuvo.world import ReplanSpec, Task

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "replan.toml"
GOAL = Task("r1", Transform2D.from_xyt((9.0, 3.0, 0.0)), (0.1, 0.1))
STILL = np.zeros(3)
MOVING = np.array([0.5, 0.0, 0.0])


def straight(x0, x1, y=3.0, n=41):
    """A path along y from x0 to x1."""
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, y), np.zeros(n)])


def sight(time=5.0, x=2.0, y=3.0, planned_at=0.0, **changes):
    """What the replanner sees: a robot on a straight path to GOAL, after a plan."""
    fields = {
        "time": time,
        "pose": Transform2D.from_xyt((x, y, 0.0)),
        "path": straight(1.0, 9.0),
        "task": GOAL,
        "planned_at": planned_at,
        "failed": False,
        "velocity": MOVING,
        "costmap": None,
    }
    return Sight(**{**fields, **changes})


def window(marks=(), centre=(3.0, 3.0), size=3.0, resolution=0.05):
    """A local costmap round `centre` with LETHAL cells at the given (x, y) points."""
    cells = round(size / resolution)
    data = np.zeros((cells, cells), np.uint8)
    origin = (centre[0] - size / 2, centre[1] - size / 2)
    grid = Costmap(data, resolution, origin)
    for x, y in marks:
        if grid.in_bounds(x, y):
            data[grid.world_to_cell(x, y)] = LETHAL
    return grid


# -- the triggers --


def test_a_robot_without_the_spec_never_plans_again():
    """No spec, no reason, whatever is seen."""
    replanner = Replanner({"r1": None})
    assert replanner.reason("r1", sight(failed=True, velocity=STILL)) is None


def test_a_robot_within_reach_of_its_goal_never_plans_again():
    """At the goal it turns to the angle or holds; a new path would be a point."""
    replanner = Replanner({"r1": ReplanSpec(timer=True)})
    assert replanner.reason("r1", sight(x=8.95)) is None
    assert replanner.reason("r1", sight(x=8.8)) == "1 s passed"


def test_no_two_plans_come_closer_than_the_period():
    """A trigger that holds sooner than the period waits for it."""
    replanner = Replanner({"r1": ReplanSpec(timer=True, period=0.5)})
    assert replanner.reason("r1", sight(time=0.4)) is None
    assert replanner.reason("r1", sight(time=0.5)) == "0.5 s passed"


def test_a_plan_that_failed_is_tried_again_every_period_whatever_the_triggers():
    """With every trigger off, a robot that has no path still asks again."""
    spec = ReplanSpec(period=1.0, blocked=False, off_path=0.0, stuck=0.0)
    replanner = Replanner({"r1": spec})
    assert replanner.reason("r1", sight(time=0.9, failed=True)) is None
    assert replanner.reason("r1", sight(time=1.0, failed=True)) == "it has no path yet"
    assert replanner.reason("r1", sight(time=1.0)) is None


def test_a_block_on_the_path_ahead_asks_for_a_plan_and_one_behind_does_not():
    """Only the part of the path still to drive counts."""
    replanner = Replanner({"r1": ReplanSpec(off_path=0.0, stuck=0.0)})
    ahead, behind = window([(3.0, 3.0)]), window([(1.5, 3.0)])
    assert replanner.reason("r1", sight(costmap=ahead)) == "the path ahead is blocked"
    assert replanner.reason("r1", sight(costmap=behind)) is None
    quiet = Replanner({"r1": ReplanSpec(blocked=False, off_path=0.0, stuck=0.0)})
    assert quiet.reason("r1", sight(costmap=ahead)) is None


def test_too_far_off_the_path_asks_for_a_plan_with_the_distance():
    """The distance is to the path's segments, in the reason."""
    replanner = Replanner({"r1": ReplanSpec(off_path=0.5, stuck=0.0)})
    assert replanner.reason("r1", sight(y=3.4)) is None
    assert replanner.reason("r1", sight(y=3.6)) == "it is 0.60 m off the path"


def test_standing_still_counts_up_and_moving_starts_it_over():
    """Still for `stuck` seconds, short of the goal, asks for a plan."""
    replanner = Replanner({"r1": ReplanSpec(off_path=0.0, stuck=2.0)})
    assert replanner.reason("r1", sight(time=10.0, velocity=STILL)) is None
    assert replanner.reason("r1", sight(time=11.0, velocity=STILL)) is None
    assert replanner.reason("r1", sight(time=11.5, velocity=MOVING)) is None
    assert replanner.reason("r1", sight(time=12.0, velocity=STILL)) is None
    assert replanner.reason("r1", sight(time=14.0, velocity=STILL)) == (
        "it has stood still for 2.0 s"
    )
    replanner.forget("r1")
    assert replanner.reason("r1", sight(time=15.0, velocity=STILL)) is None


def test_stillness_is_counted_through_the_waiting_period():
    """The period holds the plan back, not the count of still seconds."""
    replanner = Replanner({"r1": ReplanSpec(period=3.0, off_path=0.0, stuck=1.0)})
    assert replanner.reason("r1", sight(time=0.5, velocity=STILL)) is None
    assert replanner.reason("r1", sight(time=3.0, velocity=STILL)) == (
        "it has stood still for 2.5 s"
    )


def test_a_changed_local_costmap_asks_for_a_plan_once_it_changed_enough():
    """The first costmap is the reference; a body appearing in it is a change."""
    spec = ReplanSpec(blocked=False, off_path=0.0, stuck=0.0, changed=0.1)
    replanner = Replanner({"r1": spec})
    empty = window()
    assert replanner.reason("r1", sight(costmap=empty)) is None  # nothing to compare
    assert replanner.reason("r1", sight(costmap=window([(3.0, 3.6)]))) is None
    marks = [(3.0 + 0.05 * i, 3.6 + 0.05 * j) for i in range(8) for j in range(8)]
    assert replanner.reason("r1", sight(costmap=window(marks))) == (
        "the local costmap changed over 0.16 square metres"
    )
    replanner.planned("r1", window(marks))  # planned on it: the reference moves on
    assert replanner.reason("r1", sight(costmap=window(marks))) is None
    replanner.planned("r1", None)  # planned with no costmap: the next one is it
    assert replanner.reason("r1", sight(costmap=empty)) is None
    assert replanner.reason("r1", sight(costmap=window(marks))) is not None
    off = Replanner({"r1": ReplanSpec(blocked=False, off_path=0.0, stuck=0.0)})
    assert off.reason("r1", sight(costmap=empty)) is None
    assert off.reason("r1", sight(costmap=window(marks))) is None


def test_changed_area_compares_only_where_both_windows_lie():
    """The same grid, a mark added, a window that rolled on, and two apart."""
    assert changed_area(window(), window()) == 0.0
    assert changed_area(window(), window([(3.0, 3.0)])) == pytest.approx(0.0025)
    rolled = window([(1.0, 3.0)], centre=(2.0, 3.0))  # the mark is off the old grid
    assert changed_area(window(), rolled) == 0.0
    assert changed_area(window([(3.0, 3.0)]), rolled) == pytest.approx(0.0025)
    assert changed_area(window(), window([(8.0, 3.0)], centre=(8.0, 3.0))) == 0.0


def test_path_blocked_looks_from_the_nearest_point_on_and_inside_the_grid_only():
    """An empty path, a block behind, and a path leaving the grid."""
    pose = Transform2D.from_xyt((2.0, 3.0, 0.0))
    assert not path_blocked(np.empty((0, 3)), pose, window([(3.0, 3.0)]))
    assert path_blocked(straight(1.0, 9.0), pose, window([(3.0, 3.0)]))
    assert not path_blocked(straight(1.0, 9.0), pose, window([(1.6, 3.0)]))
    far = window([(3.0, 3.0)], centre=(7.0, 3.0))  # the mark lies off this grid
    assert not path_blocked(straight(1.0, 9.0), pose, far)
    assert path_blocked(straight(1.0, 9.0, n=2), pose, window([(3.0, 3.0)]))


# -- the handoff --


def planner_module(tmp_path, monkeypatch):
    """A planner that records its calls and a controller that drives straight."""
    (tmp_path / "re_nav.py").write_text(
        "import numpy as np\n"
        "from demuvo import Twist\n"
        "calls, configured, seen = [], [], []\n"
        "fail = False\n"
        "class Line:\n"
        "    def configure(self, costmap):\n"
        "        configured.append(costmap)\n"
        "    def plan(self, start, goal):\n"
        "        calls.append(tuple(np.round(start.xyt, 6)))\n"
        "        if fail:\n"
        "            return np.empty((0, 3))\n"
        "        x = np.linspace(start.xyt[0], goal.xyt[0], 41)\n"
        "        return np.column_stack([x, np.full(41, 3.0), np.zeros(41)])\n"
        "class Drive:\n"
        "    def command(self, obs):\n"
        "        seen.append((round(obs.time, 6), obs.path_changes, obs.path))\n"
        "        return Twist(vx=0.5)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    import re_nav  # pylint: disable=import-error,import-outside-toplevel

    re_nav.calls.clear()
    re_nav.configured.clear()
    re_nav.seen.clear()
    re_nav.fail = False
    return re_nav


def world_with(tmp_path, replan, lidar=False, extra=""):
    """A world file with one robot driving east, planning again as `replan` says."""
    sensors = "sensors = { lidar = { beams = 90, max_range = 4.0 } }\n" if lidar else ""
    text = (
        "version = 1\n[world]\nsize = [10.0, 6.0]\nseed = 1\n"
        "[[obstacle]]\npose = [5.0, 3.0, 0.0]\n"
        'shape = { name = "box", size = [0.4, 1.0] }\n'
        '[[robot]]\nname = "r1"\nmodel = "diff_drive"\n'
        'shape = { name = "circle", radius = 0.2 }\npose = [1.0, 3.0, 0.0]\n'
        'planner = "re_nav:Line"\ncontroller = "re_nav:Drive"\n'
        f"{sensors}replan = {replan}\n{extra}"
        '[costmap.global]\nplugins = ["inflation"]\n'
        '[[task]]\nrobot = "r1"\ngoal = [9.0, 3.0, 0.0]\n'
    )
    path = tmp_path / "replan.toml"
    path.write_text(text, encoding="utf-8")
    return path


TIMER = "{ period = 0.2, timer = true, blocked = false, off_path = 0.0, stuck = 0.0 }"


def test_a_new_path_starts_at_the_robot_and_reaches_the_controller_in_the_same_run(
    tmp_path, monkeypatch
):
    """Each plan starts from the pose the controller sees and replaces the path then."""
    re_nav = planner_module(tmp_path, monkeypatch)
    sim = Simulation(load_world(world_with(tmp_path, TIMER)), dt=0.05)
    nav = Navigation(sim.world)
    for _ in range(10):
        sim.step(nav(sim))
    # Plans at 0, 0.2 and 0.4 s; the robot drives at 0.5 m/s from x = 1.
    assert re_nav.calls == [(1.0, 3.0, 0.0), (1.1, 3.0, 0.0), (1.2, 3.0, 0.0)]
    assert [changes for _, changes, _ in re_nav.seen] == [1] * 4 + [2] * 4 + [3] * 2
    for time, _, path in re_nav.seen:
        assert path[0, 0] == pytest.approx(1.0 + 0.5 * (time - time % 0.2), abs=1e-9)
    assert nav.paths["r1"][0, 0] == pytest.approx(1.2)


def test_a_robot_that_replans_plans_on_what_its_lidar_sees(tmp_path, monkeypatch):
    """The planner is configured again with the local costmap over the global one."""
    re_nav = planner_module(tmp_path, monkeypatch)
    sim = Simulation(load_world(world_with(tmp_path, TIMER, lidar=True)), dt=0.05)
    nav = Navigation(sim.world)
    for _ in range(100):  # 5 s: the robot gets to x = 3.5, with the box in its window
        sim.step(nav(sim))
    base = sim.costmaps["r1"]
    assert len(re_nav.configured) == 25  # once for each plan
    assert re_nav.configured[0] is base  # no local costmap before the first step
    merged = re_nav.configured[-1]
    assert merged is not base
    assert (merged.resolution, merged.origin) == (base.resolution, base.origin)
    assert (merged.data >= base.data).all()
    assert base.cost_at(4.82, 3.0) == 0  # the map does not show the box
    assert sim.robots["r1"].local_costmap.cost_at(4.82, 3.0) == LETHAL  # the lidar does
    assert merged.cost_at(4.82, 3.0) == LETHAL  # so the planner is told


CHANGED = (
    "{ period = 0.2, blocked = false, off_path = 0.0, stuck = 0.0, changed = 0.1 }"
)
OTHER = (
    '[[robot]]\nname = "r2"\nmodel = "diff_drive"\n'
    'shape = { name = "circle", radius = 0.2 }\npose = [3.0, 5.5, -1.5708]\n'
)


def test_another_robot_driving_into_the_local_costmap_makes_the_robot_plan_again(
    tmp_path, monkeypatch, caplog
):
    """r2 comes south into r1's window; r1, planned once, plans again and says why."""
    re_nav = planner_module(tmp_path, monkeypatch)
    world = load_world(world_with(tmp_path, CHANGED, lidar=True, extra=OTHER))
    sim = Simulation(world, dt=0.05)
    nav = Navigation(world)
    with caplog.at_level(logging.INFO, logger="demuvo.nav"):
        for _ in range(16):  # 0.8 s: r2, at y 5.1, is still clear of the window
            sim.step({**nav(sim), "r2": Twist(vx=0.5)})
        assert len(re_nav.calls) == 1
        for _ in range(24):  # 2 s: r2's body has come into the window, at y 4.5
            sim.step({**nav(sim), "r2": Twist(vx=0.5)})
    assert len(re_nav.calls) >= 2
    reasons = [
        r.getMessage() for r in caplog.records if "planned task" in r.getMessage()
    ]
    assert reasons and all("the local costmap changed over" in r for r in reasons)
    assert sim.poses["r2"].xyt[1] - 0.2 < 4.5  # its body is in the window


def test_a_path_from_outside_is_never_planned_again(tmp_path, monkeypatch):
    """`set_path` puts the planner aside, timer or not."""
    re_nav = planner_module(tmp_path, monkeypatch)
    sim = Simulation(load_world(world_with(tmp_path, TIMER)), dt=0.05)
    nav = Navigation(sim.world)
    nav.set_path("r1", straight(1.0, 9.0))
    for _ in range(10):
        sim.step(nav(sim))
    assert re_nav.calls == []
    assert {changes for _, changes, _ in re_nav.seen} == {1}


def test_a_plan_that_finds_nothing_keeps_the_path_and_says_so_once(
    tmp_path, monkeypatch, caplog
):
    """The old path stays, one warning, quiet retries, then a path again."""
    re_nav = planner_module(tmp_path, monkeypatch)
    sim = Simulation(load_world(world_with(tmp_path, TIMER)), dt=0.05)
    nav = Navigation(sim.world)
    sim.step(nav(sim))
    first = nav.paths["r1"]
    re_nav.fail = True
    with caplog.at_level(logging.WARNING, logger="demuvo.nav"):
        for _ in range(12):  # retries at 0.2, 0.4 and 0.6 s
            sim.step(nav(sim))
    assert nav.paths["r1"] is first
    assert len(re_nav.calls) == 4
    assert [r.getMessage() for r in caplog.records] == [
        "robot 'r1': no new path for task 1, though 0.2 s passed; it keeps the "
        "path it has and tries again"
    ]
    assert {changes for _, changes, _ in re_nav.seen} == {1}
    re_nav.fail = False
    with caplog.at_level(logging.WARNING, logger="demuvo.nav"):
        for _ in range(4):
            sim.step(nav(sim))
    assert caplog.records[-1].getMessage() == "robot 'r1': found a path for task 1"
    assert nav.paths["r1"] is not first
    assert re_nav.seen[-1][1] == 2


def test_a_robot_with_no_path_at_all_tries_again_and_drives_once_it_has_one(
    tmp_path, monkeypatch, caplog
):
    """The first warning says it tries again; the retry that finds a path says so."""
    re_nav = planner_module(tmp_path, monkeypatch)
    re_nav.fail = True
    sim = Simulation(load_world(world_with(tmp_path, TIMER)), dt=0.05)
    nav = Navigation(sim.world)
    with caplog.at_level(logging.WARNING, logger="demuvo.nav"):
        for _ in range(8):
            sim.step(nav(sim))
        re_nav.fail = False
        for _ in range(4):
            sim.step(nav(sim))
    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "robot 'r1': no path for task 1, from (1.00, 3.00) to (9.00, 3.00); it "
        "stays still and tries again every 0.2 s",
        "robot 'r1': found a path for task 1",
    ]
    assert len(re_nav.calls) == 3  # at 0, 0.2 and 0.4 s
    assert [c for t, c, _ in re_nav.seen if t < 0.4] == [1] * 8
    assert [c for t, c, _ in re_nav.seen if t >= 0.4] == [2] * 4


def test_the_example_world_needs_replanning_to_get_past_the_box():
    """On a map with no obstacles, only the robot that plans again gets through."""
    run = Run(EXAMPLE)
    assert run.until_done(40.0)
    assert run.result()[-1] == "0 collisions"
    text = EXAMPLE.read_text(encoding="utf-8")
    world = load_world(EXAMPLE)
    assert world.robots[0].replan == ReplanSpec()
    once = Run(
        world.__class__(
            **{
                **vars(world),
                "robots": (
                    world.robots[0].__class__(
                        **{**vars(world.robots[0]), "replan": None}
                    ),
                ),
            }
        )
    )
    assert not once.until_done(40.0)
    assert once.result()[-1] == "1 collision"
    assert "replan = " in text
