"""Tests for joining planners and controllers to a simulation."""

import math
from pathlib import Path

import pytest

from demuvo import Navigation, Simulation, Twist, WorldError, load_world

DEMO = Path(__file__).resolve().parents[1] / "examples" / "worlds" / "demo.toml"
PURE_PURSUIT = 'controller = "demuvo.controllers:PurePursuit"'


def world_file(tmp_path, **changes):
    """Write the demo world with some text replaced, and return its path."""
    text = DEMO.read_text(encoding="utf-8")
    for old, new in changes.items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "world.toml"
    path.write_text(text, encoding="utf-8")
    return path


def run(path, steps):
    """Run a world file with its own planner and controller; return the simulation."""
    sim = Simulation(load_world(path))
    commands = Navigation(sim.world)
    for _ in range(steps):
        sim.step(commands(sim))
    return sim


def error(sim):
    """Distance and angle between the robot and its goal."""
    x, y, yaw = sim.poses["r1"].xyt
    gx, gy, gyaw = sim.specs["r1"].goal
    return math.hypot(gx - x, gy - y), abs(
        (gyaw - yaw + math.pi) % (2 * math.pi) - math.pi
    )


def test_the_demo_robot_reaches_its_goal_and_turns_to_its_angle():
    """The demo file plans a line, follows it, and ends within both tolerances."""
    sim = run(DEMO, 800)
    distance, angle = error(sim)
    assert distance <= 0.1
    assert angle <= 0.1


def test_with_the_final_turn_off_the_robot_ends_facing_along_the_path(tmp_path):
    """The switch in the file changes the behaviour: position right, angle not."""
    path = world_file(
        tmp_path,
        **{"speed = 0.5, lookahead = 0.6": "speed = 0.5, rotate_to_goal = false"},
    )
    distance, angle = error(run(path, 800))
    assert distance <= 0.1
    assert angle > 0.3


def test_a_robot_with_no_controller_gets_no_command(tmp_path):
    """The commands function is empty, so the robot stays where it is."""
    lines = DEMO.read_text(encoding="utf-8").splitlines()
    text = "\n".join(
        line
        for line in lines
        if not line.startswith(("controller", "planner", "goal_tolerance"))
    )
    path = tmp_path / "w.toml"
    path.write_text(text, encoding="utf-8")
    sim = Simulation(load_world(path))
    assert not Navigation(sim.world)(sim)


def user_module(tmp_path, monkeypatch):
    """A module with a planner and a controller of the user's own."""
    (tmp_path / "my_nav.py").write_text(
        "import numpy as np\n"
        "from demuvo import Twist\n"
        "calls = []\n"
        "class Points:\n"
        "    def __init__(self, n=2):\n"
        "        self.n = n\n"
        "    def plan(self, world, start, goal):\n"
        "        calls.append((start, goal))\n"
        "        return np.zeros((self.n, 3))\n"
        "class Count:\n"
        "    def command(self, obs):\n"
        "        return Twist(vx=float(len(obs.path)))\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    import my_nav  # pylint: disable=import-error,import-outside-toplevel

    my_nav.calls.clear()  # the module stays loaded between tests
    return my_nav


def test_the_planner_runs_once_and_its_path_reaches_the_controller(
    tmp_path, monkeypatch
):
    """Params reach both classes; the path is planned at the first step only."""
    my_nav = user_module(tmp_path, monkeypatch)
    path = world_file(
        tmp_path,
        **{
            PURE_PURSUIT: 'controller = "my_nav:Count"',
            'planner = "demuvo.planners:StraightLine"': 'planner = "my_nav:Points"',
            "controller_params = { speed = 0.5, lookahead = 0.6 }": "",
        },
    )
    path.write_text(path.read_text() + "planner_params = { n = 7 }\n")
    sim = Simulation(load_world(path))
    commands = Navigation(sim.world)
    assert [commands(sim) for _ in range(3)] == [{"r1": Twist(vx=7.0)}] * 3
    assert my_nav.calls == [((1.0, 1.0, 0.0), (9.0, 5.0, 0.0))]


def test_a_robot_with_no_goal_gets_an_empty_path(tmp_path, monkeypatch):
    """The controller still runs; its path is empty."""
    user_module(tmp_path, monkeypatch)
    lines = DEMO.read_text(encoding="utf-8").splitlines()
    text = "\n".join(
        line
        for line in lines
        if not line.startswith(("goal", "planner", "controller_params"))
    ).replace("demuvo.controllers:PurePursuit", "my_nav:Count")
    path = tmp_path / "w.toml"
    path.write_text(text, encoding="utf-8")
    sim = Simulation(load_world(path))
    assert Navigation(sim.world)(sim) == {"r1": Twist(vx=0.0)}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (PURE_PURSUIT, 'controller = "no_such_module_xyz:Thing"', "controller"),
        (PURE_PURSUIT, 'controller = "demuvo.controllers:Nope"', "controller"),
        (PURE_PURSUIT, 'controller = "demuvo.controllers:Controller"', "controller"),
        ("speed = 0.5", "speed = -1", "speed must be a positive number"),
        ("speed = 0.5", "pace = 0.5", "controller 'demuvo.controllers:PurePursuit'"),
        ("demuvo.planners:StraightLine", "demuvo.planners:Nope", "planner"),
        ("demuvo.planners:StraightLine", "no_such_module_xyz:Thing", "planner"),
    ],
)
def test_a_class_that_cannot_load_names_the_robot_and_the_kind(
    tmp_path, old, new, message
):
    """A missing module or class, a bad argument, or a bad value."""
    path = world_file(tmp_path, **{old: new})
    with pytest.raises(WorldError, match=f"robot 'r1': cannot load.*{message}"):
        Navigation(load_world(path))


def counting_world(tmp_path, monkeypatch):
    """The demo world with the user's counting planner and controller."""
    my_nav = user_module(tmp_path, monkeypatch)
    path = world_file(
        tmp_path,
        **{
            PURE_PURSUIT: 'controller = "my_nav:Count"',
            'planner = "demuvo.planners:StraightLine"': 'planner = "my_nav:Points"',
            "controller_params = { speed = 0.5, lookahead = 0.6 }": "",
        },
    )
    sim = Simulation(load_world(path))
    return sim, Navigation(sim.world), my_nav


def test_a_new_goal_is_planned_again(tmp_path, monkeypatch):
    """set_goal drops the old path; the planner runs again from where the robot is."""
    sim, nav, my_nav = counting_world(tmp_path, monkeypatch)
    nav(sim)
    nav.set_goal("r1", (2.0, 2.0, 0.5))
    nav(sim)
    nav(sim)
    assert my_nav.calls == [
        ((1.0, 1.0, 0.0), (9.0, 5.0, 0.0)),
        ((1.0, 1.0, 0.0), (2.0, 2.0, 0.5)),
    ]
    assert nav.goals["r1"] == (2.0, 2.0, 0.5)


def test_a_path_from_outside_is_used_and_not_replanned(tmp_path, monkeypatch):
    """set_path replaces the planner's path, and the planner is never asked."""
    sim, nav, my_nav = counting_world(tmp_path, monkeypatch)
    nav.set_path("r1", [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
    assert nav(sim) == {"r1": Twist(vx=3.0)}
    assert my_nav.calls == []


def test_the_end_of_an_outside_path_is_the_goal_when_there_is_none(
    tmp_path, monkeypatch
):
    """With no goal set, the controller is told the last pose of the path."""
    sim, nav, _ = counting_world(tmp_path, monkeypatch)
    seen = []
    nav._controllers["r1"].command = lambda obs: seen.append(obs.goal) or Twist()  # pylint: disable=protected-access
    nav.set_goal("r1", None)
    nav.set_path("r1", [[1.0, 1.0, 0.0], [4.0, 2.0, 0.7]])
    nav(sim)
    assert seen == [(4.0, 2.0, 0.7)]


@pytest.mark.parametrize(
    "path",
    [[], [1.0, 2.0, 3.0], [[1.0, 2.0]], [[1.0, 2.0, math.nan]], [[0.0, 0.0, math.inf]]],
)
def test_a_bad_outside_path_is_refused(tmp_path, monkeypatch, path):
    """The path must be rows of three finite numbers, at least one row."""
    _, nav, _ = counting_world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="path must"):
        nav.set_path("r1", path)


def test_a_goal_or_path_for_an_unknown_robot_is_refused(tmp_path, monkeypatch):
    """Only the robots of the world can be given a goal or a path."""
    _, nav, _ = counting_world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no robot named 'r2'"):
        nav.set_goal("r2", (1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="no robot named 'r2'"):
        nav.set_path("r2", [[1.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    ("changes", "kind", "method"),
    [
        (
            {
                PURE_PURSUIT: 'controller = "demuvo.planners:StraightLine"',
                "controller_params = { speed = 0.5, lookahead = 0.6 }": "",
            },
            "controller",
            "command",
        ),
        (
            {"demuvo.planners:StraightLine": "demuvo.controllers:PurePursuit"},
            "planner",
            "plan",
        ),
    ],
)
def test_a_class_with_the_wrong_methods_is_refused_before_the_first_step(
    tmp_path, changes, kind, method
):
    """A planner named as a controller, or the reverse, fails at load, by name."""
    path = world_file(tmp_path, **changes)
    with pytest.raises(WorldError, match=f"robot 'r1': {kind} .* has no '{method}'"):
        Navigation(load_world(path))
