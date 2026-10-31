"""Tests for joining planners and controllers to a simulation."""

import math
from pathlib import Path

import pytest
from robomath import Transform2D

from demuvo import Navigation, Simulation, Task, Twist, WorldError, load_world
from demuvo.judge import reached

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
    sim = Simulation(load_world(path), dt=0.05)
    commands = Navigation(sim.world)
    for _ in range(steps):
        sim.step(commands(sim))
    return sim


def error(sim):
    """Distance and angle between the robot and its goal."""
    x, y, yaw = sim.poses["r1"].xyt
    gx, gy, gyaw = sim.world.tasks[-1].goal.xyt
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
        "        calls.append((tuple(start.xyt), tuple(goal.xyt)))\n"
        "        return np.zeros((self.n, 3))\n"
        "class Count:\n"
        "    def command(self, obs):\n"
        "        return Twist(vx=float(len(obs.path)))\n"
        "seen = []\n"
        "class Log:\n"
        "    def command(self, obs):\n"
        "        seen.append((round(obs.time, 6), round(obs.dt, 6)))\n"
        "        return Twist(vx=obs.time)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    import my_nav  # pylint: disable=import-error,import-outside-toplevel

    my_nav.calls.clear()  # the module stays loaded between tests
    my_nav.seen.clear()
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
            'planner = "demuvo.planners:StraightLine"': (
                'planner = "my_nav:Points"\nplanner_params = { n = 7 }'
            ),
            "controller_params = { speed = 0.5, lookahead = 0.6 }": "",
        },
    )
    sim = Simulation(load_world(path))
    commands = Navigation(sim.world)
    assert [commands(sim) for _ in range(3)] == [{"r1": Twist(vx=7.0)}] * 3
    assert my_nav.calls == [((1.0, 1.0, 0.0), (9.0, 5.0, 0.0))]


def test_a_robot_with_no_task_gets_an_empty_path(tmp_path, monkeypatch):
    """The controller still runs; its path is empty."""
    user_module(tmp_path, monkeypatch)
    lines = DEMO.read_text(encoding="utf-8").split("[[task]]")[0].splitlines()
    text = "\n".join(
        line for line in lines if not line.startswith(("planner", "controller_params"))
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
    sim = Simulation(load_world(path), dt=0.05)
    return sim, Navigation(sim.world), my_nav


def test_a_new_goal_is_planned_again(tmp_path, monkeypatch):
    """set_goal drops the old path; the planner runs again from where the robot is."""
    sim, nav, my_nav = counting_world(tmp_path, monkeypatch)
    nav(sim)
    nav.set_goal("r1", Transform2D.from_xyt((2.0, 2.0, 0.5)))
    sim.step({})
    nav(sim)
    assert my_nav.calls == [
        ((1.0, 1.0, 0.0), (9.0, 5.0, 0.0)),
        ((1.0, 1.0, 0.0), (2.0, 2.0, 0.5)),
    ]
    assert tuple(nav.tasks["r1"][0].goal.xyt) == (2.0, 2.0, 0.5)


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
    assert tuple(seen[0].xyt) == pytest.approx((4.0, 2.0, 0.7))


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
        nav.set_goal("r2", Transform2D.from_xyt((1.0, 1.0, 0.0)))
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


def test_reached_needs_both_the_distance_and_the_angle():
    """Inside the distance but not the angle is not reached."""
    task = Task("r1", Transform2D.from_xyt((3.0, 3.0, 0.0)), (0.1, 0.2))
    assert reached(Transform2D.from_xyt((3.05, 3.0, 0.1)), task)
    assert not reached(Transform2D.from_xyt((3.05, 3.0, 0.5)), task)
    assert not reached(Transform2D.from_xyt((3.3, 3.0, 0.0)), task)


def test_a_robot_visits_its_waypoints_in_order(tmp_path):
    """Two tasks: the robot reaches the first, goes on, and ends at the second."""
    text = DEMO.read_text(encoding="utf-8")
    text = text.replace("goal = [9.0, 5.0, 0.0]", "goal = [3.0, 1.0, 0.0]")
    text += '\n[[task]]\nrobot = "r1"\ngoal = [3.0, 3.0, 1.5]\n'
    path = tmp_path / "waypoints.toml"
    path.write_text(text, encoding="utf-8")
    sim = Simulation(load_world(path))
    nav = Navigation(sim.world)
    first = sim.world.tasks[0].goal.translation
    closest = 99.0
    for _ in range(1500):
        sim.step(nav(sim))
        x, y, _ = sim.poses["r1"].xyt
        closest = min(closest, math.hypot(x - first[0], y - first[1]))
    assert closest <= 0.1
    assert reached(sim.poses["r1"], sim.world.tasks[1])


def test_a_goal_set_from_outside_replaces_the_tasks(tmp_path, monkeypatch):
    """set_goal leaves one task; None leaves none and the robot has no goal."""
    sim, nav, _ = counting_world(tmp_path, monkeypatch)
    nav.set_goal("r1", Transform2D.from_xyt((2.0, 2.0, 0.0)), (0.3, 0.4))
    assert [t.goal_tolerance for t in nav.tasks["r1"]] == [(0.3, 0.4)]
    nav.set_goal("r1", None)
    assert nav.tasks["r1"] == []
    assert nav(sim) == {"r1": Twist(vx=0.0)}


def logging_run(tmp_path, monkeypatch, dt, period, steps):
    """Run the demo robot with the Log controller; return the log and the commands."""
    my_nav = user_module(tmp_path, monkeypatch)
    path = world_file(
        tmp_path,
        **{
            PURE_PURSUIT: 'controller = "my_nav:Log"',
            "controller_params = { speed = 0.5, lookahead = 0.6 }": "",
            "control_period = 0.05": f"control_period = {period}",
        },
    )
    sim = Simulation(load_world(path), dt=dt)
    nav = Navigation(sim.world)
    vx = []
    for _ in range(steps):
        commands = nav(sim)
        vx.append(commands["r1"].vx)
        sim.step(commands)
    return my_nav.seen, vx


def test_the_controller_runs_once_per_period_and_its_command_is_held(
    tmp_path, monkeypatch
):
    """Period 0.05 with dt 0.01: a run every 5 steps; the command stays between."""
    seen, vx = logging_run(tmp_path, monkeypatch, dt=0.01, period=0.05, steps=12)
    assert seen == [(0.0, 0.05), (0.05, 0.05), (0.1, 0.05)]
    assert vx == [0.0] * 5 + [0.05] * 5 + [0.1] * 2


def test_a_period_that_is_not_a_multiple_alternates_two_gaps(tmp_path, monkeypatch):
    """Period 0.03, dt 0.02: runs on steps 0, 2, 3, 5, 6, 8; the mean gap is 0.03."""
    with pytest.warns(UserWarning, match="not a whole number of steps"):
        seen, _ = logging_run(tmp_path, monkeypatch, dt=0.02, period=0.03, steps=10)
    assert seen == [
        (0.0, 0.03),
        (0.04, 0.04),
        (0.06, 0.02),
        (0.1, 0.04),
        (0.12, 0.02),
        (0.16, 0.04),
        (0.18, 0.02),
    ]


def test_the_period_is_checked_once_for_each_dt(tmp_path, monkeypatch, recwarn):
    """A new dt is checked again; the same dt is not."""
    sim, nav, _ = counting_world(tmp_path, monkeypatch)
    nav.check(0.02)
    nav.check(0.02)
    assert len(recwarn) == 1
    nav.check(0.03)
    assert len(recwarn) == 2
    nav.check(0.01)
    assert len(recwarn) == 2
    assert sim.dt == 0.05


def test_a_period_below_dt_is_refused_by_name(tmp_path, monkeypatch):
    """A controller cannot run twice in one step."""
    user_module(tmp_path, monkeypatch)
    path = world_file(tmp_path, **{"control_period = 0.05": "control_period = 0.005"})
    sim = Simulation(load_world(path), dt=0.01)
    nav = Navigation(sim.world)
    with pytest.raises(
        WorldError, match="robot 'r1': control_period 0.005 is below dt 0.01"
    ):
        nav.check(sim.dt)
    with pytest.raises(WorldError, match="below dt"):
        nav(sim)


def test_a_robot_with_no_controller_has_no_period_to_check():
    """Only robots with a controller are checked."""
    world = load_world(DEMO)
    world = type(world)(
        world.size,
        world.seed,
        world.obstacles,
        (
            type(world.robots[0])(
                "r1",
                "diff_drive",
                world.robots[0].shape,
                world.robots[0].pose,
                control_period=0.001,
            ),
        ),
    )
    Navigation(world).check(0.01)


def waiting_world(tmp_path, wait=0.5):
    """The demo robot with a first task at its own start and a second elsewhere."""
    text = DEMO.read_text(encoding="utf-8")
    text = text.replace(
        "goal = [9.0, 5.0, 0.0]", f"goal = [1.0, 1.0, 0.0]\nwait = {wait}"
    )
    text += '\n[[task]]\nrobot = "r1"\ngoal = [3.0, 1.0, 0.0]\n'
    path = tmp_path / "waiting.toml"
    path.write_text(text, encoding="utf-8")
    sim = Simulation(load_world(path), dt=0.05)
    return sim, Navigation(sim.world)


def test_a_task_with_a_wait_holds_the_robot_until_the_time_has_passed(tmp_path):
    """The robot sits on the first goal for 0.5 s; only then does it plan the second."""
    sim, nav = waiting_world(tmp_path)
    for _ in range(5):
        sim.step(nav(sim))
    assert list(nav.paths["r1"][-1, :2]) == [1.0, 1.0]
    assert nav.judge.done == {}
    for _ in range(10):
        sim.step(nav(sim))
    assert nav.judge.done == {("r1", 0): 0.0}
    assert list(nav.paths["r1"][-1, :2]) == [3.0, 1.0]


def test_a_task_that_must_stop_is_not_done_while_the_robot_moves(tmp_path):
    """At the goal but still moving does not finish a task with a wait."""
    sim, nav = waiting_world(tmp_path, wait=0.05)
    sim.velocity["r1"][:] = [0.5, 0.0, 0.0]
    commands = nav(sim)
    assert nav.judge.done == {}
    sim.step(commands)  # the robot is told to hold, so it is still from now on
    for _ in range(2):
        sim.step(nav(sim))
    assert nav.judge.done == pytest.approx({("r1", 0): 0.05})


def test_the_last_task_notes_the_time_it_was_first_met():
    """The demo robot finishes its only task; the time is noted once."""
    sim = Simulation(load_world(DEMO), dt=0.05)
    nav = Navigation(sim.world)
    for _ in range(800):
        sim.step(nav(sim))
    assert list(nav.judge.done) == [("r1", 0)]
    assert 0.0 < nav.judge.done[("r1", 0)] <= sim.time
