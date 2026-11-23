"""Tests for the `demuvo generate` command."""

import math

import pytest

from demuvo.generate import Clearance, check_connectivity
from demuvo.run.cli import main
from demuvo.runner import Run
from demuvo.world import load_world

BASE = """
version = 1

[world]
size = [20.0, 14.0]
seed = 2

[[robot]]
name = "car"
model = "ackermann"
shape = {{ name = "box", size = [0.6, 0.4] }}
pose = [2.0, 2.0, 0.0]
planner = "demuvo.planners:AStar"
controller = "demuvo.controllers:PurePursuit"
controller_params = {{ speed = 0.7, lookahead = 0.7 }}
{extra}
"""


def base_file(tmp_path, extra=""):
    """A world file for the command to draw into."""
    path = tmp_path / "base.toml"
    path.write_text(BASE.format(extra=extra))
    return str(path)


def drawn(tmp_path, style, *extra, name="out.toml"):
    """Draw a world of this style into a file, and give back the exit code and path."""
    path = tmp_path / name
    argv = [
        "generate",
        style,
        "--size",
        "20",
        "14",
        "--seed",
        "5",
        "--radius",
        "0.25",
        "--out",
        str(path),
        *extra,
    ]
    return main(argv), path


@pytest.mark.parametrize("style", ["field", "rooms", "caves", "maze"])
def test_each_style_writes_a_world_file_that_reads_back(tmp_path, capsys, style):
    """The command's one job: a file the loader takes."""
    code, path = drawn(tmp_path, style)
    assert code == 0
    world = load_world(path)
    assert world.obstacles and world.size == (20.0, 14.0) and world.seed == 5
    printed = capsys.readouterr().out
    assert f"{style}, seed 5" in printed
    assert "free " in printed and "region" in printed
    assert f"# demuvo generate {style}" in printed  # the command that draws it again


@pytest.mark.parametrize("style", ["field", "rooms", "caves", "maze"])
def test_a_world_with_a_pair_runs_to_its_goal_as_it_is_written(tmp_path, style):
    """A map, a robot that can drive and somewhere to drive to, in one command."""
    code, path = drawn(tmp_path, style, "--pair")
    assert code == 0
    world = load_world(path)
    assert len(world.robots) == 1 and len(world.tasks) == 1
    run = Run(world)
    assert run.until_done(300.0)


def test_a_pair_is_as_far_apart_as_asked_and_the_robot_faces_its_goal(tmp_path):
    """The start looks the way it must go, which is what a reader would expect."""
    code, path = drawn(tmp_path, "caves", "--pair", "--min-distance", "12")
    assert code == 0
    world = load_world(path)
    start = world.robots[0].pose.translation.tolist()
    goal = world.tasks[0].goal.translation.tolist()
    assert math.dist(start, goal) >= 12.0
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    assert world.robots[0].pose.xyt[2] == pytest.approx(heading)


def test_the_goal_is_a_place_and_not_a_heading(tmp_path):
    """Any heading counts as reaching a goal that was picked at random."""
    _, path = drawn(tmp_path, "rooms", "--pair")
    assert load_world(path).tasks[0].goal_tolerance[1] == pytest.approx(math.pi)


def test_the_world_drawn_into_keeps_its_robot_and_gains_the_goal(tmp_path, capsys):
    """The car stays a car, with its own controller, moved to the start."""
    path = tmp_path / "out.toml"
    code = main(
        [
            "generate",
            "rooms",
            "--from",
            base_file(tmp_path),
            "--pair",
            "--out",
            str(path),
        ]
    )
    assert code == 0
    world = load_world(path)
    assert len(world.robots) == 1
    assert world.robots[0].name == "car" and world.robots[0].model == "ackermann"
    assert world.robots[0].controller == "demuvo.controllers:PurePursuit"
    assert world.robots[0].pose.translation.tolist() != [2.0, 2.0]  # moved to the start
    assert [t.robot for t in world.tasks] == ["car"]
    assert "for the robot of the base world" in capsys.readouterr().out


def test_a_task_the_world_already_had_makes_way_for_the_new_one(tmp_path):
    """One robot, one goal: the pair replaces what the base world asked for."""
    extra = '\n[[task]]\nrobot = "car"\ngoal = [18.0, 12.0, 0.0]\n'
    path = tmp_path / "out.toml"
    code = main(
        [
            "generate",
            "rooms",
            "--from",
            base_file(tmp_path, extra),
            "--pair",
            "--out",
            str(path),
        ]
    )
    assert code == 0
    tasks = load_world(path).tasks
    assert len(tasks) == 1 and tasks[0].goal.translation.tolist() != [18.0, 12.0]


def test_a_world_drawn_into_keeps_its_own_size(tmp_path):
    """The base world decides how big the world is, so it need not be said twice."""
    path = tmp_path / "out.toml"
    assert (
        main(["generate", "caves", "--from", base_file(tmp_path), "--out", str(path)])
        == 0
    )
    assert load_world(path).size == (20.0, 14.0)


def test_the_settings_of_a_style_are_options_of_its_own(tmp_path):
    """Each style takes what it takes, and a denser field leaves less room.

    The count of the shapes is no measure of it: more fill makes the blobs
    bigger and not more of them, and both of these come out as 29. What the
    setting changes is how much of the world is left to drive in.

    0.45 is not used here: for a robot of 0.25 m on a world this size it is too
    much clutter, and twenty fields of it all fail the check.
    """
    _, thin = drawn(tmp_path, "field", "--fill", "0.15", name="thin.toml")
    _, thick = drawn(tmp_path, "field", "--fill", "0.30", name="thick.toml")
    robot = Clearance(radius=0.25)
    room = [
        check_connectivity(load_world(path), robot).free_fraction
        for path in (thin, thick)
    ]
    assert room[0] > room[1]


def test_a_style_takes_only_the_settings_it_has(tmp_path, capsys):
    """A building has no fill, and saying so is an error and not a silence."""
    path = tmp_path / "out.toml"
    with pytest.raises(SystemExit):
        main(
            [
                "generate",
                "rooms",
                "--size",
                "20",
                "14",
                "--fill",
                "0.3",
                "--out",
                str(path),
            ]
        )
    assert "--fill" in capsys.readouterr().err


def test_boxes_instead_of_polygons(tmp_path):
    """The shapes a map is made of can be asked for."""
    _, path = drawn(tmp_path, "caves", "--pieces", "boxes")
    assert all(o.shape.name == "box" for o in load_world(path).obstacles)


def test_the_same_seed_writes_the_same_file(tmp_path):
    """A world file is the record of a map, so it must be the same one twice."""
    _, one = drawn(tmp_path, "caves", "--pair", name="one.toml")
    _, two = drawn(tmp_path, "caves", "--pair", name="two.toml")
    assert one.read_text() == two.read_text()


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "nothing says how big the robot is"),
        (["--radius", "0.2", "--fill", "1.0", "--attempts", "2"], "no map passed"),
        (["--radius", "0.2", "--pair", "--min-distance", "99"], "no two places"),
    ],
)
def test_what_cannot_be_drawn_is_said_and_not_written(tmp_path, capsys, extra, message):
    """Every refusal names what went wrong and leaves no file behind."""
    path = tmp_path / "out.toml"
    argv = ["generate", "caves", "--size", "10", "6", "--out", str(path), *extra]
    assert main(argv) == 1
    assert message in capsys.readouterr().err
    assert not path.exists()


def test_a_size_is_needed_when_there_is_no_world_to_draw_into(tmp_path, capsys):
    """Nothing says how big the world is, so nothing is drawn."""
    assert (
        main(["generate", "caves", "--radius", "0.2", "--out", str(tmp_path / "x")])
        == 1
    )
    assert "--size is needed" in capsys.readouterr().err


def test_a_size_beside_a_world_to_draw_into_is_refused(tmp_path, capsys):
    """Two answers to one question, so the command asks for one."""
    argv = [
        "generate",
        "caves",
        "--from",
        base_file(tmp_path),
        "--size",
        "9",
        "9",
        "--out",
        str(tmp_path / "x"),
    ]
    assert main(argv) == 1
    assert "belong to the world --from names" in capsys.readouterr().err


def test_a_file_that_cannot_be_written_is_said(tmp_path, capsys):
    """The map was drawn and there is nowhere to put it."""
    argv = [
        "generate",
        "caves",
        "--size",
        "10",
        "6",
        "--radius",
        "0.2",
        "--out",
        str(tmp_path / "no" / "such" / "x.toml"),
    ]
    assert main(argv) == 1
    assert "demuvo generate:" in capsys.readouterr().err


def test_the_command_is_asked_for_by_name_and_leaves_the_rest_alone(capsys):
    """`demuvo generate` is a command of its own; `demuvo world.toml` is not it."""
    with pytest.raises(SystemExit) as raised:
        main(["generate"])  # a style is needed
    assert raised.value.code == 2
    assert "STYLE" in capsys.readouterr().err


def test_a_maze_takes_its_loops_from_the_command_line(tmp_path):
    """The style's own option reaches the style, and more loops open more walls."""
    _, plain = drawn(tmp_path, "maze", name="plain.toml")
    _, looped = drawn(tmp_path, "maze", "--loops", "0.5", name="looped.toml")
    assert len(load_world(looped).obstacles) != len(load_world(plain).obstacles)


def test_a_pair_seed_moves_the_start_and_goal_and_leaves_the_map(tmp_path):
    """The map is drawn from --seed and the pair from --pair-seed."""
    _, one = drawn(tmp_path, "rooms", "--pair", name="one.toml")
    _, two = drawn(tmp_path, "rooms", "--pair", "--pair-seed", "9", name="two.toml")
    first, second = load_world(one), load_world(two)
    assert first.obstacles == second.obstacles
    assert first.robots[0].pose != second.robots[0].pose
    _, same = drawn(tmp_path, "rooms", "--pair", "--pair-seed", "5", name="same.toml")
    assert load_world(same) == first
