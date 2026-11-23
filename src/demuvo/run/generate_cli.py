"""The `demuvo generate` command - draw a world from a seed and write it to a file.

The Python API of `demuvo.generate` is the contract; this makes it usable without
writing a script. One style is one subcommand, so `demuvo generate rooms --help`
lists what a building takes and nothing else.

A map is written where it was asked for and nowhere else: `--out` is required. What
the map is for is said by `--radius`, or by the robots of the world `--from` names,
since no style draws for a robot nobody named.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import fields, replace

from robomath import Transform2D

from ..generate import (
    CaveStyle,
    Clearance,
    FieldStyle,
    Generated,
    MazeStyle,
    RoomsStyle,
    generate_caves,
    generate_field,
    generate_maze,
    generate_rooms,
    place_pair,
)
from ..world import RobotSpec, Task, World, WorldError, load_world, save_world
from ..world.shapes import Circle

# The style, the settings class and the function of each subcommand.
STYLES = {
    "field": (FieldStyle, generate_field),
    "rooms": (RoomsStyle, generate_rooms),
    "caves": (CaveStyle, generate_caves),
    "maze": (MazeStyle, generate_maze),
}
# What a robot invented for a map is given, so the file runs as it is written.
PLANNER = "demuvo.planners:AStar"
CONTROLLER = "demuvo.controllers:PurePursuit"
CONTROLLER_PARAMS = {"speed": 0.5, "lookahead": 0.6}
# A goal is a place and not a heading, so any heading counts as reaching it.
GOAL_TOLERANCE = (0.1, math.pi)


def _common() -> argparse.ArgumentParser:
    """The arguments every style takes."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--out", required=True, metavar="FILE", help="where to write")
    parent.add_argument(
        "--size",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        help="the world in metres; the base world's size with --from",
    )
    parent.add_argument(
        "--from",
        dest="base",
        metavar="FILE",
        help="a world file to draw into, whose robots, tasks and settings are kept "
        "and whose widest robot the map is drawn for",
    )
    parent.add_argument("--seed", type=int, default=1, help="the seed (default 1)")
    parent.add_argument(
        "--radius",
        type=float,
        help="inscribed radius in metres of the robot the map is for; needed "
        "unless --from brings one",
    )
    parent.add_argument(
        "--margin", type=float, default=0.0, help="metres of room beyond the radius"
    )
    parent.add_argument(
        "--origin",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="the lower-left corner of the world (default 0 0)",
    )
    parent.add_argument("--resolution", type=float, help="side of a cell in metres")
    parent.add_argument(
        "--pieces",
        choices=("polygons", "boxes"),
        help="trace each body as one polygon, or tile them with boxes",
    )
    parent.add_argument(
        "--attempts", type=int, default=20, help="maps to draw before giving up"
    )
    parent.add_argument(
        "--pair",
        action="store_true",
        help="place a start and a goal at random in the free space, which the "
        "check has already found to be one piece. The first robot of the world "
        "is moved to the start and given the goal; a world with no robot gets a "
        f"round one of --radius with {PLANNER} and {CONTROLLER}",
    )
    parent.add_argument(
        "--min-distance",
        type=float,
        help="the least a start and a goal may be apart (default half the world "
        "corner to corner)",
    )
    return parent


def _add(command: argparse.ArgumentParser, style, skip=()) -> None:
    """Give a style's own settings as options, one for each of its fields."""
    helps = {
        "fill": "share of the world that starts blocked, 0 to 1",
        "scale": "size of the largest features in metres",
        "octaves": "how many finer lattices are added",
        "persistence": "the weight of each octave against the one before",
        "smooth": "passes of the majority rule",
        "min_area": "smallest body drawn, in square metres",
        "min_pocket": "pockets smaller than this are filled (default: the robot's)",
        "border": "metres kept clear, or of rock, round the edge",
        "corridor": "width of a corridor in metres (default: the robot's)",
        "min_room": "shortest side of a room in metres (default: three corridors)",
        "wall": "thickness of a wall in metres",
        "depth": "how many times the building is cut in two",
        "loops": "share of the inner walls opened to make loops, 0 to 1",
    }
    whole = ("octaves", "smooth", "depth")
    for item in fields(style):
        if item.name in skip or item.name not in helps:
            continue
        kind = int if item.name in whole else float
        command.add_argument(
            f"--{item.name.replace('_', '-')}", type=kind, help=helps[item.name]
        )


def parser() -> argparse.ArgumentParser:
    """The command line of `demuvo generate`."""
    found = argparse.ArgumentParser(
        prog="demuvo generate",
        description="Draw a world from a seed and write it to a file.",
    )
    styles = found.add_subparsers(dest="style", required=True, metavar="STYLE")
    common = _common()
    for name, (settings, _) in STYLES.items():
        one = styles.add_parser(
            name,
            parents=[common],
            help={
                "field": "clutter from a smooth random field",
                "rooms": "a building of rooms joined by corridors",
                "caves": "caverns worn out of noise",
                "maze": "a maze of corridors, with loops if asked",
            }[name],
        )
        _add(one, settings, skip=("resolution", "pieces"))
    return found


def _style(args):
    """The settings of the style, with only what the command line asked for."""
    settings, _ = STYLES[args.style]
    given = {
        item.name: getattr(args, item.name)
        for item in fields(settings)
        if getattr(args, item.name, None) is not None
    }
    return settings(**given)


def _size_and_origin(args, base: World | None):
    """The size and origin to draw at, which the base world decides if there is one.

    Raises:
        WorldError: if neither says how big the world is, or if both do.
    """
    if base is not None:
        if args.size is not None or args.origin is not None:
            raise WorldError(
                "--size and --origin belong to the world --from names; leave them out"
            )
        return base.size, base.origin
    if args.size is None:
        raise WorldError("--size is needed, unless --from brings a world to draw into")
    return (args.size[0], args.size[1]), tuple(args.origin or (0.0, 0.0))


def _with_pair(got: Generated, args) -> World:
    """The world with a start and a goal placed at random in its free space.

    The first robot of the world is moved to the start and given the goal, its own
    tasks making way; a world with no robot of its own gets one that can drive, so
    that the file runs as it is written.
    """
    world = got.world
    least = args.min_distance
    if least is None:
        least = math.hypot(*world.size) / 2
    start, goal = place_pair(got.connectivity, args.seed, min_distance=least)
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    pose = Transform2D.from_xyt([start[0], start[1], heading])
    where = Transform2D.from_xyt([goal[0], goal[1], 0.0])
    if world.robots:
        first = world.robots[0]
        robots = (replace(first, pose=pose), *world.robots[1:])
    else:
        first = RobotSpec(
            "r1",
            "diff_drive",
            Circle(args.radius),
            pose,
            planner=PLANNER,
            controller=CONTROLLER,
            controller_params=dict(CONTROLLER_PARAMS),
        )
        robots = (first,)
    kept = tuple(task for task in world.tasks if task.robot != first.name)
    task = Task(first.name, where, goal_tolerance=GOAL_TOLERANCE)
    return replace(world, robots=robots, tasks=(*kept, task))


def _report(got: Generated, world: World, args, argv: list[str]) -> None:
    """Say what was drawn, how it measures, and the command that draws it again."""
    found = got.connectivity
    robot = f"{(args.radius if args.radius is not None else 0.0):.2f} m"
    if args.radius is None and world.robots:
        robot = "the robot of the base world"
    print(f"{args.out}: {args.style}, seed {args.seed}, {len(world.obstacles)} shapes")
    print(f"  drawn on attempt {got.attempt + 1} of {args.attempts}, for {robot}")
    print(
        f"  free {found.free_fraction:.0%} of the world, {found.components} region"
        f"{'' if found.components == 1 else 's'}, the largest holding "
        f"{found.main_fraction:.0%} of it"
    )
    if args.pair and world.tasks:
        start = world.robots[0].pose.translation.tolist()
        goal = world.tasks[-1].goal.translation.tolist()
        print(
            f"  start ({start[0]:.2f}, {start[1]:.2f}) and goal "
            f"({goal[0]:.2f}, {goal[1]:.2f}), {math.dist(start, goal):.2f} m apart"
        )
    print(f"# demuvo generate {' '.join(argv)}")


def main(argv: list[str]) -> int:
    """Draw a world and write it.

    Args:
        argv: the arguments after `generate`.

    Returns:
        0 when the file was written, 1 when nothing could be drawn or written.
    """
    args = parser().parse_args(argv)
    try:
        base = load_world(args.base) if args.base else None
        size, origin = _size_and_origin(args, base)
        clearance = Clearance(radius=args.radius, margin=args.margin)
        _, draw = STYLES[args.style]
        got = draw(
            size,
            args.seed,
            _style(args),
            clearance,
            origin=origin,
            base=base,
            attempts=args.attempts,
        )
        world = _with_pair(got, args) if args.pair else got.world
        save_world(world, args.out)
    except (OSError, ValueError) as error:
        print(f"demuvo generate: {error}", file=sys.stderr)
        return 1
    _report(got, world, args, argv)
    return 0
