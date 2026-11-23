"""Tests for the maze style: the walk, the loops, the corridors and the robot."""

# The rooms style is held to the same contract, so some tests read alike on purpose.
# pylint: disable=duplicate-code

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.costmap import build_costmaps
from demuvo.generate import (
    Clearance,
    GenerateError,
    MazeStyle,
    check_connectivity,
    generate_maze,
    maze_corridor,
)
from demuvo.planners import AStar
from demuvo.world import (
    Obstacle,
    Task,
    WorldError,
    load_world,
    save_world,
)
from demuvo.world.regions import components
from demuvo.world.shapes import Box

from .worlds import ROBOT, SIZE, a_robot, base_world


def test_a_seed_gives_the_same_maze_every_time():
    """Two calls with equal arguments agree, wall for wall."""
    one, two = (
        generate_maze(SIZE, 5, clearance=ROBOT),
        generate_maze(SIZE, 5, clearance=ROBOT),
    )
    assert one.world == two.world
    assert np.array_equal(one.mask, two.mask)


def test_another_seed_gives_another_maze():
    """The seed is what the walk is drawn from."""
    assert not np.array_equal(
        generate_maze(SIZE, 5, clearance=ROBOT).mask,
        generate_maze(SIZE, 6, clearance=ROBOT).mask,
    )


def test_a_maze_has_no_field_behind_it():
    """There is no continuous field to cut here, and the result says so."""
    assert generate_maze(SIZE, 1, clearance=ROBOT).field is None


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("pieces", ["polygons", "boxes"])
def test_a_maze_is_written_and_read_back_unchanged(tmp_path, seed, pieces):
    """Every wall passes the gate a world file passes; this is the proof."""
    got = generate_maze(SIZE, seed, MazeStyle(pieces=pieces), ROBOT)
    path = tmp_path / "maze.toml"
    save_world(got.world, path)
    assert load_world(path) == got.world


def test_a_maze_with_no_loops_has_one_route_between_two_places():
    """The walls of a perfect maze are one piece, so no wall is an island."""
    for seed in range(5):
        got = generate_maze(SIZE, seed, MazeStyle(loops=0.0), ROBOT)
        assert len(components(got.mask)) == 1
        assert got.connectivity.ok


def test_loops_leave_walls_standing_alone():
    """Opening a wall in a ring of corridors cuts a piece of wall free."""
    got = generate_maze(SIZE, 5, MazeStyle(loops=0.4), ROBOT)
    assert len(components(got.mask)) > 1
    assert got.connectivity.ok


def test_more_loops_open_more_of_the_world():
    """The share of walls opened shows in the free space."""
    free = [
        (~generate_maze(SIZE, 5, MazeStyle(loops=share), ROBOT).mask).sum()
        for share in (0.0, 0.3, 1.0)
    ]
    assert free[0] < free[1] < free[2]


def test_the_corridors_are_as_wide_as_the_robot_needs():
    """Body across and three cells of slack each side; no width written in metres."""
    style = MazeStyle(resolution=0.1)
    assert maze_corridor(style, 0.25) == pytest.approx(1.1)
    assert maze_corridor(style, 0.5) == pytest.approx(1.6)
    assert maze_corridor(MazeStyle(corridor=0.9), 0.5) == 0.9


def test_a_maze_fits_the_robot_it_was_drawn_for_and_not_a_much_bigger_one():
    """Sizing by the robot is the whole point, so a bigger body does not fit."""
    got = generate_maze(SIZE, 7, clearance=Clearance(radius=0.2))
    assert check_connectivity(got.world, Clearance(radius=0.2)).ok
    assert not check_connectivity(got.world, Clearance(radius=0.9)).ok


def test_the_robot_of_the_base_world_sizes_the_corridors():
    """Nothing need be said twice: the world already carries its robot."""
    small = generate_maze(SIZE, 3, base=base_world(a_robot(radius=0.2)))
    large = generate_maze(SIZE, 3, base=base_world(a_robot(radius=0.6)))
    assert (~large.mask).sum() != (~small.mask).sum()
    assert large.connectivity.ok and small.connectivity.ok


def test_a_maze_is_refused_when_nothing_says_how_big_the_robot_is():
    """A corridor cannot be sized by a robot nobody named."""
    with pytest.raises(WorldError, match="nothing says how big the robot is"):
        generate_maze(SIZE, 1)


def test_a_corridor_too_narrow_for_the_robot_is_refused_before_any_work():
    """No maze of it could pass the check, so it is not drawn twenty times."""
    with pytest.raises(WorldError, match="has no room for a robot that needs"):
        generate_maze(SIZE, 1, MazeStyle(corridor=0.3), Clearance(radius=0.25))


def test_the_outer_wall_is_solid_all_the_way_round():
    """A maze is shut: its outer wall is the border of the world."""
    got = generate_maze(SIZE, 4, MazeStyle(wall=0.3), ROBOT)
    rings = 3  # 0.3 m of wall at 0.1 m cells
    assert got.mask[:rings, :].all() and got.mask[-rings:, :].all()
    assert got.mask[:, :rings].all() and got.mask[:, -rings:].all()


def test_the_robots_and_the_goals_of_the_base_world_are_joined_to_the_maze():
    """A robot never starts inside a wall, and its goal is in the same free space."""
    base = base_world(
        a_robot(), tasks=(Task("r1", Transform2D.from_xyt([18.0, 12.0, 0.0])),)
    )
    got = generate_maze(SIZE, 8, base=base, anchors=((10.0, 7.0),))
    assert got.connectivity.anchors == (0, 0, 0)
    assert got.connectivity.ok


def test_the_walls_are_named_in_order():
    """The window lists them by name, so the names must not jump about."""
    got = generate_maze(SIZE, 2, clearance=ROBOT)
    names = [o.name for o in got.world.obstacles]
    assert names == [f"wall {i}" for i in range(1, len(names) + 1)]


def test_a_base_world_that_already_uses_the_names_is_refused():
    """Two obstacles of one name is a world file error, so it is caught here."""
    taken = Obstacle(Transform2D.from_xyt([10.0, 7.0, 0.0]), Box((0.5, 0.5)), "wall 1")
    with pytest.raises(WorldError, match="already has an obstacle called 'wall 1'"):
        generate_maze(SIZE, 1, base=base_world(obstacles=(taken,)))


def test_a_world_with_no_room_for_a_cell_is_refused_after_the_last_attempt():
    """Walls thicker than the world leave it solid, and the message says why."""
    with pytest.raises(GenerateError, match="in 2 attempts"):
        generate_maze((1.0, 1.0), 1, MazeStyle(wall=0.6), ROBOT, attempts=2)


def test_asking_for_no_clearance_keeps_the_first_maze():
    """A caller that wants the map anyway gets it, and the report still says."""
    got = generate_maze((1.0, 1.0), 1, MazeStyle(wall=0.6), clearance=None)
    assert got.attempt == 0 and not got.connectivity.ok


def test_a_world_that_reaches_west_and_south_is_built_where_it_lies():
    """The origin moves the maze, and no wall leaves the world."""
    got = generate_maze(SIZE, 3, clearance=ROBOT, origin=(-10.0, -7.0))
    assert got.world.origin == (-10.0, -7.0)
    for obstacle in got.world.obstacles:
        for x, y in obstacle.shape.points(obstacle.pose):
            assert -10.0 <= x <= 10.0 and -7.0 <= y <= 7.0


def test_a_robot_can_plan_its_way_through_a_maze():
    """What the check calls connected, A* can plan; that is what it is for."""
    base = base_world(
        a_robot(), tasks=(Task("r1", Transform2D.from_xyt([18.0, 12.0, 0.0])),)
    )
    got = generate_maze(SIZE, 3, base=base)
    planner = AStar()
    planner.configure(build_costmaps(got.world)["r1"])
    path = planner.plan(
        Transform2D.from_xyt([1.5, 1.5, 0.0]), Transform2D.from_xyt([18.0, 12.0, 0.0])
    )
    assert len(path) > 0


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"resolution": 0.0}, "resolution must be a positive number"),
        ({"wall": -0.1}, "wall must be a positive number"),
        ({"corridor": 0.0}, "corridor must be a positive number"),
        ({"loops": -0.1}, "loops must be from 0 to 1"),
        ({"loops": 1.5}, "loops must be from 0 to 1"),
        ({"pieces": "circles"}, "pieces must be one of polygons, boxes"),
    ],
)
def test_a_style_that_cannot_be_is_refused(settings, message):
    """Each setting is checked when the style is made, before any work."""
    with pytest.raises(ValueError, match=message):
        MazeStyle(**settings)


@pytest.mark.parametrize("settings", [{"attempts": 0}, {"attempts": 1.5}])
def test_an_argument_that_cannot_be_is_refused(settings):
    """The call itself is checked too."""
    with pytest.raises(ValueError, match="attempts must be 1 or more"):
        generate_maze(SIZE, 1, clearance=ROBOT, **settings)
