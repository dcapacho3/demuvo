"""Tests for the rooms-and-corridors style: the cuts, the corridors and the robot."""

# pylint: disable=protected-access  # some tests check the style's own lengths

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.costmap import build_costmaps
from demuvo.generate import (
    Clearance,
    GenerateError,
    RoomsStyle,
    check_connectivity,
    corridor_width,
    generate_rooms,
)
from demuvo.generate.rooms import _plan
from demuvo.planners import AStar
from demuvo.world import (
    Obstacle,
    RobotSpec,
    Task,
    World,
    WorldError,
    load_world,
    save_world,
)
from demuvo.world.shapes import Box, Circle

SIZE = (20.0, 14.0)
# The robot a building is laid out for when no base world carries one.
ROBOT = Clearance(radius=0.25)


def a_robot(name="r1", x=1.5, y=1.5, radius=0.2):
    """A round robot at a place."""
    return RobotSpec(
        name, "diff_drive", Circle(radius), Transform2D.from_xyt([x, y, 0.0])
    )


def base_world(*robots, tasks=(), obstacles=(), size=SIZE):
    """A world for a building to be put into."""
    return World(
        size=size,
        seed=1,
        obstacles=tuple(obstacles),
        robots=tuple(robots) or (a_robot(),),
        tasks=tuple(tasks),
    )


def test_a_seed_gives_the_same_building_every_time():
    """Two calls with equal arguments agree, wall for wall."""
    one, two = (
        generate_rooms(SIZE, 5, clearance=ROBOT),
        generate_rooms(SIZE, 5, clearance=ROBOT),
    )
    assert one.world == two.world
    assert np.array_equal(one.mask, two.mask)


def test_another_seed_gives_another_building():
    """The seed is what the layout is drawn from."""
    assert not np.array_equal(
        generate_rooms(SIZE, 5, clearance=ROBOT).mask,
        generate_rooms(SIZE, 6, clearance=ROBOT).mask,
    )


def test_a_building_has_no_field_behind_it():
    """There is no continuous field to cut here, and the result says so."""
    assert generate_rooms(SIZE, 1, clearance=ROBOT).field is None


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("pieces", ["polygons", "boxes"])
def test_a_building_is_written_and_read_back_unchanged(tmp_path, seed, pieces):
    """Every wall passes the gate a world file passes; this is the proof."""
    got = generate_rooms(SIZE, seed, RoomsStyle(pieces=pieces), ROBOT)
    path = tmp_path / "rooms.toml"
    save_world(got.world, path)
    assert load_world(path) == got.world


def test_the_corridors_are_as_wide_as_the_robot_needs():
    """Body across, and a cell of slack on each side; no width written in metres."""
    style = RoomsStyle(resolution=0.1)
    assert corridor_width(style, 0.15) == pytest.approx(0.5)
    assert corridor_width(style, 0.25) == pytest.approx(0.7)
    assert corridor_width(style, 0.5) == pytest.approx(1.2)
    assert corridor_width(RoomsStyle(corridor=0.9), 0.5) == 0.9


def test_every_length_of_a_building_is_cut_from_the_robot():
    """Corridor, door and smallest room all follow the body; none is a fixed number.

    A bigger robot does not simply give more free space: its corridors are wider,
    but its rooms must be bigger too, so fewer cuts fit in the same world. What
    grows with the robot is every length the building is laid out by.
    """
    style = RoomsStyle(resolution=0.1)
    small, large = _plan(style, 0.2), _plan(style, 0.5)
    assert small.corridor == 6 and large.corridor == 12  # body and a cell each side
    assert small.room == 18 and large.room == 36  # three corridors across
    assert large.least > small.least  # so a piece must be bigger before it is cut
    assert _plan(RoomsStyle(resolution=0.1, min_room=1.0), 0.2).room == 10


def test_a_building_fits_the_robot_it_was_drawn_for_and_not_a_much_bigger_one():
    """Sizing by the robot is the whole point, so a bigger body does not fit."""
    got = generate_rooms(SIZE, 7, clearance=Clearance(radius=0.2))
    assert check_connectivity(got.world, Clearance(radius=0.2)).ok
    assert not check_connectivity(got.world, Clearance(radius=0.9)).ok


def test_the_robot_of_the_base_world_sizes_the_corridors():
    """Nothing need be said twice: the world already carries its robot."""
    small = generate_rooms(SIZE, 3, base=base_world(a_robot(radius=0.2)))
    large = generate_rooms(SIZE, 3, base=base_world(a_robot(radius=0.6)))
    assert (~large.mask).sum() > (~small.mask).sum()
    assert large.connectivity.ok and small.connectivity.ok


def test_the_widest_robot_of_the_base_world_decides():
    """A map the car can cross, the small robot can cross too."""
    mixed = base_world(a_robot("small", radius=0.2), a_robot("big", 3.0, 1.5, 0.6))
    assert generate_rooms(SIZE, 3, base=mixed).connectivity.ok


def test_a_building_is_refused_when_nothing_says_how_big_the_robot_is():
    """A corridor cannot be sized by a robot nobody named."""
    with pytest.raises(WorldError, match="nothing says how big the robot is"):
        generate_rooms(SIZE, 1)


def test_a_corridor_too_narrow_for_the_robot_is_refused_before_any_work():
    """No building of it could pass the check, so it is not drawn twenty times."""
    with pytest.raises(WorldError, match="has no room for a robot that needs"):
        generate_rooms(SIZE, 1, RoomsStyle(corridor=0.3), Clearance(radius=0.25))


def test_the_outer_wall_is_solid_all_the_way_round():
    """A building is shut: its outer wall is the border of the world."""
    got = generate_rooms(SIZE, 4, RoomsStyle(wall=0.3), ROBOT)
    rings = 3  # 0.3 m of wall at 0.1 m cells
    assert got.mask[:rings, :].all() and got.mask[-rings:, :].all()
    assert got.mask[:, :rings].all() and got.mask[:, -rings:].all()


def test_the_robots_and_the_goals_of_the_base_world_are_joined_to_the_building():
    """A robot never starts inside a wall, and its goal is in the same free space."""
    base = base_world(
        a_robot(), tasks=(Task("r1", Transform2D.from_xyt([18.0, 12.0, 0.0])),)
    )
    got = generate_rooms(SIZE, 8, base=base, anchors=((10.0, 7.0),))
    assert got.connectivity.anchors == (0, 0, 0)
    assert got.connectivity.ok


def test_a_building_with_no_cuts_is_one_room():
    """Depth 0 leaves the building whole, which is still a world to drive in."""
    got = generate_rooms(SIZE, 1, RoomsStyle(depth=0), ROBOT)
    free = ~got.mask
    rows = np.flatnonzero(free.any(axis=1))
    cols = np.flatnonzero(free.any(axis=0))
    # the one room is a filled rectangle
    assert free[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].all()


def test_every_room_holds_the_smallest_room_asked_for():
    """A cut that would leave a piece too small for a room is not made."""
    got = generate_rooms(SIZE, 9, RoomsStyle(depth=6, min_room=2.0), ROBOT)
    free = ~got.mask
    # no free run of a row is shorter than a corridor, and the rooms are 2 m across
    widest = max(int(free[r].sum()) for r in range(free.shape[0]))
    assert widest >= 20  # 2.0 m at 0.1 m cells


def test_the_walls_are_named_in_order():
    """The window lists them by name, so the names must not jump about."""
    got = generate_rooms(SIZE, 2, clearance=ROBOT)
    names = [o.name for o in got.world.obstacles]
    assert names == [f"wall {i}" for i in range(1, len(names) + 1)]


def test_a_base_world_that_already_uses_the_names_is_refused():
    """Two obstacles of one name is a world file error, so it is caught here."""
    taken = Obstacle(Transform2D.from_xyt([10.0, 7.0, 0.0]), Box((0.5, 0.5)), "wall 1")
    with pytest.raises(WorldError, match="already has an obstacle called 'wall 1'"):
        generate_rooms(SIZE, 1, base=base_world(obstacles=(taken,)))


def test_a_base_world_of_another_size_is_refused():
    """The size and the origin belong to the base world, and are not guessed."""
    with pytest.raises(WorldError, match="leave the size and the origin"):
        generate_rooms((4.0, 4.0), 1, base=base_world())


def test_a_world_with_no_room_for_a_building_is_refused_after_the_last_attempt():
    """Walls thicker than the world leave it solid, and the message says why."""
    with pytest.raises(GenerateError, match="in 2 attempts"):
        generate_rooms((1.0, 1.0), 1, RoomsStyle(wall=0.6), ROBOT, attempts=2)


def test_asking_for_no_clearance_keeps_the_first_building():
    """A caller that wants the map anyway gets it, and the report still says."""
    got = generate_rooms((1.0, 1.0), 1, RoomsStyle(wall=0.6), clearance=None)
    assert got.attempt == 0 and not got.connectivity.ok


def test_a_world_that_reaches_west_and_south_is_built_where_it_lies():
    """The origin moves the building, and no wall leaves the world."""
    got = generate_rooms(SIZE, 3, clearance=ROBOT, origin=(-10.0, -7.0))
    assert got.world.origin == (-10.0, -7.0)
    for obstacle in got.world.obstacles:
        for x, y in obstacle.shape.points(obstacle.pose):
            assert -10.0 <= x <= 10.0 and -7.0 <= y <= 7.0


def test_a_robot_can_plan_its_way_across_a_building():
    """What the check calls connected, A* can plan; that is what it is for."""
    base = base_world(
        a_robot(), tasks=(Task("r1", Transform2D.from_xyt([18.0, 12.0, 0.0])),)
    )
    got = generate_rooms(SIZE, 3, base=base)
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
        ({"min_room": -1.0}, "min_room must be a positive number"),
        ({"depth": -1}, "depth must be 0 or more"),
        ({"depth": 1.5}, "depth must be a whole number"),
        ({"pieces": "circles"}, "pieces must be one of polygons, boxes"),
    ],
)
def test_a_style_that_cannot_be_is_refused(settings, message):
    """Each setting is checked when the style is made, before any work."""
    with pytest.raises(ValueError, match=message):
        RoomsStyle(**settings)


@pytest.mark.parametrize("settings", [{"attempts": 0}, {"attempts": 1.5}])
def test_an_argument_that_cannot_be_is_refused(settings):
    """The call itself is checked too."""
    with pytest.raises(ValueError, match="attempts must be 1 or more"):
        generate_rooms(SIZE, 1, clearance=ROBOT, **settings)


@pytest.mark.parametrize("size", [(0.0, 5.0), (5.0, -1.0)])
def test_a_world_of_no_size_is_refused(size):
    """A world must have room to build in."""
    with pytest.raises(ValueError, match="size must be two lengths above 0"):
        generate_rooms(size, 1, clearance=ROBOT)
