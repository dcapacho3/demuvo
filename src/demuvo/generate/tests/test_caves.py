"""Tests for the caves style: the rule, the scale it is run at, and the one cave."""

# These read like the rooms tests because the styles answer to one contract, and a
# style that did not would be the bug.
# pylint: disable=duplicate-code

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.costmap import build_costmaps
from demuvo.generate import (
    CaveStyle,
    Clearance,
    GenerateError,
    cavern,
    check_connectivity,
    generate_caves,
    lane,
    numbered,
    within,
)
from demuvo.generate.carve import majority
from demuvo.planners import AStar
from demuvo.sensing import stream
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
# The robot a cave is worn for when no base world carries one.
ROBOT = Clearance(radius=0.25)


def a_robot(name="r1", x=1.5, y=1.5, radius=0.25):
    """A round robot at a place."""
    return RobotSpec(
        name, "diff_drive", Circle(radius), Transform2D.from_xyt([x, y, 0.0])
    )


def base_world(*robots, tasks=(), obstacles=(), size=SIZE):
    """A world for a cave to be worn into."""
    return World(
        size=size,
        seed=1,
        obstacles=tuple(obstacles),
        robots=tuple(robots) or (a_robot(),),
        tasks=tuple(tasks),
    )


def patches(mask, resolution, reach):
    """How many separate patches of ground a robot of this reach can stand on."""
    _, sizes = numbered(~within(mask, resolution, reach))
    return len(sizes)


def test_a_seed_gives_the_same_cave_every_time():
    """Two calls with equal arguments agree, rock for rock."""
    one = generate_caves(SIZE, 5, clearance=ROBOT)
    two = generate_caves(SIZE, 5, clearance=ROBOT)
    assert one.world == two.world
    assert np.array_equal(one.mask, two.mask)


def test_another_seed_gives_another_cave():
    """The seed is what the rock is tossed from."""
    assert not np.array_equal(
        generate_caves(SIZE, 5, clearance=ROBOT).mask,
        generate_caves(SIZE, 6, clearance=ROBOT).mask,
    )


def test_a_cave_has_no_field_behind_it():
    """The first toss of each cell is noise, not a field to be cut."""
    assert generate_caves(SIZE, 1, clearance=ROBOT).field is None


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("pieces", ["polygons", "boxes"])
def test_a_cave_is_written_and_read_back_unchanged(tmp_path, seed, pieces):
    """Every body of rock passes the gate a world file passes; this is the proof."""
    got = generate_caves(SIZE, seed, CaveStyle(pieces=pieces), ROBOT)
    path = tmp_path / "caves.toml"
    save_world(got.world, path)
    assert load_world(path) == got.world


def test_the_rule_is_run_at_the_scale_of_the_robots_own_lane():
    """Tossed at the map's scale the rule strands the robot; at its lane it does not.

    This is the whole of the style. On a grid of 0.1 m cells the rule leaves
    caverns joined by cracks the robot cannot pass, and the ground it can stand on
    falls into scores of patches. On a grid whose cell is the robot's lane, the
    narrowest thing the rule can draw is a way the robot fits through, and the
    ground it can stand on is one patch.
    """
    rows, cols, resolution, reach = 140, 200, 0.1, 0.25
    rng = stream(1, "generate", "caves", "0")
    fine = rng.random((rows, cols)) < 0.5
    for _ in range(4):
        fine = majority(fine)
    assert patches(fine, resolution, reach) > 20

    worn = cavern(stream(1, "generate", "caves", "0"), rows, cols, CaveStyle(), reach)
    assert patches(worn, resolution, reach) == 1


def test_only_the_one_cave_the_robot_can_move_about_in_is_left():
    """A pocket it could never reach is rock, whatever the dice said."""
    got = generate_caves(SIZE, 11, clearance=ROBOT)
    assert patches(got.mask, got.resolution, ROBOT.reach) == 1
    assert got.connectivity.ok


def test_a_cave_fits_the_robot_it_was_worn_for_and_not_a_much_bigger_one():
    """Sizing the ways through by the robot is the point, so a bigger body sticks."""
    got = generate_caves(SIZE, 7, clearance=Clearance(radius=0.2))
    assert check_connectivity(got.world, Clearance(radius=0.2)).ok
    assert not check_connectivity(got.world, Clearance(radius=0.9)).ok


def test_a_bigger_robot_gets_wider_ways_through():
    """A cave cell is one lane, so every way through grows with the body."""
    assert lane(0.2, 0.1) == pytest.approx(0.6)
    assert lane(0.5, 0.1) == pytest.approx(1.2)
    small = generate_caves(SIZE, 2, clearance=Clearance(radius=0.2))
    large = generate_caves(SIZE, 2, clearance=Clearance(radius=0.6))
    assert patches(small.mask, small.resolution, 0.2) == 1
    assert patches(large.mask, large.resolution, 0.6) == 1


def test_the_robot_of_the_base_world_wears_the_cave():
    """Nothing need be said twice: the world already carries its robot."""
    got = generate_caves(SIZE, 3, base=base_world(a_robot(radius=0.4)))
    assert got.connectivity.ok
    assert patches(got.mask, got.resolution, 0.4) == 1


def test_a_cave_is_refused_when_nothing_says_how_big_the_robot_is():
    """A way through cannot be sized by a robot nobody named."""
    with pytest.raises(WorldError, match="nothing says how big the robot is"):
        generate_caves(SIZE, 1)


def test_the_robots_and_the_goals_of_the_base_world_are_tunnelled_into_the_cave():
    """A robot never starts inside the rock because the dice put the cave elsewhere."""
    base = base_world(
        a_robot(), tasks=(Task("r1", Transform2D.from_xyt([18.0, 12.0, 0.0])),)
    )
    got = generate_caves(SIZE, 8, base=base, anchors=((10.0, 7.0),))
    assert got.connectivity.anchors == (0, 0, 0)
    assert got.connectivity.ok


def test_the_edge_of_the_world_is_solid_rock():
    """A cave is shut in, so its border is not a way out."""
    got = generate_caves(SIZE, 4, CaveStyle(border=0.3), ROBOT)
    rings = 3  # 0.3 m at 0.1 m cells
    assert got.mask[:rings, :].all() and got.mask[-rings:, :].all()
    assert got.mask[:, :rings].all() and got.mask[:, -rings:].all()


def test_a_world_of_no_rock_is_one_open_hall():
    """A fill of nothing leaves the border and nothing else."""
    got = generate_caves(SIZE, 1, CaveStyle(fill=0.0), ROBOT)
    assert got.connectivity.ok and got.connectivity.components == 1
    assert got.connectivity.free_fraction > 0.8


def test_a_world_of_nothing_but_rock_is_refused():
    """Solid rock is no cave, and the message says the robot can stand nowhere."""
    with pytest.raises(GenerateError, match="can stand on 0%"):
        generate_caves(SIZE, 1, CaveStyle(fill=1.0), ROBOT, attempts=2)


def test_asking_for_no_clearance_keeps_the_first_cave():
    """A caller that wants the map anyway gets it, and the report still says."""
    got = generate_caves(SIZE, 1, CaveStyle(fill=1.0), clearance=None)
    assert got.attempt == 0 and not got.connectivity.ok


def test_the_rocks_are_named_in_order():
    """The window lists them by name, so the names must not jump about."""
    got = generate_caves(SIZE, 2, clearance=ROBOT)
    names = [o.name for o in got.world.obstacles]
    assert names == [f"rock {i}" for i in range(1, len(names) + 1)]


def test_a_base_world_that_already_uses_the_names_is_refused():
    """Two obstacles of one name is a world file error, so it is caught here."""
    taken = Obstacle(Transform2D.from_xyt([10.0, 7.0, 0.0]), Box((0.5, 0.5)), "rock 1")
    with pytest.raises(WorldError, match="already has an obstacle called 'rock 1'"):
        generate_caves(SIZE, 1, base=base_world(obstacles=(taken,)))


def test_a_base_world_of_another_size_is_refused():
    """The size and the origin belong to the base world, and are not guessed."""
    with pytest.raises(WorldError, match="leave the size and the origin"):
        generate_caves((4.0, 4.0), 1, base=base_world())


def test_a_world_that_reaches_west_and_south_is_worn_where_it_lies():
    """The origin moves the cave, and no rock leaves the world."""
    got = generate_caves(SIZE, 3, clearance=ROBOT, origin=(-10.0, -7.0))
    assert got.world.origin == (-10.0, -7.0)
    for obstacle in got.world.obstacles:
        for x, y in obstacle.shape.points(obstacle.pose):
            assert -10.0 <= x <= 10.0 and -7.0 <= y <= 7.0


def test_a_robot_can_plan_its_way_through_a_cave():
    """What the check calls connected, A* can plan; that is what it is for."""
    base = base_world(
        a_robot(), tasks=(Task("r1", Transform2D.from_xyt([18.0, 12.0, 0.0])),)
    )
    got = generate_caves(SIZE, 3, base=base)
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
        ({"border": -0.1}, "border must be a positive number"),
        ({"fill": 1.5}, "fill must be from 0 to 1"),
        ({"smooth": -1}, "smooth must be 0 or more"),
        ({"smooth": 1.5}, "smooth must be a whole number"),
        ({"pieces": "circles"}, "pieces must be one of polygons, boxes"),
    ],
)
def test_a_style_that_cannot_be_is_refused(settings, message):
    """Each setting is checked when the style is made, before any work."""
    with pytest.raises(ValueError, match=message):
        CaveStyle(**settings)


@pytest.mark.parametrize("settings", [{"attempts": 0}, {"attempts": 1.5}])
def test_an_argument_that_cannot_be_is_refused(settings):
    """The call itself is checked too."""
    with pytest.raises(ValueError, match="attempts must be 1 or more"):
        generate_caves(SIZE, 1, clearance=ROBOT, **settings)


@pytest.mark.parametrize("size", [(0.0, 5.0), (5.0, -1.0)])
def test_a_world_of_no_size_is_refused(size):
    """A world must have room to wear a cave into."""
    with pytest.raises(ValueError, match="size must be two lengths above 0"):
        generate_caves(size, 1, clearance=ROBOT)
