"""Tests for the free-space check that every generated world must pass."""

import dataclasses

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.generate import (
    Clearance,
    Connectivity,
    check_connectivity,
    check_grid,
    clearance_for,
    lane,
    room,
    within,
)
from demuvo.world import Obstacle, RobotSpec, Task, World, WorldError
from demuvo.world.shapes import Box, Circle

SIZE = (6.0, 4.0)
# The robot a test measures for when its world has none to take one from.
ROBOT = Clearance(radius=0.2)


def box(x, y, length, width):
    """A box obstacle at a place."""
    return Obstacle(Transform2D.from_xyt([x, y, 0.0]), Box((length, width)), "")


def robot(name, x, y):
    """A small round robot at a place."""
    return RobotSpec(
        name, "diff_drive", Circle(0.15), Transform2D.from_xyt([x, y, 0.0])
    )


def world_of(*obstacles, robots=(), tasks=(), size=SIZE, origin=(0.0, 0.0)):
    """A world with these obstacles and robots and nothing else."""
    return World(
        size=size,
        seed=1,
        obstacles=tuple(obstacles),
        robots=tuple(robots),
        tasks=tuple(tasks),
        origin=origin,
    )


def test_an_empty_world_is_one_free_region_the_robot_fills():
    """Only the border takes space, and what is left is whole."""
    found = check_connectivity(world_of(), ROBOT)
    assert found.ok and not found.reasons
    assert found.components == 1 and found.main_fraction == 1.0
    # the border wall fills the outer cells, and the robot's centre stays 0.2 m
    # off them, so it may stand on the middle 5.5 by 3.5 m
    assert found.free_fraction == pytest.approx(5.5 * 3.5 / 24.0, abs=0.01)
    assert found.resolution == 0.05 and found.origin == (0.0, 0.0)


def test_a_wall_across_the_world_cuts_the_free_space_in_two():
    """Two rooms, and the report says which condition failed."""
    found = check_connectivity(world_of(box(3.0, 2.0, 0.2, 4.0)))
    assert not found.ok and found.components == 2
    assert found.main_fraction == pytest.approx(0.5, abs=0.05)
    assert "the largest of 2 free regions holds 50%" in found.reasons[0]
    assert "90% is asked for" in found.reasons[0]


def test_a_door_the_robot_fits_through_joins_the_rooms_and_a_narrow_one_does_not():
    """The check measures with the planner's ruler: the door must take the body."""
    radius = 0.2
    wide = world_of(box(3.0, 0.65, 0.2, 1.3), box(3.0, 3.35, 0.2, 1.3))
    narrow = world_of(box(3.0, 0.95, 0.2, 1.9), box(3.0, 3.05, 0.2, 1.9))
    clearance = Clearance(radius=radius)
    assert check_connectivity(wide, clearance).components == 1  # a 1.4 m door
    assert check_connectivity(narrow, clearance).components == 2  # a 0.2 m door


def test_a_robot_shut_in_a_small_room_is_not_in_the_largest_region():
    """The room is small enough that only the anchor fails."""
    walls = (
        box(1.0, 0.6, 1.2, 0.1),
        box(1.0, 1.8, 1.2, 0.1),
        box(0.45, 1.2, 0.1, 1.3),
        box(1.55, 1.2, 0.1, 1.3),
    )
    found = check_connectivity(world_of(*walls, robots=[robot("r1", 1.0, 1.2)]))
    assert not found.ok and found.anchors == (1,)
    assert found.reasons == (
        "robot 'r1' at (1.00, 1.20) is not in the largest free region",
    )


def test_a_robot_inside_an_obstacle_has_no_room_at_all():
    """A blocked anchor has no region, and the message says so."""
    found = check_connectivity(
        world_of(box(3.0, 2.0, 1.0, 1.0), robots=[robot("r1", 3.0, 2.0)])
    )
    assert not found.ok and found.anchors == (None,)
    assert found.reasons == ("robot 'r1' at (3.00, 2.00) has no room for the robot",)


def test_the_goal_of_a_task_and_a_loose_anchor_are_both_checked():
    """Every place that matters is an anchor, and each is named in its own way."""
    world = world_of(
        box(3.0, 2.0, 1.0, 1.0),
        robots=[robot("r1", 1.0, 1.0)],
        tasks=[Task("r1", Transform2D.from_xyt([3.0, 2.0, 0.0]))],
    )
    found = check_connectivity(world, Clearance(), ((3.0, 2.0),))
    assert found.anchors == (0, None, None)
    assert found.reasons == (
        "the goal of task 1 ('r1') at (3.00, 2.00) has no room for the robot",
        "anchor 1 at (3.00, 2.00) has no room for the robot",
    )


def test_a_world_with_almost_no_room_fails_the_free_share():
    """One lane through a solid block passes `min_main` and still is no world."""
    blocks = [
        box(x + 0.5, y + 0.5, 0.98, 0.98)
        for x in np.arange(0.0, 6.0, 1.0)
        for y in np.arange(0.0, 4.0, 1.0)
        if y != 1.0  # one lane of free ground across the middle
    ]
    found = check_connectivity(world_of(*blocks), ROBOT)
    assert not found.ok and found.main_fraction == 1.0
    assert "can stand on 14% of the world" in found.reasons[0]
    assert "20% is asked for" in found.reasons[0]


def test_a_point_robot_stands_where_a_wide_one_cannot():
    """With no radius there is no inflation, and only the obstacles are taken."""
    world = world_of(box(3.0, 2.0, 1.0, 1.0))
    point = check_connectivity(world, Clearance(radius=0.0))
    wide = check_connectivity(world, Clearance(radius=0.5))
    assert point.free_fraction > wide.free_fraction
    assert point.ok and point.components == 1


def test_a_margin_takes_as_much_room_as_a_radius():
    """Only their sum counts, since both keep the robot's centre off the wall.

    The two are the same room to within a cell: `0.2 + 0.1` is not `0.3` as a
    float, and a cell centre that lands on the line falls the other way.
    """
    world = world_of(box(3.0, 2.0, 1.0, 1.0))
    split = check_connectivity(world, Clearance(radius=0.2, margin=0.1))
    whole = check_connectivity(world, Clearance(radius=0.3))
    assert split.components == whole.components == 1
    assert split.free_fraction == pytest.approx(whole.free_fraction, abs=0.05)


def test_the_check_runs_at_the_resolution_it_is_given():
    """A coarser grid has fewer cells, and the report says which it used."""
    coarse = check_connectivity(world_of(), Clearance(resolution=0.1))
    fine = check_connectivity(world_of(), Clearance(resolution=0.05))
    assert coarse.resolution == 0.1 and coarse.labels.shape == (40, 60)
    assert fine.labels.shape == (80, 120)


def test_the_check_follows_the_origin_of_the_world():
    """A world that reaches west and south is measured where it lies."""
    world = world_of(box(-2.0, -1.0, 1.0, 1.0), origin=(-5.0, -3.0))
    found = check_connectivity(world, Clearance(), ((-2.0, -1.0),))
    assert found.origin == (-5.0, -3.0)
    assert found.anchors == (None,)  # the anchor sits in the box


def test_a_free_pocket_smaller_than_min_component_is_no_region():
    """A shut-in gap of four cells is free, and still is not a room to be counted."""
    blocked = np.zeros((20, 20), dtype=bool)
    blocked[2:6, 2:6] = True
    blocked[3:5, 3:5] = False  # four free cells walled in, 0.04 square metres
    point = Clearance(radius=0.0)
    many = check_grid(blocked, 0.1, dataclasses.replace(point, min_component=0.0))
    few = check_grid(blocked, 0.1, dataclasses.replace(point, min_component=0.05))
    assert many.components == 2 and few.components == 1
    assert few.main_fraction < 1.0  # the pocket is free space all the same


def test_an_anchor_off_the_grid_has_no_room():
    """A place outside the world is blocked, not a region."""
    blocked = np.zeros((10, 10), dtype=bool)
    found = check_grid(blocked, 0.1, Clearance(radius=0.0), ((-1, 5), (5, 99), (5, 5)))
    assert found.anchors == (None, None, 0)


def test_a_grid_with_no_room_at_all_reports_nothing_free():
    """Every cell blocked gives no region and no share, and does not divide by zero."""
    found = check_grid(np.ones((8, 8), dtype=bool), 0.1, Clearance(radius=0.0))
    assert not found.ok
    assert found.free_fraction == 0.0 and found.main_fraction == 0.0
    assert found.components == 0


def test_regions_are_numbered_largest_first_and_the_same_way_every_time():
    """Two rooms of equal size keep the order of their first cell."""
    blocked = np.zeros((9, 9), dtype=bool)
    blocked[4, :] = True
    found = check_grid(blocked, 0.1, Clearance(radius=0.0))
    assert found.labels[0, 0] == 0 and found.labels[8, 8] == 1
    assert check_grid(blocked, 0.1, Clearance(radius=0.0)).labels.tolist() == (
        found.labels.tolist()
    )


def test_two_reports_with_the_same_numbers_are_equal_whatever_their_labels():
    """The labels are a working grid, not part of what the report says."""
    found = check_connectivity(world_of())
    other = dataclasses.replace(found, labels=np.zeros((2, 2), dtype=np.int32))
    assert found == other and "labels" not in repr(found)


def test_a_report_can_be_made_with_no_labels_at_all():
    """The default is an empty grid, so a report is cheap to write by hand."""
    assert Connectivity(True, (), 1.0, 1, 1.0).labels.shape == (0, 0)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"radius": -0.1}, "radius must be 0 or more"),
        ({"margin": float("nan")}, "margin must be 0 or more"),
        ({"min_main": 1.5}, "min_main must be from 0 to 1"),
        ({"min_free": -0.1}, "min_free must be from 0 to 1"),
        ({"min_component": -1.0}, "min_component must be 0 or more"),
        ({"resolution": 0.0}, "resolution must be a positive number"),
    ],
)
def test_a_clearance_that_cannot_be_is_refused(settings, message):
    """Each setting is checked when the clearance is made."""
    with pytest.raises(ValueError, match=message):
        Clearance(**settings)


def test_a_grid_of_no_cells_at_all_is_measured_without_dividing_by_zero():
    """A world of nothing has no free share, and the check still answers."""
    found = check_grid(np.zeros((0, 0), dtype=bool), 0.1, Clearance(radius=0.0))
    assert found.free_fraction == 0.0 and found.components == 0 and not found.ok


def test_a_clearance_with_no_radius_takes_the_widest_robot_of_the_world():
    """A map the car can cross, the small robot can cross too."""
    shapes = [Circle(0.15), Box((1.0, 0.6)), Circle(0.4)]
    assert Clearance().for_robots(shapes).radius == pytest.approx(0.4)
    assert Clearance().for_robots([]).radius == 0.0  # no body to measure for
    assert Clearance(radius=0.1).for_robots(shapes).radius == 0.1  # its own wins


def test_the_check_measures_a_world_for_the_robot_that_will_drive_it():
    """Nothing need be said twice: the world already carries its robot."""
    door = (box(3.0, 0.75, 0.2, 1.5), box(3.0, 3.25, 0.2, 1.5))  # a 1 m door
    small = check_connectivity(world_of(*door, robots=[robot("r1", 1.0, 1.0)]))
    big = RobotSpec(
        "r1", "diff_drive", Circle(0.6), Transform2D.from_xyt([1.0, 2.0, 0.0])
    )
    wide = check_connectivity(world_of(*door, robots=[big]))
    assert small.components == 1 and small.ok  # 0.15 m body, the door is a door
    assert wide.components == 2  # 0.6 m body, the door is a wall


def test_a_clearance_that_has_not_met_its_robot_cannot_say_how_far_it_reaches():
    """A radius of None is a question, not a number, and is never guessed at."""
    with pytest.raises(ValueError, match="this clearance has no radius yet"):
        _ = Clearance().reach
    assert Clearance(radius=0.2, margin=0.1).reach == pytest.approx(0.3)


def test_the_smallest_region_that_counts_is_the_room_the_robot_needs():
    """What is too small to be a region follows the body, not a number here."""
    assert room(0.0) == 0.0 and room(0.25) == pytest.approx(0.25)
    assert Clearance(radius=0.25).smallest_region == pytest.approx(room(0.25))
    assert Clearance(radius=0.25, min_component=0.3).smallest_region == 0.3


def test_a_generator_is_never_left_to_guess_the_robot():
    """With no radius and no robot there is nothing to size a map by."""
    empty = World(size=SIZE, seed=1, obstacles=(), robots=())
    with pytest.raises(WorldError, match="nothing says how big the robot is"):
        clearance_for(Clearance(), empty)
    with pytest.raises(WorldError, match="nothing says how big the robot is"):
        clearance_for(Clearance(), None)
    # a map that need not pass the check promises nothing, so a point robot will do
    assert clearance_for(None, None).radius == 0.0
    assert clearance_for(None, empty).radius == 0.0
    # and either source answers the question
    assert clearance_for(Clearance(radius=0.3), None).radius == 0.3
    with_robot = World(size=SIZE, seed=1, obstacles=(), robots=(robot("r1", 1.0, 1.0),))
    assert clearance_for(Clearance(), with_robot).radius == pytest.approx(0.15)


def test_how_far_a_thing_spreads_over_a_grid_is_said_in_one_place():
    """The check and the styles that clean by it use the same rule, `within`."""
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    assert np.array_equal(within(mask, 0.1, 0.0), mask)  # no reach, no spread
    near = within(mask, 0.1, 0.2)
    assert near[5, 5] and near[3, 5] and near[5, 3]  # two cells each way
    assert not near[2, 5] and not near[3, 3]  # and no further, nor round a corner
    assert not within(np.zeros((5, 5), dtype=bool), 0.1, 0.3).any()


def test_a_way_through_is_the_robot_and_a_cell_of_slack_on_each_side():
    """Every corridor, tunnel and way out of an anchor is this wide."""
    assert lane(0.2, 0.1) == pytest.approx(0.6)
    assert lane(0.25, 0.05) == pytest.approx(0.6)
    assert lane(0.2, 0.1, slack=2) == pytest.approx(0.8)
