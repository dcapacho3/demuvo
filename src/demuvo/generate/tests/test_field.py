"""Tests for the field style: the noise, the cut, the cleaning and the retries."""

import dataclasses

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.costmap import build_costmaps
from demuvo.generate import (
    Clearance,
    FieldStyle,
    GenerateError,
    cut_mask,
    field_of,
    generate_field,
)
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

SIZE = (10.0, 6.0)
# The robot the maps of most tests are drawn for. A generator is never left to
# guess one, so every call either says this or brings a base world with a robot.
ROBOT = Clearance(radius=0.2)
# A style with no cleaning at all, so a test sees the cut itself.
RAW = FieldStyle(smooth=0, min_area=0.0, min_pocket=0.0)


def styled(**settings):
    """The raw style with some settings replaced."""
    return dataclasses.replace(RAW, **settings)


def a_table(name="table", x=5.0, y=3.0):
    """A box obstacle for a base world."""
    return Obstacle(Transform2D.from_xyt([x, y, 0.0]), Box((0.5, 0.5)), name)


def a_robot(name="r1", x=1.0, y=1.0):
    """A small round robot at a place."""
    return RobotSpec(name, "diff_drive", Circle(0.2), Transform2D.from_xyt([x, y, 0.0]))


def blocks_field():
    """A field of 68 high cells in two blocks, a speck of 4 and a blob of 64."""
    values = np.random.default_rng(0).random((20, 20)) * 0.5
    values[2:4, 2:4] = 1.0
    values[8:16, 8:16] = 1.0
    return values


def test_a_seed_gives_the_same_world_every_time():
    """Two calls with equal arguments agree, pose for pose and vertex for vertex."""
    one, two = (
        generate_field(SIZE, 7, clearance=ROBOT),
        generate_field(SIZE, 7, clearance=ROBOT),
    )
    assert one.world == two.world
    assert np.array_equal(one.mask, two.mask)
    assert np.array_equal(one.field, two.field)
    assert one.attempt == two.attempt


def test_another_seed_gives_another_world():
    """The seed is what the map is drawn from."""
    assert not np.array_equal(
        generate_field(SIZE, 7, clearance=ROBOT).mask,
        generate_field(SIZE, 8, clearance=ROBOT).mask,
    )


def test_the_world_carries_the_seed_it_was_drawn_from():
    """So a run of it, and the seeds after it, start from a known place."""
    assert generate_field(SIZE, 11, clearance=ROBOT).world.seed == 11


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("pieces", ["polygons", "boxes"])
def test_a_generated_world_is_written_and_read_back_unchanged(tmp_path, seed, pieces):
    """Every shape passes the gate a world file passes; this is the proof."""
    got = generate_field(SIZE, seed, FieldStyle(pieces=pieces), ROBOT)
    path = tmp_path / "field.toml"
    save_world(got.world, path)
    assert load_world(path) == got.world


def test_a_world_that_reaches_west_and_south_is_drawn_where_it_lies():
    """The origin moves the field, and no shape leaves the world."""
    got = generate_field(SIZE, 3, clearance=ROBOT, origin=(-5.0, -3.0))
    assert got.world.origin == (-5.0, -3.0)
    for obstacle in got.world.obstacles:
        for x, y in obstacle.shape.points(obstacle.pose):
            assert -5.0 <= x <= 5.0 and -3.0 <= y <= 3.0


def test_the_field_is_smooth_and_covers_minus_one_to_one():
    """A lattice of random values, eased out, stays in range and has no jumps."""
    values = field_of(np.random.default_rng(1), 60, 100, FieldStyle())
    assert values.shape == (60, 100)
    assert -1.0 <= values.min() and values.max() <= 1.0
    # neighbouring cells of a field of features 2 m wide are close together
    assert np.abs(np.diff(values, axis=1)).max() < 0.2


def test_fewer_octaves_make_a_smoother_field():
    """Each octave adds detail, which shows as a bigger step between cells."""
    rng = np.random.default_rng(2)
    one = field_of(rng, 60, 100, FieldStyle(octaves=1))
    many = field_of(rng, 60, 100, FieldStyle(octaves=4))
    assert np.abs(np.diff(one)).mean() < np.abs(np.diff(many)).mean()


def test_the_cut_blocks_the_share_of_the_world_that_was_asked_for():
    """`fill` is a quantile, so it means the same on every field."""
    values = field_of(np.random.default_rng(3), 200, 200, FieldStyle())
    for fill in (0.2, 0.35, 0.5):
        style = FieldStyle(smooth=0, min_area=0.0, min_pocket=0.0, fill=fill)
        mask = cut_mask(values, style, (0.0, 0.0))
        assert mask.mean() == pytest.approx(fill, abs=0.05)


def test_a_fill_of_nothing_leaves_an_empty_world():
    """0 asks for no clutter, and the quantile is not consulted."""
    got = generate_field(SIZE, 1, FieldStyle(fill=0.0), ROBOT)
    assert not got.world.obstacles and not got.mask.any()
    assert got.connectivity.ok and got.connectivity.components == 1


def test_a_speck_smaller_than_min_area_is_dropped():
    """Four cells are noise; sixty-four are a blob."""
    kept = cut_mask(blocks_field(), styled(fill=0.17), (0.0, 0.0))
    dropped = cut_mask(blocks_field(), styled(fill=0.17, min_area=0.05), (0.0, 0.0))
    assert kept.sum() == 68 and dropped.sum() == 64


def test_a_pocket_a_robot_could_never_enter_is_filled():
    """A hole of sixteen cells inside a blob is not a room."""
    values = blocks_field()
    values[10:14, 10:14] = 0.0  # a hole through the middle of the blob
    assert cut_mask(values, styled(fill=0.13), (0.0, 0.0)).sum() == 52
    assert cut_mask(values, styled(fill=0.13, min_pocket=0.2), (0.0, 0.0)).sum() == 68


def test_the_border_of_the_world_is_always_left_free():
    """Two cells at least, so no vertex of a blob can land on the edge."""
    values = np.ones((20, 20))
    values[::2, ::2] = 0.0  # a field with two values, so the cut takes half of it
    mask = cut_mask(values, styled(fill=0.5), (0.0, 0.0))
    assert not mask[:2, :].any() and not mask[-2:, :].any()
    assert not mask[:, :2].any() and not mask[:, -2:].any()
    wide = cut_mask(values, styled(fill=0.5, border=0.5), (0.0, 0.0))
    assert not wide[:5, :].any() and not wide[:, :5].any()


def test_the_ground_under_an_anchor_is_cleared_for_the_robot():
    """A robot or a goal of the base world always has room to stand."""
    values = np.ones((20, 20))
    style = styled(fill=1.0)
    mask = cut_mask(values, style, (0.0, 0.0), ((1.0, 1.0),), 0.2)
    assert not mask[10, 10]  # the cell of the anchor, at 0.1 m cells
    assert mask[5, 5]  # and the rest of the world is still blocked
    assert not cut_mask(values, style, (0.0, 0.0), ((-5.0, -5.0),), 0.2)[0, 0]


def test_smoothing_takes_away_a_ridge_one_cell_wide():
    """The majority rule keeps a blob and drops what is too thin to be one."""
    values = np.random.default_rng(4).random((20, 20)) * 0.5
    values[10, 2:18] = 1.0  # a ridge of sixteen cells, one cell thick
    values[4:8, 4:8] = 1.0  # a blob of sixteen cells, four by four
    rough = cut_mask(values, styled(fill=0.08), (0.0, 0.0))
    smoothed = cut_mask(values, styled(fill=0.08, smooth=1), (0.0, 0.0))
    assert rough.sum() == 32
    assert not smoothed[10, 2:18].any() and smoothed[5:7, 5:7].all()


def test_boxes_instead_of_polygons_gives_only_boxes():
    """A style that asks for boxes gets no polygon at all."""
    got = generate_field(SIZE, 5, FieldStyle(pieces="boxes"), ROBOT)
    assert got.world.obstacles
    assert all(o.shape.name == "box" for o in got.world.obstacles)


def test_the_obstacles_are_named_in_order():
    """The window lists them by name, so the names must not jump about."""
    got = generate_field(SIZE, 2, clearance=ROBOT)
    names = [o.name for o in got.world.obstacles]
    assert names == [f"field {i}" for i in range(1, len(names) + 1)]


def test_a_base_world_keeps_everything_it_had():
    """The clutter is added to it; its robots, tasks and settings stay."""
    base = World(
        size=SIZE,
        seed=99,
        obstacles=(),
        robots=(a_robot(),),
        tasks=(Task("r1", Transform2D.from_xyt([9.0, 5.0, 0.0])),),
    )
    got = generate_field(SIZE, 4, base=base)
    assert got.world.robots == base.robots and got.world.tasks == base.tasks
    assert got.world.seed == 4  # the seed asked for, not the base's
    assert got.world.obstacles  # the clutter was added to it
    # the robot and its goal are anchors, so both have room and connect
    assert got.connectivity.anchors == (0, 0)


def test_the_obstacles_of_a_base_world_come_first_and_keep_their_names():
    """A table of the base world stays a table, and the clutter follows it."""
    base = World(size=SIZE, seed=1, obstacles=(a_table(),), robots=(a_robot(),))
    got = generate_field(SIZE, 6, base=base)
    assert got.world.obstacles[0] == a_table()
    assert got.world.obstacles[1].name == "field 1"


def test_a_base_world_of_another_size_is_refused():
    """The size and the origin belong to the base world, and are not guessed."""
    base = World(size=(4.0, 4.0), seed=1, obstacles=(), robots=())
    with pytest.raises(WorldError, match="leave the size and the origin"):
        generate_field(SIZE, 1, base=base)


def test_a_base_world_that_already_uses_the_names_is_refused():
    """Two obstacles of one name is a world file error, so it is caught here."""
    base = World(size=SIZE, seed=1, obstacles=(a_table("field 1"),), robots=())
    with pytest.raises(WorldError, match="already has an obstacle called 'field 1'"):
        generate_field(SIZE, 1, base=base)


def test_a_map_that_cannot_be_connected_is_refused_after_the_last_attempt():
    """The message says how many were tried and what the last one was like."""
    with pytest.raises(GenerateError) as raised:
        generate_field(SIZE, 1, FieldStyle(fill=0.9), ROBOT, attempts=2)
    assert "in 2 attempts" in str(raised.value)
    assert "Lower fill" in str(raised.value)


def test_one_attempt_is_counted_as_one():
    """The message is written for a person, so it says 'attempt' once."""
    with pytest.raises(GenerateError, match="in 1 attempt."):
        generate_field(SIZE, 1, FieldStyle(fill=0.9), ROBOT, attempts=1)


def test_asking_for_no_clearance_keeps_the_first_field_whatever_it_is():
    """A caller that wants the map anyway gets it, and the report still says."""
    got = generate_field(SIZE, 1, FieldStyle(fill=0.9), clearance=None)
    assert got.attempt == 0 and not got.connectivity.ok


def test_a_world_the_robot_can_cross_is_one_a_planner_can_cross():
    """What the check calls connected, A* can plan; that is what it is for."""
    base = World(size=SIZE, seed=1, obstacles=(), robots=(a_robot(),))
    got = generate_field(SIZE, 12, base=base, anchors=((9.0, 5.0),))
    costmap = build_costmaps(got.world)["r1"]
    planner = AStar()
    planner.configure(costmap)
    path = planner.plan(
        Transform2D.from_xyt([1.0, 1.0, 0.0]), Transform2D.from_xyt([9.0, 5.0, 0.0])
    )
    assert len(path) > 0


def test_a_robot_shut_in_by_the_clutter_is_never_given_back():
    """An anchor must be in the largest free region, which is what A* searches."""
    got = generate_field(
        SIZE, 12, base=World(size=SIZE, seed=1, obstacles=(), robots=(a_robot(),))
    )
    assert got.connectivity.anchors == (0,)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"resolution": 0.0}, "resolution must be a positive number"),
        ({"scale": -1.0}, "scale must be a positive number"),
        ({"persistence": 1.5}, "persistence must be from 0 to 1"),
        ({"fill": 1.5}, "fill must be from 0 to 1"),
        ({"min_area": -1.0}, "min_area must be 0 or more"),
        ({"min_pocket": -1.0}, "min_pocket must be 0 or more"),
        ({"border": -1.0}, "border must be 0 or more"),
        ({"octaves": 0}, "octaves must be 1 or more"),
        ({"octaves": 1.5}, "octaves must be a whole number"),
        ({"smooth": -1}, "smooth must be a whole number"),
        ({"pieces": "circles"}, "pieces must be one of polygons, boxes"),
    ],
)
def test_a_style_that_cannot_be_is_refused(settings, message):
    """Each setting is checked when the style is made, before any work."""
    with pytest.raises(ValueError, match=message):
        FieldStyle(**settings)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"attempts": 0}, "attempts must be 1 or more"),
        ({"attempts": 1.5}, "attempts must be 1 or more"),
    ],
)
def test_an_argument_that_cannot_be_is_refused(settings, message):
    """The call itself is checked too."""
    with pytest.raises(ValueError, match=message):
        generate_field(SIZE, 1, clearance=ROBOT, **settings)


@pytest.mark.parametrize("size", [(0.0, 5.0), (5.0, -1.0)])
def test_a_world_of_no_size_is_refused(size):
    """A world must have room to draw in."""
    with pytest.raises(ValueError, match="size must be two lengths above 0"):
        generate_field(size, 1, clearance=ROBOT)


def test_what_counts_as_a_pocket_follows_the_robot():
    """A gap a small robot can use is a gap; for a wide one it is a hole to fill."""
    values = blocks_field()
    values[10:14, 10:14] = 0.0  # a hole of sixteen cells, 0.16 square metres
    style = FieldStyle(smooth=0, min_area=0.0, fill=0.13)  # min_pocket from the robot
    small = cut_mask(values, style, (0.0, 0.0), reach=0.1)  # needs 0.04 m2
    wide = cut_mask(values, style, (0.0, 0.0), reach=0.25)  # needs 0.25 m2
    assert small.sum() == 52  # the hole is left open
    assert wide.sum() == 68  # the hole is filled in


def test_a_field_is_refused_when_nothing_says_how_big_the_robot_is():
    """Clutter is cleaned and checked by the robot, so one must be named."""
    with pytest.raises(WorldError, match="nothing says how big the robot is"):
        generate_field(SIZE, 1)


def test_the_robot_of_the_base_world_measures_the_clutter():
    """A base world carries its robot, so the caller need not say it twice."""
    base = World(size=SIZE, seed=1, obstacles=(), robots=(a_robot(),))
    got = generate_field(SIZE, 4, base=base)
    assert got.connectivity.ok and got.connectivity.anchors == (0,)
