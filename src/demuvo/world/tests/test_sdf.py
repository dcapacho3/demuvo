"""Tests for reading a Gazebo world: its shapes become obstacles."""

import math
import warnings
from pathlib import Path

import pytest

from demuvo import Run, edit, load_world
from demuvo.world import WorldError
from demuvo.world.sdf import read_sdf

GAZEBO = Path(__file__).resolve().parents[4] / "examples" / "gazebo"

WORLD = """
<sdf version='1.7'><world name='w'>
  <model name='ground'><link name='l'><collision name='c'><geometry>
    <plane><normal>0 0 1</normal><size>100 100</size></plane>
  </geometry></collision></link></model>
  <model name='crate'><pose>2 1 0 0 0 1.5708</pose>
    <link name='l'><pose>1 0 0.5 0 0 0</pose>
      <collision name='c'><geometry>
        <box><size>2 1 1</size></box></geometry></collision>
      <collision name='floor'><pose>0 0 -0.5 0 0 0</pose><geometry>
        <box><size>9 9 0.05</size></box></geometry></collision>
      <collision name='roof'><pose>0 0 3 0 0 0</pose><geometry>
        <box><size>1 1 0.5</size></box></geometry></collision>
      <collision name='ramp'><pose>0 0 0.5 0 0.5 0</pose><geometry>
        <box><size>1 1 1</size></box></geometry></collision>
      <collision name='shelf'><geometry>
        <mesh><uri>x.dae</uri></mesh></geometry></collision>
      <collision name='nothing'/>
    </link></model>
  <model name='post'><pose>-1 -1 0.5 0 0 0</pose><link name='l'>
    <collision name='c'><geometry>
      <cylinder><radius>0.2</radius><length>1</length></cylinder>
    </geometry></collision>
    <collision name='ball'><pose>1 0 0 0 0 0</pose><geometry>
      <sphere><radius>0.3</radius></sphere></geometry></collision>
  </link></model>
</world></sdf>
"""


@pytest.fixture(name="world_file")
def make_world_file(tmp_path):
    """A small Gazebo world on disk."""
    path = tmp_path / "small.world"
    path.write_text(WORLD)
    return path


def test_the_shapes_that_stand_on_the_floor_become_obstacles(world_file):
    """The crate turns with its model, and floor, roof, ramp and plane are left out."""
    with pytest.warns(UserWarning, match="1 mesh shapes were left out"):
        data = read_sdf(world_file)
    shapes = [
        (o["shape"]["name"], [round(v, 2) for v in o["pose"]]) for o in data["obstacle"]
    ]
    assert shapes[0] == (
        "box",
        [2.0, 2.0, 1.57],
    )  # (1, 0) turned a quarter, from (2, 1)
    assert shapes[1:] == [("circle", [-1.0, -1.0, 0.0]), ("circle", [0.0, -1.0, 0.0])]
    assert data["obstacle"][0]["shape"]["size"] == [2.0, 1.0]
    assert data["obstacle"][1]["shape"]["radius"] == 0.2
    assert data["obstacle"][2]["shape"]["radius"] == 0.3


def test_the_world_is_sized_round_the_shapes_and_has_no_robot(world_file):
    """A margin of whole metres, and, as in an empty Gazebo world, no robot."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = load_world(world_file)
    assert world.origin == (-3.0, -3.0) and world.size == (7.0, 7.0)
    assert len(world.obstacles) == 3 and not world.robots


def test_a_file_that_is_not_xml_or_has_no_shape_is_refused(tmp_path):
    """Each gives a plain message."""
    bad = tmp_path / "bad.world"
    bad.write_text("not xml")
    with pytest.raises(WorldError, match="not valid XML"):
        read_sdf(bad)
    empty = tmp_path / "empty.world"
    empty.write_text("<sdf><world><model name='m'/></world></sdf>")
    with pytest.raises(WorldError, match="no box, cylinder or sphere"):
        read_sdf(empty)


def test_a_gazebo_world_with_leading_blank_lines_is_read(tmp_path):
    """Some files start with a newline before the XML declaration."""
    path = tmp_path / "gap.world"
    path.write_text(
        "\n<?xml version='1.0'?><sdf><world><model name='m'><link name='l'><collision "
        "name='c'><geometry><box><size>1 1 1</size></box></geometry></collision>"
        "</link></model></world></sdf>"
    )
    assert len(read_sdf(path)["obstacle"]) == 1


@pytest.mark.filterwarnings("ignore:.*mesh shapes")
@pytest.mark.parametrize("name", ["supermarket.world", "turtlebot3_world.sdf"])
def test_a_robot_drives_across_a_gazebo_example_with_a_planner_that_goes_round(name):
    """The imported walls and shelves are in the way, and A* finds the way."""
    world = load_world(GAZEBO / name)
    x0, y0 = world.origin
    world = edit.add_robot(
        world,
        "r1",
        (x0 + 1.0, y0 + 1.0, 0.0),
        planner="demuvo.planners:AStar",
        controller="demuvo.controllers:PurePursuit",
    )
    free = _far_free_place(world)
    world = edit.set_goal(world, "r1", (*free, 0.0))
    run = Run(world)
    assert run.until_done(300.0)
    assert run.result()[0].startswith("r1 task 1: reached")
    assert run.result()[-1] == "0 collisions"


def _far_free_place(world):
    """The free place, on a half metre grid, farthest from the robot."""
    start = world.robots[0].pose.translation
    x0, y0, x1, y1 = world.bounds
    best, far = None, 0.0
    x = x0 + 0.5
    while x <= x1 - 0.5:
        y = y0 + 0.5
        while y <= y1 - 0.5:
            clear = all(
                math.hypot(x - o.pose.xyt[0], y - o.pose.xyt[1]) > 1.5
                for o in world.obstacles
            )
            reach = math.hypot(x - start[0], y - start[1])
            if clear and reach > far:
                best, far = (x, y), reach
            y += 0.5
        x += 0.5
    return best
