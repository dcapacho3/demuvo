"""Tests for reading and checking world files."""

from pathlib import Path

import pytest

from demuvo.shapes import Box, Circle, LineString, Polygon
from demuvo.world import Obstacle, WorldError, load_world

GOOD = """
version = 1

[world]
size = [10.0, 6.0]
seed = 7

[[obstacle]]
pose = [4.0, 3.0, 0.0]
shape = { name = "box", size = [1.0, 2.0] }

[[obstacle]]
pose = [7.0, 2.0, 0.0]
shape = { name = "circle", radius = 0.5 }

[[robot]]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 1.0, 0.0]
goal = [9.0, 5.0, 0.0]
controller = "demuvo.controllers:GoToGoal"
"""


def write(tmp_path, text):
    """Write a world file and return its path."""
    path = tmp_path / "world.toml"
    path.write_text(text)
    return path


def load(tmp_path, text):
    """Load a world from text."""
    return load_world(write(tmp_path, text))


def test_loads_a_good_world(tmp_path):
    """Every field comes back, in file order, as tuples."""
    world = load(tmp_path, GOOD)
    assert world.size == (10.0, 6.0)
    assert world.seed == 7
    assert world.obstacles == (
        Obstacle((4.0, 3.0, 0.0), Box((1.0, 2.0))),
        Obstacle((7.0, 2.0, 0.0), Circle(0.5)),
    )
    (robot,) = world.robots
    assert robot.name == "r1"
    assert robot.shape == Circle(0.2)
    assert robot.pose == (1.0, 1.0, 0.0)
    assert robot.goal == (9.0, 5.0, 0.0)
    assert robot.controller == "demuvo.controllers:GoToGoal"


def test_the_example_world_loads():
    """The example file in the repository is a valid world."""
    path = Path(__file__).resolve().parents[1] / "examples" / "worlds" / "demo.toml"
    assert len(load_world(path).robots) == 1


def test_a_missing_file_is_not_a_world_error(tmp_path):
    """A file that does not exist raises the ordinary error."""
    with pytest.raises(FileNotFoundError):
        load_world(tmp_path / "nothing.toml")


def test_the_version_is_required(tmp_path):
    """No `version` key: the message names it."""
    with pytest.raises(WorldError, match="version"):
        load(tmp_path, GOOD.replace("version = 1", ""))


def test_a_newer_version_is_refused(tmp_path):
    """A version this code does not know is an error, not a guess."""
    with pytest.raises(WorldError, match="version"):
        load(tmp_path, GOOD.replace("version = 1", "version = 2"))


def test_a_missing_key_is_named(tmp_path):
    """No world `size`: the message names the key."""
    with pytest.raises(WorldError, match="size"):
        load(tmp_path, GOOD.replace("size = [10.0, 6.0]\n", ""))


def test_a_world_size_must_be_positive(tmp_path):
    """A zero or negative world size is refused."""
    with pytest.raises(WorldError, match="size"):
        load(tmp_path, GOOD.replace("size = [10.0, 6.0]", "size = [10.0, 0.0]"))


def test_an_unknown_shape_is_named(tmp_path):
    """The message names the shape and the obstacle."""
    with pytest.raises(WorldError, match=r"obstacle 1.*triangle"):
        load(tmp_path, GOOD.replace('name = "box"', 'name = "triangle"'))


def test_an_obstacle_outside_the_world_is_named(tmp_path):
    """A circle that sticks out of the world names the obstacle."""
    with pytest.raises(WorldError, match=r"obstacle 2.*inside"):
        load(tmp_path, GOOD.replace("pose = [7.0, 2.0, 0.0]", "pose = [9.8, 2.0, 0.0]"))


def test_a_rotated_box_uses_its_corners(tmp_path):
    """A 2 x 1 box at x = 1.02 fits only if it is not rotated."""
    old = 'pose = [4.0, 3.0, 0.0]\nshape = { name = "box", size = [1.0, 2.0] }'
    new = 'pose = [1.02, 3.0, ANGLE]\nshape = { name = "box", size = [2.0, 1.0] }'
    text = GOOD.replace(old, new)
    load(tmp_path, text.replace("ANGLE", "0.0"))
    with pytest.raises(WorldError, match=r"obstacle 1.*inside"):
        load(tmp_path, text.replace("ANGLE", "0.7853981633974483"))


def test_a_robot_pose_outside_the_world_is_named(tmp_path):
    """The message names the robot."""
    with pytest.raises(WorldError, match=r"robot 'r1'.*pose"):
        load(tmp_path, GOOD.replace("pose = [1.0, 1.0, 0.0]", "pose = [0.1, 1.0, 0.0]"))


def test_a_robot_goal_outside_the_world_is_named(tmp_path):
    """The goal must be inside the world too."""
    with pytest.raises(WorldError, match=r"robot 'r1'.*goal"):
        load(tmp_path, GOOD.replace("goal = [9.0, 5.0, 0.0]", "goal = [9.0, 6.5, 0.0]"))


def test_robot_names_must_be_unique(tmp_path):
    """Two robots with one name: the message names it."""
    second = GOOD[GOOD.index("[[robot]]") :]
    with pytest.raises(WorldError, match="r1"):
        load(tmp_path, GOOD + second)


def test_a_controller_needs_a_module_and_a_class(tmp_path):
    """The controller is written as module:Class."""
    with pytest.raises(WorldError, match=r"robot 'r1'.*controller"):
        load(tmp_path, GOOD.replace("demuvo.controllers:GoToGoal", "GoToGoal"))


def test_a_misspelled_key_is_refused(tmp_path):
    """A typo in a key is an error, not a silently ignored setting."""
    with pytest.raises(WorldError, match=r"robot 'r1' shape.*raduis"):
        load(tmp_path, GOOD.replace("radius = 0.2", "raduis = 0.2"))


def test_a_world_without_a_robot_is_refused(tmp_path):
    """A world with no robot has nothing to run."""
    with pytest.raises(WorldError, match="robot"):
        load(tmp_path, GOOD[: GOOD.index("[[robot]]")])


def test_a_world_without_obstacles_loads(tmp_path):
    """Obstacles are optional."""
    text = GOOD[: GOOD.index("[[obstacle]]")] + GOOD[GOOD.index("[[robot]]") :]
    assert not load(tmp_path, text).obstacles


def test_the_seed_must_be_an_integer(tmp_path):
    """A float seed is refused."""
    with pytest.raises(WorldError, match="seed"):
        load(tmp_path, GOOD.replace("seed = 7", "seed = 7.5"))


def test_a_number_must_be_finite(tmp_path):
    """TOML accepts nan and inf; the world does not."""
    with pytest.raises(WorldError, match="size"):
        load(tmp_path, GOOD.replace("size = [10.0, 6.0]", "size = [inf, 6.0]"))


def test_a_list_of_the_wrong_length_is_named(tmp_path):
    """A pose with two numbers instead of three names the obstacle and key."""
    with pytest.raises(WorldError, match=r"obstacle 1.*pose"):
        load(tmp_path, GOOD.replace("pose = [4.0, 3.0, 0.0]", "pose = [4.0, 3.0]"))


def test_a_text_value_must_not_be_empty(tmp_path):
    """An empty model name is refused."""
    with pytest.raises(WorldError, match=r"robot 'r1'.*model"):
        load(tmp_path, GOOD.replace('model = "diff_drive"', 'model = ""'))


def test_obstacles_must_be_written_as_a_table_array(tmp_path):
    """`obstacle = 3` instead of `[[obstacle]]` is refused."""
    head = GOOD[: GOOD.index("[[obstacle]]")]
    robot = GOOD[GOOD.index("[[robot]]") :]
    with pytest.raises(WorldError, match="obstacle"):
        load(tmp_path, "obstacle = 3\n" + head + robot)


def test_an_obstacle_must_be_a_table(tmp_path):
    """A list of numbers in place of tables is refused."""
    head = GOOD[: GOOD.index("[[obstacle]]")]
    robot = GOOD[GOOD.index("[[robot]]") :]
    with pytest.raises(WorldError, match="obstacle 1"):
        load(tmp_path, "obstacle = [1, 2]\n" + head + robot)


def test_the_world_must_be_a_table(tmp_path):
    """`world = 3` is refused."""
    with pytest.raises(WorldError, match="world"):
        load(tmp_path, "version = 1\nworld = 3\n")


def test_goal_and_controller_are_optional(tmp_path):
    """A robot needs only a name, a model, a shape and a pose."""
    text = GOOD[: GOOD.index("goal =")]
    (robot,) = load(tmp_path, text).robots
    assert robot.goal is None
    assert robot.controller is None


def test_a_robot_can_be_a_box(tmp_path):
    """A rectangular robot is checked by its corners."""
    box = 'shape = { name = "box", size = [0.6, 0.4] }'
    text = GOOD.replace('shape = { name = "circle", radius = 0.2 }', box)
    assert load(tmp_path, text).robots[0].shape == Box((0.6, 0.4))
    at = "pose = [0.25, 1.0, ANGLE]"
    text = text.replace("pose = [1.0, 1.0, 0.0]", at)
    load(tmp_path, text.replace("ANGLE", "1.5707963267948966"))
    with pytest.raises(WorldError, match=r"robot 'r1'.*pose"):
        load(tmp_path, text.replace("ANGLE", "0.0"))


def test_a_robot_cannot_be_a_line(tmp_path):
    """A linestring has no area."""
    line = 'shape = { name = "linestring", vertices = [[0, 0], [1, 0]] }'
    text = GOOD.replace('shape = { name = "circle", radius = 0.2 }', line)
    with pytest.raises(WorldError, match=r"robot 'r1'.*linestring"):
        load(tmp_path, text)


def test_a_polygon_obstacle_and_a_wall_load(tmp_path):
    """Polygon and linestring obstacles; both may leave out the pose."""
    extra = (
        '[[obstacle]]\nshape = { name = "polygon",'
        " vertices = [[5, 4], [6, 4], [5.5, 5]] }\n"
        '[[obstacle]]\nshape = { name = "linestring",'
        " vertices = [[2, 5], [2, 3.5]] }\n"
    )
    text = GOOD.replace("[[robot]]", extra + "[[robot]]")
    *_, polygon, wall = load(tmp_path, text).obstacles
    assert polygon == Obstacle(
        (0.0, 0.0, 0.0), Polygon(((5.0, 4.0), (6.0, 4.0), (5.5, 5.0)))
    )
    assert wall.shape == LineString(((2.0, 5.0), (2.0, 3.5)))


def test_a_polygon_needs_three_vertices(tmp_path):
    """Two points do not enclose an area."""
    bad = '[[obstacle]]\nshape = { name = "polygon", vertices = [[5, 4], [6, 4]] }\n'
    with pytest.raises(WorldError, match=r"obstacle 3.*vertices"):
        load(tmp_path, GOOD.replace("[[robot]]", bad + "[[robot]]"))


def test_a_polygon_vertex_outside_the_world_is_named(tmp_path):
    """Every vertex must lie inside the world."""
    bad = (
        '[[obstacle]]\nshape = { name = "polygon",'
        " vertices = [[5, 4], [11, 4], [5.5, 5]] }\n"
    )
    with pytest.raises(WorldError, match=r"obstacle 3.*inside"):
        load(tmp_path, GOOD.replace("[[robot]]", bad + "[[robot]]"))


def test_the_shape_is_required_and_must_be_a_table(tmp_path):
    """A missing or malformed shape names the object."""
    no_shape = GOOD.replace('shape = { name = "circle", radius = 0.5 }\n', "")
    with pytest.raises(WorldError, match=r"obstacle 2.*shape"):
        load(tmp_path, no_shape)
    with pytest.raises(WorldError, match=r"obstacle 2 shape.*table"):
        load(tmp_path, GOOD.replace('{ name = "circle", radius = 0.5 }', '"circle"'))


def test_the_known_shapes_are_listed_in_the_error(tmp_path):
    """The message says which names are allowed."""
    with pytest.raises(WorldError, match=r"triangle.*circle.*box.*polygon.*linestring"):
        load(tmp_path, GOOD.replace('name = "box"', 'name = "triangle"'))


def test_a_circle_radius_must_be_positive(tmp_path):
    """A negative radius is refused, and the message names the obstacle."""
    with pytest.raises(WorldError, match=r"obstacle 2.*radius"):
        load(tmp_path, GOOD.replace("radius = 0.5", "radius = -0.5"))


def test_an_unknown_model_is_named(tmp_path):
    """The message names the robot, the model and the known models."""
    with pytest.raises(WorldError, match=r"robot 'r1'.*hovercraft.*diff_drive"):
        load(tmp_path, GOOD.replace('"diff_drive"', '"hovercraft"'))


def test_params_replace_the_model_defaults(tmp_path):
    """The values in `params` come back on the robot."""
    text = GOOD.replace(
        'controller = "demuvo.controllers:GoToGoal"',
        'controller = "demuvo.controllers:GoToGoal"\n'
        "params = { max_v = 0.5, max_w = 2.0 }",
    )
    assert load(tmp_path, text).robots[0].params == {"max_v": 0.5, "max_w": 2.0}


def test_robots_have_no_params_by_default(tmp_path):
    """No `params` key gives an empty dict."""
    assert load(tmp_path, GOOD).robots[0].params == {}


def test_an_unknown_param_is_named(tmp_path):
    """A parameter the model does not have is refused."""
    text = GOOD.replace(
        'controller = "demuvo.controllers:GoToGoal"',
        'controller = "demuvo.controllers:GoToGoal"\nparams = { max_speed = 1.0 }',
    )
    with pytest.raises(WorldError, match=r"robot 'r1' params.*max_speed"):
        load(tmp_path, text)


def test_a_param_must_be_positive(tmp_path):
    """A zero or negative limit is refused."""
    text = GOOD.replace(
        'controller = "demuvo.controllers:GoToGoal"',
        'controller = "demuvo.controllers:GoToGoal"\nparams = { max_v = 0 }',
    )
    with pytest.raises(WorldError, match=r"robot 'r1' params.*max_v"):
        load(tmp_path, text)


def test_a_toml_syntax_error_is_a_world_error(tmp_path):
    """A file that is not TOML raises WorldError, like any other bad file."""
    with pytest.raises(WorldError, match="not valid TOML"):
        load(tmp_path, "version = = 1")
