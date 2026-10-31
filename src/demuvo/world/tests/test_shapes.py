"""Tests for the shapes and their checks."""

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import SHAPES, Box, Circle, LineString, Polygon, WorldError
from demuvo.world.shapes import parse_shape


def shape(**table):
    """Parse a shape table as an obstacle."""
    return parse_shape(table, "obstacle 1")


def test_the_table_names_every_shape():
    """The world file finds a shape by its name."""
    assert SHAPES == {
        "circle": Circle,
        "box": Box,
        "polygon": Polygon,
        "linestring": LineString,
    }


def test_a_box_has_four_corners_around_the_pose():
    """The corners go around the box, turned and moved by the pose."""
    corners = Box((4.0, 2.0)).points(
        Transform2D.from_xyt((10.0, 20.0, 3.141592653589793 / 2))
    )
    assert np.array(corners) == pytest.approx(
        np.array([(11, 18), (11, 22), (9, 22), (9, 18)])
    )


def test_a_circle_gives_the_square_around_it():
    """Two opposite corners of the square, whatever the angle."""
    assert Circle(0.5).points(Transform2D.from_xyt((1.0, 2.0, 0.7))) == [
        (0.5, 1.5),
        (1.5, 2.5),
    ]


def test_a_vertex_error_names_the_vertex():
    """The third vertex is the bad one, and the message says so."""
    with pytest.raises(WorldError, match="obstacle 1: 'vertex 3' needs 2"):
        shape(name="polygon", vertices=[[0, 0], [1, 0], [1], [0, 1]])


def test_a_polygon_needs_three_vertices():
    """Two vertices enclose nothing."""
    with pytest.raises(WorldError, match="at least 3"):
        shape(name="polygon", vertices=[[0, 0], [1, 0]])


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        ([[0, 0], [1, 0], [1, 0], [0, 1]], "two vertices in a row"),
        ([[0, 0], [1, 0], [1, 1], [0, 0]], "first and last"),
        ([[0, 0], [1, 1], [2, 2]], "no area"),
        ([[0, 0], [2, 0], [1, 0], [1, 1]], "doubles back"),
        ([[0, 0], [3, 3], [3, 0], [0, 2]], "edges 1 and 3 touch"),
        ([[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]], "touch"),
        ([[0, 0], [4, 0], [4, 2], [2, 0], [0, 2]], "touch"),
    ],
)
def test_a_bad_polygon_is_refused(vertices, message):
    """Repeats, no area, a spike and crossing or touching edges all fail."""
    with pytest.raises(WorldError, match=message):
        shape(name="polygon", vertices=vertices)


def test_a_good_polygon_is_accepted_in_either_order():
    """Clockwise and counter-clockwise both load; so does a concave one."""
    shape(name="polygon", vertices=[[0, 0], [2, 0], [2, 2]])
    shape(name="polygon", vertices=[[0, 0], [2, 2], [2, 0]])
    shape(name="polygon", vertices=[[0, 0], [4, 0], [2, 1], [4, 4], [0, 4]])


def test_a_line_string_may_cross_itself_but_not_repeat_a_point():
    """A wall can cross itself; two equal neighbours are a mistake."""
    shape(name="linestring", vertices=[[0, 0], [2, 2], [2, 0], [0, 2]])
    with pytest.raises(WorldError, match="two vertices in a row"):
        shape(name="linestring", vertices=[[0, 0], [1, 1], [1, 1]])
