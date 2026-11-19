"""Tests for reading triangles from Collada, STL and OBJ files."""

import struct

import numpy as np
import pytest

from demuvo.world import WorldError
from demuvo.world.meshes import MAX_DEPTH, is_closed, read_mesh

# A unit cube as 12 triangles with corners at 0 and 1.
CUBE = [
    (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
    (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
]  # fmt: skip
CORNERS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]  # fmt: skip


def cube_triangles() -> np.ndarray:
    """The cube's triangles, shape (12, 3, 3)."""
    return np.array(CORNERS, dtype=float)[np.array(CUBE)]


def dae(body: str, asset: str = "", scene: str = "") -> str:
    """A Collada file with the given geometry library, asset lines and scene."""
    return (
        '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
        f"<asset>{asset}</asset><library_geometries>{body}</library_geometries>"
        f"{scene}</COLLADA>"
    )


SQUARE = """
<geometry id="g1" name="Square"><mesh>
  <source id="p"><float_array id="pa" count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
    <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common>
  </source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  {primitive}
</mesh></geometry>
"""
TRIANGLES = (
    '<triangles count="2"><input semantic="VERTEX" source="#v" offset="0"/>'
    '<input semantic="NORMAL" source="#n" offset="1"/>'
    "<p>0 0 1 0 2 0 0 0 2 0 3 0</p></triangles>"
)
POLYLIST = (
    '<polylist count="2"><input semantic="VERTEX" source="#v" offset="0"/>'
    "<vcount>4 3</vcount><p>0 1 2 3 0 1 2</p></polylist>"
)
POLYGONS = (
    '<polygons count="2"><input semantic="VERTEX" source="#v" offset="0"/>'
    "<p>0 1 2 3</p><p>0 1</p><p>0 1 2</p></polygons>"
)


def write(tmp_path, name, text):
    """Write a file and give its path."""
    path = tmp_path / name
    path.write_text(text)
    return path


def total_area(part) -> float:
    """The area of all the triangles of a part."""
    a, b, c = part.triangles[:, 0], part.triangles[:, 1], part.triangles[:, 2]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)


def test_a_cube_is_closed_and_a_square_is_not():
    """Every edge of a cube is shared by two triangles; a loose triangle has none."""
    assert is_closed(cube_triangles())
    assert not is_closed(cube_triangles()[:-1])
    assert not is_closed(np.zeros((0, 3, 3)))


def test_a_binary_stl_gives_its_triangles(tmp_path):
    """The header, the count and 50 bytes a triangle are read."""
    data = b"\0" * 80 + struct.pack("<I", 12)
    for tri in cube_triangles():
        data += struct.pack("<12fH", 0, 0, 0, *tri.reshape(-1), 0)
    path = tmp_path / "cube.stl"
    path.write_bytes(data)
    (part,) = read_mesh(path)
    assert part.triangles.shape == (12, 3, 3) and part.closed
    assert total_area(part) == pytest.approx(6.0)


def test_a_text_stl_gives_its_triangles_and_a_broken_one_is_refused(tmp_path):
    """Corner lines are read; none, or a part of a triangle, is an error."""
    one = "solid s\nfacet normal 0 0 1\nouter loop\n" + "".join(
        f"vertex {x} {y} 0\n" for x, y in ((0, 0), (1, 0), (0, 1))
    )
    path = write(tmp_path, "one.stl", one + "endloop\nendfacet\nendsolid\n")
    (part,) = read_mesh(path)
    assert part.triangles.shape == (1, 3, 3)
    bad = write(tmp_path, "bad.stl", "solid s\nvertex 0 0 0\nvertex 1 0 0\nendsolid\n")
    with pytest.raises(WorldError, match="not a mesh demuvo can read"):
        read_mesh(bad)
    with pytest.raises(WorldError, match="no whole triangle"):
        read_mesh(write(tmp_path, "empty.stl", "nothing"))


def test_an_obj_gives_a_part_for_each_group_and_cuts_polygons_into_triangles(tmp_path):
    """Quads are fanned, negative indices count back, and groups name the parts."""
    text = (
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\n"
        "o floor\nf 1 2 3 4\ng wall\nf -5/1/1 -4 -1\n"
    )
    parts = read_mesh(write(tmp_path, "m.obj", text))
    assert [sorted(p.names) for p in parts] == [["floor"], ["wall"]]
    assert [len(p.triangles) for p in parts] == [2, 1]
    plain = read_mesh(
        write(tmp_path, "p.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\n\nf 1 2 3\n")
    )
    assert plain[0].names == frozenset()


def test_an_obj_with_no_face_or_a_face_off_the_end_is_refused(tmp_path):
    """Each is a plain error that names the file."""
    with pytest.raises(WorldError, match="holds no face"):
        read_mesh(write(tmp_path, "none.obj", "v 0 0 0\n"))
    with pytest.raises(WorldError, match="is not a mesh demuvo can read"):
        read_mesh(write(tmp_path, "off.obj", "v 0 0 0\nf 1 2 3\n"))


def test_another_kind_of_file_or_a_missing_one_is_refused_by_name(tmp_path):
    """The message says what demuvo reads, or why the file could not be opened."""
    with pytest.raises(WorldError, match=r"reads \.dae, \.stl and \.obj"):
        read_mesh(write(tmp_path, "m.fbx", ""))
    with pytest.raises(WorldError, match="no suffix"):
        read_mesh(write(tmp_path, "mesh", ""))
    with pytest.raises(WorldError, match="cannot be read"):
        read_mesh(tmp_path / "missing.dae")


def test_a_collada_triangle_list_with_two_inputs_gives_its_square(tmp_path):
    """The index stride counts every input; the unit scales the corners."""
    text = dae(
        SQUARE.format(primitive=TRIANGLES),
        asset='<unit meter="0.01" name="centimeter"/>',
    )
    (part,) = read_mesh(write(tmp_path, "t.dae", text))
    assert total_area(part) == pytest.approx(1.0 * 0.01**2)
    assert part.names == frozenset({"g1", "Square"})


def test_a_collada_polylist_and_polygons_are_cut_into_triangles(tmp_path):
    """A quad gives two triangles, a triangle one; one-corner polygons are skipped."""
    (part,) = read_mesh(
        write(tmp_path, "l.dae", dae(SQUARE.format(primitive=POLYLIST)))
    )
    assert len(part.triangles) == 3
    (part,) = read_mesh(
        write(tmp_path, "g.dae", dae(SQUARE.format(primitive=POLYGONS)))
    )
    assert len(part.triangles) == 3


def test_a_node_places_its_geometry_with_its_steps_in_order(tmp_path):
    """Translate, rotate, scale and a matrix, in nested nodes, and a shared node."""
    scene = """
    <library_nodes><node id="shared"><translate>0 0 5</translate>
      <instance_geometry url="#g1"/></node></library_nodes>
    <library_visual_scenes><visual_scene id="s">
      <node name="outer"><translate>1 0 0</translate>
        <node name="inner"><rotate>0 0 1 90</rotate><scale>2 2 1</scale>
          <instance_geometry url="#g1"/></node>
        <instance_node url="#shared"/></node>
      <node name="flat"><matrix>1 0 0 0 0 1 0 0 0 0 1 7 0 0 0 1</matrix>
        <extra/><instance_geometry url="#g1"/></node>
    </visual_scene></library_visual_scenes>
    """
    parts = read_mesh(
        write(tmp_path, "n.dae", dae(SQUARE.format(primitive=TRIANGLES), "", scene))
    )
    assert len(parts) == 3
    inner, shared, flat = parts
    assert {"inner"} <= inner.names and {"shared"} <= shared.names
    corners = inner.triangles.reshape(-1, 3)
    # turned a quarter and doubled: x runs from 1 - 2 to 1, y from 0 to 2
    assert corners[:, 0].min() == pytest.approx(-1.0) and corners[:, 1].max() == 2.0
    assert shared.triangles[:, :, 2].min() == pytest.approx(5.0)
    assert flat.triangles[:, :, 2].min() == pytest.approx(7.0)


def test_a_collada_file_without_a_scene_gives_each_geometry(tmp_path):
    """The geometry library alone is read, with the unit applied."""
    text = dae(SQUARE.format(primitive=TRIANGLES))
    (part,) = read_mesh(write(tmp_path, "bare.dae", text))
    assert len(part.triangles) == 2


def test_a_collada_file_with_no_namespace_is_read(tmp_path):
    """Some exporters leave the namespace out."""
    text = dae(SQUARE.format(primitive=TRIANGLES)).replace(
        ' xmlns="http://www.collada.org/2005/11/COLLADASchema"', ""
    )
    assert len(read_mesh(write(tmp_path, "plain.dae", text))[0].triangles) == 2


@pytest.mark.parametrize(
    ("body", "scene", "message"),
    [
        (
            SQUARE.format(primitive="<tristrips/>"),
            "",
            "tristrips> are not supported",
        ),
        (SQUARE.format(primitive="<lines/>"), "", "holds no triangle"),
        (
            SQUARE.format(primitive=TRIANGLES),
            "<library_visual_scenes><visual_scene><node><lookat>0 0 0 0 0 0 0 0 0"
            "</lookat></node></visual_scene></library_visual_scenes>",
            "lookat> in a node is not supported",
        ),
        (
            SQUARE.format(primitive=TRIANGLES),
            "<library_visual_scenes><visual_scene><node>"
            '<instance_geometry url="#nothing"/></node></visual_scene>'
            "</library_visual_scenes>",
            "a geometry, #nothing, is missing",
        ),
        (
            SQUARE.format(primitive=TRIANGLES),
            "<library_visual_scenes><visual_scene><node>"
            '<instance_node url="#nothing"/></node></visual_scene>'
            "</library_visual_scenes>",
            "an instanced node, #nothing, is missing",
        ),
        (
            SQUARE.format(primitive=TRIANGLES),
            '<library_nodes><node id="a"><instance_node url="#a"/></node>'
            "</library_nodes><library_visual_scenes><visual_scene><node>"
            '<instance_node url="#a"/></node></visual_scene></library_visual_scenes>',
            "in a loop",
        ),
        ("", "", "it places no geometry"),
    ],
)
def test_a_collada_file_demuvo_cannot_follow_is_refused_with_the_reason(
    tmp_path, body, scene, message
):
    """Strips, loops, missing parts and the like name what is wrong."""
    path = write(tmp_path, "bad.dae", dae(body, "", scene))
    with pytest.raises(WorldError, match=message):
        read_mesh(path)


def test_the_depth_limit_is_what_stops_a_node_loop():
    """A chain longer than the limit would be refused, and the limit is not tiny."""
    assert MAX_DEPTH >= 8
