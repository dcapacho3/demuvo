"""Tests for the meshes, includes and warnings of the Gazebo reader."""

# pylint: disable=protected-access  # one test checks a helper of the module

import math
import warnings

import numpy as np
import pytest

from demuvo import load_world
from demuvo.world import WorldError, sdf
from demuvo.world.sdf import read_sdf

from .test_meshes import CORNERS, CUBE, SQUARE, TRIANGLES, dae

CORNER_TEXT = " ".join(str(v) for c in CORNERS for v in c)
INDEX_TEXT = " ".join(str(i) for t in CUBE for i in t)
CUBE_DAE = dae(
    '<geometry id="cube" name="Cube"><mesh>'
    f'<source id="p"><float_array id="pa" count="24">{CORNER_TEXT}</float_array>'
    '<technique_common><accessor source="#pa" count="8" stride="3"/>'
    "</technique_common></source>"
    '<vertices id="v"><input semantic="POSITION" source="#p"/></vertices>'
    '<triangles count="12"><input semantic="VERTEX" source="#v" offset="0"/>'
    f"<p>{INDEX_TEXT}</p></triangles></mesh></geometry>"
)


def world_text(body: str, name: str = "w") -> str:
    """A Gazebo world with the given models."""
    return f"<sdf version='1.9'><world name='{name}'>{body}</world></sdf>"


def mesh_model(
    uri: str = "cube.dae", pose="0 0 0 0 0 0", scale="1 1 1", kind="collision"
) -> str:
    """A model whose one link has a mesh of the given kind."""
    return (
        f"<model name='m'><pose>{pose}</pose><link name='l'><{kind} name='c'>"
        f"<geometry><mesh><uri>{uri}</uri><scale>{scale}</scale></mesh></geometry>"
        f"</{kind}></link></model>"
    )


def read(tmp_path, text, name="w.world"):
    """Write a world file and read it, giving the data and the warnings."""
    path = tmp_path / name
    path.write_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = read_sdf(path)
    return data, [str(w.message) for w in caught]


@pytest.fixture(name="folder")
def make_folder(tmp_path, monkeypatch):
    """A folder with a cube mesh, and no outside folders to look in."""
    monkeypatch.delenv("GZ_SIM_RESOURCE_PATH", raising=False)
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "cube.dae").write_text(CUBE_DAE)
    return tmp_path


def test_a_mesh_cube_on_the_floor_becomes_a_box_at_its_place(folder):
    """The cube is scaled by 2 and moved to (3, 1): a box of 2 by 2 there."""
    text = world_text(mesh_model(pose="3 1 0 0 0 0", scale="2 2 2"))
    data, messages = read(folder, text)
    (box,) = data["obstacle"]
    assert box["shape"]["name"] == "box" and messages == []
    assert box["pose"][:2] == pytest.approx([4.0, 2.0], abs=0.03)
    assert box["shape"]["size"] == pytest.approx([2.0, 2.0], abs=0.06)
    world = load_world(folder / "w.world")
    assert len(world.obstacles) == 1 and not world.robots


def test_a_mesh_scaled_by_one_number_and_given_in_a_folder_is_found(folder):
    """A single scale applies to all three axes; a relative address is from the file."""
    (folder / "meshes").mkdir()
    (folder / "meshes" / "cube.dae").write_text(CUBE_DAE)
    text = world_text(mesh_model("meshes/cube.dae", scale="3"))
    data, _ = read(folder, text)
    assert data["obstacle"][0]["shape"]["size"] == pytest.approx([3.0, 3.0], abs=0.06)


def test_the_roof_and_the_floor_of_a_mesh_are_not_cut(folder):
    """A cube raised above the cut, and one sunk below it, give no obstacle."""
    high = world_text(mesh_model(pose="0 0 3 0 0 0"))
    with pytest.raises(WorldError, match="nothing between"):
        read(folder, high)
    low = world_text(mesh_model(pose="0 0 -3 0 0 0"))
    with pytest.raises(WorldError, match="nothing between 0.1 and 1"):
        read(folder, low)


def test_a_mesh_that_is_not_found_is_told_with_where_it_was_looked_for(folder):
    """The shapes that can be read stay; the warning names the mesh and the places."""
    body = (
        "<model name='box'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>0 0 0.5 0 0 0</pose>"
        "</collision></link></model>" + mesh_model("model://nothing/meshes/x.dae")
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    (message,) = messages
    assert "model://nothing/meshes/x.dae" in message and "was not found" in message
    assert "GZ_SIM_RESOURCE_PATH" in message and "more places" in message


def test_a_mesh_in_a_folder_named_by_the_environment_is_found(folder, monkeypatch):
    """GZ_SIM_RESOURCE_PATH, and the share folders of ROS, are searched."""
    models = folder / "elsewhere"
    (models / "thing" / "meshes").mkdir(parents=True)
    (models / "thing" / "meshes" / "cube.dae").write_text(CUBE_DAE)
    monkeypatch.setenv("GZ_SIM_RESOURCE_PATH", str(models))
    data, messages = read(
        folder, world_text(mesh_model("model://thing/meshes/cube.dae"))
    )
    assert len(data["obstacle"]) == 1 and messages == []
    share = folder / "ros" / "share" / "pkg" / "models" / "pack" / "meshes"
    share.mkdir(parents=True)
    (share / "cube.dae").write_text(CUBE_DAE)
    monkeypatch.setenv("AMENT_PREFIX_PATH", f"{folder / 'ros'}:")
    monkeypatch.delenv("GZ_SIM_RESOURCE_PATH")
    data, messages = read(
        folder, world_text(mesh_model("model://pack/meshes/cube.dae"))
    )
    assert len(data["obstacle"]) == 1 and messages == []
    data, messages = read(
        folder, world_text(mesh_model("package://pkg/models/pack/meshes/cube.dae"))
    )
    assert len(data["obstacle"]) == 1 and messages == []


def test_a_file_uri_and_an_unusable_address_are_followed_or_told(folder):
    """A file:// address is a place on disk; one that goes nowhere is a warning."""
    here = f"file://{folder / 'cube.dae'}"
    data, messages = read(folder, world_text(mesh_model(here)))
    assert len(data["obstacle"]) == 1 and messages == []
    body = mesh_model("https://example.org/not/a/model") + mesh_model("")
    with pytest.raises(WorldError) as caught:
        read(folder, world_text(body))
    assert "a <mesh> with no <uri>" in str(caught.value)
    assert "was not found" in str(caught.value)


def test_a_mesh_that_cannot_be_read_is_told_once_with_its_reason(folder):
    """A broken file is named, once, however many shapes use it."""
    (folder / "bad.dae").write_text("not xml")
    body = mesh_model("bad.dae") * 3 + (
        "<model name='box'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>0 0 0.5 0 0 0</pose>"
        "</collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    assert len(messages) == 1 and "bad.dae" in messages[0]
    assert "is not a mesh demuvo can read" in messages[0]


def test_a_submesh_picks_one_part_and_a_missing_one_is_told(folder):
    """Only the named piece is cut; a name that is not there leaves the mesh out."""
    two = dae(
        (SQUARE.format(primitive=TRIANGLES) + SQUARE.format(primitive=TRIANGLES))
        .replace('id="g1" name="Square"', 'id="a" name="A"', 1)
        .replace('id="g1" name="Square"', 'id="b" name="B"', 1),
        "",
        '<library_visual_scenes><visual_scene><node name="A">'
        '<instance_geometry url="#a"/></node><node name="B">'
        '<translate>9 0 0</translate><instance_geometry url="#b"/></node>'
        "</visual_scene></library_visual_scenes>",
    )
    (folder / "two.dae").write_text(two)
    text = world_text(
        "<model name='m'><link name='l'><collision name='c'><geometry><mesh>"
        "<uri>two.dae</uri><submesh><name>B</name></submesh></mesh></geometry>"
        "<pose>0 0 0.5 1.5708 0 0</pose></collision></link></model>"
    )
    data, _ = read(folder, text)
    (shape,) = data["obstacle"]
    assert shape["pose"][0] > 8
    missing = text.replace("<name>B</name>", "<name>C</name>")
    with pytest.raises(WorldError, match="no part named 'C'"):
        read(folder, missing)


def test_a_world_with_no_collision_shape_is_read_from_its_visual_meshes(folder):
    """Visuals stand in, and a warning says so."""
    data, messages = read(folder, world_text(mesh_model(kind="visual")))
    assert len(data["obstacle"]) == 1
    assert any("visual shapes were used" in m for m in messages)


def test_a_world_with_nothing_to_read_says_what_was_missed(folder):
    """The error lists every reason, from the first reading and from the visuals."""
    text = world_text(mesh_model("missing.dae") + mesh_model("lost.dae", kind="visual"))
    with pytest.raises(WorldError) as caught:
        read(folder, text)
    assert "missing.dae" in str(caught.value) and "lost.dae" in str(caught.value)


def test_a_world_with_no_shape_at_all_gives_the_plain_error(folder):
    """No reason to add when nothing was left out."""
    with pytest.raises(WorldError, match=r"stands on the floor between 0.1 and 1 m$"):
        read(folder, world_text("<model name='m'/>"))


def test_the_shapes_the_heights_leave_out_are_counted_in_one_warning(folder):
    """A floor slab, a roof and a ramp are named with how many there were."""
    body = "".join(
        f"<model name='m{i}'><pose>{p}</pose><link name='l'><collision name='c'>"
        f"<geometry>"
        f"<box><size>1 1 {z}</size></box></geometry></collision></link></model>"
        for i, (p, z) in enumerate(
            [("0 0 0 0 0 0", 0.05), ("0 0 3 0 0 0", 0.5), ("0 0 0.5 0 0.9 0", 1)]
        )
    ) + (
        "<model name='ok'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>5 5 0.5 0 0 0</pose>"
        "</collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    (message,) = messages
    assert "3 shapes were left out" in message
    assert (
        "above 1 m" in message and "lower than 0.1 m" in message and "tilted" in message
    )


def test_a_shape_demuvo_cannot_read_is_told(folder):
    """A heightmap is a warning, and the other shape is still read."""
    body = (
        "<model name='m'><link name='l'><collision name='c'><geometry><heightmap/>"
        "</geometry></collision><collision name='d'><geometry/></collision>"
        "<collision name='b'><geometry><box><size>1 1 1</size></box></geometry>"
        "<pose>0 0 0.5 0 0 0</pose></collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    assert any("a 'heightmap' shape cannot be read" in m for m in messages)
    assert any("a 'empty' shape cannot be read" in m for m in messages)


def test_a_pose_in_degrees_and_a_pose_that_demuvo_cannot_follow(folder):
    """Degrees are turned to radians; another rotation format or a frame is told."""
    body = (
        "<model name='m'><link name='l'><collision name='c'><geometry>"
        "<box><size>2 1 1</size></box></geometry>"
        "<pose degrees='true'>1 1 0.5 0 0 90</pose></collision>"
        "<collision name='q'><geometry><box><size>1 1 1</size></box></geometry>"
        "<pose rotation_format='quat_xyzw'>5 5 0.5 0 0 0 1</pose></collision>"
        "<collision name='f'><geometry><box><size>1 1 1</size></box></geometry>"
        "<pose relative_to='other'>8 8 0.5 0 0 0</pose></collision>"
        "<collision name='e'><geometry><box><size>1 1 1</size></box></geometry>"
        "<pose> </pose></collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert data["obstacle"][0]["pose"][2] == pytest.approx(math.pi / 2)
    assert any("rotation format 'quat_xyzw'" in m for m in messages)
    assert any("relative_to a frame" in m for m in messages)


def test_a_model_inside_a_model_is_placed_by_both(folder):
    """The pose of the outer model carries the inner one."""
    body = (
        "<model name='outer'><pose>10 0 0 0 0 0</pose><model name='inner'>"
        "<pose>0 5 0 0 0 0</pose><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>0 0 0.5 0 0 0</pose>"
        "</collision></link></model></model>"
    )
    data, _ = read(folder, world_text(body))
    assert data["obstacle"][0]["pose"][:2] == pytest.approx([10.0, 5.0])


def include(uri, pose=""):
    """An `<include>` of an address, with an optional pose."""
    extra = f"<pose>{pose}</pose>" if pose else ""
    return f"<include><uri>{uri}</uri>{extra}</include>"


def make_model(folder, name="thing", body=None, config=True):
    """A model folder with a model.sdf holding a mesh cube, and a model.config."""
    model = folder / "models" / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "cube.dae").write_text(CUBE_DAE)
    text = body or (
        "<sdf version='1.9'><model name='thing'><pose>1 0 0 0 0 0</pose>"
        "<link name='l'><collision name='c'><geometry><mesh><uri>cube.dae</uri>"
        "</mesh></geometry></collision></link></model></sdf>"
    )
    (model / "model.sdf").write_text(text)
    if config:
        (model / "model.config").write_text(
            "<model><sdf version='1.0'>old.sdf</sdf><sdf version='1.9'>model.sdf</sdf>"
            "</model>"
        )
    return model


def test_an_included_model_is_read_with_its_own_pose_or_the_one_asked_for(folder):
    """The model's pose is used; a pose on the include replaces it."""
    make_model(folder)
    data, messages = read(folder, world_text(include("model://thing")))
    assert data["obstacle"][0]["pose"][0] == pytest.approx(1.5, abs=0.03)
    assert messages == []
    data, _ = read(folder, world_text(include("model://thing", "10 20 0 0 0 0")))
    assert data["obstacle"][0]["pose"][:2] == pytest.approx([10.5, 20.5], abs=0.03)


def test_an_include_of_a_model_with_no_config_reads_model_sdf(folder):
    """With no model.config, model.sdf is the file."""
    make_model(folder, config=False)
    data, _ = read(folder, world_text(include("model://thing")))
    assert len(data["obstacle"]) == 1


def test_a_model_config_that_names_a_missing_file_or_is_broken_falls_back(folder):
    """The config is a hint: model.sdf is read when it cannot be followed."""
    model = make_model(folder)
    (model / "model.config").write_text("<model><sdf>gone.sdf</sdf></model>")
    assert len(read(folder, world_text(include("model://thing")))[0]["obstacle"]) == 1
    (model / "model.config").write_text("not xml")
    assert len(read(folder, world_text(include("model://thing")))[0]["obstacle"]) == 1
    (model / "model.config").write_text("<model/>")
    assert len(read(folder, world_text(include("model://thing")))[0]["obstacle"]) == 1


def test_a_fuel_address_is_found_in_the_cache_with_the_newest_version(folder):
    """The cache holds numbered versions; the highest is read."""
    base = folder / "home" / ".gz" / "fuel" / "fuel.gazebosim.org"
    model = base / "openrobotics" / "models" / "some thing"
    for version in ("2", "10", "notes"):
        (model / version).mkdir(parents=True)
    (model / "notes" / "x").mkdir()
    (model / "10" / "cube.dae").write_text(CUBE_DAE)
    (model / "10" / "model.sdf").write_text(
        "<sdf><model name='t'><link name='l'><collision name='c'><geometry><mesh>"
        "<uri>cube.dae</uri></mesh></geometry></collision></link></model></sdf>"
    )
    address = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Some%20Thing"
    data, messages = read(folder, world_text(include(address)))
    assert len(data["obstacle"]) == 1 and messages == []
    pinned = include(address + "/10")
    assert len(read(folder, world_text(pinned))[0]["obstacle"]) == 1


def test_a_fuel_address_that_is_not_a_model_page_is_not_found(folder):
    """Only owner/models/name addresses are mapped to the cache."""
    with pytest.raises(WorldError, match="was not found"):
        read(folder, world_text(include("https://fuel.gazebosim.org/1.0/x/worlds/y")))


@pytest.mark.parametrize(
    ("element", "message"),
    [
        ("<include/>", "an <include> with no <uri>"),
        (include("model://absent"), "include 'model://absent' was not found"),
    ],
)
def test_an_include_that_cannot_be_followed_is_told(folder, element, message):
    """With a real shape beside it, the world is read and the include is a warning."""
    body = element + (
        "<model name='ok'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>0 0 0.5 0 0 0</pose>"
        "</collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    assert any(message in m for m in messages)


def test_an_include_of_an_empty_folder_or_a_broken_file_is_told(folder):
    """No model.sdf, and a model.sdf that is not XML, each give a reason."""
    (folder / "models" / "empty").mkdir(parents=True)
    (folder / "models" / "bad").mkdir(parents=True)
    (folder / "models" / "bad" / "model.sdf").write_text("not xml")
    ok = (
        "<model name='ok'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>0 0 0.5 0 0 0</pose>"
        "</collision></link></model>"
    )
    data, messages = read(
        folder, world_text(include("model://empty") + include("model://bad") + ok)
    )
    assert len(data["obstacle"]) == 1
    assert any("holds no model.sdf or model.config" in m for m in messages)
    assert any("cannot be read" in m for m in messages)


def test_a_model_that_includes_itself_stops_and_says_so(folder):
    """A loop is cut, with a warning, and what else was found is kept."""
    make_model(
        folder,
        "loop",
        "<sdf><model name='a'>"
        + include("model://loop")
        + "<link name='l'><collision name='c'><geometry><box><size>1 1 1</size></box>"
        "</geometry><pose>0 0 0.5 0 0 0</pose></collision></link></model></sdf>",
    )
    data, messages = read(folder, world_text(include("model://loop")))
    assert len(data["obstacle"]) == 1
    assert any("includes itself or goes too deep" in m for m in messages)


def test_an_include_may_name_a_file_and_not_a_folder(folder):
    """A path to the model.sdf itself is read as it is."""
    model = make_model(folder)
    data, _ = read(folder, world_text(include(str(model / "model.sdf"))))
    assert len(data["obstacle"]) == 1


def test_too_many_shapes_make_the_cells_coarser_and_say_so(folder, monkeypatch):
    """With room for one shape, the cells grow to the coarsest and a warning follows."""
    monkeypatch.setattr(sdf, "MAX_SHAPES", 1)
    body = mesh_model(pose="0 0 0 0 0 0") + mesh_model(pose="5 0 0 0 0 0")
    data, messages = read(folder, world_text(body))
    assert any("cells of 0.4 m were used" in m for m in messages)
    assert any("2 shapes; the world may run slowly" in m for m in messages)
    assert len(data["obstacle"]) == 2


def test_a_mesh_too_wide_to_cut_is_left_out_with_the_reason(folder, monkeypatch):
    """The grid limit is a warning, and the other shapes remain."""
    monkeypatch.setattr("demuvo.world.footprint.MAX_CELLS", 10)
    body = mesh_model() + (
        "<model name='ok'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>9 9 0.5 0 0 0</pose>"
        "</collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    assert any("the meshes were left out: " in m for m in messages)


def test_specks_of_a_mesh_are_dropped_and_counted(folder):
    """A post a hundredth of a metre wide is noise, and the warning says so."""
    body = mesh_model(scale="0.01 0.01 2") + (
        "<model name='ok'><link name='l'><collision name='c'><geometry>"
        "<box><size>1 1 1</size></box></geometry><pose>9 9 0.5 0 0 0</pose>"
        "</collision></link></model>"
    )
    data, messages = read(folder, world_text(body))
    assert len(data["obstacle"]) == 1
    assert any("specks of the meshes" in m for m in messages)


def test_the_world_is_sized_round_a_polygon():
    """A polygon's corners set the size, as a box's do."""
    hexagon = np.array(
        [
            (2 + math.cos(k * math.pi / 3), 2 + math.sin(k * math.pi / 3))
            for k in range(6)
        ]
    )
    assert hexagon.shape == (6, 2)
    extents = sdf._extent({"shape": {"name": "polygon", "vertices": hexagon.tolist()}})
    assert extents == pytest.approx(
        (1.0, 2 - math.sqrt(3) / 2, 3.0, 2 + math.sqrt(3) / 2)
    )
