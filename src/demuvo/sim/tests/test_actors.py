"""Tests for the actors holder: motions built from the world, poses at a time."""

import math

import numpy as np
import pytest

from demuvo.actors import Still, Waypoints
from demuvo.sim.actors import Actors
from demuvo.world import WorldError
from demuvo.world.tests.small_world import small_world

WALKER = """
[[actor]]
name = "walker"
shape = {{ name = "circle", radius = 0.25 }}
pose = [1.0, 3.0, 0.0]
motion = "demuvo.actors:Waypoints"
motion_params = {{ points = [[3.0, 3.0]], speed = 0.5, mode = "{mode}" }}
"""

CRATE = """
[[actor]]
name = "crate"
shape = { name = "box", size = [0.4, 0.4] }
pose = [4.0, 1.0, 0.3]
"""


def actors_of(tmp_path, extra) -> Actors:
    """The actors of the small world with this extra text."""
    return Actors(small_world(tmp_path, extra))


def test_a_world_with_no_actor_has_an_empty_holder(tmp_path):
    """Nothing to build, nothing to move; the holder is false."""
    actors = actors_of(tmp_path, "")
    assert not actors
    assert len(actors) == 0
    assert actors.poses_at(1.0) == {}
    assert actors.parts_at(1.0) == {}


def test_an_actor_with_no_motion_stands_still(tmp_path):
    """It gets `Still` and its pose is the file's, whatever the time."""
    actors = actors_of(tmp_path, CRATE)
    assert isinstance(actors.motions["crate"], Still)
    for t in (0.0, 1.0, 1e6):
        assert actors.poses_at(t)["crate"].xyt == pytest.approx([4.0, 1.0, 0.3])


def test_a_motion_is_built_from_the_params_and_configured_at_the_start(tmp_path):
    """The class the file names, with its settings, leaving from the actor's pose."""
    actors = actors_of(tmp_path, WALKER.format(mode="once"))
    motion = actors.motions["walker"]
    assert isinstance(motion, Waypoints)
    assert motion.speed == 0.5
    assert actors.poses_at(0.0)["walker"].xyt == pytest.approx([1.0, 3.0, 0.0])
    assert actors.poses_at(2.0)["walker"].xyt == pytest.approx([2.0, 3.0, 0.0])


def test_poses_and_parts_come_by_name_in_file_order(tmp_path):
    """Several actors, and each body is cut into parts at where it is now."""
    actors = actors_of(tmp_path, WALKER.format(mode="loop") + CRATE)
    assert list(actors.poses_at(0.0)) == ["walker", "crate"]
    parts = actors.parts_at(2.0)
    assert set(parts) == {"walker", "crate"}
    (circle,) = parts["walker"]
    assert circle.radius == 0.25
    assert circle.vertices[0] == pytest.approx((2.0, 3.0))
    (box,) = parts["crate"]
    assert box.radius == 0.0 and len(box.vertices) == 4


def test_the_pose_is_a_function_of_time_not_of_the_calls(tmp_path):
    """Asking at 10 s, then at 2 s, gives the pose of 2 s."""
    actors = actors_of(tmp_path, WALKER.format(mode="bounce"))
    late = actors.poses_at(10.0)["walker"].xyt.copy()
    early = actors.poses_at(2.0)["walker"].xyt
    assert early == pytest.approx([2.0, 3.0, 0.0])
    assert actors.poses_at(10.0)["walker"].xyt.tolist() == late.tolist()


def test_each_actor_gets_its_own_stream_from_the_world_seed(tmp_path, monkeypatch):
    """The stream depends on the seed and the actor's name, not on the other actors."""
    seen = {}

    def configure(self, start, rng):
        seen[self.points[0][0]] = rng.random()

    monkeypatch.setattr(Waypoints, "configure", configure)
    two = WALKER.format(mode="loop") + WALKER.format(mode="loop").replace(
        "walker", "other"
    ).replace("[3.0, 3.0]", "[3.5, 3.0]")
    actors_of(tmp_path, two)
    first = dict(seen)
    seen.clear()
    actors_of(tmp_path, WALKER.format(mode="loop"))
    assert seen[3.0] == first[3.0]
    assert first[3.0] != first[3.5]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            WALKER.format(mode="once").replace("demuvo.actors:Waypoints", "nowhere:X"),
            "actor 'walker': cannot load motion 'nowhere:X'",
        ),
        (
            CRATE + 'motion = "demuvo.planners:StraightLine"\n',
            "actor 'crate': motion 'demuvo.planners:StraightLine' has no 'pose_at'",
        ),
        (
            WALKER.format(mode="once").replace("speed = 0.5", "speed = 0"),
            "actor 'walker': cannot load motion .*speed must be a positive",
        ),
        (
            WALKER.format(mode="once").replace("[3.0, 3.0]", "[1.0, 3.0]"),
            "actor 'walker': motion 'demuvo.actors:Waypoints': points: two in a row",
        ),
    ],
)
def test_a_motion_that_cannot_be_built_names_the_actor(tmp_path, extra, message):
    """Both a class that will not load and one that refuses its start pose."""
    with pytest.raises(WorldError, match=message):
        actors_of(tmp_path, extra)


def test_the_walker_turns_round_in_bounce_mode(tmp_path):
    """A sanity check through the holder of the motion's direction of travel."""
    actors = actors_of(tmp_path, WALKER.format(mode="bounce"))
    assert actors.poses_at(6.0)["walker"].xyt[2] == pytest.approx(math.pi)
    assert np.allclose(actors.poses_at(8.0)["walker"].xyt[:2], [1.0, 3.0])
