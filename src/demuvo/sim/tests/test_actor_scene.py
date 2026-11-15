"""Tests that a lidar, and so a local costmap, sees the actors of a world."""

import numpy as np
import pytest

from demuvo import Simulation, load_world
from demuvo.costmap import LETHAL
from demuvo.sensing import Scene, cast
from demuvo.sim.scene import actor_owner, parts_scene, with_robots
from demuvo.world.shapes import Part

WORLD = """
version = 1

[world]
size = [10.0, 6.0]
seed = 5

[[actor]]
name = "walker"
shape = {{ name = "circle", radius = 0.3 }}
pose = [3.2, 2.5, 0.0]
{motion}{flags}
[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.25 }}
pose = [2.0, 2.5, 0.0]
sensors = {{ lidar = {{ beams = 90, period = 0.01, max_range = 6.0 }} }}
"""
# Straight along x at 1 m/s, so where it is at a time is easy to say.
MOTION = (
    'motion = "demuvo.actors:Waypoints"\n'
    'motion_params = { points = [[8.0, 2.5]], speed = 1.0, mode = "once" }\n'
)


def sim_of(tmp_path, motion="", flags=""):
    """A simulation of one robot facing one actor 1.2 m ahead of it."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(motion=motion, flags=flags))
    return Simulation(load_world(path))


def ahead(sim) -> float:
    """The range of a ray cast along +x from the robot, through the scene now."""
    return float(cast(np.array([2.0, 2.5]), np.zeros(1), sim.scene(), 50.0, 0)[0])


def test_an_actor_is_numbered_after_the_robots():
    """So a robot's own index still names its own body."""
    assert actor_owner(2, 0) == 2
    assert actor_owner(2, 1) == 3
    assert actor_owner(0, 0) == 0


def test_with_robots_gives_each_actor_its_own_owner():
    """Two robots and two actors: owners 0, 1 for the robots and 2, 3 for the actors."""
    empty = parts_scene([])
    disc = [Part(((0.0, 0.0),), 0.5)]
    scene = with_robots(empty, [disc, disc], [disc, disc])
    assert scene.circle_owner.tolist() == [0, 1, 2, 3]
    assert Scene.FIXED not in scene.circle_owner.tolist()


def test_a_ray_hits_an_actor_that_stands_still(tmp_path):
    """The actor is a body in the scene like any other."""
    sim = sim_of(tmp_path)
    assert ahead(sim) == pytest.approx(0.9)  # 3.2 - 0.3 - 2.0


def test_the_scene_follows_the_actor_as_it_moves(tmp_path):
    """The range to it grows as it drives away, step by step."""
    sim = sim_of(tmp_path, MOTION)
    assert ahead(sim) == pytest.approx(0.9)
    for _ in range(100):  # 100 steps of 0.01 s is 1 s, so 1 m further off
        sim.step({})
    assert sim.actor_poses()["walker"].xyt[:2] == pytest.approx([4.2, 2.5])
    assert ahead(sim) == pytest.approx(1.9)


def test_an_actor_is_in_the_robots_local_costmap(tmp_path):
    """What the lidar found of the actor is marked, which is the point of actors."""
    sim = sim_of(tmp_path)
    sim.step({})
    costmap = sim.robots["a"].local_costmap
    assert costmap.cost_at(2.9, 2.5) == LETHAL  # the near face of the actor
    assert 0 < costmap.cost_at(2.6, 2.5) < LETHAL  # inflated short of it
    assert costmap.cost_at(1.0, 2.5) == 0  # clear behind the robot


def test_an_actor_is_not_in_the_global_costmap(tmp_path):
    """The planner plans as if the actor were not there; only the lidar finds it."""
    sim = sim_of(tmp_path)
    assert sim.costmaps["a"].cost_at(3.2, 2.5) == 0


def test_a_non_solid_actor_is_seen_just_the_same(tmp_path):
    """`solid` is about being pushed out, not about being seen."""
    solid = sim_of(tmp_path)
    ghost = sim_of(tmp_path, flags="solid = false\n")
    solid.step({})
    ghost.step({})
    assert ahead(ghost) == pytest.approx(ahead(solid))
    assert ghost.robots["a"].local_costmap.cost_at(2.9, 2.5) == LETHAL


def test_a_robot_still_leaves_out_its_own_body(tmp_path):
    """The skip is the robot's index, which no actor can take."""
    sim = sim_of(tmp_path)
    sim.step({})
    scan = sim.robots["a"].scans()["lidar"]
    assert np.isfinite(scan.ranges).any()
    assert scan.ranges.min() > 0.25  # nothing at its own radius


def test_a_world_with_no_actor_builds_the_same_scene_as_before(tmp_path):
    """The actors are an addition: a world without them is untouched."""
    path = tmp_path / "w.toml"
    path.write_text(
        WORLD.format(motion="", flags="").replace(
            """[[actor]]
name = "walker"
shape = { name = "circle", radius = 0.3 }
pose = [3.2, 2.5, 0.0]
""",
            "",
        )
    )
    sim = Simulation(load_world(path))
    assert not sim.actors
    assert sim.actor_poses() == {}
    assert np.isinf(ahead(sim)) or ahead(sim) > 7.0  # only the far border
