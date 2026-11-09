"""Tests for sensors, estimators, the robot that joins them and its use."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import (
    MODELS,
    DiffDrive,
    Navigation,
    Simulation,
    Twist,
    WorldError,
    load_world,
)
from demuvo.evaluation import result_lines
from demuvo.sensing import (
    DeadReckoning,
    Encoders,
    GroundTruth,
    PerfectPose,
    Reading,
    StepTruth,
    stream,
)

DEMO = Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml"
DEAD = "control_period = 0.05     # optional: seconds between controller runs"
SECOND_ROBOT = """
[[robot]]
name = "a0"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 3.0, 0.0]
estimator = "dead_reckoning"
sensors = { encoders = { noise = 0.05 } }
"""


def world_file(tmp_path, estimator="dead_reckoning", sensors="", extra=""):
    """The demo world with an estimator and sensor settings; return its path."""
    text = DEMO.read_text(encoding="utf-8")
    robot = f'{DEAD}\nestimator = "{estimator}"\n{sensors}'
    text = text.replace(DEAD, robot)
    path = tmp_path / "world.toml"
    path.write_text(text + extra, encoding="utf-8")
    return path


def drive(world, steps=150, dt=0.02, command=Twist(vx=0.6, wz=0.4)):
    """Run a world with one command for every robot; return the simulation."""
    sim = Simulation(world, dt=dt)
    for _ in range(steps):
        sim.step({name: command for name in sim.models})
    return sim


def truth(command=Twist(vx=0.5), dt=0.1):
    """A step truth for a robot at the origin."""
    return StepTruth(dt, Transform2D.from_xyt((1.0, 2.0, 0.3)), command, dt)


# streams


def test_a_stream_depends_on_the_seed_and_the_names_only():
    """The same names give the same numbers; other names or seeds do not."""
    first = stream(7, "r1", "encoders").standard_normal(3)
    assert stream(7, "r1", "encoders").standard_normal(3) == pytest.approx(first)
    for other in (stream(8, "r1", "encoders"), stream(7, "r2", "encoders")):
        assert not np.allclose(other.standard_normal(3), first)


# sensors


def test_the_perfect_pose_reads_the_true_pose():
    """Its data is the true (x, y, angle)."""
    reading = PerfectPose(None, None).read(truth())
    assert reading == Reading("perfect_pose", 0.1, reading.data)
    assert reading.data == pytest.approx([1.0, 2.0, 0.3])


def encoders(rng_seed=1, **options):
    """Encoders on a diff-drive with their own stream."""
    return Encoders(DiffDrive(), stream(rng_seed, "r1", "encoders"), **options)


def test_encoders_with_no_settings_read_the_ideal_wheels():
    """No noise and no rounding: exactly what the model says."""
    model = DiffDrive()
    command = Twist(vx=0.5, wz=0.2)
    got = Encoders(model, stream(1, "r1", "encoders")).read(truth(command)).data
    assert got == pytest.approx(model.encoders(command, 0.1))


def test_noise_changes_a_turning_wheel_and_leaves_a_still_one_alone():
    """The error is a fraction of the increment, so zero stays zero."""
    noisy = encoders(noise=0.1).read(truth(Twist(vx=0.5))).data
    ideal = DiffDrive().encoders(Twist(vx=0.5), 0.1)
    assert not np.allclose(noisy, ideal)
    assert encoders(noise=0.1).read(truth(Twist())).data == pytest.approx([0.0, 0.0])


def test_the_same_stream_repeats_and_another_robot_differs():
    """Same seed and names: same readings. A different robot name: different."""
    a = encoders(noise=0.1).read(truth()).data
    assert encoders(noise=0.1).read(truth()).data == pytest.approx(a)
    other = Encoders(DiffDrive(), stream(1, "r2", "encoders"), noise=0.1)
    assert not np.allclose(other.read(truth()).data, a)


def test_rounding_keeps_what_it_drops_for_the_next_step():
    """Every reading is whole ticks, and the total is within one tick of the true."""
    ticks = 100
    sensor = encoders(ticks_per_rev=ticks)
    command = Twist(vx=0.07, wz=0.03)  # a step turns the wheels less than one tick
    total = np.zeros(2)
    for _ in range(500):
        data = sensor.read(truth(command, 0.1)).data
        assert np.allclose(data * ticks / math.tau, np.round(data * ticks / math.tau))
        total += data
    ideal = 500 * DiffDrive().encoders(command, 0.1)
    assert np.abs(total - ideal).max() <= math.tau / ticks
    assert total.max() > 0  # without the carry, every step would round to zero


def test_negative_settings_are_refused():
    """Noise and ticks cannot be below zero."""
    with pytest.raises(ValueError, match="negative"):
        encoders(noise=-0.1)
    with pytest.raises(ValueError, match="negative"):
        encoders(ticks_per_rev=-1)


# estimators


def test_ground_truth_takes_the_pose_from_the_sensor():
    """It passes the perfect pose through."""
    reading = PerfectPose(None, None).read(truth())
    estimator = GroundTruth(DiffDrive(), Transform2D.identity())
    assert estimator.update(Twist(), 0.1, [reading]).pose.xyt == pytest.approx(
        [1.0, 2.0, 0.3]
    )


def test_an_estimator_without_its_reading_says_so():
    """No reading from the sensor it needs is an error, not a guess."""
    estimator = GroundTruth(DiffDrive(), Transform2D.identity())
    with pytest.raises(ValueError, match="perfect_pose"):
        estimator.update(Twist(), 0.1, [])


def test_dead_reckoning_adds_up_the_steps():
    """Two straight 5 cm steps from (1, 1, 0) end at (1.1, 1, 0)."""
    model = DiffDrive(wheel_radius=0.1)
    estimator = DeadReckoning(model, Transform2D.from_xyt((1.0, 1.0, 0.0)))
    reading = Reading("encoders", 0.1, np.array([0.5, 0.5]))
    estimator.update(Twist(), 0.1, [reading])
    estimate = estimator.update(Twist(), 0.1, [reading])
    assert estimate.pose.xyt == pytest.approx([1.1, 1.0, 0.0])
    assert estimate.cov is None
    assert estimate.map is None


# the robot in the simulation


def test_the_default_estimate_is_the_true_pose():
    """With no estimator in the file the controller sees the truth."""
    sim = drive(load_world(DEMO), steps=50)
    assert sim.estimates["r1"].pose.xyt == pytest.approx(sim.poses["r1"].xyt)
    assert sim.robots["r1"].spec.estimator == "ground_truth"


def test_dead_reckoning_without_noise_follows_the_true_pose(tmp_path):
    """With exact wheels the estimate and the truth agree along a turn."""
    sim = drive(load_world(world_file(tmp_path)))
    assert sim.estimates["r1"].pose.xyt == pytest.approx(sim.poses["r1"].xyt, abs=1e-9)
    assert abs(sim.poses["r1"].xyt[2]) > 1.0  # it did turn


def test_noisy_wheels_make_the_estimate_drift(tmp_path):
    """The estimate leaves the truth, and the same seed gives the same estimate."""
    sensors = "sensors = { encoders = { noise = 0.05, ticks_per_rev = 1024 } }"
    path = world_file(tmp_path, sensors=sensors)
    first = drive(load_world(path))
    again = drive(load_world(path))
    error = np.hypot(*(first.estimates["r1"].pose.xyt[:2] - first.poses["r1"].xyt[:2]))
    assert 0.001 < error < 5.0
    assert first.estimates["r1"].pose.xyt == pytest.approx(
        again.estimates["r1"].pose.xyt
    )


def test_the_robot_setting_replaces_the_world_default_key_by_key(tmp_path):
    """The world says noise 0.1; one robot says 0 and follows the truth."""
    shared = "\n[sensors.encoders]\nnoise = 0.1\n"
    noisy = drive(load_world(world_file(tmp_path, extra=shared)))
    assert not np.allclose(noisy.estimates["r1"].pose.xyt, noisy.poses["r1"].xyt)
    own = "sensors = { encoders = { noise = 0.0 } }"
    exact = drive(load_world(world_file(tmp_path, sensors=own, extra=shared)))
    assert exact.estimates["r1"].pose.xyt == pytest.approx(
        exact.poses["r1"].xyt, abs=1e-9
    )


def test_adding_a_robot_does_not_change_the_noise_of_another(tmp_path):
    """The streams come from the names, so r1 is the same with or without a0."""
    sensors = "sensors = { encoders = { noise = 0.05 } }"
    alone = drive(load_world(world_file(tmp_path, sensors=sensors)))
    crowd = drive(load_world(world_file(tmp_path, sensors=sensors, extra=SECOND_ROBOT)))
    assert crowd.estimates["r1"].pose.xyt == pytest.approx(
        alone.estimates["r1"].pose.xyt
    )
    assert not np.allclose(
        crowd.estimates["a0"].pose.xyt, crowd.estimates["r1"].pose.xyt
    )


def test_a_wheel_keeps_turning_against_a_wall(tmp_path):
    """The encoders read the command, so the estimate runs on past the border."""
    sim = drive(load_world(world_file(tmp_path)), steps=1300, command=Twist(vx=1.0))
    assert sim.poses["r1"].xyt[0] < 10.0  # stopped by the border
    assert sim.estimates["r1"].pose.xyt[0] > 12.0


# a model with no wheel form


@dataclass(frozen=True)
class Plain:
    """A model that has no wheels to read."""

    name: ClassVar[str] = "plain"
    uses: ClassVar[tuple[str, ...]] = ("vx", "wz")

    def clip(self, command):
        """No limits."""
        return command

    def step(self, pose, command, dt):
        """Same motion as a diff-drive."""
        return DiffDrive().step(pose, command, dt)


def test_a_model_without_encoders_falls_back_to_ground_truth(tmp_path, monkeypatch):
    """The file asks for dead reckoning; it warns and the robot uses the truth."""
    monkeypatch.setitem(MODELS, "plain", Plain)
    path = world_file(tmp_path)
    text = path.read_text().replace('model = "diff_drive"', 'model = "plain"')
    path.write_text(text)
    with pytest.warns(UserWarning, match="model 'plain' has no encoder form"):
        world = load_world(path)
    assert world.robots[0].estimator == "ground_truth"
    sim = drive(world, steps=20)
    assert sim.estimates["r1"].pose.xyt == pytest.approx(sim.poses["r1"].xyt)


def test_a_sensor_setting_the_estimator_ignores_warns(tmp_path):
    """Encoder noise on a ground-truth robot has no effect, and the file says so."""
    path = world_file(
        tmp_path,
        estimator="ground_truth",
        sensors="sensors = { encoders = { noise = 0.05 } }",
    )
    with pytest.warns(UserWarning, match="encoders is set but estimator 'ground_t"):
        load_world(path)


# the world file


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"estimator": "kalman"}, "unknown estimator 'kalman'"),
        ({"sensors": "sensors = { sonar = {} }"}, "unknown key 'sonar'"),
        (
            {"sensors": "sensors = { encoders = { speed = 1 } }"},
            "unknown key 'speed'",
        ),
        (
            {"sensors": "sensors = { encoders = { noise = -1 } }"},
            "'noise' must not be negative",
        ),
        ({"sensors": "sensors = { encoders = 3 }"}, "must be a table"),
    ],
)
def test_a_bad_estimator_or_sensor_setting_is_refused(tmp_path, kwargs, message):
    """The error names the robot and the key."""
    with pytest.raises(WorldError, match=message):
        load_world(world_file(tmp_path, **kwargs))


def test_a_bad_world_default_is_refused(tmp_path):
    """The world's own sensor table is checked the same way."""
    with pytest.raises(WorldError, match="file sensors.encoders: unknown key"):
        load_world(world_file(tmp_path, extra="\n[sensors.encoders]\nbad = 1\n"))


# what the controller and the result see


class Recorder:
    """A controller that notes what it is shown and drives forward."""

    seen: ClassVar[list] = []

    def command(self, obs):
        """Keep the observation."""
        self.seen.append(obs)
        return Twist(vx=0.5)


def recorded(tmp_path, steps=60, extra=""):
    """Run the noisy world with the recorder.

    Returns the simulation, the navigation, and for each step the true pose and
    the estimate as they were when the controller was called.
    """
    Recorder.seen.clear()
    path = world_file(
        tmp_path, sensors=extra + "sensors = { encoders = { noise = 0.1 } }"
    )
    text = (
        path.read_text()
        .replace(
            'controller = "demuvo.controllers:PurePursuit"',
            'controller = "demuvo.sensing.tests.test_sensing:Recorder"',
        )
        .replace("controller_params = { speed = 0.5, lookahead = 0.6 }", "")
    )
    path.write_text(text)
    sim = Simulation(load_world(path), dt=0.05)
    nav = Navigation(sim.world)
    before = []
    for _ in range(steps):
        before.append((sim.poses["r1"], sim.estimates["r1"].pose))
        sim.step(nav(sim))
    return sim, nav, before


def test_the_controller_sees_the_estimate_and_not_the_truth(tmp_path):
    """Each observation holds the estimate of that moment, which is not the truth."""
    _, _, before = recorded(tmp_path)
    assert len(Recorder.seen) == len(before)
    for obs, (_, estimate) in zip(Recorder.seen, before):
        assert obs.pose.xyt == pytest.approx(estimate.xyt, abs=1e-12)
    truth_xyt, estimate = before[-1][0].xyt, before[-1][1].xyt
    assert np.hypot(*(truth_xyt[:2] - estimate[:2])) > 1e-4


def test_the_controller_can_see_the_truth_while_the_estimator_runs(tmp_path):
    """control_pose "truth": the controller gets the true pose, the estimate drifts."""
    sim, nav, before = recorded(tmp_path, extra='control_pose = "truth"\n')
    for obs, (truth_pose, _) in zip(Recorder.seen, before):
        assert obs.pose.xyt == pytest.approx(truth_pose.xyt, abs=1e-12)
    nav.judge.update(sim)
    assert any(
        "pose from dead_reckoning" in line for line in result_lines(sim, nav.judge)
    )
    assert not np.allclose(sim.estimates["r1"].pose.xyt, sim.poses["r1"].xyt)


def test_control_pose_must_be_estimate_or_truth(tmp_path):
    """Any other word is refused, naming the robot."""
    with pytest.raises(WorldError, match="robot 'r1': 'control_pose' must be"):
        load_world(world_file(tmp_path, sensors='control_pose = "both"'))


def test_the_path_counter_changes_with_each_new_path(tmp_path):
    """A plan, a new goal and an outside path each add one."""
    sim, nav, _ = recorded(tmp_path, steps=5)
    assert {obs.path_changes for obs in Recorder.seen} == {1}
    nav.set_goal("r1", Transform2D.from_xyt((5.0, 5.0, 0.0)))
    sim.step(nav(sim))
    assert Recorder.seen[-1].path_changes == 2
    nav.set_path("r1", [(1.0, 1.0, 0.0), (2.0, 1.0, 0.0)])
    sim.step(nav(sim))
    assert Recorder.seen[-1].path_changes == 3
    sim.step(nav(sim))
    assert Recorder.seen[-1].path_changes == 3


def test_the_result_says_how_far_the_estimate_is(tmp_path):
    """A robot that is not on ground truth gets a line with the error."""
    sim, nav, _ = recorded(tmp_path)
    nav.judge.update(sim)
    lines = result_lines(sim, nav.judge)
    line = next(line for line in lines if "pose from dead_reckoning" in line)
    assert line.startswith("r1 pose from dead_reckoning, ")
    assert line.endswith(" m from the true pose at the end")


def test_the_result_has_no_such_line_on_ground_truth():
    """The default adds nothing to the result."""
    sim = Simulation(load_world(DEMO))
    nav = Navigation(sim.world)
    sim.step(nav(sim))
    assert not any("pose from" in line for line in result_lines(sim, nav.judge))
