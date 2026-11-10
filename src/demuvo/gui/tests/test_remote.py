"""Tests for the simulation in a process of its own."""

import multiprocessing
import threading
import time

import numpy as np
import pytest

from demuvo import Run, edit
from demuvo.gui import remote as module
from demuvo.gui.remote import RemoteRun

from .fakes import EXAMPLES

DEMO, LIDAR = EXAMPLES / "demo.toml", EXAMPLES / "lidar.toml"


@pytest.fixture(name="pipe")
def served():
    """A worker running `serve` in a thread, and the end of the pipe to talk to it."""
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(target=module.serve, args=(child,), daemon=True)
    thread.start()
    yield parent, thread
    if thread.is_alive():
        parent.send(("stop",))
        thread.join(timeout=5)


def wait_for(run, seconds=10.0):
    """Poll until the worker's answer to `advance` has come."""
    end = time.perf_counter() + seconds
    while not run.poll():
        assert time.perf_counter() < end, "no answer from the worker"
        time.sleep(0.002)


def test_a_remote_run_starts_at_time_zero_with_the_poses_of_the_world(pipe):
    """The first snapshot comes with the load."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    assert run.time == 0.0
    assert run.poses["r1"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert run.sim.dt == 0.01 and run.sim.costmaps["r1"] is not None
    assert run.world.robots[0].name == "r1"


def test_steps_in_another_process_give_exactly_the_poses_of_a_run_here(pipe):
    """The same code on the same world: bit for bit."""
    far, near = RemoteRun(str(LIDAR), connection=pipe[0]), Run(str(LIDAR))
    far.step(300)
    near.step(300)
    assert far.time == near.time
    assert np.array_equal(far.poses["r1"], near.poses["r1"])
    assert np.array_equal(far.path("r1"), near.path("r1"))
    assert len(far.scans("r1")) == 2
    assert np.array_equal(far.scans("r1")[0].ranges, near.scans("r1")[0].ranges)
    assert far.local_costmap("r1") is not None
    assert np.array_equal(far.local_costmap("r1").data, near.local_costmap("r1").data)


def test_advance_does_not_wait_and_the_answer_arrives_later(pipe):
    """The call returns at once; polling brings the new time."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    assert run.advance(0.3) == 0
    wait_for(run)
    assert run.time == pytest.approx(0.3, abs=0.011)
    assert run.poll() is False


def test_time_asked_for_while_the_worker_is_busy_is_owed_and_sent_when_it_is_idle(pipe):
    """Two requests in a row: one in flight and one owed; both get done."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    run.advance(0.2)
    run.advance(0.2)
    wait_for(run)
    run.advance(0.0)
    wait_for(run)
    assert run.time == pytest.approx(0.4, abs=0.02)


def test_one_request_is_capped_so_a_slow_worker_cannot_spiral(pipe):
    """Ten owed seconds become at most MAX_REQUEST of simulated time."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    run.advance(10.0)
    wait_for(run)
    assert run.time == pytest.approx(module.MAX_REQUEST, abs=0.011)


def test_a_command_after_an_advance_waits_for_its_answer_first(pipe):
    """Step while an advance is in flight: both happen, in order."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    run.advance(0.1)
    run.step(10)
    assert run.time == pytest.approx(0.2, abs=0.011)


def test_reset_and_a_new_world_start_again_from_time_zero(pipe):
    """The poses and costmaps follow the new world; owed time is forgotten."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    run.step(50)
    run.reset()
    assert run.time == 0.0
    world = edit.add_robot(edit.new_world(), "r2", (1.0, 1.0, 0.0))
    run.set_world(world)
    assert set(run.poses) == {"r1", "r2"} and run.world is world
    assert set(run.sim.costmaps) == {"r1", "r2"} and run.time == 0.0


def test_the_result_and_the_finish_come_from_the_worker(pipe):
    """The demo robot reaches its goal; until_done says so."""
    run = RemoteRun(str(DEMO), connection=pipe[0])
    assert run.finished is False
    assert run.until_done(60.0) is True
    assert any("reached at" in line for line in run.result())


def test_a_world_in_memory_is_loaded_and_a_step_length_of_its_own_is_used(pipe):
    """A `World` object, with dt given."""
    world = edit.new_world()
    run = RemoteRun(world, dt=0.025, connection=pipe[0])
    assert run.sim.dt == 0.025
    run.step(4)
    assert run.time == pytest.approx(0.1)


def test_close_stops_the_worker(pipe):
    """The loop ends, also when an advance is still in flight."""
    parent, thread = pipe
    run = RemoteRun(str(DEMO), connection=parent)
    run.advance(0.1)
    run.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_a_real_worker_process_gives_the_same_poses_and_stops():
    """Spawn a process, step it, compare with a run here, and close it."""
    far = RemoteRun(str(DEMO))
    try:
        far.step(200)
        near = Run(str(DEMO))
        near.step(200)
        assert np.array_equal(far.poses["r1"], near.poses["r1"])
    finally:
        far.close()
    assert not far._process.is_alive()  # pylint: disable=protected-access
