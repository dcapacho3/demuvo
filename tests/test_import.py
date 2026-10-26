"""The package and its dependencies must import."""

import robomath

import demuvo


def test_import():
    """Demuvo and robomath must exist in the environment."""
    assert demuvo is not None
    assert robomath is not None
