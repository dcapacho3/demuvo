"""Tests for the motion of an actor that names none."""

import numpy as np
import pytest
from robomath import Transform2D

from demuvo.actors import Still

START = Transform2D.from_xyt(np.array([2.0, 4.0, 1.0]))


def test_it_stays_at_the_start_pose_for_good():
    """Whatever the time, and whichever order the times come in."""
    motion = Still()
    motion.configure(START, np.random.default_rng(0))
    for t in (0.0, 7.5, 1e6, 0.5):
        assert motion.pose_at(t).xyt == pytest.approx([2.0, 4.0, 1.0])


def test_pose_at_before_configure_is_refused():
    """It does not know where it was put yet."""
    with pytest.raises(RuntimeError, match="configure"):
        Still().pose_at(0.0)
