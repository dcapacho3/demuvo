"""The test that every model with a wheel form must pass."""

import math

import numpy as np
import pytest
from robomath import Transform2D

from demuvo import MODELS, Twist

WHEELED = {
    name: cls
    for name, cls in MODELS.items()
    if hasattr(cls, "encoders") and hasattr(cls, "from_encoders")
}


def test_at_least_one_model_is_checked():
    """Without this the test below could pass by checking nothing."""
    assert "diff_drive" in WHEELED


@pytest.mark.filterwarnings("ignore:.*ignores vy")
@pytest.mark.parametrize("name", sorted(WHEELED))
def test_the_wheels_give_back_the_motion_of_the_step(name):
    """`from_encoders(encoders(c, dt))` is the step that `step(c, dt)` makes."""
    model = WHEELED[name]()
    rng = np.random.default_rng(1)
    start = Transform2D.from_xyt((1.0, 2.0, 0.7))
    for _ in range(50):
        command = model.clip(
            Twist(vx=rng.uniform(-1, 1), vy=rng.uniform(-1, 1), wz=rng.uniform(-2, 2))
        )
        dt = rng.uniform(0.001, 0.2)
        by_wheels = start.rplus(model.from_encoders(model.encoders(command, dt)))
        by_model = model.step(start, command, dt)
        assert by_wheels.xyt == pytest.approx(by_model.xyt, abs=1e-9)


def test_a_model_without_vy_still_passes_with_a_sideways_request():
    """The contract is on the clipped command, so ignored fields do not matter."""
    model = WHEELED["diff_drive"]()
    with pytest.warns(UserWarning, match="ignores vy"):
        command = model.clip(Twist(vx=0.5, vy=0.3, wz=math.pi / 8))
    assert command.vy == 0.0
