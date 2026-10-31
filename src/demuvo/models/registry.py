"""The one place that lists the models, and the check of their parameters."""

from __future__ import annotations

from dataclasses import fields

from ..verify import check_keys, number, table
from .base import Model
from .diff_drive import DiffDrive

# The one place that lists the models, a new model class needs only an entry here.
MODELS: dict[str, type[Model]] = {cls.name: cls for cls in (DiffDrive,)}


def parse_params(value, model: str, label: str) -> dict[str, float]:
    """Check the `params` table of a robot against the fields of its model.

    Args:
        value: the table read from the file.
        model: name of the robot's model, which must be in `MODELS`.
        label: the robot, for example "robot 'r1'".

    Returns:
        The parameters as keyword arguments for the model class.
    """
    data = table(value, f"{label} params")
    check_keys(data, {f.name for f in fields(MODELS[model])}, f"{label} params")
    return {key: number(data, key, f"{label} params") for key in data}
