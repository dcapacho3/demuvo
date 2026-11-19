"""World text that the simulation tests build in the same way."""

HEAD = """
version = 1
[world]
size = [10.0, 6.0]
seed = 1
dt = 0.1
"""


def robot(name, x, y, angle=0.0, *, radius=0.5, extra=""):
    """A `[[robot]]` block: a disc, with any extra lines of the table after it."""
    return f"""
[[robot]]
name = "{name}"
model = "diff_drive"
shape = {{ name = "circle", radius = {radius} }}
pose = [{x}, {y}, {angle}]
{extra}"""
