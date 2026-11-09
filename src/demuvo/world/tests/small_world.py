"""A small world file for the tests of one table or one key."""

from demuvo.world import load_world

WORLD = """
version = 1

[world]
size = [5.0, 4.0]
seed = 2

[[robot]]
name = "a"
model = "diff_drive"
shape = {{ name = "circle", radius = 0.1 }}
pose = [1.0, 1.0, 0.0]
{extra}
"""


def small_world(tmp_path, extra=""):
    """Load a small world with one robot and the extra text at the end of the file."""
    path = tmp_path / "w.toml"
    path.write_text(WORLD.format(extra=extra))
    return load_world(path)
