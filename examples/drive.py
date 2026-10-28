"""Drive the robot of a world file in a circle and watch it.

python examples/drive.py                      open a window
python examples/drive.py --gif circle.gif     save a GIF, open nothing
"""

import argparse
from pathlib import Path

from demuvo import Simulation, Twist, load_world
from demuvo.view import play, record

DEMO = Path(__file__).parent / "worlds" / "demo.toml"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("world", nargs="?", default=DEMO, help="world file (TOML)")
parser.add_argument("--gif", help="save a GIF here and open no window")
parser.add_argument("--steps", type=int, default=250)
args = parser.parse_args()

sim = Simulation(load_world(args.world))


def commands(_):
    """Drive in a circle of 0.5 m radius: 0.5 m/s forward and 1 rad/s turn."""
    return {"r1": Twist(vx=0.5, wz=1.0)}


if args.gif:
    record(sim, commands, args.steps, args.gif)
else:
    play(sim, commands, args.steps)
