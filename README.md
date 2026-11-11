<p align="center">
  <img src="https://raw.githubusercontent.com/dcapacho3/demuvo/main/docs/logo.svg" alt="demuvo" width="560">
</p>

<p align="center">
  <a href="https://github.com/dcapacho3/demuvo/actions/workflows/ci_pytest.yml"><img src="https://github.com/dcapacho3/demuvo/actions/workflows/ci_pytest.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license">
</p>

**A deterministic 2D robot simulator.**

demuvo is a flat 2D world that runs with no physics engine: robots move exactly as their
kinematic model says. You describe a world in a file or draw it in a window, put robots in
it, choose how each one senses, plans and steers, and press play. You see the path before
the robot moves, the costmap it plans on, and what its lidar sees. The same world and the
same seed always give the same run, so a result can be repeated and compared.

> **Work in progress.** demuvo is under active development and is not released yet. The
> API and the world file may still change. The [to do list](#to-do) shows what is next.

![A session in the sandbox: draw a box, a circle and a line, rename one, give the robot a
goal, drag an obstacle, and play the run.](docs/sandbox.gif)

## Install and open

```bash
pip install -e .
sudo apt install libglfw3 libgl1      # the window; Ubuntu and Debian
demuvo                                # a blank world to build in
demuvo examples/worlds/demo.toml      # open a world file; Space runs it
demuvo --install-desktop              # optional: add demuvo and its icon to your launcher
```

`demuvo world.toml --bench` opens a world fixed, so every run is the same. `--check` checks
a file and opens nothing. `--result` prints how a run went when the window closes. After
the demo robot has driven to the star it prints:

```text
r1 task 1: reached at 19.26 s
r1 drove 8.86 m
```

## How to use it

**Build a world in the window.** Pick a tool from the tool bar, or press its key, and use
the pointer:

| Tool | Key | What it does |
| --- | --- | --- |
| Select | V | click a robot or obstacle; drag it to move it |
| Goal | G | click the floor to set the selected robot's goal; Shift and click adds a waypoint |
| Box, Circle | B, C | drag on the floor |
| Polygon, Line | P, L | click the corners; double click, right click or Enter to finish |
| Add robot, Delete | Ctrl+N, Delete | add a robot, delete the selected one |

Select a robot, an obstacle, or nothing, and the inspector on the right shows its form:
pose, goals, body, model, planner, smoother, controller, sensors, costmaps. Every field
applies as you leave it. Double click a name in the scene list (or press F2) to rename it.
Ctrl+Z and Ctrl+Y undo and redo. Space plays, S steps, R resets. Warnings from a run appear
in the log panel at the bottom. [The sandbox](docs/sandbox.md) has everything, with all the
keys.

**Save and open.** The File menu saves the world as a TOML file and opens one. The world
file is plain text; [its format](docs/world-file.md) covers obstacles, robots, goals,
costmaps, lidar and time. A file is checked when it loads, and a mistake is named by object
and key.

**Run it from a script.**

```python
from demuvo import Run, edit

world = edit.new_world()  # 10 by 6 m, one robot, r1
world = edit.set_goal(world, "r1", (8.0, 1.0, 0.0))
run = Run(world)
run.until_done(60.0)
print("\n".join(run.result()))
```

[Python](docs/python.md) lists the edit functions.

**Run many seeds, with no window.**

```bash
demuvo examples/worlds/demo.toml --seeds 4 --jobs 4
```

```text
seed  done  time (s)  collisions
   7   yes     19.26           0
   8   yes     19.26           0
   9   yes     19.26           0
  10   yes     19.26           0
4 of 4 runs done
```

The demo world has no noise, so the rows match. A world with encoder or lidar noise gives
rows that differ, and the same seed always gives the same row.

## How to add your own algorithm

A planner, a smoother or a controller is a class. Name it as `module:Class` in the world
file and demuvo builds it and runs it beside the built-in ones:

```toml
[[robot]]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 1.0, 0.0]
planner = "breadth_first:BreadthFirst"          # your class, as module:Class
controller = "turn_then_drive:TurnThenDrive"    # and this one
controller_params = { speed = 0.6 }
```

```bash
PYTHONPATH=examples/plugins demuvo examples/worlds/plugins.toml
```

![Two robots race along two lanes. r1 uses a planner and a controller from
examples/plugins; r2 uses A* and pure pursuit.](docs/plugins.gif)

A planner has `configure(costmap)` and `plan(start, goal)`, which returns an array of x, y,
angle. A smoother has `configure(costmap)` and `smooth(path)`. A controller has
`command(obs)` and returns a `Twist`. The two classes in `examples/plugins` are about thirty
lines each. [Your own algorithms](docs/plugins.md) has the details, and how to make a class
show up in the window's lists.

## What is inside

- **A window** to build a world, run it, undo, rename, save and open. It is drawn by
  demuvo on numpy and shown through the system's GLFW and OpenGL, so it needs no Python
  package for the window.
- **A flat 2D world** with no physics engine: obstacles (box, circle, polygon, wall), solid
  to robots, and maps read from a `map.yaml`.
- **Robots** with a differential-drive model driven by a ROS-style `Twist`, wheel encoders
  with noise, and a dead-reckoning estimator.
- **Sensing:** any number of lidars on a robot, each with its mount, noise and scan rate,
  and a local costmap that the robot builds from their scans.
- **Planning and control:** `StraightLine` and `AStar` planners, a `LineOfSight` smoother and
  a `PurePursuit` controller, on a global costmap.
- **Exact time:** a step length and a controller rate that you set apart, and results that
  repeat to the last bit on one machine.
- **Scripting:** a `Run` to step a world from a script or a benchmark, edit functions that
  build and save a world, and a recorder that writes GIFs of any world with no window.

## To do

- [ ] More robot models, each with the controller and the encoders that suit it: car-like
  (Ackermann), mecanum, skid-steer and forklift.
- [ ] An extended Kalman filter (EKF), written from scratch, as a robot's estimator.
- [ ] SLAM, written from scratch, so a robot builds a map and finds itself in it.
- [ ] A map generator: a seeded Gaussian random field turned into obstacles, so each seed
  makes a different, repeatable world (clutter, caves), with start and goal checked to be
  connected.
- [ ] An optional terrain layer, also from a seeded Gaussian map: friction, slip or a speed
  limit that makes a world harder. Off by default, so the ideal world stays as it is.
- [ ] Swarms: many robots, each running the same rule from its own sensors and its
  neighbours, with collisions and time reported over seeds.
- [ ] A local planner that steers round what the lidar sees and the map does not show.
- [ ] Moving actors, and replanning while the robot drives.
- [ ] A benchmark runner that compares planners and controllers over worlds and seeds, with
  success, time, path length and collision metrics, a table and a CSV file.
- [ ] Run logs and replay.
- [ ] Bridges to other software, ROS 2 first, so a program outside demuvo can drive a robot.
- [ ] In the window: reshape the corners of a polygon or line, and more body shapes.
- [ ] A Debian package, and a faster core in C++ and Rust under the same Python interface.
- [ ] Later: a reset and step interface so learning-based methods, reinforcement learning
  among them, can drive a world, and experiments with learned planners and controllers,
  including training and distillation.

## Documentation

| Page | What is in it |
| --- | --- |
| [The sandbox](docs/sandbox.md) | the window: tools, forms, files, undo, the log, all the keys |
| [Your own algorithms](docs/plugins.md) | write a planner, a smoother or a controller |
| [The world file](docs/world-file.md) | every key: obstacles, robots, tasks, costmaps, lidar, time |
| [Python](docs/python.md) | `Run`, the edit functions, seeds, what stays exact |

## Develop

```bash
git clone https://github.com/dcapacho3/demuvo.git
cd demuvo
pip install -e . --group dev     # needs pip 25.1 or later
pre-commit install
python -m pytest
```

The window tests need `libglfw3`, `libgl1` and `xvfb`, and are skipped where they are
missing.

Each part of the code is a folder under `src/demuvo/` (`world`, `models`, `costmap`,
`sim`, `planners`, `controllers`, `nav`, `evaluation`, `run`, `gui`), plus the files
`options.py`, `edit.py` and `runner.py`, and its tests sit in a `tests` folder inside it.

Pose maths comes from [robomath](https://github.com/dcapacho3/robomath).

## License

Apache 2.0. Copyright 2026 David Capacho. See [LICENSE](LICENSE).
