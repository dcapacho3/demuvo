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
kinematic model says. A world is described in a file or drawn in a window, and each robot
is given its sensors, planner and controller. The window shows the planned path before a
robot moves, the costmap it plans on, and what its lidar sees. The same world and the same
seed always give the same run, so a result can be repeated and compared.

> **Work in progress.** demuvo is under active development and is not released yet. The
> API and the world file may still change. The [to do list](#to-do) shows what is next.

![A session in the sandbox: draw a box, a circle and a line, rename one, give the robot a
goal, drag an obstacle, and play the run.](docs/sandbox.gif)

## Installation

```bash
pip install -e .
sudo apt install libglfw3 libgl1      # the window; Ubuntu and Debian
```

demuvo needs Python 3.12 or later. The window uses the system's GLFW and OpenGL
(`libglfw3` and `libgl1`) and no Python package. Running without a window needs neither.

## Command line

```text
demuvo [world.toml] [options]
```

| Option | Meaning |
| --- | --- |
| *(no file)* | open the sandbox with a blank world |
| `world.toml` | open the sandbox with that world |
| `--bench` | open the world fixed, as a benchmark: every run is the same |
| `--check` | check the file, print a summary, open nothing |
| `--result` | print how each task went when the window closes |
| `--dt S` | step length in seconds, replacing the file's |
| `--seconds S` | close the window after `S` seconds |
| `--one-process` | simulate in the window's process, not in a process of its own |
| `--seeds N` | open no window; run `N` seeds to the end and print a table |
| `--jobs J`, `--limit S` | with `--seeds`: runs at once, and simulated seconds before a run stops |
| `--install-desktop` | add demuvo and its icon to the desktop launcher, then exit |

`--result` prints, for each task, whether the robot reached it and when, and the distance
driven. For `examples/worlds/demo.toml` after the robot reaches the star:

```text
r1 task 1: reached at 19.26 s
r1 drove 8.86 m
```

## The sandbox

The sandbox is the window `demuvo` opens. It builds, runs, edits, saves and opens worlds.
The scene list and layer switches are on the left, the world with a playback pill is in
the middle, the inspector is on the right, and the log is at the bottom. Every panel
resizes by dragging its edge.

| Tool | Key | Action |
| --- | --- | --- |
| Select | V | click a robot or obstacle to select it; drag it to move it |
| Goal | G | click the floor to set the selected robot's goal; Shift and click adds a waypoint |
| Box, Circle | B, C | drag on the floor |
| Polygon, Line | P, L | click the corners; double click, right click or Enter finishes |
| Add robot, Delete | Ctrl+N, Delete | add a robot, delete the selected one |

The inspector shows a form for what is selected: for a robot, its pose, goals, body,
model, planner, smoother, controller, sensors and control period; for an obstacle, its pose
and size; for nothing, the world and its costmaps. A field applies as it is left. A name
in the scene list is renamed by a double click or F2. Ctrl+Z and Ctrl+Y undo and redo,
Space plays, S steps and R resets. Warnings from a run go to the log panel. The File menu
saves the world as TOML and opens one. [The sandbox](docs/sandbox.md) documents the whole
window and every key.

## World files

A world is a TOML file: size, seed, obstacles (box, circle, polygon, wall), robots, tasks
(goals and waypoints), sensors and costmaps. A file is checked when it loads, and a
mistake is named by object and key. [The world file](docs/world-file.md) describes every key.

## Python

`Run` steps a world with no window, and the functions in `demuvo.edit` build and change a
world. Each returns a new world.

```python
from demuvo import Run, edit

world = edit.new_world()  # 10 by 6 m, one robot, r1
world = edit.set_goal(world, "r1", (8.0, 1.0, 0.0))
run = Run(world)
run.until_done(60.0)
print("\n".join(run.result()))
```

[Python](docs/python.md) lists the edit functions and the `Run` methods.

## Running many seeds

`--seeds N` runs the world's seed and the `N - 1` after it, each to the end or to
`--limit` simulated seconds, with no window. `--jobs J` runs `J` at once; the rows are the
same for any `J`.

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

## Adding a planner, smoother or controller

A planner, a smoother and a controller are classes, named as `module:Class` in a robot of
the world file. demuvo builds the class and runs it beside the built-in ones.

```toml
[[robot]]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 1.0, 0.0]
planner = "breadth_first:BreadthFirst"
controller = "turn_then_drive:TurnThenDrive"
controller_params = { speed = 0.6 }
```

| Kind | Methods |
| --- | --- |
| Planner | `configure(costmap)`; `plan(start, goal)` returns an array of x, y, angle, or an empty array |
| Smoother | `configure(costmap)`; `smooth(path)` returns a path |
| Controller | `command(obs)` returns a `Twist` |

The module must be importable. `examples/plugins` holds a breadth-first planner and a
turn-then-drive controller, and `examples/worlds/plugins.toml` races them against A* and
pure pursuit:

```bash
PYTHONPATH=examples/plugins demuvo examples/worlds/plugins.toml
```

![Two robots race along two lanes. r1 uses a planner and a controller from
examples/plugins; r2 uses A* and pure pursuit.](docs/plugins.gif)

[Your own algorithms](docs/plugins.md) documents what each class receives and returns, and
how a class is listed in the window.

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
- **Planning and control:** `StraightLine` and `AStar` planners and a `LineOfSight` smoother
  on a global costmap, a `PurePursuit` controller, and a `DWB` controller, a local planner
  that steers round what the lidar sees in the local costmap.
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

| Page | Contents |
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
