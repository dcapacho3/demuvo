<p align="center">
  <img src="https://raw.githubusercontent.com/dcapacho3/demuvo/main/docs/logo.svg" alt="demuvo" width="560">
</p>

<p align="center">
  <a href="https://github.com/dcapacho3/demuvo/actions/workflows/ci_pytest.yml"><img src="https://github.com/dcapacho3/demuvo/actions/workflows/ci_pytest.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license">
</p>

**A small, deterministic 2D world for robots.**

Demuvo is a top-down world you describe in a file: its size, its obstacles and its
robots. Robots move with known kinematic models, sense with simulated encoders and
lidar, and are driven by controllers, planners or filters that anyone can plug in. The
same world and the same seed always give the same run, so two algorithms can be
compared fairly.

## Status

Early development. Nothing is released yet and the API will change.

Working today:

- a world file (TOML) with a world size, obstacles (box, circle, polygon, wall) and
  robots, checked when it loads;
- one kinematic model, differential drive, driven by a ROS-style `Twist` command;
- a planner and a controller for each robot, chosen in the file as `module:Class`;
  built in: `StraightLine` and `PurePursuit`;
- a step length and a controller rate that you set apart (see "Time" below);
- a top-down view, and GIF export from Python.

Not yet: collisions, noisy wheel encoders, a lidar, the other kinematic models
(car-like, forklift, mecanum, skid-steer), and a runner that compares controllers on
the same worlds and seeds.

## Try it

```bash
pip install -e ".[view]"
demuvo examples/worlds/demo.toml     # a robot drives to the star; --no-trail hides its path
demuvo examples/worlds/demo.toml --check   # check the file and open nothing
```

A robot that has a task gets a path from its planner, and its controller follows it.
Several tasks for one robot, in file order, are waypoints:

```toml
[[robot]]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 1.0, 0.0]                         # x, y, angle
controller = "demuvo.controllers:PurePursuit"
controller_params = { speed = 0.5, lookahead = 0.6, rotate_to_goal = true }

[[task]]
robot = "r1"
goal = [9.0, 5.0, 0.0]
goal_tolerance = [0.1, 0.1]                    # metres, radians
wait = 2.0                                     # optional: seconds to hold the goal
```

A task is done when the robot is within the tolerance and, if the task asks, standing
still (below 0.01 m/s and 0.01 rad/s) for `wait` seconds. `stop = true` or `false` sets
the standing-still test; the default is yes for a robot's last task and for a task with
a `wait`, and no for the waypoints in between.

To use your own, write a class with one method and name it in the file. A planner has
`plan(world, start, goal)` and returns an array of shape (N, 3) of x, y, angle. A
controller has `command(obs)` and returns a `Twist`; it sees only the observation (its
pose, the path, the goal), not the simulation. From Python, `Navigation` also takes a
goal or a path from outside with `set_goal` and `set_path`.

## Time

The simulation moves in steps of `dt` seconds, set in `[world]` (default 0.01). Each
robot's controller runs on its own clock, `control_period` seconds apart (default 0.05,
the 20 Hz of the Nav2 controller), and its last command stays in force between runs:

```toml
[world]
size = [10.0, 6.0]
seed = 7
dt = 0.01                  # optional: seconds per step

[[robot]]
name = "r1"
model = "diff_drive"
shape = { name = "circle", radius = 0.2 }
pose = [1.0, 1.0, 0.0]
controller = "demuvo.controllers:PurePursuit"
control_period = 0.05      # optional: seconds between controller runs
```

`demuvo file.toml --dt 0.02` replaces the file's `dt`. A period shorter than `dt` is an
error, since a controller cannot run twice in one step. A period that is not a whole
number of steps works, with one warning: the gaps then alternate between two lengths,
one step apart, and their mean is the period. A new goal or path takes effect at the
controller's next run, up to one period later, as in Nav2. The controller's
`Observation.dt` is the time since it last ran.

## Develop

```bash
git clone https://github.com/dcapacho3/demuvo.git
cd demuvo
pip install -e . --group dev     # needs pip 25.1 or later
pre-commit install
python -m pytest
```

Pose maths comes from [robomath](https://github.com/dcapacho3/robomath).

## License

Apache 2.0. Copyright 2026 David Capacho. See [LICENSE](LICENSE).
