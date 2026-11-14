# Use it from Python

`Run` steps a world with no window, so a script, a test or a benchmark needs nothing
else. The `edit` functions build or change a world, and `save_world` writes it to a file.
Each edit returns a new world and leaves the old one as it was. A bad edit is refused with
a message that names the robot and the setting.

```python
from demuvo import Run, edit
from demuvo.world import save_world

world = edit.new_world()  # 10 by 6 m, one robot, r1
world = edit.set_goal(world, "r1", (8.0, 1.0, 0.0))
world = edit.add_robot(world, "r2", edit.free_pose(world))
world = edit.set_goal(world, "r2", (1.0, 5.0, 0.0))
save_world(world, "mine.toml")  # a file `demuvo mine.toml` opens

run = Run(world)
run.until_done(60.0)  # step until every goal is reached
print("\n".join(run.result()))
```

```text
r1 task 1: reached at 9.16 s
r1 drove 3.58 m
r2 task 1: reached at 14.06 s
r2 drove 4.36 m
0 collisions
```

`run.step(n)` takes `n` steps. `run.advance(seconds, budget)` covers some simulated time
within a wall-clock budget. `run.poses`, `run.scans(name)`, `run.local_costmap(name)` and
`run.path(name)` read the state. A task is "reached at" the time the robot first met all
its conditions, and the distance is the exact length of the path, so the same seed gives
the same lines.

## The edits

| Function | Does |
| --- | --- |
| `new_world`, `validate` | a blank world; check that a world can run |
| `add_robot`, `remove_robot`, `rename_robot`, `free_pose` | robots |
| `set_pose`, `set_robot_shape`, `set_model`, `set_part`, `set_settings` | a robot's place, body, model and parts |
| `set_estimator`, `set_control_pose`, `set_control_period`, `set_sensor` | what the controller sees, and the sensors |
| `set_replan` | when a robot plans again while it drives |
| `set_goal`, `add_goal`, `move_goal`, `set_goal_options`, `remove_goal` | goals and waypoints |
| `add_obstacle`, `add_outline`, `move_obstacle`, `resize_obstacle`, `remove_obstacle` | obstacles |
| `rename_obstacle`, `set_outline_pose`, `set_line_length`, `obstacle_label` | names, and turning a polygon or line |
| `set_size`, `set_origin`, `set_seed`, `set_step`, `set_costmap` | the world and its costmaps |

## Seeds from the command line

To compare seeds, run a world with no window. `--seeds 4` runs the world's seed and the
three after it, each to the end or to `--limit` simulated seconds (120 by default).
`--jobs 4` runs four at once. The rows are the same for any number of jobs.

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

The demo world has no noise, so every seed gives the same row. A world with encoder or
lidar noise gives rows that differ.

`--result` prints how the run went when the window closes, in the same lines as
`run.result()`.

## The metrics file

`run.metrics()` gives the run as numbers instead of lines: one row per robot with the
world and seed, the time and steps, the tasks done and when, the distance driven, the
goal error left, the paths the robot was handed (replans included), its collisions, the
parts it ran (model, planner, smoother, controller, replan and estimator) and its pose
error at the end. The parts are there so rows of different configurations compare in
one table; a robot driven from outside demuvo has empty parts. `write_metrics` turns
the rows into a CSV file:

```python
from demuvo.evaluation import write_metrics

run = Run("mine.toml")
run.until_done(60.0)
print(write_metrics(run.metrics("mine")))  # demuvo_metrics_world_<time>.csv, here
```

`run.metrics("mine")` names the world in the rows; a run does not keep its file's name.

`write_metrics(rows, "runs/mine.csv")` names the file instead. The default place and
name are `METRICS_DIR` and `METRICS_NAME` in `demuvo.evaluation.metrics`, one line
each, so a benchmark that wants its files elsewhere changes them in one place. The
run-level columns repeat on every row, so rows from many runs or seeds concatenate
into one flat table. `--metrics [FILE]` writes the same file when the window closes.
The format is locked: [the metrics file](../README.md#the-metrics-file) in the README
names every column in order and says when a cell is left empty.

## What stays exact

The same world and seed give the same run on one machine and one numpy build. On another
machine the last few bits of a float may differ, so compare results with a small
tolerance, not for equality, when they cross machines.
