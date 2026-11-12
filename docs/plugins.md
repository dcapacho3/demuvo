# Your own algorithms

A planner, a smoother and a controller are each a class that you name in the world file.
demuvo builds it, gives it what it needs, and runs it next to the built-in ones, in the
same world and with the same seed. That makes a small, fair place to try an idea.

![Two robots race along two lanes. r1 uses the planner and controller in
examples/plugins; r2 uses A* and pure pursuit.](plugins.gif)

```bash
PYTHONPATH=examples/plugins demuvo examples/worlds/plugins.toml
```

`PYTHONPATH` puts the folder with your files where Python looks for modules; `--check`
needs it too, since checking a file loads the classes it names. The window
opens the race; press Space. With the numbers from this world, r1 reaches its goal at
18.86 s after 9.00 m and r2 at 16.01 s after 8.62 m, with no collision.

## The three kinds

You write a class and name it as `module:Class`:

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

A setting in `planner_params`, `smoother_params` or `controller_params` is passed to the
class as a keyword argument. A class with number or true/false arguments that have a
default also appears, with a field for each, in the inspector of the window.

**A planner** has two methods.

- `configure(costmap)` is called once, before the first plan, with the robot's global
  costmap (`None` if the world turns it off). The costmap is the same object afterwards,
  so keep it.
- `plan(start, goal)` returns the path as an array of shape (N, 3), rows of x, y, angle,
  or an array of shape (0, 3) if there is none. `start` and `goal` are poses with
  `.xyt` and `.translation`.

**A smoother** has `configure(costmap)` and `smooth(path)`, which returns a better path of
the same shape.

**A controller** has `command(obs)` and returns a `Twist`. It runs every `control_period`
seconds, and its last command stays in force between runs. It sees only the observation:
`obs.pose`, `obs.path`, `obs.goal`, `obs.goal_tolerance`, `obs.dt`, `obs.scans` (the
latest scan of each lidar), `obs.costmap` (the local costmap) and `obs.path_changes`
(a count that grows when the path is replaced, so a controller that keeps state knows
to start over). It does not see the simulation.

A planner that finds no way returns an empty path. The robot then stays still, and the log
says so.

## A planner in thirty lines

`examples/plugins/breadth_first.py` is breadth-first search on the costmap. It visits the
cells in rings round the start until it reaches the goal. It moves in four directions, so
its paths have square corners. Read it, then change one line.

```python
class BreadthFirst:
    def configure(self, costmap):
        self.costmap = costmap

    def plan(self, start, goal):
        ...  # breadth-first search over cells that are not avoided
        return with_headings(points, goal.xyt[2])
```

Things to try:

- Lower `AVOID` and the path keeps farther from the walls. Raise it toward 252 and it
  hugs them.
- Add `smoother = "demuvo.smoothers.line_of_sight:LineOfSight"` to r1 and watch the
  corners go.
- Change `STEPS` to eight directions. What changes in the path? In the time?
- Make it A*: keep a priority queue and order it by cost so far plus the distance left.

## A controller in thirty lines

`examples/plugins/turn_then_drive.py` turns on the spot until it faces a point of the
path a little way ahead, then drives and steers a little. It is simpler than `PurePursuit`
and loses time turning. Change `speed`, `ahead`, `turn_rate` and `face_within` in
`controller_params` and watch it.

## Keep a run exact

A run repeats only if your code does. Do not use the clock, and do not use `random`
without a seed. If your algorithm needs chance, take a `seed` in `planner_params` or
`controller_params` and make a generator from it.

## Show a class in the window's lists

The planner, smoother and controller lists of the inspector hold the built-in classes and
any that a package names in its entry points:

```toml
[project.entry-points."demuvo.planners"]
breadth_first = "mypackage.planners:BreadthFirst"
```

The groups are `demuvo.planners`, `demuvo.smoothers` and `demuvo.controllers`. Only the
`module:Class` text is read at start; the class is loaded when a robot uses it.

## From Python

`Run` takes a world file or a `World` and needs the same `PYTHONPATH`. `Navigation` also
takes a goal or a path from outside with `set_goal` and `set_path`, for a program that
plans somewhere else; the planner and smoother then stay out of it.
