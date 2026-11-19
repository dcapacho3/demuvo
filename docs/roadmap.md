# Roadmap

What is built, what the first release still needs, and the longer list. Move a line to
**Built** when it ships, and add new ideas at the end under **Ideas to sort**.

Sizes: **S** is one small commit, **M** a few commits, **L** a series of parts. Every
item keeps demuvo's rules: simulation only, an ideal world with no physics and no slip
(sensor noise is fine), kinematic models, and a dependency
only if its licence allows it. Planners written from papers.

## Built

- **Window:** build a world, run it, undo and redo, rename, save and open, drag a goal
  to aim it, Shift for waypoints, a card before unsaved work is lost, inspector groups
  with help text, layers, an empty default world.
- **World:** box, circle, polygon and wall obstacles, maps from `map.yaml`, Gazebo
  `.world` and `.sdf` import (boxes, cylinders, spheres, includes and meshes cut at a
  robot's height), moving actors on waypoints.
- **Models:** differential drive, car (Ackermann), mecanum.
- **Sensing:** lidar with mount, noise and rate; encoders; a local costmap built from
  scans; ground-truth and dead-reckoning estimators.
- **Planners:** straight line, A*, Dubins, Reeds-Shepp, Hybrid A*. Smoother: line of
  sight.
- **Controllers:** pure pursuit (with reversing), DWB, omni pursuit.
- **Replanning** on a blocked path, a path left, a stall or a timer.
- **Running:** `Run` from Python, edit functions, `set_goal` and `set_path` from outside,
  seeded runs with `--seeds` and `--jobs`, a metrics table and CSV file, a GIF recorder.
- **Your own algorithms:** plug-in planners, smoothers and controllers, with examples.
- **Checks:** 100% test coverage, ruff and pylint clean, a rule on which part imports which.

## What the first release still needs

The base: a person installs demuvo, loads or builds a world, runs a planner and a
controller of their own beside the built-in ones, and gets numbers that repeat, a replay
and a table over seeds, with docs that stand alone.

| # | Item | Size |
|---|------|------|
| 1 | **Trace recorder and replay.** Save the poses, commands and paths of a run in a versioned file; play it back in the window with a scrubber and step back. | M |
| 2 | **Determinism check.** A `--verify-determinism` command that runs a world twice and compares every step; run it in CI. | S |
| 3 | **More metrics.** Path length against the shortest, clearance, planner expansions, stuck count, success weighted by path length. Expansions let planners compare on work done, not Python time. | S-M |
| 4 | **Benchmark runner.** Every pairing of planner, controller and model over worlds and seeds from one TOML file, in one table and one CSV. `--seeds` does one world today. | M |
| 5 | **Map and instance generator.** Seeded random maps (clutter, caves, rooms) with a check that start and goal connect, and random start and goal pairs. | M |
| 6 | **The two missing models:** skid-steer (an effective track width, no slip) and a rear-steer forklift. | S, M |
| 7 | **Enough built-in algorithms to compare:** Dijkstra and Theta* planners; Stanley and regulated pure pursuit controllers. | S each |
| 8 | **Cars in tight rooms.** `DWB` stalls there, and `HybridAStar` without reversing often finds no path. A parking controller that reaches the goal angle exactly. | M |
| 9 | **Release.** A PyPI release with a changelog and release notes, a plug-in template (`demuvo new planner`), and an example gallery with a GIF per world. | M |
| 10 | **Pylint in CI**, now that the code is clean. | S |

## Next: the teaching layer

| Item | Size |
|------|------|
| Planner debug drawing: open and closed sets, trees and cost fields in the window. | M |
| Step through a planner one expansion at a time. | M |
| Two planners side by side in one window, on one world. | M |
| Exercises that ask for a prediction first, then run (a lesson pack, in English and Spanish). | M |
| Autograder: run a student's planner on hidden worlds with a time limit. | M-L |
| A plug-in loaded by entry point, so `pip install my-planner` appears in the window. | M |
| Notebook support: draw a run in Jupyter. | M |
| A first-run tour and a keyboard help screen. | S |
| Scenario browser with thumbnails; a folder of named scenarios shipped with demuvo. | S |
| Live plots of speed, error and clearance beside the scene. | M |
| An event log with filters (collisions, goal reached, replans). | S |
| Estimate against truth: a display and an error plot. | S |
| "Why this path": the cost along the path, split by term. | S |

## Robot models (kinematic only)

| Item | Size |
|------|------|
| Swerve drive. | M |
| Tractor and one trailer, then several. | M |
| Articulated vehicle with hinge steering. | M |
| Dubins and Reeds-Shepp cars as models, not only planners. | S |
| Acceleration and jerk limits on any model. | S |
| Command delay and update rate per robot. | S |
| Footprints that are not circles: rectangle, polygon, a forklift with forks. | M |
| An attached load that adds to the footprint while carried. | M |
| Point-mass 2D drone. | S |
| A model checker that says whether a path or a command sequence is feasible. | S |
| A model written in a TOML table, with no Python. | M |
| A wheel-speed limit for mecanum, so x, y and turning share one budget. | S |
| A battery counter and a charging task. | S |

## Global planners (written from the papers)

| Item | Size |
|------|------|
| Weighted A* and anytime A*. | S |
| Jump point search. | M |
| D* Lite, for a map that changes. | M |
| State lattice with generated motion primitives. | L |
| RRT, RRT-Connect, RRT* and Informed RRT*. | M each |
| PRM and a lazy PRM that keeps its roadmap between queries. | M |
| FMT* and BIT*. | M |
| Potential field and navigation function. | S |
| Visibility graph and Voronoi roadmap. | M |
| Wavefront and fast marching distance fields. | S |
| Coverage planners: back-and-forth and spiral. | M |
| Frontier exploration. | M |
| A route graph for warehouses with one-way aisles. | M |

## Local planners and controllers

| Item | Size |
|------|------|
| Pure pursuit with an adaptive lookahead. | S |
| Vector pursuit. | S |
| Graceful controller (a smooth control law). | M |
| LQR path tracking. | M |
| Kinematic MPC. | L |
| MPPI with a list of cost terms, per model. | L |
| Elastic band (TEB style). | L |
| Velocity obstacles: RVO and ORCA, for crowds and several robots. | M |
| Social force model for people actors. | M |
| Gap following and VFH, from lidar alone. | M |
| Rotate-first shim as a wrapper around any controller. | S |
| Velocity smoother and limiter as a wrapper. | S |
| Collision monitor: slow or stop in a zone before a hit. | M |
| Recovery behaviours: wait, back up, spin. | M |
| Goal, progress and stuck checkers you can choose. | S |
| Docking controller, to line up on a pallet or a charger. | M |

## Smoothers and path tools

| Item | Size |
|------|------|
| Gradient-descent smoother with a clearance term. | M |
| B-spline and Bezier smoothers. | S |
| Clothoid smoother with continuous curvature. | M |
| Moving-average and Savitzky-Golay smoothers. | S |
| Time parametrisation of a path under speed and acceleration limits. | M |
| Path simplifier, resampler and a curvature profile. | S |

## Sensing, estimation and mapping

| Item | Size |
|------|------|
| Extended Kalman filter, written from scratch. | M |
| Unscented Kalman filter. | M |
| Particle filter with map matching. | L |
| Landmark sensor with range, bearing, noise and false matches. | M |
| Wheel odometry drift per model. | S |
| IMU, GPS-like and compass sensors with bias drift. | S |
| Sensor faults: dropout, stuck value, spike, delay. | S |
| A camera-like field-of-view sensor in 2D, with occlusion. | M |
| Sonar and bump sensors. | S |
| Occupancy grid mapping from lidar. | M |
| ICP scan matching. | M |
| SLAM: EKF first, then particle and pose-graph. | L |
| Localisation drift as a world option, so a robot must find itself again. | S |

## Worlds and maps

| Item | Size |
|------|------|
| Cave, maze (with loops), rooms-and-corridors generators. | S each |
| Warehouse, supermarket, parking lot and street-grid generators. | M each |
| A difficulty score for a world (clutter, corridor width, path winding) as its own tool. | S |
| Occupancy image import (PNG and YAML). | S |
| Gazebo import of more shapes (heightmaps, capsules, ellipsoids) and of a model that sits in the Fuel cloud only; export of a world to `.sdf`. | M |
| Import from SVG or DXF drawings. | M |
| Terrain layer: a speed limit, cost or label from a seeded map; off by default. | M |
| A world that changes in time: doors, blocked aisles, schedules. | M |
| Keep-out and speed zones as map layers. | S |
| Random walkers and crowds as actors, beside the waypoint walkers. | M |
| Pallets and boxes that a robot moves by a simple rule, with no physics. | M |
| A JSON Schema for the world file, with versions and a migration tool. | M |
| In the window: reshape the corners of a polygon or line, and more body shapes. | M |

## Many robots

| Item | Size |
|------|------|
| Swarm runner: one rule, many robots, each with its own sensors and neighbours. | L |
| Flocking, consensus, rendezvous and formation control. | M each |
| Priority planning, then conflict-based search. | M, L |
| Reservation tables for aisles and crossings. | M |
| Deadlock and livelock detection in the metrics. | S |
| Task allocation: Hungarian, auction, CBBA. | M |
| Shopping carts that share a list and split the aisles. | L |
| A communication model: range, delay, loss. | M |
| Leader-follower and platoons. | M |
| Fleet metrics: throughput, waiting time, fairness. | S |
| Multi-robot coverage and exploration. | L |

## Missions and behaviours

| Item | Size |
|------|------|
| Task queues: patrol, pick and deliver, arrivals over time. | M |
| A behaviour-tree engine of your own, with recovery nodes. | L |
| A state machine per robot, drawn in the window. | M |
| Mission files in TOML that script a sequence of tasks. | S |
| Replanning rules: when to replan and when to keep the path. | M |
| A path-switching metric: how often, and how much, a replan changes the route. | S |

## Evaluation and benchmarks

| Item | Size |
|------|------|
| Parallel runs that give equal results (a seed per job). | M |
| Parquet export beside CSV. | S |
| Statistics: bootstrap intervals, Wilcoxon, Friedman with Nemenyi, a critical-difference plot. | M |
| Performance profiles and Pareto fronts of planner pairs. | M |
| LaTeX table and figure export. | S |
| A table of golden numbers in CI that must not drift. | S |
| A cross-machine mode with its own trigonometry and no fused multiply-add. | M |
| An HTML report with every plot and a leaderboard. | M |
| Sweeps over noise, delay and drift against the success rate. | M |
| A feasibility audit of each planner's output for each model. | S |
| Near-miss and time-to-collision metrics. | S |
| Rank agreement between demuvo and Gazebo or Nav2 results. | M |

## Recording and visuals

| Item | Size |
|------|------|
| Trace diff: the first step where two runs differ. | S |
| Branch a run from a step ("what if the goal moves here"). | M |
| Headless render to PNG frames, GIF and MP4. | M |
| Cost value under the cursor. | S |
| Heat maps of visited cells and collisions over many runs. | S |
| An overlay of the paths from different seeds. | S |
| Camera follow, minimap, a measuring tool, snapshots. | S each |

## Window and editor

| Item | Size |
|------|------|
| Grid snapping, copy and paste, align, group. | M |
| A parameter sweep from the window, with a results table. | M |
| Dark and light themes and scalable text. | S |
| Spanish and English interface. | M |
| Rows inside a folded inspector group indented, so a group in a group reads as one. | S |

## Links to other software

| Item | Size |
|------|------|
| ROS 2 bridge: `cmd_vel`, `odom`, `scan`, TF and the map. | L |
| Run a Nav2 planner or controller in a demuvo world through the bridge. | L |
| A stable Python API with a version policy. | S |
| Server mode with a websocket and a browser viewer. | L |
| A Docker image for headless benchmarks. | S |
| A GitHub Action that runs a benchmark file on every push. | S |
| Command line subcommands: `run`, `bench`, `convert`, `list`, `verify`. | M |
| Debian package and conda-forge. | M |

## Engineering

| Item | Size |
|------|------|
| A faster core in C++ or Rust under the same Python interface: the search loops of A*, Hybrid A* and the DWB wavefront first, then the collision check. | L |
| Jump Point Search for A*, which skips the cells of an open grid. A plan across the Depot takes 0.4 s today, and every replan pays it. | M |
| Hybrid A*: vectorise the arcs of each pose, and stop a search that has no path sooner than 20,000 poses (3 s on the Depot). | M |
| DWB: find the poses of each candidate trajectory by the closed-form arc, x, y and angle, and not through 3 by 3 matrix products. The products take 18% of a DWB run. | S |
| Build the simulation once for an edit; today one build checks the edit and another runs it. | S |
| Keep the scene of fixed obstacles between edits, as the global costmap now is. | S |
| Open a big Gazebo file faster: the mesh cut takes 1.3 s for the Depot, and could be kept beside the file. | S |
| Lidar for many robots: batch the ray casting across robots. | M |
| Property tests: a planner's path is valid, and Dijkstra never costs more than A*. | M |
| Fuzz the world loader. | S |
| Mutation tests on the planners. | S |
| Type checking in CI, a docs site and an API reference. | M |
| A `--profile` flag and counters. | S |
