# The sandbox

The sandbox is the window `demuvo` opens. You build a world with the pointer and the
forms, run it, change it, and save it as a file.

![A session in the sandbox: a box, a circle and a line are drawn, one is renamed, the
robot gets a goal, an obstacle is dragged, and the run plays.](sandbox.gif)

```bash
sudo apt install libglfw3 libgl1      # the window; Ubuntu and Debian
demuvo                                # a blank world to build in
demuvo examples/worlds/demo.toml      # open a world file
demuvo examples/worlds/demo.toml --bench   # the same world, fixed
```

A blank world is 10 by 6 metres and empty, like an empty Gazebo world: no obstacles,
actors or robots. Add a robot with the robot tool, then give it a goal. Any robot can be
deleted, the last one too.

## Two modes

- **Sandbox** (the default): the world is yours. You change it only while no run is in
  progress. Reset (R) ends the run and unlocks editing.
- **Benchmark** (`--bench`): the world is fixed, so every run is the same and Reset
  repeats it. Layers and panels still work. New, Open and the edit tools are off, and
  Save as still writes the world.

A change that cannot work is refused in the status bar, in words that name the robot and
the setting, and the world stays as it was.

## The window

| Where | What |
| --- | --- |
| Left | the scene list (robots, actors and obstacles) and the layer switches |
| Middle | the world, with the playback pill (play, step, reset, speed, the clock) |
| Right | the inspector: the form for what is selected |
| Bottom | the log, which opens when a run gives a warning |
| Top | the menus and the tool bar |

Drag the edge of the left, right or bottom panel to resize it. Ctrl+1, Ctrl+2 and
Ctrl+3 show and hide the scene list, the inspector and the log.

## Zoom and move

A big world, such as a warehouse, needs a closer look.

| Do | How |
| --- | --- |
| Zoom in or out | Turn the wheel over the world. The place under the pointer stays where it is. The `+` and `-` keys, Ctrl with them, and View > Zoom in and Zoom out zoom about the middle. |
| Move the view | Press the middle button and drag, or pick the Hand tool (H) and drag with the left button, which suits a trackpad. |
| See the whole world again | Press `0` or Home, or View > Fit the world. |
| Zoom to what is selected | Press `F`, or View > Zoom to selection, with a robot, an actor or an obstacle selected. |

The scale bar at the bottom left of the world gives a round number of metres and, once
you have zoomed, the zoom as a percentage of the whole-world view. You can zoom out to
40 percent and in to 400 pixels a metre, where a cell of the default costmap is 20
pixels across, and the world cannot be moved out of sight. The zoom stays while you
edit and while the window is resized; a world of another size starts whole.

## Building a world

Pick a tool from the tool bar or by key, then use the pointer.

| Tool | Key | What it does |
| --- | --- | --- |
| Select | V | Click a robot, an actor or an obstacle to select it. Drag it to move it. |
| Hand | H | Drag to move the view; see [Zoom and move](#zoom-and-move). It works while a run is in progress. |
| Goal | G | Press on the floor to put the selected robot's goal there, drag to turn its arrow and let go to set it, as RViz's 2D Goal Pose does; a click alone sets the heading to 0. Hold Shift to add a waypoint after the goals it has instead, which is how strategy games queue moves; the robot form's Add waypoint does the same. It is grey until a robot is selected. |
| Box | B | Drag from one corner to the other. |
| Circle | C | Drag from the centre outwards. |
| Polygon | P | Click the corners. Click near the first corner, double click, right click or press Enter to close it. |
| Line | L | Click the points. Double click, right click or press Enter to finish. |
| Add robot | Ctrl+N | Puts a robot at a free place. |
| Add actor | Ctrl+M | Puts an actor at a free place. |
| Delete | Delete | Removes the selected robot, actor or obstacle. |

After you draw a shape, the pointer goes back to Select, so the next drag moves it. Esc
cancels a polygon or line that is half drawn.

While you drag a robot, the world follows it a few times a second, so its planned path
moves with it. A dragged obstacle shows an outline and the world changes when you let go.

### Names

Double click a row of the scene list, or select something and press F2, to rename it in
place. Enter keeps the name and Esc drops it. A robot's goals follow its new name. An
obstacle with no name is listed by its shape, such as "box 1". Two robots, or two
obstacles, cannot share a name.

## The inspector

Every field applies as you leave it. A number field selects its number when you click, so
typing replaces it; Left, Right, Home, End, Delete and Ctrl+A work as in a text box, and
Up, Down and the wheel change it by a step.

**Nothing selected** shows the world: its width, height, seed and step length, the
corner, the sensors every robot starts with, and the global and local costmap.

**An actor** shows where it starts, its body, whether it is `solid` and whether it can
`collide` with other actors, and its motion with that motion's settings; see
[Actors](#actors).

**An obstacle** shows its pose and size. A polygon or line shows its middle and its
rotation instead, and turns about that middle. A line of two points also has a length.
The corners of a polygon or line are set in the world file.

**A robot** shows:

- its pose and its goals, each with a tolerance, a stop rule and a wait; Add waypoint adds
  one more;
- its body, a disc or a box, and its size;
- the model, planner, smoother and controller, each with its settings, and the control
  period. A setting that does nothing for this robot is left out: pure pursuit's
  reversing switch for a robot that turns in place, DWB's sideways speeds for one that cannot move
  sideways, and the wheel sizes until something reads the encoders;
- whether it plans again while it drives, with the period between plans and how far off
  the path it may stray; the other [replanning](world-file.md#replanning) settings are
  set in the world file;
- where the controller gets its pose (the true pose or an estimate) and the sensors:
  encoders, and any number of lidars.

Before the first step, a robot with a goal already shows the path its planner, and its
smoother, will make. The preview does not change the run.

## Files

The File menu has New world, Open, Save and Save as.

- Open and Save as show a file browser drawn by demuvo. Click a folder twice to go in,
  click a file twice to take it, use Up and Home, or type a name or a whole path in the
  Name box. Save as adds `.toml` and asks before it replaces a file.
- Ctrl+S saves to the file the world came from, and asks for one if there is none.
- The title bar shows the file and a star if there are unsaved changes. New world,
  Quit and the window's close button ask first, in a card with Cancel and Discard.
  Cancel is what Enter and Esc answer. Open always shows its browser, and its title
  says when the unsaved changes will be lost.

Open also takes a Gazebo `.world` or `.sdf` file, with no robot, as in an empty Gazebo
world. What stands between 0.1 and 1 m above the floor becomes obstacles: boxes,
cylinders and spheres as they are, and meshes (`.dae`, `.stl`, `.obj`) cut at those
heights, so a roof or a ceiling adds nothing. Each region of a mesh becomes one box
(turned, if it is) or one polygon, not a staircase of boxes. A closed mesh is a solid
and a room is open. Models the file includes are read from the folders that Gazebo
searches, `GZ_SIM_RESOURCE_PATH` among them, and from the Fuel cache; nothing is
downloaded. A world with no collision shapes, such as one that only includes a model
with visual meshes, is read from the visuals, and the log says so. The log also names
anything that could not be read: a mesh or an include that is missing, a shape demuvo
does not know, the shapes the heights left out. The world has no file of its own after
that, so Save asks for a name and never writes over the Gazebo file.

Saving writes a new file. Comments and the order of keys in the old file are not kept.
The [world file](world-file.md) has the format.

## Running

| Key | Does |
| --- | --- |
| Space | play or pause |
| S | step by one control period |
| R | reset |

The speed goes from 0.25x to 8x. While a run plays, the status bar shows "real time
100%": the simulated seconds gained for each real second, as a share of the speed you
chose. It turns amber below 90%, which means the run is falling behind.

The layers are the global and local costmap, obstacles, actors, actor routes, trails, paths, goals,
lidar rays and robots, with an opacity and a palette for the costmaps. The floor is always shown. A
world with up to ten robots starts with the lidar rays on. The rays appear after the
first step.

## Actors

An actor is a moving obstacle that the global costmap is not built from, so a planner
never sees one: a robot finds an actor with its lidar, puts it in its local costmap and,
with `replan`, plans around it. Add one with the tool bar button, Ctrl+M or
Edit > Add actor. It arrives at a free place, standing still, and it is drawn in amber
with a tick that shows which way it faces.

![The actors world in the sandbox: the scene list has an Actors group, the walker is
selected, and the inspector shows its pose, body, the solid and collide switches and
its Waypoints motion with mode, speed, wait, delay and face. The cart, which is not
solid, is drawn hollow.](actors.png)

Click an actor to select it, drag it to move it, F2 to rename it, Delete to remove it.
The inspector then shows:

| Field | What it does |
| --- | --- |
| x, y, heading | where it starts |
| radius, or length and width | its body, a disc or a box |
| solid | on, a robot is pushed out of it; off, the robot drives through and the collision is still counted |
| collide | on, it meets other actors that also have it on; the contact is recorded and neither is moved |
| motion | how it moves: "stands still", or a class such as Waypoints |

Picking a motion shows its settings under it. **Waypoints** goes from where the actor
starts through its points at `speed`, facing the way it goes: `mode` is `once` (stay at
the last point), `loop` (back to the start and round again) or `bounce` (turn round and
come back), with `wait` at every point it reaches, `delay` before it first moves, and
`face` off to slide without turning. A new motion starts with one point.

The route of an actor with waypoints is drawn as a dashed amber line with a numbered
dot at each point and a ring at the start; the selected actor's route is the strong
one. Edit it in the window:

- **Waypoints fold** - "Show waypoints (N)" in the inspector opens an x and y row for
  each point with its own wait and a switch to turn to a heading there, a Remove
  button and an Add waypoint button.
- **Drag a dot** to move that point.
- **Drag the middle of a leg** to put a new point there.
- **Shift and click the floor** to add a point at the end.

See
[the world file](world-file.md#actors) for every key, and
[your own algorithms](plugins.md) to write a motion of your own.

An actor that is not solid is drawn hollow. Actors and their routes each have a layer
switch, so the routes can be hidden.

## The log

Warnings and log lines of a run appear in the log panel, not on the terminal: a planner
that finds no way, or a control period that is not a whole number of steps. The panel
opens by itself the first time a message comes. The tool bar button, View > Log and
Ctrl+3 show and hide it, and Clear empties it. The same message twice in a row is one
line with a count.

## Undo

Ctrl+Z takes back the last edit and Ctrl+Y makes it again. A drag is one step. Opening or
starting a world forgets the list. Undo is off while a run is in progress.

## All the keys

| Key | Does |
| --- | --- |
| Space, S, R | play or pause, step, reset |
| V, H, G, B, C, P, L | the select, hand, goal, box, circle, polygon and line tools |
| +, -, 0, F | zoom in, zoom out, show the whole world, zoom to the selection |
| Enter, Esc | finish or cancel a polygon or line |
| F2 | rename the selected robot, actor or obstacle |
| Ctrl+N, Ctrl+M, Delete | add a robot, add an actor, delete the selected one |
| Ctrl+O, Ctrl+S | open a world file, save the world |
| Ctrl+Z, Ctrl+Y | undo, redo |
| Ctrl+1, Ctrl+2, Ctrl+3 | the scene list, the inspector, the log |
| Ctrl+0 | put the panels back to their first sizes |
| Tab, Shift+Tab | move the keyboard focus |
| Ctrl+Q | quit |

## The launcher and the icon

```bash
demuvo --install-desktop
```

adds demuvo and its icon to your desktop's launcher, using files under `~/.local/share`
and nothing else. A package installed for the whole system will do the same.

## Recording a GIF

A GIF or a folder of PNG frames of any world is made with no window:

```bash
python -m demuvo.gui.record examples/worlds/lidar.toml run.gif --seconds 12 --fps 8 \
    --layers rays,local          # a path ending in / writes one PNG for each frame
```

A GIF stores only the part of each frame that changed, so a still scene with a few moving
robots makes a small file.
