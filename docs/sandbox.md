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

A blank world is 10 by 6 metres with one robot, `r1`, in the middle. The world needs at
least one robot, so the last one cannot be deleted.

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
| Left | the scene list (robots and obstacles) and the layer switches |
| Middle | the world, with the playback pill (play, step, reset, speed, the clock) |
| Right | the inspector: the form for what is selected |
| Bottom | the log, which opens when a run gives a warning |
| Top | the menus and the tool bar |

Drag the edge of the left, right or bottom panel to resize it. Ctrl+1, Ctrl+2 and
Ctrl+3 show and hide the scene list, the inspector and the log.

## Building a world

Pick a tool from the tool bar or by key, then use the pointer.

| Tool | Key | What it does |
| --- | --- | --- |
| Select | V | Click a robot or an obstacle to select it. Drag it to move it. |
| Goal | G | Click the floor to set the selected robot's goal. Shift and click adds a waypoint after the goals it has. It is grey until a robot is selected. |
| Box | B | Drag from one corner to the other. |
| Circle | C | Drag from the centre outwards. |
| Polygon | P | Click the corners. Click near the first corner, double click, right click or press Enter to close it. |
| Line | L | Click the points. Double click, right click or press Enter to finish. |
| Add robot | Ctrl+N | Puts a robot at a free place. |
| Delete | Delete | Removes the selected robot or obstacle. |

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

**An obstacle** shows its pose and size. A polygon or line shows its middle and its
rotation instead, and turns about that middle. A line of two points also has a length.
The corners of a polygon or line are set in the world file.

**A robot** shows:

- its pose and its goals, each with a tolerance, a stop rule and a wait; Add waypoint adds
  one more;
- its body, a disc or a box, and its size;
- the model, planner, smoother and controller, each with its settings, and the control
  period;
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
- The title bar shows the file and a star if there are unsaved changes. New, Open and
  Quit say so the first time and go ahead the second.

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

The layers are the global and local costmap, obstacles, trails, paths, goals, lidar rays
and robots, with an opacity and a palette for the costmaps. The floor is always shown. A
world with up to ten robots starts with the lidar rays on. The rays appear after the
first step.

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
| V, G, B, C, P, L | the select, goal, box, circle, polygon and line tools |
| Enter, Esc | finish or cancel a polygon or line |
| F2 | rename the selected robot or obstacle |
| Ctrl+N, Delete | add a robot, delete the selected one |
| Ctrl+O, Ctrl+S | open a world file, save the world |
| Ctrl+Z, Ctrl+Y | undo, redo |
| Ctrl+1, Ctrl+2, Ctrl+3 | the scene list, the inspector, the log |
| Ctrl+0 | fit the panels to the window |
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
