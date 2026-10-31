"""A top-down view of a simulation, drawn with pygame.

This module is not imported by `demuvo`, so the core runs without pygame.
Install the `view` extra of the package to use it.

The window and the GIF share one drawing routine: a frame is drawn at twice
the size and shrunk, which smooths the edges.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Callable, Mapping

import numpy as np
import pygame
from PIL import Image
from pygame.locals import (
    K_DOWN,
    K_ESCAPE,
    K_RIGHT,
    K_SPACE,
    K_UP,
    KEYDOWN,
    QUIT,
    SRCALPHA,
)
from robomath import Transform2D

from ..models import Twist
from ..sim.simulation import Simulation
from ..world.shapes import Circle, LineString, Shape

Commands = Callable[[Simulation], Mapping[str, Twist]]

SS = 2
TRAIL = 4000  # past positions kept for each robot's trail
HUD = 34
MARGIN = 0.35
TILE = 1.0
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ROBOT_COLORS = ("#1e88e5", "#e53935", "#43a047", "#fb8c00", "#8e24aa", "#00acc1")
OUTSIDE = "#90a4ae"
FLOOR = ("#eceff1", "#dde2e6")
WALL = "#263238"
OBSTACLE = "#607d8b"
BAR = "#1c262b"
TEXT = "#eceff1"
SHADOW = (0, 0, 0, 55)


def outline(shape: Shape, pose: Transform2D) -> np.ndarray:
    """The points to draw for a shape at a pose, as an array of shape (n, 2)."""
    if isinstance(shape, Circle):
        x, y = pose.translation
        angles = np.linspace(0.0, 2 * math.pi, 65)[:-1]
        return np.column_stack(
            [x + shape.radius * np.cos(angles), y + shape.radius * np.sin(angles)]
        )
    return np.array(shape.points(pose))


def _darker(color: str) -> pygame.Color:
    """The same colour, darker, for outlines."""
    c = pygame.Color(color)
    return pygame.Color(int(c.r * 0.6), int(c.g * 0.6), int(c.b * 0.6))


def _reach(shape: Shape) -> float:
    """Distance from the pose to the farthest point of the shape."""
    return max(math.hypot(*p) for p in outline(shape, Transform2D.identity()))


class View:
    """Draws a simulation as an image, frame after frame.

    Call `step` to advance the simulation and record the trails, then `render`
    for the picture.

    Attributes:
        sim: the simulation that is drawn.
        size: size in pixels of a rendered frame.
        reach: for each robot name, the distance from its pose to the farthest
            point of its body.
        paused: shown in the top bar.
        speed: shown in the top bar, as a multiple of real time.
        done: shown in the top bar when the run has ended.
    """

    def __init__(
        self, sim: Simulation, width: int = 960, height: int = 640, trail: int = TRAIL
    ) -> None:
        """Work out the scale and draw the parts that do not move.

        Args:
            sim: the simulation to draw.
            width: largest width of the world area in pixels.
            height: largest height of the world area in pixels.
            trail: number of past positions kept for each robot's trail; 0
                draws no trail.
        """
        pygame.font.init()
        self.sim = sim
        w, h = sim.world.size
        self.scale = min(width / (w + 2 * MARGIN), height / (h + 2 * MARGIN))
        self._scene = (
            round((w + 2 * MARGIN) * self.scale),
            round((h + 2 * MARGIN) * self.scale),
        )
        self.size = (self._scene[0], self._scene[1] + HUD)
        self.trails = {
            name: deque([tuple(pose.translation)], maxlen=trail)
            for name, pose in sim.poses.items()
        }
        self.reach = {r.name: _reach(r.shape) for r in sim.world.robots}
        self.goals = {
            r.name: [t.goal.translation for t in sim.world.tasks if t.robot == r.name]
            for r in sim.world.robots
        }
        self.paused = False
        self.speed = 1.0
        self.done = False
        self._font = pygame.font.Font(None, 24)
        self._small = pygame.font.Font(None, 20)
        self._background = self._draw_background()

    def to_pixel(self, x: float, y: float, k: int = 1) -> tuple[float, float]:
        """Metres to scene pixels at k times the size. The y axis points up."""
        h = self.sim.world.size[1]
        return ((x + MARGIN) * self.scale * k, (h + MARGIN - y) * self.scale * k)

    def _points(self, points, k: int = SS) -> list[tuple[float, float]]:
        """Many points from metres to pixels."""
        return [self.to_pixel(x, y, k) for x, y in points]

    def _layer(self, alpha: bool = False) -> pygame.Surface:
        """A blank surface for the scene at SS times the size."""
        size = (self._scene[0] * SS, self._scene[1] * SS)
        return pygame.Surface(size, SRCALPHA if alpha else 0)

    def _shrink(self, layer: pygame.Surface) -> pygame.Surface:
        """Shrink a layer to the scene size."""
        return pygame.transform.smoothscale(layer, self._scene)

    def _draw_background(self) -> pygame.Surface:
        """The floor, shadows, obstacles and walls, with the top bar."""
        w, h = self.sim.world.size
        big = self._layer()
        big.fill(OUTSIDE)
        for i, j in itertools.product(
            range(math.ceil(w / TILE)), range(math.ceil(h / TILE))
        ):
            x0, y0 = map(round, self.to_pixel(i * TILE, min((j + 1) * TILE, h), SS))
            x1, y1 = map(round, self.to_pixel(min((i + 1) * TILE, w), j * TILE, SS))
            big.fill(FLOOR[(i + j) % 2], (x0, y0, x1 - x0, y1 - y0))
        shadows = self._layer(alpha=True)
        for obstacle in self.sim.world.obstacles:
            x, y, angle = obstacle.pose.xyt
            shifted = Transform2D.from_xyt((x + 0.07, y - 0.07, angle))
            self._draw_obstacle(shadows, obstacle.shape, shifted)
        big.blit(shadows, (0, 0))
        for obstacle in self.sim.world.obstacles:
            self._draw_obstacle(big, obstacle.shape, obstacle.pose, solid=True)
        x0, y0 = self.to_pixel(0, h, SS)
        x1, y1 = self.to_pixel(w, 0, SS)
        border = pygame.Rect(
            round(x0), round(y0), round(x1) - round(x0), round(y1) - round(y0)
        )
        pygame.draw.rect(big, WALL, border, width=round(0.08 * self.scale * SS))
        background = pygame.Surface(self.size)
        background.fill(BAR)
        background.blit(self._shrink(big), (0, HUD))
        return background

    def _draw_obstacle(
        self,
        surface: pygame.Surface,
        shape: Shape,
        pose: Transform2D,
        solid: bool = False,
    ) -> None:
        """Draw one obstacle, as a shadow unless `solid`."""
        points = self._points(outline(shape, pose))
        if isinstance(shape, LineString):
            width = round(0.08 * self.scale * SS)
            colour = WALL if solid else SHADOW
            pygame.draw.lines(surface, colour, False, points, width)
            for point in points:
                pygame.draw.circle(surface, colour, point, width / 2)
        else:
            pygame.draw.polygon(surface, OBSTACLE if solid else SHADOW, points)
            if solid:
                pygame.draw.polygon(surface, _darker(OBSTACLE), points, width=SS * 2)

    def step(self, commands: Commands) -> None:
        """Advance the simulation one step and record where each robot is."""
        self.sim.step(commands(self.sim))
        for name, pose in self.sim.poses.items():
            self.trails[name].append(tuple(pose.translation))

    def render(self) -> pygame.Surface:
        """The picture of the simulation as it is now."""
        layer = self._layer(alpha=True)
        robots = list(enumerate(self.sim.world.robots))
        for i, robot in robots:
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            if len(self.trails[robot.name]) > 1:
                width = round(0.03 * self.scale * SS)
                trail = (*pygame.Color(color)[:3], 170)
                points = self._points(self.trails[robot.name])
                pygame.draw.lines(layer, trail, False, points, width)
            for goal in self.goals[robot.name]:
                self._draw_goal(layer, goal, color)
        for i, robot in robots:
            self._draw_robot(layer, robot, ROBOT_COLORS[i % len(ROBOT_COLORS)])
        frame = self._background.copy()
        frame.blit(self._shrink(layer), (0, HUD))
        for _, robot in robots:
            self._draw_label(frame, robot)
        self._draw_bar(frame)
        return frame

    def _draw_goal(self, layer: pygame.Surface, goal, color: str) -> None:
        """A star in the colour of the robot."""
        x, y = goal
        points = []
        for k in range(10):
            r = 0.22 if k % 2 == 0 else 0.09
            angle = math.pi / 2 + k * math.pi / 5
            points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
        pixels = self._points(points)
        pygame.draw.polygon(layer, color, pixels)
        pygame.draw.polygon(layer, WALL, pixels, width=SS)

    def _draw_robot(self, layer: pygame.Surface, robot, color: str) -> None:
        """The body, with a white line from the centre toward the front."""
        pose = self.sim.poses[robot.name]
        points = self._points(outline(robot.shape, pose))
        pygame.draw.polygon(layer, color, points)
        pygame.draw.polygon(layer, _darker(color), points, width=SS * 2)
        x, y, angle = pose.xyt
        reach = 0.8 * self.reach[robot.name]
        centre = self.to_pixel(x, y, SS)
        front = self.to_pixel(
            x + reach * math.cos(angle), y + reach * math.sin(angle), SS
        )
        width = max(2, round(0.035 * self.scale * SS))
        pygame.draw.line(layer, "white", centre, front, width)
        pygame.draw.circle(layer, "white", centre, width)

    def _draw_label(self, frame: pygame.Surface, robot) -> None:
        """The robot's name above it."""
        x, y, _ = self.sim.poses[robot.name].xyt
        text = self._small.render(robot.name, True, WALL)
        cx, cy = self.to_pixel(x, y)
        top = cy + HUD - self.reach[robot.name] * self.scale - 16
        frame.blit(text, (cx - text.get_width() / 2, top))

    def _draw_bar(self, frame: pygame.Surface) -> None:
        """Time and speed on the left, the keys on the right."""
        state = "   DONE" if self.done else "   PAUSED" if self.paused else ""
        left = f"t = {self.sim.time:6.2f} s    speed x{self.speed:g}{state}"
        keys = "SPACE pause   RIGHT step   UP/DOWN speed   ESC quit"
        frame.blit(self._font.render(left, True, TEXT), (12, 8))
        right = self._small.render(keys, True, OUTSIDE)
        frame.blit(right, (self.size[0] - right.get_width() - 12, 10))

    def image(self) -> Image.Image:
        """The current picture as a Pillow image."""
        data = pygame.image.tobytes(self.render(), "RGB")
        return Image.frombytes("RGB", self.size, data)


def record(
    sim: Simulation,
    commands: Commands,
    steps: int,
    path: str,
    *,
    fps: int = 20,
    trail: int = TRAIL,
) -> None:
    """Run `steps` steps and save the run as a GIF, one frame every 1 / fps seconds.

    This needs no display.

    Args:
        sim: the simulation to run.
        commands: called before each step with the simulation; returns the
            command of each robot, by name.
        steps: number of steps to run.
        path: where to save the GIF.
        fps: frames per second of the GIF.
        trail: past positions kept for each robot's trail; 0 draws none.
    """
    view = View(sim, trail=trail)
    skip = max(1, round(1 / (fps * sim.dt)))
    frames = [view.image()]
    for i in range(1, steps + 1):
        view.step(commands)
        if i % skip == 0 or i == steps:
            frames.append(view.image())
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
    )


def _change_speed(speed: float, up: bool) -> float:
    """The next speed in `SPEEDS` above or below `speed`, or `speed` at the end."""
    if up:
        return min((s for s in SPEEDS if s > speed), default=speed)
    return max((s for s in SPEEDS if s < speed), default=speed)


def _on_key(view: View, key: int) -> None:
    """Apply a key that changes only the view: pause or speed."""
    if key == K_SPACE:
        view.paused = not view.paused
    elif key in (K_UP, K_DOWN):
        view.speed = _change_speed(view.speed, key == K_UP)


def _read_events(view: View) -> tuple[bool, bool]:
    """Handle the waiting events.

    Returns:
        Whether to keep running, and whether the user asked for a single step.
    """
    keep, step_once = True, False
    for event in pygame.event.get():
        if event.type == QUIT:
            keep = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                keep = False
            elif event.key == K_RIGHT:
                view.paused, step_once = True, True
            else:
                _on_key(view, event.key)
    return keep, step_once


def _window(view: View, commands: Commands, steps: int | None, caption: str) -> None:
    """Show the simulation in a window until it is closed."""
    pygame.display.init()
    screen = pygame.display.set_mode(view.size)
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    sim = view.sim
    owed = 0.0  # steps due but not yet taken, so the speed is exact on average
    running = True
    while running:
        running, step_once = _read_events(view)
        if not running:
            break
        elapsed = clock.tick(60) / 1000
        if step_once:
            owed = 1.0
        elif view.paused:
            owed = 0.0
        else:
            owed = min(owed + view.speed * elapsed / sim.dt, 20.0)
        while owed >= 1.0 and (steps is None or sim.steps < steps):
            view.step(commands)
            owed -= 1.0
        view.done = steps is not None and sim.steps >= steps
        screen.blit(view.render(), (0, 0))
        pygame.display.flip()
    pygame.display.quit()


def play(
    sim: Simulation,
    commands: Commands,
    steps: int | None = None,
    caption: str = "demuvo",
    trail: int = TRAIL,
) -> None:
    """Run the simulation in a window until it is closed.

    Args:
        sim: the simulation to run.
        commands: called before each step with the simulation; returns the
            command of each robot, by name.
        steps: number of steps to run before the run counts as done, or None
            to run for as long as the window is open.
        caption: title of the window.
        trail: past positions kept for each robot's trail; 0 draws none.
    """
    _window(View(sim, trail=trail), commands, steps, caption)
