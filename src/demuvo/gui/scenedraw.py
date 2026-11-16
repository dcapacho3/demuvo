"""Drawing a world - floor, costmaps, obstacles, paths, goals, trails and lidar rays.

The layers that change rarely (floor, global costmap, obstacles, planned paths,
goals) are drawn once into a cut-out of the scene and copied each frame. The ones
that change every step (trails, rays, the local costmap) are drawn over it, inside
the board. Robots, on top of all, are in `robots`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from robomath import Transform2D

from ..world.shapes import Circle, LineString
from . import font, look
from .raster import Canvas, is_convex, rgba
from .robots import draw_robots, footprint, robot_at, robot_boxes
from .sprites import SpriteCache
from .strokes import polyline, polylines, segments
from .view import View

# The layers, bottom first: name, label and whether it shows at first.
LAYERS = {
    "floor": ("floor", True),
    "costmap": ("global costmap", True),
    "local": ("local costmap", False),
    "obstacles": ("obstacles", True),
    "actors": ("actors", True),
    "routes": ("actor routes", True),
    "trails": ("trails", True),
    "paths": ("paths", True),
    "goals": ("goals", True),
    "rays": ("lidar rays", False),
    "robots": ("robots", True),
}
# A click this close, in pixels, to an edge counts as on the shape.
CLICK_PIXELS = 6.0
# A click this close, in pixels, to a waypoint dot or a leg's middle picks it.
ROUTE_PIXELS = 10.0
# Points kept in the trail of a robot.
TRAIL = 400
# A changed path redraws the cut-out at most this often, in seconds.
PATH_CHECK = 0.35
# Length of the goal arrow in metres.
GOAL_ARROW = 0.62
# The layers that go into the cut-out.
STATIC = ("floor", "costmap", "obstacles", "paths", "goals")


@dataclass
class Layers:
    """Which layers show.

    Attributes:
        shown: how far each layer is faded in, 0 to 1; animating it fades a layer.
        opacity: the opacity a layer has when it is on, 0 to 1.
    """

    shown: dict[str, float] = field(
        default_factory=lambda: {n: 1.0 if on else 0.0 for n, (_, on) in LAYERS.items()}
    )
    opacity: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(LAYERS, 1.0)
    )

    def alpha(self, name: str) -> float:
        """The opacity a layer is drawn with now."""
        return self.shown[name] * self.opacity[name]


@dataclass
class _Cutout:
    """The cached pixels of the layers that change rarely."""

    pixels: np.ndarray | None = None
    key: object = None
    signature: tuple = ()
    checked: float = 0.0
    keep: tuple = ()
    world: object = None


class SceneDraw:
    """Draws a run's world with the renderer.

    Args:
        run: a `Run`; the world, its costmaps, poses, paths and scans come from it.

    Attributes:
        run: the run being drawn.
        world: its world.
        view: how metres map to pixels; see `fit`.
        layers: which layers show.
        palette: the costmap palette, one of `look.PALETTES`.
        selected: the name of the selected robot, or None.
        selected_obstacle: the 0-based place of the selected obstacle, or None.
        selected_actor: the 0-based place of the selected actor, or None.
        trails: the recent places of each robot, by name.
        colors: the colour of each robot, by name.
        feet: the footprint of each robot, by name.
        sprites: the cache of small pictures.
    """

    def __init__(self, run) -> None:
        """Start drawing a run."""
        self.layers = Layers()
        self.palette = look.PALETTES[0]
        self.view: View | None = None
        self.selected: str | None = None
        self.selected_obstacle: int | None = None
        self.selected_actor: int | None = None
        self._dragged_actor: dict[int, np.ndarray] = {}
        # The waypoint being dragged, as (its place, x, y), drawn there until dropped.
        self.drag_point: tuple[int, float, float] | None = None
        self._actor_poses: dict[str, np.ndarray] = {}
        self._dragged: dict[str, np.ndarray] = {}
        self.sprites = SpriteCache()
        self._cutout = _Cutout()
        self._local: dict[str, tuple[object, np.ndarray, str]] = {}
        self._poses: dict[str, np.ndarray] = {}
        self.set_run(run)

    @property
    def world(self):
        """The world of the run."""
        return self.run.world

    def set_run(self, run) -> None:
        """Draw another run, or the same one after its world was replaced."""
        self.run = run
        # keep the window rectangle, so an edit does not shrink the board to a corner
        rect = self.view.rect if self.view is not None else View.rect
        self.view = View(self.world.bounds, rect=rect).fit(rect)
        self._poses = {}
        self.trails: dict[str, deque] = {
            r.name: deque([tuple(r.pose.xyt[:2])], maxlen=TRAIL)
            for r in self.world.robots
        }
        self.colors = {
            r.name: look.robot_color(i) for i, r in enumerate(self.world.robots)
        }
        self.feet = {r.name: footprint(r.shape) for r in self.world.robots}
        self._last_boxes: dict[str, tuple[int, int, int, int]] | None = None
        self._local.clear()
        self.sprites.clear()

    def fit(
        self, rect: tuple[float, float, float, float], bottom_room: float = 60.0
    ) -> None:
        """Fit the board into a window rectangle."""
        fitted = self.view.fit(rect, bottom_room)
        if fitted != self.view:  # the same view keeps the cached layers
            self.view = fitted
            self._cutout.key = None

    def refresh_poses(self) -> None:
        """Read every robot's and actor's pose once; one at a time rebuilds all."""
        self._poses = self.run.poses
        self._actor_poses = self.run.actor_poses

    def drag(self, name: str, x: float, y: float, angle: float) -> None:
        """Draw a robot at a pose of its own, while the user drags it."""
        self._dragged = {name: np.array([x, y, angle])}

    def drop(self) -> None:
        """Draw the dragged robot where the run has it again."""
        self._dragged = {}

    def pose(self, name: str) -> np.ndarray:
        """The pose of a robot as drawn: where it is dragged to, else where it is."""
        if name in self._dragged:
            return self._dragged[name]
        if name not in self._poses:
            self.refresh_poses()
        return self._poses[name]

    def drag_actor(self, index: int, x: float, y: float, angle: float) -> None:
        """Draw an actor at a pose of its own, while the user drags it."""
        self._dragged_actor = {index: np.array([x, y, angle])}

    def drop_actor(self) -> None:
        """Draw the dragged actor where its motion has it again."""
        self._dragged_actor = {}

    def actor_pose(self, index: int) -> np.ndarray:
        """An actor's pose as drawn: where it is dragged to, else where it is."""
        if index in self._dragged_actor:
            return self._dragged_actor[index]
        name = self.world.actors[index].name
        if name not in self._actor_poses:
            self.refresh_poses()
        return self._actor_poses[name]

    def record_trails(self) -> None:
        """Add each robot's place to its trail."""
        self.refresh_poses()
        for name, pose in self._poses.items():
            self.trails[name].append((float(pose[0]), float(pose[1])))

    def reset_history(self) -> None:
        """Empty the trails, each keeping the robot's place now."""
        for robot in self.world.robots:
            self.trails[robot.name].clear()
            self.trails[robot.name].append(tuple(self.pose(robot.name)[:2]))

    def _path_signature(self) -> tuple:
        """A short description of every planned path, to see when one changed."""
        found = []
        for robot in self.world.robots:
            path = self.run.path(robot.name)
            ends = None
            if path is not None and len(path):
                ends = (
                    len(path),
                    round(float(path[-1, 0]), 3),
                    round(float(path[-1, 1]), 3),
                )
            found.append(ends)
        return tuple(found)

    def cutout_key(self) -> tuple[dict[str, float], tuple]:
        """The opacities of the cached layers, and the key to tell a stale cut-out."""
        alphas = {n: round(self.layers.alpha(n), 3) for n in STATIC}
        now = time.perf_counter()
        if (
            now - self._cutout.checked > PATH_CHECK
            or self._cutout.world is not self.world
        ):
            self._cutout.signature = self._path_signature()
            self._cutout.checked = now
            self._cutout.world = self.world
        world = self.world
        # Edits leave the obstacles and tasks they do not touch as the same objects,
        # and a rebuilt simulation has an equal costmap, so a robot that moves keeps
        # the cache. `keep` holds them so an id is never taken for another object.
        key = (
            self.view,
            self.palette,
            tuple(alphas.values()),
            self._costmap_key(),
            self._cutout.signature,
            id(world.obstacles),
            id(world.tasks),
            world.size,
            world.origin,
            tuple(r.name for r in world.robots),
        )
        self._cutout.keep = (world.obstacles, world.tasks)
        return alphas, key

    def _costmap_key(self) -> tuple | None:
        """What the drawn global costmap holds, so an equal one is not drawn again."""
        costmap = self.run.sim.costmaps[self.world.robots[0].name]
        if costmap is None:
            return None
        return (hash(costmap.data.tobytes()), costmap.resolution, tuple(costmap.origin))

    def _draw_cutout(self, alphas: dict[str, float], key: object) -> None:
        """Draw the cached layers into a cut-out the size of the scene."""
        x, y, width, height = (round(v) for v in self.view.rect)
        canvas = Canvas(width, height, look.WINDOW)
        view = self.view.shifted(-x, -y)
        if alphas["floor"] > 0:
            self._floor(canvas, view, alphas["floor"])
        if alphas["costmap"] > 0.001:
            self._costmap(canvas, view, alphas["costmap"])
        if alphas["obstacles"] > 0.001:
            self._obstacles(canvas, view, alphas["obstacles"])
        left, top, right, bottom = view.board()
        outline = (left - 1, top - 1, right - left + 2, bottom - top + 2)
        canvas.rect_outline(
            outline, rgba(look.NAVY, alphas["obstacles"] * 0.9 + 0.1), width=3
        )
        canvas.set_clip(
            max(round(left), 0),
            max(round(top), 0),
            min(round(right), width),
            min(round(bottom), height),
        )
        if alphas["paths"] > 0.001:
            self._paths(canvas, view, alphas["paths"])
        if alphas["goals"] > 0.001:
            self._goals(canvas, view, alphas["goals"])
        self._cutout.pixels, self._cutout.key = canvas.px, key

    def draw_cutout(self, canvas: Canvas) -> None:
        """Copy the cached layers into the canvas inside its clip, redrawn if stale."""
        x, y, width, height = (round(v) for v in self.view.rect)
        if width < 2 or height < 2:
            return
        alphas, key = self.cutout_key()
        if key != self._cutout.key:
            self._draw_cutout(alphas, key)
        cx0, cy0 = max(canvas.clip[0], x), max(canvas.clip[1], y)
        cx1, cy1 = min(canvas.clip[2], x + width), min(canvas.clip[3], y + height)
        if cx1 > cx0 and cy1 > cy0:
            canvas.px[cy0:cy1, cx0:cx1] = self._cutout.pixels[
                cy0 - y : cy1 - y, cx0 - x : cx1 - x
            ]

    def _floor(self, canvas: Canvas, view: View, opacity: float) -> None:
        """The chess-board floor, one tile a metre."""
        x0, y0, _, _ = view.bounds
        left, top, right, bottom = view.board()
        i0, j0 = max(int(left), 0), max(int(top), 0)
        i1, j1 = (
            min(math.ceil(right), canvas.width),
            min(math.ceil(bottom), canvas.height),
        )
        if i1 <= i0 or j1 <= j0:
            return
        xs = (np.arange(i0, i1) + 0.5 - view.ox) / view.scale - x0
        ys = (view.oy - (np.arange(j0, j1) + 0.5)) / view.scale - y0
        tile = (np.floor(xs)[None, :] + np.floor(ys)[:, None]).astype(int) % 2
        colors = np.array([rgba(look.MINT)[:3], rgba(look.MINT2)[:3]], dtype=np.float32)
        base = np.array(rgba(look.WINDOW)[:3], dtype=np.float32)
        canvas.px[j0:j1, i0:i1] = base + (colors[tile] - base) * opacity

    def _costmap(self, canvas: Canvas, view: View, opacity: float) -> None:
        """The global costmap of the first robot, if the world has one."""
        costmap = self.run.sim.costmaps[self.world.robots[0].name]
        if costmap is not None:
            self._picture(
                canvas,
                view,
                costmap,
                look.costmap_rgba(costmap.data, self.palette),
                opacity,
            )

    @staticmethod
    def _picture(canvas: Canvas, view: View, costmap, picture, opacity: float) -> None:
        """Draw a costmap picture where the costmap lies."""
        x0, y0 = costmap.origin
        top = view.to_screen(x0, y0 + costmap.data.shape[0] * costmap.resolution)
        origin = (float(top[0]), float(top[1]))
        canvas.image_scaled(picture, origin, costmap.resolution * view.scale, opacity)

    def _obstacles(self, canvas: Canvas, view: View, opacity: float) -> None:
        """Circles as discs, lines as thick strokes, other shapes as filled polygons."""
        fill, edge = rgba(look.NAVY2, opacity), rgba(look.NAVY, opacity)
        for obstacle in self.world.obstacles:
            shape = obstacle.shape
            if isinstance(shape, Circle):
                x, y = view.to_screen(*obstacle.pose.translation)
                canvas.circle(
                    (float(x), float(y)),
                    shape.radius * view.scale,
                    fill=fill,
                    stroke=edge,
                    width=2,
                )
                continue
            corners = np.array(shape.points(obstacle.pose), dtype=float)
            sx, sy = view.to_screen(corners[:, 0], corners[:, 1])
            screen = np.column_stack([sx, sy])
            if isinstance(shape, LineString):
                polyline(canvas, screen, 6, edge)
            elif is_convex(screen):
                canvas.convex(screen, fill=fill, stroke=edge, width=2)
            else:
                canvas.polygon(screen, fill)

    def _draw_actors(self, canvas: Canvas, opacity: float) -> None:
        """Every actor where it is now; one that is not solid is drawn hollow.

        Actors move every step, so they are drawn here and not into the cut-out.
        """
        view = self.view
        edge = rgba(look.NAVY, opacity)
        for index, spec in enumerate(self.world.actors):
            fill = rgba(look.AMBER, opacity * (0.35 if not spec.solid else 1.0))
            x, y, angle = (float(v) for v in self.actor_pose(index))
            pose = Transform2D.from_xyt(np.array([x, y, angle]))
            if isinstance(spec.shape, Circle):
                px, py = view.to_screen(x, y)
                canvas.circle(
                    (float(px), float(py)),
                    spec.shape.radius * view.scale,
                    fill=fill,
                    stroke=edge,
                    width=2,
                )
            else:
                corners = np.array(spec.shape.points(pose), dtype=float)
                sx, sy = view.to_screen(corners[:, 0], corners[:, 1])
                canvas.convex(
                    np.column_stack([sx, sy]), fill=fill, stroke=edge, width=2
                )
            self._actor_nose(canvas, spec, x, y, angle, edge)

    def route(self, index: int) -> np.ndarray:
        """The start and waypoints of an actor as drawn, with a dragged point moved."""
        stops = route_of(self.world.actors[index])
        if self.drag_point is not None and index == self.selected_actor:
            place, x, y = self.drag_point
            stops[place + 1] = (x, y)
        return stops

    def point_at(self, sx: float, sy: float) -> int | None:
        """The place of the selected actor's waypoint under a pixel, or None."""
        return self._near_route(sx, sy, lambda stops: stops[1:])

    def leg_at(self, sx: float, sy: float) -> int | None:
        """The place a new waypoint takes if the middle of a leg is dragged, or None.

        The legs are those of the selected actor. The leg that ends at waypoint
        k puts the new one at k, just before it.
        """
        return self._near_route(sx, sy, lambda stops: (stops[:-1] + stops[1:]) / 2)

    def _near_route(self, sx, sy, picks) -> int | None:
        """The place of the nearest picked route spot within a dot's reach, or None."""
        index = self.selected_actor
        if index is None:
            return None
        spots = picks(self.route(index))
        if not len(spots):
            return None
        px, py = self.view.to_screen(spots[:, 0], spots[:, 1])
        away = np.hypot(px - sx, py - sy)
        nearest = int(np.argmin(away))
        return nearest if away[nearest] <= ROUTE_PIXELS else None

    def _draw_routes(self, canvas: Canvas, opacity: float) -> None:
        """The route of each actor that has waypoints: a dashed line and numbered dots.

        The selected actor's route is drawn in full, the others' fainter. A loop
        closes back to the start, which is drawn as a ring.
        """
        view = self.view
        for index, spec in enumerate(self.world.actors):
            stops = self.route(index)
            if len(stops) < 2:
                continue
            faint = opacity * (1.0 if index == self.selected_actor else 0.45)
            color = rgba(look.AMBER, faint)
            sx, sy = view.to_screen(stops[:, 0], stops[:, 1])
            line = np.column_stack([sx, sy])
            if spec.motion_params.get("mode", "loop") == "loop":
                line = np.vstack([line, line[:1]])
            polyline(canvas, line, 2, color, dash=(7, 5))
            canvas.circle((float(sx[0]), float(sy[0])), 5, stroke=color, width=2)
            for number, (x, y) in enumerate(line[1 : len(stops)], 1):
                canvas.circle((float(x), float(y)), 9, fill=color)
                canvas.text(
                    font.bold(),
                    str(number),
                    (float(x) - 3 * len(str(number)), float(y) + 4),
                    size=11,
                    color=rgba(look.NAVY, faint),
                )

    def _actor_nose(self, canvas: Canvas, spec, x, y, angle, color) -> None:
        """A short tick from the middle, so which way an actor faces can be seen."""
        reach = _actor_reach(spec.shape)
        tip = (x + reach * math.cos(angle), y + reach * math.sin(angle))
        start = self.view.to_screen(x, y)
        end = self.view.to_screen(*tip)
        polyline(canvas, np.array([start, end], dtype=float), 2, color)

    def actor_boxes(self) -> dict[str, tuple[int, int, int, int]]:
        """The screen box of each actor now, by name, with room for its outline."""
        boxes = {}
        for index, spec in enumerate(self.world.actors):
            x, y, _ = (float(v) for v in self.actor_pose(index))
            reach = _actor_reach(spec.shape) * self.view.scale + 6
            px, py = (float(v) for v in self.view.to_screen(x, y))
            boxes[spec.name] = (
                int(px - reach),
                int(py - reach),
                int(px + reach) + 1,
                int(py + reach) + 1,
            )
        return boxes

    def actor_at(self, sx: float, sy: float) -> int | None:
        """The place of the topmost actor under a pixel, or None."""
        x, y = self.view.to_world(sx, sy)
        for index in range(len(self.world.actors) - 1, -1, -1):
            shape = self.world.actors[index].shape
            pose = Transform2D.from_xyt(np.array(self.actor_pose(index), dtype=float))
            if self._body_hit(x, y, shape, pose):
                return index
        return None

    def _draw_selected_actor(self, canvas: Canvas) -> None:
        """Outline the selected actor, which moves, so it is drawn every frame."""
        index = self.selected_actor
        if index is None or index >= len(self.world.actors):
            return
        spec = self.world.actors[index]
        x, y, angle = (float(v) for v in self.actor_pose(index))
        color = rgba(look.TEAL, 1.0)
        if isinstance(spec.shape, Circle):
            px, py = self.view.to_screen(x, y)
            radius = spec.shape.radius * self.view.scale + 3
            canvas.circle((float(px), float(py)), radius, stroke=color, width=2)
            return
        pose = Transform2D.from_xyt(np.array([x, y, angle]))
        corners = np.array(spec.shape.points(pose), dtype=float)
        sx, sy = self.view.to_screen(corners[:, 0], corners[:, 1])
        ring = np.column_stack([sx, sy])
        polyline(canvas, np.vstack([ring, ring[:1]]), 3, color, dash=(6, 4))

    def _paths(self, canvas: Canvas, view: View, opacity: float) -> None:
        """A dashed line for each robot with a goal: its path, else straight there."""
        goals = {t.robot: t.goal.xyt for t in self.world.tasks}
        paths, colors = [], []
        for robot in self.world.robots:
            if robot.name not in goals:
                continue
            planned = self.run.path(robot.name)
            if planned is not None and len(planned) > 1:
                sx, sy = view.to_screen(planned[:, 0], planned[:, 1])
            else:
                start, goal = robot.pose.xyt, goals[robot.name]
                sx, sy = view.to_screen(
                    np.array([start[0], goal[0]]), np.array([start[1], goal[1]])
                )
            paths.append(np.column_stack([sx, sy]))
            colors.append(rgba(self.colors[robot.name], opacity))
        polylines(canvas, paths, colors, 2.4, dash=(7, 5))

    def _goals(self, canvas: Canvas, view: View, opacity: float) -> None:
        """Each goal as an arrow along its heading, in a ring of the tolerance."""
        for task in self.world.tasks:
            color = self.colors[task.robot]
            gx, gy, heading = task.goal.xyt
            x, y = (float(v) for v in view.to_screen(gx, gy))
            ring = max(task.goal_tolerance[0] * view.scale, 4.0)
            length = GOAL_ARROW * view.scale
            key = (
                "goal",
                color,
                round(view.scale, 1),
                round(float(heading), 2),
                round(ring, 1),
            )
            sprite = self.sprites.get(
                key,
                2 * (length + ring) + 12,
                goal_drawing(color, ring, heading, length),
            )
            canvas.blend_rgba(
                round(x - sprite.width / 2),
                round(y - sprite.height / 2),
                sprite.rgba,
                opacity,
            )

    def _local_picture(self, name: str):
        """The local costmap of a robot and its picture, kept until either changes."""
        costmap = self.run.local_costmap(name)
        if costmap is None:
            return None
        kept = self._local.get(name)
        if kept is None or kept[0] is not costmap or kept[2] != self.palette:
            kept = (
                costmap,
                look.costmap_rgba(costmap.data, self.palette),
                self.palette,
            )
            self._local[name] = kept
        return costmap, kept[1]

    def _draw_local(self, canvas: Canvas, opacity: float) -> None:
        """The local costmap of every robot that has one."""
        for robot in self.world.robots:
            found = self._local_picture(robot.name)
            if found:
                self._picture(canvas, self.view, found[0], found[1], opacity)

    def _draw_trails(self, canvas: Canvas, opacity: float) -> None:
        """The recent path of each robot, in its colour."""
        paths, colors = [], []
        for robot in self.world.robots:
            points = np.array(self.trails[robot.name], dtype=float)
            if len(points) > 1:
                sx, sy = self.view.to_screen(points[:, 0], points[:, 1])
                paths.append(np.column_stack([sx, sy]))
                colors.append(rgba(self.colors[robot.name], 0.9 * opacity))
        polylines(canvas, paths, colors, 3)

    def _draw_rays(self, canvas: Canvas, opacity: float) -> None:
        """The beams of every lidar; a ray that hit nothing reaches its maximum."""
        for robot in self.world.robots:
            for beams in self.run.scans(robot.name):
                reach = np.where(
                    np.isfinite(beams.ranges), beams.ranges, beams.max_range
                )
                ox, oy = beams.origin
                sx, sy = self.view.to_screen(ox, oy)
                ex, ey = self.view.to_screen(
                    ox + reach * np.cos(beams.angles), oy + reach * np.sin(beams.angles)
                )
                ends = np.column_stack(
                    [np.full(len(ex), float(sx)), np.full(len(ex), float(sy)), ex, ey]
                )
                segments(
                    canvas, ends, 1.0, rgba(self.colors[robot.name], 0.45 * opacity)
                )

    def robot_at(self, sx: float, sy: float) -> str | None:
        """The robot under a pixel, or None."""
        return robot_at(self, sx, sy)

    def heading_to(self, name: str, x: float, y: float) -> float:
        """The heading from a robot towards a world point."""
        rx, ry = self.pose(name)[:2]
        return math.atan2(y - ry, x - rx)

    def obstacle_at(self, sx: float, sy: float) -> int | None:
        """The place of the topmost obstacle under a pixel, or None."""
        x, y = self.view.to_world(sx, sy)
        for index in range(len(self.world.obstacles) - 1, -1, -1):
            obstacle = self.world.obstacles[index]
            if self._body_hit(x, y, obstacle.shape, obstacle.pose):
                return index
        return None

    def _body_hit(self, x: float, y: float, shape, pose) -> bool:
        """True if a world point is on a body: inside it, or near the edge of a line."""
        if isinstance(shape, Circle):
            cx, cy = pose.translation
            return math.hypot(x - cx, y - cy) <= shape.radius
        points = shape.points(pose)
        closed = shape.closed
        reach = CLICK_PIXELS / self.view.scale
        return _near_edge(x, y, points, reach, closed=closed) or (
            closed and _in_polygon(x, y, points)
        )

    def _draw_selected_obstacle(self, canvas: Canvas) -> None:
        """Outline the selected obstacle; it is drawn apart from the cached layers."""
        index = self.selected_obstacle
        if index is None or index >= len(self.world.obstacles):
            return
        obstacle = self.world.obstacles[index]
        color = rgba(look.TEAL, 1.0)
        if isinstance(obstacle.shape, Circle):
            x, y = self.view.to_screen(*obstacle.pose.translation)
            radius = obstacle.shape.radius * self.view.scale + 3
            canvas.circle((float(x), float(y)), radius, stroke=color, width=2)
            return
        corners = np.array(obstacle.shape.points(obstacle.pose), dtype=float)
        sx, sy = self.view.to_screen(corners[:, 0], corners[:, 1])
        ring = np.column_stack([sx, sy])
        polyline(canvas, np.vstack([ring, ring[:1]]), 3, color, dash=(6, 4))

    def draw(self, canvas: Canvas) -> None:
        """Draw the whole scene inside the canvas's clip, which may be a part of it.

        The cached layers come first, then, inside the board, the layers that
        change every step, and last the robots, which may spill over the board.
        """
        x, y, width, height = self.view.rect
        saved = canvas.clip
        self.refresh_poses()
        left, top = max(saved[0], int(x)), max(saved[1], int(y))
        right, bottom = min(saved[2], int(x + width)), min(saved[3], int(y + height))
        if right <= left or bottom <= top:
            return
        canvas.set_clip(left, top, right, bottom)
        self.draw_cutout(canvas)
        board = (round(v) for v in self.view.board())
        b0, b1, b2, b3 = board
        canvas.set_clip(max(left, b0), max(top, b1), min(right, b2), min(bottom, b3))
        self.draw_overlays(canvas)
        self._draw_selected_obstacle(canvas)
        self._draw_selected_actor(canvas)
        canvas.set_clip(left, top, right, bottom)
        if self.layers.alpha("robots") > 0.001:
            draw_robots(self, canvas, self.layers.alpha("robots"))
        canvas.clip = saved

    def dirty_rect(self) -> tuple[int, int, int, int]:
        """The smallest box that holds what changed since the last call.

        That is where each robot was and is now. The whole scene is returned for the
        first call, and when something global changed: the cached layers, the rays,
        the local costmap, or the robots.
        """
        x, y, width, height = self.view.rect
        full = (int(x), int(y), int(x + width), int(y + height))
        self.refresh_poses()
        _, key = self.cutout_key()
        boxes = robot_boxes(self) | self.actor_boxes()
        last, self._last_boxes = self._last_boxes, boxes
        everything = (
            last is None
            or key != self._cutout.key
            or self.layers.alpha("rays") > 0.001
            or self.layers.alpha("local") > 0.001
            or set(last) != set(boxes)
        )
        if everything:
            return full
        every = [last[n] for n in boxes] + list(boxes.values())
        x0, y0 = min(b[0] for b in every), min(b[1] for b in every)
        x1, y1 = max(b[2] for b in every), max(b[3] for b in every)
        return (max(x0, full[0]), max(y0, full[1]), min(x1, full[2]), min(y1, full[3]))

    def draw_overlays(self, canvas: Canvas) -> None:
        """Draw the layers that change every step, inside the board and the clip."""
        alpha = self.layers.alpha
        if alpha("local") > 0.001:
            self._draw_local(canvas, alpha("local"))
        if alpha("trails") > 0.001:
            self._draw_trails(canvas, alpha("trails"))
        if alpha("rays") > 0.001:
            self._draw_rays(canvas, alpha("rays"))
        if alpha("routes") > 0.001:
            self._draw_routes(canvas, alpha("routes"))
        if alpha("actors") > 0.001:
            self._draw_actors(canvas, alpha("actors"))


def route_of(spec) -> np.ndarray:
    """The start and then the waypoints of an actor, as rows of (x, y)."""
    start = [float(v) for v in spec.pose.xyt[:2]]
    points = spec.motion_params.get("points", []) if spec.motion else []
    return np.array([start, *points], dtype=float).reshape(-1, 2)


def _actor_reach(shape) -> float:
    """How far an actor's body reaches from its middle, in metres."""
    if isinstance(shape, Circle):
        return shape.radius
    return max(math.hypot(x, y) for x, y in shape.points(Transform2D.identity()))


def _near_edge(x: float, y: float, points, reach: float, *, closed: bool) -> bool:
    """True if a point is within `reach` of an edge, so a thin line can be clicked."""
    ends = zip(points, [*points[1:], points[0]] if closed else points[1:])
    for (ax, ay), (bx, by) in ends:
        dx, dy = bx - ax, by - ay
        along = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy or 1.0)
        along = min(max(along, 0.0), 1.0)
        if math.hypot(x - (ax + along * dx), y - (ay + along * dy)) <= reach:
            return True
    return False


def _in_polygon(x: float, y: float, points) -> bool:
    """True if a point is inside a polygon, by counting edges to its right."""
    inside = False
    for (x0, y0), (x1, y1) in zip(points, [*points[1:], points[0]]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def goal_drawing(color: str, ring: float, heading: float, length: float):
    """A function that draws a goal marker at the centre of a sprite."""

    def draw(canvas: Canvas, cx: float, cy: float) -> None:
        canvas.circle((cx, cy), ring, stroke=rgba(color), width=2)
        arrow(canvas, (cx, cy), -heading, length, rgba(color), outline=rgba("#ffffff"))

    return draw


def arrow(
    canvas: Canvas, origin, angle: float, length: float, fill, *, outline=None
) -> None:
    """A shaft and a head along a screen angle (y down), with an optional outline."""
    cos, sin = math.cos(angle), math.sin(angle)
    head = 0.34 * length

    def point(u, v):
        return (origin[0] + u * cos - v * sin, origin[1] + u * sin + v * cos)

    half = 0.07 * length
    shaft = [
        point(0, -half),
        point(length - head, -half),
        point(length - head, half),
        point(0, half),
    ]
    tip = [
        point(length - head, -0.22 * length),
        point(length, 0),
        point(length - head, 0.22 * length),
    ]
    for part in (shaft, tip):
        if outline is not None:
            canvas.convex(part, fill=outline, stroke=outline, width=2.0)
    for part in (shaft, tip):
        canvas.convex(part, fill=fill)
