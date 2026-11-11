"""The objects a world file describes."""

from __future__ import annotations

from dataclasses import dataclass, field

from robomath import Transform2D

from ..plugins import check_positive
from .shapes import Shape

DEFAULT_GOAL_TOLERANCE = (0.1, 0.1)
DEFAULT_DT = 0.01
DEFAULT_CONTROL_PERIOD = 0.05


@dataclass(frozen=True)
class Obstacle:
    """A fixed object.

    Attributes:
        pose: where its origin is and which way it points.
        shape: its shape.
        name: what it is called, or "" for no name. Names are for people: the
            window lists an obstacle by name, and a collision names it.
    """

    pose: Transform2D
    shape: Shape
    name: str = ""


@dataclass(frozen=True)
class RobotSpec:
    """A robot as the world file describes it.

    Attributes:
        name: unique name.
        model: name of a kinematic model, for example "diff_drive".
        shape: its body, a shape that encloses an area.
        pose: where it starts.
        controller: "module:Class" of the controller, or None.
        params: values that replace the model defaults, such as max_v.
        planner: "module:Class" of the planner, or None for the default.
        controller_params: keyword arguments for the controller class.
        planner_params: keyword arguments for the planner class.
        control_period: seconds between two runs of the controller; at least
            the simulation step. The command stays in force between runs.
        estimator: name of the estimator that gives the controller its pose,
            "ground_truth" (the true pose) or "dead_reckoning" (from the
            encoders). A name the model cannot run was already replaced by
            "ground_truth" when the file was read.
        sensors: settings of this robot's sensors by name, such as
            {"encoders": {"noise": 0.01}}; they replace the world's, key by key.
            A lidar can have a name of its own and a "type" of "lidar", so a robot
            can carry several.
        control_pose: the pose the controller and planner see: "estimate" (the
            estimator's) or "truth" (the true pose, while the estimator still
            runs and is scored).
        smoother: "module:Class" of the smoother that improves the planner's
            path, or None for none.
        smoother_params: keyword arguments for the smoother class.
    """

    name: str
    model: str
    shape: Shape
    pose: Transform2D
    controller: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    planner: str | None = None
    controller_params: dict = field(default_factory=dict)
    planner_params: dict = field(default_factory=dict)
    control_period: float = DEFAULT_CONTROL_PERIOD
    estimator: str = "ground_truth"
    sensors: dict = field(default_factory=dict)
    control_pose: str = "estimate"
    smoother: str | None = None
    smoother_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """Something a robot is to do: go to a goal.

    Several tasks for one robot, in file order, are waypoints.

    Attributes:
        robot: name of the robot.
        goal: where it should end.
        goal_tolerance: distance in metres and angle in radians within which
            the robot counts as at its goal.
        stop: True if the robot must also be standing still to finish the
            task. None means yes for the robot's last task and for any task
            with a `wait`, no for the others.
        wait: seconds the robot must hold the goal before the task is done.
    """

    robot: str
    goal: Transform2D
    goal_tolerance: tuple[float, float] = DEFAULT_GOAL_TOLERANCE
    stop: bool | None = None
    wait: float = 0.0


@dataclass(frozen=True)
class CostmapSpec:
    """How the global costmap is built.

    Attributes:
        enabled: False gives no costmap, and a planner gets None.
        resolution: side of a cell in metres. A map read from a file keeps its own.
        plugins: the layers to use, "static" then "inflation".
        source: "generate" (from the obstacles of the world) or the path of a
            `map.yaml`.
        unknown: what a cell of a map file between its free and occupied
            thresholds counts as, "free" or "blocked" (no information, which a
            planner will not cross).
        inflation_radius: how far cost spreads from an obstacle, in metres.
        cost_scaling_factor: how fast cost falls with distance, higher is faster.
    """

    enabled: bool = True
    resolution: float = 0.05
    plugins: tuple[str, ...] = ("static", "inflation")
    source: str = "generate"
    inflation_radius: float = 0.55
    cost_scaling_factor: float = 10.0
    unknown: str = "free"


LOCAL_LAYERS = ("obstacle", "inflation")
# The local costmap settings that are lengths or times, so must be above 0.
LOCAL_POSITIVE = (
    "window",
    "resolution",
    "period",
    "obstacle_max_range",
    "raytrace_max_range",
    "inflation_radius",
    "cost_scaling_factor",
)


@dataclass(frozen=True)
class LocalCostmapSpec:
    """How the local costmap of a robot with a lidar is built.

    Attributes:
        enabled: False gives no local costmap, whatever the robot carries.
        window: side of the square window round the robot, in metres.
        resolution: side of a cell in metres.
        period: seconds between updates.
        obstacle_max_range: the farthest a hit is marked as an obstacle, in metres.
        raytrace_max_range: the farthest a beam clears cells, in metres; a beam
            that hit nothing clears up to it.
        inflation_radius: how far cost spreads from an obstacle, in metres.
        cost_scaling_factor: how fast cost falls with distance; higher is faster.
        plugins: the layers to use, "obstacle" then "inflation".
    """

    enabled: bool = True
    window: float = 3.0
    resolution: float = 0.05
    period: float = 0.2
    obstacle_max_range: float = 2.5
    raytrace_max_range: float = 3.0
    inflation_radius: float = 0.55
    cost_scaling_factor: float = 10.0
    plugins: tuple[str, ...] = LOCAL_LAYERS

    def __post_init__(self) -> None:
        """Check the settings."""
        check_positive(self, *LOCAL_POSITIVE)
        if list(self.plugins) != [p for p in LOCAL_LAYERS if p in self.plugins]:
            raise ValueError(
                f"plugins must list 'obstacle' and 'inflation', in that order, "
                f"each at most once, got {self.plugins!r}"
            )


@dataclass(frozen=True)
class World:
    """Everything in a world file.

    Attributes:
        size: width along x and height along y in metres; the border is a wall.
        seed: seed for every random number the run uses.
        obstacles: the obstacles, in file order.
        robots: the robots, in file order.
        tasks: what the robots are to do, in file order.
        dt: length of a simulation step in seconds.
        sensors: default settings of the sensors of every robot, by sensor
            name, such as {"encoders": {"noise": 0.01, "ticks_per_rev": 2048}}.
        costmap: how the global costmap is built.
        local_costmap: how the local costmap of a robot with a lidar is built.
        origin: where the lower-left corner of the world is, in metres. The world
            spans from there to the origin plus `size`.
    """

    size: tuple[float, float]
    seed: int
    obstacles: tuple[Obstacle, ...]
    robots: tuple[RobotSpec, ...]
    tasks: tuple[Task, ...] = ()
    dt: float = DEFAULT_DT
    sensors: dict = field(default_factory=dict)
    costmap: CostmapSpec = field(default_factory=CostmapSpec)
    local_costmap: LocalCostmapSpec = field(default_factory=LocalCostmapSpec)
    origin: tuple[float, float] = (0.0, 0.0)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The corners of the world as (x_min, y_min, x_max, y_max)."""
        (x, y), (width, height) = self.origin, self.size
        return x, y, x + width, y + height
