"""Reading and checking a world file."""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

from robomath import Transform2D

from ..models import MODELS, parse_params
from ..sensing import ESTIMATORS, PAYLOAD, SENSORS, resolve_estimator
from ..verify import (
    WorldError,
    check_keys,
    inside,
    items,
    number,
    numbers,
    require_key,
    table,
    text,
)
from .shapes import parse_shape
from .spec import (
    DEFAULT_CONTROL_PERIOD,
    DEFAULT_DT,
    DEFAULT_GOAL_TOLERANCE,
    CostmapSpec,
    LocalCostmapSpec,
    Obstacle,
    RobotSpec,
    Task,
    World,
)

FORMAT_VERSION = 1


def _parse_obstacle(item: dict, index: int, bounds) -> Obstacle:
    """Build and check one `[[obstacle]]`."""
    label = f"obstacle {index}"
    item = table(item, label)
    check_keys(item, {"shape", "pose"}, label)
    shape = parse_shape(require_key(item, "shape", label), label)
    if "pose" in item or not shape.pose_optional:
        pose = _pose(item, label)
    else:
        pose = Transform2D.identity()
    if not inside(shape.points(pose), bounds):
        raise WorldError(f"{label}: must lie inside the world")
    return Obstacle(pose, shape)


def _pose(item: dict, label: str, key: str = "pose") -> Transform2D:
    """Read a required (x, y, angle) as a pose."""
    return Transform2D.from_xyt(numbers(require_key(item, key, label), 3, label, key))


def _plugin(item: dict, key: str, label: str) -> str | None:
    """Read an optional "module:Class" string."""
    if key not in item:
        return None
    value = text(item, key, label)
    module, _, cls = value.partition(":")
    if not (module and cls):
        raise WorldError(f"{label}: '{key}' must be 'module:Class'")
    return value


def _plugin_params(item: dict, key: str, plugin: str, label: str) -> dict:
    """Read the keyword arguments of a plugin; they need the plugin's key."""
    if key not in item:
        return {}
    if plugin not in item:
        raise WorldError(f"{label}: '{key}' needs '{plugin}'")
    return dict(table(item[key], f"{label} {key}"))


def _sensor_kind(name: str, settings: dict, label: str, named: bool) -> str:
    """The kind of sensor a `sensors` entry is: its `type` if named, else its key."""
    if named and "type" in settings:
        kind = settings["type"]
        if kind not in PAYLOAD:
            known = ", ".join(sorted(PAYLOAD))
            raise WorldError(
                f"{label} sensors.{name}: 'type' must be one of {known}, got {kind!r}"
            )
        return kind
    if name not in SENSORS:
        hint = ", or give it a 'type'" if named else ""
        raise WorldError(f"{label} sensors: unknown key '{name}'{hint}")
    return name


def _parse_sensors(value, label: str, named: bool = False) -> dict[str, dict]:
    """Check a `sensors` table: sensors, each with its own settings.

    A sensor is named by its kind. With `named`, a lidar may have any name and a
    `type`, so a robot can carry several.
    """
    data = table(value, f"{label} sensors")
    result = {}
    for name, settings in data.items():
        where = f"{label} sensors.{name}"
        settings = table(settings, where)
        kind = _sensor_kind(name, settings, label, named)
        cls = SENSORS[kind]
        check_keys(settings, set(cls.options) | ({"type"} if named else set()), where)
        checked = {"type": kind} if "type" in settings else {}
        for key, number_or_list in settings.items():
            if key == "type":
                continue
            if key in getattr(cls, "vectors", {}):
                checked[key] = numbers(number_or_list, cls.vectors[key], where, key)
                continue
            numbers([number_or_list], 1, where, key)  # finite; zero is allowed
            if number_or_list < 0:
                raise WorldError(f"{where}: '{key}' must not be negative")
            checked[key] = float(number_or_list)
        result[name] = checked
    return result


def _warn_ignored_sensors(sensors: dict, estimator: str, label: str) -> None:
    """Warn about a sensor setting that nothing reads: the estimator or a payload."""
    for sensor, settings in sensors.items():
        kind = settings.get("type", sensor)
        if kind not in ESTIMATORS[estimator].needs and kind not in PAYLOAD:
            warnings.warn(
                f"{label}: sensors.{sensor} is set but estimator '{estimator}' "
                "does not read it, so the setting has no effect"
            )


def _parse_robot(item: dict, index: int, bounds, names: set[str]) -> RobotSpec:
    """Build and check one `[[robot]]`; `names` holds the names read so far."""
    item = table(item, f"robot {index}")
    name = text(item, "name", f"robot {index}")
    label = f"robot '{name}'"
    if name in names:
        raise WorldError(f"{label}: name used twice")
    names.add(name)
    check_keys(
        item,
        {
            "name",
            "model",
            "shape",
            "pose",
            "planner",
            "planner_params",
            "smoother",
            "smoother_params",
            "controller",
            "controller_params",
            "params",
            "control_period",
            "estimator",
            "sensors",
            "control_pose",
        },
        label,
    )
    model = text(item, "model", label)
    if model not in MODELS:
        known = ", ".join(MODELS)
        raise WorldError(f"{label}: unknown model '{model}', expected one of {known}")
    params = parse_params(item.get("params", {}), model, label)
    shape = parse_shape(require_key(item, "shape", label), label, robot=True)
    pose = _pose(item, label)
    controller = _plugin(item, "controller", label)
    planner = _plugin(item, "planner", label)
    controller_params = _plugin_params(item, "controller_params", "controller", label)
    planner_params = _plugin_params(item, "planner_params", "planner", label)
    period = DEFAULT_CONTROL_PERIOD
    if "control_period" in item:
        period = number(item, "control_period", label)
    if not inside(shape.points(pose), bounds):
        raise WorldError(f"{label}: 'pose' must lie inside the world")
    estimator = (
        text(item, "estimator", label) if "estimator" in item else "ground_truth"
    )
    if estimator not in ESTIMATORS:
        known = ", ".join(ESTIMATORS)
        raise WorldError(
            f"{label}: unknown estimator '{estimator}', expected one of {known}"
        )
    estimator = resolve_estimator(estimator, model, MODELS[model], label)
    sensors = _parse_sensors(item.get("sensors", {}), label, named=True)
    _warn_ignored_sensors(sensors, estimator, label)
    control_pose = (
        text(item, "control_pose", label) if "control_pose" in item else "estimate"
    )
    if control_pose not in ("estimate", "truth"):
        raise WorldError(f"{label}: 'control_pose' must be 'estimate' or 'truth'")
    return RobotSpec(
        name,
        model,
        shape,
        pose,
        controller,
        params,
        planner,
        controller_params,
        planner_params,
        period,
        estimator,
        sensors,
        control_pose,
        _plugin(item, "smoother", label),
        _plugin_params(item, "smoother_params", "smoother", label),
    )


def _parse_task(item: dict, index: int, bounds, robots: dict[str, RobotSpec]) -> Task:
    """Build and check one `[[task]]`; the robot must be in `robots`."""
    label = f"task {index}"
    item = table(item, label)
    check_keys(item, {"robot", "goal", "goal_tolerance", "stop", "wait"}, label)
    name = text(item, "robot", label)
    if name not in robots:
        raise WorldError(f"{label}: no robot named '{name}'")
    goal = _pose(item, label, "goal")
    if not inside(robots[name].shape.points(goal), bounds):
        raise WorldError(f"{label}: 'goal' must lie inside the world")
    tolerance = DEFAULT_GOAL_TOLERANCE
    if "goal_tolerance" in item:
        tolerance = numbers(item["goal_tolerance"], 2, label, "goal_tolerance", True)
    stop = item.get("stop")
    if stop is not None and not isinstance(stop, bool):
        raise WorldError(f"{label}: 'stop' must be true or false")
    wait = number(item, "wait", label) if "wait" in item else 0.0
    return Task(name, goal, tolerance, stop, wait)


def _setting(table_: dict, key: str, label: str, default: float) -> float:
    """A positive number from the table, or the default if the key is absent."""
    return number(table_, key, label) if key in table_ else default


def _layers(table_: dict, order: tuple[str, ...], label: str) -> tuple[str, ...]:
    """The `plugins` of a costmap table: layers in the fixed order, each once."""
    plugins = table_.get("plugins", list(order))
    if not isinstance(plugins, list) or not set(plugins) <= set(order):
        names = " and ".join(f"'{name}'" for name in order)
        raise WorldError(f"{label}: 'plugins' must list {names}")
    if plugins != [p for p in order if p in plugins]:
        raise WorldError(
            f"{label}: 'plugins' names a layer twice or puts "
            f"{order[1]} before {order[0]}"
        )
    return tuple(plugins)


def _layer_table(table_: dict, layer: str, keys: set[str], label: str) -> dict:
    """The settings table of one layer, with its keys checked."""
    settings = table(table_.get(layer, {}), f"{label}.{layer}")
    check_keys(settings, keys, f"{label}.{layer}")
    return settings


def _warn_unused_layers(plugins: tuple[str, ...], layers: dict, label: str) -> None:
    """Warn about the settings of a layer that is not in `plugins`."""
    for layer, settings in layers.items():
        if settings and layer not in plugins:
            warnings.warn(
                f"{label}: {layer} is set but not in 'plugins', so it has no effect"
            )


def _enabled(table_: dict, label: str, default: bool) -> bool:
    """The `enabled` switch of a costmap table, which must be true or false."""
    enabled = table_.get("enabled", default)
    if not isinstance(enabled, bool):
        raise WorldError(f"{label}: 'enabled' must be true or false")
    return enabled


INFLATION = {"inflation_radius", "cost_scaling_factor"}


def _parse_global(value, base: Path) -> CostmapSpec:
    """Check `[costmap.global]`; a map path is taken relative to `base`."""
    label = "costmap.global"
    g = table(value, label)
    check_keys(g, {"enabled", "resolution", "plugins", "static", "inflation"}, label)
    default = CostmapSpec()
    plugins = _layers(g, ("static", "inflation"), label)
    static = _layer_table(g, "static", {"source"}, label)
    inflation = _layer_table(g, "inflation", INFLATION, label)
    _warn_unused_layers(plugins, {"static": static, "inflation": inflation}, label)
    source = (
        text(static, "source", f"{label}.static") if "source" in static else "generate"
    )
    if source != "generate":
        path = base / source
        if not path.is_file():
            raise WorldError(f"{label}.static: no map file '{path}'")
        source = str(path)
    return CostmapSpec(
        _enabled(g, label, default.enabled),
        _setting(g, "resolution", label, default.resolution),
        plugins,
        source,
        _setting(
            inflation,
            "inflation_radius",
            f"{label}.inflation",
            default.inflation_radius,
        ),
        _setting(
            inflation,
            "cost_scaling_factor",
            f"{label}.inflation",
            default.cost_scaling_factor,
        ),
    )


def _parse_local(value) -> LocalCostmapSpec:
    """Check `[costmap.local]`."""
    label = "costmap.local"
    local = table(value, label)
    check_keys(
        local,
        {
            "enabled",
            "window",
            "resolution",
            "period",
            "plugins",
            "obstacle",
            "inflation",
        },
        label,
    )
    default = LocalCostmapSpec()
    plugins = _layers(local, ("obstacle", "inflation"), label)
    obstacle = _layer_table(
        local, "obstacle", {"obstacle_max_range", "raytrace_max_range"}, label
    )
    inflation = _layer_table(local, "inflation", INFLATION, label)
    _warn_unused_layers(plugins, {"obstacle": obstacle, "inflation": inflation}, label)
    return LocalCostmapSpec(
        enabled=_enabled(local, label, default.enabled),
        window=_setting(local, "window", label, default.window),
        resolution=_setting(local, "resolution", label, default.resolution),
        period=_setting(local, "period", label, default.period),
        obstacle_max_range=_setting(
            obstacle,
            "obstacle_max_range",
            f"{label}.obstacle",
            default.obstacle_max_range,
        ),
        raytrace_max_range=_setting(
            obstacle,
            "raytrace_max_range",
            f"{label}.obstacle",
            default.raytrace_max_range,
        ),
        inflation_radius=_setting(
            inflation,
            "inflation_radius",
            f"{label}.inflation",
            default.inflation_radius,
        ),
        cost_scaling_factor=_setting(
            inflation,
            "cost_scaling_factor",
            f"{label}.inflation",
            default.cost_scaling_factor,
        ),
        plugins=plugins,
    )


def _parse_costmaps(value, base: Path) -> tuple[CostmapSpec, LocalCostmapSpec]:
    """Check the `[costmap]` table: the global costmap and the local one."""
    data = table(value, "costmap")
    check_keys(data, {"global", "local"}, "costmap")
    return _parse_global(data.get("global", {}), base), _parse_local(
        data.get("local", {})
    )


def load_world(path: str | Path) -> World:
    """Read a world file and check it.

    Args:
        path: path of a TOML file in the format of `examples/worlds/demo.toml`.

    Returns:
        The world.

    Raises:
        WorldError: if anything in the file is wrong. The message says which
            object (for example "obstacle 2" or "robot 'r1'") and which key.
        OSError: if the file cannot be read.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as error:
        raise WorldError(f"file: not valid TOML: {error}") from error

    version = require_key(data, "version", "file")
    if version != FORMAT_VERSION:
        raise WorldError(
            f"file: version {version} is not supported, expected {FORMAT_VERSION}"
        )
    check_keys(
        data,
        {"version", "world", "obstacle", "robot", "task", "sensors", "costmap"},
        "file",
    )

    world = table(require_key(data, "world", "file"), "world")
    check_keys(world, {"size", "seed", "dt"}, "world")
    size = numbers(require_key(world, "size", "world"), 2, "world", "size", True)
    bounds = (0.0, 0.0, size[0], size[1])
    seed = require_key(world, "seed", "world")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorldError("world: 'seed' must be an integer")
    dt = number(world, "dt", "world") if "dt" in world else DEFAULT_DT

    obstacles = tuple(
        _parse_obstacle(item, i, bounds)
        for i, item in enumerate(items(data, "obstacle"), start=1)
    )

    names: set[str] = set()
    robots = tuple(
        _parse_robot(item, i, bounds, names)
        for i, item in enumerate(items(data, "robot"), start=1)
    )
    if not robots:
        raise WorldError("file: needs at least one [[robot]]")

    by_name = {r.name: r for r in robots}
    tasks = tuple(
        _parse_task(item, i, bounds, by_name)
        for i, item in enumerate(items(data, "task"), start=1)
    )
    sensors = _parse_sensors(data.get("sensors", {}), "file")
    costmap, local_costmap = _parse_costmaps(
        data.get("costmap", {}), Path(path).resolve().parent
    )
    return World(
        size, seed, obstacles, robots, tasks, dt, sensors, costmap, local_costmap
    )
