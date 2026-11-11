"""The inspector's forms for the two costmaps and the sensors every robot starts with.

Each field applies its edit through the session and says why when it is refused.
"""

from __future__ import annotations

from .. import edit
from .formbase import FormBase
from .inputs import NumberSpec
from .widgets import Section

CELL = NumberSpec(0.01, 5.0, 0.01, 3, " m")
DISTANCE = NumberSpec(0.01, 1000.0, 0.05, 2, " m")
SECONDS = NumberSpec(0.01, 100.0, 0.05, 2, " s")
FACTOR = NumberSpec(0.01, 1000.0, 0.5, 2)
# The layers of each costmap, in the order they are applied.
LAYERS = {"global": ("static", "inflation"), "local": ("obstacle", "inflation")}
# Settings each costmap has as a number, by name, label and how to show it.
NUMBERS = {
    "global": (
        ("resolution", "cell size", CELL),
        ("inflation_radius", "inflation radius", DISTANCE),
        ("cost_scaling_factor", "cost scaling", FACTOR),
    ),
    "local": (
        ("window", "window", DISTANCE),
        ("resolution", "cell size", CELL),
        ("period", "update period", SECONDS),
        ("obstacle_max_range", "obstacle range", DISTANCE),
        ("raytrace_max_range", "clearing range", DISTANCE),
        ("inflation_radius", "inflation radius", DISTANCE),
        ("cost_scaling_factor", "cost scaling", FACTOR),
    ),
}


class CostmapForm(FormBase):
    """Builds the global and local costmap sections into a column."""

    def costmap(self, which: str) -> None:
        """The switch, numbers and layers of the "global" or "local" costmap."""
        world = self.session.world
        spec = world.costmap if which == "global" else world.local_costmap
        prefix = "cm" if which == "global" else "lc"
        self.column.add(Section(f"{which} costmap"))
        self.switch(
            f"{prefix}.enabled",
            "enabled",
            spec.enabled,
            lambda on: self._set(which, {"enabled": on}),
        )
        for key, label, shape in NUMBERS[which]:
            self.number(
                f"{prefix}.{key}",
                label,
                getattr(spec, key),
                shape,
                lambda k=key: self._set(which, {k: self.fields[f"{prefix}.{k}"].value}),
            )
        for layer in LAYERS[which]:
            self.switch(
                f"{prefix}.{layer}",
                f"{layer} layer",
                layer in spec.plugins,
                lambda _on: self._layers(which, prefix),
            )
        if which == "global":
            self.text(
                "cm.source",
                "map file",
                spec.source,
                lambda t: self._set(which, {"source": t}),
            )
            self.choice(
                "cm.unknown",
                "unknown cells",
                ["free", "blocked"],
                spec.unknown,
                lambda t: self._set(which, {"unknown": t}),
            )

    def _layers(self, which: str, prefix: str) -> None:
        """A layer switch changed; the layers on are read back in order."""
        plugins = [
            layer for layer in LAYERS[which] if self.fields[f"{prefix}.{layer}"].checked
        ]
        self._set(which, {"plugins": plugins})

    def _set(self, which: str, values: dict) -> None:
        """Apply costmap settings to the world."""
        self.done(
            self.session.edits.change_world(
                lambda w: edit.set_costmap(w, which, values)
            )
        )
