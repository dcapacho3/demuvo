"""The dependency rule: each part of demuvo imports only the parts below it."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "demuvo"

# Each part and the parts it may import. A new part must be added here, on
# purpose. Nothing imports `run` or `view`, and planners and controllers never
# import `sim`: they see the world and the observation, not the simulation.
ALLOWED = {
    "verify": set(),
    "plugins": {"verify"},
    "models": {"verify"},
    "world": {"verify", "models"},
    "sim": {"models", "world"},
    "planners": {"plugins", "world"},
    "controllers": {"models", "plugins"},
    "evaluation": {"sim", "world"},
    "nav": {
        "controllers",
        "evaluation",
        "models",
        "planners",
        "plugins",
        "sim",
        "world",
    },
    "view": {"models", "sim", "world"},
    "run": {"evaluation", "nav", "sim", "world"},
}


def areas_imported(source: str, root_level: int) -> set[str]:
    """The parts of demuvo that a file imports.

    Args:
        source: the text of the file.
        root_level: how many dots a relative import needs to reach the demuvo
            folder: 1 for a file in it, 2 for a file one folder down.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == root_level:
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
            )
            for name in names:
                parts = name.split(".")
                if (
                    parts[0] == "demuvo"
                    and len(parts) > 1
                    and not getattr(node, "level", 0)
                ):
                    found.add(parts[1])
    return found


def imports_by_area() -> dict[str, set[str]]:
    """What each part of the source imports, from the files themselves."""
    found: dict[str, set[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if "tests" in rel.parts or len(rel.parts) == 1 and rel.stem.startswith("__"):
            continue
        area = rel.parts[0].removesuffix(".py")
        imported = areas_imported(path.read_text(encoding="utf-8"), len(rel.parts))
        found.setdefault(area, set()).update(imported - {area})
    return found


def test_the_reader_sees_relative_and_absolute_imports():
    """Without this the rule below could pass because it reads nothing."""
    source = (
        "from ..sim import A\nfrom ..world.shapes import B\n"
        "import demuvo.nav\nfrom . import c\n"
    )
    assert areas_imported(source, 2) == {"sim", "world", "nav"}
    assert areas_imported("from .verify import A\nfrom .. import x\n", 1) == {"verify"}
    assert areas_imported("from demuvo.view import play\n", 2) == {"view"}


def test_every_part_is_in_the_rule():
    """A new folder or module must be added to ALLOWED."""
    assert set(imports_by_area()) == set(ALLOWED)


@pytest.mark.parametrize("area", sorted(ALLOWED))
def test_a_part_imports_only_the_parts_it_may(area):
    """No part reaches above itself."""
    assert imports_by_area()[area] <= ALLOWED[area]
