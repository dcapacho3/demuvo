"""Put demuvo in the desktop's launcher, with its icon.

A desktop shows an icon for a window by matching it to a launcher entry, so a program
run from a terminal has none until one is installed. This writes the entry and the
icons into the user's own folders, and nothing outside them. A package installed
for the whole system does the same job.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "gui" / "assets"
SIZES = (32, 64, 128)
ENTRY = """[Desktop Entry]
Type=Application
Name=demuvo
Comment=A small deterministic 2D world for robots
Exec={exe} %f
Icon=demuvo
Terminal=false
Categories=Development;Science;Robotics;
StartupWMClass=demuvo
"""


def install(home: Path | None = None, exe: str | None = None) -> list[Path]:
    """Write the launcher entry and the icons under a home folder.

    Args:
        home: the folder to write under; the user's own by default.
        exe: the command the entry runs; the `demuvo` found on the path by default.

    Returns:
        The files written.
    """
    home = home or Path.home()
    exe = exe or shutil.which("demuvo") or f"{sys.executable} -m demuvo"
    data = home / ".local" / "share"
    written = []
    for size in SIZES:
        target = data / "icons" / "hicolor" / f"{size}x{size}" / "apps" / "demuvo.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ASSETS / f"icon_{size}.png", target)
        written.append(target)
    entry = data / "applications" / "demuvo.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(ENTRY.format(exe=exe), encoding="utf-8")
    written.append(entry)
    _refresh(data)
    return written


def _refresh(data: Path) -> None:
    """Ask the desktop's own tools to re-read what was written, where they exist.

    Without this the desktop may keep an older list of icons and show a gear.
    """
    for command in (
        ["gtk-update-icon-cache", "-f", "-t", str(data / "icons" / "hicolor")],
        ["update-desktop-database", str(data / "applications")],
    ):
        if shutil.which(command[0]):
            subprocess.run(command, check=False, capture_output=True)
