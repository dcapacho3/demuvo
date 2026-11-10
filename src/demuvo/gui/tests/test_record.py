"""Tests for the headless recorder."""

import runpy
import sys

import numpy as np
import pytest

from demuvo.gui import record as module
from demuvo.gui.imgio import read_png

from .fakes import EXAMPLES
from .test_imgio import read_gif

DEMO, LIDAR = EXAMPLES / "demo.toml", EXAMPLES / "lidar.toml"


def test_a_world_is_recorded_as_a_gif_of_the_asked_size_and_length(tmp_path):
    """0.6 s at 10 frames a second is 6 frames of 320 by 198 pixels."""
    out = tmp_path / "run.gif"
    assert (
        module.record(DEMO, str(out), seconds=0.6, fps=10, framing=module.Framing(320))
        == 6
    )
    (width, height), loop, delays, frames = read_gif(out.read_bytes())
    assert (width, height) == (320, 198) and len(frames) == 6
    assert loop == 0 and delays == [10] * 6


def test_the_frames_change_as_the_robot_moves(tmp_path):
    """The first and last frames differ."""
    out = tmp_path / "run.gif"
    module.record(DEMO, str(out), seconds=3.0, fps=5, framing=module.Framing(320))
    frames = read_gif(out.read_bytes())[3]
    assert not np.array_equal(frames[0], frames[-1])


def test_an_output_folder_gets_one_png_for_each_frame(tmp_path):
    """The names count up from zero."""
    folder = f"{tmp_path}/frames/"
    assert (
        module.record(DEMO, folder, seconds=0.3, fps=10, framing=module.Framing(200))
        == 3
    )
    names = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert names == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert read_png(tmp_path / "frames" / names[0]).shape == (124, 200, 3)


def test_layers_can_be_switched_on_and_the_panels_added(tmp_path):
    """Rays add ink to the lidar world; panels make the frame wider and taller."""
    plain, rays, wide = (f"{tmp_path}/{n}/" for n in ("plain", "rays", "wide"))
    module.record(LIDAR, plain, seconds=1.0, fps=1, framing=module.Framing(320))
    module.record(
        LIDAR, rays, seconds=1.0, fps=1, framing=module.Framing(320, ("rays", "local"))
    )
    module.record(
        LIDAR, wide, seconds=1.0, fps=1, framing=module.Framing(320, panels=True)
    )
    a = read_png(f"{plain}frame_00000.png")
    b = read_png(f"{rays}frame_00000.png")
    c = read_png(f"{wide}frame_00000.png")
    assert not np.array_equal(a, b) and a.shape == b.shape == (198, 320, 3)
    assert c.shape == (198 + 112, 320 + 560, 3)


def test_the_command_records_and_says_how_many_frames(tmp_path, capsys):
    """Arguments become the call; the count and the place are printed."""
    out = tmp_path / "a.gif"
    code = module.main(
        [
            str(DEMO),
            str(out),
            "--seconds",
            "0.4",
            "--fps",
            "5",
            "--width",
            "200",
            "--layers",
            "rays,",
        ]
    )
    assert code == 0 and out.exists()
    assert capsys.readouterr().out == f"2 frames to {out}\n"
    with pytest.raises(SystemExit):
        module.main([])


def test_python_dash_m_runs_the_recorder(tmp_path, monkeypatch, capsys):
    """`python -m demuvo.gui.record` records and prints the same line."""
    out = tmp_path / "m.gif"
    monkeypatch.setattr(
        "sys.argv", ["record", str(DEMO), str(out), "--seconds", "0.2", "--fps", "5"]
    )
    monkeypatch.delitem(sys.modules, "demuvo.gui.record")
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("demuvo.gui.record", run_name="__main__")
    assert stopped.value.code == 0 and out.exists()
    assert capsys.readouterr().out.startswith("1 frames to ")
