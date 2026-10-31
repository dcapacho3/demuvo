"""Tests for the `demuvo` command."""

import runpy
import sys
from pathlib import Path

import pytest

from demuvo.run.cli import main

DEMO = str(Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml")


def test_check_reads_the_file_and_opens_nothing(capsys):
    """--check prints a summary and returns 0."""
    assert main([DEMO, "--check"]) == 0
    assert "3 obstacles, 1 robot" in capsys.readouterr().out


def test_a_wrong_file_names_the_problem(tmp_path, capsys):
    """A world error goes to stderr with the file name, and the code is 1."""
    path = tmp_path / "bad.toml"
    path.write_text("version = 2\n")
    assert main([str(path), "--check"]) == 1
    err = capsys.readouterr().err
    assert "bad.toml" in err
    assert "version" in err


def test_a_missing_file_is_reported(tmp_path, capsys):
    """A file that does not exist is an error message, not a traceback."""
    assert main([str(tmp_path / "nothing.toml")]) == 1
    assert "nothing.toml" in capsys.readouterr().err


def test_a_bad_step_length_is_reported(capsys):
    """--dt 0 is refused by the simulation and reported."""
    assert main([DEMO, "--dt", "0"]) == 1
    assert "dt" in capsys.readouterr().err


def test_the_window_opens_with_the_file_name(monkeypatch):
    """With no options the view gets the simulation and a caption."""
    view = pytest.importorskip("demuvo.view")
    calls = []
    monkeypatch.setattr("demuvo.view.play", lambda *a, **k: calls.append((a, k)))
    assert main([DEMO]) == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 2
    assert kwargs == {"caption": "demuvo - demo.toml", "trail": view.TRAIL}
    commands = args[1]
    assert list(commands(args[0])) == ["r1"]


def test_no_trail_turns_the_path_off(monkeypatch):
    """--no-trail asks the view for a trail of 0 points."""
    pytest.importorskip("pygame")
    calls = []
    monkeypatch.setattr("demuvo.view.play", lambda *a, **k: calls.append(k))
    assert main([DEMO, "--no-trail"]) == 0
    assert calls[0]["trail"] == 0


def test_without_pygame_the_message_says_how_to_install(monkeypatch, capsys):
    """A missing package is a hint on stderr that names it, not a traceback."""
    for name in [n for n in sys.modules if n.startswith("demuvo.view")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "pygame", None)
    assert main([DEMO]) == 1
    err = capsys.readouterr().err
    assert "'pygame'" in err
    assert "'view' extra" in err


def test_a_bad_option_ends_with_code_2():
    """Argparse refuses a step length that is not a number."""
    with pytest.raises(SystemExit) as info:
        main([DEMO, "--dt", "abc"])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2


def test_python_dash_m_runs_the_command(monkeypatch):
    """`python -m demuvo world.toml --check` exits with code 0."""
    monkeypatch.setattr(sys, "argv", ["demuvo", DEMO, "--check"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("demuvo", run_name="__main__")
    assert info.value.code == 0


def test_a_controller_that_cannot_load_is_reported(tmp_path, capsys):
    """--check stops on a controller the file names but demuvo cannot find."""
    text = Path(DEMO).read_text(encoding="utf-8").replace("PurePursuit", "Nothing")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    assert main([str(path), "--check"]) == 1
    assert "robot 'r1': cannot load controller" in capsys.readouterr().err


def test_a_period_below_dt_is_reported_before_the_window(capsys):
    """--dt 0.1 is above the demo's 0.05 s control period, so the run is refused."""
    assert main([DEMO, "--dt", "0.1", "--check"]) == 1
    assert "control_period" in capsys.readouterr().err
