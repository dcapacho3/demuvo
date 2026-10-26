"""Tests for the `demuvo` command."""

import runpy
import sys
from pathlib import Path

import pytest

from demuvo.cli import main

DEMO = str(Path(__file__).resolve().parents[1] / "examples" / "worlds" / "demo.toml")


def test_check_reads_the_file_and_opens_nothing(capsys):
    """--check prints a summary and returns 0."""
    assert main([DEMO, "--check"]) == 0
    assert "3 obstacles, 1 robots" in capsys.readouterr().out


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
    pytest.importorskip("pygame")
    calls = []
    monkeypatch.setattr("demuvo.view.play", lambda *a, **k: calls.append((a, k)))
    assert main([DEMO]) == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 2
    assert kwargs == {"caption": "demuvo - demo.toml"}


def test_without_pygame_the_message_says_how_to_install(monkeypatch, capsys):
    """A missing view module is a hint, not a traceback."""
    monkeypatch.setitem(sys.modules, "demuvo.view", None)
    assert main([DEMO]) == 1
    assert "demuvo[view]" in capsys.readouterr().out


def test_python_dash_m_runs_the_command(monkeypatch):
    """`python -m demuvo world.toml --check` exits with code 0."""
    monkeypatch.setattr(sys, "argv", ["demuvo", DEMO, "--check"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("demuvo", run_name="__main__")
    assert info.value.code == 0
