"""Tests for the `demuvo` command."""

import runpy
import sys
from pathlib import Path

import pytest

from demuvo.run import cli
from demuvo.run.cli import main

DEMO = str(Path(__file__).resolve().parents[4] / "examples" / "worlds" / "demo.toml")


@pytest.fixture(name="windows")
def stand_in_window(monkeypatch):
    """Replace the window with a function that notes the session and returns 0."""
    shown = []

    def show(session, seconds):
        shown.append((session.mode, session.world.robots[0].name, seconds))
        return 0

    real = cli.Session
    monkeypatch.setattr(cli, "show", show)
    monkeypatch.setattr(
        cli, "Session", lambda world, mode, remote, **_named: real(world, mode)
    )
    return shown


def test_check_reads_the_file_and_opens_nothing(capsys, windows):
    """--check prints a summary and returns 0."""
    assert main([DEMO, "--check"]) == 0
    assert "3 obstacles, 1 robot" in capsys.readouterr().out
    assert windows == []


def test_check_needs_a_file(capsys):
    """With no file there is nothing to check."""
    assert main(["--check"]) == 1
    assert capsys.readouterr().err == "demuvo: --check needs a world file\n"


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


def test_a_benchmark_needs_a_file(capsys):
    """--bench alone is refused in plain words."""
    assert main(["--bench"]) == 1
    assert capsys.readouterr().err == "demuvo: a benchmark needs a world file\n"


def test_a_bad_step_length_is_reported(capsys):
    """--dt 0 is refused by the simulation and reported."""
    assert main([DEMO, "--dt", "0"]) == 1
    assert "dt" in capsys.readouterr().err


def test_the_window_opens_on_the_sandbox_by_default(windows):
    """A file opens editable; the time limit is passed on."""
    assert main([DEMO, "--seconds", "3"]) == 0
    assert windows == [("sandbox", "r1", 3.0)]


def test_bench_opens_a_fixed_benchmark_and_no_file_a_blank_sandbox(windows):
    """The mode follows the option; a blank world has its starter robot."""
    assert main([DEMO, "--bench"]) == 0
    assert main([]) == 0
    assert windows == [("benchmark", "r1", None), ("sandbox", "r1", None)]


def test_the_simulation_gets_a_process_of_its_own_unless_told_not_to(monkeypatch):
    """`remote` is True by default, False with --one-process; the session closes."""
    seen, closed = [], []

    class Probe:
        """A session stand-in."""

        def __init__(self, _world, _mode, remote, **_named):
            seen.append(remote)

        def close(self):
            """Note the close."""
            closed.append(1)

    monkeypatch.setattr(cli, "Session", Probe)
    monkeypatch.setattr(cli, "show", lambda session, seconds: 0)
    assert main([DEMO]) == 0 and main([DEMO, "--one-process"]) == 0
    assert seen == [True, False] and closed == [1, 1]


def test_the_session_is_closed_even_if_the_window_fails(monkeypatch):
    """A crash in the window does not leave the worker running."""
    closed = []

    class Probe:
        """A session stand-in."""

        def __init__(self, _world, _mode, remote, **_named):
            """Nothing to set up."""
            assert remote

        def close(self):
            """Note the close."""
            closed.append(1)

    def crash(session, seconds):
        raise RuntimeError("window")

    monkeypatch.setattr(cli, "Session", Probe)
    monkeypatch.setattr(cli, "show", crash)
    with pytest.raises(RuntimeError):
        main([DEMO])
    assert closed == [1]


def test_dt_replaces_the_step_length_of_the_file(monkeypatch):
    """The session is made with the new step."""
    steps = []
    monkeypatch.setattr(
        cli, "show", lambda session, seconds: steps.append(session.run.sim.dt) or 0
    )
    assert main([DEMO, "--dt", "0.025"]) == 0
    assert steps == [0.025]


def test_a_window_that_cannot_open_gives_its_exit_code(monkeypatch):
    """The code of the window is the code of the command, and no result is printed."""
    monkeypatch.setattr(cli, "show", lambda session, seconds: 1)
    assert main([DEMO, "--result"]) == 1


def test_a_bad_option_ends_with_code_2():
    """Argparse refuses a step length that is not a number."""
    with pytest.raises(SystemExit) as info:
        main([DEMO, "--dt", "abc"])
    assert info.value.code == 2


def test_python_dash_m_runs_the_command(monkeypatch):
    """`python -m demuvo world.toml --check` exits with code 0."""
    monkeypatch.setattr(sys, "argv", ["demuvo", DEMO, "--check"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("demuvo", run_name="__main__")
    assert info.value.code == 0


def test_a_controller_that_cannot_load_is_reported(tmp_path, capsys):
    """The file names a controller demuvo cannot find."""
    text = Path(DEMO).read_text(encoding="utf-8").replace("PurePursuit", "Nothing")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    assert main([str(path), "--check"]) == 1
    assert "robot 'r1': cannot load controller" in capsys.readouterr().err


def test_a_period_below_dt_is_reported_before_the_window(capsys):
    """--dt 0.1 is above the demo's 0.05 s control period, so the run is refused."""
    assert main([DEMO, "--dt", "0.1", "--check"]) == 1
    assert "control_period" in capsys.readouterr().err


def test_the_result_is_printed_only_when_asked(monkeypatch, capsys):
    """With --result, each task and the distance driven go to stdout; else nothing."""

    def drive_to_the_goal(session, seconds):
        assert seconds is None
        session.run.until_done(60.0)
        return 0

    monkeypatch.setattr(cli, "show", drive_to_the_goal)
    assert main([DEMO]) == 0
    assert capsys.readouterr().out == ""
    assert main([DEMO, "--result"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("r1 task 1: reached at ")
    assert lines[1].startswith("r1 drove ")


def test_with_nothing_done_no_result_lines_are_printed(monkeypatch, capsys):
    """A blank world has no task, so there is nothing to say."""
    monkeypatch.setattr(cli, "show", lambda session, seconds: 0)
    assert main(["--result"]) == 0
    assert capsys.readouterr().out == ""


def test_seeds_print_a_table_and_open_no_window(capsys, windows):
    """--seeds runs each seed to the end and prints one row each."""
    assert main([DEMO, "--seeds", "2", "--jobs", "2", "--limit", "120"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("seed")
    assert lines[-1] == "2 of 2 runs done"
    assert windows == []


def test_seeds_need_a_file(capsys):
    """Without a world there is nothing to run."""
    assert main(["--seeds", "2"]) == 1
    assert capsys.readouterr().err == "demuvo: --seeds needs a world file\n"


def test_seeds_refuse_a_count_below_one(capsys):
    """A count, job number or limit of zero is a message, not a hang."""
    assert main([DEMO, "--seeds", "0"]) == 1
    assert "must be above 0" in capsys.readouterr().err


def test_install_desktop_writes_the_launcher_entry_and_the_icons(tmp_path, monkeypatch):
    """The entry names the command and the window class; three icon sizes follow."""
    monkeypatch.setattr(cli.desktop.Path, "home", classmethod(lambda cls: tmp_path))
    assert main(["--install-desktop"]) == 0
    entry = tmp_path / ".local" / "share" / "applications" / "demuvo.desktop"
    text = entry.read_text()
    assert "StartupWMClass=demuvo" in text and "Icon=demuvo" in text
    for size in (32, 64, 128):
        icon = (
            tmp_path / ".local/share/icons/hicolor" / f"{size}x{size}/apps/demuvo.png"
        )
        assert icon.read_bytes()[:4] == b"\x89PNG"


def test_install_desktop_uses_the_python_module_when_no_command_is_on_the_path(
    tmp_path, monkeypatch
):
    """Without a `demuvo` command the entry runs the module."""
    monkeypatch.setattr(cli.desktop.shutil, "which", lambda name: None)
    cli.desktop.install(tmp_path)
    entry = tmp_path / ".local/share/applications/demuvo.desktop"
    assert "-m demuvo" in entry.read_text()


def test_install_desktop_asks_the_desktop_tools_to_refresh_when_they_exist(
    tmp_path, monkeypatch
):
    """The icon cache and the launcher list are rebuilt; a missing tool is skipped."""
    ran = []
    monkeypatch.setattr(
        cli.desktop.shutil,
        "which",
        lambda name: name if name != "update-desktop-database" else None,
    )
    monkeypatch.setattr(
        cli.desktop.subprocess, "run", lambda command, **_k: ran.append(command)
    )
    cli.desktop.install(tmp_path, exe="demuvo")
    assert [c[0] for c in ran] == ["gtk-update-icon-cache"]
    assert ran[0][-1].endswith("hicolor")
