"""Tests for the file dialog: browsing, choosing, naming and replacing."""

from pathlib import Path

import pytest

from demuvo.gui import filedialog as module
from demuvo.gui import look
from demuvo.gui.filedialog import Browse, FileDialog, list_folder
from demuvo.gui.keys import KEY
from demuvo.gui.raster import Canvas
from demuvo.gui.widgets import Event

from .fakes import FakeHost

PLACE = (20, 20, 640, 440)


@pytest.fixture(name="tree")
def make_tree(tmp_path):
    """A folder with two folders, two world files, a text file and a hidden file."""
    (tmp_path / "beta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "b.toml").write_text("x")
    (tmp_path / "A.toml").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".secret.toml").write_text("x")
    return tmp_path


def make(folder, name="", *, save=False):
    """A dialog on a fake host, and the paths it hands over."""
    chosen = []
    dialog = FileDialog(Browse("Open", chosen.append, name, save), folder, PLACE)
    host = FakeHost()
    dialog.adopt(host)
    return dialog, host, chosen


def row_point(dialog, index):
    """A pixel inside the row at a place of the visible list."""
    x, y, _, _ = dialog._list_rect()  # pylint: disable=protected-access
    return x + 50, y + module.ROW * (index - dialog.view.scroll) + 5


def key(code):
    """A key event."""
    return Event(key=code)


def test_a_folder_lists_its_folders_then_its_world_files_without_hidden_names(tree):
    """Sorted without regard to case; other files and dot names are left out."""
    assert list_folder(tree, ".toml") == (["Alpha", "beta"], ["A.toml", "b.toml"])


def test_the_dialog_starts_in_a_folder_and_takes_the_focus_for_the_name(tree):
    """The path box shows the folder; typing goes to the name box."""
    dialog, host, _ = make(tree)
    assert dialog.path_field.buffer == str(tree)
    assert host.focus is dialog.name_field and dialog.name_field.editing
    canvas = Canvas(700, 500, look.WINDOW)
    dialog.draw(canvas)
    assert (canvas.px != look.WINDOW[:3]).any()


def test_a_click_selects_a_row_and_a_file_click_fills_the_name(tree, monkeypatch):
    """A folder is lit; a file's name goes to the name box."""
    monkeypatch.setattr(module.time, "perf_counter", lambda: 100.0)
    dialog, _, _ = make(tree)
    dialog.on_press(Event(*row_point(dialog, 0)))
    assert dialog.view.selected == 0 and dialog.name_field.buffer == ""
    dialog.on_press(Event(*row_point(dialog, 2)))
    assert dialog.view.selected == 2 and dialog.name_field.buffer == "A.toml"


def test_a_double_click_enters_a_folder_and_takes_a_file(tree, monkeypatch):
    """Two quick clicks on a folder go in; on a file they choose it."""
    clock = [100.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    dialog, host, chosen = make(tree)
    (tree / "Alpha" / "inner.toml").write_text("x")
    dialog.on_press(Event(*row_point(dialog, 0)))
    clock[0] += 0.1
    dialog.on_press(Event(*row_point(dialog, 0)))
    assert dialog.folder == tree / "Alpha" and dialog.view.files == ["inner.toml"]
    clock[0] += 5.0
    dialog.on_press(Event(*row_point(dialog, 0)))
    clock[0] += 0.1
    dialog.on_press(Event(*row_point(dialog, 0)))
    assert chosen == [str(tree / "Alpha" / "inner.toml")]
    assert host.closed == 1 and host.focus is None


def test_up_home_and_the_path_box_move_between_folders(tree, monkeypatch):
    """Up goes to the parent, Home to the home folder, a typed path to that folder."""
    dialog, _, _ = make(tree / "Alpha")
    dialog.buttons["up"].activate()
    assert dialog.folder == tree
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: tree / "beta"))
    dialog.buttons["home"].activate()
    assert dialog.folder == tree / "beta"
    dialog.path_field.buffer = str(tree)
    dialog.path_field.on_submit(str(tree))
    assert dialog.folder == tree and dialog.path_field.buffer == str(tree)
    dialog.path_field.on_submit(str(tree / "nowhere"))
    assert dialog.note == "That folder does not exist." and dialog.folder == tree


def test_an_unreadable_folder_is_reported_and_the_old_one_stays(tree, monkeypatch):
    """A permission error becomes a note, not a crash."""
    dialog, _, _ = make(tree)

    def refuse(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(module.Path, "iterdir", refuse)
    dialog._read(tree / "beta")  # pylint: disable=protected-access
    assert "Permission denied" in dialog.note and dialog.folder == tree


def test_open_takes_an_existing_file_by_name_or_path_and_refuses_a_missing_one(tree):
    """A bare name is looked for in the folder; a path is taken as it is."""
    dialog, host, chosen = make(tree)
    dialog.name_field.on_submit("A.toml")
    assert chosen == [str(tree / "A.toml")] and host.closed == 1
    dialog, host, chosen = make(tree)
    dialog.name_field.on_submit(str(tree / "b.toml"))
    assert chosen == [str(tree / "b.toml")]
    dialog, host, chosen = make(tree)
    dialog.name_field.on_submit("nothing.toml")
    assert dialog.note == "There is no such file." and not chosen and host.closed == 0
    dialog.name_field.on_submit("beta")
    assert dialog.folder == tree / "beta" and not chosen


def test_save_adds_the_ending_and_asks_before_replacing_a_file(tree):
    """A new name is saved; an old one needs a second Save."""
    dialog, host, chosen = make(tree, "fresh", save=True)
    dialog.buttons["ok"].activate()
    assert chosen == [str(tree / "fresh.toml")] and host.closed == 1
    dialog, host, chosen = make(tree, "A.toml", save=True)
    dialog.buttons["ok"].activate()
    assert "exists" in dialog.note and not chosen
    dialog.buttons["ok"].activate()
    assert chosen == [str(tree / "A.toml")]


def test_the_cancel_button_and_escape_close_without_choosing(tree):
    """Nothing is handed over."""
    dialog, host, chosen = make(tree)
    dialog.buttons["cancel"].activate()
    assert host.closed == 1 and not chosen
    dialog, host, chosen = make(tree)
    dialog.name_field.on_key(key(KEY["escape"]))
    assert host.closed == 1 and not chosen


def test_clicks_reach_the_boxes_and_buttons_and_the_pointer_lights_a_row(tree):
    """A press on a box is the box's; a move over a row lights it and a button."""
    dialog, _, _ = make(tree)
    box = dialog.path_field
    dialog.on_press(Event(box.x + 5, box.y + 5))
    assert box.editing
    dialog.on_release(Event(box.x + 5, box.y + 5))
    ok = dialog.buttons["ok"]
    dialog.on_move(Event(ok.x + 5, ok.y + 5))
    assert ok.hover
    dialog.on_move(Event(*row_point(dialog, 1)))
    assert dialog.view.hover == 1
    dialog.on_move(Event(5, 5))
    assert dialog.view.hover is None and not ok.hover
    dialog.on_press(Event(5, 5))
    dialog.on_release(Event(5, 5))
    assert dialog.view.selected is None


def test_the_wheel_scrolls_a_long_list_and_stops_at_the_ends(tmp_path):
    """Thirty files do not fit; the wheel moves the first row shown."""
    for k in range(30):
        (tmp_path / f"w{k:02d}.toml").write_text("x")
    dialog, _, _ = make(tmp_path)
    assert dialog.on_scroll(Event(dy=-1))
    assert dialog.view.scroll == 1
    for _ in range(40):
        dialog.on_scroll(Event(dy=-1))
    assert dialog.view.scroll == 30 - dialog._visible_rows()  # pylint: disable=protected-access
    for _ in range(40):
        dialog.on_scroll(Event(dy=1))
    assert dialog.view.scroll == 0
    canvas = Canvas(700, 500, look.WINDOW)
    dialog.draw(canvas)


def test_an_empty_folder_says_so(tmp_path):
    """The list is not left blank."""
    dialog, _, _ = make(tmp_path)
    empty = Canvas(700, 500, look.WINDOW)
    dialog.draw(empty)
    (tmp_path / "w.toml").write_text("x")
    dialog._read(tmp_path)  # pylint: disable=protected-access
    full = Canvas(700, 500, look.WINDOW)
    dialog.draw(full)
    assert (empty.px != full.px).any()
    assert Path(dialog.folder) == tmp_path


def test_a_selected_row_and_a_row_under_the_pointer_are_lit(tree, monkeypatch):
    """Selection and hover each change the picture of the list."""
    monkeypatch.setattr(module.time, "perf_counter", lambda: 100.0)
    dialog, _, _ = make(tree)
    plain = Canvas(700, 500, look.WINDOW)
    dialog.draw(plain)
    dialog.on_move(Event(*row_point(dialog, 1)))
    hovered = Canvas(700, 500, look.WINDOW)
    dialog.draw(hovered)
    dialog.on_press(Event(*row_point(dialog, 0)))
    selected = Canvas(700, 500, look.WINDOW)
    dialog.draw(selected)
    assert (plain.px != hovered.px).any() and (hovered.px != selected.px).any()
