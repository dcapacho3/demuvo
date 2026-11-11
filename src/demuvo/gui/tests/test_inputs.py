"""Tests for the number field, the drop-down, the menu popup and the list."""

import pytest

from demuvo.gui import inputs as module_inputs
from demuvo.gui import look
from demuvo.gui.inputs import (
    Combo,
    MenuItem,
    NameBox,
    NumberSpec,
    Popup,
    Row,
    Spin,
    TextField,
    TreeList,
)
from demuvo.gui.keys import KEY
from demuvo.gui.raster import Canvas
from demuvo.gui.widgets import Event

from .fakes import FakeHost, click, hosted, ink


def key(code, ctrl=False):
    """A key event."""
    return Event(key=code, ctrl=ctrl)


def field(value=1.5, **kwargs):
    """A hosted number field from 0 to 10 in steps of 0.5."""
    seen = []
    spec = NumberSpec(0, 10, 0.5, **kwargs)
    spin = Spin(value, spec, seen.append)
    return spin, hosted(spin), seen


def test_a_number_field_shows_its_value_with_decimals_and_suffix():
    """Two decimals by default, none for whole numbers."""
    assert Spin(1.5, NumberSpec(0, 10, 0.5, suffix=" m")).text() == "1.50 m"
    assert Spin(3.0, NumberSpec(0, 10, 1, 0, integer=True)).text() == "3"
    assert Spin(2.0, NumberSpec(0, 10, 1, decimals=3)).text() == "2.000"


def test_set_holds_the_value_between_the_ends_and_tells_the_handler_of_a_change():
    """Clamped, rounded, and silent when nothing changed or when asked."""
    spin, _, seen = field(1.5)
    spin.set(99)
    spin.set(10)
    spin.set(-4, notify=False)
    assert spin.value == 0 and seen == [10]
    spin.set(0.12345)
    assert spin.value == 0.123 and seen == [10, 0.123]
    whole = Spin(2, NumberSpec(0, 10, 1, integer=True))
    whole.set(4.6)
    assert whole.value == 5 and isinstance(whole.value, int)


def test_a_click_gives_the_field_the_focus_and_starts_editing():
    """The buffer holds the value as text; a disabled field ignores the click."""
    spin, host, _ = field(1.5)
    click(spin)
    assert host.focus is spin and spin.editing and spin.buffer == "1.50"
    off, off_host, _ = field()
    off.enabled = False
    click(off)
    assert off_host.focus is None and not off.editing
    whole = Spin(7, NumberSpec(0, 10, 1, integer=True))
    hosted(whole)
    whole.on_focus_change(True)
    assert whole.buffer == "7"


def test_typing_changes_the_buffer_and_enter_accepts_it():
    """Digits and a point go in, letters do not; Backspace removes; Enter commits."""
    spin, host, seen = field(1.5)
    spin.on_char("5")
    assert spin.buffer == ""
    click(spin)
    spin.buffer = ""
    for char in "7x.5":
        spin.on_char(char)
    assert spin.buffer == "7.5"
    spin.on_key(key(KEY["backspace"]))
    assert spin.buffer == "7."
    spin.on_char("5")
    assert spin.on_key(key(KEY["enter"])) is True
    assert spin.value == 7.5 and seen == [7.5] and not spin.editing
    assert host.focus is None


def test_a_keypad_enter_accepts_too_and_a_bad_number_keeps_the_old_value():
    """Text that is not a number is dropped quietly."""
    spin, _, seen = field(2.0)
    click(spin)
    spin.buffer = "1-2"
    spin.on_key(key(KEY["kp_enter"]))
    assert spin.value == 2.0 and not seen


def test_escape_cancels_the_edit_and_leaving_accepts_it():
    """Escape drops the typed text; a click elsewhere commits it."""
    spin, host, seen = field(2.0)
    click(spin)
    spin.buffer = "9"
    spin.on_key(key(KEY["escape"]))
    assert spin.value == 2.0 and not spin.editing and host.focus is None
    click(spin)
    spin.buffer = "9"
    host.set_focus(None)
    assert spin.value == 9.0 and seen == [9.0]


def test_the_arrow_keys_step_the_value_and_refresh_the_buffer():
    """Up adds a step, down takes one away, within the ends."""
    spin, _, seen = field(1.5)
    click(spin)
    spin.on_key(key(KEY["up"]))
    spin.on_key(key(KEY["up"]))
    spin.on_key(key(KEY["down"]))
    assert spin.value == 2.0 and spin.buffer == "2.00" and seen == [2.0, 2.5, 2.0]
    assert spin.editing


def test_copy_puts_the_buffer_or_the_value_on_the_clipboard_and_paste_takes_digits():
    """Ctrl+C copies the buffer, or the value when not editing; Ctrl+V keeps digits."""
    spin, host, _ = field(1.5)
    spin.on_key(key(KEY["c"], ctrl=True))
    assert host.clipboard == "1.5"
    click(spin)
    spin.buffer = "3.25"
    spin.on_key(key(KEY["c"], ctrl=True))
    assert host.clipboard == "3.25"
    host.clipboard = " 4,5 m "
    spin.on_key(key(KEY["v"], ctrl=True))
    assert spin.buffer == "45" and spin.editing
    host.clipboard = "none"
    spin.on_key(key(KEY["v"], ctrl=True))
    assert spin.buffer == "45"


def test_other_keys_are_not_used_and_a_disabled_field_uses_none():
    """The key goes on to the window; Backspace outside editing is not used either."""
    spin, _, _ = field()
    assert spin.on_key(key(KEY["a"])) is False
    assert spin.on_key(key(KEY["backspace"])) is False
    spin.enabled = False
    assert spin.on_key(key(KEY["up"])) is False


def test_the_wheel_over_the_field_steps_the_value_and_off_it_does_nothing():
    """Up adds, down subtracts; a disabled or unhovered field ignores it."""
    spin, _, seen = field(1.5)
    assert spin.on_scroll(Event(dy=1)) is False
    spin.hover = True
    assert spin.on_scroll(Event(dy=1)) is True and spin.on_scroll(Event(dy=-1)) is True
    assert seen == [2.0, 1.5]
    spin.enabled = False
    assert spin.on_scroll(Event(dy=1)) is False


def test_a_number_field_draws_idle_hover_editing_and_disabled():
    """Editing has a teal edge and a cursor; disabled is faint."""

    def paint(setup):
        spin, _, _ = field(1.5, suffix=" m")
        setup(spin)
        canvas = Canvas(140, 52)
        spin.draw(canvas)
        return canvas

    idle = paint(lambda s: None)
    hover = paint(lambda s: setattr(s, "hover", True))
    editing = paint(lambda s: (setattr(s, "editing", True), setattr(s, "buffer", "12")))
    off = paint(lambda s: setattr(s, "enabled", False))
    assert hover.px[10, 40, 0] < 1.0 and ink(hover) > ink(idle)
    assert editing.px[11, 70, 0] < 0.3 < idle.px[11, 70, 0]
    assert ink(editing, (0, 0, 140, 12)) > ink(idle, (0, 0, 140, 12))
    assert ink(off) < ink(idle)


def lines(*texts, **kwargs):
    """A popup over menu items made from texts."""
    items = [MenuItem(text, **kwargs) for text in texts]
    popup = Popup(items, 20, 20)
    return popup, hosted(popup, (20, 20, popup.width, popup.height))


def test_a_popup_is_as_wide_as_its_widest_line_and_as_tall_as_its_lines():
    """A shortcut and a check column add width; a separator is a short row."""
    plain = Popup([MenuItem("m" * 24), MenuItem("Delete")], 0, 0)
    ticked = Popup([MenuItem("m" * 24, checked=True), MenuItem("Delete")], 0, 0)
    keys = Popup([MenuItem("Add", shortcut="Ctrl+Shift+Alt+N")], 0, 0)
    assert ticked.has_check and not plain.has_check
    assert ticked.width == pytest.approx(plain.width + 20)
    assert keys.width > 160
    assert Popup([MenuItem("Add")], 0, 0).width == 160
    assert plain.height == 2 * 32 + 12
    mixed = Popup([MenuItem("a"), MenuItem(separator=True), MenuItem("b")], 0, 0)
    assert mixed.height == 32 * 2 + 10 + 12
    assert Popup([], 0, 0).width == 160


def test_row_at_finds_selectable_lines_only():
    """Separators and the space outside give -1."""
    popup = Popup([MenuItem("a"), MenuItem(separator=True), MenuItem("b")], 0, 0)
    assert popup.row_at(6 + 10) == 0
    assert popup.row_at(6 + 32 + 5) == -1
    assert popup.row_at(6 + 32 + 10 + 5) == 2
    assert popup.row_at(-5) == -1


def test_the_pointer_lights_a_line_and_the_host_redraws_only_on_a_change():
    """Moving within one line asks for one redraw."""
    popup, host = lines("a", "b")
    popup.on_move(Event(30, 20 + 6 + 10))
    popup.on_move(Event(31, 20 + 6 + 12))
    assert popup.hover_row == 0 and len(host.invalidated) == 1
    popup.on_move(Event(30, 20 + 6 + 32 + 10))
    assert popup.hover_row == 1 and len(host.invalidated) == 2


def test_a_click_closes_the_popup_and_runs_the_action_of_the_line():
    """Close first, then act; a disabled line, a separator and the margin do nothing."""
    done = []
    items = [
        MenuItem("go", action=lambda: done.append("go")),
        MenuItem("off", enabled=False, action=lambda: done.append("off")),
        MenuItem(separator=True),
        MenuItem("none"),
    ]
    popup = Popup(items, 20, 20)
    host = hosted(popup, (20, 20, popup.width, popup.height))
    popup.on_release(Event(30, 20 + 6 + 10))
    assert done == ["go"] and host.closed == 1
    popup.on_release(Event(30, 20 + 6 + 32 + 10))
    popup.on_release(Event(30, 20 + 6 + 64 + 5))
    popup.on_release(Event(30, 20 + 6 + 64 + 10 + 10))
    assert done == ["go"] and host.closed == 2
    popup.on_release(Event(30, 2))
    assert host.closed == 2


def test_a_popup_draws_a_card_a_tick_the_keys_and_a_highlight():
    """Ink for the card and text; a tick when checked; a tint under the pointer."""
    items = [
        MenuItem("Scene panel", checked=True, shortcut="Ctrl+1"),
        MenuItem(separator=True),
        MenuItem("Inspector", checked=False, enabled=False),
    ]
    popup = Popup(items, 10, 10)
    hosted(popup, (10, 10, popup.width, popup.height))
    canvas = Canvas(int(popup.width) + 30, int(popup.height) + 30, look.WINDOW)
    popup.draw(canvas)
    lit = Canvas(canvas.width, canvas.height, look.WINDOW)
    popup.hover_row = 0
    popup.draw(lit)
    assert ink(canvas) > 100
    assert lit.px[10 + 6 + 16, 18, 0] < canvas.px[10 + 6 + 16, 18, 0]
    assert canvas.px[10 + 6 + 32 + 4, 40, 0] < 0.9


def test_a_drop_down_opens_a_popup_of_its_choices_and_a_choice_changes_it():
    """Space opens it, the current one is ticked, choosing calls the handler once."""
    chosen = []
    combo = Combo(["1x", "2x", "4x"], "1x", chosen.append)
    host = hosted(combo, (10, 10, 100, 32))
    combo.activate()
    popup = host.popups[-1]
    assert [i.text for i in popup.items] == ["1x", "2x", "4x"]
    assert [i.checked for i in popup.items] == [True, False, False]
    assert popup.x == 10 and popup.y == 10 + 32 + 4 and popup.width >= 100
    popup.items[1].action()
    popup.items[1].action()
    assert combo.current == "2x" and chosen == ["2x"]
    combo.choose("4x", notify=False)
    assert combo.current == "4x" and chosen == ["2x"]


def test_a_click_opens_the_drop_down_and_a_disabled_one_stays_shut():
    """Press and release on it opens; off it, or disabled, does not."""
    combo = Combo(["a", "b"], "a")
    host = hosted(combo)
    click(combo)
    assert len(host.popups) == 1
    click(combo, outside=True)
    assert len(host.popups) == 1
    combo.enabled = False
    combo.activate()
    click(combo)
    assert len(host.popups) == 1
    Combo(["a"]).choose("b")


def test_a_drop_down_draws_the_choice_and_an_arrow_and_fades_when_disabled():
    """Ink for both; hover darkens the edge."""

    def paint(setup):
        combo = Combo(["1x", "2x"], "1x")
        hosted(combo, (4, 4, 100, 32))
        setup(combo)
        canvas = Canvas(110, 40)
        combo.draw(canvas)
        return canvas

    idle = paint(lambda c: None)
    assert ink(idle, (60, 0, 110, 40)) > 5
    assert paint(lambda c: setattr(c, "hover", True)).px[4, 50, 0] < idle.px[4, 50, 0]
    assert ink(paint(lambda c: setattr(c, "enabled", False))) < ink(idle)


def test_a_list_selects_an_entry_on_click_and_ignores_headings():
    """The key is what the handler gets; a heading has an empty key."""
    chosen = []
    tree = TreeList(chosen.append)
    hosted(tree, (0, 0, 200, 200))
    tree.set_rows(
        [Row("", "Robots (2)"), Row("r1", "r1", "#1fb5a6", 1), Row("r2", "r2")]
    )
    click(tree, 50, 32 + 10)
    click(tree, 50, 10)
    click(tree, 50, 500)
    click(tree, 50, 32 + 10, outside=True)
    assert chosen == ["r1"]
    plain = TreeList()
    hosted(plain, (0, 0, 200, 200))
    plain.set_rows([Row("a", "a")])
    click(plain, 50, 10)


def test_a_list_follows_the_pointer_and_forgets_it_on_leaving():
    """Hover is tracked by key; headings never light; one redraw per change."""
    tree = TreeList()
    host = hosted(tree, (0, 0, 200, 200))
    tree.set_rows([Row("", "Robots"), Row("r1", "r1")])
    count = len(host.invalidated)
    tree.on_move(Event(20, 10))
    assert tree.hover_key is None and len(host.invalidated) == count
    tree.on_move(Event(20, 40))
    tree.on_move(Event(25, 41))
    assert tree.hover_key == "r1" and len(host.invalidated) == count + 1
    tree.on_hover(False)
    assert tree.hover_key is None
    tree.on_move(Event(20, 400))
    assert tree.hover_key is None


def test_a_list_draws_selection_hover_dots_and_indent():
    """A selected row is tinted deeper than a hovered one; a dot adds colour."""
    tree = TreeList()
    hosted(tree, (0, 0, 200, 96))
    tree.set_rows([Row("", "Robots"), Row("r1", "r1", "#f5a623", 1), Row("r2", "r2")])
    tree.selected = "r1"
    tree.hover_key = "r2"
    canvas = Canvas(200, 96)
    tree.draw(canvas)
    assert canvas.px[32 + 4, 100, 0] < canvas.px[64 + 4, 100, 0] < 1.0
    assert canvas.px[10, 100, 0] == 1.0
    assert canvas.px[32 + 16, 10 + 14 + 9, 2] < 0.5


def test_a_click_selects_the_number_so_typing_replaces_it():
    """Typing after a click gives the new number, not the old one with more digits."""
    spin, _, seen = field(5.0)
    click(spin)
    assert spin.editing and spin.selected and spin.buffer == "5.00"
    for char in "7.5":
        spin.on_char(char)
    spin.on_key(key(KEY["enter"]))
    assert spin.value == 7.5 and seen == [7.5]


def test_the_arrow_keys_home_and_end_move_the_caret_and_delete_works_there():
    """Edit the middle of a number: caret keys, Backspace and Delete."""
    spin, _, _ = field(5.0)
    click(spin)
    spin.on_key(key(KEY["right"]))
    assert not spin.selected and spin.cursor == 4
    spin.on_key(key(KEY["left"]))
    spin.on_key(key(KEY["left"]))
    assert spin.cursor == 2
    spin.on_key(key(KEY["backspace"]))
    assert spin.buffer == "500" and spin.cursor == 1
    spin.on_key(key(KEY["delete"]))
    assert spin.buffer == "50"
    spin.on_key(key(KEY["home"]))
    spin.on_char("1")
    assert spin.buffer == "150" and spin.cursor == 1
    spin.on_key(key(KEY["end"]))
    assert spin.cursor == 3
    spin.on_key(key(KEY["backspace"]))
    spin.on_key(key(KEY["backspace"]))
    spin.on_key(key(KEY["backspace"]))
    spin.on_key(key(KEY["backspace"]))
    spin.on_key(key(KEY["delete"]))
    assert spin.buffer == "" and spin.cursor == 0


def test_left_and_right_over_a_selection_go_to_its_ends_and_ctrl_a_selects_all():
    """Left goes to the start, Right to the end, Ctrl+A selects it all again."""
    spin, _, _ = field(5.0)
    click(spin)
    spin.on_key(key(KEY["left"]))
    assert spin.cursor == 0 and not spin.selected
    spin.on_key(key(KEY["a"], ctrl=True))
    assert spin.selected
    spin.on_key(key(KEY["right"]))
    assert spin.cursor == 4
    spin.on_key(key(KEY["delete"]))
    assert spin.buffer == "5.00"


def test_a_second_click_puts_the_caret_between_the_digits_it_lands_on():
    """The click's x picks the nearest gap; no second click selects again."""
    spin, _, _ = field(5.0)
    click(spin)
    spin.on_press(Event(spin.x + 10, spin.y + 5))
    assert spin.cursor == 0 and not spin.selected
    spin.on_press(Event(spin.x + spin.width, spin.y + 5))
    assert spin.cursor == 4


def test_the_caret_and_the_selection_are_drawn():
    """The edited field shows a teal mark: a caret, or a band over a selection."""
    spin, _, _ = field(5.0)
    plain = Canvas(200, 40, look.WINDOW)
    spin.draw(plain)
    click(spin)
    band = Canvas(200, 40, look.WINDOW)
    spin.draw(band)
    spin.on_key(key(KEY["left"]))
    caret = Canvas(200, 40, look.WINDOW)
    spin.draw(caret)
    assert ink(band) != ink(caret) != ink(plain)
    assert (band.px != plain.px).any()


def test_typing_over_a_selection_or_deleting_it_leaves_only_what_is_typed():
    """Backspace on a selected number empties the field."""
    spin, _, _ = field(5.0)
    click(spin)
    spin.on_key(key(KEY["backspace"]))
    assert spin.buffer == "" and not spin.selected
    spin.on_char("3")
    assert spin.buffer == "3"


def text_box(text="world.toml"):
    """A hosted text box that records what it hands over."""
    done = []
    box = TextField(text, done.append, lambda: done.append("cancelled"))
    return box, hosted(box, (10, 10, 300, 32)), done


def test_a_text_box_is_selected_by_a_click_and_typing_replaces_it():
    """Any printable character goes in; Enter hands the text over."""
    box, _, done = text_box()
    click(box)
    assert box.editing and box.selected
    for char in "a/b c.toml":
        box.on_char(char)
    box.on_char("\n")
    assert box.buffer == "a/b c.toml"
    box.on_key(key(KEY["enter"]))
    assert done == ["a/b c.toml"]


def test_a_text_box_gives_up_on_escape_and_is_ignored_when_disabled():
    """Esc calls the cancel handler; a disabled box takes no key."""
    box, _, done = text_box()
    click(box)
    box.on_key(key(KEY["escape"]))
    assert done == ["cancelled"]
    box.editing = False
    box.enabled = False
    assert not box.on_key(key(KEY["enter"]))
    box.on_press(Event(box.x + 5, box.y + 5))
    assert not box.editing
    assert not text_box()[0].on_key(key(KEY["up"]))


def test_a_text_box_copies_pastes_and_moves_the_caret():
    """Ctrl+C copies, Ctrl+V pastes at the caret, a second click moves the caret."""
    box, host, _ = text_box("abc")
    click(box)
    box.on_key(key(KEY["c"], ctrl=True))
    assert host.clipboard == "abc"
    host.clipboard = "XY\n"
    box.on_key(key(KEY["right"]))
    box.on_key(key(KEY["v"], ctrl=True))
    assert box.buffer == "abcXY" and box.cursor == 5
    box.on_press(Event(box.x + 10, box.y + 5))
    assert box.cursor == 0 and box.editing
    box.on_focus_change(False)
    assert not box.editing
    canvas = Canvas(320, 50, look.WINDOW)
    box.draw(canvas)
    assert (canvas.px != look.WINDOW[:3]).any()


def test_two_quick_clicks_on_a_row_ask_for_a_rename_and_a_slow_pair_does_not(
    monkeypatch,
):
    """The handler gets the key and the row's rectangle; headings are ignored."""
    clock = [50.0]
    monkeypatch.setattr(module_inputs.time, "perf_counter", lambda: clock[0])
    asked = []
    tree = TreeList(on_select=lambda key: None, on_rename=lambda *a: asked.append(a))
    tree.set_rows([Row("", "Robots"), Row("r1", "r1"), Row("r2", "r2")])
    hosted(tree, (0, 0, 200, 96))
    click(tree, 20, 45)
    clock[0] += 0.2
    click(tree, 20, 45)
    assert asked == [("r1", tree.row_rect("r1"))]
    clock[0] += 0.2
    click(tree, 20, 45)
    assert len(asked) == 1
    clock[0] += 3.0
    click(tree, 20, 45)
    clock[0] += 3.0
    click(tree, 20, 45)
    clock[0] += 0.1
    click(tree, 20, 16)
    clock[0] += 0.1
    click(tree, 20, 16)
    assert len(asked) == 1
    click(tree, 20, 80)
    clock[0] += 0.1
    click(tree, 20, 45)
    assert len(asked) == 1


def test_a_name_box_hands_over_the_text_and_gives_the_focus_back_however_it_closes():
    """Enter submits; Esc and a click elsewhere only close it."""
    seen = []
    box = NameBox("r1", seen.append, (10, 10, 160, 32))
    host = FakeHost()
    box.adopt(host)
    assert host.focus is box.field and box.field.selected
    box.on_press(Event(box.field.x + 5, box.field.y + 5))
    box.on_move(Event(20, 20))
    box.on_release(Event(20, 20))
    canvas = Canvas(200, 60, look.WINDOW)
    box.draw(canvas)
    for char in "scout":
        box.field.on_char(char)
    box.field.on_key(key(KEY["enter"]))
    assert seen == ["scoutr1"]
    assert host.closed == 1
    box.on_close()
    assert host.focus is None
    box.field.on_key(key(KEY["escape"]))
    assert host.closed == 2 and len(seen) == 1
    Popup([], 0, 0).on_close()
