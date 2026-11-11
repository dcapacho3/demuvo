"""Tests for the shell: layout, panels, events, focus, menus, tooltips and rendering."""

import time

import numpy as np
import pytest

from demuvo.gui import shell as module
from demuvo.gui.inputs import MenuItem, NumberSpec, Popup, Spin
from demuvo.gui.keys import CTRL, KEY, SHIFT
from demuvo.gui.shell import Animator, Damage, Shell
from demuvo.gui.widgets import Checkbox, Event, IconButton, Widget


class Probe(Widget):
    """A pane that records the events it gets and can use the wheel."""

    def __init__(self):
        """Start with no events."""
        super().__init__()
        self.seen, self.uses_wheel = [], False

    def on_hover(self, inside):
        """Note the pointer coming and going."""
        self.seen.append(("hover", inside))
        super().on_hover(inside)

    def on_press(self, event):
        """Note a press."""
        self.seen.append(("press", event.x, event.y))
        super().on_press(event)

    def on_release(self, event):
        """Note a release."""
        self.seen.append(("release", event.x, event.y))
        super().on_release(event)

    def on_move(self, event):
        """Note a move."""
        self.seen.append(("move", event.x, event.y))

    def on_scroll(self, event):
        """Use the wheel if asked."""
        self.seen.append(("scroll", event.dy))
        return self.uses_wheel


class ToyProgram:
    """A program with a probe pane, two tool buttons, a checkbox and a number field."""

    menu_titles = ("File", "Edit", "View")

    def __init__(self):
        """Start with nothing recorded."""
        self.commands, self.busy_now, self.ticked = [], False, []
        self.groups, self.pane = {}, Probe()
        self.one = IconButton("select", checkable=True, group="tool")
        self.two = IconButton("goal", tooltip="Goal", checkable=True, group="tool")
        self.box = Checkbox("Rays")
        self.field = Spin(1.0, NumberSpec(0, 10, 1))

    def build(self, window):
        """Fill the panels and set the pane."""
        window.layout.pane = self.pane
        chrome = window.layout.chrome
        self.groups["tool"] = [self.one, self.two]
        chrome.toolbar.add(self.one, self.two)
        chrome.left.add(self.box, self.field)
        self.box.set_rect(10, 90, 100, 28)
        self.field.set_rect(10, 130, 100, 30)

    def menu_items(self, title):
        """A line that says which menu it is in."""
        return [MenuItem(f"{title} item", action=lambda: self.commands.append(title))]

    @staticmethod
    def can_quit():
        """The toy never has unsaved work."""
        return True

    def command(self, key, ctrl):
        """Note a key nothing else used."""
        self.commands.append((key, ctrl))
        return True

    def layout_toolbar(self, width):
        """Two buttons in a row."""
        assert width > 0
        self.one.set_rect(8, module.MENU_H + 6, 32, 32)
        self.two.set_rect(42, module.MENU_H + 6, 32, 32)

    def busy(self):
        """Busy when the test says so."""
        return self.busy_now

    def after_tick(self, now):
        """Note a tick."""
        self.ticked.append(now)

    def world_at(self, x, y):
        """Pretend a pixel of the pane is a world point, a hundredth of a metre each."""
        pane = self.pane
        inside = (
            pane.x <= x < pane.x + pane.width and pane.y <= y < pane.y + pane.height
        )
        return (x / 100, y / 100) if inside else None


@pytest.fixture(name="toy")
def make_toy():
    """A 1000 by 700 shell over the toy program."""
    return Shell(ToyProgram(), 1000, 700)


def centre(widget):
    """The centre of a widget."""
    return widget.x + widget.width / 2, widget.y + widget.height / 2


def test_the_window_is_laid_out_in_bars_panels_and_a_pane(toy):
    """Menu, tool bar and status take strips; the panels and pane fill the rest."""
    chrome, pane = toy.layout.chrome, toy.layout.pane
    top = module.MENU_H + module.TOOL_H
    body = 700 - top - module.STATUS_H
    assert (chrome.menubar.height, chrome.toolbar.y, chrome.toolbar.height) == (
        34,
        34,
        44,
    )
    assert (chrome.status.y, chrome.status.height) == (700 - 26, 26)
    assert (chrome.left.width, chrome.right.width) == (248, 312)
    assert (chrome.left.y, chrome.left.height) == (top, body)
    assert (pane.x, pane.width) == (248, 1000 - 248 - 312)
    assert chrome.right.x == 1000 - 312
    assert (toy.screen.width, toy.screen.height) == (1000, 700)


def test_in_a_narrow_window_the_panels_give_way_to_keep_the_scene_wide_enough():
    """The scene keeps its minimum while the panels have room to give."""
    narrow = Shell(ToyProgram(), 500, 700)
    assert narrow.layout.pane.width == pytest.approx(module.SCENE_MIN)
    chrome = narrow.layout.chrome
    assert chrome.left.width + chrome.right.width == pytest.approx(500 - 240)
    tiny = Shell(ToyProgram(), 200, 700).layout.chrome
    assert tiny.left.visible is False and tiny.right.visible is False


def test_a_render_with_a_new_size_remakes_the_frame_and_lays_out_again(toy):
    """The frame has the new size and the status bar sits at the new foot."""
    assert toy.render(800, 600).shape == (600, 800, 3)
    assert toy.layout.chrome.status.y == 600 - 26
    assert toy.render().shape == (600, 800, 3)


def test_a_panel_slides_shut_and_open_through_an_animation(toy):
    """Ticking the animation to its end leaves the dock closed, then open."""
    toy.layout.slide("left", False)
    assert toy.layout.docks["left"].open is False
    toy.animator.tick(time.perf_counter() + 1.0)
    assert toy.layout.docks["left"].width == 0.0 and not toy.layout.chrome.left.visible
    assert toy.layout.pane.x == 0
    toy.layout.slide("left", True)
    toy.animator.tick(time.perf_counter() + 1.0)
    assert toy.layout.docks["left"].width == 248 and toy.layout.chrome.left.visible


def test_an_animation_eases_out_and_a_new_one_under_the_same_key_replaces_it():
    """Half way in time is over half way in value; a second start replaces the first."""
    animator, values = Animator(), []
    animator.start("a", 0.0, 10.0, 100.0, values.append)
    assert animator.busy()
    animator.tick(time.perf_counter() + 0.05)
    assert 5.0 < values[-1] < 10.0
    animator.start("a", values[-1], 0.0, 100.0, values.append)
    animator.tick(time.perf_counter() + 1.0)
    assert values[-1] == 0.0 and not animator.busy()


def test_far_apart_damage_boxes_stay_apart_and_near_ones_join():
    """A change in the status bar does not redraw the scene between."""
    damage = Damage(200, 100)
    damage.take()
    damage.add((0, 0, 200, 10))
    damage.add((0, 90, 200, 100))
    assert damage.take() == [(0, 0, 200, 10), (0, 90, 200, 100)]
    damage.add((0, 0, 50, 50))
    damage.add((40, 40, 90, 90))
    assert damage.take() == [(0, 0, 90, 90)]
    damage.add((0, 0, 10, 10))
    damage.add((12, 0, 22, 10))
    damage.add((100, 0, 110, 10))
    damage.add((20, 0, 100, 10))
    assert damage.take() == [(0, 0, 110, 10)]


def test_the_damage_box_grows_to_hold_each_change_and_is_taken_clipped():
    """Boxes far apart stay two; the union holds both; taking clips and clears."""
    damage = Damage(100, 80)
    damage.take()
    assert not damage.dirty
    damage.add((10, 10, 20, 20))
    damage.add((50, 40, 120, 90))
    assert damage.rect == (10, 10, 120, 90)
    assert damage.take() == [(10, 10, 20, 20), (50, 40, 100, 80)]
    assert damage.rect is None and not damage.dirty
    damage.widget(Widget())
    assert damage.rect == (-8, -8, 8, 8)
    assert damage.take() == [(0, 0, 8, 8)]
    assert damage.take() == [(0, 0, 100, 80)]


def test_the_first_render_draws_everything_and_a_later_one_only_what_changed(toy):
    """A change outside the damage box does not reach the frame."""
    first = toy.render().copy()
    assert toy.dirty is False and first.max() > 0
    toy.screen.surface.canvas.px[300:310, 400:410] = (1.0, 0.0, 0.0)
    toy.screen.damage.add((0, 0, 50, 50))
    again = toy.render()
    assert np.array_equal(again[300:310, 400:410], first[300:310, 400:410])
    assert again is toy.screen.surface.frame


def test_invalidate_marks_a_widget_or_everything(toy):
    """A widget's own box, or the whole window."""
    toy.render()
    one = toy.program.one
    toy.invalidate(one)
    assert toy.dirty_rect[0] == int(one.x) - module.DIRTY_PAD
    toy.render()
    toy.invalidate()
    assert toy.dirty_rect == (0, 0, 1000, 700)
    toy.render()
    toy.invalidate_all()
    assert toy.dirty


def test_the_pointer_enters_and_leaves_widgets_and_a_press_is_captured_until_release(
    toy,
):
    """The probe sees hover on, a move, a press, then release even off the pane."""
    pane = toy.program.pane
    x, y = centre(pane)
    toy.mouse_move(x, y)
    assert toy.input.pointer.hovered is pane
    toy.mouse_press(Event(x, y))
    toy.mouse_move(5, 5)
    toy.mouse_release(Event(5, 5))
    kinds = [item[0] for item in pane.seen]
    assert (
        kinds[:2] == ["hover", "move"] and "press" in kinds and kinds[-1] == "release"
    )
    assert ("move", 5, 5) in pane.seen
    toy.mouse_move(1, 1)
    assert pane.seen.count(("hover", False)) == 1


def test_only_the_left_button_presses(toy):
    """The right button is ignored."""
    pane = toy.program.pane
    toy.mouse_press(Event(*centre(pane), button=1))
    assert not any(item[0] == "press" for item in pane.seen)


def test_the_status_bar_shows_the_pointers_world_place_while_over_the_pane(toy):
    """Over the pane the place is set; off it, it is cleared."""
    x, y = centre(toy.program.pane)
    toy.mouse_move(x, y)
    assert toy.layout.chrome.status.coords == (x / 100, y / 100)
    toy.mouse_move(5, 5)
    assert toy.layout.chrome.status.coords is None


def test_the_wheel_goes_to_the_widget_under_the_pointer_and_then_up_the_tree(toy):
    """A pane that does not use it passes it up; one that does stops there."""
    pane = toy.program.pane
    toy.mouse_move(*centre(pane))
    toy.mouse_scroll(-1.0)
    assert ("scroll", -1.0) in pane.seen
    pane.uses_wheel = True
    toy.mouse_scroll(1.0)
    assert pane.seen[-1] == ("scroll", 1.0)
    toy.mouse_move(5, 400)
    toy.mouse_scroll(1.0)


def test_a_click_off_a_text_field_drops_the_focus_and_a_click_on_one_keeps_it(toy):
    """The number field takes the focus; the pane does not."""
    field = toy.program.field
    toy.mouse_press(Event(*centre(field)))
    toy.mouse_release(Event(*centre(field)))
    assert toy.focus is field and field.editing
    toy.mouse_press(Event(*centre(toy.program.pane)))
    assert toy.focus is None and toy.keyboard_focus is False


def test_tab_walks_the_focus_forward_and_back_with_a_ring(toy):
    """Tab goes through the focusable widgets; Shift+Tab goes back; both wrap."""
    program = toy.program
    order = toy.input.focusables()
    assert order[:2] == [program.one, program.two]
    assert program.box in order and program.field in order
    toy.key(KEY["tab"], 1, 0)
    assert toy.focus is order[0] and toy.keyboard_focus
    toy.key(KEY["tab"], 1, 0)
    assert toy.focus is order[1]
    toy.key(KEY["tab"], 1, SHIFT)
    assert toy.focus is order[0]
    toy.key(KEY["tab"], 1, SHIFT)
    assert toy.focus is order[-1]


def test_tab_with_nothing_to_focus_does_nothing(toy):
    """Every focusable widget is disabled."""
    for widget in toy.input.focusables():
        widget.enabled = False
    toy.key(KEY["tab"], 1, 0)
    assert toy.focus is None


def test_space_or_enter_clicks_the_focused_button_and_escape_clears_the_focus(toy):
    """A focused checkbox is flipped by Space; Escape drops the focus and the ring."""
    box = toy.program.box
    toy.set_focus(box)
    toy.keyboard_focus = True
    toy.key(KEY["space"], 1, 0)
    assert box.checked
    toy.key(KEY["enter"], 1, 0)
    assert not box.checked
    toy.key(KEY["escape"], 1, 0)
    assert toy.focus is None and not toy.keyboard_focus


def test_a_focused_number_field_takes_the_keys_it_uses_and_passes_the_rest_on(toy):
    """Typing goes to the field; a key it does not use reaches the commands."""
    field = toy.program.field
    toy.mouse_press(Event(*centre(field)))
    toy.char("7")
    assert field.buffer.endswith("7")
    toy.key(KEY["a"], 1, 0)
    assert toy.program.commands[-1] == (KEY["a"], False)
    toy.set_focus(None)
    toy.char("x")


def test_ctrl_q_quits_and_other_keys_go_to_the_programs_commands(toy):
    """Ctrl+Q asks to end; Ctrl+1 and a plain key reach `command`."""
    toy.key(KEY["q"], 1, CTRL)
    assert toy.quit_requested
    toy.key(KEY["1"], 1, CTRL)
    assert toy.program.commands[-1] == (KEY["1"], True)
    toy.key(KEY["g"], 1, 0)
    assert toy.program.commands[-1] == (KEY["g"], False)


def test_a_button_group_checks_one_and_clears_the_rest(toy):
    """Clicking one tool button un-checks the other."""
    one, two = toy.program.one, toy.program.two
    for button in (one, two):
        toy.mouse_press(Event(*centre(button)))
        toy.mouse_release(Event(*centre(button)))
        assert button.checked and not (two if button is one else one).checked
    toy.select_in_group("none", one)


def test_a_menu_opens_under_its_title_and_closes_on_a_click_elsewhere(toy):
    """The title lights while open; a click in the pane only closes it."""
    menubar = toy.layout.chrome.menubar
    _, left, width = menubar.places[1]
    toy.mouse_press(Event(left + width / 2, 10))
    popup = toy.input.popup
    assert popup is not None and popup.owner == "Edit"
    assert menubar.open_title == "Edit" and popup.x == left
    assert toy.busy()
    toy.mouse_press(Event(*centre(toy.program.pane)))
    assert toy.input.popup is None and menubar.open_title is None
    assert not any(item[0] == "press" for item in toy.program.pane.seen)


def test_moving_to_another_title_with_a_menu_open_opens_that_menu(toy):
    """The popup changes owner as the pointer slides along the bar."""
    places = toy.layout.chrome.menubar.places
    (_, l0, w0), (_, l1, w1) = places[0], places[1]
    toy.mouse_press(Event(l0 + w0 / 2, 10))
    assert toy.input.popup.owner == "File"
    toy.mouse_move(l1 + w1 / 2, 10)
    assert toy.input.popup.owner == "Edit"
    toy.mouse_move(l1 + w1 / 2 + 3, 10)
    assert toy.input.popup.owner == "Edit"


def test_a_click_on_a_menu_line_runs_it_and_closes_the_menu(toy):
    """The action is the program's; the popup is gone after."""
    _, left, width = toy.layout.chrome.menubar.places[0]
    toy.mouse_press(Event(left + width / 2, 10))
    popup = toy.input.popup
    x, y = popup.x + 20, popup.y + 6 + 16
    toy.mouse_move(x, y)
    toy.mouse_press(Event(x, y))
    toy.mouse_release(Event(x, y))
    assert toy.program.commands == ["File"] and toy.input.popup is None


def test_a_click_on_the_bar_outside_the_titles_opens_nothing(toy):
    """No title there."""
    menubar = toy.layout.chrome.menubar
    toy.mouse_press(Event(900, 10))
    assert toy.input.popup is None
    assert menubar.title_at(900, 10) is None and menubar.title_at(20, 100) is None


def test_the_menu_bar_lights_the_title_under_the_pointer(toy):
    """Hover is tracked and redrawn only when it changes."""
    menubar = toy.layout.chrome.menubar
    _, left, width = menubar.places[2]
    toy.render()
    toy.mouse_move(left + width / 2, 10)
    assert menubar.hover_title == "View" and toy.dirty
    toy.render()
    toy.mouse_move(left + width / 2 + 2, 10)
    toy.mouse_move(900, 10)
    assert menubar.hover_title is None


def test_a_tooltip_shows_after_the_pointer_rests_and_goes_when_it_moves(toy):
    """Half a second over a button with a tip shows it; a move clears it."""
    two, pointer = toy.program.two, toy.input.pointer
    toy.mouse_move(*centre(two))
    toy.tick(pointer.since + 0.1)
    assert pointer.tip is None
    toy.tick(pointer.since + 1.0)
    text, _, y = pointer.tip
    assert text == "Goal" and y > two.y + two.height
    toy.render()
    toy.mouse_move(*centre(two))
    assert pointer.tip is None
    toy.mouse_move(*centre(toy.program.pane))
    toy.tick(pointer.since + 1.0)
    assert pointer.tip is None


def test_a_tooltip_is_drawn_inside_the_window(toy):
    """Dark box near the button."""
    toy.mouse_move(*centre(toy.program.two))
    pointer = toy.input.pointer
    toy.tick(pointer.since + 1.0)
    frame = toy.render().astype(int)
    _, x, y = pointer.tip
    assert frame[int(y) + 3, int(x), :].sum() < 3 * 120


def test_a_message_shows_in_the_status_bar_and_returns_to_the_default(toy):
    """After its time the default text is back."""
    status = toy.layout.chrome.status
    status.default = "Ready."
    status.show("Something happened.", seconds=2.0)
    assert status.message == "Something happened."
    toy.tick(time.perf_counter() + 5.0)
    assert status.message == "Ready." and status.until == 0.0


def test_the_loop_stays_awake_while_something_moves(toy):
    """Idle, then busy for an animation, an open menu, and the program's own reason."""
    assert not toy.busy()
    toy.layout.slide("right", False)
    assert toy.busy()
    toy.animator.tick(time.perf_counter() + 1.0)
    toy.program.busy_now = True
    assert toy.busy()
    toy.program.busy_now = False
    toy.input.open_menu("View")
    assert toy.busy()
    toy.close_popup()
    toy.close_popup()
    assert not toy.busy()


def test_ticks_reach_the_programs_hook(toy):
    """Each tick calls `after_tick` with the time."""
    toy.tick(5.0)
    assert toy.program.ticked == [5.0]


def test_a_popup_can_be_opened_and_closed_through_the_window_contract(toy):
    """The widgets' way in: `open_popup` and `close_popup` on the shell."""
    toy.open_popup(Popup([MenuItem("x")], 10, 10))
    assert toy.input.popup is not None and toy.input.popup.host is toy
    toy.close_popup()
    assert toy.input.popup is None


def test_the_clipboard_is_kept_until_the_window_replaces_it(toy):
    """A plain store."""
    toy.clip_set("1.5")
    assert toy.clip_get() == "1.5"


def test_the_status_and_menu_bars_draw_their_text(toy):
    """Titles in the menu bar; message, place and count in the status bar."""
    status = toy.layout.chrome.status
    status.message, status.count, status.coords = "Hello", "2 robots", (1.5, 2.25)
    frame = toy.render().astype(int)
    assert (frame[8:26, 10:140].sum(axis=2) < 3 * 120).any()
    assert (frame[700 - 22 : 700 - 4, 10:60].sum(axis=2) < 3 * 180).any()
    assert (frame[700 - 22 : 700 - 4, 900:990].sum(axis=2) < 3 * 180).any()


def test_a_key_that_the_focused_widget_uses_does_not_reach_the_commands(toy):
    """Enter in a number field is the field's: it accepts the value."""
    field = toy.program.field
    toy.mouse_press(Event(*centre(field)))
    field.buffer = "4"
    toy.key(KEY["enter"], 1, 0)
    assert field.value == 4.0 and not toy.program.commands


def test_an_open_menu_is_drawn_over_the_window(toy):
    """The popup's card is in the frame once it is open."""
    toy.render()
    _, left, width = toy.layout.chrome.menubar.places[0]
    toy.mouse_press(Event(left + width / 2, 10))
    popup = toy.input.popup
    frame = toy.render().astype(int)
    x, y = int(popup.x + 30), int(popup.y + 22)
    assert (frame[y - 6 : y + 6, x : x + 60].sum(axis=2) < 3 * 200).any()


def test_a_popup_that_would_leave_the_window_goes_above_its_anchor_or_up(toy):
    """Near the bottom a list opens upward; with no anchor it moves up; right too."""
    height = toy.screen.height
    items = [MenuItem(f"line {k}") for k in range(6)]
    anchored = Popup(items, 20, height - 20)
    anchored.flip_to = height - 40
    toy.open_popup(anchored)
    assert anchored.y + anchored.height == pytest.approx(height - 40)
    plain = Popup(items, 20, height - 20)
    toy.open_popup(plain)
    assert plain.y + plain.height == pytest.approx(height - 4)
    wide = Popup(items, toy.screen.width - 10, 100)
    toy.open_popup(wide)
    assert wide.x + wide.width <= toy.screen.width - 4
    fits = Popup(items, 30, 100)
    toy.open_popup(fits)
    assert (fits.x, fits.y) == (30, 100)
    cramped = Popup(items, 20, 10)
    cramped.flip_to = 2
    cramped.set_rect(20, height - 30, cramped.width, cramped.height)
    toy.open_popup(cramped)
    assert cramped.y + cramped.height == pytest.approx(height - 4)


def test_a_right_click_goes_to_the_widget_under_it_unless_a_menu_is_open(toy):
    """The right button reaches `on_secondary`; other buttons do nothing."""
    seen = []
    toy.program.one.on_secondary = seen.append
    x, y = toy.program.one.x + 5, toy.program.one.y + 5
    toy.mouse_press(Event(x, y, button=1))
    assert len(seen) == 1
    toy.mouse_press(Event(x, y, button=2))
    toy.open_popup(Popup([MenuItem("a")], 400, 200))
    toy.mouse_press(Event(x, y, button=1))
    assert len(seen) == 1
    toy.mouse_press(Event(900, 650, button=1))
