"""Tests for the base widget, text, buttons, checkbox, slider and panels."""

import pytest

from demuvo.gui import font, look
from demuvo.gui.raster import Canvas, rgba
from demuvo.gui.widgets import (
    Checkbox,
    Column,
    Event,
    IconButton,
    Label,
    Panel,
    Section,
    Slider,
    TextButton,
    Widget,
    text_baseline,
)

from .fakes import FakeHost, click, hosted, ink


def test_added_widgets_know_their_parent_and_the_host_reaches_all_levels():
    """A grandchild added before the host is set gets it when the root adopts one."""
    root, middle, leaf = Widget(), Widget(), Widget()
    middle.add(leaf)
    root.add(middle)
    host = FakeHost()
    root.adopt(host)
    assert (leaf.parent, middle.parent) == (middle, root)
    assert leaf.host is host and middle.host is host
    late = Widget()
    root.add(late)
    assert late.host is host
    root.clear()
    assert not root.children


def test_a_click_finds_the_front_most_interactive_widget():
    """The front one wins; a label, a hidden widget and a miss pass through."""
    root, back, front, label = Widget(), Widget(), Widget(), Label("x")
    root.add(back, front, label)
    for widget in (root, back, front, label):
        widget.set_rect(0, 0, 50, 50)
    assert root.hit(10, 10) is front
    front.visible = False
    assert root.hit(10, 10) is back
    assert root.hit(60, 10) is None
    assert Widget().hit(0, 0) is None
    assert root.contains(0, 0) and not root.contains(50, 0)


def test_a_root_that_is_not_interactive_lets_the_click_through():
    """Nothing behind it takes the click either."""
    root = Widget()
    root.interactive = False
    root.set_rect(0, 0, 10, 10)
    assert root.hit(5, 5) is None


def test_invalidate_tells_the_host_and_does_nothing_without_one():
    """The widget itself is what is passed on."""
    widget = Widget()
    widget.invalidate()
    host = hosted(widget)
    widget.invalidate()
    assert host.invalidated[-1] is widget


def test_the_focus_ring_shows_only_for_the_keyboard_focus():
    """Nothing for a mouse focus, a ring for a keyboard one."""
    button = IconButton("play")
    host = hosted(button, (20, 20, 32, 32))
    host.focus = button
    plain = Canvas(80, 80)
    button.ring(plain)
    assert ink(plain) == 0.0
    host.keyboard_focus = True
    ringed = Canvas(80, 80)
    button.ring(ringed)
    assert ink(ringed) > 20
    button.host = None
    button.ring(Canvas(80, 80))


def test_the_pointer_and_the_press_set_the_flags_and_redraw():
    """Enter, press, release, leave."""
    widget = Widget()
    host = hosted(widget)
    widget.on_hover(True)
    assert widget.hover
    widget.on_press(Event())
    assert widget.down
    widget.on_release(Event())
    assert not widget.down
    widget.on_press(Event())
    widget.on_hover(False)
    assert not widget.hover and not widget.down
    assert len(host.invalidated) == 5
    widget.on_move(Event())
    assert widget.on_scroll(Event()) is False and widget.on_key(Event()) is False
    widget.on_char("x")
    widget.activate()
    widget.on_focus_change(True)
    widget.on_focus_change(False)


def test_a_child_outside_the_clip_or_hidden_is_not_drawn():
    """Only the visible child in the box is asked to draw."""
    drawn = []

    class Spy(Widget):
        """A widget that notes that it was drawn."""

        def draw(self, canvas):
            drawn.append(self)

    root, near, far, hidden = Widget(), Spy(), Spy(), Spy()
    root.add(near, far, hidden)
    near.set_rect(0, 0, 10, 10)
    far.set_rect(500, 500, 10, 10)
    hidden.set_rect(0, 0, 10, 10)
    hidden.visible = False
    root.draw(Canvas(100, 100))
    assert drawn == [near]


def test_text_sits_on_a_baseline_that_centres_it():
    """About a third of the size below the middle."""
    assert text_baseline(10, 32, 13) == pytest.approx(10 + 16 + 4.68)


def test_a_label_wraps_to_the_width_and_gives_the_height_it_needs():
    """Short words fill a line; a long word on its own gets its own line."""
    label = Label("alpha beta gamma delta", 13, wrap=True)
    fits = font.regular().width("alpha beta", 13) + 3
    assert label.lines(fits)[:2] == ["alpha beta", "gamma"]
    assert len(label.lines(60)) > 2
    assert Label("alpha beta", wrap=True).lines(1000) == ["alpha beta"]
    assert (
        Label("supercalifragilistic is long", wrap=True)
        .lines(40)[0]
        .startswith("super")
    )
    assert label.height_for(fits) == pytest.approx(
        len(label.lines(fits)) * 13 * 1.45 + 4
    )
    assert Label("x y z").lines(5) == ["x y z"]
    label.set_rect(0, 0, fits, 80)
    assert label.lines() == label.lines(fits)


def test_a_label_draws_left_right_and_centred_and_fades_when_disabled():
    """The ink lands in the part of the box that the alignment names."""
    boxes = {}
    for align in ("left", "right", "center"):
        label = Label("Hi", 14, align=align)
        label.set_rect(0, 0, 100, 30)
        canvas = Canvas(100, 30)
        label.draw(canvas)
        boxes[align] = [x for x in range(100) if canvas.px[:, x, 0].min() < 0.9]
    assert max(boxes["left"]) < 30 and min(boxes["right"]) > 70
    assert 30 < min(boxes["center"]) and max(boxes["center"]) < 70
    dim = Label("Hi", 14)
    dim.set_rect(0, 0, 100, 30)
    dim.enabled = False
    faded, full = Canvas(100, 30), Canvas(100, 30)
    dim.draw(faded)
    on = Label("Hi", 14)
    on.set_rect(0, 0, 100, 30)
    on.draw(full)
    assert ink(faded) < ink(full)


def test_a_wrapped_label_and_a_bold_one_draw_their_lines():
    """Two lines of ink for wrapped text, and more ink for bold."""
    wrapped = Label("one two three four five six", 13, wrap=True)
    wrapped.set_rect(0, 0, 60, 80)
    canvas = Canvas(100, 80)
    wrapped.draw(canvas)
    assert ink(canvas, (0, 0, 100, 20)) > 0 and ink(canvas, (0, 20, 100, 60)) > 0
    thin, thick = Label("MMM", 20), Label("MMM", 20, bold=True)
    for label in (thin, thick):
        label.set_rect(0, 0, 100, 30)
    a, b = Canvas(100, 30), Canvas(100, 30)
    thin.draw(a)
    thick.draw(b)
    assert ink(b) > ink(a)


def test_a_section_is_a_small_upper_case_heading():
    """Upper case, muted, bold, and a little taller than a line."""
    section = Section("Pose")
    assert (section.text, section.color, section.bold) == ("POSE", look.MUTED, True)
    assert section.preferred_height == 30.0


def test_a_button_clicks_on_release_inside_and_not_outside():
    """Press and release on it calls the handler; releasing elsewhere does not."""
    calls = []
    button = IconButton("play", lambda: calls.append(1), tooltip="Play")
    hosted(button, (10, 10, 32, 32))
    click(button)
    assert calls == [1]
    click(button, outside=True)
    assert calls == [1]
    button.enabled = False
    click(button)
    button.activate()
    assert calls == [1]
    assert button.tooltip == "Play"


def test_a_checkable_button_flips_and_a_group_button_asks_the_host():
    """The handler gets the state; in a group the host checks one."""
    states = []
    toggle = IconButton("goal", states.append, checkable=True)
    hosted(toggle)
    click(toggle)
    click(toggle)
    assert states == [True, False]
    chosen = IconButton("select", checkable=True, group="tool")
    host = hosted(chosen)
    click(chosen)
    assert host.groups == [("tool", chosen)] and chosen.checked
    chosen.activate()
    assert len(host.groups) == 2


def test_a_group_button_without_a_host_just_toggles():
    """No window, no group."""
    button = IconButton("select", checkable=True, group="tool")
    button.activate()
    assert button.checked


def test_a_button_draws_its_states():
    """Pressed, checked and hovered are lit, and disabled is faint."""

    def paint(setup):
        button = IconButton("play", checkable=True)
        host = hosted(button, (4, 4, 32, 32))
        setup(button, host)
        canvas = Canvas(40, 40)
        button.draw(canvas)
        return canvas

    idle = paint(lambda b, h: None)
    down = paint(lambda b, h: setattr(b, "down", True))
    hover = paint(lambda b, h: setattr(b, "hover", True))
    checked = paint(lambda b, h: setattr(b, "checked", True))
    off = paint(lambda b, h: setattr(b, "enabled", False))
    assert idle.px[6, 6].tolist() == [1.0, 1.0, 1.0]
    assert down.px[6, 6].tolist() == pytest.approx(list(rgba(look.MINT2)[:3]), abs=0.03)
    assert down.px[6, 6, 0] < hover.px[6, 6, 0] < 1.0
    assert checked.px[6, 6, 0] == pytest.approx(down.px[6, 6, 0], abs=0.01)
    assert ink(off) < ink(idle)


def test_a_checkbox_flips_on_click_and_on_enter_and_can_be_set_quietly():
    """The handler sees each new state; `notify=False` stays silent."""
    seen = []
    box = Checkbox("Rays", False, seen.append)
    hosted(box, (10, 10, 120, 28))
    click(box)
    box.activate()
    assert seen == [True, False]
    box.set(True, notify=False)
    assert box.checked and seen == [True, False]
    click(box, outside=True)
    assert box.checked
    box.enabled = False
    box.activate()
    assert box.checked
    Checkbox("plain").set(True)


def test_a_checkbox_draws_a_filled_box_when_on_and_an_outline_when_off():
    """Teal fill with a tick, or white with a border; hover tints the row."""

    def paint(checked, hover=False, enabled=True):
        box = Checkbox("Rays", checked)
        hosted(box, (0, 0, 120, 28))
        box.hover, box.enabled = hover, enabled
        canvas = Canvas(120, 28)
        box.draw(canvas)
        return canvas

    on, off = paint(True), paint(False)
    assert on.px[14, 17].tolist() != off.px[14, 17].tolist()
    assert on.px[7, 12, 0] < 0.5
    assert paint(False, hover=True).px[2, 60, 0] < 1.0
    assert ink(paint(True, enabled=False)) < ink(on)


def test_a_slider_sets_its_value_from_the_pointer_and_clamps_at_the_ends():
    """Press sets it, move drags only while down, and the ends hold."""
    values = []
    slider = Slider(0.5, 0.0, 1.0, values.append)
    hosted(slider, (0, 0, 116, 28))
    slider.on_press(Event(8 + 50, 10))
    assert slider.value == pytest.approx(0.5)
    slider.on_move(Event(8 + 100, 10))
    assert slider.value == pytest.approx(1.0)
    slider.on_move(Event(-100, 10))
    assert slider.value == pytest.approx(0.0)
    slider.on_release(Event())
    slider.on_move(Event(60, 10))
    assert slider.value == pytest.approx(0.0)
    assert values[0] == pytest.approx(0.5)
    slider.enabled = False
    slider.on_press(Event(60, 10))
    slider.on_move(Event(60, 10))
    assert slider.value == pytest.approx(0.0)
    Slider().on_press(Event(5, 5))


def test_a_slider_draws_a_filled_part_up_to_the_handle():
    """More teal to the left of the handle than to the right of it."""
    slider = Slider(0.5, 0.0, 1.0)
    hosted(slider, (0, 0, 116, 28))
    canvas = Canvas(116, 28)
    slider.draw(canvas)
    assert canvas.px[14, 20, 0] < canvas.px[14, 100, 0]
    slider.enabled = False
    dim = Canvas(116, 28)
    slider.draw(dim)
    assert ink(dim) < ink(canvas)


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom", None])
def test_a_panel_fills_draws_its_edge_and_clips_its_children(edge):
    """The edge is a one pixel line on its side; a child cannot draw outside."""
    panel = Panel("#ffffff", edge)

    class Splash(Widget):
        """A widget that paints far more than its own box."""

        def draw(self, canvas):
            canvas.rect((0, 0, 200, 200), rgba("#000000"))

    panel.add(Splash())
    panel.set_rect(10, 10, 40, 40)
    panel.children[0].set_rect(10, 10, 40, 40)
    canvas = Canvas(80, 80, "#ff0000")
    canvas.set_clip(0, 0, 80, 80)
    panel.draw(canvas)
    assert canvas.px[5, 5, 1] == 0.0 and canvas.px[55, 55, 1] == 0.0
    assert canvas.px[30, 30, 0] == 0.0
    assert canvas.clip == (0, 0, 80, 80)


def test_a_panel_edge_line_is_on_the_side_it_names():
    """A right edge is the last column of the panel, in the line colour."""
    panel = Panel("#ffffff", "right")
    panel.set_rect(10, 10, 40, 40)
    canvas = Canvas(80, 80, "#ff0000")
    panel.draw(canvas)
    assert canvas.px[30, 49].tolist() == pytest.approx(
        list(rgba(look.LINE)[:3]), abs=0.01
    )
    assert canvas.px[30, 48].tolist() == [1.0, 1.0, 1.0]


def test_a_column_stacks_its_children_with_padding_gap_and_wrapped_text():
    """Each child gets its height; a wrapped label gets the height its text needs."""
    column = Column(pad=10, gap=4)
    first, hidden = Widget(), Widget()
    first.preferred_height = 20.0
    hidden.visible = False
    note = Label("one two three four five six seven eight", 13, wrap=True)
    plain = Label("plain")
    column.add(first, hidden, note, plain)
    column.set_rect(0, 0, 100, 300)
    assert (first.x, first.y, first.width, first.height) == (10, 10, 80, 20)
    assert note.y == 10 + 20 + 4
    assert note.height == pytest.approx(note.height_for(80))
    assert plain.height == plain.preferred_height
    assert hidden.height == 0.0
    assert column.content_height == pytest.approx(plain.y + plain.height + 10 - 0)


def test_a_column_scrolls_with_the_wheel_when_the_content_is_taller():
    """The wheel moves the content and clamps; a short column ignores it."""
    column = Column(pad=10, gap=4)
    rows = [Widget() for _ in range(10)]
    column.add(*rows)
    column.set_rect(0, 0, 100, 100)
    host = hosted(column, (0, 0, 100, 100))
    assert column.on_scroll(Event(dy=-1)) is True
    assert column.scroll == 40 and rows[0].y == 10 - 40
    column.on_scroll(Event(dy=-100))
    assert column.scroll == column.content_height - 100
    column.on_scroll(Event(dy=100))
    assert column.scroll == 0
    assert host.invalidated
    short = Column()
    short.add(Widget())
    short.set_rect(0, 0, 100, 400)
    assert short.on_scroll(Event(dy=-1)) is False


def test_a_column_that_overflows_draws_a_scroll_bar_and_one_that_fits_does_not():
    """A narrow bar near the right edge."""
    tall = Column()
    tall.add(*[Widget() for _ in range(10)])
    tall.set_rect(0, 0, 100, 100)
    canvas = Canvas(100, 100)
    tall.draw(canvas)
    assert canvas.px[10, 94, 0] < 1.0
    fits = Column()
    fits.add(Widget())
    fits.set_rect(0, 0, 100, 100)
    clean = Canvas(100, 100)
    fits.draw(clean)
    assert clean.px[10, 94, 0] == 1.0


def test_a_text_button_clicks_on_release_inside_and_by_key_and_not_when_disabled():
    """Press and release on it, or Enter, calls the handler; a grey one does not."""
    calls = []
    button = TextButton("Add goal", lambda: calls.append(1))
    hosted(button, (10, 10, 120, 34))
    click(button)
    button.activate()
    assert calls == [1, 1]
    click(button, outside=True)
    assert calls == [1, 1]
    button.enabled = False
    click(button)
    button.activate()
    assert calls == [1, 1]
    assert TextButton("Nothing").click() is None


def test_a_text_button_draws_its_words_and_lights_when_hovered_or_pressed():
    """Ink for the label; a hover and a press each change the picture."""
    button = TextButton("Add goal")
    hosted(button, (10, 10, 120, 34))
    plain = Canvas(150, 60, look.WINDOW)
    button.draw(plain)
    button.on_hover(True)
    hover = Canvas(150, 60, look.WINDOW)
    button.draw(hover)
    button.on_press(Event(20, 20))
    pressed = Canvas(150, 60, look.WINDOW)
    button.draw(pressed)
    assert (plain.px != hover.px).any() and (hover.px != pressed.px).any()
    button.enabled = False
    grey = Canvas(150, 60, look.WINDOW)
    button.draw(grey)
    assert ink(grey) < ink(plain)
