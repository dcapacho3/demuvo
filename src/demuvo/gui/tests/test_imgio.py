"""Tests for the picture writers, with a GIF decoder of their own to read them back."""

import struct
import zlib

import numpy as np
import pytest

from demuvo.gui import imgio


def lzw_decode(data: bytes, min_code: int) -> bytes:
    """GIF's LZW, read the standard way, so the writer is checked against the format."""
    clear, end = 1 << min_code, (1 << min_code) + 1
    table = [bytes([i]) for i in range(clear)] + [b"", b""]
    width, previous, bits, count, position, out = (
        min_code + 1,
        None,
        0,
        0,
        0,
        bytearray(),
    )
    while True:
        while count < width:
            bits |= data[position] << count
            count += 8
            position += 1
        code, bits, count = bits & ((1 << width) - 1), bits >> width, count - width
        if code == clear:
            table, width, previous = table[: end + 1], min_code + 1, None
            continue
        if code == end:
            return bytes(out)
        if previous is None:
            entry = table[code]
        else:
            entry = table[code] if code < len(table) else previous + previous[:1]
            table.append(previous + entry[:1])
        out += entry
        previous = entry
        if len(table) == (1 << width) and width < 12:
            width += 1


def read_gif(data: bytes):
    """Decode a GIF the writer made: (size, loop count, delays, RGB frames)."""
    assert data[:6] == b"GIF89a"
    width, height, flags, _, _ = struct.unpack("<HHBBB", data[6:13])
    size = 3 * (2 << (flags & 7))
    palette = np.frombuffer(data[13 : 13 + size], np.uint8).reshape(-1, 3)
    position, loop, delays, frames = 13 + size, None, [], []
    while data[position] != 0x3B:
        block = data[position]
        position += 1
        if block == 0x21:
            label, parts = data[position], []
            position += 1
            while data[position]:
                parts.append(data[position + 1 : position + 1 + data[position]])
                position += 1 + data[position]
            position += 1
            if label == 0xF9:
                delays.append(struct.unpack("<H", parts[0][1:3])[0])
            else:
                loop = struct.unpack("<H", parts[1][1:3])[0]
        else:
            assert block == 0x2C
            position += 9
            min_code, position, packed = data[position], position + 1, b""
            while data[position]:
                packed += data[position + 1 : position + 1 + data[position]]
                position += 1 + data[position]
            position += 1
            indices = np.frombuffer(lzw_decode(packed, min_code), np.uint8)
            frames.append(palette[indices].reshape(height, width, 3))
    return (width, height), loop, delays, frames


@pytest.mark.parametrize("channels", [3, 4])
def test_a_png_reads_back_exactly(tmp_path, channels):
    """Random pixels in colour, and in colour with opacity."""
    picture = np.random.default_rng(1).integers(0, 256, (17, 23, channels), np.uint8)
    imgio.write_png(tmp_path / "a.png", picture)
    assert (tmp_path / "a.png").read_bytes()[:8] == imgio.PNG_SIGNATURE
    assert np.array_equal(imgio.read_png(tmp_path / "a.png"), picture)


def test_a_smooth_picture_makes_a_small_png():
    """The row filter earns its place: a gradient is far smaller than its pixels."""
    gradient = np.tile(np.arange(200, dtype=np.uint8)[:, None, None], (1, 200, 3))
    assert len(imgio.png_bytes(gradient)) < gradient.size // 10


@pytest.mark.parametrize(
    "picture",
    [
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 2), np.uint8),
        np.zeros((4, 4, 3), float),
        np.zeros((0, 4, 3), np.uint8),
    ],
)
def test_a_picture_of_the_wrong_kind_is_refused(picture):
    """The message says what a picture is."""
    with pytest.raises(ValueError, match="a picture is a uint8 array"):
        imgio.png_bytes(picture)


def test_a_file_that_is_not_a_png_this_writer_made_is_refused(tmp_path):
    """Not a PNG at all, a PNG with 16 bits, and one with another row filter."""
    (tmp_path / "x.png").write_bytes(b"not a picture")
    with pytest.raises(ValueError, match="not a PNG"):
        imgio.read_png(tmp_path / "x.png")
    good = imgio.png_bytes(np.zeros((2, 2, 3), np.uint8))
    deep = good.replace(b"\x08\x02\x00\x00\x00", b"\x10\x02\x00\x00\x00", 1)
    (tmp_path / "deep.png").write_bytes(deep)
    with pytest.raises(ValueError, match="only 8-bit colour"):
        imgio.read_png(tmp_path / "deep.png")
    gray = good.replace(b"\x08\x02\x00\x00\x00", b"\x08\x00\x00\x00\x00", 1)
    (tmp_path / "gray.png").write_bytes(gray)
    with pytest.raises(ValueError, match="only 8-bit colour"):
        imgio.read_png(tmp_path / "gray.png")


def test_a_png_with_another_row_filter_is_refused(tmp_path):
    """Only filter 1 is read, which is the only one the writer uses."""
    raw = zlib.compress(bytes([0]) + bytes(6))
    body = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
    data = (
        imgio.PNG_SIGNATURE
        + imgio._chunk(b"IHDR", body)  # pylint: disable=protected-access
        + imgio._chunk(b"IDAT", raw)  # pylint: disable=protected-access
        + imgio._chunk(b"IEND", b"")  # pylint: disable=protected-access
    )
    (tmp_path / "f0.png").write_bytes(data)
    with pytest.raises(ValueError, match="filter 1"):
        imgio.read_png(tmp_path / "f0.png")


def moving_dot(count=5):
    """Frames of a bright disc on a pale floor, a few colours and smooth edges."""
    rows, cols = np.mgrid[0:32, 0:48]
    frames = []
    for k in range(count):
        frame = np.full((32, 48, 3), (230, 246, 244), np.uint8)
        frame[(rows - 16) ** 2 + (cols - 8 - 6 * k) ** 2 < 36] = (31, 181, 166)
        frames.append(frame)
    return frames


def test_a_gif_has_the_size_the_frames_the_delay_and_the_loop():
    """Five frames at 10 per second that play for ever."""
    frames = moving_dot()
    size, loop, delays, decoded = read_gif(imgio.gif_bytes(frames, fps=10))
    assert size == (48, 32)
    assert (loop, delays) == (0, [10] * 5)
    assert len(decoded) == 5


def test_a_picture_with_few_colours_comes_back_exactly():
    """Two colours fit the palette, so nothing is lost."""
    frames = moving_dot()
    decoded = read_gif(imgio.gif_bytes(frames))[3]
    assert all(np.array_equal(a, b) for a, b in zip(frames, decoded))


def test_a_picture_with_many_colours_comes_back_close():
    """More than 256 colours share a palette, so each pixel is near, not equal."""
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (60, 80, 3), np.uint8) for _ in range(2)]
    decoded = read_gif(imgio.gif_bytes(frames))[3]
    error = np.abs(decoded[1].astype(int) - frames[1].astype(int)).mean()
    assert error < 20


def test_a_long_frame_fills_and_resets_the_code_table():
    """10 000 pixels of 200 colours need over 4096 codes and read back exactly."""
    rng = np.random.default_rng(5)
    colors = rng.integers(0, 256, (200, 3), np.uint8)
    frame = colors[rng.integers(0, 200, (100, 100))]
    assert np.array_equal(read_gif(imgio.gif_bytes([frame]))[3][0], frame)


def test_a_frame_of_one_colour_makes_a_palette_of_one():
    """Nothing to split, and the frame still reads back."""
    frame = np.full((3, 5, 3), 77, np.uint8)
    assert np.array_equal(read_gif(imgio.gif_bytes([frame]))[3][0], frame)


def test_write_gif_writes_a_file_that_plays(tmp_path):
    """The file holds the same bytes as `gif_bytes`."""
    frames = moving_dot(2)
    imgio.write_gif(tmp_path / "a.gif", frames, fps=5)
    assert (tmp_path / "a.gif").read_bytes() == imgio.gif_bytes(frames, fps=5)


@pytest.mark.parametrize(
    ("frames", "fps", "message"),
    [
        ([], 20, "at least one frame"),
        ([np.zeros((2, 2, 3), np.uint8)], 0, "fps must be positive"),
        (
            [np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2, 3), np.uint8)],
            20,
            "same size",
        ),
        ([np.zeros((2, 2, 4), np.uint8)], 20, "a picture is a uint8 array"),
    ],
)
def test_a_gif_that_cannot_be_made_is_refused(frames, fps, message):
    """No frames, a bad rate, frames of two sizes, and a frame with opacity."""
    with pytest.raises(ValueError, match=message):
        imgio.gif_bytes(frames, fps=fps)
