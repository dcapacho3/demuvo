"""Picture files without a picture library - PNG, and animated GIF.

Pictures are numpy arrays of `uint8`, shape (height, width, 3) for colour and
(height, width, 4) for colour with opacity.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# PNG colour types by the number of channels.
PNG_COLOR_TYPE = {3: 2, 4: 6}
PALETTE_SIZE = 256
# GIF codes are at most 12 bits wide, so a table holds 4096 entries.
GIF_TABLE_SIZE = 4096
GIF_CODE_BITS = 12
# Pixels taken from each frame to choose the palette.
SAMPLE_PIXELS = 20000


def _check_picture(picture: np.ndarray, channels: tuple[int, ...]) -> None:
    """Raise unless the picture is a uint8 array of the right shape."""
    if (
        picture.dtype != np.uint8
        or picture.ndim != 3
        or picture.shape[2] not in channels
        or picture.size == 0
    ):
        raise ValueError(
            "a picture is a uint8 array, not empty, of shape (height, width, "
            f"{' or '.join(map(str, channels))}), got {picture.dtype} {picture.shape}"
        )


def _chunk(kind: bytes, data: bytes) -> bytes:
    """One PNG chunk: length, kind, data and checksum."""
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def png_bytes(picture: np.ndarray) -> bytes:
    """The bytes of a PNG file.

    Each row is stored as its difference from the pixel to its left (PNG filter
    1), which makes smooth areas compress well.

    Raises:
        ValueError: if the picture is not uint8 with 3 or 4 channels.
    """
    _check_picture(picture, (3, 4))
    height, width, channels = picture.shape
    difference = picture.copy()
    difference[:, 1:] -= picture[:, :-1]  # unsigned, so it wraps as PNG expects
    rows = np.concatenate(
        [np.ones((height, 1), np.uint8), difference.reshape(height, width * channels)],
        axis=1,
    )
    header = struct.pack(
        ">IIBBBBB", width, height, 8, PNG_COLOR_TYPE[channels], 0, 0, 0
    )
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def write_png(path: str | Path, picture: np.ndarray) -> None:
    """Write a picture as a PNG file."""
    Path(path).write_bytes(png_bytes(picture))


def read_png(path: str | Path) -> np.ndarray:
    """Read a PNG that `write_png` wrote: 8 bits, colour or colour with opacity.

    Raises:
        ValueError: if the file is any other kind of PNG, or not a PNG.
    """
    data = Path(path).read_bytes()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError(f"{path}: not a PNG file")
    position, compressed, shape = 8, b"", None
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color_type, *_ = struct.unpack(">IIBBBBB", body)
            channels = {code: n for n, code in PNG_COLOR_TYPE.items()}.get(color_type)
            shape = (height, width, channels)
            if depth != 8 or channels is None:
                raise ValueError(f"{path}: only 8-bit colour PNG files can be read")
        elif kind == b"IDAT":
            compressed += body
        position += 12 + length
    height, width, channels = shape
    rows = np.frombuffer(zlib.decompress(compressed), np.uint8)
    rows = rows.reshape(height, 1 + width * channels)
    if (rows[:, 0] != 1).any():
        raise ValueError(f"{path}: only PNG files with filter 1 can be read")
    pixels = rows[:, 1:].reshape(height, width, channels)
    return np.cumsum(pixels, axis=1, dtype=np.uint8)


def _spread(box: np.ndarray) -> float:
    """The widest range of any channel among the colours of a box."""
    return float((box.max(axis=0) - box.min(axis=0)).max())


def _median_cut(colors: np.ndarray, count: int = PALETTE_SIZE) -> np.ndarray:
    """A palette of `count` colours for an (n, 3) array with more than `count` colours.

    It keeps splitting the box of colours with the most pixels and the widest
    range, at the median of its widest channel, and takes the mean of each box.
    As there are more colours than boxes, some box always holds two different ones.
    """
    boxes = [colors]
    while len(boxes) < count:
        _, index = max((len(box) * _spread(box), i) for i, box in enumerate(boxes))
        box = boxes[index]
        channel = int((box.max(axis=0) - box.min(axis=0)).argmax())
        order = np.argsort(box[:, channel], kind="stable")
        half = len(box) // 2
        boxes[index : index + 1] = [box[order[:half]], box[order[half:]]]
    return np.array([box.mean(axis=0) for box in boxes], dtype=np.uint8)


def _palette(sample: np.ndarray) -> np.ndarray:
    """The colours of a GIF: those of the sample if they fit, else a cut of them."""
    packed = (
        (sample[:, 0].astype(np.int32) << 16)
        | (sample[:, 1].astype(np.int32) << 8)
        | sample[:, 2]
    )
    unique = np.unique(packed)
    if len(unique) <= PALETTE_SIZE:
        return np.column_stack(
            [(unique >> 16) & 255, (unique >> 8) & 255, unique & 255]
        ).astype(np.uint8)
    return _median_cut(sample)


def _lzw(indices: bytes, min_code: int) -> bytes:
    """GIF's variable-width LZW compression of palette indices."""
    clear, end = 1 << min_code, (1 << min_code) + 1
    out, bits, bit_count = bytearray(), 0, 0

    def emit(code: int, width: int) -> None:
        nonlocal bits, bit_count
        bits |= code << bit_count
        bit_count += width
        while bit_count >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            bit_count -= 8

    table = {bytes([i]): i for i in range(clear)}
    width, next_code = min_code + 1, end + 1
    emit(clear, width)
    run = b""
    for byte in indices:
        candidate = run + bytes([byte])
        if candidate in table:
            run = candidate
            continue
        emit(table[run], width)
        table[candidate] = next_code
        next_code += 1
        if next_code > (1 << width) and width < GIF_CODE_BITS:
            width += 1
        if next_code >= GIF_TABLE_SIZE:
            emit(clear, width)
            table = {bytes([i]): i for i in range(clear)}
            width, next_code = min_code + 1, end + 1
        run = bytes([byte])
    emit(table[run], width)
    emit(end, width)
    if bit_count:
        out.append(bits & 0xFF)
    return bytes(out)


def _sub_blocks(data: bytes) -> bytes:
    """Data cut into GIF blocks of at most 255 bytes, each with its length."""
    blocks = bytearray()
    for start in range(0, len(data), 255):
        piece = data[start : start + 255]
        blocks += bytes([len(piece)]) + piece
    return bytes(blocks) + b"\x00"


def _indices(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """The palette entry nearest to each pixel; each colour is looked up once."""
    flat = frame.reshape(-1, 3).astype(np.int32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    unique, inverse = np.unique(packed, return_inverse=True)
    colors = np.column_stack([(unique >> 16) & 255, (unique >> 8) & 255, unique & 255])
    distance = ((colors[:, None, :] - palette[None].astype(np.int32)) ** 2).sum(axis=2)
    return distance.argmin(axis=1).astype(np.uint8)[inverse]


def gif_bytes(frames: list[np.ndarray], fps: int = 20, loop: int = 0) -> bytes:
    """The bytes of an animated GIF.

    All frames share one palette of up to 256 colours, cut from a sample of every
    frame. A picture with no more colours than that is exact; one with more is close.

    Args:
        frames: pictures of one size, uint8 with 3 channels.
        fps: frames per second.
        loop: how many times to play; 0 plays for ever.

    Raises:
        ValueError: if there are no frames, they differ in size or kind, or fps
            is not positive.
    """
    if not frames:
        raise ValueError("a GIF needs at least one frame")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    for frame in frames:
        _check_picture(frame, (3,))
        if frame.shape != frames[0].shape:
            raise ValueError("all frames of a GIF must have the same size")
    height, width = frames[0].shape[:2]
    step = max(1, height * width // SAMPLE_PIXELS)
    sample = np.concatenate([frame.reshape(-1, 3)[::step] for frame in frames])
    found = _palette(sample)
    palette = np.zeros((PALETTE_SIZE, 3), np.uint8)
    palette[: len(found)] = found
    out = bytearray(b"GIF89a" + struct.pack("<HHBBB", width, height, 0xF7, 0, 0))
    out += palette.tobytes()
    out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + struct.pack("<H", loop) + b"\x00"
    for frame in frames:
        data = _lzw(_indices(frame, palette[: len(found)]).tobytes(), 8)
        out += b"\x21\xf9\x04\x00" + struct.pack("<H", round(100 / fps)) + b"\x00\x00"
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, width, height, 0) + b"\x08"
        out += _sub_blocks(data)
    return bytes(out + b"\x3b")


def write_gif(path: str | Path, frames: list[np.ndarray], fps: int = 20) -> None:
    """Write frames as an animated GIF file that plays for ever."""
    Path(path).write_bytes(gif_bytes(frames, fps))
