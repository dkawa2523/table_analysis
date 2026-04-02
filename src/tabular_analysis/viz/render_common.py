from __future__ import annotations
from pathlib import Path
from typing import Any
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_RENDER_RECOVERABLE_ERRORS = _OPTIONAL_IMPORT_ERRORS + (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
def plotly_go() -> Any | None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    return go
def _plotly_go() -> Any | None:
    return plotly_go()
def write_minimal_png(path: Path) -> None:
    import struct
    import zlib
    width = 1
    height = 1
    raw = b"\x00\xff\xff\xff\xff"
    compressed = zlib.compress(raw)
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
def _write_minimal_png(path: Path) -> None:
    write_minimal_png(path)
def render_placeholder(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw  # type: ignore
        img = Image.new("RGB", (640, 420), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (630, 410)], outline=(0, 0, 0), width=2)
        draw.text((20, 20), title, fill=(0, 0, 0))
        img.save(path, format="PNG")
        return
    except _RENDER_RECOVERABLE_ERRORS:
        write_minimal_png(path)
def _render_placeholder(path: Path, title: str) -> None:
    render_placeholder(path, title)
def fallback_image(path: Path | None, title: str) -> Path | None:
    if path is None:
        return None
    render_placeholder(path, title)
    return path
__all__ = [
    "plotly_go",
    "_plotly_go",
    "write_minimal_png",
    "_write_minimal_png",
    "render_placeholder",
    "_render_placeholder",
    "fallback_image",
]
