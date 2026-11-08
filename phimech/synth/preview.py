"""The 7-view preview contact sheet, and a deterministic silhouette comparison.

Why this exists
---------------
Every numeric gate in PhiMech can pass on a part that is manifold, printable, correctly walled,
fits the bed -- and is the wrong shape. A box where a cylinder was asked for has the right bbox,
the right wall and a positive volume. Nothing numeric notices. The one thing that does is a
*picture*: render the part from the seven canonical directions (iso, front, back, left, right, top,
bottom), compose them into one image, and hand that image back to the agent that asked for the
part, so it can look before it answers. Two independent projects (CADAM, agentcad) converged on
exactly this loop; see the public docs/README.md.

The sheet is **advisory**. No gate verdict is ever derived from an image. What *is* machine-checkable
is ``compare_sheets``: a per-view silhouette IoU between a reference sheet and a candidate sheet,
which is the deterministic proxy the wrong-shape benchmark (``phimech/bench/wrong_shape.py``) uses
to put a number on "does it look right".

Design constraints, and why
---------------------------
* **stdlib only.** This module runs under ``/usr/bin/python3`` 3.9, FreeCAD's python and the MCP
  venv alike, so the PNGs OpenSCAD writes are decoded (``zlib`` + PNG filter reversal) and the
  sheet re-encoded by hand. OpenSCAD writes 8-bit RGB/RGBA non-interlaced PNGs; that is what is
  supported, and anything else is refused with a clear error rather than mis-decoded.
* **OpenSCAD only.** The views come from ``openscad --camera ... --projection=o`` as a child
  process; the binary lookup is the OpenSCAD backend's own. FreeCAD's offscreen rendering is not
  used: one renderer means one set of pixels to reason about.
* **Everything is normalised to an STL first.** A CSG AST or a ``.scad`` file is rendered once to
  a mesh, measured with the backend's stdlib STL reader, and the seven views are drawn from *that*
  mesh. What the sheet shows is therefore exactly what was measured.
* **Units are mm and mm3** in every number this module reports; images are pixels.

Integration hooks (wired by the coordinator, not here): ``register_cli(subparsers)`` adds the
``preview`` CLI command, ``for_build(out_dir)`` is the one-call entry for the synth pipeline, and
``doctor_section()`` is a JSON-serialisable health block. ``phimech/mcp/tools_preview.py`` exposes
the MCP tools.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from phimech.backends.base import BackendError

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

#: OpenSCAD ``--camera`` rotations (``$vpr`` semantics) for each canonical view. Verified against
#: the GUI's View menu: front looks at the -Y face, right looks at the +X face, top looks down -Z.
CAMERAS: Dict[str, Tuple[float, float, float]] = {
    "iso": (55.0, 0.0, 25.0),
    "front": (90.0, 0.0, 0.0),
    "back": (90.0, 0.0, 180.0),
    "left": (90.0, 0.0, 270.0),
    "right": (90.0, 0.0, 90.0),
    "top": (0.0, 0.0, 0.0),
    "bottom": (180.0, 0.0, 0.0),
}

DEFAULT_VIEWS: Tuple[str, ...] = ("iso", "front", "back", "left", "right", "top", "bottom")

#: For each face-on view: (axis the camera looks along, +1 if it looks toward +axis). Used to
#: shade the part by depth - see `_depth_banded_scene`.
VIEW_DEPTH_AXIS: Dict[str, Tuple[int, int]] = {
    "front": (1, +1),
    "back": (1, -1),
    "left": (0, +1),
    "right": (0, -1),
    "top": (2, -1),
    "bottom": (2, +1),
}

#: Depth bands for the face-on views, as fractions of the part's extent along the view axis,
#: and the brightness of each band (nearest first). OpenSCAD shades a face by its normal only,
#: so in an orthographic face-on view every face that points at the camera is one flat colour
#: whatever its depth: a window in the +X wall of a hollow box shows the inner face of the -X
#: wall, which points the same way, and the window vanishes - LEFT and RIGHT become mirror
#: images of one blank wall. Colouring the material by how far it is from the camera puts the
#: recess back in the picture. The first band is thin so a wall (and a shallow pocket) reads as
#: "near" and anything seen through an opening reads as clearly deeper.
DEPTH_BANDS: Tuple[float, ...] = (0.0, 0.08, 0.2, 0.4, 0.7, 1.0)
DEPTH_SHADES: Tuple[float, ...] = (1.0, 0.82, 0.66, 0.52, 0.40)

#: Base face colour for the depth-banded views: the Tomorrow scheme's own face blue, so the
#: near band looks exactly like the iso tile and like sheets rendered before banding.
_PART_RGB = (0.27, 0.45, 0.72)

#: Sheet layout: 4 columns x 2 rows; the last slot is the text tile with bbox and volume.
GRID_COLUMNS = 4
GRID_ROWS = 2
INFO_SLOT = GRID_COLUMNS * GRID_ROWS - 1

#: Silhouette IoU below which two sheets are called different shapes. Calibrated on the wrong-shape
#: benchmark (``fixtures/bench/preview/results.json``): an identical re-render scores 1.0, the same
#: shape 2 mm (5 %) taller scores 0.954 - a size change the numeric gates own, so it must *not*
#: be flagged - and the closest wrong shape in the suite scores 0.870. 0.92 sits between with at
#: least 0.03 of clearance on both sides, which is also the headroom for anti-aliasing differences
#: between OpenSCAD builds (a re-render on one machine is pixel-identical).
DEFAULT_IOU_THRESHOLD = 0.92

#: A pixel is foreground when any channel differs from the tile background by more than this.
FOREGROUND_DELTA = 40

#: Width of the tile border ring, excluded from silhouettes because it is drawn, not rendered.
BORDER_PX = 1

DEFAULT_COLORSCHEME = "Tomorrow"
_FALLBACK_BACKGROUND = (248, 248, 248)  # Tomorrow's background, used only before a view is decoded
_LABEL_FG = (255, 255, 255)
_LABEL_BG = (24, 24, 24)
_INFO_FG = (24, 24, 24)
_BORDER_RGB = (200, 200, 200)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_BYTES_PER_PIXEL = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


# --------------------------------------------------------------------------------------
# a 5x7 bitmap font, enough to burn view names and numbers into a tile
# --------------------------------------------------------------------------------------

_FONT: Dict[str, Tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "01100", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}
_GLYPH_W, _GLYPH_H, _GLYPH_ADVANCE = 5, 7, 6


def font_scale(tile_px: int) -> int:
    """Integer glyph scale for a tile: 1 at the fixture size (160 px), 3 at the default 400 px."""
    return max(1, int(tile_px) // 130)


def label_band_px(tile_px: int) -> int:
    """Height of the top band a burnt-in label may occupy; ``compare_sheets`` ignores it.

    Two scale units of margin, a label box of seven glyph rows plus one unit of padding on each
    side, and two more units below - a pure function of the tile size, so a reference and a
    candidate sheet exclude exactly the same rows.
    """
    scale = font_scale(tile_px)
    return 13 * scale


# --------------------------------------------------------------------------------------
# raster helpers (rows of RGB bytearrays)
# --------------------------------------------------------------------------------------

Canvas = List[bytearray]


def _blank(width: int, height: int, rgb: Tuple[int, int, int]) -> Canvas:
    row = bytearray(bytes(rgb) * width)
    return [bytearray(row) for _ in range(height)]


def _fill_rect(canvas: Canvas, x: int, y: int, w: int, h: int, rgb: Tuple[int, int, int]) -> None:
    if w <= 0 or h <= 0 or not canvas:
        return
    width = len(canvas[0]) // 3
    x0, x1 = max(0, x), min(width, x + w)
    if x1 <= x0:
        return
    run = bytes(rgb) * (x1 - x0)
    for row in range(max(0, y), min(len(canvas), y + h)):
        canvas[row][x0 * 3 : x1 * 3] = run


def _blit(canvas: Canvas, x: int, y: int, tile: Canvas) -> None:
    for offset, row in enumerate(tile):
        target = y + offset
        if 0 <= target < len(canvas):
            canvas[target][x * 3 : x * 3 + len(row)] = row


def _draw_text(
    canvas: Canvas, x: int, y: int, text: str, scale: int, rgb: Tuple[int, int, int]
) -> int:
    """Draw ``text`` with the bitmap font; returns the x just past the last glyph."""
    cursor = x
    for char in text.upper():
        glyph = _FONT.get(char) or _FONT["?"]
        for row_index, bits in enumerate(glyph):
            for col_index, bit in enumerate(bits):
                if bit == "1":
                    _fill_rect(
                        canvas,
                        cursor + col_index * scale,
                        y + row_index * scale,
                        scale,
                        scale,
                        rgb,
                    )
        cursor += _GLYPH_ADVANCE * scale
    return cursor


def text_width_px(text: str, scale: int) -> int:
    return len(text) * _GLYPH_ADVANCE * scale - scale if text else 0


# --------------------------------------------------------------------------------------
# PNG codec: 8-bit, non-interlaced, all five scanline filters
# --------------------------------------------------------------------------------------


def _unfilter(kind: int, cur: bytearray, prev: bytearray, bpp: int) -> None:
    """Reverse one PNG scanline filter in place (PNG spec 9.2). Plain loops: no numpy here."""
    n = len(cur)
    if kind == 0:
        return
    if kind == 1:  # Sub
        for i in range(bpp, n):
            cur[i] = (cur[i] + cur[i - bpp]) & 0xFF
    elif kind == 2:  # Up
        for i in range(n):
            cur[i] = (cur[i] + prev[i]) & 0xFF
    elif kind == 3:  # Average
        for i in range(n):
            left = cur[i - bpp] if i >= bpp else 0
            cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
    elif kind == 4:  # Paeth
        for i in range(n):
            if i >= bpp:
                a = cur[i - bpp]
                c = prev[i - bpp]
            else:
                a = c = 0
            b = prev[i]
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            if pa <= pb and pa <= pc:
                pred = a
            elif pb <= pc:
                pred = b
            else:
                pred = c
            cur[i] = (cur[i] + pred) & 0xFF
    else:
        raise ValueError("unknown PNG filter type {}".format(kind))


def _to_rgb(row: bytearray, color_type: int, palette: Optional[bytes]) -> bytearray:
    if color_type == 2:
        return row
    if color_type == 6:
        out = bytearray(len(row) // 4 * 3)
        out[0::3], out[1::3], out[2::3] = row[0::4], row[1::4], row[2::4]
        return out
    if color_type == 0:
        out = bytearray(len(row) * 3)
        out[0::3] = out[1::3] = out[2::3] = row
        return out
    if color_type == 4:
        grey = row[0::2]
        out = bytearray(len(grey) * 3)
        out[0::3] = out[1::3] = out[2::3] = grey
        return out
    if color_type == 3:
        if palette is None:
            raise ValueError("palette PNG without a PLTE chunk")
        out = bytearray()
        for index in row:
            out += palette[index * 3 : index * 3 + 3]
        return out
    raise ValueError("unsupported PNG colour type {}".format(color_type))


def read_png(path: Union[str, Path]) -> Tuple[int, int, Canvas]:
    """Decode an 8-bit non-interlaced PNG into ``(width, height, rows)`` of packed RGB bytes.

    This is the subset OpenSCAD (via lodepng) writes and the subset this module writes. A 16-bit
    or interlaced file is refused explicitly so it cannot be silently mis-decoded into a mask.
    """
    data = Path(path).read_bytes()
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG file: {}".format(path))
    header: Optional[Tuple[int, ...]] = None
    palette: Optional[bytes] = None
    idat: List[bytes] = []
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        chunk_type = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif chunk_type == b"PLTE":
            palette = bytes(body)
        elif chunk_type == b"IDAT":
            idat.append(body)
        elif chunk_type == b"IEND":
            break
    if header is None:
        raise ValueError("PNG without an IHDR chunk: {}".format(path))
    width, height, depth, color_type, _, _, interlace = header
    if depth != 8:
        raise ValueError("only 8-bit PNGs are supported, got {}-bit: {}".format(depth, path))
    if interlace:
        raise ValueError("interlaced PNGs are not supported: {}".format(path))
    if color_type not in _BYTES_PER_PIXEL:
        raise ValueError("unsupported PNG colour type {}: {}".format(color_type, path))
    if not idat:
        raise ValueError("PNG without image data: {}".format(path))
    bpp = _BYTES_PER_PIXEL[color_type]
    stride = width * bpp
    raw = zlib.decompress(b"".join(idat))
    if len(raw) < height * (stride + 1):
        raise ValueError("PNG image data is truncated: {}".format(path))
    rows: Canvas = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        kind = raw[offset]
        cur = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        _unfilter(kind, cur, prev, bpp)
        rows.append(_to_rgb(cur, color_type, palette))
        prev = cur
    return width, height, rows


def write_png(path: Union[str, Path], width: int, height: int, rows: Canvas) -> int:
    """Encode packed-RGB rows as an 8-bit RGB PNG (filter 0, zlib level 9). Returns the size."""
    if len(rows) != height or any(len(row) != width * 3 for row in rows):
        raise ValueError("canvas does not match {}x{} RGB".format(width, height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    payload = (
        _PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return len(payload)


# --------------------------------------------------------------------------------------
# OpenSCAD driving
# --------------------------------------------------------------------------------------


def _openscad_binary() -> str:
    """The OpenSCAD backend's own lookup (PHIMECH_OPENSCAD, PATH, then the .app bundle)."""
    from phimech.backends import openscad as backend_module

    binary = backend_module._candidate_binary()  # the backend owns the lookup order
    if binary is None:
        raise BackendError(
            "OpenSCAD executable not found; install it or set PHIMECH_OPENSCAD "
            "(run `phimech doctor`)"
        )
    return binary


def _run_openscad(command: Sequence[str], timeout: float, what: str) -> str:
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendError(
            "OpenSCAD timed out after {:.0f}s while rendering {}".format(timeout, what),
            stderr=str(exc.stderr or ""),
        ) from exc
    except OSError as exc:
        raise BackendError("unable to execute OpenSCAD", stderr=str(exc)) from exc
    log = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise BackendError(
            "OpenSCAD failed (exit {}) while rendering {}".format(completed.returncode, what),
            stderr=log,
            returncode=completed.returncode,
        )
    return log


def _camera_argument(view: str) -> str:
    rx, ry, rz = CAMERAS[view]
    # The trailing distance is a placeholder: --viewall recomputes it from the bounding box.
    return "--camera=0,0,0,{:g},{:g},{:g},140".format(rx, ry, rz)


def render_view(
    scad_path: Path,
    view: str,
    out_png: Path,
    *,
    size_px: int,
    colorscheme: str = DEFAULT_COLORSCHEME,
    timeout: float = 120.0,
) -> float:
    """Render one orthographic view of a .scad file to a square PNG. Returns seconds taken.

    Preview mode (no ``--render``) is deliberate: the input is already a mesh, so a full CGAL
    evaluation would only cost time, and it would refuse a mesh that is not quite manifold - the
    very part an agent most needs to see.
    """
    if view not in CAMERAS:
        raise ValueError("unknown view {!r}; choose from {}".format(view, ", ".join(CAMERAS)))
    started = time.monotonic()
    command = [
        _openscad_binary(),
        "-o",
        str(out_png),
        _camera_argument(view),
        "--projection=o",
        "--autocenter",
        "--viewall",
        "--imgsize={0},{0}".format(int(size_px)),
        "--colorscheme={}".format(colorscheme),
        str(scad_path),
    ]
    log = _run_openscad(command, timeout, "the {} view".format(view))
    if not out_png.is_file() or out_png.stat().st_size == 0:
        raise BackendError("OpenSCAD wrote no image for the {} view".format(view), stderr=log)
    return time.monotonic() - started


Point3 = Tuple[float, float, float]


def _clip_polygon(polygon: List[Point3], axis: int, bound: float, keep_above: bool
                  ) -> List[Point3]:
    """Sutherland-Hodgman against one axis-aligned plane; keeps the winding."""
    def inside(point: Point3) -> bool:
        return point[axis] >= bound if keep_above else point[axis] <= bound

    out: List[Point3] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        if inside(current):
            if not inside(previous):
                out.append(_cross_plane(previous, current, axis, bound))
            out.append(current)
        elif inside(previous):
            out.append(_cross_plane(previous, current, axis, bound))
    return out


def _cross_plane(a: Point3, b: Point3, axis: int, bound: float) -> Point3:
    t = (bound - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))


def _write_binary_stl(path: Path, triangles: Sequence[Sequence[Point3]]) -> None:
    with path.open("wb") as handle:
        handle.write(b"phimech preview depth band".ljust(80, b" "))
        handle.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            handle.write(struct.pack("<12fH", 0.0, 0.0, 0.0,
                                     *(c for point in triangle for c in point), 0))


def _depth_banded_scene(triangles: Sequence[Sequence[Point3]], stl: Path, view: str,
                        bbox_min: Sequence[float], bbox_max: Sequence[float],
                        workdir: Path) -> Any:
    """The part's surface split into depth bands along ``view``'s axis, each its own shade.

    Returns a CSG node; the bare import for a view with no depth axis (iso). The triangles are
    clipped at the band planes in Python and each band is written as its own STL, so OpenSCAD
    only *draws* meshes - no boolean in preview mode, where OpenCSG mis-renders a non-convex
    import cut by a slab (a convexity-1 import loses whole walls). The bands partition the
    surface exactly, so the silhouette - and every `compare_sheets` IoU - is unchanged; only
    the shading inside it gains a depth cue.
    """
    from phimech.csg.ast import Color, Import, Union as CsgUnion

    if view not in VIEW_DEPTH_AXIS:
        return Import(str(stl))
    axis, direction = VIEW_DEPTH_AXIS[view]
    extent = float(bbox_max[axis]) - float(bbox_min[axis])
    if extent <= 0 or not triangles:
        return Import(str(stl))
    near = float(bbox_min[axis]) if direction > 0 else float(bbox_max[axis])
    infinity = 1e9
    bands = []
    for index, shade in enumerate(DEPTH_SHADES):
        d0 = -infinity if index == 0 else DEPTH_BANDS[index] * extent
        d1 = infinity if index == len(DEPTH_SHADES) - 1 else DEPTH_BANDS[index + 1] * extent
        lo, hi = sorted((near + direction * d0, near + direction * d1))
        clipped: List[Tuple[Point3, Point3, Point3]] = []
        for triangle in triangles:
            values = [point[axis] for point in triangle]
            if max(values) < lo or min(values) > hi:
                continue
            polygon: List[Point3] = [tuple(point) for point in triangle]  # type: ignore[misc]
            if min(values) < lo:
                polygon = _clip_polygon(polygon, axis, lo, True)
            if polygon and max(values) > hi:
                polygon = _clip_polygon(polygon, axis, hi, False)
            for k in range(1, len(polygon) - 1):
                clipped.append((polygon[0], polygon[k], polygon[k + 1]))
        if not clipped:
            continue
        band_stl = workdir / "{}-band{}.stl".format(view, index)
        _write_binary_stl(band_stl, clipped)
        rgba = tuple(round(c * shade, 4) for c in _PART_RGB) + (1.0,)
        bands.append(Color(rgba=rgba, children=[Import(str(band_stl))]))
    return CsgUnion(children=bands) if bands else Import(str(stl))


def _export_stl_from_scad(scad_path: Path, stl_path: Path, timeout: float) -> None:
    from phimech.backends.openscad import OpenSCADBackend

    command = [_openscad_binary(), "-o", str(stl_path)]
    probe = getattr(OpenSCADBackend(), "_supports_manifold", None)
    if callable(probe) and probe():
        command.extend(["--backend", "manifold"])
    command.append(str(scad_path))
    log = _run_openscad(command, timeout, "an STL of {}".format(scad_path.name))
    if not stl_path.is_file() or stl_path.stat().st_size == 0:
        raise BackendError("OpenSCAD wrote no STL for {}".format(scad_path), stderr=log)


#: The three files a SolidWorks handoff bundle must carry; see `phimech/adapters/solidworks`.
SOLIDWORKS_BUNDLE_FILES = ("manifest.json", "model.step", "source.json")


def is_solidworks_bundle(path: Path) -> bool:
    """True for a directory shaped like a validated SolidWorks handoff, not a synth build."""
    return path.is_dir() and all((path / name).is_file() for name in SOLIDWORKS_BUNDLE_FILES)


def _bundle_to_stl(bundle: Path, workdir: Path, timeout: float) -> Tuple[Path, str]:
    """Validate a SolidWorks handoff bundle, then tessellate the STEP it bound.

    The bundle exists so the mesh can be traced back to an exact native document: the adapter
    refuses a hash mismatch, a non-mm unit, a tessellated body, a dirty or partially-selected
    part, and any export whose translation diagnostics were not cleared. Drawing straight off
    ``model.step`` would skip all of that, so the validator runs first and the sheet is labelled
    with the *native* document name rather than the bundle-local filename.
    """
    from phimech.adapters.solidworks.bundle import validate_bundle

    record = validate_bundle(bundle, timeout=timeout)
    step = Path(record["step_path"])
    component = record.get("manifest", {}).get("component", {})
    label = component.get("name") or step.name
    return _step_to_stl(step, workdir, timeout), label


STEP_SUFFIXES = (".step", ".stp")

#: A STEP whose kernel volume and tessellated volume disagree by more than this fraction did not
#: survive translation intact -- a silently-dropped body or a scaled import looks like a valid but
#: wrong mesh, and every downstream number would be measured off that wrong mesh.
STEP_VOLUME_TOLERANCE = 0.01


def _step_to_stl(step: Path, workdir: Path, timeout: float) -> Path:
    """Tessellate a STEP solid to an STL through the FreeCAD subprocess.

    This is the SolidWorks entry point: SolidWorks exports STEP, and every PhiMech surface that
    measures geometry measures a mesh. Three things are checked *before* the mesh is trusted,
    because each one fails silently otherwise:

    * **Units must already be mm** (rule 6). An inch STEP is perfectly valid and 25.4x too big;
      it would pass every geometric gate. PhiMech refuses it rather than converting.
    * **The body must be a real solid, not a tessellated one.** A mesh-in-STEP arrives as a
      graphics body with no closed shell, and its "faces" are triangles.
    * **Kernel volume and mesh volume must agree** within ``STEP_VOLUME_TOLERANCE``. That is what
      catches a dropped body or a botched tessellation, which otherwise yields a plausible sheet
      full of numbers measured off geometry that is not the part.
    """
    from phimech.backends.freecad import FreeCADBackend
    from phimech.backends.openscad import measure_stl
    from phimech.csg.ast import Import
    from phimech.interop.step import inspect

    facts = inspect(step)
    if facts.parse_error is not None:
        raise ValueError("{}: cannot parse STEP: {}".format(step.name, facts.parse_error))
    if facts.length_unit not in (None, "mm"):
        raise ValueError(
            "{}: STEP declares {} -- PhiMech is millimetres throughout and never converts. "
            "Re-export from the source CAD in mm.".format(step.name, facts.length_unit)
        )
    if facts.looks_tessellated or facts.solids < 1:
        raise ValueError(
            "{}: not a solid STEP ({} solid(s), {} tessellated entities). A mesh STEP carries no "
            "faces to measure; re-export as solid geometry.".format(
                step.name, facts.solids, facts.mesh_entities)
        )

    backend = FreeCADBackend()
    if not backend.available():
        raise BackendError(
            "STEP input needs FreeCAD (OpenSCAD cannot read STEP); `bin/phimech doctor` shows "
            "which binary is used and where it was found"
        )
    stl = workdir / "model.stl"
    result = backend.render(Import(str(step.resolve())), stl, "stl", timeout=timeout)

    kernel = result.measurement.volume if result.measurement else None
    mesh = measure_stl(stl).volume
    if mesh <= 0:
        raise ValueError("{}: tessellated to an empty mesh".format(step.name))
    if kernel and kernel > 0:
        drift = abs(kernel - mesh) / kernel
        if drift > STEP_VOLUME_TOLERANCE:
            raise ValueError(
                "{}: tessellation lost the part -- kernel volume {:.1f} mm3 vs mesh {:.1f} mm3 "
                "({:.1%} > {:.0%}). The mesh is not the solid; refusing to measure it.".format(
                    step.name, kernel, mesh, drift, STEP_VOLUME_TOLERANCE)
            )
    return stl


def _resolve_source(source: Any, workdir: Path, timeout: float) -> Tuple[Path, str]:
    """Normalise a CSG Node, an STL, a CSG AST JSON file or a .scad file to an STL on disk.

    Returns the STL path and a short label for the info tile. A design IR is refused with a hint:
    the sheet previews *geometry*, and an IR only becomes geometry through ``synth build``.
    """
    from phimech.csg.ast import Node, from_dict

    if isinstance(source, Node):
        from phimech.backends.openscad import OpenSCADBackend

        stl = workdir / "model.stl"
        OpenSCADBackend().render(source, stl, "stl", timeout=timeout)
        return stl, "csg ast"

    path = Path(str(source)).expanduser()
    if is_solidworks_bundle(path):
        return _bundle_to_stl(path, workdir, timeout)
    if not path.is_file():
        raise FileNotFoundError("model not found: {}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return path.resolve(), path.name
    if suffix == ".scad":
        stl = workdir / "model.stl"
        _export_stl_from_scad(path.resolve(), stl, timeout)
        return stl, path.name
    if suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "kind" in document:
            node = from_dict(document)
            from phimech.backends.openscad import OpenSCADBackend

            stl = workdir / "model.stl"
            OpenSCADBackend().render(node, stl, "stl", timeout=timeout)
            return stl, path.name
        if isinstance(document, dict) and "components" in document:
            raise ValueError(
                "{} is a design IR, not geometry: build it with `phimech synth build` and "
                "preview the build directory (preview.for_build / `preview sheet out/`)".format(
                    path.name
                )
            )
        raise ValueError("{} is neither a CSG AST nor a design IR".format(path))
    if suffix in STEP_SUFFIXES:
        return _step_to_stl(path.resolve(), workdir, timeout), path.name
    raise ValueError(
        "unsupported model {}: pass an .stl, a .step, a CSG AST .json, or a .scad file".format(
            path.name)
    )


# --------------------------------------------------------------------------------------
# sheet composition
# --------------------------------------------------------------------------------------


@dataclass
class SheetResult:
    """What ``render_sheet`` produced. All lengths mm, volume mm3, timings seconds."""

    path: str
    views: Dict[str, Dict[str, Any]]
    bbox_mm: Dict[str, List[float]]
    volume_mm3: float
    render_s: float
    tile_px: int = 400
    source: str = ""
    part: Optional[str] = None
    backend: str = "openscad"
    openscad_version: Optional[str] = None
    bytes: int = 0
    seconds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "views": self.views,
            "bbox_mm": self.bbox_mm,
            "volume_mm3": self.volume_mm3,
            "render_s": self.render_s,
            "tile_px": self.tile_px,
            "source": self.source,
            "part": self.part,
            "backend": self.backend,
            "openscad_version": self.openscad_version,
            "bytes": self.bytes,
            "seconds": self.seconds,
            "advisory": "a picture, not a gate: no gate verdict is derived from this image",
        }


def _validate_views(views: Sequence[str]) -> Tuple[str, ...]:
    chosen = tuple(str(v) for v in views)
    if not chosen:
        raise ValueError("at least one view is required")
    if len(chosen) > INFO_SLOT:
        raise ValueError("at most {} views fit on a sheet".format(INFO_SLOT))
    if len(set(chosen)) != len(chosen):
        raise ValueError("views must be unique")
    unknown = [v for v in chosen if v not in CAMERAS]
    if unknown:
        raise ValueError("unknown view(s) {}; choose from {}".format(unknown, ", ".join(CAMERAS)))
    return chosen


def _slot_origin(index: int, tile_px: int) -> Tuple[int, int]:
    return (index % GRID_COLUMNS) * tile_px, (index // GRID_COLUMNS) * tile_px


def _corner_background(tile: Canvas) -> Tuple[int, int, int]:
    """The tile's background colour: the most common colour along its edge ring."""
    return _ring_mode(tile, 0, 0, len(tile[0]) // 3, len(tile), inset=0)


def _ring_mode(
    rows: Canvas, x: int, y: int, w: int, h: int, *, inset: int
) -> Tuple[int, int, int]:
    counter: Counter = Counter()
    x0, y0, x1, y1 = x + inset, y + inset, x + w - 1 - inset, y + h - 1 - inset
    for col in range(x0, x1 + 1):
        for row in (y0, y1):
            counter[bytes(rows[row][col * 3 : col * 3 + 3])] += 1
    for row in range(y0 + 1, y1):
        for col in (x0, x1):
            counter[bytes(rows[row][col * 3 : col * 3 + 3])] += 1
    colour = counter.most_common(1)[0][0]
    return (colour[0], colour[1], colour[2])


def _draw_label(canvas: Canvas, x: int, y: int, text: str, tile_px: int) -> None:
    scale = font_scale(tile_px)
    pad = 2 * scale
    box_w = text_width_px(text, scale) + 2 * scale
    box_h = _GLYPH_H * scale + 2 * scale
    _fill_rect(canvas, x + pad, y + pad, box_w, box_h, _LABEL_BG)
    _draw_text(canvas, x + pad + scale, y + pad + scale, text, scale, _LABEL_FG)


def _draw_border(canvas: Canvas, x: int, y: int, tile_px: int) -> None:
    _fill_rect(canvas, x, y, tile_px, BORDER_PX, _BORDER_RGB)
    _fill_rect(canvas, x, y + tile_px - BORDER_PX, tile_px, BORDER_PX, _BORDER_RGB)
    _fill_rect(canvas, x, y, BORDER_PX, tile_px, _BORDER_RGB)
    _fill_rect(canvas, x + tile_px - BORDER_PX, y, BORDER_PX, tile_px, _BORDER_RGB)


def _fit_line(text: str, tile_px: int, scale: int) -> str:
    """Truncate a line so it fits the tile with a margin, marking the cut with '..'."""
    max_chars = max(4, (tile_px - 4 * scale) // (_GLYPH_ADVANCE * scale))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 2] + ".."


def info_lines(
    bbox_mm: Dict[str, List[float]], volume_mm3: float, source: str, version: Optional[str]
) -> List[str]:
    size = bbox_mm.get("size") or [0.0, 0.0, 0.0]
    lines = [
        "PHIMECH PREVIEW",
        source,
        "SIZE MM",
        "X {:.1f}".format(size[0]),
        "Y {:.1f}".format(size[1]),
        "Z {:.1f}".format(size[2]),
        "VOLUME MM3",
        "{:,.0f}".format(volume_mm3),
    ]
    if version:
        lines.append("OPENSCAD {}".format(version))
    return lines


def _draw_info_tile(
    canvas: Canvas, x: int, y: int, tile_px: int, lines: Sequence[str]
) -> None:
    scale = font_scale(tile_px)
    pad = 2 * scale
    line_h = (_GLYPH_H + 2) * scale
    cursor_y = y + pad + scale
    for line in lines:
        if cursor_y + _GLYPH_H * scale > y + tile_px - pad:
            break
        _draw_text(canvas, x + pad + scale, cursor_y, _fit_line(line, tile_px, scale), scale,
                   _INFO_FG)
        cursor_y += line_h


def compose_sheet(
    tiles: Dict[str, Canvas],
    views: Sequence[str],
    tile_px: int,
    lines: Sequence[str],
    background: Optional[Tuple[int, int, int]] = None,
) -> Canvas:
    """Lay decoded view tiles out on the 4x2 grid, label them, and add the text tile.

    Pure function of its inputs, which is what makes sheets comparable: two sheets composed from
    the same tile size put every view at the same pixel offsets.
    """
    chosen = _validate_views(views)
    for name in chosen:
        tile = tiles.get(name)
        if tile is None:
            raise ValueError("no rendered tile for view {!r}".format(name))
        if len(tile) != tile_px or any(len(row) != tile_px * 3 for row in tile):
            raise ValueError("tile {0!r} is not {1}x{1} RGB".format(name, tile_px))
    if background is None:
        background = _corner_background(tiles[chosen[0]])
    width, height = GRID_COLUMNS * tile_px, GRID_ROWS * tile_px
    canvas = _blank(width, height, background)
    for index, name in enumerate(chosen):
        x, y = _slot_origin(index, tile_px)
        _blit(canvas, x, y, tiles[name])
        _draw_label(canvas, x, y, name, tile_px)
    x, y = _slot_origin(INFO_SLOT, tile_px)
    _draw_info_tile(canvas, x, y, tile_px, lines)
    for index in range(GRID_COLUMNS * GRID_ROWS):
        x, y = _slot_origin(index, tile_px)
        _draw_border(canvas, x, y, tile_px)
    return canvas


def _openscad_version() -> Optional[str]:
    try:
        from phimech.backends.openscad import OpenSCADBackend

        return OpenSCADBackend().version()
    except Exception:  # version is decoration on the sheet, never a reason to fail it
        return None


def render_sheet(
    stl_or_ast: Any,
    out_png: Union[str, Path],
    *,
    views: Sequence[str] = DEFAULT_VIEWS,
    tile_px: int = 400,
    keep_views: bool = False,
    colorscheme: str = DEFAULT_COLORSCHEME,
    timeout: float = 120.0,
    jobs: int = 4,
) -> SheetResult:
    """Render the contact sheet for an STL path, a CSG ``Node``, a CSG AST JSON or a .scad file.

    Each view is an orthographic OpenSCAD render at ``tile_px`` square; the tiles are composed
    4x2 with the view name burnt into each and the last tile carrying the measured bbox (mm) and
    volume (mm3). With ``keep_views`` the individual view PNGs are written beside the sheet as
    ``<stem>-<view>.png``; otherwise they live in a temporary directory and the result records
    ``path: None`` for them. Views render in parallel (``jobs`` processes) because OpenSCAD
    start-up dominates at this image size.
    """
    from phimech.backends.openscad import measure_stl

    chosen = _validate_views(views)
    if int(tile_px) < 64:
        raise ValueError("tile_px must be at least 64 pixels")
    tile_px = int(tile_px)
    destination = Path(out_png).expanduser()
    if destination.suffix.lower() != ".png":
        raise ValueError("out_png must end in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seconds: Dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="phimech-preview-") as temporary:
        workdir = Path(temporary)
        t0 = time.monotonic()
        stl, label = _resolve_source(stl_or_ast, workdir, timeout=max(timeout, 600.0))
        measurement = measure_stl(stl)
        seconds["prepare"] = round(time.monotonic() - t0, 3)

        from phimech.backends.scad_emit import emit_file
        # One scene per view: face-on views are shaded by depth (`_depth_banded_scene`), the
        # iso view keeps the plain import, whose oblique lighting already separates the faces.
        from phimech.backends.openscad import _triangles_from_ascii, _triangles_from_binary

        data = stl.read_bytes()
        triangles = _triangles_from_ascii(data)
        if triangles is None:
            triangles = _triangles_from_binary(data)
        scenes: Dict[str, Path] = {}
        for name in chosen:
            scenes[name] = workdir / "view-{}.scad".format(name)
            emit_file(_depth_banded_scene(triangles, stl, name, measurement.bbox.min,
                                          measurement.bbox.max, workdir), scenes[name])

        view_paths = {name: workdir / "{}.png".format(name) for name in chosen}

        def job(name: str) -> Tuple[str, float]:
            return name, render_view(
                scenes[name], name, view_paths[name], size_px=tile_px, colorscheme=colorscheme,
                timeout=timeout,
            )

        workers = max(1, min(int(jobs), len(chosen)))
        if workers == 1:
            timings = [job(name) for name in chosen]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                timings = list(pool.map(job, chosen))
        for name, taken in timings:
            seconds["view:" + name] = round(taken, 3)

        t0 = time.monotonic()
        tiles: Dict[str, Canvas] = {}
        for name in chosen:
            width, height, rows = read_png(view_paths[name])
            if (width, height) != (tile_px, tile_px):
                raise BackendError(
                    "OpenSCAD wrote a {}x{} image for the {} view, expected {}".format(
                        width, height, name, tile_px
                    )
                )
            tiles[name] = rows
        bbox = measurement.bbox.to_dict()
        bbox_mm = {key: [float(v) for v in value] for key, value in bbox.items()}
        version = _openscad_version()
        canvas = compose_sheet(
            tiles, chosen, tile_px, info_lines(bbox_mm, measurement.volume, label, version)
        )
        size = write_png(destination, GRID_COLUMNS * tile_px, GRID_ROWS * tile_px, canvas)
        seconds["compose"] = round(time.monotonic() - t0, 3)

        views_out: Dict[str, Dict[str, Any]] = {}
        for name in chosen:
            kept: Optional[str] = None
            if keep_views:
                target = destination.with_name("{}-{}.png".format(destination.stem, name))
                shutil.copyfile(str(view_paths[name]), str(target))
                kept = str(target)
            views_out[name] = {
                "path": kept,
                "camera": [float(v) for v in CAMERAS[name]],
                "seconds": seconds["view:" + name],
            }

    return SheetResult(
        path=str(destination),
        views=views_out,
        bbox_mm=bbox_mm,
        volume_mm3=float(measurement.volume),
        render_s=round(time.monotonic() - started, 3),
        tile_px=tile_px,
        source=str(stl_or_ast) if not hasattr(stl_or_ast, "kind") else "csg ast",
        openscad_version=version,
        bytes=size,
        seconds=seconds,
    )


# --------------------------------------------------------------------------------------
# one-call entry for a build directory
# --------------------------------------------------------------------------------------


def _build_stl_candidates(out_dir: Path) -> List[Tuple[str, Path, float]]:
    """(part, stl, volume) for every printable part a build wrote, from report.json if present."""
    found: List[Tuple[str, Path, float]] = []
    report_path = out_dir / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError:
            report = {}
        printable = set(str(p) for p in (report.get("printable_parts") or []))
        measurements = report.get("measurements") if isinstance(report, dict) else None
        for output in report.get("outputs") or []:
            if not isinstance(output, dict) or str(output.get("fmt", "")).lower() != "stl":
                continue
            part = str(output.get("part", ""))
            if printable and part not in printable:
                continue
            raw = Path(str(output.get("path", "")))
            stl = raw if raw.is_absolute() else out_dir / raw.name
            if not stl.is_file() and raw.is_file():
                stl = raw
            if not stl.is_file():
                continue
            volume = 0.0
            entry = measurements.get(part) if isinstance(measurements, dict) else None
            if isinstance(entry, dict):
                try:
                    volume = float(entry.get("volume_mm3") or 0.0)
                except (TypeError, ValueError):
                    volume = 0.0
            found.append((part, stl.resolve(), volume))
    if not found:
        for stl in sorted(out_dir.glob("*.stl")):
            found.append((stl.stem, stl.resolve(), float(stl.stat().st_size)))
    return found


def for_build(
    out_dir: Union[str, Path],
    *,
    part: Optional[str] = None,
    tile_px: int = 400,
    keep_views: bool = False,
    filename: str = "preview.png",
) -> SheetResult:
    """Render ``preview.png`` beside a ``synth build`` output: the enclosure's main printable part.

    The part is the printable part with the largest measured volume (the base of an enclosure),
    or the one named by ``part``. Meant to be called by the pipeline after a build and by agents
    that want to look at what they just built.
    """
    directory = Path(out_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError("build directory not found: {}".format(directory))
    candidates = _build_stl_candidates(directory)
    if not candidates:
        raise FileNotFoundError("no STL to preview in {}".format(directory))
    if part is not None:
        matches = [c for c in candidates if c[0] == part or c[1].stem.endswith("-" + part)]
        if not matches:
            raise ValueError(
                "no printable part {!r} in {}; have {}".format(
                    part, directory, ", ".join(sorted(c[0] for c in candidates))
                )
            )
        chosen = matches[0]
    else:
        chosen = max(candidates, key=lambda c: (c[2], c[1].stat().st_size))
    result = render_sheet(chosen[1], directory / filename, tile_px=tile_px, keep_views=keep_views)
    result.part = chosen[0]
    return result


# --------------------------------------------------------------------------------------
# the numeric proxy: silhouette IoU per view
# --------------------------------------------------------------------------------------


def _tile_mask(
    rows: Canvas, x: int, y: int, tile_px: int
) -> Tuple[List[bytearray], Tuple[int, int, int]]:
    """Foreground mask of one tile as rows of 0/1 bytes; the label band and border are zeroed.

    The background is the most common colour on the tile's inner edge ring rather than a fixed
    colour scheme value, so the comparison keeps working when the scheme changes.
    """
    background = _ring_mode(rows, x, y, tile_px, tile_px, inset=BORDER_PX)
    band = label_band_px(tile_px)
    r0, g0, b0 = background
    mask: List[bytearray] = []
    lo, hi = BORDER_PX, tile_px - BORDER_PX
    for row in range(tile_px):
        line = bytearray(tile_px)
        if band <= row < hi and row >= lo:
            source = rows[y + row]
            base = x * 3
            for col in range(lo, hi):
                index = base + col * 3
                if (
                    abs(source[index] - r0) > FOREGROUND_DELTA
                    or abs(source[index + 1] - g0) > FOREGROUND_DELTA
                    or abs(source[index + 2] - b0) > FOREGROUND_DELTA
                ):
                    line[col] = 1
        mask.append(line)
    return mask, background


def mask_iou(a: Sequence[bytearray], b: Sequence[bytearray]) -> Tuple[float, int, int, int]:
    """IoU of two equally-sized 0/1 masks, plus (intersection, |a|, |b|) pixel counts.

    Two empty masks are the same (empty) silhouette, so they score 1.0 rather than 0/0.
    """
    inter = union = count_a = count_b = 0
    for row_a, row_b in zip(a, b):
        sa, sb = sum(row_a), sum(row_b)
        count_a += sa
        count_b += sb
        if sa and sb:
            both = sum(1 for pa, pb in zip(row_a, row_b) if pa and pb)
        else:
            both = 0
        inter += both
        union += sa + sb - both
    iou = 1.0 if union == 0 else inter / float(union)
    return iou, inter, count_a, count_b


def sheet_masks(
    path: Union[str, Path], views: Sequence[str] = DEFAULT_VIEWS
) -> Tuple[int, Dict[str, List[bytearray]]]:
    """Silhouette masks for every view tile of a sheet; returns ``(tile_px, {view: mask})``."""
    chosen = _validate_views(views)
    width, height, rows = read_png(path)
    if width % GRID_COLUMNS or height % GRID_ROWS or width // GRID_COLUMNS != height // GRID_ROWS:
        raise ValueError(
            "{} is not a {}x{} sheet of square tiles ({}x{})".format(
                path, GRID_COLUMNS, GRID_ROWS, width, height
            )
        )
    tile_px = width // GRID_COLUMNS
    masks: Dict[str, List[bytearray]] = {}
    for index, name in enumerate(chosen):
        x, y = _slot_origin(index, tile_px)
        masks[name], _ = _tile_mask(rows, x, y, tile_px)
    return tile_px, masks


def compare_sheets(
    reference_png: Union[str, Path],
    candidate_png: Union[str, Path],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    views: Sequence[str] = DEFAULT_VIEWS,
) -> Dict[str, Any]:
    """Per-view silhouette IoU between two sheets made with the same tile size and view order.

    Deterministic and model-free: this is the numeric wrong-shape detector. ``same_shape`` is
    ``min_iou >= threshold``. Because ``--viewall`` normalises scale, the comparison is about
    *shape*, not size - size is the numeric gates' job. Silhouettes cannot see concavities that
    do not reach the outline (an open-top tray and an open-front tray look identical from every
    axis), which is why this proxy complements, and never replaces, looking at the sheet.
    """
    chosen = _validate_views(views)
    ref_px, reference = sheet_masks(reference_png, chosen)
    cand_px, candidate = sheet_masks(candidate_png, chosen)
    if ref_px != cand_px:
        raise ValueError(
            "sheets have different tile sizes ({} vs {} px); render both with the same "
            "tile_px".format(ref_px, cand_px)
        )
    per_view: Dict[str, float] = {}
    pixels: Dict[str, Dict[str, int]] = {}
    for name in chosen:
        iou, inter, count_a, count_b = mask_iou(reference[name], candidate[name])
        per_view[name] = round(iou, 4)
        pixels[name] = {"intersection": inter, "reference": count_a, "candidate": count_b}
    min_iou = min(per_view.values())
    worst = min(chosen, key=lambda name: per_view[name])
    return {
        "per_view": per_view,
        "min_iou": round(min_iou, 4),
        "mean_iou": round(sum(per_view.values()) / float(len(per_view)), 4),
        "worst_view": worst,
        "threshold": float(threshold),
        "same_shape": min_iou >= float(threshold),
        "tile_px": ref_px,
        "foreground_px": pixels,
        "advisory": "silhouette-only; a concavity that never reaches the outline is invisible",
    }


# --------------------------------------------------------------------------------------
# doctor + CLI hooks (wired by the coordinator)
# --------------------------------------------------------------------------------------


def doctor_section() -> Dict[str, Any]:
    """Health block for `phimech doctor`: can the sheet be rendered at all?"""
    try:
        from phimech.backends.openscad import OpenSCADBackend

        backend = OpenSCADBackend()
        available = bool(backend.available())
        version = backend.version() if available else None
    except Exception as exc:  # a broken backend import is a finding, not a crash
        return {"name": "preview", "ok": False, "error": str(exc)}
    return {
        "name": "preview",
        "ok": available,
        "openscad": version,
        "views": list(DEFAULT_VIEWS),
        "iou_threshold": DEFAULT_IOU_THRESHOLD,
        "note": "7-view sheet renders through OpenSCAD only; images are advisory, never a gate"
        if available
        else "OpenSCAD not found: install it or set PHIMECH_OPENSCAD",
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _handle_sheet(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser()
    if model.is_dir():
        result = for_build(
            model, part=args.part, tile_px=args.tile_px, keep_views=args.keep_views,
            filename=Path(args.output).name if args.output else "preview.png",
        )
        if args.output and Path(args.output).expanduser().resolve() != Path(result.path):
            shutil.copyfile(result.path, str(Path(args.output).expanduser()))
            result.path = str(Path(args.output).expanduser())
    else:
        if not args.output:
            print("error: -o/--output is required for a model file", file=sys.stderr)
            return 2
        result = render_sheet(
            model, args.output, tile_px=args.tile_px, keep_views=args.keep_views,
            views=tuple(args.views.split(",")) if args.views else DEFAULT_VIEWS,
        )
    if args.json:
        _print_json(result.to_dict())
        return 0
    size = result.bbox_mm.get("size", [0, 0, 0])
    print("sheet: {}".format(result.path))
    print("views: {}".format(", ".join(result.views)))
    print("size mm: {:.1f} x {:.1f} x {:.1f}   volume mm3: {:,.1f}".format(
        size[0], size[1], size[2], result.volume_mm3))
    print("render: {:.2f} s ({} px tiles, openscad {})".format(
        result.render_s, result.tile_px, result.openscad_version or "?"))
    print("advisory: look at it; no gate verdict comes from this image")
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    result = compare_sheets(args.reference_png, args.candidate_png, threshold=args.threshold)
    if args.json:
        _print_json(result)
    else:
        for name, iou in result["per_view"].items():
            print("{:<8} iou {:.3f}".format(name, iou))
        print("min {:.3f} (worst: {})  mean {:.3f}  threshold {:.2f}  -> {}".format(
            result["min_iou"], result["worst_view"], result["mean_iou"], result["threshold"],
            "SAME SHAPE" if result["same_shape"] else "DIFFERENT SHAPE"))
    return 0 if result["same_shape"] else 1


def register_cli(subparsers: Any) -> None:
    """Add ``preview sheet`` / ``preview compare`` / ``preview bench`` to a phimech parser."""
    parser = subparsers.add_parser(
        "preview", help="render a 7-view contact sheet and compare silhouettes"
    )
    commands = parser.add_subparsers(dest="preview_command", metavar="COMMAND")

    sheet = commands.add_parser(
        "sheet", help="render the 7-view sheet of an STL, CSG AST, .scad, or a build directory"
    )
    sheet.add_argument("model", help="path to .stl / CSG .json / .scad, or a synth build dir")
    sheet.add_argument("-o", "--output", default=None, help="sheet PNG (required for a file)")
    sheet.add_argument("--tile-px", type=int, default=400, dest="tile_px",
                       help="pixels per view tile (default 400; fixtures use 160)")
    sheet.add_argument("--keep-views", action="store_true", dest="keep_views",
                       help="also write each view PNG beside the sheet")
    sheet.add_argument("--part", default=None,
                       help="for a build directory: the printable part to preview")
    sheet.add_argument("--views", default=None,
                       help="comma-separated subset of iso,front,back,left,right,top,bottom")
    sheet.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sheet.set_defaults(handler=_handle_sheet)

    compare = commands.add_parser(
        "compare", help="silhouette IoU per view between two sheets; exit 1 if the shape differs"
    )
    compare.add_argument("reference_png")
    compare.add_argument("candidate_png")
    compare.add_argument("--threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                         help="min IoU to call the shapes the same (default %.2f)"
                         % DEFAULT_IOU_THRESHOLD)
    compare.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    compare.set_defaults(handler=_handle_compare)

    try:
        from phimech.bench import wrong_shape
    except ImportError:  # the bench is optional at CLI-build time
        return
    wrong_shape.register_cli(commands)


__all__ = [
    "CAMERAS",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_VIEWS",
    "SheetResult",
    "compare_sheets",
    "compose_sheet",
    "doctor_section",
    "for_build",
    "label_band_px",
    "mask_iou",
    "read_png",
    "register_cli",
    "render_sheet",
    "render_view",
    "sheet_masks",
    "write_png",
]
