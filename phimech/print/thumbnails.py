"""Plate thumbnails for a headless slice: rendered by OpenSCAD, injected into G-code and 3MF.

Why this exists
---------------
OrcaSlicer draws the plate thumbnail with OpenGL. Run headless (``QT_QPA_PLATFORM=offscreen``) it
cannot make a GL context ("Invalid OpenGL version" ... "skip thumbnail generating"), so the
``.gcode.3mf`` it writes has *no* image: ``_rels/.rels`` still points at
``Metadata/plate_1.png`` and ``plate_1_small.png``, but neither file is in the zip, and the G-code
has no ``; thumbnail begin`` block. The QIDI Max 4 touchscreen and the Fluidd/Mainsail file
lists then show a blank tile for every PhiMech print.

OpenSCAD renders headless fine and is already a hard dependency, so this module renders the part
itself and writes the images where each consumer looks:

* **G-code header** - the standard PNG blocks OrcaSlicer writes when it *can* render
  (``src/libslic3r/GCode/Thumbnails.hpp``)::

      ; THUMBNAIL_BLOCK_START
      ;
      ; thumbnail begin 380x380 <base64 length>
      ; <78 base64 chars per line>
      ; thumbnail end
      ;
      ; THUMBNAIL_BLOCK_END

  placed right after ``; HEADER_BLOCK_END`` - where OrcaSlicer puts it for a non-Bambu printer.
  Moonraker's ``metadata.py`` (upstream and QIDI's Max 4 fork) matches
  ``; thumbnail begin ... ; thumbnail end`` in the first MiB, checks the base64 length against
  the header and saves ``.thumbs/<name>-WxH.png``. One block per size in the machine profile's
  ``thumbnails`` list (``380x380/COLPIC, 210x210/COLPIC, 110x110/PNG`` for the Max 4). Every size
  is written as **PNG**: QIDI's ``;gimage:``/``;simage:`` COLPIC encoding is for older QIDI
  screens, and both Moonraker forks read the PNG block.
* **The .gcode.3mf** - ``Metadata/plate_1.png`` (512x512) and ``Metadata/plate_1_small.png``
  (128x128), the files and sizes OrcaSlicer writes (``bbs_3mf.cpp``; ``THUMBNAIL_SIZE_3MF``),
  stored uncompressed like OrcaSlicer's, plus ``thumbnail_file`` in ``model_settings.config``'s
  ``<plate>``. The relationships and ``[Content_Types].xml`` already declare them. The G-code
  inside the 3MF gets the same header blocks, and ``Metadata/plate_1.gcode.md5`` (uppercase hex,
  no newline) is recomputed so the archive stays self-consistent.

Nothing here can fail a slice: :func:`add_to_slice` returns warnings, never raises. A slice
whose G-code already carries ``; thumbnail begin`` (a GL-capable OrcaSlicer) is left untouched.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: OrcaSlicer's own default when a machine profile sets no ``thumbnails``.
DEFAULT_GCODE_SIZES: Tuple[Tuple[int, int], ...] = ((48, 48), (300, 300))

#: The 3MF images OrcaSlicer writes, and their sizes (``THUMBNAIL_SIZE_3MF`` and its 1/4 copy).
PLATE_PNG = "Metadata/plate_1.png"
PLATE_SMALL_PNG = "Metadata/plate_1_small.png"
PLATE_PNG_SIZE = (512, 512)
PLATE_SMALL_PNG_SIZE = (128, 128)

GCODE_MEMBER = "Metadata/plate_1.gcode"
MD5_MEMBER = "Metadata/plate_1.gcode.md5"
MODEL_SETTINGS_MEMBER = "Metadata/model_settings.config"

#: Base64 characters per G-code comment line (OrcaSlicer's ``max_row_length``).
LINE_CHARS = 78
_HEADER_END = b"; HEADER_BLOCK_END"
_BEGIN_MARKER = b"; thumbnail begin "
#: Only the head is searched for the header end and an existing block: Moonraker reads 1 MiB.
_SEARCH_BYTES = 1024 * 1024

#: Iso camera ($vpr) shared with the preview sheet's ISO tile, so the printer shows the same view.
_CAMERA = "--camera=0,0,0,55,0,25,140"
#: A dark scheme: the Max 4 touchscreen and the Fluidd/Mainsail lists are dark UIs.
_COLORSCHEME = "Tomorrow Night"

_SIZE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*(?:/\s*([A-Za-z_]+))?\s*$")


def gcode_sizes(machine_profile: Optional[Path]) -> List[Tuple[int, int]]:
    """Thumbnail sizes the machine profile asks for, largest first, de-duplicated.

    Reads ``thumbnails`` (OrcaSlicer's key) then ``thumbnail_size`` (QIDIStudio's, which Orca
    renames on load) from the flattened ``machine.json``; each entry is ``"WxH"`` or
    ``"WxH/FORMAT"``, as a list or one comma-separated string. Falls back to OrcaSlicer's
    default ``48x48, 300x300``.
    """
    data: Dict[str, object] = {}
    if machine_profile is not None:
        try:
            loaded = json.loads(Path(machine_profile).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    for key in ("thumbnails", "thumbnail_size"):
        value = data.get(key)
        entries: Sequence[object]
        if isinstance(value, str):
            entries = value.split(",")
        elif isinstance(value, list):
            entries = [part for item in value for part in str(item).split(",")]
        else:
            continue
        sizes: List[Tuple[int, int]] = []
        for entry in entries:
            match = _SIZE_RE.match(str(entry))
            if match:
                size = (int(match.group(1)), int(match.group(2)))
                if size[0] > 0 and size[1] > 0 and size not in sizes:
                    sizes.append(size)
        if sizes:
            return sorted(sizes, key=lambda s: s[0] * s[1], reverse=True)
    return sorted(DEFAULT_GCODE_SIZES, key=lambda s: s[0] * s[1], reverse=True)


def render_pngs(stl: Path, sizes: Sequence[Tuple[int, int]], workdir: Path,
                timeout: float = 120.0) -> Dict[Tuple[int, int], bytes]:
    """Render ``stl`` once per size with OpenSCAD (headless, preview mode) and return the PNGs."""
    from phimech.backends.scad_emit import emit_file
    from phimech.csg.ast import Import
    from phimech.synth.preview import _openscad_binary

    if not Path(stl).is_file():  # OpenSCAD would warn and render an empty scene, exit 0
        raise FileNotFoundError("model not found: {}".format(stl))
    binary = _openscad_binary()
    scad = workdir / "thumbnail.scad"
    emit_file(Import(str(Path(stl).resolve())), scad)
    images: Dict[Tuple[int, int], bytes] = {}
    for width, height in sizes:
        out = workdir / "thumb-{}x{}.png".format(width, height)
        completed = subprocess.run(
            [binary, "-o", str(out), _CAMERA, "--projection=o", "--autocenter", "--viewall",
             "--imgsize={},{}".format(width, height), "--colorscheme={}".format(_COLORSCHEME),
             str(scad)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
        if completed.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            tail = completed.stdout.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise RuntimeError("OpenSCAD could not render a {}x{} thumbnail: {}".format(
                width, height, " | ".join(tail) or "no output"))
        images[(width, height)] = out.read_bytes()
    return images


def gcode_block(images: Dict[Tuple[int, int], bytes]) -> bytes:
    """The ``THUMBNAIL_BLOCK`` OrcaSlicer writes for PNG thumbnails, largest image first."""
    lines = ["; THUMBNAIL_BLOCK_START"]
    for (width, height), png in sorted(images.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
        encoded = base64.b64encode(png).decode("ascii")
        lines.append(";")
        lines.append("; thumbnail begin {}x{} {}".format(width, height, len(encoded)))
        for start in range(0, len(encoded), LINE_CHARS):
            lines.append("; " + encoded[start:start + LINE_CHARS])
        lines.append("; thumbnail end")
        lines.append(";")
    lines.append("; THUMBNAIL_BLOCK_END")
    return ("\n".join(lines) + "\n\n").encode("ascii")


def _insertion_offset(head: bytes) -> int:
    """Byte offset just after ``; HEADER_BLOCK_END`` and its blank line; 0 if there is none."""
    index = head.find(_HEADER_END)
    if index < 0:
        return 0
    newline = head.find(b"\n", index)
    if newline < 0:
        return len(head)
    offset = newline + 1
    if head[offset:offset + 1] == b"\n":
        offset += 1
    return offset


def has_thumbnail(head: bytes) -> bool:
    return _BEGIN_MARKER in head


def inject_stream(source, destination, block: bytes) -> bool:
    """Copy G-code from ``source`` to ``destination`` with ``block`` after the header.

    Streams, so a 500 MB file is never held in memory. Returns False (copying unchanged) when
    the G-code already carries a thumbnail.
    """
    head = source.read(_SEARCH_BYTES)
    if has_thumbnail(head):
        destination.write(head)
        shutil.copyfileobj(source, destination)
        return False
    offset = _insertion_offset(head)
    destination.write(head[:offset])
    destination.write(block)
    destination.write(head[offset:])
    shutil.copyfileobj(source, destination)
    return True


def inject_gcode_file(path: Path, block: bytes) -> bool:
    """Rewrite a bare ``.gcode`` file in place with ``block`` after its header."""
    path = Path(path)
    temporary = path.with_name(path.name + ".thumb-tmp")
    with path.open("rb") as source, temporary.open("wb") as destination:
        changed = inject_stream(source, destination, block)
    if changed:
        os.replace(str(temporary), str(path))
    else:
        temporary.unlink()
    return changed


class _HashingWriter:
    """File-like wrapper that MD5s whatever is written through it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.md5 = hashlib.md5()

    def write(self, data: bytes) -> int:
        self.md5.update(data)
        return self.inner.write(data)


def _with_thumbnail_metadata(text: str) -> str:
    """Add ``thumbnail_file`` to the first ``<plate>`` after ``gcode_file``, as OrcaSlicer does."""
    if 'key="thumbnail_file"' in text:
        return text
    match = re.search(r'([ \t]*)<metadata key="gcode_file" value="[^"]*"/>\n', text)
    if match is None:
        return text
    line = '{}<metadata key="thumbnail_file" value="{}"/>\n'.format(match.group(1), PLATE_PNG)
    return text[:match.end()] + line + text[match.end():]


def inject_3mf(path: Path, plate_png: bytes, small_png: bytes, block: bytes) -> bool:
    """Rebuild the ``.gcode.3mf`` with plate images, a thumbnailed G-code and its new MD5.

    A zip member cannot be replaced in place, so every other member is copied across with its
    original compression. Returns False, leaving the file untouched, when the archive has no
    ``Metadata/plate_1.gcode`` (not a sliced plate) or already has ``plate_1.png``.
    """
    path = Path(path)
    with zipfile.ZipFile(str(path)) as archive:
        names = archive.namelist()
        if GCODE_MEMBER not in names or PLATE_PNG in names:
            return False
    temporary = path.with_name(path.name + ".thumb-tmp")
    try:
        with zipfile.ZipFile(str(path)) as archive, \
                zipfile.ZipFile(str(temporary), "w", zipfile.ZIP_DEFLATED) as out:
            digest: Optional[str] = None
            for info in archive.infolist():
                if info.filename in (MD5_MEMBER, PLATE_SMALL_PNG):
                    continue
                if info.filename == GCODE_MEMBER:
                    target = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                    target.compress_type = info.compress_type
                    with archive.open(info) as source, out.open(target, "w",
                                                                force_zip64=True) as sink:
                        writer = _HashingWriter(sink)
                        inject_stream(source, writer, block)
                    digest = writer.md5.hexdigest().upper()
                    out.writestr(zipfile.ZipInfo(MD5_MEMBER, date_time=info.date_time),
                                 digest, compress_type=zipfile.ZIP_DEFLATED)
                    continue
                data = archive.read(info)
                if info.filename == MODEL_SETTINGS_MEMBER:
                    data = _with_thumbnail_metadata(data.decode("utf-8")).encode("utf-8")
                out.writestr(info, data, compress_type=info.compress_type)
            # OrcaSlicer stores its PNGs uncompressed (MZ_NO_COMPRESSION).
            out.writestr(PLATE_PNG, plate_png, compress_type=zipfile.ZIP_STORED)
            out.writestr(PLATE_SMALL_PNG, small_png, compress_type=zipfile.ZIP_STORED)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def add_to_slice(stl: Path, gcode_3mf: Path, gcode: Optional[Path],
                 machine_profile: Optional[Path]) -> List[str]:
    """Render thumbnails of ``stl`` and inject them into a finished slice. Never raises.

    ``gcode`` is the bare G-code copy beside the 3MF (``None`` if there is none). Returns
    warnings for the slice result; an empty list means every target got its images.
    """
    sizes = gcode_sizes(machine_profile)
    wanted = list(dict.fromkeys(list(sizes) + [PLATE_PNG_SIZE, PLATE_SMALL_PNG_SIZE]))
    try:
        with tempfile.TemporaryDirectory(prefix="phimech-thumb-") as scratch:
            images = render_pngs(Path(stl), wanted, Path(scratch))
    except Exception as exc:  # thumbnails are cosmetic; a slice must never fail on them
        return ["thumbnails skipped: {}".format(exc)]
    block = gcode_block({size: images[size] for size in sizes})
    warnings: List[str] = []
    try:
        if gcode is not None and Path(gcode).is_file():
            inject_gcode_file(Path(gcode), block)
    except Exception as exc:
        warnings.append("thumbnails not added to {}: {}".format(Path(gcode).name, exc))
    try:
        inject_3mf(Path(gcode_3mf), images[PLATE_PNG_SIZE], images[PLATE_SMALL_PNG_SIZE], block)
    except Exception as exc:
        warnings.append("thumbnails not added to {}: {}".format(Path(gcode_3mf).name, exc))
    return warnings


def read_gcode_thumbnails(head: bytes) -> List[Tuple[int, int, bytes]]:
    """Decode every ``; thumbnail begin WxH LEN`` block the way Moonraker does.

    Raises ``ValueError`` when a block's base64 length disagrees with its header - the check
    Moonraker applies before it trusts a thumbnail.
    """
    found: List[Tuple[int, int, bytes]] = []
    pattern = re.compile(rb"; thumbnail begin (\d+)x(\d+) (\d+)\n(.*?); thumbnail end",
                         re.DOTALL)
    for match in pattern.finditer(head):
        payload = b"".join(line[2:] if line.startswith(b"; ") else line.lstrip(b"; ")
                           for line in match.group(4).split(b"\n") if line.strip())
        if len(payload) != int(match.group(3)):
            raise ValueError("thumbnail {}x{}: base64 length {} != header {}".format(
                int(match.group(1)), int(match.group(2)), len(payload), int(match.group(3))))
        found.append((int(match.group(1)), int(match.group(2)), base64.b64decode(payload)))
    return found


__all__ = [
    "DEFAULT_GCODE_SIZES",
    "PLATE_PNG",
    "PLATE_SMALL_PNG",
    "add_to_slice",
    "gcode_block",
    "gcode_sizes",
    "inject_3mf",
    "inject_gcode_file",
    "read_gcode_thumbnails",
    "render_pngs",
]
