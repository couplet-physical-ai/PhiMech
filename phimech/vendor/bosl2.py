"""BOSL2 glue: where the vendored library is, which includes are legal, what is worth knowing.

This module is the single place that knows the layout of `phimech/vendor/BOSL2/`. Three
consumers share it so they cannot drift apart:

* `phimech.csg.validate` — `resolve_include()` decides whether a `scad_module` node's `include`
  names a vendored file (and refuses anything that would escape `phimech/vendor/`).
* `phimech.backends.scad_emit` / `phimech.backends.openscad` — `required_includes()` gives the
  include lines to emit and `VENDOR_DIR` is what goes on `OPENSCADPATH`.
* `phimech.mcp.tools_bosl2` and the doctor — `include_files()`, `CURATED_MODULES`, `pinned()`
  and `doctor_section()` describe the copy to an agent or a human.

Nothing here reads BOSL2 source to *interpret* it. The curated table is hand-written from the
upstream documentation and is checked by `tests/test_bosl2.py` against `module <name>(` lines in
the vendored files, so a refresh that renames a module fails the gate instead of the agent.

stdlib only; imports under python 3.9.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phimech.vendor import VENDOR_DIR

#: Library directory, and the include every other BOSL2 file documents as its prerequisite.
LIBRARY = "BOSL2"
BOSL2_DIR: Path = VENDOR_DIR / LIBRARY
STD_INCLUDE = LIBRARY + "/std.scad"
VENDORED_MD: Path = BOSL2_DIR / "VENDORED.md"

#: An OpenSCAD module name. Anything else (a call, a `;`, whitespace) would let a node inject
#: source, so the validator rejects it before the emitter ever sees it.
MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: A named-argument key. `$fn`, `$fa`, `$fs` are legal named arguments in OpenSCAD and are how a
#: caller controls a BOSL2 module's tessellation, hence the optional leading `$`.
ARG_NAME = re.compile(r"^\$?[A-Za-z_][A-Za-z0-9_]*$")

#: The characters an include path may contain. `<` and `>` delimit OpenSCAD's include syntax,
#: so they can never appear inside one; backslashes and whitespace are refused to keep the line
#: unambiguous on every platform.
INCLUDE_TEXT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.scad$")


class IncludeError(ValueError):
    """An `include` string that must not reach OpenSCAD, with the reason in the message."""


def resolve_include(include: Any) -> Path:
    """Return the vendored file an ``include <…>`` line names, or raise ``IncludeError``.

    The string is checked as text first (allowed characters, ``.scad`` suffix, no ``..`` or
    ``.`` segments, not absolute), then resolved and required to sit inside ``phimech/vendor``,
    then required to exist. The last check is deliberate: OpenSCAD only *warns* about a missing
    include and then fails on the unknown module with a message that never mentions the include,
    so catching it here is the difference between a one-line fix and a hunt.
    """
    if not isinstance(include, str) or not include:
        raise IncludeError("include must be a non-empty string such as {!r}".format(STD_INCLUDE))
    if not INCLUDE_TEXT.match(include):
        raise IncludeError(
            "include {!r} must be a relative path of letters, digits, '_', '-', '.', '/' ending in "
            ".scad, for example {!r}".format(include, STD_INCLUDE)
        )
    parts = include.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise IncludeError(
            "include {!r} must not contain empty, '.' or '..' path segments; it may not leave "
            "phimech/vendor".format(include)
        )
    root = VENDOR_DIR.resolve()
    candidate = (root / include).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise IncludeError(
            "include {!r} resolves outside phimech/vendor ({}); only vendored libraries may be "
            "included".format(include, candidate)
        ) from None
    if not candidate.is_file():
        raise IncludeError(
            "include {!r} is not a vendored library file (looked for {}); available: {}".format(
                include, candidate, ", ".join(include_files()[:6]) + ", …"
            )
        )
    return candidate


def required_includes(include: str) -> List[str]:
    """The include lines a node's ``include`` implies, in emission order.

    BOSL2 documents ``include <BOSL2/std.scad>`` as a prerequisite of every other file
    (``threading.scad`` echoes a warning without it and its dependencies are missing), so any
    ``BOSL2/…`` include is preceded by ``std.scad``. Other libraries get just themselves.
    """
    if include == STD_INCLUDE or not include.startswith(LIBRARY + "/"):
        return [include]
    return [STD_INCLUDE, include]


def include_files() -> List[str]:
    """Every vendored library file, as the string a node's ``include`` field would hold."""
    if not BOSL2_DIR.is_dir():
        return []
    return sorted(
        "{}/{}".format(LIBRARY, path.name) for path in BOSL2_DIR.iterdir()
        if path.suffix == ".scad" and path.is_file()
    )


def vendored_size_bytes() -> int:
    if not BOSL2_DIR.is_dir():
        return 0
    return sum(path.stat().st_size for path in BOSL2_DIR.iterdir() if path.is_file())


_PIN_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("commit", r"^\|\s*Commit\s*\|\s*`([0-9a-f]{40})`"),
    ("commit_date", r"^\|\s*Commit date\s*\|\s*([0-9T:+\-]+)"),
    ("version", r"BOSL_VERSION\s*=\s*(\[[0-9, ]+\])"),
)


def pinned() -> Dict[str, Optional[str]]:
    """The upstream pin recorded in ``VENDORED.md`` (commit, date, BOSL_VERSION), or Nones."""
    result: Dict[str, Optional[str]] = {"commit": None, "commit_date": None, "version": None}
    if not VENDORED_MD.is_file():
        return result
    text = VENDORED_MD.read_text(encoding="utf-8")
    for key, pattern in _PIN_FIELDS:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            result[key] = match.group(1).strip()
    return result


def openscadpath(existing: Optional[str] = None) -> str:
    """The ``OPENSCADPATH`` value the OpenSCAD backend passes to its subprocess.

    Ours goes first so a user's own BOSL2 checkout cannot silently shadow the pinned copy the
    gate was measured against; theirs is kept behind it. ``os.pathsep`` is OpenSCAD's separator
    (``:`` on POSIX, ``;`` on Windows).
    """
    ours = str(VENDOR_DIR)
    if not existing:
        return ours
    kept = [entry for entry in existing.split(os.pathsep) if entry and entry != ours]
    return os.pathsep.join([ours] + kept)


def defines_module(module: str) -> Optional[str]:
    """Which vendored file defines ``module <name>(``, or None. A source scan, not an evaluation."""
    if not MODULE_NAME.match(module or "") or not BOSL2_DIR.is_dir():
        return None
    pattern = re.compile(r"^\s*module\s+" + re.escape(module) + r"\s*\(", re.MULTILINE)
    for path in sorted(BOSL2_DIR.glob("*.scad")):
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                return "{}/{}".format(LIBRARY, path.name)
        except OSError:
            continue
    return None


#: The modules an agent reaches for most, with the arguments that matter. Units are mm and
#: degrees throughout; every entry names the include the node needs. Hand-curated from upstream
#: docs; `tests/test_bosl2.py` proves each `module` is defined in the vendored source.
CURATED_MODULES: Tuple[Dict[str, Any], ...] = (
    {
        "module": "threaded_rod", "include": "BOSL2/threading.scad",
        "summary": "External ISO/UTS thread. M3 = d=3, pitch=0.5; gate-measured Ø3.0 × 12.0 mm.",
        "args": {"d": "major diameter mm", "l": "length mm", "pitch": "thread pitch mm",
                 "internal": "true to make a cutter for a tapped hole",
                 "bevel": "true to bevel both ends", "$fn": "segments per turn (48 is smooth)"},
    },
    {
        "module": "threaded_nut", "include": "BOSL2/threading.scad",
        "summary": "Hex nut with an internal ISO thread matching threaded_rod.",
        "args": {"nutwidth": "flat-to-flat mm", "id": "thread major diameter mm",
                 "h": "height mm", "pitch": "thread pitch mm", "shape": "\"hex\" or \"square\""},
    },
    {
        "module": "screw", "include": "BOSL2/screws.scad",
        "summary": "A metric or UTS screw from a spec string, e.g. \"M3x8\", with head and drive.",
        "args": {"spec": "\"M3x8\", \"M2.5\", \"#6-32,1/2\"", "head": "\"socket\", \"pan\", \"flat\", \"hex\"…",
                 "drive": "\"hex\", \"torx\", \"phillips\", \"slot\"", "length": "shaft length mm",
                 "thread": "false for an unthreaded shaft (fast)"},
    },
    {
        "module": "screw_hole", "include": "BOSL2/screws.scad",
        "summary": "The matching hole/counterbore cutter — subtract it with a Difference node.",
        "args": {"spec": "same spec string as screw()", "head": "head recess to cut",
                 "length": "hole depth mm", "thread": "true to cut real threads",
                 "tolerance": "\"close\", \"normal\", \"loose\" or a number mm"},
    },
    {
        "module": "nut", "include": "BOSL2/screws.scad",
        "summary": "A standard nut for a spec string.",
        "args": {"spec": "\"M3\"", "shape": "\"hex\" or \"square\"", "thickness": "mm (default from spec)"},
    },
    {
        "module": "cuboid", "include": "BOSL2/std.scad",
        "summary": "Centred box with rounded or chamfered edges — the rounded_cube of BOSL2.",
        "args": {"size": "[x, y, z] mm", "rounding": "edge radius mm", "chamfer": "edge chamfer mm",
                 "edges": "which edges, e.g. \"Z\" for the vertical ones", "anchor": "e.g. BOTTOM"},
    },
    {
        "module": "cyl", "include": "BOSL2/std.scad",
        "summary": "Centred cylinder/cone with per-end chamfer or rounding.",
        "args": {"h": "height mm", "d": "diameter mm (or d1/d2 for a cone)",
                 "chamfer": "both ends mm", "chamfer1": "bottom mm", "chamfer2": "top mm",
                 "rounding": "both ends mm", "$fn": "segments"},
    },
    {
        "module": "tube", "include": "BOSL2/std.scad",
        "summary": "Hollow cylinder from outer diameter and wall or inner diameter.",
        "args": {"h": "height mm", "od": "outer diameter mm", "id": "inner diameter mm",
                 "wall": "wall thickness mm (instead of id)"},
    },
    {
        "module": "prismoid", "include": "BOSL2/std.scad",
        "summary": "Rectangular frustum: different bottom and top sizes, optional rounding.",
        "args": {"size1": "[x, y] at the bottom mm", "size2": "[x, y] at the top mm", "h": "height mm",
                 "shift": "[x, y] offset of the top mm", "rounding": "vertical edge radius mm"},
    },
    {
        "module": "hull_points", "include": "BOSL2/std.scad",
        "summary": "Convex hull of an explicit 3D point list (the core AST's Hull hulls children).",
        "args": {"points": "[[x, y, z], …] mm", "fast": "true skips the degenerate-input checks"},
    },
    {
        "module": "linear_sweep", "include": "BOSL2/std.scad",
        "summary": "Extrude a 2D polygon given as a point list, with twist, scale or texture.",
        "args": {"region": "[[x, y], …] mm — the profile, as points, not a child",
                 "height": "mm", "twist": "degrees", "scale": "factor at the top",
                 "texture": "a texture name, e.g. \"diamonds\""},
    },
    {
        "module": "path_sweep", "include": "BOSL2/std.scad",
        "summary": "Sweep a 2D profile (points) along a 2D/3D path (points). The swept-handle example.",
        "args": {"shape": "[[x, y], …] mm profile", "path": "[[x, y(, z)], …] mm centreline",
                 "closed": "true for a loop", "caps": "true to close the ends of an open sweep",
                 "method": "\"incremental\" (default), \"natural\", \"manual\""},
    },
    {
        "module": "offset_sweep", "include": "BOSL2/rounding.scad",
        "summary": "Extrude a 2D outline with rounded/chamfered top and bottom edges.",
        "args": {"path": "[[x, y], …] mm outline", "height": "mm",
                 "bottom": "an os_*() struct as a literal list, e.g. [[\"type\",\"circle\"],[\"r\",2]]",
                 "top": "same shape as bottom"},
    },
    {
        "module": "rounded_prism", "include": "BOSL2/rounding.scad",
        "summary": "Prism between two polygons with continuous-curvature rounded edges.",
        "args": {"bottom": "[[x, y], …] mm", "top": "[[x, y], …] mm (optional)", "height": "mm",
                 "joint_top": "rounding size at the top mm", "joint_bot": "bottom mm",
                 "joint_sides": "vertical edges mm"},
    },
    {
        "module": "skin", "include": "BOSL2/std.scad",
        "summary": "Loft a solid through a list of 3D profiles (each a point list).",
        "args": {"profiles": "[[[x, y, z], …], …] mm", "slices": "interpolated steps between profiles",
                 "method": "\"direct\", \"reindex\", \"distance\", \"tangent\"",
                 "caps": "true to close the ends"},
    },
)


def doctor_section() -> Dict[str, Any]:
    """JSON-serialisable health of the vendored copy, for `phimech doctor` to wire in.

    `ok` means the copy is usable by the OpenSCAD backend: the directory exists, `std.scad` is
    present, the licence travelled with it, and the pin is recorded. It says nothing about
    whether OpenSCAD itself is installed — that is the backend's own section.
    """
    files = include_files()
    pin = pinned()
    problems: List[str] = []
    if not BOSL2_DIR.is_dir():
        problems.append("missing {}".format(BOSL2_DIR))
    if STD_INCLUDE not in files:
        problems.append("std.scad is not vendored")
    if not (BOSL2_DIR / "LICENSE").is_file():
        problems.append("LICENSE is missing beside the library")
    if not pin.get("commit"):
        problems.append("VENDORED.md does not pin an upstream commit")
    return {
        "ok": not problems,
        "library": LIBRARY,
        "dir": str(BOSL2_DIR),
        "files": len(files),
        "size_bytes": vendored_size_bytes(),
        "commit": pin.get("commit"),
        "commit_date": pin.get("commit_date"),
        "version": pin.get("version"),
        "openscadpath": str(VENDOR_DIR),
        "problems": problems,
    }


__all__ = [
    "ARG_NAME",
    "BOSL2_DIR",
    "CURATED_MODULES",
    "INCLUDE_TEXT",
    "IncludeError",
    "LIBRARY",
    "MODULE_NAME",
    "STD_INCLUDE",
    "VENDOR_DIR",
    "defines_module",
    "doctor_section",
    "include_files",
    "openscadpath",
    "pinned",
    "required_includes",
    "resolve_include",
    "vendored_size_bytes",
]
