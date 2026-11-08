"""Static + JSON server behind the PhiMech 3D viewer.

Two responsibilities, deliberately kept small:

* serve the asset bundle (``assets/``) at ``/`` and the build output tree under ``/files/``;
* flatten ``report.json`` + ``design.json`` into one ``/api/manifest`` payload so the browser
  never has to know PhiMech's internal report shape.

Everything is stdlib (``http.server``, ``json``, ``urllib``) because this module has to import
under ``/usr/bin/python3`` 3.9 like the rest of the core.

The manifest endpoint never raises: a partial, stale, or missing report yields whatever could be
read plus a ``warnings`` array. A viewer that 500s on a half-finished build is useless precisely
when you most want to look at the geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import socket
import struct
import sys
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

#: The repository's ``docs/`` tree, served read-only at ``/docs/<path>`` so the documentation and
#: the tool live at one address. ``None`` when PhiMech is installed as a wheel, which does not ship
#: docs — the docs page then says so instead of 404-ing card by card.
def _find_docs_dir() -> Optional[Path]:
    """Walk up from this file looking for the repo's ``docs/``.

    Deliberately a search rather than a constant: the viewer runs from a checkout, from an
    editable install and from a wheel, and only the first two have docs next to the package.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs"
        if (candidate / "README.md").is_file():
            return candidate
    return None


DOCS_DIR = _find_docs_dir()

DOWNLOADS_DIR = None  # No bundled release; explicit --downloads only.

#: What ``/downloads/`` will hand out. Anything else in the directory (``.gitkeep``, a stray
#: build log) is a 404: the route publishes releases, it is not a file browser.
DOWNLOAD_TYPES = {
    ".zip": "application/zip",
    ".json": "application/json; charset=utf-8",
    ".sha256": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}

#: Curated order and blurbs for ``/docs``. Kept here rather than in the page so the list is data
#: the server can also serve as JSON — and so a doc that vanishes cannot leave a dead card.
DOC_PAGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("start", "README.md", "Start here", "Public setup and examples."),
)

#: Bed of the QiDi Max4 in mm. The viewer draws this as the printable envelope.
BED_SIZE_MM = (360.0, 360.0)
BED_HEIGHT_MM = 360.0

#: Density used to turn a measured volume into a mass when the design names no material.
#: Generic PLA, 1.24 g/cm^3. Reported alongside ``mass_basis`` so the number is never a mystery.
DEFAULT_DENSITY_G_CM3 = 1.24

MATERIAL_DENSITY_G_CM3 = {
    "pla": 1.24,
    "petg": 1.27,
    "abs": 1.04,
    "asa": 1.07,
    "tpu": 1.21,
    "nylon": 1.14,
    "pa": 1.14,
    "pc": 1.20,
    "resin": 1.10,
}

#: Neutral palette for distinguishing adjacent bodies.
PART_COLORS = [
    "#868C98",
    "#6194B8",
    "#CDD0D5",
    "#6194B8",
    "#525866",
    "#E2E4E9",
    "#6194B8",
    "#31353F",
]

MIME_TYPES = {
    ".stl": "model/stl",
    ".step": "model/step",
    ".stp": "model/step",
    ".3mf": "model/3mf",
    ".gcode": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".scad": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}

_PORT_SEARCH_ATTEMPTS = 40


# --------------------------------------------------------------------------------------------
# build discovery
# --------------------------------------------------------------------------------------------


def _is_build_dir(path: Path) -> bool:
    """A directory counts as a build when it holds a report, a design, or renderable geometry."""
    if not path.is_dir():
        return False
    if (path / "report.json").is_file() or (path / "design.json").is_file():
        return True
    try:
        return any(path.glob("*.stl"))
    except OSError:
        return False


def discover_builds(out_dir: Path) -> List[Dict[str, Any]]:
    """List the builds reachable from ``out_dir``.

    Supports both shapes the CLI produces: ``-o out/box`` (``out_dir`` *is* the build) and
    ``out/final`` (a parent holding several). A parent that is itself a build reports itself
    first, so pointing the viewer at either level always works.
    """
    out_dir = Path(out_dir)
    builds: List[Dict[str, Any]] = []
    if not out_dir.is_dir():
        return builds

    if _is_build_dir(out_dir):
        builds.append(_build_record(out_dir, out_dir, "."))

    try:
        children = sorted(p for p in out_dir.iterdir() if p.is_dir())
    except OSError:
        children = []
    for child in children:
        if child.name.startswith(".") or child.name == "vendor":
            continue
        if _is_build_dir(child):
            builds.append(_build_record(out_dir, child, child.name))
    return builds


def _build_record(root: Path, path: Path, build_id: str) -> Dict[str, Any]:
    """One entry for ``/api/builds``: enough to populate the switcher without a manifest fetch."""
    report = _read_json(path / "report.json") or {}
    stls = sorted(p.name for p in path.glob("*.stl"))
    ok = report.get("ok")
    return {
        "id": build_id,
        "name": str(report.get("project") or path.name),
        "title": str(report.get("title") or report.get("project") or path.name),
        "rev": report.get("rev"),
        "ok": ok if isinstance(ok, bool) else None,
        "timestamp": report.get("timestamp"),
        "has_report": (path / "report.json").is_file(),
        "part_count": len(stls),
        "path": str(path),
        "relpath": _relative_posix(root, path),
    }


def _relative_posix(root: Path, path: Path) -> str:
    """``path`` relative to ``root`` as a URL fragment; ``""`` when they are the same directory."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return ""
    return "" if str(rel) == "." else rel.as_posix()


def _resolve_build(out_dir: Path, build_id: Optional[str]) -> Tuple[Path, List[str]]:
    """Map a ``?build=`` id onto a directory, rejecting anything that escapes ``out_dir``.

    With no id, prefer the first real build UNDER ``out_dir`` when ``out_dir`` is itself only a
    container. Returning the container unconditionally meant pointing the viewer at a parent of
    several builds (the natural thing to do, and what ``bin/phimech view out/final`` does)
    opened an empty scene reading "no STL parts found" while /api/builds happily listed five.
    """
    warnings: List[str] = []
    out_dir = Path(out_dir)
    if not build_id or build_id in (".", ""):
        if not _is_build_dir(out_dir):
            children = discover_builds(out_dir)
            if children:
                first = out_dir / str(children[0]["relpath"])
                if first.is_dir():
                    return first, warnings
        return out_dir, warnings
    candidate = (out_dir / build_id).resolve()
    if not _within(out_dir.resolve(), candidate) or not candidate.is_dir():
        warnings.append("unknown build {!r}; showing {}".format(build_id, out_dir.name))
        return out_dir, warnings
    return candidate, warnings


# --------------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON, returning ``None`` for missing/corrupt files — callers add their own warning."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _num(value: Any) -> Optional[float]:
    """Coerce to float, mapping bools/NaN/strings-that-aren't-numbers to ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # NaN != NaN


def _parameters(report: Mapping[str, Any], design: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalise the editable parameters into a stable, sorted list of slider descriptors.

    Read from the report first and the design second: the report is what this build actually used,
    and the design on disk may already have been edited since. A parameter with no numeric value is
    dropped rather than surfaced as a broken control.

    ``min``/``max`` may be ``None``: that means the design declared no bound, and a UI should offer
    a number field rather than inventing a slider range.
    """
    source = report.get("parameters")
    if not isinstance(source, dict) or not source:
        source = design.get("parameters") if isinstance(design.get("parameters"), dict) else {}
    rows: List[Dict[str, Any]] = []
    for name in sorted(source or {}):
        spec = source[name]
        if not isinstance(spec, dict):
            continue
        value = _num(spec.get("value"))
        if value is None:
            continue
        unit = spec.get("unit")
        rows.append(
            {
                "name": str(name),
                "value": value,
                "min": _num(spec.get("min")),
                "max": _num(spec.get("max")),
                "step": _num(spec.get("step")),
                "unit": str(unit) if isinstance(unit, str) and unit else "mm",
                "description": str(spec.get("description") or ""),
            }
        )
    return rows


def _size3(value: Any) -> Optional[List[float]]:
    """Normalise a 3-vector of numbers, or ``None`` if it is not one."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = [_num(v) for v in value]
    if any(v is None for v in out):
        return None
    return [v for v in out if v is not None]


def _bbox(value: Any) -> Optional[Dict[str, List[float]]]:
    """Normalise a ``{min,max[,size]}`` bbox, deriving ``size`` when it is absent."""
    if not isinstance(value, dict):
        return None
    low, high = _size3(value.get("min")), _size3(value.get("max"))
    if low is None or high is None:
        return None
    size = _size3(value.get("size")) or [high[i] - low[i] for i in range(3)]
    return {"min": low, "max": high, "size": size}


def stl_triangle_count(path: Path) -> Optional[int]:
    """Triangle count of an STL, binary or ASCII, without loading the mesh.

    Binary STLs carry the count in the header, so this is a 84-byte read for the common case.
    The viewer shows the browser's own count once loaded; this is for the manifest and lets the
    tests assert a real number without a GPU.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(84)
            if len(header) < 84:
                return None
            declared = int(struct.unpack("<I", header[80:84])[0])
            looks_ascii = header[:5].lower().lstrip().startswith(b"solid")
            # A binary STL is exactly 84 + 50*n bytes. If the file says "solid" AND the size does
            # not match, it really is ASCII (OpenSCAD's output) and must be counted by facet.
            if looks_ascii and size != 84 + 50 * declared:
                handle.seek(0)
                count = 0
                for line in handle:
                    if line.lstrip()[:5].lower() == b"facet":
                        count += 1
                return count
            return declared
    except (OSError, struct.error):
        return None


def _density_for(design: Dict[str, Any]) -> Tuple[float, str]:
    """Pick a density (g/cm^3) from the design's material name, else generic PLA."""
    material = ""
    for key in ("manufacturing", "material", "print"):
        section = design.get(key)
        if isinstance(section, dict):
            for field in ("material", "filament", "name"):
                value = section.get(field)
                if isinstance(value, str) and value.strip():
                    material = value.strip().lower()
                    break
        elif isinstance(section, str) and section.strip():
            material = section.strip().lower()
        if material:
            break
    for name, density in MATERIAL_DENSITY_G_CM3.items():
        if name in material:
            return density, "{} @ {:.2f} g/cm3".format(name.upper(), density)
    return DEFAULT_DENSITY_G_CM3, "assumed PLA @ {:.2f} g/cm3".format(DEFAULT_DENSITY_G_CM3)


def _file_url(prefix: str, name: str) -> str:
    """URL under ``/files/`` for a build-relative filename."""
    parts = [p for p in (prefix, name) if p]
    return "/files/" + "/".join(parts)


def _collect_parts(
    build_dir: Path,
    prefix: str,
    report: Dict[str, Any],
    density: float,
    warnings: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split the report's parts into renderable STL parts and preview-only entries.

    ``report['parts']`` includes ``assembly``/``exploded`` pseudo-parts that only have SCAD/PNG
    output. Those must not reach the 3D scene (nothing to load) but are still worth showing as
    preview images, so they come back separately.
    """
    parts: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    raw_measurements = report.get("measurements")
    measurements = raw_measurements if isinstance(raw_measurements, dict) else {}
    reported = report.get("parts") if isinstance(report.get("parts"), dict) else {}

    names: List[str] = [str(n) for n in reported.keys()]
    # Any STL on disk that the report forgot (partial build, hand-dropped file) still renders.
    for stl in sorted(build_dir.glob("*.stl")):
        if not any(_part_stl(build_dir, reported.get(n), n) == stl for n in names):
            guess = _name_from_filename(stl, report)
            if guess not in names:
                names.append(guess)
                reported = dict(reported)
                reported[guess] = {"path": str(stl), "printable": True}
                warnings.append("STL {} was not in report.parts; added it".format(stl.name))

    # Colour index counts only renderable parts: letting the assembly/exploded pseudo-parts eat
    # palette slots leaves the real geometry on clashing colours further down the list.
    color_index = 0
    for name in names:
        entry = reported.get(name) if isinstance(reported.get(name), dict) else {}
        stl_path = _part_stl(build_dir, entry, name)
        png = _first_existing(build_dir, ["{}-{}.png".format(report.get("project", ""), name)])
        raw_measure = measurements.get(name)
        measurement = raw_measure if isinstance(raw_measure, dict) else {}
        bbox = _bbox(measurement.get("bbox"))
        solids = measurement.get("solids")
        size = _size3(entry.get("size")) or (bbox["size"] if bbox else None)
        volume = _num(measurement.get("volume_mm3"))
        if stl_path is not None:
            color = PART_COLORS[color_index % len(PART_COLORS)]
            color_index += 1
        else:
            color = "#525866"
        record: Dict[str, Any] = {
            "name": name,
            "color": color,
            "printable": (
                bool(entry.get("printable")) if "printable" in entry else stl_path is not None
            ),
            "stl": _file_url(prefix, stl_path.name) if stl_path else None,
            "png": _file_url(prefix, png.name) if png else None,
            "formats": [str(f) for f in entry.get("formats", []) if isinstance(f, str)],
            "bbox": bbox,
            "size_mm": size,
            "volume_mm3": volume,
            "volume_cm3": round(volume / 1000.0, 3) if volume is not None else None,
            "area_mm2": _num(measurement.get("area_mm2")),
            "solids": solids if isinstance(solids, int) else None,
            "mass_g": round(volume / 1000.0 * density, 2) if volume is not None else None,
            "triangles": stl_triangle_count(stl_path) if stl_path else None,
            "bytes": stl_path.stat().st_size if stl_path else None,
        }
        if stl_path is None:
            previews.append(
                {k: record[k] for k in ("name", "color", "png", "formats", "size_mm", "printable")}
            )
        else:
            parts.append(record)
    # The 7-view preview sheet `synth build` writes (synth/preview.py). It is a picture of the
    # main printable part for a human or vision model to *look* at - the wrong-shape detector -
    # so it belongs with the other rendered images, never in the scene.
    sheet = build_dir / "preview.png"
    if sheet.is_file():
        previews.append({
            "name": "sheet",
            "color": "#525866",
            "png": _file_url(prefix, sheet.name),
            "formats": ["png"],
            "size_mm": None,
            "printable": False,
        })
    return parts, previews


def _part_stl(build_dir: Path, entry: Any, name: str) -> Optional[Path]:
    """Locate a part's STL: the reported path first, then the conventional filename."""
    if isinstance(entry, dict):
        raw = entry.get("path")
        if isinstance(raw, str) and raw:
            candidate = Path(raw)
            if candidate.suffix.lower() != ".stl":
                candidate = candidate.with_suffix(".stl")
            # Reports store absolute paths from the build host; fall back to the local name.
            for probe in (candidate, build_dir / candidate.name):
                if probe.is_file():
                    return probe
    matches = sorted(build_dir.glob("*-{}.stl".format(name))) or sorted(
        build_dir.glob("{}.stl".format(name))
    )
    return matches[0] if matches else None


def _name_from_filename(stl: Path, report: Dict[str, Any]) -> str:
    """``sample-box-base.stl`` -> ``base`` when the project prefix is known."""
    stem = stl.stem
    project = str(report.get("project") or "")
    if project and stem.startswith(project + "-"):
        return stem[len(project) + 1 :]
    return stem


def _first_existing(build_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        if not name:
            continue
        candidate = build_dir / name
        if candidate.is_file():
            return candidate
    return None


def _collect_gates(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise gate results; ``passed`` stays tri-state (True/False/None-for-skipped)."""
    gates = report.get("gates")
    results = gates.get("results") if isinstance(gates, dict) else None
    if not isinstance(results, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        passed = item.get("passed")
        out.append(
            {
                "name": str(item.get("name", "gate")),
                "passed": passed if isinstance(passed, bool) else None,
                "value": _num(item.get("value")),
                "threshold": _num(item.get("threshold")),
                "severity": str(item.get("severity") or "error"),
                "message": str(item.get("message") or ""),
            }
        )
    return out


def _collect_print_info(
    build_dir: Path, prefix: str, report: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Slice/print facts if the print slice ran: time, filament, gcode/3mf links.

    Tolerant by design — the print pipeline is a sibling slice and its report keys may land after
    this one. Anything recognisable is surfaced; anything missing is simply absent.
    """
    info: Dict[str, Any] = {}
    for key in ("print", "slice", "slicing", "printing"):
        section = report.get(key)
        if isinstance(section, dict):
            info.update(section)
    for key in ("print_time_s", "print_time", "estimated_time_s", "filament_g", "filament_mm"):
        if key in report:
            info[key] = report[key]
    files: List[Dict[str, str]] = []
    for pattern in ("*.gcode", "*.gcode.3mf", "*.3mf"):
        for path in sorted(build_dir.glob(pattern)):
            files.append({"name": path.name, "url": _file_url(prefix, path.name),
                          "bytes": str(path.stat().st_size)})
    if files:
        info["files"] = files
    if not info:
        return None

    seconds = None
    for key in ("print_time_s", "estimated_time_s", "time_s", "print_time", "estimated_time"):
        seconds = _num(info.get(key))
        if seconds is not None:
            break
    grams = None
    for key in ("filament_g", "filament_grams", "material_g", "weight_g"):
        grams = _num(info.get(key))
        if grams is not None:
            break
    normalised = {"raw": info}
    if seconds is not None:
        normalised["print_time_s"] = seconds
        normalised["print_time_text"] = _format_duration(seconds)
    if grams is not None:
        normalised["filament_g"] = round(grams, 1)
    if files:
        normalised["files"] = files
    return normalised


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes = rest // 60
    if hours:
        return "{} h {:02d} min".format(hours, minutes)
    return "{} min".format(minutes) if minutes else "{} s".format(total)


def _collect_enclosure(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The enclosure summary, flattened for the info panel."""
    encl = report.get("enclosure")
    if not isinstance(encl, dict):
        return None
    return {
        "style": encl.get("style"),
        "fastening": encl.get("fastening"),
        "outer_size_mm": _size3(encl.get("outer_size_mm")),
        "cavity_size_mm": _size3(encl.get("cavity_size_mm")),
        "outer_bbox": _bbox(encl.get("outer_bbox")),
        "cavity_bbox": _bbox(encl.get("cavity_bbox")),
        "wall_mm": _num(encl.get("wall_mm")),
        "floor_mm": _num(encl.get("floor_mm")),
        "standoffs": encl.get("standoffs"),
        "screws": encl.get("screws"),
        "cutouts": encl.get("cutouts"),
        "world_offset": _size3(encl.get("world_offset")),
        "estimated_material_mm3": _num(encl.get("estimated_material_mm3")),
        "notes": [str(n) for n in encl.get("notes", []) if isinstance(n, str)],
        "parts": [str(p) for p in encl.get("parts", []) if isinstance(p, str)],
    }


def _collect_components(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = report.get("components")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "ref": str(item.get("ref") or "?"),
                "name": str(item.get("name") or ""),
                "size_mm": _size3(item.get("size")),
                "position_mm": _size3(item.get("position")),
                "rotation_deg": _size3(item.get("rotation")),
                "source": str(item.get("source") or ""),
            }
        )
    return out


def _collect_stages(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = report.get("stages")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "stage": str(item.get("stage") or "?"),
                "seconds": _num(item.get("seconds")),
                "ok": item.get("ok") if isinstance(item.get("ok"), bool) else None,
                "note": str(item.get("note") or ""),
            }
        )
    return out


def build_manifest(build_dir: Path, prefix: str = "") -> Dict[str, Any]:
    """Everything the UI needs for one build, in one payload.

    ``prefix`` is the build's path under ``/files/`` so returned URLs are ready to fetch. Always
    returns a dict; problems land in ``warnings`` rather than raising.
    """
    build_dir = Path(build_dir)
    warnings: List[str] = []

    report = _read_json(build_dir / "report.json")
    if report is None:
        report = {}
        if (build_dir / "report.json").exists():
            warnings.append("report.json is present but unreadable; showing geometry only")
        else:
            warnings.append("no report.json in {}; showing geometry only".format(build_dir.name))

    design = _read_json(build_dir / "design.json") or {}
    if not design:
        warnings.append("no design.json; material and manufacturing detail unavailable")

    density, mass_basis = _density_for(design)
    parts, previews = _collect_parts(build_dir, prefix, report, density, warnings)
    if not parts:
        warnings.append("no STL parts found in {}".format(build_dir.name))

    gates = _collect_gates(report)
    failed = [g["name"] for g in gates if g["passed"] is False]
    skipped = [g["name"] for g in gates if g["passed"] is None]
    ok = report.get("ok")
    if not isinstance(ok, bool):
        ok = None if not gates else not failed

    raw_mfg = design.get("manufacturing")
    manufacturing = raw_mfg if isinstance(raw_mfg, dict) else {}
    total_volume = sum(p["volume_mm3"] for p in parts if p["volume_mm3"] is not None)
    printable = [p for p in parts if p.get("printable")]

    manifest: Dict[str, Any] = {
        "schema": 1,
        "build_id": prefix or ".",
        "build_dir": str(build_dir),
        "project": str(report.get("project") or design.get("project") or build_dir.name),
        "title": str(report.get("title") or design.get("title") or build_dir.name),
        "company": str(report.get("company") or design.get("company") or ""),
        "rev": report.get("rev") or design.get("rev"),
        "generated": report.get("timestamp"),
        "served": datetime.now().replace(microsecond=0).isoformat(),
        "phimech_version": report.get("phimech_version"),
        "backend": report.get("backend"),
        "elapsed_s": _num(report.get("elapsed_s")),
        "ok": ok,
        "gates": gates,
        "gates_summary": {
            "total": len(gates),
            "passed": len([g for g in gates if g["passed"] is True]),
            "failed": len(failed),
            "skipped": len(skipped),
            "failed_names": failed,
        },
        "parts": parts,
        "previews": previews,
        "enclosure": _collect_enclosure(report),
        "components": _collect_components(report),
        "stages": _collect_stages(report),
        "notes": [str(n) for n in report.get("notes", []) if isinstance(n, str)],
        "errors": [str(e) for e in report.get("errors", []) if isinstance(e, str)],
        "print": _collect_print_info(build_dir, prefix, report),
        "totals": {
            "parts": len(parts),
            "printable_parts": len(printable),
            "volume_mm3": round(total_volume, 2) if parts else None,
            "volume_cm3": round(total_volume / 1000.0, 2) if parts else None,
            "mass_g": round(total_volume / 1000.0 * density, 1) if parts else None,
            "triangles": sum(p["triangles"] or 0 for p in parts) or None,
        },
        "material": {
            "density_g_cm3": density,
            "mass_basis": mass_basis,
            "process": manufacturing.get("process"),
            "nozzle_mm": _num(manufacturing.get("nozzle")),
            "layer_height_mm": _num(manufacturing.get("layer_height")),
            "min_wall_mm": _num(manufacturing.get("min_wall")),
        },
        "bed": {"size_mm": list(BED_SIZE_MM), "height_mm": BED_HEIGHT_MM, "name": "QiDi Max4"},
        # One entry per editable number, each with its bounds: a UI renders these as sliders. The
        # list is normalised here (name inlined, unit defaulted) so a client never has to guess.
        # Empty for a design that declares no parameters - render no controls rather than inventing
        # them, because a number the design did not declare editable has no safe bounds.
        "parameters": _parameters(report, design),
        "files": {
            "report_json": _file_url(prefix, "report.json")
            if (build_dir / "report.json").is_file()
            else None,
            "report_md": _file_url(prefix, "REPORT.md")
            if (build_dir / "REPORT.md").is_file()
            else None,
            "design_json": _file_url(prefix, "design.json")
            if (build_dir / "design.json").is_file()
            else None,
            # The production drawing, when the build ran with --drawing (synth build) or a
            # drawing was dropped beside report.json. Always present as keys, null when absent,
            # so a UI can render a "Drawing" tab conditionally without probing /files/.
            "drawing_svg": _file_url(prefix, "drawing.svg")
            if (build_dir / "drawing.svg").is_file()
            else None,
            "drawing_pdf": _file_url(prefix, "drawing.pdf")
            if (build_dir / "drawing.pdf").is_file()
            else None,
            "drawing_png": _file_url(prefix, "drawing.png")
            if (build_dir / "drawing.png").is_file()
            else None,
            "drawing_dxf": _file_url(prefix, "drawing.dxf")
            if (build_dir / "drawing.dxf").is_file()
            else None,
        },
        "warnings": warnings,
    }
    return manifest


# --------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------


def _within(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` is ``root`` or below it. The path-traversal gate."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    if candidate_resolved == root_resolved:
        return True
    return root_resolved in candidate_resolved.parents


def _safe_join(root: Path, url_path: str) -> Optional[Path]:
    """Resolve a URL path under ``root``, or ``None`` if it tries to escape.

    Rejects absolute paths, ``..`` segments, drive letters, and NUL — and then re-checks the
    resolved result against ``root`` so a symlink cannot smuggle you out either.
    """
    raw = unquote(url_path)
    if "\x00" in raw:
        return None
    parts: List[str] = []
    for segment in raw.replace("\\", "/").split("/"):
        if segment in ("", ".", ):
            continue
        if segment == ".." or ":" in segment:
            return None
        parts.append(segment)
    candidate = root.joinpath(*parts) if parts else root
    if not _within(root, candidate):
        return None
    return candidate


def _parse_range(header: Optional[str], size: int) -> Any:
    """``(start, end)`` inclusive for a single ``bytes=`` range, ``None`` to send the whole file,
    or ``"unsatisfiable"`` for a range that starts past the end.

    Multiple ranges, other units and malformed values all return ``None``: serving the full
    representation is always a valid answer to a Range request.
    """
    if not header:
        return None
    unit, _, spec = header.strip().partition("=")
    if unit.strip().lower() != "bytes" or "," in spec:
        return None
    first, dash, last = spec.strip().partition("-")
    if not dash or not (first.isdigit() or last.isdigit()):
        return None
    if not first.isdigit():
        # bytes=-N: the last N bytes.
        suffix = int(last)
        if suffix == 0:
            return "unsatisfiable"
        return (max(0, size - suffix), size - 1) if size else "unsatisfiable"
    start = int(first)
    if start >= size:
        return "unsatisfiable"
    end = min(int(last), size - 1) if last.isdigit() else size - 1
    if end < start:
        return None
    return (start, end)


def _guess_type(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


class ViewerHandler(BaseHTTPRequestHandler):
    """Request handler; ``out_dir`` is injected by :func:`serve` as a class attribute."""

    out_dir: Path = Path(".")
    downloads_dir: Optional[Path] = None
    quiet: bool = False
    server_version = "PhiMech/1.0"
    protocol_version = "HTTP/1.1"

    # -- logging: one line per request, and nothing at all for the asset bundle --------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        if self.quiet:
            return
        message = fmt % args
        if any(token in message for token in ("/vendor/", "/assets/", ".css", ".js", "favicon")):
            return
        sys.stderr.write("[viewer] {}\n".format(message))

    def log_error(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        if not self.quiet:
            sys.stderr.write("[viewer] {}\n".format(fmt % args))

    # -- routing ----------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        self._handle(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib signature
        self._handle(head_only=True)

    def _handle(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        path = posixpath.normpath(parsed.path or "/")
        query = _parse_query(parsed.query)
        try:
            if path == "/api/manifest":
                self._api_manifest(query, head_only)
            elif path == "/api/builds":
                self._api_builds(head_only)
            elif path == "/api/health":
                self._send_json({"ok": True, "out_dir": str(self.out_dir)}, head_only)
            elif path == "/api/docs":
                self._api_docs(head_only)
            elif path.startswith("/files"):
                self._serve_files(path[len("/files") :], head_only)
            elif path == "/docs" or path == "/docs/":
                self._serve_asset("/docs.html", head_only)
            elif path.startswith("/docs/"):
                self._serve_doc(path[len("/docs/") :], head_only)
            elif path == "/install":
                if parsed.path.endswith("/"):
                    # The page uses ./-relative links like every other page here, and from a
                    # trailing-slash URL those resolve under /install/. Redirect rather than
                    # serve a page whose download button points at /install/downloads/.
                    self._redirect("../install")
                else:
                    self._serve_asset("/install.html", head_only)
            elif (parsed.path == "/downloads" or parsed.path.startswith("/downloads/")
                  or path.startswith("/downloads/")):
                # Sliced from the RAW path, not the normalised one: normpath would quietly turn
                # /downloads/../x into /x and serve whatever that is. The guard must see (and
                # refuse) the attempt itself.
                raw = parsed.path
                name = raw[len("/downloads"):].lstrip("/") if raw.startswith("/downloads") else ""
                self._serve_download(name, head_only)
            elif path in ("/viewer", "/viewer/"):
                self._serve_asset("/index.html", head_only)
            else:
                self._serve_asset(path, head_only)
        except BrokenPipeError:  # pragma: no cover - browser navigated away mid-response
            pass
        except Exception as exc:  # pragma: no cover - last-resort guard, never leak a traceback
            self._send_json({"error": "{}: {}".format(type(exc).__name__, exc)}, head_only,
                            status=HTTPStatus.INTERNAL_SERVER_ERROR)

    # -- endpoints --------------------------------------------------------------------------
    def _api_manifest(self, query: Dict[str, str], head_only: bool) -> None:
        build_dir, warnings = _resolve_build(self.out_dir, query.get("build"))
        prefix = _relative_posix(self.out_dir, build_dir)
        manifest = build_manifest(build_dir, prefix)
        manifest["warnings"] = warnings + list(manifest.get("warnings", []))
        manifest["builds"] = discover_builds(self.out_dir)
        self._send_json(manifest, head_only)

    def _api_builds(self, head_only: bool) -> None:
        builds = discover_builds(self.out_dir)
        self._send_json(
            {"out_dir": str(self.out_dir), "count": len(builds), "builds": builds}, head_only
        )

    def _api_docs(self, head_only: bool) -> None:
        """The documentation index as data: title, blurb, and whether the file is actually here."""
        pages = []
        for slug, rel, title, blurb in DOC_PAGES:
            target = _safe_join(DOCS_DIR, rel) if DOCS_DIR else None
            pages.append({
                "slug": slug,
                "title": title,
                "blurb": blurb,
                "path": rel,
                "url": "/docs/" + rel,
                "available": bool(target and target.is_file()),
            })
        self._send_json({
            "schema": 1,
            "docs_dir": str(DOCS_DIR) if DOCS_DIR else None,
            "count": sum(1 for page in pages if page["available"]),
            "pages": pages,
        }, head_only)

    def _serve_doc(self, rel: str, head_only: bool) -> None:
        """Serve one markdown file from the repo ``docs/`` tree, read-only.

        Markdown is sent as text and rendered in the browser. That keeps the server stdlib-only
        (rule 9) and means the file on disk stays the single source of truth — no build step can
        make the served docs disagree with the repo.
        """
        if DOCS_DIR is None:
            self._send_error_json(HTTPStatus.NOT_FOUND,
                                  "this install has no docs/ tree; read them in the repository")
            return
        target = _safe_join(DOCS_DIR, rel)
        if target is None:
            self._send_error_json(HTTPStatus.FORBIDDEN, "path outside the docs directory")
            return
        if target.is_dir():
            target = target / "README.md"
        if not target.is_file() or target.suffix.lower() not in (".md", ".png", ".svg", ".jpg"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "no such doc: {}".format(rel))
            return
        self._send_file(target, head_only)

    def _serve_files(self, rel: str, head_only: bool) -> None:
        target = _safe_join(self.out_dir, rel)
        if target is None:
            self._send_error_json(HTTPStatus.FORBIDDEN, "path outside the build directory")
            return
        if target.is_dir():
            listing = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            self._send_json({"dir": rel or ".", "entries": listing}, head_only)
            return
        if not target.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "no such file: {}".format(rel.lstrip("/")))
            return
        self._send_file(target, head_only)

    def _serve_download(self, name: str, head_only: bool) -> None:
        """Serve one release file from the downloads directory, read-only.

        Flat names only, a fixed set of types, and every refusal is the same 404: a traversal
        attempt learns nothing a typo would not. The zip is streamed rather than read whole,
        because an installer bundle is the one file here that can be large.
        """
        root = self.downloads_dir
        target = _safe_join(root, name) if (root is not None and name) else None
        if (root is None or target is None or target.parent.resolve() != root.resolve()
                or target.name.startswith(".") or not target.is_file()
                or target.suffix.lower() not in DOWNLOAD_TYPES):
            self._send_error_json(HTTPStatus.NOT_FOUND,
                                  "no such download: {}".format(unquote(name) or "(none)"))
            return
        suffix = target.suffix.lower()
        try:
            size = target.stat().st_size
            handle = target.open("rb")
        except OSError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "no such download: {}".format(name))
            return
        with handle:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", DOWNLOAD_TYPES[suffix])
            self.send_header("Content-Length", str(size))
            if suffix == ".zip":
                self.send_header("Content-Disposition",
                                 'attachment; filename="{}"'.format(target.name.replace('"', "")))
            # latest.json must never be stale: it is how the install page finds the new zip. The
            # zips get the same header so a re-published file of the same name is re-validated.
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head_only:
                shutil.copyfileobj(handle, self.wfile, 1024 * 1024)

    def _serve_asset(self, path: str, head_only: bool) -> None:
        # `/` is the home page and `/viewer` is the workspace shell. This used to be one page; the
        # split exists so a first-time visitor lands on something that explains the tool, while
        # the workspace keeps a stable address of its own.
        rel = "home.html" if path in ("/", "") else path.lstrip("/")
        target = _safe_join(ASSETS_DIR, rel)
        if target is None:
            self._send_error_json(HTTPStatus.FORBIDDEN, "bad asset path")
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "no such asset: {}".format(rel))
            return
        self._send_file(target, head_only)

    # -- response helpers -------------------------------------------------------------------
    def _send_file(self, path: Path, head_only: bool) -> None:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, "unreadable: {}".format(exc))
            return
        # One byte range, because Safari will not play a <video> whose server ignores Range (the
        # home page's demo). Anything this cannot parse falls back to the whole file, which RFC
        # 9110 allows; a range starting past the end is a 416.
        byte_range = _parse_range(self.headers.get("Range"), len(payload))
        if byte_range == "unsatisfiable":
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", "bytes */{}".format(len(payload)))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = HTTPStatus.OK
        body = payload
        if byte_range is not None:
            start, end = byte_range
            status = HTTPStatus.PARTIAL_CONTENT
            body = payload[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Type", _guess_type(path))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if byte_range is not None:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, len(payload)))
        # No caching: the whole point is to re-read a build you just re-ran.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(self, payload: Any, head_only: bool, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=1, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message, "status": int(status)}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _parse_query(query: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in query.split("&"):
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        out[unquote(key)] = unquote(value.replace("+", " "))
    return out


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def serve(
    out_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    quiet: bool = False,
    downloads_dir: Optional[Path] = None,
) -> Tuple[ThreadingHTTPServer, str]:
    """Start the viewer server and return ``(httpd, url)`` without blocking.

    Walks upward from ``port`` when it is taken (and finally asks the OS for any free port), so
    two viewers can run side by side. The returned URL always names the port actually bound —
    never assume the requested one.

    Downloads are disabled by default. Pass ``downloads_dir`` explicitly to serve local files.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        raise NotADirectoryError("out_dir does not exist: {}".format(out_dir))

    downloads = (Path(downloads_dir).expanduser().resolve() if downloads_dir is not None
                 else DOWNLOADS_DIR)
    handler = type(
        "BoundViewerHandler", (ViewerHandler,),
        {"out_dir": out_dir, "quiet": bool(quiet), "downloads_dir": downloads},
    )

    httpd: Optional[ThreadingHTTPServer] = None
    candidates = [port + offset for offset in range(_PORT_SEARCH_ATTEMPTS)] if port else []
    candidates.append(0)
    last_error: Optional[OSError] = None
    for candidate in candidates:
        try:
            httpd = ThreadingHTTPServer((host, candidate), handler)  # type: ignore[arg-type]
            break
        except OSError as exc:
            last_error = exc
            continue
    if httpd is None:  # pragma: no cover - only if even port 0 fails
        raise OSError("could not bind a port on {}: {}".format(host, last_error))

    httpd.daemon_threads = True
    actual_port = httpd.server_address[1]
    url = "http://{}:{}/".format(host, actual_port)
    if port and actual_port != port and not quiet:
        sys.stderr.write("[viewer] port {} busy; using {}\n".format(port, actual_port))
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    return httpd, url


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Blocking CLI entry point: ``python -m phimech.viewer.server out/final``."""
    parser = argparse.ArgumentParser(
        prog="phimech-viewer",
        description="Serve the PhiMech 3D viewer over a build output directory.",
    )
    parser.add_argument("out_dir", nargs="?", default="out",
                        help="build directory, or a parent holding several (default: out)")
    parser.add_argument("--host", default=os.environ.get("PHIMECH_VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PHIMECH_VIEWER_PORT", "8765")))
    parser.add_argument("--open", dest="open_browser", action="store_true",
                        help="open the URL in the default browser")
    parser.add_argument("--quiet", action="store_true", help="suppress per-request logging")
    parser.add_argument("--downloads", default=os.environ.get("PHIMECH_DOWNLOADS_DIR"),
                        help="directory served at /downloads/ "
                             "(disabled by default)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        httpd, url = serve(Path(args.out_dir), args.host, args.port, args.open_browser, args.quiet,
                           downloads_dir=Path(args.downloads) if args.downloads else None)
    except (NotADirectoryError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2

    builds = discover_builds(Path(args.out_dir).expanduser().resolve())
    sys.stderr.write("[viewer] serving {} ({} build(s)) at {}\n".format(
        Path(args.out_dir).resolve(), len(builds), url))
    for build in builds:
        sys.stderr.write("[viewer]   {:<24} {} part(s){}\n".format(
            build["id"], build["part_count"], "" if build["ok"] is not False else "  GATES FAILED"))
    sys.stderr.write("[viewer] Ctrl-C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[viewer] stopped\n")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
