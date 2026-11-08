"""Read-only Moonraker/Klipper client for the QiDi Max 4 — slice, upload, query. Never print.

SAFETY CONTRACT (non-negotiable, enforced by a test that greps this module's own source):

    This module NEVER starts, pauses, resumes, or cancels a print.  It NEVER sets a temperature.
    It NEVER homes or moves an axis.  It NEVER runs a G-code command on the printer.

    It does exactly three things: query read-only state, upload a file, and list files.  Uploads
    are submitted with the "do not print" flag; the file simply lands in the printer's `gcodes`
    root for a human to pick.

    Starting a print belongs to the separate `qidi-mcp` server, behind its own two-step confirm
    token.  Do not add a start/cancel/temperature/motion helper here "for convenience" — the
    interlocks and the audit log live in that server, and duplicating the capability here would
    route plastic-moving commands around them.

Everything is stdlib ``urllib``.  Two operational details worth knowing:

* **Proxies are explicitly disabled.**  The printer is a LAN host; a proxy can interfere with local device requests.  Every request is
  built with an empty :class:`urllib.request.ProxyHandler`, so this works regardless of the
  ambient ``HTTPS_PROXY``.
* **Nothing here raises on a network failure.**  A missing printer must degrade a report, not
  crash a build, so queries return ``reachable: False`` with the error text.

The default printer volume is a generic example; configure your own machine before use.

"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_PRINTER_URL = "http://127.0.0.1:7125"

# The safe part envelope, in mm. Derived from the bed mesh area (15…375 in X and Y) and the Z
# axis maximum, NOT from the axis minima/maxima: the toolhead can reach past the mesh, but a part
# printed out there is not on measured bed.
BUILD_VOLUME_MM: Tuple[float, float, float] = (360.0, 360.0, 340.0)

GCODE_ROOT = "gcodes"

_TIMEOUT_DEFAULT = 10


def resolve_url(explicit: Optional[str] = None) -> str:
    """Resolve the Moonraker base URL: argument, ``$PHIMECH_PRINTER_URL``, then localhost."""
    value = explicit or os.environ.get("PHIMECH_PRINTER_URL") or DEFAULT_PRINTER_URL
    return value.rstrip("/")


def _opener() -> urllib.request.OpenerDirector:
    """An opener with proxies disabled — the printer is a LAN host."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(
    url: str,
    timeout: int = _TIMEOUT_DEFAULT,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
) -> Dict[str, object]:
    """Perform one HTTP request and JSON-decode the body. Raises on failure; callers wrap it."""
    request = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with _opener().open(request, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        return {}
    try:
        decoded = json.loads(payload.decode("utf-8", "replace"))
    except ValueError:
        return {"raw": payload.decode("utf-8", "replace")}
    return decoded if isinstance(decoded, dict) else {"result": decoded}


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP {} {}".format(exc.code, exc.reason)
    if isinstance(exc, urllib.error.URLError):
        return "{}".format(exc.reason)
    return "{}: {}".format(type(exc).__name__, exc)


def _get(path: str, url: str, timeout: int) -> Dict[str, object]:
    return _request("{}{}".format(url, path), timeout=timeout)


def printer_info(
    base_url: Optional[str] = None, timeout: int = _TIMEOUT_DEFAULT
) -> Dict[str, object]:
    """Query the printer's identity, axis limits, and temperature ceilings. Never raises.

    On any network failure returns ``{"reachable": False, "error": "<why>", ...}`` with the
    documented envelope still filled in, so a caller can gate a design against the build volume
    without the printer being on.
    """
    url = resolve_url(base_url)
    info: Dict[str, object] = {
        "url": url,
        "reachable": False,
        "klippy_state": None,
        "hostname": None,
        "software_version": None,
        "axis_minimum": None,
        "axis_maximum": None,
        "build_volume_mm": list(BUILD_VOLUME_MM),
        # Provenance of `build_volume_mm`, because a gate must never present a compiled-in
        # constant as a machine reading. "documented" = the module constant below, which is
        # what an unreachable printer leaves in place; "queried" = derived from this machine's
        # own bed mesh and Z axis in this call.
        "build_volume_source": "documented",
        "bed_mesh": None,
        "nozzle_mm": None,
        "max_temps": {},
        "roots": [],
        "error": None,
    }
    try:
        result = _get("/printer/info", url, timeout).get("result", {})
    except Exception as exc:  # noqa: BLE001 - a missing printer must not raise
        info["error"] = _error_text(exc)
        return info
    if not isinstance(result, dict):
        info["error"] = "unexpected /printer/info payload"
        return info
    info["reachable"] = True
    info["klippy_state"] = result.get("state")
    info["state_message"] = result.get("state_message")
    info["hostname"] = result.get("hostname")
    info["software_version"] = result.get("software_version")

    try:
        query = _get("/printer/objects/query?toolhead", url, timeout)
        toolhead = query.get("result", {}).get("status", {}).get("toolhead", {})
        if isinstance(toolhead, dict):
            minimum = toolhead.get("axis_minimum")
            maximum = toolhead.get("axis_maximum")
            if isinstance(minimum, list):
                info["axis_minimum"] = [float(v) for v in minimum[:3]]
            if isinstance(maximum, list):
                info["axis_maximum"] = [float(v) for v in maximum[:3]]
            for key in ("homed_axes", "max_velocity", "max_accel"):
                if key in toolhead:
                    info[key] = toolhead[key]
    except Exception as exc:  # noqa: BLE001
        info["error"] = _error_text(exc)

    try:
        settings = (
            _get("/printer/objects/query?configfile=settings", url, timeout)
            .get("result", {})
            .get("status", {})
            .get("configfile", {})
            .get("settings", {})
        )
        if isinstance(settings, dict):
            temps: Dict[str, float] = {}
            extruder = settings.get("extruder")
            if isinstance(extruder, dict):
                if extruder.get("max_temp") is not None:
                    temps["extruder_c"] = float(extruder["max_temp"])
                if extruder.get("nozzle_diameter") is not None:
                    info["nozzle_mm"] = float(extruder["nozzle_diameter"])
                if extruder.get("filament_diameter") is not None:
                    info["filament_diameter_mm"] = float(extruder["filament_diameter"])
                if extruder.get("min_extrude_temp") is not None:
                    temps["min_extrude_c"] = float(extruder["min_extrude_temp"])
            bed = settings.get("heater_bed")
            if isinstance(bed, dict) and bed.get("max_temp") is not None:
                temps["bed_c"] = float(bed["max_temp"])
            chamber = settings.get("heater_generic chamber") or settings.get("chamber")
            if isinstance(chamber, dict) and chamber.get("max_temp") is not None:
                temps["chamber_c"] = float(chamber["max_temp"])
                info["heated_chamber"] = True
            info["max_temps"] = temps
            machine = settings.get("printer")
            if isinstance(machine, dict):
                info["kinematics"] = machine.get("kinematics")
            _apply_queried_envelope(info, settings)
    except Exception as exc:  # noqa: BLE001
        if info["error"] is None:
            info["error"] = _error_text(exc)

    try:
        roots = _get("/server/files/roots", url, timeout).get("result", [])
        if isinstance(roots, list):
            info["roots"] = [
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "permissions": item.get("permissions"),
                }
                for item in roots
                if isinstance(item, dict)
            ]
    except Exception as exc:  # noqa: BLE001
        if info["error"] is None:
            info["error"] = _error_text(exc)

    return info


def _apply_queried_envelope(info: Dict[str, object], settings: Dict[str, object]) -> None:
    """Replace the documented envelope with one derived from *this* machine's own config.

    The usable part envelope is the levelled bed-mesh area in X/Y and the Z axis maximum, not
    the axis maxima in X/Y: the toolhead can travel past the mesh (X reaches 392, Y 410 on the
    Max 4) but a part printed out there sits on unmeasured bed.  ``BUILD_VOLUME_MM`` encodes
    that same derivation for the lab machine as a constant; doing it live means a re-meshed or
    different printer gates against its real area, and means the report can say which of the two
    a number came from.

    Silent about failure on purpose: if the sections are absent the documented envelope and its
    ``"documented"`` provenance simply stay in place, and a caller that requires a real reading
    checks ``build_volume_source`` rather than guessing.
    """
    mesh = settings.get("bed_mesh")
    if not isinstance(mesh, dict):
        return
    mesh_min = mesh.get("mesh_min")
    mesh_max = mesh.get("mesh_max")
    if not isinstance(mesh_min, list) or not isinstance(mesh_max, list):
        return
    if len(mesh_min) < 2 or len(mesh_max) < 2:
        return
    try:
        x = float(mesh_max[0]) - float(mesh_min[0])
        y = float(mesh_max[1]) - float(mesh_min[1])
    except (TypeError, ValueError):
        return
    axis_maximum = info.get("axis_maximum")
    if not isinstance(axis_maximum, list) or len(axis_maximum) < 3:
        return
    try:
        z = float(axis_maximum[2])
    except (TypeError, ValueError):
        return
    if x <= 0.0 or y <= 0.0 or z <= 0.0:
        return
    info["build_volume_mm"] = [x, y, z]
    info["build_volume_source"] = "queried"
    info["bed_mesh"] = {
        "min": [float(mesh_min[0]), float(mesh_min[1])],
        "max": [float(mesh_max[0]), float(mesh_max[1])],
    }


def _multipart_body(
    file_path: Path, fields: Dict[str, str], boundary: str
) -> bytes:
    """Build a ``multipart/form-data`` body by hand (no third-party ``requests``)."""
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    marker = "--{}".format(boundary).encode("ascii")
    parts: List[bytes] = []
    for key, value in fields.items():
        parts.append(marker)
        parts.append(
            'Content-Disposition: form-data; name="{}"'.format(key).encode("utf-8")
        )
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))
    parts.append(marker)
    parts.append(
        'Content-Disposition: form-data; name="file"; filename="{}"'.format(
            file_path.name
        ).encode("utf-8")
    )
    parts.append("Content-Type: {}".format(content_type).encode("ascii"))
    parts.append(b"")
    body = b"\r\n".join(parts) + b"\r\n"
    body += file_path.read_bytes()
    body += b"\r\n" + marker + b"--\r\n"
    return body


def upload_gcode(
    path,
    base_url: Optional[str] = None,
    subdir: str = "",
    timeout: int = 120,
) -> Dict[str, object]:
    """Upload a sliced file into the printer's ``gcodes`` root. Does NOT start a print.

    The request carries the "do not print" flag explicitly rather than relying on Moonraker's
    default, so no future default change can turn an upload into a job.
    """
    source = Path(path).expanduser().resolve()
    url = resolve_url(base_url)
    out: Dict[str, object] = {
        "url": url,
        "file": str(source),
        "uploaded": False,
        "remote_path": None,
        "size_bytes": None,
        "seconds": 0.0,
        "started_print": False,
        "error": None,
    }
    if not source.is_file():
        out["error"] = "file not found: {}".format(source)
        return out
    out["size_bytes"] = source.stat().st_size

    clean_subdir = subdir.strip("/") if subdir else ""
    fields = {
        "root": GCODE_ROOT,
        # Moonraker treats the string "false" as false; being explicit is the whole point.
        "print": "false",
    }
    if clean_subdir:
        fields["path"] = clean_subdir

    boundary = "phimech{}".format(uuid.uuid4().hex)
    body = _multipart_body(source, fields, boundary)
    headers = {
        "Content-Type": "multipart/form-data; boundary={}".format(boundary),
        "Content-Length": str(len(body)),
    }
    started = time.time()
    try:
        result = _request(
            "{}/server/files/upload".format(url),
            timeout=timeout,
            data=body,
            headers=headers,
            method="POST",
        )
    except Exception as exc:  # noqa: BLE001
        out["seconds"] = round(time.time() - started, 3)
        out["error"] = _error_text(exc)
        return out
    out["seconds"] = round(time.time() - started, 3)
    out["uploaded"] = True
    payload = result.get("result", result)
    if isinstance(payload, dict):
        item = payload.get("item")
        if isinstance(item, dict):
            out["remote_path"] = item.get("path")
            out["remote_root"] = item.get("root")
        out["print_started"] = payload.get("print_started", False)
        out["started_print"] = bool(payload.get("print_started", False))
    if out["remote_path"] is None:
        out["remote_path"] = (
            "{}/{}".format(clean_subdir, source.name) if clean_subdir else source.name
        )
    return out


def list_jobs(
    base_url: Optional[str] = None, limit: int = 20, timeout: int = _TIMEOUT_DEFAULT
) -> Dict[str, object]:
    """List the most recently modified files in the printer's ``gcodes`` root. Never raises."""
    url = resolve_url(base_url)
    out: Dict[str, object] = {"url": url, "reachable": False, "files": [], "error": None}
    try:
        result = _get("/server/files/list?root={}".format(GCODE_ROOT), url, timeout).get(
            "result", []
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = _error_text(exc)
        return out
    out["reachable"] = True
    if not isinstance(result, list):
        return out
    files = [item for item in result if isinstance(item, dict)]
    files.sort(key=lambda item: float(item.get("modified") or 0.0), reverse=True)
    out["count"] = len(files)
    out["files"] = [
        {
            "path": item.get("path"),
            "size_bytes": item.get("size"),
            "modified": item.get("modified"),
        }
        for item in files[: max(0, limit)]
    ]
    return out


def fits_build_volume(
    size_mm: Sequence[float],
    base_url: Optional[str] = None,
    envelope: Optional[Sequence[float]] = None,
) -> Tuple[bool, str]:
    """Does a part of ``size_mm`` (x, y, z) fit the 360 x 360 x 340 mm envelope?

    A 90 degree rotation about Z is tried before failing.  On the Max 4 the envelope is square,
    so that rotation cannot change the verdict — it matters for the rectangular beds this same
    check will be pointed at, and ``envelope`` exists so a caller (and the tests) can supply one.

    The default envelope is the levelled bed-mesh area, NOT the axis maxima (X reaches 392 and Y
    410): the toolhead can travel past the mesh, but a part out there is printed on unmeasured
    bed.  ``base_url`` is accepted for signature symmetry; it never widens the verdict.
    """
    if len(size_mm) != 3:
        raise ValueError("size_mm must be (x, y, z) in mm, got {!r}".format(size_mm))
    x, y, z = (float(value) for value in size_mm)
    limits = tuple(float(value) for value in (envelope or BUILD_VOLUME_MM))
    if len(limits) != 3:
        raise ValueError("envelope must be (x, y, z) in mm, got {!r}".format(envelope))
    max_x, max_y, max_z = limits
    described = "{:g} x {:g} x {:g} mm".format(max_x, max_y, max_z)
    part = "part {:.2f} x {:.2f} x {:.2f} mm".format(x, y, z)

    if z > max_z:
        return False, "{} is too tall: {:.2f} > Z {:g} mm (envelope {})".format(
            part, z, max_z, described
        )
    if x <= max_x and y <= max_y:
        return True, "{} fits {} as-is".format(part, described)
    if y <= max_x and x <= max_y:
        return True, "{} fits {} rotated 90 degrees about Z".format(part, described)

    # Report the overage of whichever orientation comes closest.
    best = min(
        ((a, b, max(0.0, a - max_x), max(0.0, b - max_y)) for a, b in ((x, y), (y, x))),
        key=lambda item: item[2] + item[3],
    )
    return False, (
        "{} does not fit {} in either Z orientation (best orientation {:.2f} x {:.2f} mm is over "
        "by {:.2f} mm in X, {:.2f} mm in Y)".format(
            part, described, best[0], best[1], best[2], best[3]
        )
    )


__all__ = [
    "BUILD_VOLUME_MM",
    "DEFAULT_PRINTER_URL",
    "GCODE_ROOT",
    "fits_build_volume",
    "list_jobs",
    "printer_info",
    "resolve_url",
    "upload_gcode",
]
