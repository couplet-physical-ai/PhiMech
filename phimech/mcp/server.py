#!/usr/bin/env python3
"""PhiMech's small, fork-native MCP surface.

The server intentionally keeps the MCP process thin.  Geometry, IR loading, synthesis, and
CAD backends remain in their own layers; imports happen inside tools so a partial checkout can
still start and return an actionable error to an agent.  The stdio client may sanitize the
child environment, so path arguments and the ``PHIMECH_*`` registration block are explicit.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP

# Resolve the repository from the environment first.  This is the repository root for
# <repo>/phimech/mcp/server.py, not the process working directory.
REPO = Path(
    os.environ.get("PHIMECH_REPO", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
sys.path.insert(0, str(REPO))

mcp = FastMCP("phimech")

# Viewer processes are intentionally held by the MCP process so an agent can start and stop a
# browser server without leaking a daemon between tool calls.
_VIEWERS: Dict[str, Any] = {}

# Keep this list next to the decorators.  Besides documenting the public surface, it gives
# the registration/handshake tests a dependency-free way to detect an accidental omission.
TOOLS = [
    "phimech_doctor",
    "ir_schema",
    "ir_validate",
    "ir_example",
    "ir_parameters",
    "ir_set_parameter",
    "bench_prompts",
    "bench_grade",
    "synth_check",
    "synth_build",
    "print_slice",
    "print_send",
    "printer_info",
    "print_profiles",
    "view_start",
    "view_stop",
    "csg_render",
    "csg_measure",
    "csg_emit_scad",
    "csg_validate",
    "pcb_inspect",
    "parts_list",
    "parts_show",
    "gates_catalog",
    "machine_capabilities",
    "machine_check",
    "render_preview",
    "write_ir",
]


# --------------------------------------------------------------------------------------
# Shared result and path helpers
# --------------------------------------------------------------------------------------


def _failure(error: Any, hint: str) -> Dict[str, Any]:
    """Return an agent-readable failure instead of allowing a tool exception to escape."""
    message = str(error) or error.__class__.__name__
    return {"ok": False, "error": message, "hint": hint}


def _jsonable(value: Any) -> Any:
    """Convert the few useful non-JSON values returned by the core into plain JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


def _success(value: Any) -> Dict[str, Any]:
    """Normalize a core result to the MCP contract while preserving its useful fields."""
    if isinstance(value, dict):
        result = _jsonable(value)
        result.setdefault("ok", True)
        return result
    return {"ok": True, "result": _jsonable(value)}


def _absolute_path(
    raw: str, label: str, *, must_exist: bool = False, kind: str = "file"
) -> Path:
    """Require an absolute path so a client cannot accidentally address another checkout."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "%s is required" % label
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "%s must be an absolute path (hint: pass /full/path/to/%s)"
            % (label, label.replace("_", "."))
        )
    path = path.resolve()
    if must_exist:
        # `kind` distinguishes the two so the error names what was actually wrong: an existing
        # directory passed where a file is wanted used to report "not found", which sends a caller
        # looking for a missing path that is right there.
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            if path.exists():
                raise ValueError(
                    "%s must be a %s, but %s is a %s"
                    % (label, kind, path, "directory" if path.is_dir() else "file")
                )
            raise FileNotFoundError("%s not found: %s" % (label, path))
    return path


def _absolute_output(raw: str, label: str) -> Path:
    """Validate an output path without requiring the file to exist yet."""
    path = _absolute_path(raw, label)
    if path.exists() and path.is_dir():
        raise IsADirectoryError("%s is a directory: %s" % (label, path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> Any:
    """Read one JSON document after the caller has applied the absolute-path policy."""
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _load_ast(ast_json_path: str) -> Tuple[Path, Any]:
    """Load a CSG AST lazily so the MCP server does not import geometry at startup."""
    path = _absolute_path(ast_json_path, "ast_json_path", must_exist=True)
    from phimech.csg.ast import from_dict

    document = _read_json(path)
    if not isinstance(document, dict):
        raise ValueError("CSG AST JSON must contain an object")
    return path, from_dict(document)


def _backend(name: str, output_format: Optional[str] = None) -> Any:
    """Resolve a requested backend only when a rendering or measurement tool needs it."""
    from phimech.backends import base

    if name == "auto":
        return base.for_format(output_format or "stl")
    if not name:
        raise ValueError("backend must be 'auto', 'openscad', or 'freecad'")
    return base.get(name)


def _formats(raw: str) -> Optional[List[str]]:
    """Turn the MCP-friendly comma-separated format option into the pipeline list."""
    if not raw or not raw.strip():
        return None
    result = [item.strip().lower().lstrip(".") for item in raw.split(",") if item.strip()]
    if not result:
        return None
    return result


def _available_examples() -> List[str]:
    """Return example stems, sorted for deterministic agent discovery."""
    examples = REPO / "examples"
    if not examples.is_dir():
        return []
    return sorted(
        path.stem for path in examples.glob("*.json") if path.is_file()
    )


def _find_png(value: Any, root: Path) -> Optional[Path]:
    """Find a generated preview in either a pipeline report or its temporary directory."""
    if isinstance(value, dict):
        for item in value.values():
            found = _find_png(item, root)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_png(item, root)
            if found is not None:
                return found
    elif isinstance(value, (str, Path)):
        candidate = Path(value)
        if candidate.suffix.lower() == ".png" and candidate.is_file():
            return candidate.resolve()
    candidates = sorted(path for path in root.rglob("*.png") if path.is_file())
    return candidates[0].resolve() if candidates else None


def _read_design_document(path: Path) -> Dict[str, Any]:
    """Read a design dict for ``write_ir`` without invoking the IR loader on a new file."""
    document = _read_json(path)
    if not isinstance(document, dict):
        raise ValueError("design must be a JSON object")
    return document


# --------------------------------------------------------------------------------------
# Inspection and synthesis tools
# --------------------------------------------------------------------------------------


@mcp.tool
def phimech_doctor() -> Dict[str, Any]:
    """Report the PhiMech toolchain and MCP registration health.

    Returns which Python, venv, CAD backends (OpenSCAD, FreeCAD) and ``mcp.json`` entry were
    found, and where each path came from. Call this first when any other tool fails with a
    backend error - it distinguishes "not installed" from "installed but broken".

    Units: none. Limits: probes only. Installs nothing, writes nothing, and starts no CAD
    subprocess beyond a ``--version`` call. It cannot repair a broken toolchain; run
    ``bin/phimech setup`` for that.
    """
    try:
        from phimech.doctor import doctor

        return _success(doctor())
    except Exception as exc:
        return _failure(exc, "run `bin/phimech doctor` in the PhiMech checkout")


@mcp.tool
def ir_schema() -> Dict[str, Any]:
    """Return the design IR JSON schema so an agent can author a conforming document.

    The schema is returned unwrapped, because it is a machine-readable contract clients feed
    straight into a validator.

    Units: every length in the IR is millimetres and every angle is degrees; the schema carries
    no unit fields. Limits: read-only, and a document that satisfies the schema can still fail
    ``ir_validate`` - the schema does not express cross-field rules such as placement references
    resolving to a real component.
    """
    try:
        from phimech.ir.ir import SCHEMA

        # Return the schema itself, rather than nesting it, because this is used as a
        # machine-readable contract by clients.  JSON schema already is a plain dict.
        return _jsonable(SCHEMA)
    except Exception as exc:
        return _failure(exc, "check that phimech/ir/design-ir.schema.json is present")


@mcp.tool
def ir_validate(ir_path: str) -> Dict[str, Any]:
    """Validate an IR file and return every human-readable diagnostic.

    ``ir_path`` must be an **absolute** path to an existing JSON file; a relative path is
    rejected rather than resolved, so a client cannot accidentally address another checkout.

    Units: millimetres and degrees. Limits: static validation only - no placement is resolved,
    no geometry is built and no gate is run, so passing here does not mean the design builds.
    Use ``synth_check`` for that.
    """
    try:
        path = _absolute_path(ir_path, "ir_path", must_exist=True)
        from phimech.ir.ir import load_and_validate

        _, errors = load_and_validate(path)
        return {"ok": not errors, "errors": list(errors), "path": str(path)}
    except Exception as exc:
        return _failure(exc, "pass an absolute JSON IR path and fix the reported diagnostics")


@mcp.tool
def ir_example(name: str) -> Dict[str, Any]:
    """Return one checked-in example design IR, plus the list of available example names.

    ``name`` is an example stem such as ``parametric-box``; a trailing ``.json`` is accepted
    and stripped. An unknown name fails with the available list rather than guessing.

    Units: millimetres and degrees. Limits: reads only the repository's own ``examples/``
    directory - it cannot fetch an arbitrary path (use ``ir_validate`` for that) and it does not
    build or gate what it returns.
    """
    try:
        available = _available_examples()
        requested = str(name or "").strip()
        if requested.endswith(".json"):
            requested = requested[:-5]
        if not requested or requested not in available:
            raise FileNotFoundError(
                "example %r is not available (available: %s)"
                % (name, ", ".join(available) or "none")
            )
        path = REPO / "examples" / (requested + ".json")
        return {
            "ok": True,
            "name": requested,
            "contents": path.read_text(encoding="utf-8"),
            "available": available,
        }
    except Exception as exc:
        return _failure(exc, "call ir_example with one of the names in its available list")


@mcp.tool
def bench_prompts() -> Dict[str, Any]:
    """Fetch the generation-benchmark briefs and the checks each one is graded against.

    Use this to measure how well designs are being generated. Each case gives a natural-language
    brief plus the exact machine checks that will score it - the rubric is deliberately open, since
    the point is designing to a stated requirement, not guessing a hidden one.

    Units: mm throughout. Limits: read-only, and it grades **nothing** by itself. Author one IR per
    case as ``<case-id>.json`` in a directory, then call ``bench_grade`` on that directory. There is
    no partial credit: a case passes only when every one of its checks passes.
    """
    try:
        from phimech.bench.grade import load_suite

        suite = load_suite()
        return {"ok": True, "suite": suite.get("suite"), "cases": suite.get("cases", [])}
    except Exception as exc:
        return _failure(exc, "the benchmark suite ships with phimech; reinstall if it is missing")


@mcp.tool
def bench_grade(candidates_dir: str, out_dir: str = "", build: bool = True) -> Dict[str, Any]:
    """Build and machine-score a directory of candidate designs against the benchmark.

    ``candidates_dir`` and ``out_dir`` must be **absolute** paths; ``candidates_dir`` must exist
    and hold one ``<case-id>.json`` per case (``<case-id>/design.json`` also works). ``out_dir``
    defaults to ``<candidates_dir>/_build``.

    Units: mm. Limits: with ``build=True`` this runs a real CAD build per case, so expect tens of
    seconds each and a hard dependency on OpenSCAD or FreeCAD. ``build=False`` still runs the
    IR-level gates (placement, clearance, wall, cutouts) in well under a second, but anything
    needing a mesh **cannot** pass - ``manifold``, ``part_volume_positive``, ``fits_bed``, and any
    measured size or part-count check - so those scores are **not comparable** to a built run; use
    it for fast iteration and quote only built scores. A case with no candidate scores zero rather
    than being skipped, and a **skipped gate counts as a failure**: a gate that ran nothing proved
    nothing.
    """
    try:
        candidates = _absolute_path(
            candidates_dir, "candidates_dir", must_exist=True, kind="dir"
        )
        out_root = (
            _absolute_path(out_dir, "out_dir") if out_dir else Path(candidates) / "_build"
        )
        from phimech.bench.run import run_suite

        outcome = run_suite(candidates, out_root, build=bool(build))
        outcome["ok"] = True
        return outcome
    except Exception as exc:
        return _failure(
            exc,
            "call bench_prompts first, then write one <case-id>.json per case into an "
            "absolute directory",
        )


@mcp.tool
def ir_parameters(ir_path: str) -> Dict[str, Any]:
    """List a design's editable parameters and their bounds.

    ``ir_path`` must be an **absolute** path to an existing IR. Each entry carries ``value``,
    optional ``min``/``max``/``step``, a ``unit`` and a description. Call this before
    ``ir_set_parameter`` so an edit is made inside the declared range.

    Units: as declared per parameter - mm unless the entry says ``deg``, ``g`` or ``count``.
    Limits: read-only. A design with no ``parameters`` block returns an empty list, which means
    nothing was declared editable - it does **not** mean every number is safe to change. Editing
    the IR by hand elsewhere is not equivalent: only declared parameters carry bounds.
    """
    try:
        path = _absolute_path(ir_path, "ir_path", must_exist=True)
        from phimech.ir.ir import load_and_validate

        design, errors = load_and_validate(path)
        parameters = design.get("parameters") or {}
        rows = []
        for name in sorted(parameters):
            spec = parameters[name] if isinstance(parameters[name], dict) else {}
            rows.append(
                {
                    "name": name,
                    "value": spec.get("value"),
                    "min": spec.get("min"),
                    "max": spec.get("max"),
                    "step": spec.get("step"),
                    "unit": spec.get("unit", "mm"),
                    "description": spec.get("description", ""),
                }
            )
        return {"ok": not errors, "path": str(path), "parameters": rows, "errors": errors}
    except Exception as exc:
        return _failure(exc, "pass an absolute IR path; add a `parameters` block to make a number "
                             "editable")


@mcp.tool
def ir_set_parameter(ir_path: str, name: str, value: float) -> Dict[str, Any]:
    """Change one declared parameter in a design file, then re-validate it.

    ``ir_path`` must be an **absolute** path to an existing IR. This is the supported way to adjust
    a design without regenerating it - the same operation a UI slider performs - and it edits one
    number in one place however many fields reference it.

    Units: whatever the parameter declares (mm by default). Limits: **refuses a value outside the
    declared min/max**, and refuses a name the design does not declare - widen the bound or add the
    parameter deliberately instead. It will not edit any other field: derived geometry and
    coordinates are pipeline output, and editing them would leave a design its own IR can no longer
    reproduce. The write is atomic. Validation runs but **no geometry is rebuilt and no gate is
    re-run** - call ``synth_check`` or ``synth_build`` next, because a value inside its bounds can
    still fail a gate.
    """
    try:
        path = _absolute_path(ir_path, "ir_path", must_exist=True)
        from phimech.ir.ir import load_and_validate, set_parameter

        result = set_parameter(path, str(name), float(value))
        _, errors = load_and_validate(path)
        result["errors"] = errors
        result["ok"] = not errors
        result["next"] = "run synth_check or synth_build to re-run the gates"
        return result
    except Exception as exc:
        return _failure(exc, "call ir_parameters first and choose a value inside the declared "
                             "min/max")


@mcp.tool
def synth_check(ir_path: str) -> Dict[str, Any]:
    """Validate, place, and gate an IR without writing any file.

    ``ir_path`` must be an **absolute** path to an existing IR. Runs in milliseconds because it
    stops before geometry export: use it as the fast inner loop while iterating on a design.

    Units: millimetres, degrees, grams. Limits: writes nothing and renders nothing, so
    mesh-dependent gates (``manifold``, ``part_volume_positive``, ``fits_bed``) are **skipped**,
    not passed. A skipped gate is not a passing gate - only ``synth_build`` can report them.
    """
    try:
        path = _absolute_path(ir_path, "ir_path", must_exist=True)
        from phimech.synth import pipeline

        return _success(pipeline.check(str(path)))
    except Exception as exc:
        return _failure(exc, "run ir_validate first and ensure all referenced source paths exist")


@mcp.tool
def synth_build(
    ir_path: str,
    out_dir: str,
    formats: str = "",
    backend: str = "auto",
    drawing: bool = False,
) -> Dict[str, Any]:
    """Build a placed assembly and enclosure, writing every requested format to disk.

    ``ir_path`` and ``out_dir`` must both be **absolute**; ``out_dir`` is created if missing.
    ``formats`` is a comma-separated subset of stl, step, 3mf, scad, fcstd, png, obj (empty =
    the design's own ``outputs.formats``). ``backend`` is auto, openscad or freecad.
    ``drawing`` additionally writes the dimensioned inspection sheet of the main printable part
    (``drawing.svg``, plus ``.pdf``/``.png`` when a converter exists) into ``out_dir`` and
    returns it under ``drawing`` with the sheet's deterministic ``checks`` - advisory evidence
    that never fails the build; it costs one OpenSCAD run per z level.

    Units: millimetres, degrees, grams. Limits: **this writes files and spawns CAD
    subprocesses**, so it takes seconds to minutes, not milliseconds - run ``synth_check`` first.
    It overwrites same-named outputs in ``out_dir``. OpenSCAD cannot read or write STEP, so a
    STEP request routes to FreeCAD; if that backend is missing the format is reported as skipped
    rather than silently dropped. A returned ``ok: false`` can mean a render error even when
    every gate passed.
    """
    try:
        design_path = _absolute_path(ir_path, "ir_path", must_exist=True)
        output = _absolute_path(out_dir, "out_dir")
        if output.exists() and not output.is_dir():
            raise NotADirectoryError("out_dir is not a directory: %s" % output)
        output.mkdir(parents=True, exist_ok=True)
        from phimech.synth import pipeline

        result = pipeline.build(
            str(design_path), output, formats=_formats(formats), backend=backend,
            drawing=bool(drawing),
        )
        return _success(result)
    except Exception as exc:
        return _failure(exc, "use an absolute IR path and output directory; run synth_check first")


@mcp.tool
def print_slice(
    stl_path: str,
    out_dir: str,
    material: str = "PETG",
    quality: str = "standard",
    printer: str = "qidi-max4",
) -> Dict[str, Any]:
    """Slice an absolute STL path into printer-ready G-code in an absolute output directory.

    STL dimensions are millimetres; ``material`` and ``quality`` select bundled profiles, and
    ``printer`` selects the printer profile. Use after numeric gates pass. This only creates a
    file: it does not contact the printer, heat it, move it, or start a print.
    """
    try:
        stl = _absolute_path(stl_path, "stl_path", must_exist=True)
        output = _absolute_path(out_dir, "out_dir")
        if output.exists() and not output.is_dir():
            raise NotADirectoryError("out_dir is not a directory: %s" % output)
        output.mkdir(parents=True, exist_ok=True)
        from phimech.print import slicer

        result = slicer.slice_stl(
            str(stl),
            str(output),
            material=material,
            quality=quality,
            printer=printer,
        )
        return _success({"stl_path": str(stl), "out_dir": str(output), "slice": result})
    except Exception as exc:
        return _failure(
            exc,
            "pass absolute STL/output paths and install/configure the OrcaSlicer profile",
        )


@mcp.tool
def print_send(gcode_path: str, base_url: str = "") -> Dict[str, Any]:
    """Upload an absolute ``.gcode.3mf`` file to Moonraker without starting a print.

    Use after slicing and review. ``base_url`` is the printer HTTP URL; the upload is network
    I/O only, with dimensions and temperatures controlled by the printer profile. Safety boundary:
    this tool uploads only and never starts a print, heats anything, or moves an axis; use the
    separate qidi-mcp start-print confirmation flow for a human-approved print.
    """
    try:
        path = _absolute_path(gcode_path, "gcode_path", must_exist=True)
        from phimech.print import printer

        result = printer.upload_gcode(str(path), base_url=base_url or None)
        return _success({"gcode_path": str(path), "upload": result, "print_started": False})
    except Exception as exc:
        return _failure(
            exc,
            "pass an absolute .gcode.3mf path; this tool uploads only and does not start prints",
        )


@mcp.tool
def printer_info(base_url: str = "") -> Dict[str, Any]:
    """Read printer identity, state, and numeric limits from Moonraker without motion.

    Use before fit or print planning. Dimensions are millimetres and temperatures are degrees C;
    ``base_url`` defaults to the configured PHIMECH_PRINTER_URL. This is read-only: it never
    changes temperatures, moves an axis, uploads a file, or starts a print.
    """
    try:
        from phimech.print import printer

        return _success(printer.printer_info(base_url=base_url or None))
    except Exception as exc:
        return _failure(exc, "check PHIMECH_PRINTER_URL and that Moonraker is reachable read-only")


@mcp.tool
def print_profiles() -> Dict[str, Any]:
    """List available printer material and quality profiles for slicing.

    Use before ``print_slice`` to choose profile names. Profiles describe millimetre-scale FDM
    settings and stock temperatures; this discovery call does not contact the printer and cannot
    heat, move, upload, or start a print.
    """
    try:
        from phimech.print import profiles

        return _success(
            {
                "materials": profiles.list_materials(),
                "qualities": profiles.list_qualities(),
            }
        )
    except Exception as exc:
        return _failure(exc, "check the bundled OrcaSlicer/QIDI profile installation")


@mcp.tool
def view_start(
    out_dir: str,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> Dict[str, Any]:
    """Start a local browser viewer for an absolute build-output directory and return its URL.

    Use after synthesis gates pass to inspect STL/STEP/PNG output before slicing. ``port`` is a
    TCP port, not a geometry unit. This serves files only; it has no printer, temperature, axis,
    or other external authority.

    Limits: read-only and localhost by default; it serves the given directory and nothing above
    it. ``port`` is a **request, not a promise** - if it is taken the server walks upward and then
    asks the OS for any free port, so always use the returned URL rather than assuming ``port``.
    Serving an empty or partially built directory succeeds with warnings rather than failing, so a
    returned URL is not evidence that geometry exists.
    """
    try:
        directory = _absolute_path(out_dir, "out_dir")
        if not directory.is_dir():
            raise NotADirectoryError("out_dir is not a directory: %s" % directory)
        from phimech.viewer import server

        httpd, url = server.serve(str(directory), host=host, port=int(port))
        actual_url = str(url)
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="phimech-viewer",
            daemon=True,
        )
        thread.start()
        _VIEWERS[actual_url] = httpd
        address = getattr(httpd, "server_address", (host, int(port)))
        actual_port = int(address[1]) if len(address) > 1 else int(port)
        return {
            "ok": True,
            "url": actual_url,
            "out_dir": str(directory),
            "host": host,
            "port": actual_port,
        }
    except Exception as exc:
        return _failure(exc, "pass an absolute existing build-output directory and valid TCP port")


@mcp.tool
def view_stop(url: str = "") -> Dict[str, Any]:
    """Stop a local viewer previously started by ``view_start``.

    Pass the exact URL returned by ``view_start``; omit it to stop every viewer owned by this MCP
    process. This only closes the HTTP server and cannot affect a printer, a print job, 
    hardware.
    """
    requested = str(url or "").strip()
    targets = {
        key: value for key, value in _VIEWERS.items() if not requested or key == requested
    }
    stopped: List[str] = []
    for key, httpd in targets.items():
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        _VIEWERS.pop(key, None)
        stopped.append(key)
    return {"ok": True, "stopped": sorted(stopped), "requested": requested or None}


@mcp.tool
def csg_render(
    ast_json_path: str,
    out_path: str,
    backend: str = "auto",
) -> Dict[str, Any]:
    """Render a CSG AST to a CAD file through the selected subprocess-backed backend.

    ``ast_json_path`` must be an **absolute** path to an existing JSON CSG AST, and ``out_path``
    an absolute path whose **suffix chooses the format** (.stl, .step, .3mf, .png, ...).

    Units: millimetres and degrees throughout the AST - there are no unit fields and no
    conversion. Limits: spawns a CAD subprocess with a timeout. Not every backend supports every
    format - OpenSCAD cannot write STEP, and a suffix the chosen backend cannot produce is an
    error, not a fallback. This renders a raw AST and runs **no gates**: use ``synth_build`` for
    gated output.
    """
    try:
        _, node = _load_ast(ast_json_path)
        output = _absolute_output(out_path, "out_path")
        output_format = output.suffix.lower().lstrip(".")
        if not output_format:
            raise ValueError("out_path must have a supported output suffix")
        selected = _backend(backend, output_format)
        render_result = selected.render(node, output, output_format)
        data = render_result.to_dict()
        data.update({"ok": True, "path": str(output), "bytes": output.stat().st_size})
        return _jsonable(data)
    except Exception as exc:
        return _failure(
            exc,
            "use an absolute AST/output path and run phimech_doctor for backend status",
        )


@mcp.tool
def csg_measure(ast_json_path: str, backend: str = "auto") -> Dict[str, Any]:
    """Measure an AST, returning bounding box, volume and surface area.

    ``ast_json_path`` must be an **absolute** path to an existing JSON CSG AST.

    Units: bbox and its size in millimetres, volume in mm3, area in mm2. Limits: spawns a CAD
    subprocess, so it is far slower than ``csg_validate``. Backends tessellate differently, so
    OpenSCAD and FreeCAD can disagree slightly on volume for curved geometry - compare within a
    tolerance, never for equality. Reports no mass: that needs a material density.
    """
    try:
        _, node = _load_ast(ast_json_path)
        selected = _backend(backend, "stl")
        measurement = selected.measure(node)
        return {"ok": True, "backend": selected.name, "measurement": measurement.to_dict()}
    except Exception as exc:
        return _failure(
            exc,
            "use backend='openscad' or 'freecad', and verify that its binary is installed",
        )


@mcp.tool
def csg_emit_scad(ast_json_path: str) -> Dict[str, Any]:
    """Emit deterministic OpenSCAD source text from a JSON CSG AST.

    ``ast_json_path`` must be an **absolute** path to an existing JSON CSG AST. Deterministic:
    the same AST always emits byte-identical source, which is what makes it diffable in review.

    Units: millimetres and degrees. Limits: returns text only - nothing is rendered, measured or
    written to disk, and no OpenSCAD binary is required or invoked. Emitting source is not proof
    that it renders; an AST that emits cleanly can still fail ``csg_render``.
    """
    try:
        _, node = _load_ast(ast_json_path)
        from phimech.backends.scad_emit import emit

        return {"ok": True, "scad": emit(node)}
    except Exception as exc:
        return _failure(exc, "pass an absolute path to a JSON CSG AST")


@mcp.tool
def csg_validate(ast_json_path: str) -> Dict[str, Any]:
    """Validate CSG node kinds and dimensions without invoking a CAD subprocess.

    ``ast_json_path`` must be an **absolute** path to an existing JSON CSG AST. Cheap enough to
    call on every edit - no subprocess, no temporary files.

    Units: millimetres and degrees; it rejects non-positive or non-finite dimensions. Limits:
    structural checks only. It cannot detect a geometrically empty result (a difference that
    removes everything), a non-manifold mesh, or a self-intersection - those need a real render
    and the ``manifold`` gate.
    """
    try:
        _, node = _load_ast(ast_json_path)
        from phimech.csg.validate import validate

        errors = list(validate(node))
        return {"ok": not errors, "errors": errors}
    except Exception as exc:
        return _failure(exc, "pass an absolute path to a JSON CSG AST")


@mcp.tool
def pcb_inspect(board_path: str) -> Dict[str, Any]:
    """Extract a board envelope, mounting holes and connector summary from a KiCad PCB.

    ``board_path`` must be an **absolute** path to an existing ``.kicad_pcb``. This is what the
    enclosure synthesiser sees, so call it to explain a surprising cavity or standoff position.

    Units: millimetres, in the board's own coordinate frame. Limits: parses the board file
    directly - **KiCad is never launched** and no netlist, schematic or DRC information is read.
    Component heights come from the PhiMech library or the IR, not from the PCB file, so a part
    the library does not know contributes no height.
    """
    try:
        path = _absolute_path(board_path, "board_path", must_exist=True)
        from phimech.components.pcb import inspect

        return _success(inspect(path))
    except Exception as exc:
        return _failure(exc, "pass an absolute .kicad_pcb path and check that it is readable")


@mcp.tool
def parts_list(category: str = "") -> Dict[str, Any]:
    """List the parametric component library, optionally filtered by category.

    ``category`` is matched case-insensitively for an exact category name; empty lists
    everything. Use the returned keys with ``parts_show`` or as an IR component source.

    Units: any dimensions in the returned metadata are millimetres. Limits: lists only PhiMech's
    bundled library - it is not a supplier catalogue, has no prices or stock, and does not build
    geometry. An unknown ``category`` returns an empty list rather than an error.
    """
    try:
        from phimech.components import library

        parts = library.list_parts()
        requested = (category or "").strip().lower()
        if requested:
            parts = [item for item in parts if str(item.get("category", "")).lower() == requested]
        return {"ok": True, "category": requested, "parts": _jsonable(parts)}
    except Exception as exc:
        return _failure(exc, "inspect the component library or omit category to list all parts")


@mcp.tool
def parts_show(part: str) -> Dict[str, Any]:
    """Build one library part's envelope and return it as JSON.

    ``part`` is a key from ``parts_list``; call that first rather than guessing a name.

    Units: millimetres and degrees. Limits: returns the **envelope** - the bounding geometry
    placement needs - not a detailed model of the real component, and it writes no file. Spawns
    no CAD subprocess.
    """
    try:
        from phimech.components import library

        envelope = library.make(part)
        return _success(envelope.to_dict())
    except Exception as exc:
        return _failure(exc, "call parts_list first and use one of its part keys")


@mcp.tool
def gates_catalog() -> Dict[str, Any]:
    """Return the numeric enclosure gate definitions exposed by PhiMech.

    Includes the machine gates, which only run when a design declares
    ``manufacturing.machine``. Use before ``synth_check`` to see what a report can contain.

    Units: each gate carries its own - millimetres for clearances and walls, degrees for
    overhang, grams for mass. Limits: returns definitions only. It runs **no** gate, reads no
    design and touches no geometry, so it reports what *could* be checked, never a verdict.
    """
    try:
        from phimech.check.gates import gate_catalog
        from phimech.check.machine import machine_gate_catalog

        return _success(
            {"gates": gate_catalog(), "machine_gates": machine_gate_catalog()}
        )
    except Exception as exc:
        return _failure(exc, "check that the phimech.check gate modules are present")


@mcp.tool
def machine_capabilities(base_url: str = "") -> Dict[str, Any]:
    """Read what the real printer can do, so a design can be checked against the actual machine.

    Returns the queried levelled build volume in millimetres, the fitted nozzle diameter, the
    heater ceilings in degrees C, and whether the machine has a heated chamber. ``base_url`` is
    the Moonraker URL and defaults to ``$PHIMECH_PRINTER_URL``.

    ``build_volume_source`` is the field to read first: ``"queried"`` means the envelope came
    from this machine's own bed mesh and Z axis, ``"documented"`` means the printer did not
    answer and the value is a compiled-in constant. Never treat ``"documented"`` as a machine
    reading.

    Read-only: this never starts a print, moves an axis, sets a temperature, loads filament, or
    writes config. Starting a print belongs to the separate qidi-mcp server behind its own
    two-step confirm token.
    """
    try:
        from phimech.check.machine import query_machine

        facts = query_machine(base_url)
        return _success(
            {
                "machine": facts,
                "reachable": bool(facts.get("reachable")),
                "build_volume_source": facts.get("build_volume_source"),
            }
        )
    except Exception as exc:
        return _failure(exc, "check PHIMECH_PRINTER_URL and that Moonraker is reachable read-only")


@mcp.tool
def machine_check(ir_path: str, base_url: str = "") -> Dict[str, Any]:
    """Gate a design against the real printer's queried capabilities, without printing anything.

    Runs the four machine gates for the IR at an absolute ``ir_path``: the printer answered and
    supplied its own envelope, every printable part fits that envelope in millimetres, the fitted
    nozzle matches the one the design assumed, and the machine's heater ceilings and chamber
    capability can run ``manufacturing.material``.

    Fails closed. An unreachable printer is reported as ``unknown`` and the gates FAIL; they do
    not skip and do not fall back to the documented envelope. Unlike ``synth_check`` this queries
    the machine whether or not the design declares ``manufacturing.machine``, so it is the tool
    for answering "would this fit our printer" about a design that never opted in.

    Read-only: no print is started, no axis moves, nothing is heated, no filament is loaded, no
    config is written.
    """
    try:
        path = _absolute_path(ir_path, "ir_path", must_exist=True)
        from phimech.check.machine import query_machine, run_machine_gates
        from phimech.components import providers
        from phimech.ir import ir as ir_module
        from phimech.synth import enclosure as enclosure_module
        from phimech.synth import place as place_module

        design, errors = ir_module.load_and_validate(str(path))
        facts = query_machine(base_url)
        parts: Any = None
        placement: Any = None
        note = ""
        # A design that cannot be placed still gets a machine verdict on the facts that do not
        # need geometry (reachability, nozzle, material), which is the useful half when the IR
        # is still being shaped.
        if not [message for message in errors if not str(message).startswith("warn")]:
            try:
                base_dir = (design.get("_meta") or {}).get("base_dir")
                envelopes = providers.envelopes_for(design, base_dir=base_dir)
                placement = place_module.place(design, envelopes)
                if design.get("enclosure"):
                    parts = enclosure_module.synthesize(
                        placement,
                        design.get("enclosure"),
                        design.get("manufacturing"),
                        components_by_ref={
                            str(component["ref"]): component
                            for component in design.get("components") or ()
                            if isinstance(component, dict) and component.get("ref")
                        },
                        title=design.get("title") or "",
                    )
            except Exception as exc:
                note = "could not synthesise geometry: {}: {}".format(type(exc).__name__, exc)
        else:
            note = "design has validation errors; size gate has no geometry to judge"

        report = run_machine_gates(design, placement, parts, {}, machine=facts)
        result = {
            "ir_path": str(path),
            "reachable": bool(facts.get("reachable")),
            "build_volume_source": facts.get("build_volume_source"),
            "build_volume_mm": facts.get("build_volume_mm"),
            "passed": report.passed,
            "summary": report.summary_line(),
            "gates": report.to_dict()["results"],
            "ir_errors": list(errors),
            "print_started": False,
        }
        if note:
            result["note"] = note
        return _success(result)
    except Exception as exc:
        return _failure(
            exc,
            "pass an absolute IR path; check PHIMECH_PRINTER_URL is reachable read-only",
        )


@mcp.tool
def render_preview(
    ir_or_ast_path: str,
    out_png: str,
    view: str = "iso",
) -> Dict[str, Any]:
    """Render a single PNG preview of an IR design or a raw CSG AST.

    ``ir_or_ast_path`` must be an **absolute** path to an existing JSON file (either kind is
    accepted and detected); ``out_png`` must be absolute and end in ``.png``. ``view`` is one of
    iso, front, back, left, right, top, bottom.

    Units: millimetres and degrees in the input; the output is pixels. Limits: a picture, not
    evidence - **no gate is run and no measurement is returned**, so never conclude a part is
    correct from a preview. An IR preview runs the synthesis backend and can take as long as a
    build; it writes only the one PNG requested.
    """
    try:
        source = _absolute_path(ir_or_ast_path, "ir_or_ast_path", must_exist=True)
        output = _absolute_output(out_png, "out_png")
        if output.suffix.lower() != ".png":
            raise ValueError("out_png must end in .png")
        if view not in ("iso", "front", "back", "left", "right", "top", "bottom"):
            raise ValueError("view must be iso, front, back, left, right, top, or bottom")

        document = _read_json(source)
        is_ir = isinstance(document, dict) and "components" in document
        if is_ir:
            from phimech.synth import pipeline

            temporary = Path(tempfile.mkdtemp(prefix="phimech-preview-"))
            report = pipeline.build(str(source), temporary, formats=["png"], backend="auto")
            if isinstance(report, dict) and report.get("ok") is False:
                return report
            generated = _find_png(report, temporary)
            if generated is None:
                raise FileNotFoundError(
                    "synthesis completed without a PNG preview; report=%s" % report
                )
            shutil.copyfile(str(generated), str(output))
            backend_name = report.get("backend") if isinstance(report, dict) else "auto"
        else:
            from phimech.csg.ast import from_dict

            if not isinstance(document, dict):
                raise ValueError("preview input must be an IR or CSG AST JSON object")
            node = from_dict(document)
            selected = _backend("auto", "png")
            selected.render(node, output, "png")
            backend_name = selected.name

        return {
            "ok": True,
            "path": str(output),
            "bytes": output.stat().st_size,
            "backend": backend_name,
            "view": view,
        }
    except Exception as exc:
        return _failure(exc, "pass absolute paths; IR previews need a working synthesis backend")


@mcp.tool
def write_ir(
    ir_path: str,
    design: Dict[str, Any],
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Validate an agent-authored design IR and write it canonically to an absolute path.

    ``ir_path`` must be **absolute**; parent directories are created. Defaults are applied
    *before* validation so a file written here has the same canonical shape as one the CLI
    loads. The write is atomic (temp file plus rename), so a failure cannot leave a half-written
    IR behind.

    Units: millimetres and degrees - do not add unit fields. Limits: **validation failure means
    nothing is written**; the errors come back instead. It refuses to clobber an existing
    non-JSON file unless ``overwrite=True``, but with ``overwrite=True`` it will replace an
    existing IR without further warning. Writing a valid IR does not mean it builds - run
    ``synth_check`` next.
    """
    try:
        path = _absolute_output(ir_path, "ir_path")
        if not isinstance(design, dict):
            raise ValueError("design must be a JSON object")

        if path.exists():
            try:
                _read_design_document(path)
            except Exception as exc:
                if not overwrite:
                    raise ValueError(
                        "refusing to overwrite non-JSON file %s; pass overwrite=True" % path
                    ) from exc

        from phimech.ir import ir

        # Apply defaults before validation/serialization so a file authored through MCP has
        # the same canonical shape as one loaded by the CLI.
        candidate = json.loads(json.dumps(design))
        ir.apply_defaults(candidate)
        errors = list(ir.validate(candidate))
        if errors:
            return {
                "ok": False,
                "error": "design IR validation failed",
                "hint": "fix the errors and call write_ir again",
                "errors": errors,
            }
        text = ir.dumps(candidate) + "\n"
        temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
        return {"ok": True, "path": str(path), "bytes": len(text.encode("utf-8")), "errors": []}
    except Exception as exc:
        return _failure(
            exc,
            "pass an absolute JSON path; use overwrite=True only for an intentional replacement",
        )


# --------------------------------------------------------------------------------------
# Tool modules contributed by other packages
# --------------------------------------------------------------------------------------

#: Sibling modules that add tools without editing this file. Each exposes ``register(mcp)``
#: and a ``TOOLS`` list; they are discovered by name rather than hard-coded so a new lane ships
#: its tools by dropping one file here. The guide (`phimech/guide/content.py`) and the
#: tool-docs gate (`tests/test_mcp_tool_docs.py`) scan the same ``tools_*.py`` pattern, so a
#: tool cannot exist here without appearing in the agent guide and meeting the docstring bar.
TOOL_MODULES: List[str] = sorted(
    p.stem for p in Path(__file__).resolve().parent.glob("tools_*.py")
)


def _register_tool_modules() -> Dict[str, List[str]]:
    """Import every ``tools_*`` sibling and register its tools; a broken one is reported, not
    fatal, because the core tools must stay reachable even if an optional lane fails to import.
    """
    import importlib

    registered: Dict[str, List[str]] = {}
    for stem in TOOL_MODULES:
        try:
            module = importlib.import_module("phimech.mcp." + stem)
            module.register(mcp)
            names = list(getattr(module, "TOOLS", []))
        except Exception as exc:  # pragma: no cover - only with a broken checkout
            print("phimech mcp: {} not loaded: {}".format(stem, exc), file=sys.stderr)
            continue
        registered[stem] = names
        for name in names:
            if name not in TOOLS:
                TOOLS.append(name)
    return registered


REGISTERED_TOOL_MODULES = _register_tool_modules()


if __name__ == "__main__":
    mcp.run()
