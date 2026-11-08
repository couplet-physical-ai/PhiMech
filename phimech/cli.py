"""The PhiMech command-line interface.

Handlers import synthesis, component, backend, and MCP implementations only when their
subcommand is selected.  This is important for the system Python: a fast IR/schema command
must remain usable while another workstream is still implementing an optional slice.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import importlib.util
import inspect
import io
import json
import os
import runpy
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phimech.version import __version__, git_describe


# --------------------------------------------------------------------------------------
# generic output and lazy imports
# --------------------------------------------------------------------------------------


def _lazy_import(module_name: str) -> Optional[Any]:
    """Import one command implementation and turn an incomplete slice into a CLI error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        print(
            "error: {} not available yet ({})".format(module_name, exc),
            file=sys.stderr,
        )
        return None


def _jsonable(value: Any) -> Any:
    """Convert backend dataclasses and paths into values accepted by ``json.dumps``."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return str(value)


def _print_json(value: Any) -> None:
    """Emit stable, indented JSON for both humans and automation."""
    print(json.dumps(_jsonable(value), indent=2, sort_keys=True))


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a pipeline dictionary or a report dataclass."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _repository() -> Path:
    """Find the repository used by examples and the test runner."""
    configured = os.environ.get("PHIMECH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


# --------------------------------------------------------------------------------------
# gate and pipeline presentation
# --------------------------------------------------------------------------------------


def _gate_results(value: Any) -> List[Any]:
    """Extract gate records from the few report shapes used by synth and check layers."""
    if value is None:
        return []
    results = _field(value, "results", None)
    if isinstance(results, (list, tuple)):
        return list(results)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        for key in ("gates", "gate_report", "gate_results", "gate_summary"):
            if key in value:
                found = _gate_results(value[key])
                if found:
                    return found
    return []


def _records_from_pipeline(result: Any) -> List[Any]:
    """Find gate results without imposing a concrete pipeline report class on this slice."""
    records = _gate_results(result)
    if records:
        return records
    for key in ("gates", "gate_report", "gate_results", "gate_summary"):
        candidate = _field(result, key, None)
        records = _gate_results(candidate)
        if records:
            return records
    return []


def _gate_status(record: Any) -> Tuple[str, str, str]:
    """Normalize a GateResult-like value to status, name, and message."""
    name = str(_field(record, "name", "gate"))
    message = str(_field(record, "message", ""))
    raw_status = _field(record, "status", None)
    if raw_status is not None and str(raw_status).upper() in ("PASS", "FAIL", "SKIP"):
        status = str(raw_status).upper()
    elif _field(record, "skipped", False) or _field(record, "skip", False):
        status = "SKIP"
    else:
        passed = _field(record, "passed", None)
        status = "PASS" if passed is True else "FAIL" if passed is False else "SKIP"
    return status, name, message


def _print_gates(result: Any) -> List[str]:
    """Print the required one-line gate summary and return normalized statuses."""
    statuses: List[str] = []
    for record in _records_from_pipeline(result):
        status, name, message = _gate_status(record)
        statuses.append(status)
        print("{} {} — {}".format(status, name, message))
    return statuses


def _pipeline_failed(result: Any, ignore_gates: bool = False) -> bool:
    """Infer a shell status from a pipeline report while accepting plain dictionaries."""
    statuses = [_gate_status(item)[0] for item in _records_from_pipeline(result)]
    if not ignore_gates and "FAIL" in statuses:
        return True
    if isinstance(result, dict):
        for key in ("passed", "ok", "valid", "success"):
            value = result.get(key)
            if value is False and not (ignore_gates and key in ("passed", "ok")):
                return True
        errors = result.get("errors")
        if isinstance(errors, (list, tuple)) and errors:
            return True
    return False


def _print_messages(result: Any) -> None:
    """Show validation/error strings that are not represented as gate records."""
    if not isinstance(result, dict):
        return
    for key in ("errors", "warnings"):
        values = result.get(key)
        if isinstance(values, (list, tuple)):
            for value in values:
                print("{}: {}".format(key.rstrip("s"), value))


def _print_parts(result: Any) -> None:
    """Print a small part table from a build report, regardless of its exact value shape."""
    parts = _field(result, "parts", None)
    if parts is None:
        parts = _field(result, "part_table", None)
    print("parts")
    print("name                 output                         size")
    rows: List[Tuple[str, str, str]] = []
    if isinstance(parts, dict):
        iterable = parts.items()
    elif isinstance(parts, (list, tuple)):
        iterable = ((str(index), item) for index, item in enumerate(parts))
    else:
        iterable = []
    for name, item in iterable:
        path = _field(item, "path", _field(item, "output", ""))
        size = _field(item, "size", _field(item, "bbox", ""))
        rows.append((str(name), str(path), str(size)))
    for name, path, size in rows:
        print("{:<20} {:<30} {}".format(name, path, size))


def _report_path(result: Any) -> Optional[str]:
    """Find a report path in a build result for the final human-facing line."""
    for key in ("report_path", "report"):
        value = _field(result, key, None)
        if isinstance(value, (str, Path)):
            return str(value)
        if isinstance(value, dict):
            nested = value.get("path")
            if nested:
                return str(nested)
    return None


# --------------------------------------------------------------------------------------
# command handlers
# --------------------------------------------------------------------------------------


def _handle_doctor(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.doctor")
    if module is None:
        return 2
    return int(module.main(["--json"] if args.json else []))


def _handle_synth_check(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.synth.pipeline")
    if module is None:
        return 2
    result = module.check(args.file)
    if args.json:
        _print_json(result)
    else:
        _print_messages(result)
        _print_gates(result)
    return 1 if _pipeline_failed(result) else 0


def _build_kwargs(module: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Pass only keywords supported by the agreed pipeline signature."""
    kwargs: Dict[str, Any] = {
        "formats": [item.strip() for item in args.formats.split(",") if item.strip()]
        if args.formats
        else None,
        "backend": args.backend,
    }
    try:
        parameters = inspect.signature(module.build).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "no_gates" in parameters:
        kwargs["no_gates"] = args.no_gates
    if "preview" in parameters and getattr(args, "no_preview", False):
        kwargs["preview"] = False
    if "drawing" in parameters and getattr(args, "drawing", False):
        kwargs["drawing"] = True
    return kwargs


def _optional_import(module_name: str) -> Optional[Any]:
    """Import an optional wave module without turning its absence into a build failure."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        action = "slicing" if module_name.startswith("phimech.print") else "viewer"
        print(
            "warning: {}; {} is not available ({})".format(action + " skipped", module_name, exc),
            file=sys.stderr,
        )
        return None


def _stl_paths_for_parts(result: Any) -> Dict[str, str]:
    """Find the rendered STL for each printable part in a pipeline report."""
    printable = _field(result, "printable_parts", [])
    if not isinstance(printable, (list, tuple, set)):
        printable = []
    printable_names = {str(name) for name in printable}
    if not printable_names:
        parts = _field(result, "parts", {})
        if isinstance(parts, dict):
            printable_names = {
                str(name)
                for name, value in parts.items()
                if _field(value, "printable", False)
            }
    paths: Dict[str, str] = {}
    outputs = _field(result, "outputs", [])
    if isinstance(outputs, (list, tuple)):
        for output in outputs:
            if str(_field(output, "fmt", "")).lower().lstrip(".") != "stl":
                continue
            part = str(_field(output, "part", ""))
            path = _field(output, "path", None)
            if part in printable_names and path:
                paths[part] = str(path)
    return paths


def _slice_build_outputs(result: Any, args: argparse.Namespace) -> Tuple[Any, bool]:
    """Slice printable STLs and fold numeric slicer results into the build report."""
    if _pipeline_failed(result, ignore_gates=args.no_gates):
        return result, False
    slicer = _optional_import("phimech.print.slicer")
    if slicer is None:
        if isinstance(result, dict):
            result.setdefault("warnings", []).append(
                "slicing skipped: phimech.print.slicer is not available yet"
            )
        return result, False
    available = getattr(slicer, "available", None)
    if callable(available):
        try:
            if not available():
                if isinstance(result, dict):
                    result.setdefault("warnings", []).append(
                        "slicing skipped: the configured slicer is not available"
                    )
                return result, False
        except Exception as exc:
            if isinstance(result, dict):
                result.setdefault("warnings", []).append(
                    "slicing skipped: could not probe the slicer ({})".format(exc)
                )
            return result, False

    records: Dict[str, Any] = {}
    failures = False
    for part, stl in sorted(_stl_paths_for_parts(result).items()):
        try:
            sliced = slicer.slice_stl(
                stl,
                args.output,
                material="PETG",
                quality="standard",
                printer="qidi-max4",
            )
            records[part] = _jsonable(sliced)
        except Exception as exc:
            failures = True
            records[part] = {
                "ok": False,
                "stl": stl,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
    if isinstance(result, dict):
        slice_report = {
            "ok": not failures,
            "material": "PETG",
            "quality": "standard",
            "printer": "qidi-max4",
            "parts": records,
        }
        result["print"] = slice_report
        # Keep the explicit plural key for CLI consumers while the viewer reads `print`.
        result["slices"] = slice_report
        if not records:
            failures = True
            slice_report["ok"] = False
            slice_report["error"] = "no printable STL outputs were found"
        try:
            report_module = importlib.import_module("phimech.synth.report")
            report_module.write_report(args.output, result)
        except Exception as exc:
            result.setdefault("warnings", []).append(
                "could not rewrite the report with slice results: {}".format(exc)
            )
    return result, failures


def _handle_synth_build(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.synth.pipeline")
    if module is None:
        return 2
    result = module.build(args.file, args.output, **_build_kwargs(module, args))
    slice_failed = False
    if getattr(args, "slice", False):
        result, slice_failed = _slice_build_outputs(result, args)
    if args.json:
        _print_json(result)
    else:
        _print_parts(result)
        _print_gates(result)
        _print_messages(result)
        report = _report_path(result)
        if report:
            print("report: {}".format(report))
        slices = _field(result, "slices", None)
        if slices is not None:
            print("slices: {}".format("PASS" if _field(slices, "ok", False) else "FAIL"))
    return 1 if slice_failed or _pipeline_failed(result, ignore_gates=args.no_gates) else 0


def _read_node(path: str) -> Optional[Any]:
    """Load a CSG AST through its canonical JSON decoder."""
    ast = _lazy_import("phimech.csg.ast")
    if ast is None:
        return None
    return ast.from_json(Path(path).read_text(encoding="utf-8"))


def _backend_for(name: str, fmt: str) -> Optional[Any]:
    """Select a named backend or the first available backend for an output format."""
    base = _lazy_import("phimech.backends.base")
    if base is None:
        return None
    return base.for_format(fmt) if name == "auto" else base.get(name)


def _handle_render(args: argparse.Namespace) -> int:
    node = _read_node(args.file)
    if node is None:
        return 2
    output = Path(args.output).expanduser().resolve()
    fmt = output.suffix.lstrip(".").lower()
    backend = _backend_for(args.backend, fmt)
    if backend is None:
        return 2
    rendered = backend.render(node, output)
    if args.json:
        _print_json(rendered)
    else:
        print("rendered: {}".format(rendered.path))
        print("backend: {}".format(rendered.backend))
        print("bytes: {}".format(rendered.bytes))
    return 0


def _handle_measure(args: argparse.Namespace) -> int:
    node = _read_node(args.file)
    if node is None:
        return 2
    backend = _backend_for(args.backend, "stl")
    if backend is None:
        return 2
    measurement = backend.measure(node)
    if args.json:
        _print_json(measurement)
    else:
        bbox = measurement.bbox
        print("bbox: {} → {}".format(bbox.min, bbox.max))
        print("volume: {} mm^3".format(measurement.volume))
        print("area: {} mm^2".format(measurement.area))
    return 0


def _handle_emit(args: argparse.Namespace) -> int:
    node = _read_node(args.file)
    if node is None:
        return 2
    emitter = _lazy_import("phimech.backends.scad_emit")
    if emitter is None:
        return 2
    source = emitter.emit(node)
    if args.json:
        _print_json({"source": source})
    else:
        sys.stdout.write(source)
    return 0


def _handle_pcb_inspect(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.components.pcb")
    if module is None:
        return 2
    result = module.inspect(args.file)
    if args.json:
        _print_json(result)
    else:
        _print_json(result)
    return 0


def _handle_parts_list(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.components.library")
    if module is None:
        return 2
    parts = module.list_parts()
    if args.category:
        parts = [item for item in parts if item.get("category") == args.category]
    if args.json:
        _print_json(parts)
    else:
        print("key                         category  name")
        for item in parts:
            print("{:<27} {:<9} {}".format(item["key"], item["category"], item["name"]))
    return 0


def _handle_parts_show(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.components.library")
    if module is None:
        return 2
    spec = module.get(args.key)
    envelope = module.make(args.key, ref="__preview__", **dict(spec.params))
    result = {
        "key": spec.key,
        "name": spec.name,
        "category": spec.category,
        "params": dict(spec.params),
        "size": list(envelope.size),
        "envelope": envelope.to_dict(),
    }
    if args.json:
        _print_json(result)
    else:
        print("{} ({})".format(result["name"], result["key"]))
        print("category: {}".format(result["category"]))
        print("size: {}".format(result["size"]))
        print("params: {}".format(result["params"]))
    return 0


def _handle_print_slice(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.print.slicer")
    if module is None:
        return 2
    output = args.output
    if output is None:
        output = str(Path(args.stl).expanduser().resolve().parent / "sliced")
    result = module.slice_stl(
        args.stl,
        output,
        material=args.material,
        quality=args.quality,
        printer="qidi-max4",
    )
    if args.json:
        _print_json(result)
    else:
        data = _jsonable(result)
        print("gcode: {}".format(_field(data, "gcode_3mf", "")))
        print("print time: {} s".format(_field(data, "print_time_s", "")))
        print("filament: {} g".format(_field(data, "filament_g", "")))
        print("layers: {}".format(_field(data, "layer_count", "")))
    return 0


def _handle_print_send(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.print.printer")
    if module is None:
        return 2
    upload = module.upload_gcode(args.file, base_url=args.url or None)
    result = {"file": str(Path(args.file).expanduser()), "upload": upload, "print_started": False}
    if args.json:
        _print_json(result)
    else:
        _print_json(result)
        print("upload complete; no print was started")
    return 0


def _handle_print_status(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.print.printer")
    if module is None:
        return 2
    result = module.printer_info(base_url=args.url or None)
    if args.json:
        _print_json(result)
    else:
        _print_json(result)
    return 0


def _handle_print_profiles(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.print.profiles")
    if module is None:
        return 2
    result = {
        "materials": module.list_materials(),
        "qualities": module.list_qualities(),
    }
    if args.json:
        _print_json(result)
    else:
        print("materials: {}".format(", ".join(str(item) for item in result["materials"])))
        print("qualities: {}".format(", ".join(str(item) for item in result["qualities"])))
    return 0


def _handle_view(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.viewer.server")
    if module is None:
        return 2
    out_dir = str(Path(args.out_dir).expanduser().resolve())
    httpd, url = module.serve(out_dir, host=args.host, port=args.port)
    print("VIEWER URL: {}".format(url))
    print(str(url))
    sys.stdout.flush()
    if not args.no_open:
        webbrowser.open(str(url))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("viewer stopped")
    finally:
        # serve_forever has already unwound on Ctrl-C; calling shutdown from this same
        # thread can wait on itself on some Python versions.
        httpd.server_close()
    return 0


def _handle_service_install(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.viewer.service")
    if module is None:
        return 2
    port = args.port or module.DEFAULT_PORT
    ok, notes = module.install(out_dir=args.out_dir, port=port)
    for note in notes:
        print(note)
    if args.links:
        for path in module.write_link_folder(Path(args.links), port=port):
            print("wrote {}".format(path))
    print("")
    for title, link, _ in module.links(port=port):
        print("  {:<10} {}".format(title, link))
    # Not "ok" on its own: launchd reporting success and the port answering are different claims.
    info = module.status(port=port)
    print("\n{} {}".format("✓" if info["responding"] else "✗",
                            "serving" if info["responding"] else "not answering yet"))
    return 0 if ok else 1


def _handle_service_status(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.viewer.service")
    if module is None:
        return 2
    info = module.status()
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0 if info["responding"] else 1
    for key in ("label", "plist", "installed", "loaded", "responding", "url", "out_dir"):
        print("  {:<16} {}".format(key, info[key]))
    if not info["responding"]:
        print("\nnot answering. install it with: bin/phimech service install")
    return 0 if info["responding"] else 1


def _handle_service_uninstall(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.viewer.service")
    if module is None:
        return 2
    ok, notes = module.uninstall()
    for note in notes:
        print(note)
    return 0 if ok else 1


def _handle_service_links(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.viewer.service")
    if module is None:
        return 2
    port = args.port or module.DEFAULT_PORT
    for path in module.write_link_folder(Path(args.folder), port=port):
        print("wrote {}".format(path))
    return 0


def _ir_schema_path() -> Path:
    return _repository() / "phimech" / "ir" / "design-ir.schema.json"


def _handle_ir_validate(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.ir.ir")
    if module is None:
        return 2
    _, errors = module.load_and_validate(args.file)
    result = {"file": str(args.file), "valid": not bool(errors), "errors": errors}
    if args.json:
        _print_json(result)
    elif errors:
        for error in errors:
            print(error)
    else:
        print("valid: {}".format(args.file))
    return 1 if errors else 0


def _handle_ir_params(args: argparse.Namespace) -> int:
    """List a design's editable parameters, with bounds - what a UI turns into sliders."""
    module = _lazy_import("phimech.ir.ir")
    if module is None:
        return 2
    design, errors = module.load_and_validate(args.file)
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
    if args.json:
        _print_json({"file": str(args.file), "parameters": rows, "errors": errors})
        return 1 if errors else 0
    if not rows:
        print("no parameters declared in {}".format(args.file))
        print("add a `parameters` block to make a number editable; see phimech/ir/README.md")
        return 0
    print("%-16s %10s %10s %10s %6s  %s" % ("name", "value", "min", "max", "unit", "description"))
    for row in rows:
        show = lambda item: "-" if item is None else ("%g" % item)
        print(
            "%-16s %10s %10s %10s %6s  %s"
            % (
                row["name"], show(row["value"]), show(row["min"]), show(row["max"]),
                row["unit"], row["description"],
            )
        )
    for error in errors:
        print(error)
    return 1 if errors else 0


def _handle_ir_set(args: argparse.Namespace) -> int:
    """Edit one parameter in place - the supported manual override.

    Deliberately narrow: it changes a declared number within its declared bounds and nothing else.
    Editing derived geometry would leave a design its own IR can no longer reproduce.
    """
    module = _lazy_import("phimech.ir.ir")
    if module is None:
        return 2
    assignment = str(args.assignment)
    if "=" not in assignment:
        print("expected NAME=VALUE, got %r" % assignment)
        return 2
    name, _, raw = assignment.partition("=")
    name, raw = name.strip(), raw.strip()
    try:
        value = float(raw)
    except ValueError:
        print("%s is not a number" % raw)
        return 2
    if value.is_integer():
        value = int(value)
    try:
        result = module.set_parameter(args.file, name, value, allow_new=args.new)
    except (KeyError, ValueError, OSError) as exc:
        message = str(exc).strip('"')
        if args.json:
            _print_json({"ok": False, "error": message})
        else:
            print(message)
        return 1
    _, errors = module.load_and_validate(args.file)
    result["errors"] = errors
    result["ok"] = not errors
    if args.json:
        _print_json(result)
    else:
        print("%s: %s -> %s" % (name, result["previous"], result["value"]))
        for error in errors:
            print(error)
        if not errors:
            print("design still validates; rebuild to re-run the gates")
    return 1 if errors else 0


def _handle_brep_ops(args: argparse.Namespace) -> int:
    """List the feature-program vocabulary. A closed op set is the point, so print it."""
    module = _lazy_import("phimech.brep.program")
    if module is None:
        return 2
    if args.json:
        _print_json({"version": module.VERSION, "ops": sorted(module.OPS),
                     "edge_selectors": list(module.EDGE_SELECTORS),
                     "face_selectors": list(module.FACE_SELECTORS),
                     "thread_pitch_mm": module.THREAD_PITCH})
        return 0
    print("feature program version {}\n".format(module.VERSION))
    print("ops:             {}".format(", ".join(sorted(module.OPS))))
    print("edge selectors:  {}".format(", ".join(module.EDGE_SELECTORS)))
    print("face selectors:  {}".format(", ".join(module.FACE_SELECTORS)))
    print("tapped sizes:    {}".format(", ".join(sorted(module.THREAD_PITCH))))
    return 0


def _handle_brep_build(args: argparse.Namespace) -> int:
    """Build a feature program in the B-rep kernel."""
    kernel = _lazy_import("phimech.brep.kernel")
    program_mod = _lazy_import("phimech.brep.program")
    if kernel is None or program_mod is None:
        return 2
    source = Path(args.program).expanduser()
    try:
        program = json.loads(source.read_text())
    except OSError as exc:
        print("error: cannot read {}: {}".format(source, exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print("error: {} is not valid JSON: {}".format(source, exc), file=sys.stderr)
        return 2

    outputs: Dict[str, str] = {}
    for target in args.output:
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        outputs[path.suffix.lstrip(".").lower()] = str(path)
    if not outputs:
        print("error: at least one -o/--output is required", file=sys.stderr)
        return 2

    try:
        result = kernel.build(program, outputs, timeout=args.timeout)
    except program_mod.ProgramError as exc:
        print("error: invalid program: {}".format(exc), file=sys.stderr)
        return 2
    except kernel.BrepError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    if args.json:
        _print_json(result.to_dict())
    else:
        m = result.measurements
        print("built with {} {}".format(result.kernel, result.kernel_version or ""))
        print(program_mod.describe(program))
        size = m.get("bbox", {}).get("size")
        print("  {} faces, {} edges, {} solid(s)".format(
            m.get("faces"), m.get("edges"), m.get("solids")))
        print("  {} mm bbox, {} mm3, {} mm2".format(
            size, m.get("volume_mm3"), m.get("area_mm2")))
        for fmt, path in sorted(result.written.items()):
            print("  wrote {}: {}".format(fmt, path))
    return 0


def _handle_brep_setup(args: argparse.Namespace) -> int:
    """Create build/brep-venv with the geometry kernel in it."""
    kernel = _lazy_import("phimech.brep.kernel")
    if kernel is None:
        return 2
    print("creating the B-rep venv (build123d pulls ~900 MB of OpenCascade; this is slow)",
          file=sys.stderr)
    outcome = kernel.setup(python=args.python)
    if args.json:
        _print_json(outcome)
    else:
        for step in outcome["steps"]:
            print("{}  {}".format("ok  " if step["ok"] else "FAIL", step["cmd"]))
            if step.get("error"):
                print("      {}".format(step["error"]))
        print("venv: {}".format(outcome["venv"]))
    return 0 if outcome["ok"] else 1


def _handle_interop_step(args: argparse.Namespace) -> int:
    """Report whether a STEP file transfers usefully into SolidWorks.

    Exit code is 0 when ready, 1 when a hard finding fails, and 2 when readiness could not be
    determined -- the tri-state must survive into the shell, because "unknown" is not "fine".
    """
    module = _lazy_import("phimech.interop.step")
    if module is None:
        return 2
    path = Path(args.file).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    facts = module.inspect(path)
    verdict = module.readiness(facts, expect_mm=not args.allow_any_unit)

    if args.json:
        _print_json({"facts": facts.to_dict(), "readiness": verdict})
    else:
        ready = verdict["ready"]
        mark = {True: "READY", False: "NOT READY", None: "UNKNOWN"}[ready]
        print(f"{mark}  {path}")
        print(f"  {verdict['summary']}")
        proto = f"AP{facts.protocol}" if facts.protocol else "unknown schema"
        origin = facts.originating_system or "unknown system"
        print(f"  {proto} - {facts.body_kind} - {facts.length_unit or '?'} - "
              f"from {origin} - {facts.size_bytes} bytes")
        print(f"  {facts.faces} faces, {facts.edges} edges, "
              f"{facts.analytic_surfaces} analytic + {facts.spline_surfaces} spline surfaces")
        for finding in verdict["findings"]:
            # Severity decides the glyph, not `ok`: a failed *warn* is a caveat, and printing
            # it as FAIL next to an overall READY verdict just reads as a contradiction.
            if finding["ok"] is True:
                glyph = "ok  "
            elif finding["ok"] is None:
                glyph = "??  "
            else:
                glyph = "FAIL" if finding["severity"] == "fail" else "warn"
            print(f"    {glyph} {finding['name']}: {finding['detail']}")
        print(f"  note: {verdict['caveat']}")
    return {True: 0, False: 1, None: 2}[verdict["ready"]]


def _handle_bench_prompts(args: argparse.Namespace) -> int:
    """Print the briefs for a model or harness to design against."""
    module = _lazy_import("phimech.bench.run")
    if module is None:
        return 2
    if args.json:
        grade = _lazy_import("phimech.bench.grade")
        if grade is None:
            return 2
        _print_json(grade.load_suite(args.suite))
    else:
        print(module.prompts(args.suite))
    return 0


def _handle_bench_grade(args: argparse.Namespace) -> int:
    """Build and score a directory of candidate designs.

    Exit code is the number of failed cases, so a harness can gate on it.
    """
    module = _lazy_import("phimech.bench.run")
    if module is None:
        return 2
    out_root = Path(args.out) if args.out else Path(args.candidates) / "_build"
    outcome = module.run_suite(
        args.candidates,
        out_root,
        suite_path=args.suite,
        only=args.case or None,
        build=not args.no_build,
    )
    if args.json:
        _print_json({k: v for k, v in outcome.items() if k != "table"})
    else:
        print(outcome["table"])
        for result in outcome["results"]:
            failed = [c for c in (result.get("checks") or []) if not c.get("passed")]
            if not failed:
                continue
            print("\n%s" % result["id"])
            for check in failed:
                print("  %-28s %s" % (check["name"], check["detail"]))
    return len(outcome["summary"]["failed_cases"])


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _handle_ir_schema(args: argparse.Namespace) -> int:
    schema = _read_json_file(_ir_schema_path())
    _print_json(schema)
    return 0


def _example_path(name: str) -> Path:
    """Resolve an example name without allowing it to escape the repository examples dir."""
    clean = Path(name).name
    if not clean.endswith(".json"):
        clean += ".json"
    return _repository() / "examples" / clean


def _handle_ir_example(args: argparse.Namespace) -> int:
    path = _example_path(args.name)
    if not path.is_file():
        print("error: example not found: {}".format(path), file=sys.stderr)
        return 1
    if args.json:
        _print_json(_read_json_file(path))
    else:
        sys.stdout.write(path.read_text(encoding="utf-8"))
    return 0


def _register_arguments(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    if args.dry_run:
        values.append("--dry-run")
    values.extend(["--client", args.client])
    return values


def _handle_mcp_register(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.mcp.register")
    if module is None:
        return 2
    if args.json:
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = int(module.main(_register_arguments(args)))
        _print_json({"returncode": code, "stdout": output.getvalue(), "stderr": errors.getvalue()})
        return code
    return int(module.main(_register_arguments(args)))


def _handle_mcp_serve(args: argparse.Namespace) -> int:
    del args
    try:
        runpy.run_module("phimech.mcp.server", run_name="__main__")
    except ImportError as exc:
        print(
            "error: phimech.mcp.server not available yet ({})".format(exc),
            file=sys.stderr,
        )
        return 2
    return 0


def _tool_names(module: Any) -> List[str]:
    """Best-effort introspection for FastMCP versions with different private managers."""
    names: List[str] = []
    declared = getattr(module, "TOOLS", None)
    if isinstance(declared, dict):
        names.extend(str(name) for name in declared)
    elif isinstance(declared, (list, tuple, set)):
        for item in declared:
            names.append(str(_field(item, "name", item)))
    server = getattr(module, "mcp", None)
    manager = getattr(server, "_tool_manager", None)
    for attribute in ("_tools", "tools"):
        values = getattr(manager, attribute, None)
        if isinstance(values, dict):
            names.extend(str(name) for name in values)
        elif isinstance(values, (list, tuple, set)):
            names.extend(str(_field(item, "name", item)) for item in values)
    return sorted(set(names))


def _handle_mcp_tools(args: argparse.Namespace) -> int:
    module = _lazy_import("phimech.mcp.server")
    if module is None:
        return 2
    names = _tool_names(module)
    if args.json:
        _print_json({"tools": names})
    else:
        for name in names:
            print(name)
    return 0


def _run_test_file(path: Path, repository: Path, use_pytest: bool, verbose: bool) -> Dict[str, Any]:
    """Run one test file in a child process so one failure cannot poison later files."""
    command = (
        [sys.executable, "-m", "pytest", str(path)]
        if use_pytest
        else [sys.executable, str(path)]
    )
    started = time.monotonic()
    environment = os.environ.copy()
    old_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository) + ((os.pathsep + old_path) if old_path else "")
    completed = subprocess.run(
        command,
        cwd=str(repository),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    record: Dict[str, Any] = {
        "file": str(path.relative_to(repository)),
        "passed": completed.returncode == 0,
        "seconds": round(elapsed, 3),
    }
    if completed.returncode != 0:
        record["output"] = completed.stdout[-4000:]
    if verbose:
        print(
            "{} {} ({:.2f}s)".format(
                "PASS" if record["passed"] else "FAIL", record["file"], elapsed
            )
        )
        if completed.returncode != 0 and completed.stdout:
            print(completed.stdout.rstrip())
    return record


def _run_pyflakes(repository: Path, verbose: bool) -> Dict[str, Any]:
    """Run the same lint CI runs, so CI is never the first place a dead import shows up.

    Reported as a skip rather than a failure when pyflakes is not installed: the core must stay
    importable under python 3.9 and FreeCAD's bundled interpreter, neither of which is
    guaranteed to have it.  CI always has it, so the gate cannot be silently lost.
    """
    record: Dict[str, Any] = {"name": "pyflakes"}
    if importlib.util.find_spec("pyflakes") is None:
        record.update(passed=True, skipped=True)
        if verbose:
            print("SKIP pyflakes (not installed; CI still enforces it)")
        return record
    completed = subprocess.run(
        [sys.executable, "-m", "pyflakes", "phimech", "tests"],
        cwd=str(repository),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    record["passed"] = completed.returncode == 0
    if not record["passed"]:
        record["output"] = completed.stdout[-4000:]
    if verbose:
        print("{} pyflakes".format("PASS" if record["passed"] else "FAIL"))
        if not record["passed"] and completed.stdout:
            print(completed.stdout.rstrip())
    return record


def _handle_check(args: argparse.Namespace) -> int:
    repository = _repository()
    test_paths = sorted((repository / "tests").glob("test_*.py"))
    use_pytest = bool(args.pytest and importlib.util.find_spec("pytest") is not None)
    if args.pytest and not use_pytest and not args.json:
        print("pytest is not available; running each test file directly")
    tests = [
        _run_test_file(path, repository, use_pytest, verbose=not args.json)
        for path in test_paths
    ]
    failures = [item for item in tests if not item["passed"]]

    example_records: List[Dict[str, Any]] = []
    try:
        ir_module = importlib.import_module("phimech.ir.ir")
    except ImportError:
        ir_module = None
    if ir_module is None:
        if not args.json:
            print("SKIP examples (phimech.ir.ir not available yet)")
    else:
        for path in sorted((repository / "examples").glob("*.json")):
            try:
                _, errors = ir_module.load_and_validate(path)
            except Exception as exc:
                errors = [str(exc)]
            record = {"file": str(path.relative_to(repository)), "valid": not bool(errors)}
            if errors:
                record["errors"] = errors
                failures.append(record)
            example_records.append(record)
            if not args.json:
                print("{} {}".format("PASS" if not errors else "FAIL", record["file"]))

    lint = _run_pyflakes(repository, verbose=not args.json)
    if not lint["passed"]:
        failures.append(lint)

    doctor_module = _lazy_import("phimech.doctor")
    doctor_report: Any = None
    doctor_code = 2
    if doctor_module is not None:
        doctor_report = doctor_module.doctor()
        doctor_code = 0 if doctor_report.get("ok") else 1
        if not args.json:
            doctor_module._print_table(doctor_report)

    report = {
        "tests": tests,
        "examples": example_records,
        "lint": lint,
        "doctor": doctor_report,
        "passed": not failures and doctor_code == 0,
    }
    if args.json:
        _print_json(report)
    return 0 if report["passed"] else 1


def _handle_version(args: argparse.Namespace) -> int:
    result = {"version": __version__, "git": git_describe()}
    if args.json:
        _print_json(result)
    else:
        print(__version__)
    return 0


def _handle_help(args: argparse.Namespace) -> int:
    args._root_parser.print_help()
    return 0


# --------------------------------------------------------------------------------------
# parser and entry point
# --------------------------------------------------------------------------------------


#: Subcommand hooks contributed by packages that own their own CLI surface. Each entry is a
#: ``(module, attribute)`` pair naming a ``register_cli(subparsers)`` function. They are wired
#: here, in one place, so a package can ship its command without editing this file's parser
#: body - and so an import failure in one of them degrades to a missing command with a stderr
#: note instead of taking the whole CLI down.
CLI_HOOKS: Tuple[Tuple[str, str], ...] = (
    ("phimech.adapters.solidworks.__main__", "register_cli"),
    ("phimech.synth.preview", "register_cli"),  # `phimech preview sheet|compare|bench`
    ("phimech.synth.drawing", "register_cli"),  # `phimech drawing FILE|DIR -o OUT.svg`
    ("phimech.mcp.tools_drawing_batch", "register_cli"),  # `phimech drawings PATH... -o DIR`
)

#: Hooks that hang a subcommand under `phimech interop` rather than at the top level.
INTEROP_HOOKS: Tuple[Tuple[str, str], ...] = (
    ("phimech.interop.edit_risk", "register_cli"),   # `phimech interop risk FILE`
)

#: Hooks that hang a subcommand under `phimech ir`.
IR_HOOKS: Tuple[Tuple[str, str], ...] = (
    ("phimech.ir.aliases", "register_cli"),   # `phimech ir normalise FILE [--write]`
)


def _register_hooks(
    commands: argparse._SubParsersAction,
    hooks: Tuple[Tuple[str, str], ...] = CLI_HOOKS,
) -> None:
    """Attach every hook's subcommand, tolerating a broken contributor."""
    for module_name, attribute in hooks:
        try:
            register = getattr(importlib.import_module(module_name), attribute)
            register(commands)
        except Exception as exc:  # pragma: no cover - only reachable with a broken checkout
            print("phimech: hook {}.{} not loaded: {}".format(module_name, attribute, exc),
                  file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser in one place so ``phimech help`` and argparse help agree."""
    parser = argparse.ArgumentParser(
        prog="phimech",
        description="PhiMech mechanical CAD synthesis and inspection tools",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    doctor_parser = commands.add_parser("doctor", help="probe Python, venv, CAD backends, and MCP")
    _add_json(doctor_parser)
    doctor_parser.set_defaults(handler=_handle_doctor)

    synth_parser = commands.add_parser("synth", help="compile a design IR into an assembly")
    synth_commands = synth_parser.add_subparsers(dest="synth_command", metavar="COMMAND")
    synth_check = synth_commands.add_parser(
        "check", help="validate and place an IR without writing files"
    )
    synth_check.add_argument("file")
    _add_json(synth_check)
    synth_check.set_defaults(handler=_handle_synth_check)
    synth_build = synth_commands.add_parser("build", help="build enclosure output files")
    synth_build.add_argument("file")
    synth_build.add_argument("-o", "--output", required=True)
    synth_build.add_argument("--formats", help="comma-separated output formats")
    synth_build.add_argument("--backend", choices=("auto", "openscad", "freecad"), default="auto")
    synth_build.add_argument(
        "--slice",
        action="store_true",
        help="slice each printable STL after a successful build when the slicer is available",
    )
    synth_build.add_argument("--no-gates", action="store_true")
    synth_build.add_argument(
        "--no-preview", action="store_true",
        help="skip the 7-view preview.png sheet (evidence for a look, not a gate; ~1 s)",
    )
    synth_build.add_argument(
        "--drawing", action="store_true",
        help="also write the dimensioned inspection sheet of the main part (drawing.svg, "
             "+ .pdf/.png when a converter exists) and record its checks in report.json "
             "(advisory, never a gate; one OpenSCAD run per z level)",
    )
    _add_json(synth_build)
    synth_build.set_defaults(handler=_handle_synth_build)

    render_parser = commands.add_parser("render", help="render a CSG AST to a CAD file")
    render_parser.add_argument("file")
    render_parser.add_argument("-o", "--output", required=True)
    render_parser.add_argument("--backend", choices=("auto", "openscad", "freecad"), default="auto")
    _add_json(render_parser)
    render_parser.set_defaults(handler=_handle_render)

    measure_parser = commands.add_parser("measure", help="measure a CSG AST")
    measure_parser.add_argument("file")
    measure_parser.add_argument(
        "--backend", choices=("auto", "openscad", "freecad"), default="auto"
    )
    _add_json(measure_parser)
    measure_parser.set_defaults(handler=_handle_measure)

    emit_parser = commands.add_parser("emit", help="emit OpenSCAD source for a CSG AST")
    emit_parser.add_argument("file")
    _add_json(emit_parser)
    emit_parser.set_defaults(handler=_handle_emit)

    pcb_parser = commands.add_parser("pcb", help="inspect PCB component envelopes")
    pcb_commands = pcb_parser.add_subparsers(dest="pcb_command", metavar="COMMAND")
    pcb_inspect = pcb_commands.add_parser("inspect", help="inspect a KiCad board")
    pcb_inspect.add_argument("file")
    _add_json(pcb_inspect)
    pcb_inspect.set_defaults(handler=_handle_pcb_inspect)

    parts_parser = commands.add_parser("parts", help="browse the parametric component library")
    parts_commands = parts_parser.add_subparsers(dest="parts_command", metavar="COMMAND")
    parts_list = parts_commands.add_parser("list", help="list library parts")
    parts_list.add_argument("--category")
    _add_json(parts_list)
    parts_list.set_defaults(handler=_handle_parts_list)
    parts_show = parts_commands.add_parser("show", help="show one library part")
    parts_show.add_argument("key")
    _add_json(parts_show)
    parts_show.set_defaults(handler=_handle_parts_show)

    print_parser = commands.add_parser("print", help="slice parts and upload files to a printer")
    print_commands = print_parser.add_subparsers(dest="print_command", metavar="COMMAND")
    print_slice = print_commands.add_parser(
        "slice", help="slice an STL into a printer-ready .gcode.3mf"
    )
    print_slice.add_argument("stl")
    print_slice.add_argument("-o", "--output", dest="output", help="output directory")
    print_slice.add_argument("--material", default="PETG")
    print_slice.add_argument("--quality", default="standard")
    _add_json(print_slice)
    print_slice.set_defaults(handler=_handle_print_slice)

    print_send = print_commands.add_parser(
        "send", help="upload a .gcode.3mf only; never start a print"
    )
    print_send.add_argument("file")
    print_send.add_argument("--url", help="Moonraker base URL (default: PHIMECH_PRINTER_URL)")
    _add_json(print_send)
    print_send.set_defaults(handler=_handle_print_send)

    print_status = print_commands.add_parser(
        "status", help="read printer status without moving or heating"
    )
    print_status.add_argument("--url", help="Moonraker base URL (default: PHIMECH_PRINTER_URL)")
    _add_json(print_status)
    print_status.set_defaults(handler=_handle_print_status)

    print_profiles = print_commands.add_parser(
        "profiles", help="list bundled printer materials and quality profiles"
    )
    _add_json(print_profiles)
    print_profiles.set_defaults(handler=_handle_print_profiles)

    view_parser = commands.add_parser(
        "view", help="serve build outputs in a browser and block until Ctrl-C"
    )
    view_parser.add_argument("out_dir", nargs="?", default=".")
    view_parser.add_argument("--port", type=int, default=8765)
    view_parser.add_argument("--host", default="127.0.0.1")
    view_parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    view_parser.set_defaults(handler=_handle_view)

    service_parser = commands.add_parser(
        "service", help="keep the app running at one stable URL, across reboots"
    )
    service_commands = service_parser.add_subparsers(dest="service_command", metavar="COMMAND")
    service_install = service_commands.add_parser(
        "install", help="install and start the login service"
    )
    service_install.add_argument("out_dir", nargs="?", default=None,
                                help="build directory to serve (default: ./out)")
    service_install.add_argument("--port", type=int, default=None, help="pinned port")
    service_install.add_argument("--links", metavar="FOLDER", default=None,
                                help="also write a folder of clickable links here")
    service_install.set_defaults(handler=_handle_service_install)
    service_status = service_commands.add_parser(
        "status", help="is it installed, loaded, and answering?"
    )
    _add_json(service_status)
    service_status.set_defaults(handler=_handle_service_status)
    service_uninstall = service_commands.add_parser("uninstall", help="stop and remove it")
    service_uninstall.set_defaults(handler=_handle_service_uninstall)
    service_links = service_commands.add_parser(
        "links", help="write a folder of clickable links and a text fallback"
    )
    service_links.add_argument("folder")
    service_links.add_argument("--port", type=int, default=None)
    service_links.set_defaults(handler=_handle_service_links)

    ir_parser = commands.add_parser("ir", help="validate and inspect design IR")
    ir_commands = ir_parser.add_subparsers(dest="ir_command", metavar="COMMAND")
    ir_validate = ir_commands.add_parser("validate", help="validate one design file")
    ir_validate.add_argument("file")
    _add_json(ir_validate)
    ir_validate.set_defaults(handler=_handle_ir_validate)
    ir_schema = ir_commands.add_parser("schema", help="print the design IR schema")
    _add_json(ir_schema)
    ir_schema.set_defaults(handler=_handle_ir_schema)
    ir_example = ir_commands.add_parser("example", help="print an example design")
    ir_example.add_argument("name")
    _add_json(ir_example)
    ir_example.set_defaults(handler=_handle_ir_example)
    ir_params = ir_commands.add_parser("params", help="list a design's editable parameters")
    ir_params.add_argument("file")
    _add_json(ir_params)
    ir_params.set_defaults(handler=_handle_ir_params)
    ir_set = ir_commands.add_parser("set", help="set one parameter: NAME=VALUE")
    ir_set.add_argument("file")
    ir_set.add_argument("assignment", metavar="NAME=VALUE")
    ir_set.add_argument(
        "--new", action="store_true", help="allow declaring a parameter that does not exist yet"
    )
    _add_json(ir_set)
    ir_set.set_defaults(handler=_handle_ir_set)
    _register_hooks(ir_commands, IR_HOOKS)

    brep_parser = commands.add_parser(
        "brep", help="build with the B-rep kernel: real fillets, chamfers, non-faceted STEP"
    )
    brep_commands = brep_parser.add_subparsers(dest="brep_command", metavar="COMMAND")
    brep_ops = brep_commands.add_parser("ops", help="list the feature-program vocabulary")
    _add_json(brep_ops)
    brep_ops.set_defaults(handler=_handle_brep_ops)
    brep_build = brep_commands.add_parser("build", help="build a feature program")
    brep_build.add_argument("program", help="path to a feature-program JSON file")
    brep_build.add_argument(
        "-o", "--output", action="append", default=[],
        help="output file; the format is taken from the suffix (.step, .stl, .brep). Repeatable",
    )
    brep_build.add_argument(
        "--timeout", type=float, default=300.0, help="kernel timeout in seconds (default 300)"
    )
    _add_json(brep_build)
    brep_build.set_defaults(handler=_handle_brep_build)
    brep_setup = brep_commands.add_parser("setup", help="create build/brep-venv")
    brep_setup.add_argument(
        "--python", default=None,
        help="base interpreter (default python3.12; build123d needs >=3.10,<3.13)",
    )
    _add_json(brep_setup)
    brep_setup.set_defaults(handler=_handle_brep_setup)

    interop_parser = commands.add_parser(
        "interop", help="check whether a file transfers to another CAD tool"
    )
    interop_commands = interop_parser.add_subparsers(dest="interop_command", metavar="COMMAND")
    interop_step = interop_commands.add_parser(
        "step", help="report SolidWorks readiness of a STEP file"
    )
    interop_step.add_argument("file", help="path to a .step or .stp file")
    interop_step.add_argument(
        "--allow-any-unit", action="store_true",
        help="do not fail a file whose length unit is not mm (PhiMech is mm throughout, so "
             "this only makes sense when inspecting a third-party file you intend to convert)",
    )
    _add_json(interop_step)
    interop_step.set_defaults(handler=_handle_interop_step)
    _register_hooks(interop_commands, INTEROP_HOOKS)

    bench_parser = commands.add_parser("bench", help="score AI generation against the suite")
    bench_commands = bench_parser.add_subparsers(dest="bench_command", metavar="COMMAND")
    bench_prompts = bench_commands.add_parser("prompts", help="print the design briefs")
    bench_prompts.add_argument("--suite", default=None, help="alternative suite JSON")
    _add_json(bench_prompts)
    bench_prompts.set_defaults(handler=_handle_bench_prompts)
    bench_grade = bench_commands.add_parser("grade", help="build and score candidate designs")
    bench_grade.add_argument("candidates", help="directory of <case-id>.json designs")
    bench_grade.add_argument("-o", "--out", default=None, help="build directory")
    bench_grade.add_argument("--suite", default=None, help="alternative suite JSON")
    bench_grade.add_argument(
        "--case", action="append", help="grade only this case id (repeatable)"
    )
    bench_grade.add_argument(
        "--no-build", action="store_true",
        help="grade with IR gates only, no geometry: sub-second, but mesh-dependent checks "
             "(manifold, volume, fits_bed, measured sizes) cannot pass, so scores are not "
             "comparable to a built run",
    )
    _add_json(bench_grade)
    bench_grade.set_defaults(handler=_handle_bench_grade)

    mcp_parser = commands.add_parser("mcp", help="register and inspect the MCP server")
    mcp_commands = mcp_parser.add_subparsers(dest="mcp_command", metavar="COMMAND")
    mcp_register = mcp_commands.add_parser("register", help="write the phimech MCP client entry")
    mcp_register.add_argument("--dry-run", action="store_true")
    mcp_register.add_argument("--client", default=str(Path.home() / ".pi" / "agent" / "mcp.json"))
    _add_json(mcp_register)
    mcp_register.set_defaults(handler=_handle_mcp_register)
    mcp_serve = mcp_commands.add_parser("serve", help="run the MCP server over stdio")
    _add_json(mcp_serve)
    mcp_serve.set_defaults(handler=_handle_mcp_serve)
    mcp_tools = mcp_commands.add_parser("tools", help="list registered MCP tools")
    _add_json(mcp_tools)
    mcp_tools.set_defaults(handler=_handle_mcp_tools)

    _register_hooks(commands)

    check_parser = commands.add_parser("check", help="run tests, IR examples, and doctor")
    check_parser.add_argument("--pytest", action="store_true", help="use pytest when installed")
    _add_json(check_parser)
    check_parser.set_defaults(handler=_handle_check)

    help_parser = commands.add_parser("help", help="show this help")
    _add_json(help_parser)
    help_parser.set_defaults(handler=_handle_help)

    version_parser = commands.add_parser("version", help="print the package version")
    _add_json(version_parser)
    version_parser.set_defaults(handler=_handle_version)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse and dispatch one command, never leaking an implementation traceback to users."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args._root_parser = parser
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
