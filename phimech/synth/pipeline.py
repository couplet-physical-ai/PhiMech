"""The synth pipeline: design IR in, placed assembly + enclosure + files + report out.

This module is the only place where every other layer meets, so it owns two policies that
nothing else may duplicate:

1. **Fault tolerance.** Every stage runs inside `_stage()`. A stage that raises records
   `{'stage': ..., 'ok': False, 'error': ..., 'traceback': [...]}` and the pipeline continues
   with whatever later stages can still run. `check()`/`build()` therefore *always* return a
   report dict with `ok: False` instead of propagating an exception — an agent needs the
   partial evidence (which component's envelope failed, which backend was missing) far more
   than it needs a stack trace, and the MCP surface must never lose the report.
2. **Backend routing.** `.scad` is emitted without any binary, `.step`/`.fcstd` need FreeCAD,
   `.png` needs OpenSCAD, a part containing a STEP `Import` can only go through FreeCAD, and a
   part using an AST kind in `FreeCADBackend.UNSUPPORTED` reaches STEP through an
   OpenSCAD → STL → FreeCAD-import fallback. All FreeCAD formats of one part are written by a
   single `render_many()` call because `freecadcmd` start-up dominates its cost.

Imports of sibling slices are deliberately *lazy*, inside functions: a checkout missing (say)
`phimech.check.gates` must still be able to load, validate, place, and render.

stdlib only; imports under python 3.9.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

#: The fixed stage vocabulary. `check()` runs the first six, `build()` runs all eleven
#: (`drawing` is opt-in and otherwise recorded as skipped).
STAGES: List[str] = [
    "load",
    "validate",
    "envelopes",
    "place",
    "enclosure",
    "gates_static",
    "render",
    "measure",
    "gates_full",
    "preview",
    "drawing",
    "report",
]

_CHECK_STAGES = STAGES[:6]

#: Formats that only make sense for a whole-assembly view.
_VIEW_ONLY_FORMATS = ("png",)
#: Parts that are views rather than printable geometry.
_VIEW_PARTS = ("assembly", "exploded")
#: Formats a view part is allowed to be written in (a STEP of an exploded view is noise).
_VIEW_PART_FORMATS = ("png", "scad")
#: Formats only FreeCAD can write.
_FREECAD_ONLY = ("step", "fcstd")
_STEP_SUFFIXES = (".step", ".stp")

_DEFAULT_FORMATS = ("stl",)


# --------------------------------------------------------------------------------------
# run context
# --------------------------------------------------------------------------------------


class _Run(object):
    """Mutable accumulator for one pipeline run. Not part of the public contract."""

    def __init__(self, ir_path: Union[str, Path], out_dir: Optional[Union[str, Path]]) -> None:
        self.ir_path = str(Path(str(ir_path)).expanduser())
        self.out_dir = Path(str(out_dir)).expanduser().resolve() if out_dir else None
        self.started = time.time()
        self.design: Dict[str, Any] = {}
        self.envelopes: Dict[str, Any] = {}
        self.placement: Any = None
        self.enclosure_parts: Any = None
        self.measurements: Dict[str, Any] = {}
        #: Measurements handed back by a render (FreeCAD gives them for free); keyed by part.
        self.render_measurements: Dict[str, Any] = {}
        #: Which backend produced each render measurement, so the boolean reference of a part
        #: is measured by the same kernel as the part (see `_boolean_reference`).
        self.render_measurement_backends: Dict[str, str] = {}
        #: Per-part export round-trip diagnostics: tree measurement vs the reloaded file.
        self.roundtrip: Dict[str, Dict[str, Any]] = {}
        #: {part: [fmt, ...]} of renders that raised, so the gate stage can tell a part whose
        #: geometry export died from one that merely lost a preview PNG.
        self.render_failures: Dict[str, List[str]] = {}
        self.gate_report: Any = None
        self.outputs: List[Dict[str, Any]] = []
        self.stages: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.notes: List[str] = []
        self.stop = False  # set when validation fails: later stages are meaningless

    # -- stage bookkeeping -------------------------------------------------------------

    def stage(self, name: str, fn: Callable[[], Any], *, note: str = "") -> Tuple[bool, Any]:
        """Run one stage, timing it and converting any exception into a recorded failure."""
        started = time.time()
        try:
            value = fn()
        except BaseException as exc:  # noqa: BLE001 - a stage must never abort the report
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            lines = traceback.format_exc().strip().splitlines()[-5:]
            message = "{}: {}".format(type(exc).__name__, exc)
            self.stages.append(
                {
                    "stage": name,
                    "ok": False,
                    "seconds": round(time.time() - started, 4),
                    "error": message,
                    "traceback": lines,
                    "note": note or message,
                }
            )
            self.errors.append("{}: {}".format(name, message))
            return False, None
        record: Dict[str, Any] = {
            "stage": name,
            "ok": True,
            "seconds": round(time.time() - started, 4),
        }
        if note:
            record["note"] = note
        self.stages.append(record)
        return True, value

    def note_stage(self, name: str, note: str, *, ok: bool = True) -> None:
        """Record a stage that was deliberately not executed."""
        self.stages.append({"stage": name, "ok": ok, "seconds": 0.0, "skipped": True, "note": note})

    def last_stage_note(self, note: str) -> None:
        if self.stages:
            existing = self.stages[-1].get("note")
            self.stages[-1]["note"] = "{}; {}".format(existing, note) if existing else note

    def stage_ok(self, name: str) -> bool:
        for record in self.stages:
            if record.get("stage") == name:
                return bool(record.get("ok"))
        return False


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _normalise_formats(formats: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for value in formats or ():
        text = str(value).strip().lower().lstrip(".")
        if text and text not in out:
            out.append(text)
    return out


def _design_formats(design: Dict[str, Any]) -> List[str]:
    outputs = design.get("outputs") or {}
    return _normalise_formats(outputs.get("formats")) or list(_DEFAULT_FORMATS)


def _components_by_ref(design: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for component in design.get("components") or ():
        if isinstance(component, dict) and component.get("ref"):
            result[str(component["ref"])] = component
    return result


def _component_rows(run: _Run) -> List[Dict[str, Any]]:
    """The component table the report renders: intent (name/source) plus derived placement."""
    rows: List[Dict[str, Any]] = []
    placed_by_ref: Dict[str, Any] = {}
    if run.placement is not None:
        placed_by_ref = dict(getattr(run.placement, "by_ref", {}) or {})
    for ref, envelope in run.envelopes.items():
        placed = placed_by_ref.get(ref)
        rows.append(
            {
                "ref": ref,
                "name": getattr(envelope, "name", "") or "",
                "source": getattr(envelope, "source", "") or "",
                "size": list(getattr(envelope, "size", ()) or ()),
                "position": list(getattr(placed, "position", ()) or ()) if placed else None,
                "rotation": list(getattr(placed, "rotation", ()) or ()) if placed else None,
            }
        )
    rows.sort(key=lambda row: str(row["ref"]))
    return rows


def _walk(node: Any) -> List[Any]:
    from phimech.csg.ast import walk

    return list(walk(node))


def _has_step_import(node: Any) -> bool:
    """A STEP import can only be evaluated by FreeCAD; OpenSCAD cannot read the format."""
    for sub in _walk(node):
        if getattr(sub, "kind", "") == "import":
            if Path(str(getattr(sub, "path", ""))).suffix.lower() in _STEP_SUFFIXES:
                return True
    return False


def _freecad_unsupported_kinds(node: Any) -> List[str]:
    """Reasons this part cannot go to the FreeCAD backend, as short kind-ish tokens.

    Two sources, because the static kind set is not sufficient: `UNSUPPORTED` covers whole node
    kinds (minkowski), while `can_render()` also catches node-SPECIFIC limits such as a twisted
    linear_extrude or a planar hull. Consulting only the first let a planar hull reach a FreeCAD
    subprocess, which died inside qhull and failed the STEP export of two shipped examples.
    """
    try:
        from phimech.backends import base

        backend = base.get("freecad")
    except Exception:  # backend absent: nothing to route around
        return []
    reasons = sorted(
        {sub.kind for sub in _walk(node) if getattr(sub, "kind", "") in
         (set(getattr(backend, "UNSUPPORTED", set()) or set()))}
    )
    can_render = getattr(backend, "can_render", None)
    if callable(can_render):
        try:
            why = can_render(node)
        except Exception:  # a probe must never break the build
            why = None
        if why and not reasons:
            reasons = [str(why)]
    return reasons


def _prefer_order(backend: str, fmt: str, step_import: bool) -> Tuple[str, ...]:
    """Backend preference for one format, honouring an explicit request where it can work."""
    if step_import:
        return ("freecad", "openscad")
    if fmt in _FREECAD_ONLY:
        return ("freecad", "openscad")
    if fmt in _VIEW_ONLY_FORMATS:
        return ("openscad", "freecad")
    if backend in ("openscad", "freecad"):
        other = "freecad" if backend == "openscad" else "openscad"
        return (backend, other)
    return ("openscad", "freecad")


def _measurement_with_stl(measurement: Any, stl_path: Optional[Path]) -> Any:
    """Attach the STL path so the manifold gate can mesh-check the very file we shipped."""
    if measurement is None or stl_path is None:
        return measurement
    try:
        extra = dict(getattr(measurement, "extra", {}) or {})
        extra["stl"] = str(stl_path)
        return dataclasses.replace(measurement, extra=extra)
    except Exception:
        return measurement


def _phimech_version() -> str:
    try:
        from phimech.version import __version__

        return str(__version__)
    except Exception:
        return "0.1.0"


def _command_line(run: _Run, formats: Sequence[str], backend: str, gates: bool) -> str:
    if run.out_dir is None:
        return "bin/phimech synth check {}".format(run.ir_path)
    parts = ["bin/phimech synth build", run.ir_path, "-o", str(run.out_dir)]
    if formats:
        parts.extend(["--formats", ",".join(formats)])
    if backend and backend != "auto":
        parts.extend(["--backend", backend])
    if not gates:
        parts.append("--no-gates")
    return " ".join(parts)


# --------------------------------------------------------------------------------------
# stage implementations
# --------------------------------------------------------------------------------------


def _load_and_validate(run: _Run) -> bool:
    """Run `load` then `validate`, recording `validate` as skipped when the load failed, so the
    stage list is always exactly `STAGES` no matter where the run stopped."""
    if not _stage_load(run):
        run.note_stage("validate", "skipped: the design could not be loaded", ok=False)
        return False
    return _stage_validate(run)


def _stage_load(run: _Run) -> bool:
    def work() -> Dict[str, Any]:
        from phimech.ir import ir

        return ir.load(run.ir_path)

    ok, design = run.stage("load", work)
    if ok and isinstance(design, dict):
        run.design = design
    else:
        run.stop = True
    return ok


def _stage_validate(run: _Run) -> bool:
    def work() -> List[str]:
        from phimech.ir import ir

        return list(ir.validate(run.design))

    ok, errors = run.stage("validate", work)
    if not ok:
        run.stop = True
        return False
    errors = errors or []
    if errors:
        run.stages[-1]["ok"] = False
        run.stages[-1]["note"] = "{} validation error(s)".format(len(errors))
        run.stages[-1]["errors"] = errors
        run.errors.extend("validate: {}".format(message) for message in errors)
        run.stop = True
        return False
    return True


def _stage_envelopes(run: _Run) -> bool:
    def work() -> Dict[str, Any]:
        from phimech.components import providers

        try:
            return providers.envelopes_for(run.design)
        except Exception:
            # Retry per component so the report names the ref that failed, not just the batch.
            resolved: Dict[str, Any] = {}
            failures: List[str] = []
            for component in run.design.get("components") or ():
                ref = str((component or {}).get("ref", "?"))
                try:
                    resolved[ref] = providers.envelope_for(component)
                except Exception as exc:
                    failures.append("{}: {}: {}".format(ref, type(exc).__name__, exc))
            if failures:
                raise RuntimeError("envelope failed for " + "; ".join(failures))
            return resolved

    ok, envelopes = run.stage("envelopes", work)
    if ok and isinstance(envelopes, dict):
        run.envelopes = envelopes
        run.last_stage_note("{} envelope(s)".format(len(envelopes)))
    return ok


def _stage_place(run: _Run) -> bool:
    if not run.envelopes:
        run.note_stage("place", "skipped: no envelopes", ok=False)
        return False

    auto: Dict[str, bool] = {"arranged": False}

    def work() -> Any:
        from phimech.synth import place as place_module

        constraints = run.design.get("placement") or []
        if not constraints and len(run.envelopes) > 1:
            run.design["placement"] = place_module.auto_arrange(run.envelopes)
            auto["arranged"] = True
        return place_module.place(run.design, run.envelopes)

    ok, placement = run.stage("place", work)
    if not ok:
        return False
    run.placement = placement
    notes = []
    if auto["arranged"]:
        notes.append("auto_arranged")
        run.notes.append("placement was auto-arranged on a +X shelf (no constraints in the IR)")
    errors = list(getattr(placement, "errors", []) or [])
    if errors:
        notes.append("{} placement error(s)".format(len(errors)))
        run.errors.extend("place: {}".format(message) for message in errors)
        run.stages[-1]["ok"] = False
        run.stages[-1]["errors"] = errors
    if notes:
        run.last_stage_note(", ".join(notes))
    if auto["arranged"]:
        run.stages[-1]["auto_arranged"] = True
    return bool(run.stages[-1].get("ok"))


def _stage_enclosure(run: _Run) -> bool:
    if run.placement is None:
        run.note_stage("enclosure", "skipped: no placement", ok=False)
        return False
    if not run.design.get("enclosure"):
        run.note_stage("enclosure", "skipped: design has no enclosure (bare assembly)")
        return True

    def work() -> Any:
        from phimech.synth import enclosure as enclosure_module

        return enclosure_module.synthesize(
            run.placement,
            run.design.get("enclosure"),
            run.design.get("manufacturing") or {},
            components_by_ref=_components_by_ref(run.design),
            title=str(run.design.get("title") or run.design.get("project") or ""),
        )

    ok, parts = run.stage("enclosure", work)
    if not ok:
        return False
    run.enclosure_parts = parts
    names = list(getattr(parts, "parts", {}) or {})
    run.last_stage_note("parts: {}".format(", ".join(names) or "none"))
    for note in list(getattr(parts, "notes", []) or []):
        run.notes.append("enclosure: {}".format(note))
    return True


#: Gates that can only mean something once an enclosure exists. A bare assembly has no walls,
#: floor, standoffs, screws or cutouts, so `run_gates` legitimately crashes on them; the
#: pipeline turns those crashes into skips rather than reporting a design as failing a gate
#: that does not apply to it.
_ENCLOSURE_GATES = (
    "clearance_positive",
    "wall_thickness",
    "floor_thickness",
    "min_feature",
    "screw_count_sane",
    "cutout_within_face",
    "cutout_through",
    "text_legible",
    # The interference gates read the cavity and the synthesised bosses/posts. A bare assembly
    # has none of them, so they crash for want of an enclosure rather than finding a defect.
    "standoff_within_cavity",
    "fastener_clears_components",
    "standoff_lands_on_hole",
)


def _skip_inapplicable_enclosure_gates(report: Any) -> Any:
    """Rewrite gates that cannot mean anything without an enclosure as `not_applicable`.

    Two cases: an enclosure gate that crashed for want of walls/standoffs/cutouts, and
    `manifold` — a bare assembly is a multi-body *view* of separated components, so its solid
    count is 3 by design and a manifold failure would be pure noise. Neither is a pass and
    neither is a skip in the "not asked yet" sense: the gate simply has no subject here.
    """
    from phimech.check.gates import STATUS_NOT_APPLICABLE, GateReport, GateResult

    def not_applicable(result: Any, reason: str) -> Any:
        return GateResult(
            result.name, None, None, result.threshold, reason, result.severity,
            status=STATUS_NOT_APPLICABLE,
        )

    results = []
    for result in list(getattr(report, "results", []) or []):
        crashed = result.passed is False and str(result.message).startswith("gate crashed")
        if crashed and result.name in _ENCLOSURE_GATES:
            results.append(not_applicable(result, "not applicable: design has no enclosure"))
        elif result.name == "manifold" and result.passed is not None:
            results.append(
                not_applicable(
                    result,
                    "not applicable: bare assembly is a multi-body view, not a printed part",
                )
            )
        else:
            results.append(result)
    return GateReport(results)


def _restate_undetermined(run: _Run, report: Any, parts: Dict[str, Any]) -> Any:
    """After a render was *attempted*, a measurement gate with nothing to measure is
    `undetermined`, not `skipped`.

    `run_gates` cannot tell "no render yet" (`synth check`) from "the backend died"; it says
    `skipped: no render` in both cases, and a skip does not block. Here the pipeline knows a
    render was asked for, so any printable part without a measurement means the render or
    measure layer failed -- a timeout, a crash -- and every gate that reads measurements is
    re-stated as undetermined for that layer. That includes gates that happened to pass on
    the parts that *were* measured: a manifold pass on the base says nothing about a lid
    whose render never finished.
    """
    from phimech.check.gates import MEASUREMENT_GATES, GateReport, as_undetermined

    printable = _printable_parts(run, parts) if parts else []
    missing = [part for part in printable if part not in run.measurements]
    # A lost preview PNG is not a lost part; a lost STL/STEP is.
    lost = [
        part for part in printable
        if any(fmt not in _VIEW_ONLY_FORMATS for fmt in run.render_failures.get(part, []))
    ]
    affected = sorted(set(missing) | set(lost))
    if parts and not affected:
        return report
    render_failed = bool(lost) or not run.stage_ok("render")
    layer = "render" if (render_failed or not parts) else "measure"
    culprit = next(
        (error for error in run.errors if error.startswith(layer + ":")),
        "no measurement was produced",
    )
    subject = ", ".join(affected) if affected else "every part"
    reason = "undetermined: {} layer failed for {}: {}".format(layer, subject, culprit[:200])
    results = []
    for result in list(getattr(report, "results", []) or []):
        message = str(result.message)
        reads_measurements = result.name in MEASUREMENT_GATES or message.startswith(
            ("skipped: no render", "skipped: no stl")
        )
        if reads_measurements and result.effective_status != "not_applicable":
            results.append(as_undetermined(result, reason, layer))
        else:
            results.append(result)
    return GateReport(results)


def _machine_intent(design: Dict[str, Any]) -> Tuple[bool, str]:
    """`(gate against a real printer, base URL)` from `manufacturing.machine`.

    Opt-in, and for a stronger reason than the local geometry gate set: the machine gates make a
    network call. Running them implicitly would put a Moonraker query in the millisecond
    `synth check` loop of every design ever checked, and would make a build's result depend on
    whether a printer happened to be switched on. A design asks for machine gating by declaring
    `manufacturing.machine`, and once it has asked, an unreachable printer is a FAILURE rather
    than a skip — see `phimech/check/machine.py` on failing closed.
    """
    machine = _mapping_of(design.get("manufacturing")).get("machine")
    if not isinstance(machine, Mapping):
        return False, ""
    if machine.get("check") is False:
        return False, ""
    url = machine.get("url")
    return True, str(url) if isinstance(url, str) else ""


def _mapping_of(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stage_preview(run: _Run) -> Optional[Dict[str, Any]]:
    """Write ``preview.png`` (the 7-view sheet of the main printable part) beside the build.

    A picture is how an agent notices a *wrong shape* that every numeric gate accepts (the
    preview lane's own benchmark: 0/8 wrong parts caught by gates, 7/8 by silhouette IoU). It is
    evidence for a human or a vision model, not a gate, so a failure here is a note in the
    report - never an error that turns a good build red.
    """
    started = time.time()
    try:
        from phimech.synth import preview

        sheet = preview.for_build(run.out_dir)
    except Exception as exc:  # noqa: BLE001 - optional evidence, see docstring
        run.stages.append({
            "stage": "preview", "ok": True, "skipped": True,
            "seconds": round(time.time() - started, 4),
            "note": "unavailable: {}: {}".format(type(exc).__name__, exc),
        })
        run.notes.append("preview: not rendered ({})".format(exc))
        return None
    record = sheet.to_dict() if hasattr(sheet, "to_dict") else dict(vars(sheet))
    run.stages.append({
        "stage": "preview", "ok": True,
        "seconds": round(time.time() - started, 4),
        "note": "{} ({} views)".format(Path(record.get("path", "preview.png")).name,
                                        len(record.get("views") or {})),
    })
    run.outputs.append({
        "part": record.get("part") or "preview",
        "fmt": "png",
        "path": record.get("path"),
        "bytes": int(record.get("bytes") or 0),
        "backend": record.get("backend", "openscad"),
        "seconds": round(float(record.get("render_s") or 0.0), 4),
    })
    return record


def _stage_drawing(run: _Run) -> Optional[Dict[str, Any]]:
    """Write ``drawing.svg`` (+ ``.pdf``/``.png`` when a converter exists) of the main printable
    part beside the build, then run the deterministic sheet checks over it.

    The design that produced the build is handed to the sheet as its manifest, so revision,
    material, finish and any declared tolerance are stated rather than UNSPECIFIED. The result
    carries a ``checks`` block from ``phimech.check.drawing_checks`` (views readable, text not
    clipped, every feature located, every printed number traceable to a measurement, ...) with a
    ``summary.status`` of pass / fail / needs_review. Like the preview this is evidence, never a
    gate: a failed check or a missing OpenSCAD is a note in the report, not a red build.
    """
    started = time.time()
    try:
        from phimech.check import drawing_checks
        from phimech.synth import drawing, preview

        manifest = dict(run.design) if run.design else None
        sheet = drawing.for_build(run.out_dir, manifest=manifest)
        stl_path = next((c[1] for c in preview._build_stl_candidates(run.out_dir)
                         if c[0] == sheet.part), None)
        checks = drawing_checks.run_checks(sheet, svg_path=sheet.path, source_manifest=manifest,
                                           stl_path=stl_path)
    except Exception as exc:  # noqa: BLE001 - optional evidence, see docstring
        run.stages.append({
            "stage": "drawing", "ok": True, "skipped": True,
            "seconds": round(time.time() - started, 4),
            "note": "unavailable: {}: {}".format(type(exc).__name__, exc),
        })
        run.notes.append("drawing: not rendered ({})".format(exc))
        return None
    record = sheet.to_dict()
    record["checks"] = checks
    summary = checks.get("summary") or {}
    status = summary.get("status", "unknown")
    files = [("svg", record.get("path")), ("pdf", record.get("pdf")), ("png", record.get("png")),
             ("dxf", record.get("dxf"))]
    written = [(fmt, path) for fmt, path in files if path]
    run.stages.append({
        "stage": "drawing", "ok": True,
        "seconds": round(time.time() - started, 4),
        "note": "{} ({} feature groups; checks {}{})".format(
            " + ".join(Path(p).name for _, p in written), len(record.get("groups") or []),
            status, ", first failure {}".format(summary["first_failure"])
            if summary.get("first_failure") else ""),
    })
    if status != "pass":
        run.notes.append("drawing: sheet checks {}{} (advisory; see report['drawing']['checks'])".format(
            status, " - {}".format(summary["first_failure"]) if summary.get("first_failure") else ""))
    for fmt, path in written:
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        run.outputs.append({
            "part": record.get("part") or "drawing",
            "fmt": fmt,
            "path": path,
            "bytes": int(size),
            "backend": "openscad",
            "role": "drawing",
            "seconds": round(float(record.get("render_s") or 0.0), 4) if fmt == "svg" else 0.0,
        })
    return record


def _stage_gates(run: _Run, name: str, parts: Optional[Dict[str, Any]] = None) -> bool:
    """Run the gate sets. `parts` is passed for `gates_full` only: it tells the stage a render
    was attempted, so gates left without measurements are undetermined rather than skipped."""

    def work() -> Any:
        from phimech.check import gates

        report = gates.run_gates(
            run.design, run.placement, run.enclosure_parts, run.measurements
        )
        if not run.design.get("enclosure"):
            report = _skip_inapplicable_enclosure_gates(report)
        wants_machine, machine_url = _machine_intent(run.design)
        if wants_machine:
            # Appended for the same reason, and NOT wrapped in a "never fail a good build"
            # swallow like the two above: the machine gates exist precisely to fail when the
            # machine cannot be confirmed. An import error here becomes a failed gate, because
            # silently dropping them would turn "we could not check the printer" into "PASS".
            try:
                from phimech.check import machine as machine_check
                from phimech.check.gates import GateReport

                extra = machine_check.run_machine_gates(
                    run.design,
                    run.placement,
                    run.enclosure_parts,
                    run.measurements,
                    base_url=machine_url,
                )
                report = GateReport(list(report.results) + list(extra.results))
            except Exception as exc:
                from phimech.check.gates import GateReport, GateResult

                report = GateReport(
                    list(report.results)
                    + [
                        GateResult(
                            "machine_reachable",
                            False,
                            "unknown",
                            "queried",
                            "unknown: machine gates could not run: {}: {}".format(
                                type(exc).__name__, exc
                            ),
                            "error",
                        )
                    ]
                )
                run.errors.append("{}: machine gates unavailable: {}".format(name, exc))
        if parts is not None:
            # Last, so the optional machine gate results are covered as well.
            report = _restate_undetermined(run, report, parts)
        return report

    ok, report = run.stage(name, work)
    if ok and report is not None:
        run.gate_report = report
        try:
            run.last_stage_note(report.summary_line())
        except Exception:
            pass
    return ok


# --------------------------------------------------------------------------------------
# render planning
# --------------------------------------------------------------------------------------


def _bare_assembly_node(run: _Run) -> Any:
    """Union of the placed component bodies — the only geometry a design without an enclosure
    has. Built here (not in enclosure.py) because there is no enclosure to build it from."""
    from phimech.csg.ast import Cube, Group, Rotate, Translate, Union

    children: List[Any] = []
    for placed in list(getattr(run.placement, "items", []) or []):
        envelope = placed.envelope
        body = envelope.body
        if body is None:
            body = Cube(size=tuple(float(v) for v in envelope.size))
        rotation = tuple(float(v) for v in (placed.rotation or (0.0, 0.0, 0.0)))
        node: Any = body
        if any(abs(value) > 1e-9 for value in rotation):
            node = Rotate(rotation, [node])
        node = Translate(tuple(float(v) for v in placed.position), [node])
        children.append(Group([node], name="cmp_" + placed.ref))
    if not children:
        raise ValueError("nothing to render: placement produced no component bodies")
    return Group([Union(children)], name="assembly")


def _parts_to_emit(run: _Run) -> "Dict[str, Any]":
    """Ordered {part name: node}. Printable parts first, then the views."""
    parts: Dict[str, Any] = {}
    if run.enclosure_parts is not None:
        for name, node in (getattr(run.enclosure_parts, "parts", {}) or {}).items():
            parts[str(name)] = node
        parts["assembly"] = run.enclosure_parts.assembly()
        if (run.design.get("outputs") or {}).get("exploded_view", True):
            parts["exploded"] = run.enclosure_parts.exploded()
    else:
        parts["assembly"] = _bare_assembly_node(run)
    return parts


def _printable_parts(run: _Run, parts: Dict[str, Any]) -> List[str]:
    """The parts a printer would see. Views are excluded; a bare assembly has only itself."""
    if run.enclosure_parts is not None:
        return [name for name in parts if name not in _VIEW_PARTS]
    return [name for name in parts if name == "assembly"]


def _formats_for_part(run: _Run, part: str, formats: Sequence[str]) -> List[str]:
    """A view part gets only view formats; a printable part gets everything but the views.

    A design with no enclosure is the exception: its `assembly` *is* the deliverable, so it is
    written in every requested format including STL/STEP."""
    if run.enclosure_parts is None:
        return list(formats)
    if part in _VIEW_PARTS:
        return [fmt for fmt in formats if fmt in _VIEW_PART_FORMATS]
    return [fmt for fmt in formats if fmt not in _VIEW_ONLY_FORMATS]


def _plan_part(
    run: _Run, part: str, node: Any, formats: Sequence[str], backend: str
) -> Tuple[List[Tuple[str, str, Any]], List[str]]:
    """Decide (fmt, route, backend) for one part. Routes: 'scad', 'freecad', 'direct',
    'step_via_stl'. Unroutable formats come back as notes rather than errors."""
    from phimech.backends import base

    step_import = _has_step_import(node)
    unsupported = _freecad_unsupported_kinds(node) if not step_import else []
    plan: List[Tuple[str, str, Any]] = []
    notes: List[str] = []

    for fmt in _formats_for_part(run, part, formats):
        if fmt == "scad":
            if step_import:
                notes.append("{}: skipped .scad (OpenSCAD cannot read a STEP import)".format(part))
                continue
            try:
                plan.append((fmt, "scad", base.get("openscad")))
            except Exception:
                plan.append((fmt, "scad", None))  # emit through scad_emit; needs no binary
            continue
        if fmt in _FREECAD_ONLY and unsupported:
            if fmt == "step":
                plan.append((fmt, "step_via_stl", None))
                notes.append(
                    "{}: STEP via OpenSCAD-STL-FreeCAD import ({} unsupported)".format(
                        part, ", ".join(unsupported)
                    )
                )
            else:
                notes.append(
                    "{}: skipped .{} ({} unsupported in FreeCAD)".format(
                        part, fmt, ", ".join(unsupported)
                    )
                )
            continue
        try:
            chosen = base.for_format(fmt, prefer=_prefer_order(backend, fmt, step_import))
        except Exception as exc:
            notes.append("{}: skipped .{} ({})".format(part, fmt, exc))
            continue
        if step_import and chosen.name != "freecad":
            notes.append(
                "{}: skipped .{} (a STEP import needs the freecad backend)".format(part, fmt)
            )
            continue
        plan.append((fmt, "freecad" if chosen.name == "freecad" else "direct", chosen))
    return plan, notes


def _part_table(run: _Run, parts: Dict[str, Any]) -> Dict[str, Any]:
    """{part: {path, size, formats, printable}} — one row per part, for the CLI's part table."""
    printable = set(_printable_parts(run, parts)) if parts else set()
    table: Dict[str, Any] = {}
    for part in parts:
        files = [output for output in run.outputs if output["part"] == part]
        primary = next(
            (output for output in files if output["fmt"] in ("stl", "step", "scad")),
            files[0] if files else None,
        )
        measurement = run.measurements.get(part)
        bbox = getattr(measurement, "bbox", None)
        table[part] = {
            "path": primary["path"] if primary else None,
            "size": [round(v, 3) for v in bbox.size] if bbox is not None else None,
            "formats": sorted({output["fmt"] for output in files}),
            "printable": part in printable,
        }
    return table


def _output_record(part: str, result: Any) -> Dict[str, Any]:
    return {
        "part": part,
        "fmt": result.fmt,
        "path": str(result.path),
        "bytes": int(result.bytes),
        "backend": result.backend,
        "seconds": round(float(result.seconds), 4),
    }


def _render_step_via_stl(node: Any, out_path: Path) -> Any:
    """OpenSCAD → STL → FreeCAD `Import` → STEP. The mesh loses exact surfaces, so this is a
    fallback for parts using AST kinds FreeCAD cannot evaluate (minkowski), not the main path."""
    from phimech.backends import base
    from phimech.csg.ast import Import

    openscad = base.get("openscad")
    freecad = base.get("freecad")
    with tempfile.TemporaryDirectory(prefix="phimech-step-fallback-") as temporary:
        stl = Path(temporary) / "part.stl"
        openscad.render(node, stl, "stl")
        return freecad.render(Import(str(stl)), out_path, "step")


def _stage_render(run: _Run, formats: Sequence[str], backend: str) -> Dict[str, Any]:
    """Write every planned file plus the `.ast.json` of every part. Returns {part: node}."""
    assert run.out_dir is not None
    project = str(run.design.get("project") or "design")
    parts: Dict[str, Any] = {}
    render_measurements: Dict[str, Any] = {}

    def work() -> None:
        parts.update(_parts_to_emit(run))
        for part, node in parts.items():
            ast_path = run.out_dir / "{}-{}.ast.json".format(project, part)
            ast_path.write_text(
                json.dumps(node.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            plan, notes = _plan_part(run, part, node, formats, backend)
            run.notes.extend(notes)
            freecad_batch: Dict[str, Path] = {}
            freecad_backend: Any = None
            for fmt, route, chosen in plan:
                out_path = run.out_dir / "{}-{}.{}".format(project, part, fmt)
                if route == "freecad":
                    freecad_batch[fmt] = out_path
                    freecad_backend = chosen
                    continue
                try:
                    if route == "scad":
                        if chosen is not None:
                            result = chosen.render(node, out_path, "scad")
                        else:
                            from phimech.backends import scad_emit

                            started = time.time()
                            scad_emit.emit_file(node, out_path)
                            result = _fake_result(
                                out_path, "scad", "scad_emit", time.time() - started
                            )
                    elif route == "step_via_stl":
                        result = _render_step_via_stl(node, out_path)
                    else:
                        result = chosen.render(node, out_path, fmt)
                except Exception as exc:
                    run.errors.append(
                        "render: {} .{}: {}: {}".format(part, fmt, type(exc).__name__, exc)
                    )
                    run.render_failures.setdefault(part, []).append(fmt)
                    continue
                run.outputs.append(_output_record(part, result))
                if getattr(result, "measurement", None) is not None:
                    render_measurements[part] = result.measurement
                    run.render_measurement_backends[part] = str(result.backend)
            if freecad_batch and freecad_backend is not None:
                try:
                    if hasattr(freecad_backend, "render_many"):
                        # One freecadcmd launch for every FreeCAD format of this part: the
                        # process start-up (~1 s) dominates the actual export.
                        results = freecad_backend.render_many(node, freecad_batch, measure=True)
                    else:
                        results = {
                            fmt: freecad_backend.render(node, path, fmt)
                            for fmt, path in freecad_batch.items()
                        }
                except Exception as exc:
                    run.errors.append(
                        "render: {} [{}]: {}: {}".format(
                            part, ", ".join(sorted(freecad_batch)), type(exc).__name__, exc
                        )
                    )
                    run.render_failures.setdefault(part, []).extend(sorted(freecad_batch))
                else:
                    for fmt in sorted(results):
                        result = results[fmt]
                        run.outputs.append(_output_record(part, result))
                        if getattr(result, "measurement", None) is not None:
                            render_measurements[part] = result.measurement
                            run.render_measurement_backends[part] = str(result.backend)

    ok, _ = run.stage("render", work)
    if ok:
        run.last_stage_note(
            "{} file(s) for {} part(s)".format(len(run.outputs), len(parts))
        )
        if not run.outputs:
            run.stages[-1]["ok"] = False
            run.last_stage_note("no files were produced")
    run.render_measurements = render_measurements
    return parts


def _fake_result(path: Path, fmt: str, backend: str, seconds: float) -> Any:
    from phimech.backends.base import RenderResult

    return RenderResult(
        path=path,
        fmt=fmt,
        backend=backend,
        seconds=seconds,
        bytes=path.stat().st_size if path.exists() else 0,
    )


def _artifacts_by_part(run: _Run, fmt: str) -> Dict[str, Path]:
    """{part: path} of the files of one format the render stage actually wrote."""
    found: Dict[str, Path] = {}
    for output in run.outputs:
        if output.get("fmt") == fmt:
            found[str(output["part"])] = Path(str(output["path"]))
    return found


def _bbox_delta(a: Any, b: Any) -> float:
    """Largest absolute difference between two bboxes' corner coordinates, in mm."""
    return max(
        abs(float(x) - float(y))
        for x, y in list(zip(a.min, b.min)) + list(zip(a.max, b.max))
    )


def _reload_stl(run: _Run, part: str, tree: Any, stl_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Measure the exported STL *itself* and return `(measurement to gate on, diagnostic)`.

    Why reload: the measurement a backend hands back describes its in-memory model -- FreeCAD's
    exact B-rep, or for OpenSCAD a *separate* temporary render -- not the bytes that were
    written to disk. The printer gets the file, so the geometry gates (manifold, volume, bbox,
    bed fit) are run on the file re-read from disk, and the difference between the two is
    reported as `export_roundtrip` (a diagnostic, never a gate: a tessellated cylinder is
    legitimately smaller than its B-rep).

    `solids` is kept from the tree measurement on purpose: a hollow sealed shell is one solid
    with two mesh shells, so counting shells in the file would fail `manifold` on a correct
    part. An unreadable or empty file keeps the tree measurement for the gates and records an
    error, so the build cannot go green on a file nobody can open.
    """
    from phimech.backends.base import BBox, Measurement
    from phimech.check import mesh  # module attribute lookups keep this patchable in tests

    tree_extra = dict(getattr(tree, "extra", {}) or {})
    diagnostic: Dict[str, Any] = {
        "artifact": str(stl_path),
        "format": "stl",
        "volume_before_mm3": float(tree.volume),
        # The tree's face count is only known when the backend recorded one (FreeCAD does
        # not, OpenSCAD's measurement is itself a mesh it did not count): null is honest.
        "faces_before": tree_extra.get("faces"),
    }
    try:
        triangles = mesh.read_stl(str(stl_path))
        lo, hi = mesh.bbox(triangles)
        file_volume = float(mesh.volume(triangles))
        file_area = float(mesh.area(triangles))
    except Exception as exc:
        diagnostic["error"] = "{}: {}".format(type(exc).__name__, exc)
        run.errors.append(
            "measure: {}: exported STL could not be reloaded ({})".format(part, diagnostic["error"])
        )
        return tree, diagnostic
    bbox = BBox(min=tuple(float(v) for v in lo), max=tuple(float(v) for v in hi))
    diagnostic.update(
        {
            "volume_after_mm3": file_volume,
            "volume_delta_mm3": file_volume - float(tree.volume),
            "bbox_delta_mm": _bbox_delta(tree.bbox, bbox),
            "size_before_mm": [round(float(v), 4) for v in tree.bbox.size],
            "size_after_mm": [round(float(v), 4) for v in bbox.size],
            "faces_after": len(triangles),
        }
    )
    extra = dict(tree_extra)
    extra["tree"] = {
        "volume_mm3": float(tree.volume),
        "area_mm2": float(tree.area),
        "solids": int(tree.solids),
        "bbox": tree.bbox.to_dict(),
    }
    extra["source"] = "reloaded {}".format(stl_path.name)
    measurement = Measurement(
        bbox=bbox, volume=file_volume, area=file_area, solids=int(tree.solids), extra=extra
    )
    return measurement, diagnostic


def _reload_step(run: _Run, part: str, tree: Any, step_path: Path) -> Optional[Dict[str, Any]]:
    """Re-import the exported STEP through FreeCAD and compare it with the tree measurement.

    Diagnostic only, and only when FreeCAD is installed: it is what tells a SolidWorks user
    whether the STEP carries the solids the report claims. Costs one `freecadcmd` launch.
    """
    from phimech.backends import base
    from phimech.csg.ast import Import

    try:
        freecad = base.get("freecad")
        if not freecad.available():
            return None
    except Exception:
        return None
    diagnostic: Dict[str, Any] = {
        "artifact": str(step_path),
        "format": "step",
        "volume_before_mm3": float(tree.volume),
        "solids_before": int(tree.solids),
    }
    started = time.time()
    try:
        reloaded = freecad.measure(Import(str(step_path)))
    except Exception as exc:
        diagnostic["error"] = "{}: {}".format(type(exc).__name__, exc)
        run.errors.append(
            "measure: {}: exported STEP could not be reloaded ({})".format(
                part, diagnostic["error"]
            )
        )
        return diagnostic
    diagnostic.update(
        {
            "volume_after_mm3": float(reloaded.volume),
            "volume_delta_mm3": float(reloaded.volume) - float(tree.volume),
            "bbox_delta_mm": _bbox_delta(tree.bbox, reloaded.bbox),
            "solids_after": int(reloaded.solids),
            "valid": (getattr(reloaded, "extra", {}) or {}).get("valid"),
            "seconds": round(time.time() - started, 3),
        }
    )
    return diagnostic


_TRANSPARENT_KINDS = ("group", "translate", "rotate", "mirror", "color")
#: How many levels of a part's boolean spine are checked. Each level costs one backend
#: measurement; two reach the cut that matters most in an enclosure part, the cavity:
#: `union(body, bosses...)` at the top, `difference(outer, cavity)` inside it.
_MAX_BOOLEAN_DEPTH = 2


def _top_boolean(node: Any) -> Tuple[Optional[str], List[Any]]:
    """`(kind, children)` of a node's outermost boolean, looking through the wrappers that do
    not change volume. A multi-child group is an implicit union."""
    current = node
    while getattr(current, "kind", None) in _TRANSPARENT_KINDS:
        children = list(getattr(current, "children", []) or [])
        if len(children) != 1:
            return ("union", children) if len(children) > 1 else (None, [])
        current = children[0]
    kind = getattr(current, "kind", None)
    children = list(getattr(current, "children", []) or [])
    if kind in ("difference", "union") and children:
        return kind, children
    return None, []


def _boolean_checks(
    node: Any, result_volume: float, measurer: Any, label: str, depth: int
) -> List[Dict[str, Any]]:
    """Walk the boolean spine of `node`, measuring one reference per level.

    For a `difference` the reference is its base (a cut cannot grow it); for a `union` it is
    its first child (a union cannot be smaller than any child -- and `_Part.build()` puts the
    body first, so the first child is the one worth a render). The reference then becomes the
    next level's subject, so `union(difference(outer, cavity), bosses)` yields two checks:
    part >= body, body <= outer.
    """
    kind, children = _top_boolean(node)
    if kind is None or depth >= _MAX_BOOLEAN_DEPTH:
        return []
    reference = children[0]
    reference_volume = float(measurer.measure(reference).volume)
    check: Dict[str, Any] = {
        "kind": kind,
        "node": label,
        "result_volume_mm3": float(result_volume),
        "reference_volume_mm3": reference_volume,
        "children": len(children),
    }
    if kind == "union":
        check["children_measured"] = 1
    child_label = label + (".base" if kind == "difference" else ".children[0]")
    return [check] + _boolean_checks(reference, reference_volume, measurer, child_label, depth + 1)


def _boolean_reference(
    part: str, node: Any, result_volume: float, measurer: Any
) -> Optional[Dict[str, Any]]:
    """Measure the volumes the `boolean_volume_sane` gate compares a part against.

    Measured by `measurer` -- the same backend that measured the part -- because comparing an
    exact B-rep against a tessellation would report tessellation as a boolean defect. Returns
    None for a part with no top-level boolean; a backend failure is recorded as `error` so the
    gate can say "reference unmeasured" instead of passing by omission.
    """
    if _top_boolean(node)[0] is None:
        return None
    record: Dict[str, Any] = {"backend": str(getattr(measurer, "name", "?")), "checks": []}
    started = time.time()
    try:
        record["checks"] = _boolean_checks(node, result_volume, measurer, "part", 0)
    except Exception as exc:
        record["error"] = "{}: {}: {}".format(part, type(exc).__name__, exc)
    record["seconds"] = round(time.time() - started, 3)
    return record


def _with_extra(measurement: Any, **fields: Any) -> Any:
    """A copy of `measurement` with `fields` merged into its `extra` block."""
    try:
        extra = dict(getattr(measurement, "extra", {}) or {})
        extra.update(fields)
        return dataclasses.replace(measurement, extra=extra)
    except Exception:
        return measurement


def _stage_measure(
    run: _Run,
    parts: Dict[str, Any],
    backend: str,
    *,
    reload: bool = True,
    boolean_check: bool = True,
) -> bool:
    """Fill `run.measurements` for the printable parts.

    Three measurements per part, in this order: the *tree* (a render measurement when the
    backend gave one, else `backend.measure()`), the *reloaded* exported files (`reload`), and
    the *boolean reference* for `boolean_volume_sane` (`boolean_check`). The gates read the
    reloaded STL; the tree and the reference travel in `measurement.extra`.
    """
    if not parts:
        run.note_stage("measure", "skipped: nothing was rendered", ok=False)
        return False
    from_render: Dict[str, Any] = run.render_measurements
    stl_by_part = _artifacts_by_part(run, "stl")
    step_by_part = _artifacts_by_part(run, "step")

    def work() -> None:
        from phimech.backends import base

        for part in _printable_parts(run, parts):
            node = parts[part]
            tree = from_render.get(part)
            measurer: Any = None
            if tree is not None:
                try:
                    measurer = base.get(run.render_measurement_backends.get(part, ""))
                except Exception:
                    measurer = None
            else:
                try:
                    measurer = base.for_format("stl", prefer=_prefer_order(backend, "stl", False))
                    tree = measurer.measure(node)
                except Exception as exc:
                    run.errors.append(
                        "measure: {}: {}: {}".format(part, type(exc).__name__, exc)
                    )
                    continue
            measurement = tree
            stl_path = stl_by_part.get(part)
            if reload and stl_path is not None:
                measurement, diagnostic = _reload_stl(run, part, tree, stl_path)
                run.roundtrip[part] = diagnostic
            step_path = step_by_part.get(part)
            if reload and step_path is not None:
                step_diagnostic = _reload_step(run, part, tree, step_path)
                if step_diagnostic is not None:
                    run.roundtrip.setdefault(part, {})["step"] = step_diagnostic
            if boolean_check and measurer is not None:
                reference = _boolean_reference(part, node, float(tree.volume), measurer)
                if reference is not None:
                    measurement = _with_extra(measurement, boolean_reference=reference)
            run.measurements[part] = _measurement_with_stl(measurement, stl_path)

    ok, _ = run.stage("measure", work)
    if ok:
        run.last_stage_note("{} part(s) measured".format(len(run.measurements)))
        if not run.measurements:
            run.stages[-1]["ok"] = False
    return ok


# --------------------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------------------


def _base_report(run: _Run, formats: Sequence[str], backend: str, gates: bool) -> Dict[str, Any]:
    design = run.design
    report: Dict[str, Any] = {
        "project": design.get("project") or Path(run.ir_path).stem,
        "title": design.get("title") or "",
        "rev": design.get("rev") or "A",
        "company": design.get("company") or "",
        # Carried into the report so a viewer or UI can offer an editable control per parameter
        # without parsing the design IR itself. References are already substituted by this point,
        # so this block is the only remaining record of what was editable.
        "parameters": design.get("parameters") or {},
        "ir_path": run.ir_path,
        "out_dir": str(run.out_dir) if run.out_dir else None,
        "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "phimech_version": _phimech_version(),
        "backend": backend,
        "formats": list(formats),
        "gates_enabled": bool(gates),
        "stages": run.stages,
        "errors": run.errors,
        "notes": run.notes,
        "components": _component_rows(run),
        "command": _command_line(run, formats, backend, gates),
    }
    if run.placement is not None:
        try:
            report["placement"] = run.placement.to_dict()
        except Exception as exc:
            run.errors.append("report: placement.to_dict failed: {}".format(exc))
    if run.enclosure_parts is not None:
        try:
            report["enclosure"] = run.enclosure_parts.summary()
        except Exception as exc:
            run.errors.append("report: enclosure.summary failed: {}".format(exc))
    if run.gate_report is not None:
        try:
            report["gates"] = run.gate_report.to_dict()
        except Exception as exc:
            run.errors.append("report: gates.to_dict failed: {}".format(exc))
    report["gate_summary"] = _gate_summary(run, gates)
    return report


def _gate_summary(run: _Run, gates: bool) -> Dict[str, Any]:
    """The report-level verdict, present on *every* report so a reader never has to guess
    whether a missing `gates` block means "all fine" or "never ran".

    Mirrors `report["gates"]["summary"]` when the gates ran. When they did not (disabled, or
    the design never loaded) it says so in `note`, and `is_valid` is False: validity that was
    never established is not validity.
    """
    if run.gate_report is not None:
        try:
            summary = dict(run.gate_report.summary())
            summary["note"] = None
            return summary
        except Exception as exc:
            run.errors.append("report: gates.summary failed: {}".format(exc))
    from phimech.check.gates import STATUSES

    counts = {status: 0 for status in STATUSES}
    counts["total"] = 0
    return {
        "is_valid": False,
        "first_failure": None,
        "first_undetermined": None,
        "undetermined_layer": None,
        "counts": counts,
        "note": "gates disabled" if not gates else "gates did not run",
    }


def _finish(run: _Run, report: Dict[str, Any]) -> Dict[str, Any]:
    """Compute `ok` from stages, stage errors, and the gate verdict, and stamp the runtime."""
    stage_ok = all(record.get("ok") for record in run.stages)
    gates_ok = True
    gates = report.get("gates")
    if isinstance(gates, dict) and "passed" in gates:
        gates_ok = bool(gates["passed"])
    report["ok"] = bool(stage_ok and gates_ok and not run.errors)
    report["elapsed_s"] = round(time.time() - run.started, 4)
    return report


# --------------------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------------------


def check(ir_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate, resolve envelopes, place, synthesise the enclosure, run render-free gates.

    Writes nothing and calls no backend, so it runs in milliseconds — this is the loop an
    agent iterates on while shaping the IR.
    """
    run = _Run(ir_path, None)
    if _load_and_validate(run):
        if _stage_envelopes(run):
            _stage_place(run)
            _stage_enclosure(run)
        else:
            run.note_stage("place", "skipped: envelopes failed", ok=False)
            run.note_stage("enclosure", "skipped: envelopes failed", ok=False)
        _stage_gates(run, "gates_static")
    else:
        for name in ("envelopes", "place", "enclosure", "gates_static"):
            run.note_stage(name, "skipped: the design could not be loaded/validated", ok=False)
    for name in STAGES[len(_CHECK_STAGES):]:
        run.note_stage(name, "not run by `check`")
    return _finish(run, _base_report(run, _design_formats(run.design), "auto", True))


def build(
    ir_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    formats: Optional[Sequence[str]] = None,
    backend: str = "auto",
    gates: bool = True,
    no_gates: bool = False,
    reload: bool = True,
    boolean_check: bool = True,
    preview: bool = True,
    drawing: bool = False,
) -> Dict[str, Any]:
    """Everything `check()` does, then render, measure, gate on real numbers, and report.

    `drawing=True` additionally writes the dimensioned inspection sheet of the main printable
    part (`drawing.svg`, plus `.pdf`/`.png` when a converter exists) and records the sheet's
    deterministic checks under `report["drawing"]["checks"]`. Off by default: it costs an
    OpenSCAD run per z-level and the checks are evidence for a reviewer, not gates.

    `no_gates` is accepted as an alias so the CLI can pass its flag through verbatim.
    `reload` re-reads the exported STL/STEP so the geometry gates judge the shipped file and
    `report["export_roundtrip"]` can state the tree-vs-file delta; `boolean_check` measures
    each part's top-level boolean reference for the `boolean_volume_sane` gate. Both default
    on; both cost backend time (one extra measure per part), which is why they can be turned
    off for a throwaway preview. `preview` writes the 7-view `preview.png` sheet of the main
    printable part (~1 s, OpenSCAD only); it is evidence, not a gate, and never fails a build.
    """
    if no_gates:
        gates = False
    run = _Run(ir_path, out_dir)
    assert run.out_dir is not None
    run.out_dir.mkdir(parents=True, exist_ok=True)

    parts: Dict[str, Any] = {}
    loaded = _load_and_validate(run)
    if loaded:
        if _stage_envelopes(run):
            _stage_place(run)
            _stage_enclosure(run)
        else:
            run.note_stage("place", "skipped: envelopes failed", ok=False)
            run.note_stage("enclosure", "skipped: envelopes failed", ok=False)
    else:
        for name in ("envelopes", "place", "enclosure"):
            run.note_stage(name, "skipped: the design could not be loaded/validated", ok=False)

    resolved_formats = _normalise_formats(formats) or _design_formats(run.design)
    if gates:
        _stage_gates(run, "gates_static")
    else:
        run.note_stage("gates_static", "skipped: gates disabled")

    if run.placement is not None:
        parts = _stage_render(run, resolved_formats, backend)
        _stage_measure(run, parts, backend, reload=reload, boolean_check=boolean_check)
    else:
        run.note_stage("render", "skipped: no placement to render", ok=False)
        run.note_stage("measure", "skipped: nothing was rendered", ok=False)

    if gates and run.placement is not None:
        _stage_gates(run, "gates_full", parts=parts)
    else:
        reason = "skipped: gates disabled" if not gates else "skipped: nothing was rendered"
        run.note_stage("gates_full", reason)

    preview_record: Optional[Dict[str, Any]] = None
    if not preview:
        run.note_stage("preview", "skipped: preview disabled")
    elif run.placement is not None and parts:
        preview_record = _stage_preview(run)
    else:
        run.note_stage("preview", "skipped: nothing was rendered")

    drawing_record: Optional[Dict[str, Any]] = None
    if not drawing:
        run.note_stage("drawing", "skipped: not requested (pass --drawing / drawing=True)")
    elif run.placement is not None and parts:
        drawing_record = _stage_drawing(run)
    else:
        run.note_stage("drawing", "skipped: nothing was rendered")

    report = _base_report(run, resolved_formats, backend, gates)
    report["outputs"] = run.outputs
    report["measurements"] = {
        part: measurement.to_dict() if hasattr(measurement, "to_dict") else measurement
        for part, measurement in run.measurements.items()
    }
    report["parts"] = _part_table(run, parts)
    report["printable_parts"] = _printable_parts(run, parts) if parts else []
    report["export_roundtrip"] = run.roundtrip
    report["preview"] = preview_record
    report["drawing"] = drawing_record
    report["report_path"] = str(run.out_dir / "report.json")
    report["report_md"] = str(run.out_dir / "REPORT.md")

    # The `report` stage is recorded *before* the files are written so that report.json and
    # the returned dict describe the same stage list.
    started = time.time()
    run.stages.append({"stage": "report", "ok": True, "seconds": 0.0})
    stage_record = run.stages[-1]
    try:
        from phimech.ir import ir
        from phimech.synth import report as report_module

        try:
            (run.out_dir / "design.json").write_text(ir.dumps(run.design) + "\n", encoding="utf-8")
        except Exception as exc:  # a report is still worth writing without the IR copy
            run.notes.append("report: could not copy the design IR: {}".format(exc))
        _finish(run, report)
        stage_record["note"] = "report.json + REPORT.md"
        report_module.write_report(run.out_dir, report)
    except BaseException as exc:  # noqa: BLE001 - the caller still gets the dict
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        stage_record["ok"] = False
        stage_record["error"] = "{}: {}".format(type(exc).__name__, exc)
        stage_record["traceback"] = traceback.format_exc().strip().splitlines()[-5:]
        stage_record["note"] = stage_record["error"]
        run.errors.append("report: {}".format(stage_record["error"]))
    stage_record["seconds"] = round(time.time() - started, 4)
    return _finish(run, report)


# --------------------------------------------------------------------------------------
# module CLI (the real CLI in bin/phimech calls check()/build() directly)
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python3 -m phimech.synth.pipeline check|build ...` — mirrors the CLI flags."""
    parser = argparse.ArgumentParser(prog="phimech.synth.pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    check_parser = sub.add_parser("check", help="validate + place + gate, writing nothing")
    check_parser.add_argument("file")
    check_parser.add_argument("--json", action="store_true", help="print the whole report")

    build_parser = sub.add_parser("build", help="render every part and write the report")
    build_parser.add_argument("file")
    build_parser.add_argument("-o", "--output", required=True, help="output directory")
    build_parser.add_argument("--formats", default="", help="comma-separated override")
    build_parser.add_argument("--backend", default="auto", choices=("auto", "openscad", "freecad"))
    build_parser.add_argument("--no-gates", action="store_true")
    build_parser.add_argument("--json", action="store_true", help="print the whole report")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command not in ("check", "build"):
        parser.print_help(sys.stderr)
        return 2

    if args.command == "check":
        report = check(args.file)
    else:
        report = build(
            args.file,
            args.output,
            formats=_normalise_formats(args.formats.split(",")) or None,
            backend=args.backend,
            gates=not args.no_gates,
        )

    from phimech.synth.report import summary_line

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(summary_line(report))
        for stage in report.get("stages", []):
            if not stage.get("ok"):
                print("stage {}: {}".format(stage.get("stage"), stage.get("note", "failed")))
        for message in report.get("errors", [])[:20]:
            print("error: {}".format(message))
        if report.get("report_md"):
            print("report: {}".format(report["report_md"]))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["STAGES", "check", "build", "main"]
