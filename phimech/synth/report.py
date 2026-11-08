"""Turn a pipeline report dict into the two artefacts that travel with every build.

Why two files: `report.json` is what an agent (or CI) diffs and asserts on, `REPORT.md` is what
a human reads before sending a part to a printer. They are rendered from the *same* dict, so
they can never disagree — the markdown is a pure function of the JSON.

Why this module is defensive about missing keys: `pipeline.py` is fault-tolerant by design. A
build whose `place` stage crashed still produces a report, and that report must still render.
So every accessor here degrades to a dash rather than raising: a report that cannot be written
is strictly worse than an incomplete one.

stdlib only; imports under python 3.9.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

#: PLA at 100 % infill. Only used for the "est. material" line, never for a gate.
_PLA_DENSITY_G_CM3 = 1.24

_JSON_NAME = "report.json"
_MD_NAME = "REPORT.md"

_MARK_PASS = "\u2713"  # ✓
_MARK_FAIL = "\u2717"  # ✗
_MARK_SKIP = "\u2013"  # –


# --------------------------------------------------------------------------------------
# tiny, total accessors
# --------------------------------------------------------------------------------------


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict or an object, never raising — reports come from partial runs."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any, digits: int = 2) -> str:
    number = _num(value)
    if number is None:
        return "-"
    return "{:.{d}f}".format(number, d=digits).rstrip("0").rstrip(".") or "0"


def _fmt_vec(value: Any, digits: int = 2) -> str:
    if isinstance(value, dict):
        value = value.get("size", value.get("min"))
    if not isinstance(value, (list, tuple)):
        return "-"
    return " × ".join(_fmt_num(v, digits) for v in value)


def _fmt_seconds(value: Any) -> str:
    number = _num(value)
    return "-" if number is None else "{:.3f}".format(number)


def _cell(value: Any) -> str:
    """Make a value safe for one markdown table cell."""
    if value is None:
        return "-"
    text = str(value)
    text = text.replace("|", "\\|").replace("\n", " ").strip()
    return text or "-"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    """Render a markdown table, or a single 'none' line when there are no rows."""
    if not rows:
        return ["_none_", ""]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    lines.append("")
    return lines


def _relative(path: Any, out_dir: Optional[Path]) -> str:
    if path is None:
        return "-"
    candidate = Path(str(path))
    if out_dir is not None:
        try:
            return str(candidate.relative_to(out_dir))
        except ValueError:
            pass
    return candidate.name or str(candidate)


# --------------------------------------------------------------------------------------
# derived numbers
# --------------------------------------------------------------------------------------


def _gate_results(report: Any) -> List[Any]:
    gates = _get(report, "gates")
    if gates is None:
        return []
    results = _get(gates, "results")
    if isinstance(results, (list, tuple)):
        return list(results)
    if isinstance(gates, (list, tuple)):
        return list(gates)
    return []


def gate_counts(report: Any) -> Tuple[int, int, int]:
    """(passed, failed, skipped) over every gate result in the report.

    The legacy tri-state tally: `skipped` here is every gate with `passed: null`, which since
    the status vocabulary includes `not_applicable`; `failed` includes `undetermined`. Use
    `gate_status_counts` for the five-way split.
    """
    passed = failed = skipped = 0
    for result in _gate_results(report):
        state = _get(result, "passed", None)
        if state is True:
            passed += 1
        elif state is False:
            failed += 1
        else:
            skipped += 1
    return passed, failed, skipped


def gate_status_counts(report: Any) -> Dict[str, int]:
    """One count per status word (`pass`, `fail`, `skipped`, `not_applicable`, `undetermined`).

    Works on reports written before the `status` field existed: the status is then derived
    from `passed` and the message, exactly as `check/gates.py` does.
    """
    from phimech.check.gates import STATUSES, result_status

    counts = {status: 0 for status in STATUSES}
    for result in _gate_results(report):
        status = result_status(result)
        counts[status] = counts.get(status, 0) + 1
    return counts


def gate_summary(report: Any) -> Dict[str, Any]:
    """`{is_valid, first_failure, first_undetermined, undetermined_layer, counts}` for a
    report dict: the recorded block when the pipeline wrote one, else recomputed here."""
    recorded = _get(report, "gate_summary")
    if isinstance(recorded, dict) and "is_valid" in recorded:
        return recorded
    gates = _get(report, "gates")
    nested = _get(gates, "summary") if gates is not None else None
    if isinstance(nested, dict) and "is_valid" in nested:
        return nested
    from phimech.check.gates import summarize

    return summarize(_gate_results(report))


def verdict_line(report: Any) -> str:
    """One line an agent can grep: PASS/FAIL plus the gate arithmetic behind it."""
    counts = gate_status_counts(report)
    passed, failed, skipped = counts["pass"], counts["fail"], counts["skipped"]
    ok = _get(report, "ok", None)
    gates = _get(report, "gates")
    if ok is None:
        ok = (failed + counts["undetermined"]) == 0 if gates is not None else True
    stamp = "PASS" if ok else "FAIL"
    if gates is None and not _gate_results(report):
        parts = ["no gates were run"]
    else:
        parts = ["{} gate{} passed".format(passed, "" if passed == 1 else "s")]
        if failed:
            parts.append("{} failed".format(failed))
        if skipped:
            parts.append("{} skipped".format(skipped))
        if counts["not_applicable"]:
            parts.append("{} not applicable".format(counts["not_applicable"]))
        if counts["undetermined"]:
            parts.append("{} undetermined".format(counts["undetermined"]))
    errors = _as_list(_get(report, "errors"))
    if errors:
        parts.append("{} stage error{}".format(len(errors), "" if len(errors) == 1 else "s"))
    return "{} \u2014 {}".format(stamp, ", ".join(parts))


def summary_line(report: Any) -> str:
    """The single line the CLI and MCP print. Keep it under ~120 characters."""
    passed, failed, skipped = gate_counts(report)
    project = _get(report, "project") or "design"
    ok = _get(report, "ok", None)
    stamp = "PASS" if ok or ok is None else "FAIL"
    bits = [
        "{} {}".format(stamp, project),
        "{} components".format(len(_as_list(_get(report, "components")))),
    ]
    outputs = _as_list(_get(report, "outputs"))
    if outputs:
        bits.append("{} files".format(len(outputs)))
    if passed or failed or skipped:
        bits.append("gates {}/{} ok".format(passed, passed + failed + skipped))
    if failed:
        bits.append("{} failed".format(failed))
    errors = _as_list(_get(report, "errors"))
    if errors:
        bits.append("{} stage errors".format(len(errors)))
    elapsed = _num(_get(report, "elapsed_s"))
    if elapsed is not None:
        bits.append("{:.2f}s".format(elapsed))
    return " \u2014 ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0]


def _material_grams(report: Any) -> Optional[float]:
    """Prefer real measured volume; fall back to the enclosure's bbox estimate."""
    total = 0.0
    seen = False
    for measurement in (_get(report, "measurements") or {}).values():
        volume = _num(_get(measurement, "volume_mm3"))
        if volume is None:
            volume = _num(_get(measurement, "volume"))
        if volume is not None:
            total += volume
            seen = True
    if not seen:
        estimate = _num(_get(_get(report, "enclosure") or {}, "estimated_material_mm3"))
        if estimate is None:
            return None
        total = estimate
    return total / 1000.0 * _PLA_DENSITY_G_CM3


# --------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------


def _header_lines(report: Any) -> List[str]:
    project = _get(report, "project") or "design"
    title = _get(report, "title") or project
    lines = ["# PhiMech report \u2014 {}".format(_cell(title)), ""]
    lines.append("**{}**".format(verdict_line(report)))
    lines.append("")
    rows = [
        ("project", _get(report, "project")),
        ("title", _get(report, "title")),
        ("rev", _get(report, "rev")),
        ("company", _get(report, "company")),
        ("generated", _get(report, "timestamp")),
        ("phimech", _get(report, "phimech_version")),
        ("design IR", _get(report, "ir_path")),
        ("out dir", _get(report, "out_dir")),
        ("elapsed", "{} s".format(_fmt_seconds(_get(report, "elapsed_s")))),
    ]
    lines.extend(_table(("field", "value"), [(k, v) for k, v in rows if v not in (None, "")]))
    return lines


def _stage_lines(report: Any) -> List[str]:
    rows = []
    for stage in _as_list(_get(report, "stages")):
        ok = _get(stage, "ok", None)
        if _get(stage, "skipped"):
            mark = _MARK_SKIP
        else:
            mark = _MARK_PASS if ok else _MARK_FAIL
        note = _get(stage, "note") or _get(stage, "error") or ""
        rows.append((mark, _get(stage, "stage"), _fmt_seconds(_get(stage, "seconds")), note))
    return ["## Stages", ""] + _table(("", "stage", "seconds", "note"), rows)


def _component_lines(report: Any) -> List[str]:
    rows = []
    for component in _as_list(_get(report, "components")):
        rows.append(
            (
                _get(component, "ref"),
                _get(component, "name"),
                _get(component, "source"),
                _fmt_vec(_get(component, "size")),
                _fmt_vec(_get(component, "position")),
                _fmt_vec(_get(component, "rotation"), 0),
            )
        )
    header = ("ref", "name", "source", "size mm", "position mm", "rotation \u00b0")
    return ["## Components", ""] + _table(header, rows)


def _enclosure_lines(report: Any) -> List[str]:
    enclosure = _get(report, "enclosure")
    if not enclosure:
        return ["## Enclosure", "", "_no enclosure in this design_", ""]
    grams = _material_grams(report)
    rows = [
        ("style", _get(enclosure, "style")),
        ("parts", ", ".join(str(p) for p in _as_list(_get(enclosure, "parts")))),
        ("outer size mm", _fmt_vec(_get(enclosure, "outer_size_mm"))),
        ("cavity size mm", _fmt_vec(_get(enclosure, "cavity_size_mm"))),
        ("wall / floor mm", "{} / {}".format(_fmt_num(_get(enclosure, "wall_mm")),
                                             _fmt_num(_get(enclosure, "floor_mm")))),
        ("fastening", _get(enclosure, "fastening")),
        ("standoffs", _get(enclosure, "standoffs")),
        ("cutouts", _get(enclosure, "cutouts")),
        ("screws", _get(enclosure, "screws")),
        ("est. material g", _fmt_num(grams) if grams is not None else None),
    ]
    lines = ["## Enclosure", ""]
    lines.extend(_table(("field", "value"), [(k, v) for k, v in rows if v is not None]))
    notes = _as_list(_get(enclosure, "notes"))
    if notes:
        lines.extend(["Enclosure notes:", ""] + ["- {}".format(_cell(n)) for n in notes] + [""])
    return lines


def _output_lines(report: Any, out_dir: Optional[Path]) -> List[str]:
    rows = []
    for output in _as_list(_get(report, "outputs")):
        size = _num(_get(output, "bytes"))
        rows.append(
            (
                _get(output, "part"),
                _get(output, "fmt"),
                "-" if size is None else "{:.1f}".format(size / 1024.0),
                _get(output, "backend"),
                _fmt_seconds(_get(output, "seconds")),
                _relative(_get(output, "path"), out_dir),
            )
        )
    header = ("part", "fmt", "size KB", "backend", "seconds", "file")
    return ["## Outputs", ""] + _table(header, rows)


def _measurement_lines(report: Any) -> List[str]:
    rows = []
    for part, measurement in sorted((_get(report, "measurements") or {}).items()):
        bbox = _get(measurement, "bbox") or {}
        volume = _get(measurement, "volume_mm3", _get(measurement, "volume"))
        area = _get(measurement, "area_mm2", _get(measurement, "area"))
        rows.append(
            (
                part,
                _fmt_vec(_get(bbox, "size", bbox)),
                _fmt_num(volume, 1),
                _fmt_num(area, 1),
                _get(measurement, "solids"),
            )
        )
    header = ("part", "bbox mm", "volume mm\u00b3", "area mm\u00b2", "solids")
    return ["## Measurements", ""] + _table(header, rows)


def _gate_lines(report: Any) -> List[str]:
    from phimech.check.gates import result_status

    rows = []
    for result in _gate_results(report):
        state = _get(result, "passed", None)
        mark = _MARK_PASS if state is True else _MARK_FAIL if state is False else _MARK_SKIP
        rows.append(
            (
                mark,
                _get(result, "name"),
                result_status(result),
                _get(result, "value"),
                _get(result, "threshold"),
                _get(result, "severity"),
                _get(result, "message"),
            )
        )
    header = ("", "gate", "status", "value", "threshold", "severity", "message")
    lines = ["## Gates", ""]
    if rows:
        lines.extend(_gate_summary_lines(report))
    return lines + _table(header, rows)


def _gate_summary_lines(report: Any) -> List[str]:
    """The verdict block above the gate table: validity, what blocks, what never got judged.

    `is_valid` is spelled out beside `ok` because they answer different questions (gates vs
    build), and `undetermined_layer` is named because "1 undetermined" without the layer that
    died sends a reader to the wrong place.
    """
    summary = gate_summary(report)
    counts = summary.get("counts") or {}
    tally = ", ".join(
        "{} {}".format(counts.get(key, 0), key.replace("_", " "))
        for key in ("pass", "fail", "skipped", "not_applicable", "undetermined")
        if counts.get(key)
    )
    lines = [
        "- valid: **{}** ({})".format(
            "yes" if summary.get("is_valid") else "no", tally or "no gates"
        ),
        "- first failure: {}".format(_cell(summary.get("first_failure") or "none")),
    ]
    if summary.get("undetermined_layer") or summary.get("first_undetermined"):
        lines.append(
            "- undetermined: {} (layer: {})".format(
                _cell(summary.get("first_undetermined") or "?"),
                _cell(summary.get("undetermined_layer") or "?"),
            )
        )
    lines.append("")
    return lines


def _roundtrip_lines(report: Any) -> List[str]:
    """Tree measurement vs the reloaded exported file, per part. Diagnostic: the gates above
    were run on the reloaded file, and this table says how far it sits from the model."""
    roundtrip = _get(report, "export_roundtrip") or {}
    if not isinstance(roundtrip, dict) or not roundtrip:
        return []
    rows = []
    for part, entry in sorted(roundtrip.items()):
        records = [("stl", entry)]
        step = _get(entry, "step")
        if isinstance(step, dict):
            records.append(("step", step))
        for fmt, record in records:
            if not isinstance(record, dict):
                continue
            error = _get(record, "error")
            before = _get(record, "faces_before", _get(record, "solids_before"))
            after = _get(record, "faces_after", _get(record, "solids_after"))
            rows.append(
                (
                    part,
                    fmt,
                    _fmt_num(_get(record, "volume_before_mm3"), 1),
                    _fmt_num(_get(record, "volume_after_mm3"), 1),
                    _fmt_num(_get(record, "volume_delta_mm3"), 2),
                    _fmt_num(_get(record, "bbox_delta_mm"), 4),
                    "{} → {}".format(
                        "–" if before is None else before, "–" if after is None else after
                    ),
                    _cell(error) if error else "",
                )
            )
    header = (
        "part", "file", "tree mm³", "file mm³", "Δ volume mm³", "Δ bbox mm",
        "faces/solids", "error",
    )
    return [
        "## Export round-trip",
        "",
        "Gates were run on the reloaded file; this is how far it sits from the measured tree "
        "(diagnostic, not a gate).",
        "",
    ] + _table(header, rows)


def _message_lines(report: Any) -> List[str]:
    lines: List[str] = ["## Errors and notes", ""]
    errors = _as_list(_get(report, "errors"))
    notes = _as_list(_get(report, "notes"))
    if not errors and not notes:
        lines.extend(["_none_", ""])
        return lines
    if errors:
        lines.append("Errors:")
        lines.append("")
        lines.extend("- {}".format(_cell(e)) for e in errors)
        lines.append("")
    if notes:
        lines.append("Notes:")
        lines.append("")
        lines.extend("- {}".format(_cell(n)) for n in notes)
        lines.append("")
    return lines


def _reproduce_lines(report: Any) -> List[str]:
    command = _get(report, "command")
    if not command:
        ir_path = _get(report, "ir_path") or "design.json"
        out_dir = _get(report, "out_dir")
        command = (
            "bin/phimech synth build {} -o {}".format(ir_path, out_dir)
            if out_dir
            else "bin/phimech synth check {}".format(ir_path)
        )
    return ["## How to reproduce", "", "```bash", str(command), "```", ""]


def render_markdown(report: Any) -> str:
    """Render the human-readable half of the report. Pure function of `report`."""
    out_dir_value = _get(report, "out_dir")
    out_dir = Path(str(out_dir_value)) if out_dir_value else None
    lines: List[str] = []
    lines.extend(_header_lines(report))
    lines.extend(_stage_lines(report))
    lines.extend(_component_lines(report))
    lines.extend(_enclosure_lines(report))
    lines.extend(_output_lines(report, out_dir))
    lines.extend(_measurement_lines(report))
    lines.extend(_roundtrip_lines(report))
    lines.extend(_gate_lines(report))
    lines.extend(_message_lines(report))
    lines.extend(_reproduce_lines(report))
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def write_report(out_dir: Union[str, Path], report: Dict[str, Any]) -> Path:
    """Write `report.json` + `REPORT.md` into `out_dir` and return the JSON path.

    `default=str` on the JSON dump is deliberate: reports carry `Path` objects and the odd
    dataclass, and losing the whole report to a serialisation error would defeat its purpose.
    """
    directory = Path(str(out_dir))
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(report, dict) and not report.get("timestamp"):
        report["timestamp"] = datetime.datetime.now().replace(microsecond=0).isoformat()
    json_path = directory / _JSON_NAME
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (directory / _MD_NAME).write_text(render_markdown(report), encoding="utf-8")
    return json_path


__all__ = [
    "write_report",
    "render_markdown",
    "summary_line",
    "verdict_line",
    "gate_counts",
]
