"""Grade a generated design against a benchmark case, by machine, with no opinions.

Why this module exists
----------------------
"Good AI generation" is unfalsifiable until something scores it. PhiMech already owns the hard
part - numeric gates with named thresholds - so grading a generated design is mostly a matter of
asking the gates what happened and adding the few assertions a gate cannot know (did the model
declare bounded parameters? did it use the component library instead of guessing dimensions?).

Deliberately split from *generation*. PhiMech never calls a model and never holds a credential: any
harness produces candidate IRs, and this grades them. That keeps the durable half - the grader -
testable offline and in CI, and makes the score comparable across harnesses rather than tied to one.

Every check answers to a number or a named gate. There is no partial credit and no weighting: a
case reports the checks it passed out of the checks it was given, so a low score always points at a
specific failed check rather than a vibe.

stdlib only: it must import under the same pythons as the rest of the core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SUITE_PATH = Path(__file__).resolve().parent / "suite.json"


def load_suite(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load the benchmark suite, or a caller-supplied one with the same shape."""
    suite_path = Path(path) if path else SUITE_PATH
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("suite must be an object with a `cases` array: %s" % suite_path)
    return data


def case_by_id(suite: Mapping[str, Any], case_id: str) -> Optional[Dict[str, Any]]:
    for case in suite.get("cases", []):
        if isinstance(case, dict) and case.get("id") == case_id:
            return case
    return None


# -- individual checks ---------------------------------------------------------------------
#
# Each returns (name, passed, detail). `passed` is True/False only - a check that cannot be
# evaluated is a failure with a detail saying why, never a silent skip. A skipped check would let a
# malformed candidate score well by omission.


def _gate_index(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    gates = report.get("gates")
    results = gates.get("results") if isinstance(gates, dict) else None
    index: Dict[str, Dict[str, Any]] = {}
    if isinstance(results, list):
        for gate in results:
            if isinstance(gate, dict) and isinstance(gate.get("name"), str):
                index[gate["name"]] = gate
    return index


def _outer_size(report: Mapping[str, Any]) -> Optional[List[float]]:
    """The assembled outer envelope, from the largest printable part's bbox.

    Read from measurements rather than the design, because the point is to score what was *built*.
    """
    measurements = report.get("measurements")
    if not isinstance(measurements, dict):
        return None
    best: Optional[List[float]] = None
    best_volume = -1.0
    for name, entry in measurements.items():
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox")
        size = bbox.get("size") if isinstance(bbox, dict) else None
        if not isinstance(size, list) or len(size) != 3:
            continue
        try:
            volume = float(size[0]) * float(size[1]) * float(size[2])
        except (TypeError, ValueError):
            continue
        if volume > best_volume:
            best_volume, best = volume, [float(v) for v in size]
    return best


def _check_builds(report: Mapping[str, Any], expected: bool) -> Tuple[str, bool, str]:
    actual = bool(report.get("ok"))
    errors = report.get("errors") or []
    detail = "ok=%s" % actual
    if errors:
        detail += ", %d error(s): %s" % (len(errors), str(errors[0])[:80])
    return ("builds", actual == expected, detail)


def _check_gates_pass(
    report: Mapping[str, Any], names: Sequence[str]
) -> List[Tuple[str, bool, str]]:
    index = _gate_index(report)
    out = []
    for name in names:
        gate = index.get(name)
        if gate is None:
            out.append(("gate:%s" % name, False, "gate not present in report"))
            continue
        passed = gate.get("passed")
        if passed is True:
            out.append(("gate:%s" % name, True, "pass"))
        elif passed is None:
            # The distinction that matters: a skipped gate checked nothing.
            out.append(("gate:%s" % name, False, "SKIPPED, not passed: %s"
                        % str(gate.get("message") or "")[:60]))
        else:
            out.append(("gate:%s" % name, False, str(gate.get("message") or "failed")[:70]))
    return out


def _check_gates_pass_or_warn(
    report: Mapping[str, Any], names: Sequence[str]
) -> List[Tuple[str, bool, str]]:
    """Accept a pass, or a failure whose severity is only `warn`.

    Used where a brief is legitimately hard: a warn-severity finding is information, not a defect.
    """
    index = _gate_index(report)
    out = []
    for name in names:
        gate = index.get(name)
        if gate is None:
            out.append(("gate_or_warn:%s" % name, False, "gate not present"))
            continue
        passed, severity = gate.get("passed"), gate.get("severity")
        ok = passed is True or (passed is False and severity == "warn")
        out.append(("gate_or_warn:%s" % name, ok,
                    "passed=%s severity=%s" % (passed, severity)))
    return out


def _check_not_skipped(
    report: Mapping[str, Any], names: Sequence[str]
) -> List[Tuple[str, bool, str]]:
    index = _gate_index(report)
    out = []
    for name in names:
        gate = index.get(name)
        passed = gate.get("passed") if isinstance(gate, dict) else "absent"
        out.append(("not_skipped:%s" % name, gate is not None and passed is not None,
                    "passed=%s" % passed))
    return out


def _check_size(report: Mapping[str, Any], spec: Mapping[str, Any]) -> Tuple[str, bool, str]:
    target = spec.get("value")
    tolerance = float(spec.get("tolerance", 2))
    actual = _outer_size(report)
    if not isinstance(target, list) or len(target) != 3:
        return ("outer_size_mm", False, "case has a malformed target")
    if actual is None:
        return ("outer_size_mm", False, "no measured bbox in the report")
    # Sorted comparison: orientation is the model's choice, size is not. A 48x68 box laid out as
    # 68x48 is the same design, and penalising it would score style rather than correctness.
    got, want = sorted(actual), sorted(float(v) for v in target)
    worst = max(abs(a - b) for a, b in zip(got, want))
    return ("outer_size_mm", worst <= tolerance,
            "measured %s vs target %s, worst delta %.1f mm (tolerance %.1f)"
            % ([round(v, 1) for v in actual], target, worst, tolerance))


def _check_count(label: str, actual: Any, expected: Any) -> Tuple[str, bool, str]:
    try:
        ok = int(actual) == int(expected)
    except (TypeError, ValueError):
        return (label, False, "expected %s, got %r" % (expected, actual))
    return (label, ok, "expected %s, got %s" % (expected, actual))


def _design_of(report: Mapping[str, Any], design: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(design, dict) and design:
        return dict(design)
    embedded = report.get("design")
    return dict(embedded) if isinstance(embedded, dict) else {}


def _check_parameters(
    report: Mapping[str, Any], design: Mapping[str, Any], expected: Sequence[str], bounded: bool
) -> List[Tuple[str, bool, str]]:
    declared = report.get("parameters")
    if not isinstance(declared, dict) or not declared:
        candidate = design.get("parameters")
        declared = candidate if isinstance(candidate, dict) else {}
    out: List[Tuple[str, bool, str]] = []
    for name in expected:
        # Substring match: a model may reasonably call it `wall_thickness` rather than `wall`.
        hit = next((key for key in declared if name in str(key).lower()), None)
        out.append(("parameter:%s" % name, hit is not None,
                    "declared: %s" % (", ".join(sorted(declared)) or "none")))
    if bounded:
        unbounded = [
            key for key, spec in declared.items()
            if not (isinstance(spec, dict) and ("min" in spec or "max" in spec))
        ]
        out.append(("parameters_have_bounds", not unbounded and bool(declared),
                    "without bounds: %s" % (", ".join(sorted(unbounded)) or "none")))
    return out


def _check_library_source(design: Mapping[str, Any]) -> Tuple[str, bool, str]:
    """Did the model reach for the component library instead of guessing dimensions?"""
    kinds = []
    components = design.get("components")
    if isinstance(components, list):
        for component in components:
            source = component.get("source") if isinstance(component, dict) else None
            if isinstance(source, dict) and source.get("kind"):
                kinds.append(str(source["kind"]))
    return ("uses_library_source", "library" in kinds,
            "component source kinds: %s" % (", ".join(kinds) or "none"))


def _check_cutouts(report: Mapping[str, Any], minimum: int) -> Tuple[str, bool, str]:
    enclosure = report.get("enclosure")
    count = enclosure.get("cutouts") if isinstance(enclosure, dict) else None
    try:
        ok = int(count) >= int(minimum)
    except (TypeError, ValueError):
        return ("min_cutouts", False, "no cutout count in the report")
    return ("min_cutouts", ok, "%s cutout(s), need >= %s" % (count, minimum))


def _check_min_height(report: Mapping[str, Any], minimum: float) -> Tuple[str, bool, str]:
    size = _outer_size(report)
    if size is None:
        return ("min_outer_height_mm", False, "no measured bbox")
    return ("min_outer_height_mm", size[2] >= float(minimum),
            "height %.1f mm, need >= %s" % (size[2], minimum))


# -- case grading --------------------------------------------------------------------------


def _check_preview_iou(report: Mapping[str, Any], minimum: float) -> Tuple[str, bool, str]:
    """The silhouette-IoU proxy from ``phimech.synth.preview.compare_sheets``.

    The report carries the comparison under ``preview`` (the wrong-shape runner puts it there);
    a report without one fails the check explicitly, because "nobody looked" is not a pass.
    """
    preview = report.get("preview")
    value = preview.get("min_iou") if isinstance(preview, dict) else None
    try:
        actual = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "preview_min_iou", False, "no preview comparison in the report"
    worst = preview.get("worst_view") if isinstance(preview, dict) else None
    return (
        "preview_min_iou",
        actual >= float(minimum),
        "min silhouette IoU %.3f (worst view %s), need >= %.2f" % (actual, worst, float(minimum)),
    )


def grade_case(
    case: Mapping[str, Any],
    report: Optional[Mapping[str, Any]],
    design: Optional[Mapping[str, Any]] = None,
    validates: Optional[bool] = None,
) -> Dict[str, Any]:
    """Score one candidate. Returns the per-check detail, not just a number.

    ``report`` is a ``synth build`` report; ``design`` the authored IR; ``validates`` the outcome of
    ``ir_validate`` when the candidate never built. A case with an ``either`` block passes if any
    branch passes - that is how a brief which *should* be refused is scored.
    """
    expect = case.get("expect") or {}
    if isinstance(expect.get("either"), list):
        branches = []
        for index, branch in enumerate(expect["either"]):
            sub = dict(case)
            sub["expect"] = branch
            branches.append(grade_case(sub, report, design, validates))
        winner = next((b for b in branches if b["passed"]), None)
        chosen = winner or max(branches, key=lambda b: b["score"])
        return {
            "id": case.get("id"),
            "passed": bool(winner),
            "score": chosen["score"],
            "checks": chosen["checks"],
            "note": "either-branch %s"
                    % ("satisfied" if winner else "none satisfied; showing the closest"),
        }

    checks: List[Tuple[str, bool, str]] = []
    design_doc = _design_of(report or {}, design)

    if "validates" in expect:
        want = bool(expect["validates"])
        actual = validates
        if actual is None and report is not None:
            # A report that exists at all means validation passed.
            actual = True
        checks.append(("validates", actual == want, "validates=%s, expected %s" % (actual, want)))

    if report is None:
        # Nothing built. Every remaining expectation is unmet - report it as such rather than
        # scoring an absent candidate as neutral.
        for key in expect:
            if key != "validates":
                checks.append((str(key), False, "no build report"))
    else:
        if "builds" in expect:
            checks.append(_check_builds(report, bool(expect["builds"])))
        if expect.get("gates_pass"):
            checks.extend(_check_gates_pass(report, expect["gates_pass"]))
        if expect.get("gates_pass_or_warn"):
            checks.extend(_check_gates_pass_or_warn(report, expect["gates_pass_or_warn"]))
        if expect.get("gates_not_skipped"):
            checks.extend(_check_not_skipped(report, expect["gates_not_skipped"]))
        if isinstance(expect.get("outer_size_mm"), dict):
            checks.append(_check_size(report, expect["outer_size_mm"]))
        if "printable_parts" in expect:
            printable = report.get("printable_parts")
            actual = len(printable) if isinstance(printable, list) else printable
            checks.append(_check_count("printable_parts", actual, expect["printable_parts"]))
        if "component_count" in expect:
            components = design_doc.get("components")
            actual = len(components) if isinstance(components, list) else None
            if actual is None:
                placed = report.get("placement")
                items = placed.get("items") if isinstance(placed, dict) else None
                actual = len(items) if isinstance(items, list) else None
            checks.append(_check_count("component_count", actual, expect["component_count"]))
        if expect.get("parameters_declared") or expect.get("parameters_have_bounds"):
            checks.extend(_check_parameters(
                report, design_doc, expect.get("parameters_declared") or [],
                bool(expect.get("parameters_have_bounds"))))
        if expect.get("uses_library_source"):
            checks.append(_check_library_source(design_doc))
        if "min_cutouts" in expect:
            checks.append(_check_cutouts(report, expect["min_cutouts"]))
        if "min_outer_height_mm" in expect:
            checks.append(_check_min_height(report, expect["min_outer_height_mm"]))
        if "preview_min_iou" in expect:
            checks.append(_check_preview_iou(report, float(expect["preview_min_iou"])))

    passed_count = sum(1 for _, ok, _ in checks if ok)
    return {
        "id": case.get("id"),
        "passed": bool(checks) and passed_count == len(checks),
        "score": (passed_count / float(len(checks))) if checks else 0.0,
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
    }


def summarise(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Roll per-case results into the headline numbers, cases first.

    Cases-passed leads because it is the honest measure: a candidate that half-satisfies every case
    has produced nothing usable, while its check ratio would look respectable.
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    checks = [c for r in results for c in (r.get("checks") or [])]
    checks_passed = sum(1 for c in checks if c.get("passed"))
    return {
        "cases": total,
        "cases_passed": passed,
        "case_pass_rate": round(passed / float(total), 3) if total else 0.0,
        "checks": len(checks),
        "checks_passed": checks_passed,
        "check_pass_rate": round(checks_passed / float(len(checks)), 3) if checks else 0.0,
        "failed_cases": [r.get("id") for r in results if not r.get("passed")],
    }


def format_table(results: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    """A readable scoreboard: every failed check named, because that is the actionable part."""
    lines = ["%-24s %-6s %6s  %s" % ("case", "result", "score", "failed checks")]
    for result in results:
        failed = [c["name"] for c in (result.get("checks") or []) if not c.get("passed")]
        lines.append("%-24s %-6s %5.0f%%  %s" % (
            str(result.get("id"))[:24],
            "PASS" if result.get("passed") else "FAIL",
            100.0 * float(result.get("score") or 0.0),
            ", ".join(failed[:4]) + (" ..." if len(failed) > 4 else ""),
        ))
    lines.append("")
    lines.append("cases %s/%s (%.0f%%)   checks %s/%s (%.0f%%)" % (
        summary["cases_passed"], summary["cases"], 100.0 * summary["case_pass_rate"],
        summary["checks_passed"], summary["checks"], 100.0 * summary["check_pass_rate"]))
    return "\n".join(lines)
