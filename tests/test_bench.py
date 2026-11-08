"""The grader must be strict in the ways that matter, and it must not be gameable.

Runs both ways::

    /usr/bin/python3 tests/test_bench.py
    pytest tests/test_bench.py

These tests use synthetic reports rather than real builds, deliberately: the grader is the durable
half of the benchmark, and it has to be verifiable offline, in CI, with no CAD kernel installed. The
real end-to-end run is recorded in fixtures/bench/sol-baseline/README.md and reproducible with
``bin/phimech bench grade``.

The property under test throughout is that a *bad* candidate cannot score well by omission - the
failure mode that would make the whole benchmark worthless.
"""

from __future__ import annotations

import json
import os as _os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

from phimech.bench.grade import (
    case_by_id,
    format_table,
    grade_case,
    load_suite,
    summarise,
)

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "fixtures" / "bench" / "sol-baseline"


class _Skipped(Exception):
    pass


def _skip(message: str) -> None:
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _report(gates: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """A synthetic build report. `gates` maps name -> passed (True/False/None) or a full dict."""
    results: List[Dict[str, Any]] = []
    for name, value in gates.items():
        if isinstance(value, dict):
            entry = {"name": name}
            entry.update(value)
        else:
            entry = {"name": name, "passed": value, "severity": "error", "message": ""}
        results.append(entry)
    report: Dict[str, Any] = {"ok": True, "errors": [], "gates": {"results": results}}
    report.update(extra)
    return report


# -- the suite itself ----------------------------------------------------------------------


def test_suite_is_well_formed() -> None:
    """A malformed case would silently score every candidate wrongly."""
    suite = load_suite()
    ids = [case["id"] for case in suite["cases"]]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in suite["cases"]:
        assert case.get("prompt"), "%s has no prompt" % case["id"]
        assert case.get("why"), "%s does not say why it exists" % case["id"]
        expect = case.get("expect")
        assert isinstance(expect, dict) and expect, "%s measures nothing" % case["id"]


def test_every_named_gate_exists() -> None:
    """A typo'd gate name would be an unpassable check - the case could never be satisfied."""
    try:
        from phimech.check.gates import gate_catalog
    except Exception as exc:  # pragma: no cover - only if check deps are absent
        _skip("gate catalogue unavailable: %s" % exc)
    known = {entry["name"] for entry in gate_catalog()}
    suite = load_suite()

    def named(expect: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for key in ("gates_pass", "gates_pass_or_warn", "gates_not_skipped"):
            out.extend(expect.get(key) or [])
        for branch in expect.get("either") or []:
            out.extend(named(branch))
        return out

    for case in suite["cases"]:
        for name in named(case["expect"]):
            assert name in known, "%s names unknown gate %r" % (case["id"], name)


# -- the semantics that make the score meaningful -------------------------------------------


def test_a_skipped_gate_is_not_a_pass() -> None:
    """The single most important rule here.

    A skipped gate checked nothing. Scoring it as a pass would let a model earn points by omitting
    the very declaration that makes a gate run - which is precisely what printer-bed-aware tests.
    """
    case = {"id": "t", "expect": {"gates_pass": ["fits_bed"]}}
    skipped = grade_case(case, _report({"fits_bed": None}))
    assert not skipped["passed"]
    assert "SKIP" in skipped["checks"][0]["detail"].upper()
    assert grade_case(case, _report({"fits_bed": True}))["passed"]


def test_a_missing_gate_is_not_a_pass() -> None:
    case = {"id": "t", "expect": {"gates_pass": ["manifold"]}}
    result = grade_case(case, _report({}))
    assert not result["passed"] and "not present" in result["checks"][0]["detail"]


def test_absent_candidate_scores_zero_not_neutral() -> None:
    """An unanswered case must cost as much as a wrong one, or skipping becomes a strategy."""
    case = {"id": "t", "expect": {"builds": True, "gates_pass": ["manifold", "min_feature"]}}
    result = grade_case(case, None, None, None)
    assert result["score"] == 0.0 and not result["passed"]
    assert result["checks"], "an absent candidate must still report failed checks"


def test_warn_severity_is_accepted_only_where_the_case_says_so() -> None:
    warned = _report({"overhang": {"passed": False, "severity": "warn"}})
    assert grade_case({"id": "t", "expect": {"gates_pass_or_warn": ["overhang"]}}, warned)["passed"]
    # The same gate under the strict key must fail, or `gates_pass` would mean nothing.
    assert not grade_case({"id": "t", "expect": {"gates_pass": ["overhang"]}}, warned)["passed"]


def test_error_severity_is_never_accepted_as_a_warning() -> None:
    failed = _report({"manifold": {"passed": False, "severity": "error"}})
    assert not grade_case({"id": "t", "expect": {"gates_pass_or_warn": ["manifold"]}}, failed)[
        "passed"
    ]


def test_either_branch_passes_when_one_branch_does() -> None:
    """How a brief that *should* be refused is scored: refuse it, or build it printable."""
    case = case_by_id(load_suite(), "box-thin-wall-refused")
    assert case is not None
    refused = grade_case(case, None, None, validates=False)
    assert refused["passed"], refused
    clamped = grade_case(case, _report({"wall_thickness": True}), {}, validates=True)
    assert clamped["passed"], clamped
    # Emitting the unprintable wall anyway satisfies neither branch.
    naive = grade_case(case, _report({"wall_thickness": False}), {}, validates=True)
    assert not naive["passed"]


def test_size_check_ignores_orientation_but_not_magnitude() -> None:
    """A rotated design is the same design; a wrong-sized one is not."""
    expect = {"outer_size_mm": {"value": [48, 68, 20], "tolerance": 6}}
    rotated = _report({}, measurements={"base": {"bbox": {"size": [68, 48, 20]}}})
    assert grade_case({"id": "t", "expect": expect}, rotated)["passed"]
    wrong = _report({}, measurements={"base": {"bbox": {"size": [120, 68, 20]}}})
    assert not grade_case({"id": "t", "expect": expect}, wrong)["passed"]


def test_parameters_must_carry_bounds() -> None:
    """An unbounded parameter is not slider-safe, which is the whole point of the case."""
    expect = {"parameters_declared": ["wall"], "parameters_have_bounds": True}
    bounded = _report({}, parameters={"wall": {"value": 2, "min": 1.2, "max": 5}})
    assert grade_case({"id": "t", "expect": expect}, bounded)["passed"]
    unbounded = _report({}, parameters={"wall": {"value": 2}})
    assert not grade_case({"id": "t", "expect": expect}, unbounded)["passed"]


def test_parameter_names_match_loosely() -> None:
    """`wall_thickness` is a reasonable name for `wall`; scoring style would be wrong."""
    expect = {"parameters_declared": ["wall"]}
    report = _report({}, parameters={"wall_thickness": {"value": 2, "min": 1, "max": 4}})
    assert grade_case({"id": "t", "expect": expect}, report)["passed"]


def test_library_source_is_detected_and_guessing_is_not() -> None:
    expect = {"uses_library_source": True}
    library = {"components": [{"ref": "A", "source": {"kind": "library", "part": "x"}}]}
    assert grade_case({"id": "t", "expect": expect}, _report({}), library)["passed"]
    guessed = {"components": [{"ref": "A", "source": {"kind": "box", "size": [1, 2, 3]}}]}
    assert not grade_case({"id": "t", "expect": expect}, _report({}), guessed)["passed"]


def test_build_failure_fails_even_with_green_gates() -> None:
    """synth_build can report ok=false with every gate green; the case must notice."""
    report = _report({"manifold": True}, ok=False, errors=["cannot write STEP"])
    result = grade_case({"id": "t", "expect": {"builds": True, "gates_pass": ["manifold"]}}, report)
    assert not result["passed"]


# -- reporting -----------------------------------------------------------------------------


def test_summary_leads_with_cases_not_checks() -> None:
    results = [
        {"id": "a", "passed": True, "checks": [{"name": "x", "passed": True}]},
        {"id": "b", "passed": False,
         "checks": [{"name": "y", "passed": True}, {"name": "z", "passed": False}]},
    ]
    summary = summarise(results)
    assert summary["cases"] == 2 and summary["cases_passed"] == 1
    assert summary["case_pass_rate"] == 0.5
    assert summary["checks_passed"] == 2 and summary["checks"] == 3
    assert summary["failed_cases"] == ["b"]
    table = format_table(results, summary)
    assert "PASS" in table and "FAIL" in table and "z" in table, table


def test_baseline_submission_is_present_and_complete() -> None:
    """The recorded baseline must answer every case, or its score is not comparable."""
    if not BASELINE.is_dir():
        _skip("baseline submission is absent")
    suite = load_suite()
    for case in suite["cases"]:
        candidate = BASELINE / ("%s.json" % case["id"])
        assert candidate.is_file(), "baseline does not answer %s" % case["id"]
        design = json.loads(candidate.read_text(encoding="utf-8"))
        assert design.get("project"), "%s has no project" % case["id"]


def _run_direct() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(test.__name__, exc))
        except Exception as exc:  # pragma: no cover
            failures += 1
            print("FAIL {}: {}".format(test.__name__, exc))
        else:
            print("PASS {}".format(test.__name__))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
