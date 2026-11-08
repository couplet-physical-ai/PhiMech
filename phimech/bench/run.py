"""Build every candidate design in a directory and grade it against the suite.

The split that matters: **PhiMech does not call models.** A harness (phi, Claude Code, Cursor, a
script) reads the prompts, authors ``<case-id>.json`` into a directory, and this module builds and
grades them. That keeps credentials out of PhiMech, makes the score comparable across harnesses
instead of tied to one client, and leaves the grader runnable offline and in CI.

stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from phimech.bench.grade import format_table, grade_case, load_suite, summarise


def candidate_paths(candidates_dir: Union[str, Path], suite: Mapping[str, Any]) -> Dict[str, Path]:
    """Map case id -> candidate IR path, accepting ``<id>.json`` or ``<id>/design.json``."""
    root = Path(candidates_dir)
    found: Dict[str, Path] = {}
    for case in suite.get("cases", []):
        case_id = case.get("id") if isinstance(case, dict) else None
        if not case_id:
            continue
        for candidate in (root / ("%s.json" % case_id), root / case_id / "design.json"):
            if candidate.is_file():
                found[case_id] = candidate
                break
    return found


def run_case(
    case: Mapping[str, Any],
    ir_path: Optional[Path],
    out_root: Path,
    *,
    build: bool = True,
) -> Dict[str, Any]:
    """Validate, build and grade one candidate. A missing candidate scores zero, never skips."""
    case_id = str(case.get("id"))
    if ir_path is None:
        result = grade_case(case, None, None, None)
        result["note"] = "no candidate submitted"
        return result

    from phimech.ir import ir as ir_module

    design: Optional[Dict[str, Any]] = None
    validates: Optional[bool] = None
    try:
        design, errors = ir_module.load_and_validate(ir_path)
        validates = not errors
    except Exception as exc:  # a malformed file is a generation failure, not a crash
        validates = False
        errors = [str(exc)]

    report: Optional[Dict[str, Any]] = None
    if validates:
        from phimech.synth import pipeline

        try:
            if build:
                out_dir = out_root / case_id
                out_dir.mkdir(parents=True, exist_ok=True)
                report = pipeline.build(str(ir_path), out_dir, formats=["stl"], backend="auto")
            else:
                # Still gate it, just without geometry: `check` runs placement, clearance, wall and
                # the rest, and skips only the mesh-dependent gates. Grading validation alone would
                # make the fast mode almost meaningless, since a design can validate and still be
                # unbuildable nonsense.
                report = pipeline.check(str(ir_path))
        except Exception as exc:
            report = {"ok": False, "errors": [str(exc)], "gates": {"results": []}}

    result = grade_case(case, report, design, validates)
    result["candidate"] = str(ir_path)
    if not validates:
        result["validation_errors"] = list(errors)[:5]
    return result


def run_suite(
    candidates_dir: Union[str, Path],
    out_root: Union[str, Path],
    *,
    suite_path: Union[str, Path, None] = None,
    only: Optional[Sequence[str]] = None,
    build: bool = True,
) -> Dict[str, Any]:
    """Grade a whole submission directory. Returns results plus the summary."""
    suite = load_suite(suite_path)
    found = candidate_paths(candidates_dir, suite)
    out_root = Path(out_root)
    results: List[Dict[str, Any]] = []
    for case in suite.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id"))
        if only and case_id not in only:
            continue
        results.append(run_case(case, found.get(case_id), out_root, build=build))
    summary = summarise(results)
    return {
        "suite": suite.get("suite"),
        "candidates_dir": str(candidates_dir),
        "results": results,
        "summary": summary,
        "table": format_table(results, summary),
    }


def prompts(suite_path: Union[str, Path, None] = None) -> str:
    """The briefs, as a model should receive them: id, prompt, and what will be measured.

    Handing over the expectations is deliberate. This measures whether a model can design to a
    stated requirement, not whether it can guess a hidden rubric - and a hidden rubric would only
    reward memorising this suite.
    """
    suite = load_suite(suite_path)
    blocks = [
        "PhiMech generation benchmark: %s" % suite.get("suite"),
        "",
        "For each case, author a design IR and save it as <id>.json in the submission directory.",
        "Use `bin/phimech ir schema` for the schema and `bin/phimech ir example <name>`",
        "Every case is graded by machine against the checks listed under it.",
        "",
    ]
    for case in suite.get("cases", []):
        if not isinstance(case, dict):
            continue
        blocks.append("## %s" % case.get("id"))
        blocks.append(str(case.get("prompt", "")).strip())
        expect = case.get("expect") or {}
        blocks.append("Measured: %s" % json.dumps(expect, sort_keys=True))
        blocks.append("")
    return "\n".join(blocks)
