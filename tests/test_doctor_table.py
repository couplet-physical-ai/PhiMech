"""The doctor table must print what the hook sections actually return.

Each `HOOK_SECTIONS` provider returns its own dict and `_print_table` formats a row from it; the
two are written in different PRs, so a renamed key silently degrades to a "?" placeholder (the
`bosl2` row read `file_count` while `doctor_section()` returned `files`). A "?" in a healthy row
is a contract break, not a cosmetic issue - an agent reading `doctor` cannot tell it from a
probe that failed.
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout

sys.path.insert(0, ".")

from phimech import doctor  # noqa: E402
from phimech.vendor import bosl2  # noqa: E402


def _row(report: dict, name: str) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        doctor._print_table(report)
    rows = [line for line in buffer.getvalue().splitlines() if line.startswith(name)]
    assert len(rows) == 1, "expected exactly one {!r} row, got {!r}".format(name, rows)
    return rows[0]


def _base_report() -> dict:
    return {
        "python": {}, "venv": {}, "backends": {}, "mcp": {}, "repo": {}, "examples": {},
        "brep": {}, "agent": {}, "guide": {}, "edit_risk": {}, "preview": {}, "aliases": {},
        "ok": True,
    }


def test_bosl2_row_reports_the_vendored_file_count() -> None:
    section = bosl2.doctor_section()
    assert section["ok"], section
    report = _base_report()
    report["bosl2"] = section
    row = _row(report, "bosl2")
    assert "?" not in row, row
    match = re.search(r"\((\d+) files\)", row)
    assert match, row
    assert int(match.group(1)) == section["files"] >= 50


def test_bosl2_row_prefers_a_problem_over_the_summary() -> None:
    report = _base_report()
    report["bosl2"] = {"ok": False, "problems": ["LICENSE missing"], "files": 3}
    row = _row(report, "bosl2")
    assert "LICENSE missing" in row and "files" not in row


if __name__ == "__main__":
    test_bosl2_row_reports_the_vendored_file_count()
    test_bosl2_row_prefers_a_problem_over_the_summary()
    print("doctor table tests passed")
