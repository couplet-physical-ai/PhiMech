"""Smoke tests for the PhiMech launcher and standard-library CLI surface.

These tests intentionally run the real ``bin/phimech`` process.  The module remains runnable
with /usr/bin/python3, where pytest may not be installed, so the direct runner below mirrors
the convention used by the core PhiMech tests.
"""

from __future__ import annotations

import os as _os

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin" / "phimech"
sys.path.insert(0, str(REPO))

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by system Python
    pytest = None  # type: ignore[assignment]


class _Skipped(Exception):
    """Fallback skip signal for direct execution without pytest."""


def _skip(message: str) -> None:
    # pytest.skip() raises Skipped, which derives from BaseException, so it sails through the
    # `except Exception` in _run_direct and turns a skip into a crash with a traceback and exit 1.
    # Delegating merely because pytest is IMPORTABLE was the bug: `bin/phimech check` runs each test
    # file directly with the venv python, where pytest is installed but is not driving. pytest sets
    # PYTEST_CURRENT_TEST for the duration of each test, so it tells "installed" from "running".
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _run(*arguments: str) -> subprocess.CompletedProcess:
    """Run the launcher with a stable repository import path."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [str(BIN)] + list(arguments),
        cwd=str(REPO),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_version() -> None:
    result = _run("version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_help() -> None:
    result = _run("help")
    assert result.returncode == 0, result.stderr
    assert "PhiMech" in result.stdout
    assert "doctor" in result.stdout


def test_doctor_json() -> None:
    result = _run("doctor", "--json")
    assert result.returncode in (0, 1), result.stderr
    report = json.loads(result.stdout)
    assert {"pythons", "venv", "backends", "mcp", "repo", "examples", "ok"} <= set(report)


def test_ir_schema() -> None:
    result = _run("ir", "schema")
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema["title"] == "PhiMech design IR"


def test_nonexistent_subcommand() -> None:
    result = _run("not-a-command")
    assert result.returncode != 0
    assert "traceback" not in (result.stdout + result.stderr).lower()


def test_render_tiny_cube() -> None:
    openscad = shutil.which("openscad") or "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
    if not Path(openscad).is_file() and shutil.which("openscad") is None:
        _skip("OpenSCAD is not installed")
    with tempfile.TemporaryDirectory(prefix="phimech-cli-") as directory:
        root = Path(directory)
        ast = root / "cube.json"
        output = root / "cube.stl"
        ast.write_text(json.dumps({"kind": "cube", "size": [1, 2, 3]}), encoding="utf-8")
        result = _run("render", str(ast), "-o", str(output))
        assert result.returncode == 0, result.stderr + result.stdout
        assert output.is_file()
        assert output.stat().st_size > 0


def test_synth_check_without_pipeline() -> None:
    result = _run("synth", "check", str(REPO / "examples" / "bare-assembly.json"))
    combined = result.stdout + result.stderr
    assert "traceback" not in combined.lower()
    assert result.returncode in (0, 1, 2)
    if result.returncode == 2:
        assert "not available yet" in combined


def test_setup_help() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "phimech" / "setup_venv.py"), "--help"],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--force" in result.stdout


def _run_direct() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(test.__name__, exc))
        except Exception as exc:  # pragma: no cover - direct-run failure reporting
            failures += 1
            print("FAIL {}: {}".format(test.__name__, exc))
        else:
            print("PASS {}".format(test.__name__))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
