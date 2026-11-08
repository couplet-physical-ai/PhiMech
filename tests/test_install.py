"""Non-editable wheel installation smoke test for the public CLI.

The source tree and editable installs can hide missing entry points and package data. This test
builds a wheel, installs it without dependencies into a fresh virtual environment, changes out
of the repository, and invokes only the generated ``phimech`` executable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by system Python
    pytest = None  # type: ignore[assignment]


class _Skipped(Exception):
    """Fallback skip signal for direct execution without pytest."""


def _skip(message: str) -> None:
    if pytest is not None and os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _run(
    command: List[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a smoke-test command with captured output for useful assertion failures."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _installed_command(venv: Path) -> Path:
    """Return the console-script path produced by pip on POSIX or Windows."""
    if os.name == "nt":
        return venv / "Scripts" / "phimech.exe"
    return venv / "bin" / "phimech"


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def test_non_editable_install_exposes_core_cli() -> None:
    """A wheel must work outside the checkout without either optional CAD kernel."""
    if importlib.util.find_spec("build") is None:
        _skip("the build package is not installed; install the dev extra to test wheels")

    with tempfile.TemporaryDirectory(prefix="phimech-install-") as directory:
        root = Path(directory)
        wheelhouse = root / "wheelhouse"
        built = _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheelhouse),
                str(REPO),
            ],
            cwd=root,
        )
        assert built.returncode == 0, built.stdout + built.stderr
        wheels = list(wheelhouse.glob("phimech-*.whl"))
        assert len(wheels) == 1, "expected one PhiMech wheel, found {!r}".format(wheels)

        venv = root / "venv"
        created = _run([sys.executable, "-m", "venv", str(venv)], cwd=root)
        assert created.returncode == 0, created.stdout + created.stderr
        python = _venv_python(venv)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PHIMECH_REPO", None)
        installed = _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                str(wheels[0]),
            ],
            cwd=root,
            env=environment,
        )
        assert installed.returncode == 0, installed.stdout + installed.stderr

        command = _installed_command(venv)
        assert command.is_file(), "wheel did not install the phimech console command"

        version = _run([str(command), "version", "--json"], cwd=root, env=environment)
        assert version.returncode == 0, version.stdout + version.stderr
        assert json.loads(version.stdout)["version"] == "0.1.0"

        schema = _run([str(command), "ir", "schema", "--json"], cwd=root, env=environment)
        assert schema.returncode == 0, schema.stdout + schema.stderr
        assert json.loads(schema.stdout)["title"] == "PhiMech design IR"

        # OpenSCAD falls back to known app paths when an override is merely absent. Point it at
        # an executable that cannot report a version so this probe is unavailable on every host.
        unavailable_openscad = root / "unavailable-openscad"
        unavailable_openscad.write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
        unavailable_openscad.chmod(0o755)
        environment["PHIMECH_OPENSCAD"] = str(unavailable_openscad)
        environment["PHIMECH_FREECADCMD"] = str(root / "missing-freecadcmd")
        doctor = _run([str(command), "doctor", "--json"], cwd=root, env=environment)
        assert doctor.returncode == 1, doctor.stdout + doctor.stderr
        report = json.loads(doctor.stdout)
        assert report["backends"]["openscad"]["available"] is False
        assert report["backends"]["freecad"]["available"] is False
        assert report["ok"] is False


def _run_direct() -> int:
    try:
        test_non_editable_install_exposes_core_cli()
    except _Skipped as exc:
        print("SKIP test_non_editable_install_exposes_core_cli: {}".format(exc))
        return 0
    except Exception as exc:  # pragma: no cover - direct-run failure reporting
        print("FAIL test_non_editable_install_exposes_core_cli: {}".format(exc))
        return 1
    print("PASS test_non_editable_install_exposes_core_cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_direct())
