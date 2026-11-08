"""Doctor must judge the interpreter the launcher actually runs.

`bin/phimech` picks ``PHIMECH_PYTHON``, then ``PHIMECH_VENV/bin/python``, then
``<repo>/build/venv``. Doctor looked only at ``PHIMECH_VENV`` and ``<repo>/build/venv``, so a git
worktree that borrowed another checkout's venv through ``PHIMECH_PYTHON`` ran every command fine
while doctor reported "not ready" against a venv nobody was using. And the launcher ignored
``PHIMECH_VENV`` - the override the project architecture documents - so the two disagreed in both directions.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phimech import doctor  # noqa: E402

_OVERRIDES = ("PHIMECH_PYTHON", "PHIMECH_VENV")


@contextmanager
def _environment(**values: Optional[str]) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in _OVERRIDES}
    try:
        for name in _OVERRIDES:
            os.environ.pop(name, None)
        for name, value in values.items():
            if value is not None:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fake_venv(root: Path, script: str = "") -> Path:
    """A directory shaped like a venv whose ``bin/python`` is a tiny shell script."""
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.write_text("#!/bin/sh\n" + (script or 'echo "FAKE $0 $*"\n'), encoding="utf-8")
    python.chmod(0o755)
    return python


def test_phimech_python_selects_its_venv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp) / "worktree"  # no build/venv of its own
        checkout.mkdir()
        python = _fake_venv(Path(tmp) / "other-checkout" / "build" / "venv")
        with _environment(PHIMECH_PYTHON=str(python)):
            info = doctor._venv_info(checkout)
        assert info["source"] == "PHIMECH_PYTHON", info
        assert info["python"] == str(python), info
        assert info["path"] == str(python.parent.parent), info
        assert info["exists"] is True, info


def test_phimech_python_outside_a_venv_is_reported_as_the_interpreter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with _environment(PHIMECH_PYTHON=sys.executable):
            resolved = doctor._resolve_interpreter(Path(tmp))
        assert resolved["source"] == "PHIMECH_PYTHON"
        assert resolved["python"] == Path(sys.executable)


def test_phimech_python_wins_over_phimech_venv_like_the_launcher() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        first = _fake_venv(Path(tmp) / "a")
        _fake_venv(Path(tmp) / "b")
        with _environment(PHIMECH_PYTHON=str(first), PHIMECH_VENV=str(Path(tmp) / "b")):
            resolved = doctor._resolve_interpreter(Path(tmp))
        assert resolved["python"] == first and resolved["source"] == "PHIMECH_PYTHON"


def test_default_is_the_checkouts_own_build_venv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with _environment():
            info = doctor._venv_info(Path(tmp))
        assert info["source"] == doctor.DEFAULT_VENV_SOURCE
        assert info["path"] == str(Path(tmp) / "build" / "venv")
        assert info["exists"] is False and info["ok"] is False


def test_table_names_the_override_and_does_not_tell_you_to_run_setup() -> None:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    report = {"venv": {"path": "/x/bin/python", "source": "PHIMECH_PYTHON", "exists": False,
                       "imports": {}, "ok": False}}
    with redirect_stdout(buffer):
        doctor._print_table(report)
    row = [line for line in buffer.getvalue().splitlines() if line.startswith("venv")][0]
    assert "[PHIMECH_PYTHON]" in row and "PHIMECH_PYTHON points at nothing" in row, row
    assert "bin/phimech setup" not in row, row


def _launch(environment: Dict[str, str]) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _OVERRIDES}
    env.update(environment)
    completed = subprocess.run(
        ["bash", str(REPO / "bin" / "phimech"), "version"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30,
        check=False,
    )
    return completed.stdout


def test_launcher_honours_phimech_venv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        python = _fake_venv(Path(tmp) / "venv")
        output = _launch({"PHIMECH_VENV": str(python.parent.parent)})
        assert output.startswith("FAKE {} -m phimech.cli version".format(python)), output


def test_launcher_prefers_phimech_python_over_phimech_venv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        chosen = _fake_venv(Path(tmp) / "chosen")
        ignored = _fake_venv(Path(tmp) / "ignored")
        output = _launch({"PHIMECH_PYTHON": str(chosen),
                          "PHIMECH_VENV": str(ignored.parent.parent)})
        assert output.startswith("FAKE {} ".format(chosen)), output


if __name__ == "__main__":
    test_phimech_python_selects_its_venv()
    test_phimech_python_outside_a_venv_is_reported_as_the_interpreter()
    test_phimech_python_wins_over_phimech_venv_like_the_launcher()
    test_default_is_the_checkouts_own_build_venv()
    test_table_names_the_override_and_does_not_tell_you_to_run_setup()
    test_launcher_honours_phimech_venv()
    test_launcher_prefers_phimech_python_over_phimech_venv()
    print("ok")
