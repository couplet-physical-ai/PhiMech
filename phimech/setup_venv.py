"""Create the PhiMech development virtual environment.

Setup is kept as a small standard-library script so the first run works before any optional
MCP or mesh-check packages are installed.  Package installation is retried with the explicit
runtime dependencies when editable-project packaging is the part that failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


_PROXY_NAMES = (
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


def _heal_fastmcp(venv: Path) -> None:
    """Remove a fastmcp package directory left broken by an in-place major upgrade.

    fastmcp 4.x split into `fastmcp` + `fastmcp-slim`. Installing it OVER an existing 2.x
    (the pin PhiMech inherited from PhiCad) leaves `site-packages/fastmcp/` with no
    `__init__.py`: pip reports 4.0.3 installed while `from fastmcp import FastMCP` dies with
    "cannot import name 'FastMCP' ... (unknown location)". Deleting the stale directory lets
    the install below lay it down cleanly. Cheap to check, so it runs every time.
    """
    for libdir in sorted((venv / "lib").glob("python3.*")):
        package = libdir / "site-packages" / "fastmcp"
        if package.is_dir() and not (package / "__init__.py").is_file():
            print("removing a broken fastmcp install at {}".format(package))
            shutil.rmtree(str(package), ignore_errors=True)


def _repo() -> Path:
    """Resolve the checkout path supplied by ``bin/phimech`` or the source tree."""
    configured = os.environ.get("PHIMECH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _python() -> str:
    """Select the interpreter that owns the project venv."""
    configured = os.environ.get("PHIMECH_PYTHON")
    if configured:
        return configured
    homebrew = Path("/opt/homebrew/bin/python3.13")
    if homebrew.is_file():
        return str(homebrew)
    found = shutil.which("python3")
    if found:
        return found
    raise RuntimeError("no Python interpreter found; set PHIMECH_PYTHON")


def _clean_environment() -> dict:
    """Strip proxy variables that can interfere with package index traffic."""
    environment = os.environ.copy()
    for name in _PROXY_NAMES:
        environment.pop(name, None)
    return environment


def _run(command: List[str], repository: Path) -> None:
    """Run one setup subprocess with network proxy variables removed."""
    subprocess.run(
        command,
        cwd=str(repository),
        env=_clean_environment(),
        check=True,
    )


def setup(force: bool = False) -> Path:
    """Create or reuse the project venv, then install the development extras."""
    # (see _heal_fastmcp below for why the venv is repaired before installing)
    repository = _repo()
    venv = repository / "build" / "venv"
    venv_python = venv / "bin" / "python"
    if force and venv.exists():
        print("removing {}".format(venv))
        shutil.rmtree(str(venv))

    interpreter = _python()
    if not venv_python.is_file():
        venv.parent.mkdir(parents=True, exist_ok=True)
        print("creating {} with {}".format(venv, interpreter))
        _run([interpreter, "-m", "venv", str(venv)], repository)
    else:
        print("reusing {}".format(venv))

    _heal_fastmcp(venv)

    editable = [str(venv_python), "-m", "pip", "install", "-e", ".[dev]"]
    try:
        print("installing PhiMech development dependencies")
        _run(editable, repository)
    except subprocess.CalledProcessError as exc:
        print(
            "editable install failed (exit {}); trying the runtime dependency fallback".format(
                exc.returncode
            ),
            file=sys.stderr,
        )
        fallback = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "fastmcp==2.13.1",
            "jsonschema",
            "numpy-stl",
            "pytest",
        ]
        _run(fallback, repository)

    print("setup complete: {}".format(venv))
    print("next: bin/phimech doctor")
    print("next: bin/phimech mcp register")
    return venv


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse setup options and turn subprocess failures into concise CLI errors."""
    parser = argparse.ArgumentParser(description="Create the PhiMech development venv")
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove and recreate build/venv before installing dependencies",
    )
    args = parser.parse_args(argv)
    try:
        setup(force=args.force)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "setup"]
