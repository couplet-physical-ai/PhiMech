"""Report whether the PhiMech command-line environment is ready.

The doctor intentionally uses subprocess probes instead of importing optional dependencies in
this process.  That keeps it useful with the system Python and makes failures actionable when
one of the CAD kernels or the MCP virtual environment is absent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _repo() -> Path:
    """Resolve the checkout used by the launcher, with a source-tree fallback."""
    configured = os.environ.get("PHIMECH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _version_from_output(stdout: object, stderr: object) -> Optional[str]:
    """Normalize Python's version output, which varies between stdout and stderr."""
    values = []
    for value in (stdout, stderr):
        if value:
            values.append(str(value).strip())
    text = "\n".join(item for item in values if item)
    return text.splitlines()[0] if text else None


def _python_probe(path: Path) -> Dict[str, Any]:
    """Return presence and version for one interpreter without importing it here."""
    result: Dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if not result["present"]:
        result["version"] = None
        return result
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["version"] = None
        result["error"] = str(exc)
        return result
    result["version"] = _version_from_output(completed.stdout, completed.stderr)
    result["ok"] = completed.returncode == 0 and result["version"] is not None
    return result


def _module_probe(python: Path, module: str) -> bool:
    """Probe an optional package in a child interpreter to avoid ABI contamination."""
    try:
        completed = subprocess.run(
            [str(python), "-c", "import " + module],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


#: Provenance label for the venv when no override is set; the table prints only non-defaults.
DEFAULT_VENV_SOURCE = "default (repo/build/venv)"


def _resolve_interpreter(repository: Path) -> Dict[str, Any]:
    """Pick the interpreter exactly the way ``bin/phimech`` does, and say why.

    The launcher's order is ``PHIMECH_PYTHON``, then ``PHIMECH_VENV/bin/python``, then
    ``<repo>/build/venv``, then a system python. Doctor used to look only at ``PHIMECH_VENV`` and
    ``<repo>/build/venv``, so a git worktree pointed at another checkout's venv through
    ``PHIMECH_PYTHON`` ran every command fine while doctor reported "not ready" - doctor judging
    a different interpreter from the one it was itself running under.

    Returns ``{"python", "venv", "source"}``; ``venv`` is ``None`` when ``PHIMECH_PYTHON`` names
    an interpreter that is not inside a virtual environment (no ``pyvenv.cfg`` beside ``bin/``).
    """
    configured_python = os.environ.get("PHIMECH_PYTHON", "").strip()
    if configured_python:
        python = Path(configured_python).expanduser()
        root = python.parent.parent
        venv = root if (root / "pyvenv.cfg").is_file() else None
        return {"python": python, "venv": venv, "source": "PHIMECH_PYTHON"}
    configured_venv = os.environ.get("PHIMECH_VENV", "").strip()
    if configured_venv:
        venv = Path(configured_venv).expanduser()
        return {"python": venv / "bin" / "python", "venv": venv, "source": "PHIMECH_VENV"}
    venv = repository / "build" / "venv"
    return {"python": venv / "bin" / "python", "venv": venv, "source": DEFAULT_VENV_SOURCE}


def _venv_info(repository: Path) -> Dict[str, Any]:
    """Inspect the project interpreter and the optional packages used by MCP and checks.

    Resolution is shared with the launcher through `_resolve_interpreter`, and the report names
    which override (if any) it came from, so doctor always health-checks the interpreter that
    `bin/phimech` will actually run.
    """
    resolved = _resolve_interpreter(repository)
    python: Path = resolved["python"]
    venv: Optional[Path] = resolved["venv"]
    present = python.is_file()
    imports = {
        "fastmcp": _module_probe(python, "fastmcp") if present else False,
        "jsonschema": _module_probe(python, "jsonschema") if present else False,
        # The distribution is named numpy-stl, while its import package is ``stl``.
        "numpy-stl": _module_probe(python, "stl") if present else False,
    }
    exists = venv.is_dir() if venv is not None else present
    return {
        "path": str(venv if venv is not None else python),
        "source": resolved["source"],
        "exists": exists,
        "python": str(python),
        "imports": imports,
        "ok": bool(exists and present and all(imports.values())),
    }


def _backend_info() -> Dict[str, Any]:
    """Ask every registered backend for its doctor record, tolerating missing slices."""
    records: Dict[str, Any] = {}
    try:
        base = importlib.import_module("phimech.backends.base")
        backends = base.all_backends()
    except Exception as exc:  # a doctor must report broken optional imports, not traceback
        return {
            "openscad": {"backend": "openscad", "available": False, "error": str(exc)},
            "freecad": {"backend": "freecad", "available": False, "error": str(exc)},
        }

    for backend in backends:
        name = getattr(backend, "name", backend.__class__.__name__.lower())
        try:
            records[str(name)] = backend.doctor()
        except Exception as exc:
            records[str(name)] = {
                "backend": str(name),
                "available": False,
                "error": str(exc),
            }

    # A missing concrete backend is still a diagnostic item and must make the required
    # readiness check fail.  The registry deliberately ignores ImportError for optional
    # backends, so fill in the absent names here rather than hiding that fact.
    for expected in ("openscad", "freecad"):
        if expected not in records:
            records[expected] = {
                "backend": expected,
                "available": False,
                "error": "backend is not registered",
            }
    return records


def _mcp_info() -> Dict[str, Any]:
    """Inspect the user MCP config without assuming a particular registration schema."""
    config_path = Path.home() / ".pi" / "agent" / "mcp.json"
    result: Dict[str, Any] = {
        "path": str(config_path),
        "exists": config_path.is_file(),
        "registered": False,
        "command": None,
        "command_exists": False,
    }
    if not config_path.is_file():
        return result
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
        return result
    if not isinstance(config, dict):
        result["error"] = "MCP config is not a JSON object"
        return result
    # register.py preserves whichever layout the file already uses: a top-level server map,
    # or the wrapped {"mcpServers": {...}} form. Checking only the top level reported "not
    # registered" for a server that was in fact registered and handshaking fine.
    entry = config.get("phimech")
    if not isinstance(entry, dict):
        wrapper = config.get("mcpServers")
        if isinstance(wrapper, dict):
            entry = wrapper.get("phimech")
            result["layout"] = "mcpServers"
    else:
        result["layout"] = "flat"
    result["registered"] = isinstance(entry, dict)
    if not isinstance(entry, dict):
        return result

    command = entry.get("command")
    if isinstance(command, list) and command:
        command = command[0]
    if not isinstance(command, str):
        return result
    result["command"] = command
    command_path = Path(command).expanduser()
    if not command_path.is_absolute():
        command_path = _repo() / command_path
    result["command_exists"] = bool(shutil.which(command) or command_path.is_file())
    return result


def _repo_info(repository: Path) -> Dict[str, Any]:
    """Read branch and dirty state using short, non-network Git commands."""
    result: Dict[str, Any] = {"path": str(repository), "branch": None, "dirty": None}
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
        return result
    if branch.returncode == 0:
        result["branch"] = branch.stdout.strip() or None
    if status.returncode == 0:
        result["dirty"] = bool(status.stdout.strip())
    return result


def _backend_ok(records: Dict[str, Any]) -> bool:
    """Require both supported kernels, not merely one registered backend."""
    return all(
        isinstance(records.get(name), dict) and records[name].get("available") is True
        for name in ("openscad", "freecad")
    )


def doctor() -> Dict[str, Any]:
    """Return a JSON-serializable readiness report for the current checkout."""
    repository = _repo()
    current = {
        "path": str(Path(sys.executable).resolve()),
        "version": platform.python_version(),
        "present": True,
    }
    system = _python_probe(Path("/usr/bin/python3"))
    homebrew = _python_probe(Path("/opt/homebrew/bin/python3.13"))
    pythons = {"current": current, "system": system, "homebrew": homebrew}
    venv = _venv_info(repository)
    backends = _backend_info()
    mcp = _mcp_info()
    repo = _repo_info(repository)
    examples_path = repository / "examples"
    example_count = len(list(examples_path.glob("*.json"))) if examples_path.is_dir() else 0
    examples = {"path": str(examples_path), "count": example_count}
    report = {
        "pythons": pythons,
        "venv": venv,
        "backends": backends,
        "brep": _brep_info(),
        "mcp": mcp,
        "repo": repo,
        "examples": examples,
        # `brep` is deliberately NOT part of readiness. It is an optional ~900 MB kernel; a
        # clone without it still builds everything it built before, so letting its absence
        # turn the whole doctor red would train people to ignore a red doctor.
        "ok": bool(_backend_ok(backends) and venv.get("ok") is True),
    }
    report.update(_hook_sections())
    return report


#: Doctor sections contributed by packages that own their own health check. Each entry is
#: ``(report key, module, attribute)`` naming a ``doctor_section() -> dict`` function. Like
#: ``brep``, none of them gates ``ok``: a checkout without phi, or with a stale guide, still
#: builds everything it built before, and a doctor that goes red for optional things trains
#: people to ignore a red doctor.
HOOK_SECTIONS = (
    ("solidworks", "phimech.adapters.solidworks", "doctor_section"),
    ("bosl2", "phimech.vendor.bosl2", "doctor_section"),
    ("edit_risk", "phimech.interop.edit_risk", "doctor_section"),
    ("preview", "phimech.synth.preview", "doctor_section"),
    ("drawing", "phimech.synth.drawing", "doctor_section"),
    ("aliases", "phimech.ir.aliases", "doctor_section"),
)


def _hook_sections() -> Dict[str, Any]:
    """Collect every ``HOOK_SECTIONS`` record; a failing hook is a finding, not a crash."""
    sections: Dict[str, Any] = {}
    for key, module_name, attribute in HOOK_SECTIONS:
        try:
            sections[key] = getattr(importlib.import_module(module_name), attribute)()
        except Exception as exc:
            sections[key] = {"ok": False, "error": "{}.{} failed: {}".format(
                module_name, attribute, exc)}
    return sections


def _brep_info() -> Dict[str, Any]:
    """State of the optional B-rep lane. Never raises, never affects overall readiness."""
    try:
        kernel = importlib.import_module("phimech.brep.kernel")
        return kernel.doctor()
    except Exception as exc:
        return {"available": False, "source": "import failed", "error": str(exc)}


def _mark(value: Optional[bool]) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "–"


def _print_table(report: Dict[str, Any]) -> None:
    """Print a compact human table while leaving the full structure to ``--json``."""
    print("PhiMech doctor")
    print("section    status  details")
    print("pythons    {}       current {}".format(
        _mark(bool(report.get("pythons"))),
        report.get("pythons", {}).get("current", {}).get("version", "unknown"),
    ))
    venv = report.get("venv", {})
    missing = [name for name, present in (venv.get("imports") or {}).items() if not present]
    venv_detail = venv.get("path", "")
    if venv.get("source") and venv["source"] != DEFAULT_VENV_SOURCE:
        venv_detail += "  [{}]".format(venv["source"])
    if not venv.get("exists"):
        if venv.get("source") in (None, DEFAULT_VENV_SOURCE):
            venv_detail += "  (missing — run `bin/phimech setup`)"
        else:
            venv_detail += "  (missing — {} points at nothing)".format(venv["source"])
    elif missing:
        venv_detail += "  (missing packages: {})".format(", ".join(sorted(missing)))
    print("venv       {}       {}".format(_mark(venv.get("ok")), venv_detail))
    backends = report.get("backends", {})
    for name in ("openscad", "freecad"):
        item = backends.get(name, {})
        details = item.get("version") or item.get("error") or "unavailable"
        print("backend    {}       {}: {}".format(_mark(item.get("available")), name, details))
    brep = report.get("brep", {})
    if brep.get("available"):
        brep_detail = "build123d {} ({})".format(
            brep.get("kernel_version") or "?", brep.get("source", "?")
        )
    else:
        brep_detail = brep.get("hint") or brep.get("error") or "not installed (optional)"
        if isinstance(brep_detail, list):
            brep_detail = "; ".join(str(x) for x in brep_detail)
    print("brep       {}       {}".format(
        _mark(True) if brep.get("available") else "–", brep_detail
    ))
    mcp = report.get("mcp", {})
    mcp_ok = bool(mcp.get("registered") and mcp.get("command_exists"))
    print("mcp        {}       {}".format(_mark(mcp_ok), mcp.get("path", "")))
    agent = report.get("agent", {})
    if agent.get("ok"):
        agent_detail = "phi {} as {} (key {})".format(
            agent.get("phi_version", "?"), agent.get("user", "?"),
            agent.get("key_fingerprint", "?"))
    else:
        agent_detail = agent.get("hint") or agent.get("error") or "phi not found (optional)"
    print("agent      {}       {}".format(_mark(True) if agent.get("ok") else "–", agent_detail))
    bosl2 = report.get("bosl2", {})
    bosl2_detail = bosl2.get("error") or "; ".join(bosl2.get("problems") or []) or (
        "BOSL2 {} ({} files)".format(bosl2.get("version") or bosl2.get("pin", "?")[:12],
                                      bosl2.get("files", "?")))
    print("bosl2      {}       {}".format(_mark(bool(bosl2.get("ok"))), bosl2_detail))
    risk = report.get("edit_risk", {})
    risk_detail = risk.get("error") or "tray fixture assesses {} ({} faces, {} free edges)".format(
        risk.get("level", "?"), risk.get("faces", "?"), risk.get("free_edges", "?"))
    print("edit_risk  {}       {}".format(_mark(bool(risk.get("ok"))), risk_detail))
    sheet = report.get("preview", {})
    sheet_detail = sheet.get("error") or "{} views via openscad {}, IoU threshold {}".format(
        len(sheet.get("views") or []), sheet.get("openscad", "?"),
        sheet.get("iou_threshold", "?"))
    print("preview    {}       {}".format(_mark(bool(sheet.get("ok"))), sheet_detail))
    drawing = report.get("drawing", {})
    solidworks = report.get("solidworks", {})
    print("solidworks {}       {}".format(_mark(bool(solidworks.get("ok"))),
                                         solidworks.get("note", "?")))
    drawing_detail = drawing.get("error") or drawing.get("note", "?")
    print("drawing    {}       {}".format(_mark(bool(drawing.get("ok"))), drawing_detail))
    aliases = report.get("aliases", {})
    needing = aliases.get("examples_needing_normalisation") or []
    aliases_detail = aliases.get("error") or (
        "{} aliases, {} refusal rules; {} example(s) need rewriting".format(
            aliases.get("aliases", "?"), aliases.get("refusal_rules", "?"), len(needing)))
    print("aliases    {}       {}".format(_mark(bool(aliases.get("ok"))), aliases_detail))
    repository = report.get("repo", {})
    print("repo       {}       branch={} dirty={}".format(
        _mark(repository.get("branch") is not None),
        repository.get("branch", "unknown"),
        repository.get("dirty", "unknown"),
    ))
    print(
        "examples   –       {} JSON design(s)".format(
            report.get("examples", {}).get("count", 0)
        )
    )
    print("overall    {}       {}".format(
        _mark(report.get("ok")),
        "ready" if report.get("ok") else "not ready",
    ))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the doctor command and return a shell-friendly status code."""
    parser = argparse.ArgumentParser(description="Probe the PhiMech toolchain")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = doctor()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_table(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["doctor", "main"]
