"""Host side of the B-rep lane: find the kernel python, run the worker, parse its JSON.

Stdlib only, and it never imports a geometry kernel. The kernel lives in its own virtualenv
because `build123d` drags in `cadquery-ocp`, which is roughly 900 MB of compiled OpenCascade —
too heavy for the core, and exactly the sort of binary that must not share an interpreter with
the MCP server (the core boundary, and the PhiCad protobuf/numpy lesson behind it).

Resolution order for the interpreter, first hit wins:

1. `PHIMECH_BREP_PYTHON` — an explicit override, which is also how CI pins it.
2. `build/brep-venv/bin/python` in the current repo — what `bin/phimech brep setup` creates.
3. `sys.executable`, but only if it can already import `build123d`, so a developer who
   installed it globally is not forced to make a venv.

Absence is not an error. `available()` returns False and every caller degrades to the existing
OpenSCAD/FreeCAD lane, so a clone with no kernel still builds everything it built before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_TIMEOUT_S = 300.0
_VENV_RELATIVE = Path("build") / "brep-venv" / "bin" / "python"


class BrepError(RuntimeError):
    """The worker refused or failed. The message is the worker's own, kept verbatim."""


@dataclass
class BrepResult:
    ok: bool
    measurements: Dict[str, Any] = field(default_factory=dict)
    written: Dict[str, str] = field(default_factory=dict)
    kernel: Optional[str] = None
    kernel_version: Optional[str] = None
    error: Optional[str] = None
    #: Set when the worker refused the program on a numeric geometry gate rather than failing
    #: to build it. Deliberately absent from `to_dict()`: the CLI's JSON shape is a contract.
    gate: Optional[str] = None
    stderr: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "measurements": self.measurements, "written": self.written,
            "kernel": self.kernel, "kernel_version": self.kernel_version, "error": self.error,
        }


def raise_for(result: BrepResult) -> BrepResult:
    """Return an ok result, or raise the exception its failure deserves.

    Shared by `build()` and by every caller of `build_many()`, so a batched caller reports a
    refusal exactly the way a single build does instead of re-deriving the mapping and drifting
    from it. A gate refusal is the *program's* fault and leaves as `ProgramError` (exit 2 at
    the CLI); anything else is the kernel's or the environment's and leaves as `BrepError`.
    """
    if result.ok:
        return result
    from phimech.brep.program import ProgramError

    if result.gate:
        raise ProgramError(result.error or "refused by a geometry gate")
    raise BrepError(result.error or "worker failed without a reason")


def _repo_root() -> Path:
    configured = os.environ.get("PHIMECH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _can_import_build123d(python: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(python), "-c", "import build123d"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def find_python() -> Optional[Path]:
    """Locate an interpreter that can import `build123d`, or None."""
    override = os.environ.get("PHIMECH_BREP_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.exists() else None
    venv = _repo_root() / _VENV_RELATIVE
    if venv.exists():
        return venv
    current = Path(sys.executable)
    if _can_import_build123d(current):
        return current
    return None


def available() -> bool:
    """Cheap probe. True only if an interpreter exists *and* imports the kernel."""
    python = find_python()
    return bool(python) and _can_import_build123d(python)


def doctor() -> Dict[str, Any]:
    """Human-readable state of the B-rep lane, including where the path came from."""
    override = os.environ.get("PHIMECH_BREP_PYTHON")
    venv = _repo_root() / _VENV_RELATIVE
    if override:
        source = "PHIMECH_BREP_PYTHON"
    elif venv.exists():
        source = "build/brep-venv"
    else:
        source = "sys.executable" if _can_import_build123d(Path(sys.executable)) else "not found"
    python = find_python()
    info: Dict[str, Any] = {
        "python": str(python) if python else None,
        "source": source,
        "available": False,
        "kernel_version": None,
    }
    if not python:
        info["hint"] = "run `bin/phimech brep setup` to create build/brep-venv"
        return info
    try:
        proc = subprocess.run(
            [str(python), "-c", "import build123d as b; print(getattr(b,'__version__','?'))"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = str(exc)
        return info
    if proc.returncode == 0:
        info["available"] = True
        info["kernel_version"] = proc.stdout.strip() or None
    else:
        info["error"] = (proc.stderr or "").strip().splitlines()[-1:] or ["import failed"]
        info["hint"] = "run `bin/phimech brep setup`"
    return info


def build(program: Mapping[str, Any], outputs: Mapping[str, Any], *,
          timeout: float = DEFAULT_TIMEOUT_S) -> BrepResult:
    """Build a feature program and write the requested files.

    `outputs` maps a format (`step`, `stl`, `brep`) to an absolute path. Validation happens
    here first, so a malformed program never starts a subprocess.

    Raises `ProgramError` twice over: once here for anything the numbers alone can settle, and
    again on the way back for a gate only the kernel could apply — a cutter that met no
    material. Both are the program's fault rather than the build's, so both leave by the same
    door and the CLI exits 2 on either. `BrepError` stays what it has always been: the kernel
    or the environment failed.
    """
    from phimech.brep.program import validate

    validate(program)  # raises ProgramError with an op index; cheap and worth doing twice
    payload, stderr = _invoke(
        {"program": dict(program), "outputs": _resolve_outputs(outputs)}, timeout=timeout
    )
    return raise_for(_result(payload, stderr))


def build_many(items: Sequence[Mapping[str, Any]], *,
               timeout: float = DEFAULT_TIMEOUT_S) -> List[BrepResult]:
    """Build several feature programs in **one** kernel start, one result each, in order.

    `import build123d` costs about 3 s — more than most of our programs take to build — so a
    caller with twenty small variants was spending a minute of process startup and seconds of
    geometry. This is still one bounded subprocess with an explicit timeout that exits when it
    is done, which is what rule 5 asks for; it simply hands it more than one program.

    Each item is `{"program": {...}, "outputs": {fmt: absolute path}}`. Unlike `build()` this
    does **not** raise on a program that failed: one program's refusal is not another's
    verdict. Inspect `result.ok`, or pass the one you care about through `raise_for()` to get
    exactly the exception `build()` would have raised.
    """
    from phimech.brep.program import validate

    if not items:
        return []
    payloads = []
    for item in items:
        program = item.get("program")
        validate(program)
        payloads.append({"program": dict(program),
                         "outputs": _resolve_outputs(item.get("outputs") or {})})
    payload, stderr = _invoke({"programs": payloads}, timeout=timeout)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(payloads):
        raise BrepError(
            "batch reply held {} result(s) for {} program(s)".format(
                len(results) if isinstance(results, list) else "no", len(payloads))
        )
    for entry in results:                      # the kernel identity is per invocation
        entry.setdefault("kernel", payload.get("kernel"))
        entry.setdefault("kernel_version", payload.get("kernel_version"))
    return [_result(entry, stderr) for entry in results]


def _resolve_outputs(outputs: Mapping[str, Any]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for fmt, path in outputs.items():
        target = Path(path).expanduser()
        if not target.is_absolute():
            raise BrepError(
                "output path for {!r} must be absolute, got {!r} — relative paths are rejected "
                "rather than resolved, because the worker's cwd is not the caller's".format(
                    fmt, str(path))
            )
        resolved[str(fmt)] = str(target)
    return resolved


def _result(payload: Mapping[str, Any], stderr: str) -> BrepResult:
    return BrepResult(
        ok=bool(payload.get("ok")),
        measurements=payload.get("measurements") or {},
        written=payload.get("written") or {},
        kernel=payload.get("kernel"),
        kernel_version=payload.get("kernel_version"),
        error=payload.get("error"),
        gate=payload.get("gate"),
        stderr=stderr,
    )


def _invoke(request: Dict[str, Any], *, timeout: float) -> Tuple[Dict[str, Any], str]:
    """Run the worker once and return its parsed JSON plus stderr. One place, both shapes."""
    python = find_python()
    if not python:
        raise BrepError(
            "no interpreter with build123d found; run `bin/phimech brep setup` or set "
            "PHIMECH_BREP_PYTHON"
        )

    repo = _repo_root()
    request = dict(request, repo=str(repo))
    env = dict(os.environ)
    # Keep the child from re-entering a GUI stack, and make the repo importable for program.py.
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        proc = subprocess.run(
            [str(python), "-m", "phimech.brep.worker"],
            input=json.dumps(request), capture_output=True, text=True, timeout=timeout, env=env,
            cwd=str(repo),
        )
    except subprocess.TimeoutExpired:
        raise BrepError(
            "kernel timed out after {:.0f}s; a fillet radius larger than the wall it sits on is "
            "the usual cause".format(timeout)
        )
    except OSError as exc:
        raise BrepError("could not run the worker: {}".format(exc))

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    if not stdout:
        tail = "\n".join(stderr.splitlines()[-8:])
        raise BrepError(
            "worker produced no JSON (exit {}). stderr tail:\n{}".format(proc.returncode, tail)
        )
    try:
        payload = json.loads(stdout)
    except ValueError:
        raise BrepError(
            "worker stdout was not JSON (exit {}); this usually means something wrote to fd 1 "
            "around a kernel call. First 400 chars:\n{}".format(proc.returncode, stdout[:400])
        )
    return payload, stderr


def setup(*, python: Optional[str] = None, timeout: float = 1800.0) -> Dict[str, Any]:
    """Create `build/brep-venv` with build123d in it.

    Uses `uv` when present because the OCP wheel is large and uv is markedly faster; falls back
    to `venv` + `pip`. Network calls strip a configured HTTP proxy variables (rule 8), which otherwise
    fail with `405` or a malformed HTTP response.
    """
    repo = _repo_root()
    target = repo / "build" / "brep-venv"
    base = python or "python3.12"
    env = {k: v for k, v in os.environ.items()
           if k not in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                        "ALL_PROXY", "all_proxy")}
    steps: List[Dict[str, Any]] = []

    def _run(cmd: List[str]) -> bool:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            steps.append({"cmd": " ".join(cmd), "ok": False, "error": str(exc)})
            return False
        ok = proc.returncode == 0
        steps.append({
            "cmd": " ".join(cmd), "ok": ok,
            "error": None if ok else "\n".join((proc.stderr or "").splitlines()[-4:]),
        })
        return ok

    import shutil
    uv = shutil.which("uv")
    if uv:
        ok = _run([uv, "venv", "--python", base, str(target)])
        if ok:
            ok = _run([uv, "pip", "install", "--python", str(target / "bin" / "python"),
                       "build123d"])
    else:
        ok = _run([base, "-m", "venv", str(target)])
        if ok:
            ok = _run([str(target / "bin" / "pip"), "install", "--upgrade", "pip"])
            ok = _run([str(target / "bin" / "pip"), "install", "build123d"]) and ok

    return {"ok": bool(ok), "venv": str(target), "steps": steps,
            "python": str(target / "bin" / "python") if ok else None}
