"""FreeCAD backend — the B-rep half of PhiMech: STEP out, true analytic measurements.

Why this exists next to the OpenSCAD backend
--------------------------------------------
OpenSCAD is a mesh CSG engine: fast, but its "volume" is the volume of a tessellated
approximation, it cannot write STEP, and it has no notion of a face to fillet. FreeCAD/OCC
gives us exact solids, so this backend owns:
  * `.step` (and `.fcstd`) export — what a machinist or a Windchill upload actually wants,
  * measurements that gates compare against numeric thresholds (a cylinder's volume here is
    `pi*r^2*h`, not the volume of a 32-gon prism),
  * the place where fillets/chamfers will land, because OCC can do them and OpenSCAD cannot.

How it talks to FreeCAD (the core boundary)
------------------------------------------
Never in-process. FreeCAD 1.1.3 bundles its own conda-forge python 3.11 with its own
numpy/protobuf; importing it into the PhiMech process is the exact ABI trap PhiCad already
paid for. Instead:

    node.to_dict() -> job JSON -> freecadcmd <freecad_worker.py> <job.json> -> PHIMECH_RESULT

`freecadcmd` prints a banner on stdout and OCC prints `saving...(5 %)` progress spam, so the
worker emits one machine-readable marker line and we parse only that.

The job path travels in `$PHIMECH_JOB`, not argv, because `freecadcmd` treats trailing
positional arguments as *documents to open* — a `.json` one reaches FreeCAD's FEM mesh
importer, which fails on our node dicts with `invalid literal for int(): 'kind'`. (Probed on
1.1.3; `freecad_worker.py` documents this and the `__name__` quirk that goes with it.)

Startup cost drives the API shape
---------------------------------
A `freecadcmd` process costs ~0.7-1.3 s before it runs a line of our code (measured; see
tests). So every public method here is *one* subprocess:
  * `render()` exports the file **and** measures in the same pass — `RenderResult.measurement`
    is always populated, unlike the OpenSCAD backend where measuring means re-reading an STL.
  * `render_many()` writes every requested format from a single evaluated shape. The synth
    pipeline uses it to get step+stl+3mf for the price of one FreeCAD launch.

Unsupported nodes
-----------------
`UNSUPPORTED` names the AST kinds this backend must not be handed (`minkowski`, `scad_module`,
plus twisted `linear_extrude`). `hull` *is* supported — FreeCAD 1.1.3 has no
`Part.makeConvexHull`, but it does bundle scipy, so the worker hulls a tessellated point cloud.
`can_render()` lets the pipeline route a node to OpenSCAD before paying for a launch.

`scad_module` is refused by design, not by omission: it is a call into an OpenSCAD library
(BOSL2, `phimech/vendor/BOSL2/`), and FreeCAD has no OpenSCAD interpreter to evaluate it with.
The refusal happens before the binary is even looked for, so a design that uses BOSL2 gets
"render it with the openscad backend" and never "freecadcmd not found". The consequence callers
must know: **no STEP export for parts containing a `scad_module`** — the OpenSCAD lane produces
STL/3MF/OBJ/PNG, and the pipeline's STL-to-STEP fallback yields a tessellated STEP, not a B-rep.

stdlib only, imports under python 3.9.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from phimech.backends.base import (
    BBox,
    Backend,
    BackendError,
    Measurement,
    RenderResult,
    register,
)
from phimech.csg.ast import Node, walk

WORKER = Path(__file__).with_name("freecad_worker.py")

RESULT_MARKER = "PHIMECH_RESULT"
ERROR_MARKER = "PHIMECH_ERROR"

#: Probed on macOS with the `freecad` cask; the MacOS/FreeCADCmd spelling is the older layout.
DEFAULT_BINARIES: Tuple[str, ...] = (
    "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd",
    "/usr/lib/freecad/bin/freecadcmd",
    "/usr/bin/freecadcmd",
    "/snap/bin/freecad.cmd",
)


class FreeCADBackend(Backend):
    """Drives `freecadcmd` as a child process. One subprocess per public call."""

    name = "freecad"
    supported_formats: Tuple[str, ...] = ("step", "stl", "obj", "off", "3mf", "fcstd")

    #: AST kinds this backend cannot evaluate. The pipeline checks `can_render()` and falls
    #: back to OpenSCAD rather than eating a subprocess failure. `scad_module` is a call into a
    #: vendored OpenSCAD library — there is nothing for a B-rep kernel to build from.
    UNSUPPORTED = frozenset(("minkowski", "scad_module"))

    #: Mesh tessellation defaults for stl/obj/off/3mf, in mm and radians.
    deflection = 0.05
    angular = 0.2

    def __init__(self, binary: Optional[str] = None) -> None:
        self._binary_override = binary
        self._version_cache: Optional[str] = None
        self._version_probed = False

    # -- discovery ---------------------------------------------------------------------

    @property
    def binary(self) -> Optional[str]:
        """`PHIMECH_FREECADCMD`, then `$PATH`, then the known install locations."""
        if self._binary_override:
            return self._binary_override
        env = os.environ.get("PHIMECH_FREECADCMD")
        if env:
            return env
        found = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
        if found:
            return found
        for candidate in DEFAULT_BINARIES:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def binary_source(self) -> str:
        """Where the binary came from — `phimech doctor` prints this so a wrong FreeCAD is
        obvious without guessing."""
        if self._binary_override:
            return "constructor"
        if os.environ.get("PHIMECH_FREECADCMD"):
            return "PHIMECH_FREECADCMD"
        if shutil.which("freecadcmd") or shutil.which("FreeCADCmd"):
            return "PATH"
        return "default path" if self.binary else "not found"

    def available(self) -> bool:
        binary = self.binary
        return bool(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK)

    def version(self) -> Optional[str]:
        """Parsed from `freecadcmd --version`, which returns in ~0.7 s.

        Deliberately not `-c 'import FreeCAD; print(FreeCAD.Version())'`: that pays the full
        ~1.3 s kernel import, and `-c` on freecadcmd is the stdin-hang footgun from the project architecture.
        """
        if self._version_probed:
            return self._version_cache
        self._version_probed = True
        self._version_cache = None
        if not self.available():
            return None
        try:
            proc = subprocess.run(
                [str(self.binary), "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60.0,
                env=self._env(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        # "FreeCAD 1.1.3 Revision: 20260725 (Git shallow)" -> "1.1.3"
        for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower().startswith("freecad"):
                self._version_cache = parts[1].rstrip(",")
                break
        return self._version_cache

    def doctor(self) -> Dict[str, object]:
        info = dict(Backend.doctor(self))
        info.update({
            "binary": self.binary,
            "binary_source": self.binary_source(),
            "worker": str(WORKER),
            "unsupported": sorted(self.UNSUPPORTED),
        })
        return info

    # -- capability --------------------------------------------------------------------

    # 2D primitives: a hull over these is a PLANAR hull, which qhull cannot triangulate and
    # which has no volume as a solid. OpenSCAD hulls in 2D natively, so such a node must be
    # routed there instead of crashing a FreeCAD subprocess 0.3 s later.
    _2D_KINDS = frozenset(("square", "circle", "polygon", "text"))

    def can_render(self, node: Node) -> Optional[str]:
        """None if this backend can evaluate `node`, else why it cannot (for the router)."""
        for sub in walk(node):
            if sub.kind == "scad_module":
                return (
                    "'scad_module' {}() from <{}> is OpenSCAD library source (BOSL2); the freecad "
                    "backend has no OpenSCAD interpreter, so this node renders only on the "
                    "openscad backend (stl/3mf/obj/png) and cannot be exported to STEP".format(
                        getattr(sub, "module", "?"), getattr(sub, "include", "?"))
                )
            if sub.kind in self.UNSUPPORTED:
                return "{!r} is not supported by the freecad backend".format(sub.kind)
            if sub.kind == "linear_extrude":
                twist = abs(float(getattr(sub, "twist", 0.0) or 0.0))
                if twist > 1e-9:
                    return "linear_extrude twist is not supported by the freecad backend"
            if sub.kind == "hull" and self._is_planar_hull(sub):
                return (
                    "hull over 2D children is planar; the freecad backend needs a 3D point "
                    "cloud (qhull reports a flat initial simplex)"
                )
        return None

    @classmethod
    def _is_planar_hull(cls, hull_node: Node) -> bool:
        """True when every leaf under `hull_node` is a 2D primitive.

        Checked structurally rather than by measuring extents, because the router must decide
        before any geometry is evaluated.
        """
        leaves = [
            sub.kind
            for sub in walk(hull_node)
            if not getattr(sub, "children", None) and sub.kind != "hull"
        ]
        return bool(leaves) and all(kind in cls._2D_KINDS for kind in leaves)

    # -- rendering ---------------------------------------------------------------------

    def render(
        self,
        node: Node,
        out_path: Path,
        fmt: Optional[str] = None,
        *,
        timeout: float = 600.0,
    ) -> RenderResult:
        """Write `node` to `out_path`. Measurement comes back in the same subprocess."""
        out_path = Path(out_path)
        resolved = self._fmt_from(out_path, fmt)
        results = self.render_many(node, {resolved: out_path}, timeout=timeout, measure=True)
        return results[resolved]

    def render_many(
        self,
        node: Node,
        outputs: Mapping[str, Path],
        *,
        timeout: float = 600.0,
        measure: bool = True,
    ) -> Dict[str, RenderResult]:
        """Evaluate `node` once and export every `{fmt: path}` pair in one `freecadcmd` run.

        This is the method the synth pipeline should call: FreeCAD's start-up is the dominant
        cost, so five formats in one launch beat five launches by roughly the launch count.
        """
        if not outputs:
            raise ValueError("render_many needs at least one output")
        spec: Dict[str, str] = {}
        for fmt, path in outputs.items():
            resolved = self._fmt_from(Path(path), fmt)
            if resolved not in self.supported_formats:
                raise BackendError(
                    "the freecad backend cannot write .{}; it supports {}".format(
                        resolved, ", ".join(self.supported_formats)))
            spec[resolved] = str(Path(path).resolve())

        started = time.time()
        payload = self._run_job(node, outputs=spec, measure=measure, timeout=timeout)
        elapsed = time.time() - started
        measurement = _measurement_from(payload.get("measurement")) if measure else None
        log = _log_from(payload)

        results: Dict[str, RenderResult] = {}
        for fmt, info in (payload.get("outputs") or {}).items():
            path = Path(info["path"])
            if not path.exists():
                raise BackendError("freecad reported writing {} but it is missing".format(path))
            results[fmt] = RenderResult(
                path=path,
                fmt=fmt,
                backend=self.name,
                seconds=float(info.get("seconds", elapsed)),
                bytes=int(info.get("bytes", path.stat().st_size)),
                log=log,
                measurement=measurement,
            )
        missing = sorted(set(spec) - set(results))
        if missing:
            raise BackendError(
                "freecad did not export: {}".format(", ".join(missing)), stderr=log)
        return results

    def measure(self, node: Node, *, timeout: float = 600.0) -> Measurement:
        """Exact bbox/volume/area/solid count. No file is written."""
        payload = self._run_job(node, outputs={}, measure=True, timeout=timeout)
        return _measurement_from(payload.get("measurement"))

    # -- subprocess plumbing -----------------------------------------------------------

    def _env(self, job_path: Optional[str] = None) -> Dict[str, str]:
        """Headless env for FreeCAD. Without `QT_QPA_PLATFORM=offscreen` a Qt-linked build can
        try to reach a display and stall; `PYTHONNOUSERSITE` keeps a stray `~/.local`
        site-packages from shadowing FreeCAD's own numpy. `PHIMECH_JOB` is how the job path
        reaches the worker without freecadcmd trying to open it as a document."""
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)  # our python's path must never leak into FreeCAD's
        env.pop("PYTHONHOME", None)
        if job_path:
            env["PHIMECH_JOB"] = job_path
        else:
            env.pop("PHIMECH_JOB", None)
        return env

    def _run_job(
        self,
        node: Node,
        *,
        outputs: Mapping[str, str],
        measure: bool,
        timeout: float,
    ) -> Dict[str, object]:
        self._validate_node(node)
        # Refuse what this backend can never render before asking whether it is installed: a
        # BOSL2 design on a machine without FreeCAD should hear "use openscad", not "install
        # freecad", because installing it would not help.
        reason = self.can_render(node)
        if reason:
            raise BackendError(reason + "; render it with the openscad backend instead")
        binary = self.binary
        if not self.available():
            raise BackendError(
                "freecadcmd not found (looked at PHIMECH_FREECADCMD, PATH, {}); "
                "install with `brew install --cask freecad`".format(DEFAULT_BINARIES[0]))
        if not WORKER.is_file():
            raise BackendError("the freecad worker script is missing at {}".format(WORKER))

        job = {
            "node": node.to_dict(),
            "outputs": dict(outputs),
            "measure": bool(measure),
            "deflection": self.deflection,
            "angular": self.angular,
        }
        handle, job_path = tempfile.mkstemp(prefix="phimech-freecad-", suffix=".json")
        try:
            with os.fdopen(handle, "w") as fh:
                json.dump(job, fh)
            command = [str(binary), str(WORKER)]
            try:
                proc = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,  # freecadcmd blocks on stdin if it is a tty
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    env=self._env(job_path),
                )
            except subprocess.TimeoutExpired as exc:
                raise BackendError(
                    "freecadcmd timed out after {:.0f}s: {}".format(timeout, " ".join(command)),
                    stderr=_decode(exc.stderr),
                ) from None
            except OSError as exc:
                raise BackendError("could not run {}: {}".format(binary, exc)) from None
            stdout = _decode(proc.stdout)
            stderr = _decode(proc.stderr)
            return _parse_output(stdout, stderr, proc.returncode, command)
        finally:
            try:
                os.unlink(job_path)
            except OSError:
                pass


# --------------------------------------------------------------------------------------
# output parsing
# --------------------------------------------------------------------------------------


def _decode(raw: Optional[bytes]) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def _parse_output(
    stdout: str, stderr: str, returncode: int, command: List[str]
) -> Dict[str, object]:
    """Pull the single marker line out of freecadcmd's noise.

    Scanned newest-first: OCC progress spam can precede the marker, never follow it.
    """
    result: Optional[Dict[str, object]] = None
    error: Optional[Dict[str, object]] = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if result is None and line.startswith(RESULT_MARKER):
            result = _json_or_none(line[len(RESULT_MARKER):])
        elif error is None and line.startswith(ERROR_MARKER):
            error = _json_or_none(line[len(ERROR_MARKER):])
        if result is not None:
            break
    if result is not None:
        return result
    if error is not None:
        detail = str(error.get("traceback") or "") or stderr
        raise BackendError(
            "freecad: {}".format(error.get("error") or "unknown modelling error"),
            stderr=detail,
            returncode=returncode,
        )
    raise BackendError(
        "freecadcmd produced no result marker (exit {}); command: {}".format(
            returncode, " ".join(command)),
        stderr=(stderr + "\n" + stdout).strip(),
        returncode=returncode,
    )


def _json_or_none(text: str) -> Optional[Dict[str, object]]:
    try:
        parsed = json.loads(text.strip())
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _measurement_from(raw: object) -> Measurement:
    if not isinstance(raw, dict):
        raise BackendError("freecad returned no measurement")
    bbox = raw.get("bbox") or {}
    lo = [float(x) for x in (bbox.get("min") or (0.0, 0.0, 0.0))]
    hi = [float(x) for x in (bbox.get("max") or (0.0, 0.0, 0.0))]
    extra: Dict[str, object] = {}
    for key in ("valid", "shape_type", "empty"):
        if key in raw:
            extra[key] = raw[key]
    return Measurement(
        bbox=BBox(min=(lo[0], lo[1], lo[2]), max=(hi[0], hi[1], hi[2])),
        volume=float(raw.get("volume") or 0.0),
        area=float(raw.get("area") or 0.0),
        solids=int(raw.get("solids") or 0),
        extra=extra,
    )


def _log_from(payload: Mapping[str, object]) -> str:
    """A short, agent-readable digest: timings, FreeCAD version, and any worker warnings."""
    bits = ["freecad {} on {}".format(payload.get("freecad", "?"), platform.machine())]
    if payload.get("build_seconds") is not None:
        bits.append("build {}s".format(payload["build_seconds"]))
    if payload.get("seconds") is not None:
        bits.append("total {}s".format(payload["seconds"]))
    lines = [", ".join(bits)]
    for warning in payload.get("warnings") or []:  # type: ignore[union-attr]
        lines.append("warning: {}".format(warning))
    return "\n".join(lines)


register(FreeCADBackend())

__all__ = ["FreeCADBackend", "WORKER", "DEFAULT_BINARIES"]
