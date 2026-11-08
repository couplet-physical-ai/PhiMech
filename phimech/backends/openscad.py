"""Subprocess-backed OpenSCAD implementation of the PhiMech backend contract.

OpenSCAD remains outside the PhiMech Python process: this avoids importing its GUI/runtime
libraries and makes the backend usable from the system Python and FreeCAD's Python alike.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from phimech.csg.ast import Node
from phimech.backends.base import (
    Backend,
    BackendError,
    BBox,
    Measurement,
    RenderResult,
    register,
)
from phimech.backends.scad_emit import emit_file
from phimech.vendor import bosl2


_OPENSCAD_FALLBACK = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
_SUPPORTED_FORMATS = ("stl", "3mf", "obj", "off", "amf", "scad", "png", "svg", "dxf")
_AVAILABLE: Optional[bool] = None
_VERSION: Optional[str] = None
_MANIFOLD_SUPPORTED: Optional[bool] = None


def _env() -> Dict[str, str]:
    """The subprocess environment: the caller's, plus our vendored libraries on `OPENSCADPATH`.

    Set on every render rather than only when a tree uses a library call: it is one variable,
    it makes `include <BOSL2/std.scad>` resolve identically for a render, a PNG and a `.scad`
    export re-run by hand, and putting our directory first means a user's own BOSL2 checkout
    cannot shadow the pinned copy the gate measured.
    """
    env = dict(os.environ)
    env["OPENSCADPATH"] = bosl2.openscadpath(env.get("OPENSCADPATH"))
    return env


def _candidate_binary() -> Optional[str]:
    """Find OpenSCAD in the documented override order."""
    configured = os.environ.get("PHIMECH_OPENSCAD")
    if configured:
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which("openscad")
    if found:
        return found
    if os.path.isfile(_OPENSCAD_FALLBACK) and os.access(_OPENSCAD_FALLBACK, os.X_OK):
        return _OPENSCAD_FALLBACK
    return None


def _output_text(value: object) -> str:
    """Normalize subprocess output, including timeout output on Python 3.9."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _combined_log(stdout: object, stderr: object) -> str:
    out = _output_text(stdout)
    err = _output_text(stderr)
    if out and err:
        return out.rstrip() + "\n" + err.rstrip()
    return (out or err).rstrip()


def _version_probe(binary: str) -> Optional[str]:
    """Read the version from stderr, where OpenSCAD writes it."""
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = _combined_log(completed.stdout, completed.stderr)
    match = re.search(r"OpenSCAD\s+version\s+([^\s]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if completed.returncode == 0 and text.strip():
        return text.strip().splitlines()[0].strip()
    return None


def _manifold_probe(binary: str) -> bool:
    """Check once whether this OpenSCAD build accepts the Manifold backend flag."""
    try:
        completed = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    text = _combined_log(completed.stdout, completed.stderr).lower()
    return "--backend" in text and "manifold" in text


def _triangles_from_ascii(data: bytes) -> Optional[List[Tuple[Tuple[float, float, float], ...]]]:
    """Parse STL vertex records, returning None when the data is not ASCII STL."""
    if not data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    vertices: List[Tuple[float, float, float]] = []
    found_vertex = False
    for line in text.splitlines():
        fields = line.strip().split()
        if not fields or fields[0].lower() != "vertex":
            continue
        found_vertex = True
        if len(fields) != 4:
            raise BackendError("invalid ASCII STL vertex record")
        try:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError as exc:
            raise BackendError("invalid ASCII STL vertex number") from exc
    if not found_vertex:
        return None
    if len(vertices) % 3:
        raise BackendError("ASCII STL has an incomplete triangle")
    return [
        tuple(vertices[i : i + 3]) for i in range(0, len(vertices), 3)
    ]  # type: ignore[list-item]


def _triangles_from_binary(data: bytes) -> List[Tuple[Tuple[float, float, float], ...]]:
    """Parse the fixed 50-byte triangle records in a binary STL."""
    if len(data) < 84:
        raise BackendError("STL is too short to contain a binary header")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + count * 50
    if expected > len(data):
        raise BackendError("binary STL is truncated")
    triangles: List[Tuple[Tuple[float, float, float], ...]] = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles.append(
            (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
        )
        offset += 50
    if not triangles:
        raise BackendError("STL contains no triangles")
    return triangles


def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def measure_stl(path: Path) -> Measurement:
    """Measure an ASCII or binary STL using only the standard library.

    The signed tetrahedra sum is absolute-valued for the public solid volume because STL
    files encountered in the wild may consistently wind their facets inward.  This helper
    intentionally reports one solid: connectivity analysis belongs in the check layer.
    """
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise BackendError("unable to read STL: {}".format(source), stderr=str(exc)) from exc
    triangles = _triangles_from_ascii(data)
    if triangles is None:
        triangles = _triangles_from_binary(data)

    points = [point for triangle in triangles for point in triangle]
    minimum = tuple(min(point[index] for point in points) for index in range(3))
    maximum = tuple(max(point[index] for point in points) for index in range(3))
    signed_volume = 0.0
    area = 0.0
    for v0, v1, v2 in triangles:
        edge1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        edge2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        normal = _cross(edge1, edge2)
        area += 0.5 * (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
        signed_volume += (
            v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
            + v0[1] * (v1[2] * v2[0] - v1[0] * v2[2])
            + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])
        ) / 6.0
    return Measurement(
        bbox=BBox(min=minimum, max=maximum),
        volume=abs(signed_volume),
        area=area,
        solids=1,
    )


class OpenSCADBackend(Backend):
    """OpenSCAD backend that emits source and invokes the CLI as a child process."""

    name = "openscad"
    supported_formats = _SUPPORTED_FORMATS

    def available(self) -> bool:
        """Return a cached process-wide availability result."""
        global _AVAILABLE, _VERSION
        if _AVAILABLE is not None:
            return _AVAILABLE
        if _VERSION is not None:
            _AVAILABLE = True
            return True
        binary = _candidate_binary()
        if binary is None:
            _AVAILABLE = False
            return False
        _VERSION = _version_probe(binary)
        _AVAILABLE = _VERSION is not None
        return _AVAILABLE

    def version(self) -> Optional[str]:
        """Return the parsed OpenSCAD version, probing once when necessary."""
        global _VERSION, _AVAILABLE
        if _VERSION is not None:
            return _VERSION
        binary = _candidate_binary()
        if binary is None:
            _AVAILABLE = False
            return None
        _VERSION = _version_probe(binary)
        _AVAILABLE = _VERSION is not None
        return _VERSION

    def _supports_manifold(self) -> bool:
        global _MANIFOLD_SUPPORTED
        if _MANIFOLD_SUPPORTED is None:
            binary = _candidate_binary()
            _MANIFOLD_SUPPORTED = binary is not None and _manifold_probe(binary)
        return _MANIFOLD_SUPPORTED

    def doctor(self) -> Dict[str, object]:
        """The base report plus where library includes resolve, so a missing vendor copy is
        visible next to the binary that would need it."""
        info = dict(Backend.doctor(self))
        library = bosl2.doctor_section()
        info.update({
            "openscadpath": bosl2.openscadpath(os.environ.get("OPENSCADPATH")),
            "libraries": {
                library["library"]: {
                    "ok": library["ok"],
                    "version": library.get("version"),
                    "commit": library.get("commit"),
                    "files": library.get("files"),
                }
            },
        })
        return info

    def render(
        self,
        node: Node,
        out_path: Path,
        fmt: Optional[str] = None,
        *,
        timeout: float = 600.0,
        preview: bool = False
    ) -> RenderResult:
        """Emit and render one file, keeping all OpenSCAD execution out of this process.

        `preview` only affects PNG output, where it selects OpenSCAD's preview pipeline instead of
        a full CGAL/manifold evaluation. Preview draws surfaces as they are, so a scene containing
        an imported open shell -- a vendor visualisation mesh, say -- still appears. The default
        stays on full evaluation: a solid that silently fails to be manifold must not be able to
        look fine in a gate render.
        """
        self._validate_node(node)
        destination = Path(out_path)
        output_format = self._fmt_from(destination, fmt)
        if not self.supports(output_format):
            raise BackendError("OpenSCAD does not support .{} output".format(output_format))
        destination.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()

        if output_format == "scad":
            emit_file(node, destination)
            return RenderResult(
                path=destination,
                fmt=output_format,
                backend=self.name,
                seconds=time.monotonic() - started,
                bytes=destination.stat().st_size,
            )

        binary = _candidate_binary()
        if binary is None:
            raise BackendError("OpenSCAD executable not found")
        with tempfile.TemporaryDirectory(prefix="phimech-openscad-") as temporary:
            source = Path(temporary) / "model.scad"
            emit_file(node, source)
            command = [
                binary,
                "-o",
                str(destination),
                "--export-format",
                output_format,
            ]
            if output_format != "png" and self._supports_manifold():
                command.extend(["--backend", "manifold"])
            if output_format == "png":
                command.extend(
                    [
                        "--autocenter",
                        "--viewall",
                        "--imgsize",
                        "1200,900",
                        "--colorscheme",
                        "Tomorrow",
                    ]
                )
                # `--render` is a presence flag: `--render false` still forces full evaluation, so
                # the preview path has to omit it entirely.
                if not preview:
                    command.extend(["--render", "true"])
            command.append(str(source))
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env=_env(),
                )
            except subprocess.TimeoutExpired as exc:
                log = _combined_log(exc.stdout, exc.stderr)
                raise BackendError(
                    "OpenSCAD timed out after {:.1f}s".format(timeout), stderr=log
                ) from exc
            except OSError as exc:
                raise BackendError("unable to execute OpenSCAD", stderr=str(exc)) from exc
            log = _combined_log(completed.stdout, completed.stderr)
            if completed.returncode != 0:
                raise BackendError(
                    "OpenSCAD failed with exit code {}".format(completed.returncode),
                    stderr=log,
                    returncode=completed.returncode,
                )
            if not destination.is_file():
                raise BackendError(
                    "OpenSCAD completed without creating {}".format(destination), stderr=log
                )
            try:
                size = destination.stat().st_size
            except OSError as exc:
                raise BackendError("OpenSCAD output is not readable", stderr=str(exc)) from exc
            if size == 0:
                raise BackendError("OpenSCAD created an empty {}".format(destination), stderr=log)
        return RenderResult(
            path=destination,
            fmt=output_format,
            backend=self.name,
            seconds=time.monotonic() - started,
            bytes=size,
            log=log,
        )

    def measure(self, node: Node, *, timeout: float = 600.0) -> Measurement:
        """Render to a temporary STL and measure its triangles in-process."""
        with tempfile.TemporaryDirectory(prefix="phimech-openscad-measure-") as temporary:
            stl = Path(temporary) / "measure.stl"
            self.render(node, stl, "stl", timeout=timeout)
            return measure_stl(stl)


register(OpenSCADBackend())


__all__ = ["OpenSCADBackend", "measure_stl"]
