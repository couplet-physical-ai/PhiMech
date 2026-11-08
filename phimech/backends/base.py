"""Backend interface — how a CSG `Node` becomes a file on disk.

Two implementations ship: `openscad.py` (emits a `.scad` program and runs the `openscad`
binary) and `freecad.py` (runs a script inside `freecadcmd` that builds `Part` shapes).
Both are subprocess-driven: PhiMech never imports FreeCAD or links OpenSCAD in-process, so
the MCP venv stays a plain python and the backends can be swapped, pinned, or absent.

The contract every backend honours
----------------------------------
* `render(node, out_path, fmt)` writes exactly one file at `out_path` and returns a
  `RenderResult`. It raises `BackendError` (never a bare CalledProcessError) on failure,
  with the tool's stderr attached — that text is what the agent sees, so keep it.
* `measure(node)` returns a `Measurement`: bounding box, volume, surface area — the numbers
  every downstream gate (fit, wall thickness, printability) reasons about. Volume must be
  the true solid volume, not a bbox volume.
* `available()` is a cheap probe (binary exists, responds to `--version`). `doctor()` gives
  the human-readable version of the same check.
* Paths passed in are absolute. Backends never resolve relative to their own cwd.

Formats
-------
`FORMATS` is the union of what the two backends can emit; each backend advertises its own
subset in `supported_formats`. Callers pick a backend by asking `for_format()`.
"""

from __future__ import annotations

import abc
import dataclasses
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from phimech.csg.ast import Node

Vec3 = Tuple[float, float, float]

FORMATS: Tuple[str, ...] = ("stl", "step", "3mf", "obj", "off", "amf", "scad", "fcstd", "png", "svg", "dxf")


class BackendError(RuntimeError):
    """A backend subprocess failed. `stderr` holds the tool's own diagnostics verbatim."""

    def __init__(self, message: str, *, stderr: str = "", returncode: Optional[int] = None):
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode

    def __str__(self) -> str:  # keep the tail of stderr visible to agents
        base = super().__str__()
        tail = self.stderr.strip().splitlines()[-12:]
        return base + ("\n" + "\n".join(tail) if tail else "")


@dataclass(frozen=True)
class BBox:
    min: Vec3
    max: Vec3

    @property
    def size(self) -> Vec3:
        return (self.max[0] - self.min[0], self.max[1] - self.min[1], self.max[2] - self.min[2])

    @property
    def center(self) -> Vec3:
        return tuple((a + b) / 2.0 for a, b in zip(self.min, self.max))  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, object]:
        return {"min": list(self.min), "max": list(self.max), "size": list(self.size)}


@dataclass(frozen=True)
class Measurement:
    bbox: BBox
    volume: float  # mm^3, true solid volume
    area: float  # mm^2, surface area
    solids: int = 1  # number of disjoint solids (1 for a manifold single body)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "bbox": self.bbox.to_dict(),
            "volume_mm3": self.volume,
            "area_mm2": self.area,
            "solids": self.solids,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass(frozen=True)
class RenderResult:
    path: Path
    fmt: str
    backend: str
    seconds: float
    bytes: int
    log: str = ""  # tool stdout+stderr, for the agent's benefit
    measurement: Optional[Measurement] = None

    def to_dict(self) -> Dict[str, object]:
        d = dataclasses.asdict(self)
        d["path"] = str(self.path)
        d["measurement"] = self.measurement.to_dict() if self.measurement else None
        return d


class Backend(abc.ABC):
    """One CAD kernel behind a uniform surface."""

    name: str = "abstract"
    supported_formats: Tuple[str, ...] = ()

    @abc.abstractmethod
    def available(self) -> bool:
        """Cheap probe: can this backend run at all right now?"""

    @abc.abstractmethod
    def version(self) -> Optional[str]:
        """Tool version string, or None if unavailable."""

    @abc.abstractmethod
    def render(self, node: Node, out_path: Path, fmt: Optional[str] = None, *, timeout: float = 600.0) -> RenderResult:
        """Write `node` to `out_path` (format from `fmt` or the path suffix)."""

    @abc.abstractmethod
    def measure(self, node: Node, *, timeout: float = 600.0) -> Measurement:
        """Bounding box + volume + area of the evaluated solid."""

    def doctor(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "available": self.available(),
            "version": self.version(),
            "formats": list(self.supported_formats),
        }

    def supports(self, fmt: str) -> bool:
        return fmt.lower().lstrip(".") in self.supported_formats

    @staticmethod
    def _validate_node(node: Node) -> None:
        """Reject malformed ASTs before an optional CAD executable is consulted."""
        from phimech.csg.validate import assert_valid

        try:
            assert_valid(node)
        except ValueError as exc:
            raise BackendError("invalid CSG AST: {}".format(exc)) from exc

    @staticmethod
    def _fmt_from(out_path: Path, fmt: Optional[str]) -> str:
        f = (fmt or out_path.suffix.lstrip(".")).lower()
        if f not in FORMATS:
            raise ValueError(f"unknown format {f!r}; known: {', '.join(FORMATS)}")
        return f


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------

_REGISTRY: Dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    _REGISTRY[backend.name] = backend
    return backend


def get(name: str) -> Backend:
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no backend {name!r}; have {sorted(_REGISTRY)}") from None


def all_backends() -> List[Backend]:
    _ensure_loaded()
    return list(_REGISTRY.values())


def for_format(fmt: str, prefer: Sequence[str] = ("openscad", "freecad")) -> Backend:
    """First available backend (in `prefer` order) that can emit `fmt`."""
    _ensure_loaded()
    fmt = fmt.lower().lstrip(".")
    for name in prefer:
        b = _REGISTRY.get(name)
        if b and b.supports(fmt) and b.available():
            return b
    for b in _REGISTRY.values():
        if b.supports(fmt) and b.available():
            return b
    raise BackendError(f"no available backend can produce .{fmt}; run `phimech doctor`")


def _ensure_loaded() -> None:
    """Import the concrete backends lazily so `phimech.csg` stays import-light."""
    if _REGISTRY:
        return
    for mod in ("phimech.backends.openscad", "phimech.backends.freecad"):
        try:
            __import__(mod)
        except ImportError:
            pass


def which(*names: str) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


__all__ = [
    "FORMATS",
    "Backend",
    "BackendError",
    "BBox",
    "Measurement",
    "RenderResult",
    "register",
    "get",
    "all_backends",
    "for_format",
    "which",
]
