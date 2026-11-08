"""Import STEP/STL models into measured component envelopes.

Backends are intentionally imported only when a cache miss requires measurement.  This keeps
plain IR/component work usable on machines without FreeCAD or OpenSCAD, and keeps the
component process free of FreeCAD's bundled Python ABI.

Every import also carries an **edit-risk assessment** (``phimech.interop.edit_risk``) in
``Envelope.meta["edit_risk"]``: face / edge / free-edge counts and a level with workflow
guidance, computed from the file with no kernel and cached next to the measurement under the
same mtime+size key.  It is advisory -- the envelope is identical whatever the level, because
``source.kind: import`` already treats the model as a fixed envelope to design around.  What it
changes is what an agent *knows* before it tries to fillet or boolean the import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from phimech.csg.ast import Import, Node, Scale
from phimech.components.envelope import Envelope


_CACHE_SUFFIX = ".phimech-measure.json"


def _resolved(path: Any, base_dir: Optional[Any]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir).expanduser() / candidate
    return candidate.resolve()


def _cache_path(path: Path) -> Path:
    return Path(str(path) + _CACHE_SUFFIX)


def _key_for(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _measurement_size(payload: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    bbox = payload.get("bbox")
    if not isinstance(bbox, dict):
        return None
    minimum = bbox.get("min")
    maximum = bbox.get("max")
    if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
        return None
    if len(minimum) != 3 or len(maximum) != 3:
        return None
    try:
        size = tuple(float(maximum[i]) - float(minimum[i]) for i in range(3))
        return size  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _read_cache_payload(path: Path, key: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """The whole cached record for this exact file version, or ``None``."""
    try:
        with _cache_path(path).open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(cached, dict) or cached.get("key") != key:
        return None
    if not isinstance(cached.get("measurement"), dict):
        return None
    return cached


def _read_cache(path: Path, key: Dict[str, int]) -> Optional[Tuple[float, float, float]]:
    cached = _read_cache_payload(path, key)
    if cached is None:
        return None
    return _measurement_size(cached["measurement"])


def _write_cache(
    path: Path, key: Dict[str, int], measurement: Any, edit_risk: Optional[Dict[str, Any]] = None
) -> None:
    """Best-effort cache writes must never turn a successful CAD measurement into a failure."""
    try:
        payload = {
            "key": key,
            "measurement": measurement if isinstance(measurement, dict) else measurement.to_dict(),
        }
        if edit_risk is not None:
            payload["edit_risk"] = edit_risk
        cache = _cache_path(path)
        temporary = Path(str(cache) + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(cache)
    except (AttributeError, OSError, TypeError, ValueError):
        return


def _measure(path: Path, key: Dict[str, int]) -> Tuple[float, float, float]:
    cached = _read_cache(path, key)
    if cached is not None:
        return cached

    suffix = path.suffix.lower()
    try:
        # Imports are local: components must remain importable in system Python and FreeCAD
        # must only ever run as a child process through its backend implementation.
        from phimech.backends.base import for_format, get

        if suffix == ".stl":
            backend = for_format("stl")
        elif suffix in (".step", ".stp"):
            backend = get("freecad")
            if not backend.available():
                raise RuntimeError(
                    "the FreeCAD backend is registered but its executable is unavailable"
                )
        else:
            raise RuntimeError("expected an STL (.stl) or STEP (.step/.stp) import")
        measurement = backend.measure(Import(str(path)))
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("expected an STL"):
            raise
        raise RuntimeError("cannot measure imported model {!s}: {}".format(path, exc)) from exc

    size = _measurement_size(measurement.to_dict())
    if size is None or any(value <= 0.0 for value in size):
        raise RuntimeError(
            "backend returned an invalid bounding box for imported model {!s}".format(path)
        )
    _write_cache(path, key, measurement)
    return size


def _edit_risk(path: Path, key: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """The cached assessment for this file version, computing and caching it on a miss.

    Runs *after* ``_measure`` so the kernel's bbox/volume/validity can be folded in, and is
    best-effort throughout: an assessment that could not be produced must never fail an import
    that measured fine, so the answer is then ``None`` rather than an exception.
    """
    cached = _read_cache_payload(path, key)
    if cached is not None and isinstance(cached.get("edit_risk"), dict):
        return cached["edit_risk"]
    try:
        from phimech.interop.edit_risk import assess

        measurement = cached.get("measurement") if cached is not None else None
        assessment = assess(path, measurement=measurement).to_dict()
    except Exception:
        return None
    if cached is not None:
        _write_cache(path, key, cached["measurement"], assessment)
    return assessment


def envelope_from_import(
    path: Any,
    ref: str,
    *,
    scale: float = 1.0,
    base_dir: Optional[Any] = None,
) -> Envelope:
    """Measure an external model and return an origin-relative import envelope.

    Measurement is cached by the source file's modification time and byte size.  The cached
    unscaled dimensions are multiplied locally, so using two scale factors does not invoke a
    CAD backend twice.
    """
    resolved = _resolved(path, base_dir)
    if not resolved.is_file():
        raise RuntimeError("imported model does not exist: {!s}".format(resolved))
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        raise ValueError("scale must be a positive number") from None
    if factor <= 0.0:
        raise ValueError("scale must be a positive number")

    key = _key_for(resolved)
    unscaled = _measure(resolved, key)
    size = tuple(value * factor for value in unscaled)
    body: Node = Import(str(resolved))
    if factor != 1.0:
        body = Scale((factor, factor, factor), [body])
    meta: Dict[str, Any] = {"path": str(resolved), "scale": factor}
    risk = _edit_risk(resolved, key)
    if risk is not None:
        meta["edit_risk"] = risk
        # Mirror pcb.py: a high/hostile verdict is a warning the report can show, not an error.
        # The envelope itself is unchanged - placement already treats the import as a fixed box.
        if risk.get("level") in ("high", "hostile", "unknown"):
            meta["warnings"] = [
                "{}: edit risk {} ({} faces, {} free edges) - {}".format(
                    ref, risk.get("level"), risk.get("faces"), risk.get("free_edges"),
                    (risk.get("guidance") or ["see edit_risk.guidance"])[0],
                )
            ]
    return Envelope(
        ref=ref,
        size=size,  # type: ignore[arg-type]
        body=body,
        name=resolved.stem,
        source="import",
        meta=meta,
    )


__all__ = ["envelope_from_import"]
