"""Keep export diagnostics and source identity attached to isolated-part geometry.

An export's existence is not evidence of successful translation. Validation separates
producer attestations from checks performed here, and never executes a model or edits CAD.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA = "phimech.solidworks-handoff.v1"
IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
MAX_STEP_BYTES = 64 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bytes(path: Path, limit: int) -> bytes:
    _require(not path.is_symlink() and path.is_file(), "expected regular file: {}".format(path))
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    _require(0 < len(data) <= limit, "empty or oversized file: {}".format(path))
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096


def _json(path: Path) -> Dict[str, Any]:
    def pairs(items: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    value = json.loads(_bytes(path, 1024 * 1024), object_pairs_hook=pairs)
    _require(isinstance(value, dict), "manifest must be an object")
    return value


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    _require(manifest.get("schema") == SCHEMA, "unsupported handoff schema")
    source = manifest.get("source")
    export = manifest.get("export")
    step = manifest.get("step")
    _require(all(isinstance(x, dict) for x in (source, export, step)),
             "source, export and step objects are required")
    for field in ("document", "configuration", "revision"):
        _require(_text(source.get(field)),
                 "source.{} is required (use unknown explicitly)".format(field))
    _require(source.get("document_type") == "part", "only an isolated part is supported")
    _require(source.get("active") is True and source.get("saved") is True,
             "producer must attest an active, saved part")
    _require(source.get("dirty") is False and source.get("selected_objects") == 0
             and type(source.get("selected_objects")) is int,
             "dirty or partially selected parts are refused")
    _require(_digest(source.get("sha256")), "source.sha256 is required")
    _require(manifest.get("unit") == "mm", "handoff unit must be mm")
    _require(manifest.get("transform") == IDENTITY,
             "only the default part coordinate system is supported")
    _require(step.get("file") == "model.step", "STEP must be the bundle-local model.step")
    _require(_digest(step.get("sha256")), "step.sha256 is required")
    mode = export.get("mode")
    _require(mode in ("offline_fixture", "solidworks_manual", "solidworks_api"),
             "unrecognized export mode")
    _require(export.get("completed") is True, "export did not complete")
    _require(export.get("translation_diagnostics_checked") is True,
             "translation notifications and .err diagnostics must be checked")
    _require(export.get("translation_errors") == [], "translation errors must be absent")
    _require(export.get("translation_warnings") == [], "translation warnings need human resolution")
    if mode == "solidworks_api":
        _require(export.get("save_return") is True, "SaveAs returned failure or was not captured")
        for field in ("error_flags", "warning_flags"):
            _require(type(export.get(field)) is int and export[field] == 0,
                     "{} must be captured and zero".format(field))
    else:
        flags = ("save_return", "error_flags", "warning_flags")
        _require(all(export.get(k) is None for k in flags),
                 "manual/fixture exports must not invent API return values")
    if mode != "offline_fixture":
        _require(source["document"].lower().endswith(".sldprt"),
                 "native source must identify a .sldprt")
        _require(export.get("host_platform") == "Windows", "native export requires a Windows host")
        for field in ("host_version", "solidworks_version", "operator"):
            _require(_text(export.get(field)), "export.{} is required".format(field))


def _source_manifest(capture: Dict[str, Any], protocol: str) -> Dict[str, Any]:
    source, export = capture["source"], capture["export"]
    native = export["mode"] != "offline_fixture"
    return {
        "schema_version": 1,
        "source": {
            "kind": "solidworks" if native else "step",
            "native_path": source["document"] if native else None,
            "native_sha256": source["sha256"] if native else None,
            "configuration": source["configuration"],
            "exported_by": export.get("operator"), "exported_at": export.get("exported_at"),
            "exporter": {"name": "SolidWorks" if native else "offline_fixture",
                         "version": export.get("solidworks_version")},
        },
        "step": {"path": "model.step", "sha256": capture["step"]["sha256"],
                 "protocol": "AP" + protocol, "units": "mm"},
        # Declared and left None until a producer captures them. None means UNSPECIFIED, never
        # "no tolerance" and never "as machined": a general tolerance or a finish this adapter
        # made up would be printed on a drawing someone manufactures to.
        "component": {"name": source["document"].replace("\\", "/").rsplit("/", 1)[-1],
                      "revision": None if source["revision"] == "unknown" else source["revision"],
                      "material": None, "mass_g": None,
                      "general_tolerance": None, "finish": None},
        "translation": {"ok": True, "diagnostics": []},
        "errors": [], "warnings": [],
    }


def validate_bundle(
    bundle_dir: Any, *, require_kernel: bool = True, source_path: Optional[Any] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Read a handoff without writes; default acceptance requires kernel validation.

    ``require_kernel=False`` is contract-only preflight, never geometry acceptance. Native
    source identity is a producer attestation unless its bytes are explicitly supplied.
    """
    root = Path(bundle_dir).absolute()
    _require(not root.is_symlink() and root.is_dir(), "bundle must be a real directory")
    manifest = _json(root / "manifest.json")
    _validate_manifest(manifest)
    path = root / "model.step"
    data = _bytes(path, MAX_STEP_BYTES)
    _require(_sha(data) == manifest["step"]["sha256"], "STEP hash mismatch")
    from phimech.interop.step import inspect, readiness

    facts = inspect(path)
    transfer = readiness(facts)
    _require(facts.parse_error is None and transfer["ready"] is True,
             "STEP transfer checks failed: {}".format(transfer["summary"]))
    _require(facts.solids == 1 and not facts.mesh_entities and not facts.looks_tessellated,
             "one non-tessellated solid is required")
    _require(facts.protocol in ("214", "242"), "adapter requires AP214 or AP242")
    source_manifest = _json(root / "source.json")
    _require(source_manifest == _source_manifest(manifest, facts.protocol),
             "source.json disagrees with capture provenance")
    verified_source = False
    if source_path is not None:
        verified_source = (
            _sha(_bytes(Path(source_path), MAX_STEP_BYTES)) == manifest["source"]["sha256"])
        _require(verified_source, "source hash mismatch")
    measurement = None
    if require_kernel:
        from phimech.backends.freecad import FreeCADBackend
        from phimech.csg.ast import Import

        measurement = FreeCADBackend().measure(
            Import(str(path.resolve())), timeout=timeout).to_dict()
        _require(measurement.get("extra", {}).get("valid") is True,
                 "kernel did not confirm valid geometry")
        _require(measurement.get("solids") == 1, "kernel must find exactly one solid")
        values = [measurement["volume_mm3"], measurement["area_mm2"], *measurement["bbox"]["size"]]
        _require(all(math.isfinite(v) and v > 0 for v in values),
                 "kernel dimensions, volume and area must be finite and positive")
    _require(_sha(_bytes(path, MAX_STEP_BYTES)) == manifest["step"]["sha256"],
             "STEP changed during validation")
    return {
        "schema": SCHEMA, "status": "validated" if require_kernel else "preflight_only",
        "step_path": str(path.resolve()), "manifest": source_manifest, "capture": manifest,
        "measurement": measurement,
        "source_hash_verified": verified_source,
        "native_integration_tested": False,
        "model_execution": "disabled", "manufacturing_release": False,
    }


def prepare_bundle(
    step_path: Any, metadata: Dict[str, Any], destination: Any, *,
    require_kernel: bool = True, source_path: Optional[Any] = None, timeout: float = 120.0,
) -> Dict[str, Any]:
    """Copy an export into a new directory, preserving originals and binding exact bytes."""
    data = _bytes(Path(step_path), MAX_STEP_BYTES)
    manifest = copy.deepcopy(metadata)
    manifest["schema"] = SCHEMA
    manifest["step"] = {"file": "model.step", "sha256": _sha(data)}
    _validate_manifest(manifest)
    from phimech.interop.step import inspect

    protocol = inspect(Path(step_path)).protocol
    _require(protocol in ("214", "242"), "adapter requires AP214 or AP242")
    root = Path(destination).absolute()
    root.mkdir(mode=0o700)  # exclusive: existing directories and symlinks are refused
    try:
        (root / "model.step").write_bytes(data)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (root / "source.json").write_text(
            json.dumps(_source_manifest(manifest, protocol), indent=2) + "\n", encoding="utf-8")
        result = validate_bundle(root, require_kernel=require_kernel, source_path=source_path,
                                 timeout=timeout)
        return result
    except BaseException:
        shutil.rmtree(root)
        raise


def sample_metadata(step_path: Any) -> Dict[str, Any]:
    """Label a supplied offline fixture honestly; never imply it came from SolidWorks."""
    path = Path(step_path)
    return {
        "source": {"document": path.name, "document_type": "part", "configuration": "fixture",
                   "revision": "unknown", "sha256": _sha(_bytes(path, MAX_STEP_BYTES)),
                   "active": True, "saved": True, "dirty": False, "selected_objects": 0},
        "unit": "mm", "transform": copy.deepcopy(IDENTITY),
        "export": {"mode": "offline_fixture", "completed": True, "save_return": None,
                   "error_flags": None, "warning_flags": None,
                   "translation_diagnostics_checked": True, "translation_errors": [],
                   "translation_warnings": []},
    }


def package_results(bundle_dir: Any, drawing_json: Any, destination: Any, *,
                    review_json: Optional[Any] = None) -> Dict[str, Any]:
    """Package existing hash-bound inspection artifacts; never run drawing or AI jobs."""
    validated = validate_bundle(bundle_dir)
    root = Path(bundle_dir)
    drawing_path = Path(drawing_json).absolute()
    drawing = _json(drawing_path)
    hashes = drawing.get("hashes", {})
    _require(isinstance(hashes, dict), "drawing hashes must be an object")
    _require(drawing.get("schema_version") == 1 and drawing.get("intent") == "inspection",
             "expected a schema_version 1 inspection drawing")
    _require(hashes.get("step_sha256") == validated["manifest"]["step"]["sha256"],
             "drawing is not bound to this STEP")
    entries = {name: _bytes(root / name, MAX_STEP_BYTES)
               for name in ("model.step", "source.json", "manifest.json")}
    entries["results/drawing.json"] = _bytes(drawing_path, 1024 * 1024)
    for field, suffix in (("path", "svg"), ("png", "png")):
        value = drawing.get(field)
        if value is None and suffix == "png":
            _require(hashes.get("png_sha256") is None, "missing PNG has a hash")
            continue
        _require(isinstance(value, str), "drawing artifact path is missing")
        path = Path(value)
        if not path.is_absolute():
            relative_to_json = drawing_path.parent / path
            path = (path.absolute() if path.parent != Path(".")
                    and path.absolute().parent.resolve() == drawing_path.parent.resolve()
                    else relative_to_json)
        _require(path.parent.resolve() == drawing_path.parent.resolve(),
                 "drawing artifacts must be beside drawing JSON")
        _require(path.suffix.lower() == "." + suffix, "unexpected artifact extension")
        data = _bytes(path, MAX_STEP_BYTES)
        _require(_sha(data) == hashes.get(suffix + "_sha256"), "drawing artifact hash mismatch")
        entries["results/drawing." + suffix] = data
    review_status = "unknown"
    if review_json is not None:
        review_path = Path(review_json)
        review = _json(review_path)
        _require(review.get("schema_version") == 1 and review.get("hashes") == hashes,
                 "review is not bound to these drawing artifacts")
        review_status = review.get("verdict")
        _require(review_status in ("pass", "needs_review", "fail", "unknown"),
                 "invalid review verdict")
        entries["results/drawing.review.json"] = _bytes(review_path, 1024 * 1024)
    receipt = {
        "schema_version": 1, "intent": "inspection", "manufacturing_release": False,
        "review_verdict": review_status, "review_executed_by_adapter": False,
        "model_execution": "disabled", "native_integration_tested": False,
        "files": {name: _sha(data) for name, data in entries.items()},
    }
    entries["receipt.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    _require(_sha(entries["model.step"]) == validated["manifest"]["step"]["sha256"],
             "STEP changed during packaging")
    _require(json.loads(entries["source.json"]) == validated["manifest"],
             "source manifest changed during packaging")
    _require(json.loads(entries["manifest.json"]) == validated["capture"],
             "capture changed during packaging")
    _require(json.loads(entries["results/drawing.json"]) == drawing,
             "drawing metadata changed during packaging")
    if review_json is not None:
        _require(json.loads(entries["results/drawing.review.json"]) == review,
                 "review changed during packaging")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return receipt
