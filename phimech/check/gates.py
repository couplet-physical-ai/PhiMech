"""Machine-checkable design-for-manufacture gates.

Each gate reduces a design to a numeric value, a numeric threshold, and a result whose
``status`` is one of five words (see ``STATUSES``). ``passed`` is kept as the legacy boolean
projection of that status so older readers (the viewer manifest, the benchmark grader) keep
working unchanged:

    pass -> True    fail / undetermined -> False    skipped / not_applicable -> None

The two words that are new deserve their reasons. ``not_applicable`` is a gate that had nothing
to judge -- no cutouts for `cutout_within_face`, no components for `placement_no_overlap` -- and
it used to be reported as a *pass*. A gate that ran nothing must never render as a pass, so it
now says so. ``undetermined`` is a gate whose evidence never arrived because a layer below it
failed -- a backend that timed out, a crashed gate -- and it is never a pass either; unlike
``skipped`` (the question was not asked yet, as in `synth check` before any render) it *blocks*
`GateReport.passed`, because the question was asked and could not be answered.

The code is deliberately defensive because ``synth check`` can call it before a renderer has
created measurements, and because MCP callers may provide lightweight placement/enclosure
objects rather than the full synth dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from phimech.backends.base import Measurement
from phimech.csg.ops import BOOLEAN_VOLUME_TOLERANCE

_MISSING = object()
_DEFAULT_MIN_WALL = 1.2
_DEFAULT_MIN_FEATURE = 0.8
_DEFAULT_NOZZLE = 0.4
_DEFAULT_MAX_OVERHANG = 45.0
_MIN_TEXT_SIZE = 4.0
_MAX_OVERHANG_FRACTION = 0.35
#: How far a synthesised standoff boss may sit from the mounting hole it claims to serve, mm.
#: Tight on purpose: the boss centre and the hole centre are produced by the same
#: `features.hole_to_world()` transform, so anything above floating-point noise means the boss
#: is not actually under that hole.
_HOLE_MATCH_TOL = 0.05
#: Interpenetration below this is floating-point noise from box arithmetic, not interference.
#: A post standing exactly on a keep-out boundary computes an overlap of ~1e-16 mm, which is a
#: hundred million times smaller than a printer can resolve. Anything a nozzle could express is
#: many orders of magnitude above it, so this cannot hide a real collision.
_TOUCH_EPS = 1e-6
_PLA_DENSITY = 1.24  # g/cm^3
# `boolean_volume_sane` shares `BOOLEAN_VOLUME_TOLERANCE` (0.5 %) with `csg.ops.safe_cut`, so
# the gate and the op-level guard can never disagree about what "impossible" means.

#: The status vocabulary, in the order a reader should worry about them.
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIPPED = "skipped"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_UNDETERMINED = "undetermined"
STATUSES: Tuple[str, ...] = (
    STATUS_PASS, STATUS_FAIL, STATUS_SKIPPED, STATUS_NOT_APPLICABLE, STATUS_UNDETERMINED
)
#: Message prefixes that mark a result as undetermined when no explicit status was recorded.
#: `gate crashed:` is the in-process analogue of a timeout: the gate did not finish judging.
_UNDETERMINED_PREFIXES = ("undetermined:", "gate crashed:")
_NOT_APPLICABLE_PREFIXES = ("not applicable",)


def derive_status(passed: Optional[bool], message: str) -> str:
    """The status implied by a legacy `(passed, message)` pair, for results that predate it.

    Machine gates build their results through this module's helpers and so get
    an explicit status; anything constructed positionally by hand still lands on the right word
    because the prefixes below are the conventions those modules already use.
    """
    text = str(message or "").strip().lower()
    if passed is True:
        return STATUS_PASS
    if passed is False:
        if text.startswith(_UNDETERMINED_PREFIXES):
            return STATUS_UNDETERMINED
        return STATUS_FAIL
    if text.startswith(_UNDETERMINED_PREFIXES):
        return STATUS_UNDETERMINED
    if text.startswith(_NOT_APPLICABLE_PREFIXES):
        return STATUS_NOT_APPLICABLE
    return STATUS_SKIPPED


def passed_for_status(status: str) -> Optional[bool]:
    """The legacy boolean projection of a status; the one place that mapping is written."""
    if status == STATUS_PASS:
        return True
    if status in (STATUS_FAIL, STATUS_UNDETERMINED):
        return False
    return None


@dataclass(frozen=True)
class GateResult:
    """One numeric gate decision.

    `status` is the authoritative word; `passed` is its boolean projection (see the module
    docstring). When `status` is omitted it is derived from `passed` and the message, so
    positional construction by older callers keeps working. `layer` names the pipeline stage
    whose failure left an `undetermined` result (`render`, `measure`, ...), or is None.
    """

    name: str
    passed: Optional[bool]
    value: Any
    threshold: Any
    message: str
    severity: str = "error"
    status: Optional[str] = None
    layer: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is None:
            return
        if self.status not in STATUSES:
            raise ValueError("unknown gate status {!r}; choose {}".format(self.status, STATUSES))
        if passed_for_status(self.status) is not self.passed:
            raise ValueError(
                "gate {!r}: status {!r} contradicts passed={!r}".format(
                    self.name, self.status, self.passed
                )
            )

    @property
    def effective_status(self) -> str:
        return self.status if self.status is not None else derive_status(self.passed, self.message)

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "value": _json_value(self.value),
            "threshold": _json_value(self.threshold),
            "message": self.message,
            "severity": self.severity,
            "status": self.effective_status,
        }
        if self.layer is not None:
            record["layer"] = self.layer
        return record


def result_status(result: Any) -> str:
    """Status of a live `GateResult` OR a dict re-read from `report.json` (which may predate
    the field); the report writer and the summary both go through here."""
    explicit = getattr(result, "effective_status", None)
    if isinstance(explicit, str):
        return explicit
    status = _field(result, "status", None)
    if isinstance(status, str) and status in STATUSES:
        return status
    return derive_status(_field(result, "passed", None), str(_field(result, "message", "") or ""))


def summarize(results: Sequence[Any], *, default_layer: str = "gates") -> Dict[str, Any]:
    """The report-level verdict over a list of results (live objects or report dicts).

    * `is_valid` -- no error-severity gate is `fail` or `undetermined`. Warn-severity findings
      are visible but do not decide validity, exactly as they never decided `passed`.
    * `first_failure` -- the first error-severity `fail` in report order, so it points at what
      blocks; when nothing blocks, the first advisory `fail`; else None.
    * `undetermined_layer` -- the layer recorded on the first `undetermined` result (`render`,
      `measure`), or `default_layer` when a gate itself crashed; None when nothing is undetermined.
    * `counts` -- one integer per status word plus `total`; they always add up to `total`.
    """
    counts = {status: 0 for status in STATUSES}
    first_blocking: Optional[str] = None
    first_any: Optional[str] = None
    undetermined_layer: Optional[str] = None
    first_undetermined: Optional[str] = None
    invalid = False
    for result in results:
        status = result_status(result)
        severity = str(_field(result, "severity", "error") or "error")
        name = str(_field(result, "name", "?"))
        counts[status] = counts.get(status, 0) + 1
        if status == STATUS_FAIL:
            if first_any is None:
                first_any = name
            if severity == "error":
                invalid = True
                if first_blocking is None:
                    first_blocking = name
        elif status == STATUS_UNDETERMINED:
            if severity == "error":
                invalid = True
            if first_undetermined is None:
                first_undetermined = name
                layer = _field(result, "layer", None)
                undetermined_layer = str(layer) if layer else default_layer
    counts["total"] = len(list(results))
    return {
        "is_valid": not invalid,
        "first_failure": first_blocking if first_blocking is not None else first_any,
        "first_undetermined": first_undetermined,
        "undetermined_layer": undetermined_layer,
        "counts": counts,
    }


@dataclass
class GateReport:
    """The complete gate result, with warnings excluded from the release decision."""

    results: List[GateResult]

    @property
    def passed(self) -> bool:
        return all(
            result.passed is not False
            for result in self.results
            if result.severity == "error"
        )

    def summary(self) -> Dict[str, Any]:
        """`{is_valid, first_failure, first_undetermined, undetermined_layer, counts}`."""
        return summarize(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary(),
        }

    def summary_line(self) -> str:
        counts = self.summary()["counts"]
        status = "PASS" if self.passed else "FAIL"
        line = "{}: {} gates, {} failed, {} skipped".format(
            status, len(self.results), counts[STATUS_FAIL], counts[STATUS_SKIPPED]
        )
        if counts[STATUS_NOT_APPLICABLE]:
            line += ", {} not applicable".format(counts[STATUS_NOT_APPLICABLE])
        if counts[STATUS_UNDETERMINED]:
            line += ", {} undetermined".format(counts[STATUS_UNDETERMINED])
        return line


def _json_value(value: Any) -> Any:
    """Convert common geometry values to strict JSON-shaped values for reports."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or a mapping."""
    attribute = getattr(value, name, _MISSING)
    if attribute is not _MISSING:
        return attribute
    if isinstance(value, Mapping):
        return value.get(name, default)
    return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("{} is not numeric: {!r}".format(name, value)) from None


def _manufacturing(design: Any) -> Mapping[str, Any]:
    if not isinstance(design, Mapping):
        raise ValueError("design is not a mapping")
    manufacturing = design.get("manufacturing", {})
    if manufacturing is None:
        return {}
    if not isinstance(manufacturing, Mapping):
        raise ValueError("manufacturing is not a mapping")
    return manufacturing


def _mfg_number(design: Any, key: str, default: float) -> float:
    return _number(_manufacturing(design).get(key, default), "manufacturing." + key)


def _enclosure_spec(design: Any) -> Mapping[str, Any]:
    if not isinstance(design, Mapping):
        raise ValueError("design is not a mapping")
    enclosure = design.get("enclosure", {})
    if enclosure is None:
        return {}
    if not isinstance(enclosure, Mapping):
        raise ValueError("enclosure is not a mapping")
    return enclosure


def _part_summary(parts: Any) -> Mapping[str, Any]:
    summary = getattr(parts, "summary", _MISSING)
    if summary is not _MISSING:
        summary = summary() if callable(summary) else summary
        return summary if isinstance(summary, Mapping) else {}
    if isinstance(parts, Mapping):
        value = parts.get("summary", {})
        if callable(value):
            value = value()
        return value if isinstance(value, Mapping) else {}
    return {}


def _part_value(parts: Any, summary: Mapping[str, Any], spec: Mapping[str, Any], key: str) -> Any:
    """Prefer synthesized numbers, then its summary, then the IR specification."""
    value = getattr(parts, key, _MISSING)
    if value is not _MISSING and value is not None:
        return value
    value = summary.get(key, _MISSING)
    if value is not _MISSING and value is not None:
        return value
    value = spec.get(key, _MISSING)
    return None if value is _MISSING else value


def _result(
    name: str,
    passed: Optional[bool],
    value: Any,
    threshold: Any,
    message: str,
    severity: str = "error",
) -> GateResult:
    return GateResult(name, passed, value, threshold, message, severity)


def _skipped(name: str, reason: str, threshold: Any = 0.0, severity: str = "error") -> GateResult:
    """The question was not asked yet (no render, no printer_bed declared): never a pass, and it
    does not block, because nothing was claimed."""
    return GateResult(name, None, None, threshold, reason, severity, status=STATUS_SKIPPED)


def _not_applicable(
    name: str, reason: str, threshold: Any = 0.0, severity: str = "error"
) -> GateResult:
    """The gate had nothing to judge. Reported in its own words so a design with no cutouts
    cannot collect a `cutout_within_face` pass for a check that ran nothing."""
    message = reason if reason.lower().startswith("not applicable") else "not applicable: " + reason
    return GateResult(name, None, None, threshold, message, severity, status=STATUS_NOT_APPLICABLE)


def _undetermined(
    name: str, reason: str, threshold: Any = 0.0, severity: str = "error", layer: Optional[str] = None
) -> GateResult:
    """The question was asked and could not be answered (timeout, crash). Blocks like a
    failure -- a render that never finished is not evidence of anything."""
    message = reason
    if not reason.lower().startswith(_UNDETERMINED_PREFIXES):
        message = "undetermined: " + reason
    return GateResult(
        name, False, None, threshold, message, severity, status=STATUS_UNDETERMINED, layer=layer
    )


def as_undetermined(result: GateResult, reason: str, layer: Optional[str]) -> GateResult:
    """Re-state a result as undetermined, keeping its name, threshold and severity.

    Used by the pipeline when a gate reported `skipped: no render` *after* a render was attempted
    and failed: the gate itself cannot tell "not asked" from "asked, backend died", the pipeline
    can.
    """
    return _undetermined(result.name, reason, result.threshold, result.severity, layer)


def _gate_ir_valid(design: Any) -> GateResult:
    """Validation error count, validated HERE when the caller has not already done it.

    This gate used to read `design["_errors"]` and nothing else. Nothing in the repository ever
    wrote that key - it was read in exactly one place and assigned in none - so the gate reported
    `errors=0 threshold=0 PASS` unconditionally, for every design ever checked. A gate whose value
    is structurally always zero is worse than a missing gate, because `PASS ir_valid` reads as
    positive evidence that the IR was validated.

    It was caught by building a design with a deliberately invalid `source.kind`: the pipeline
    surfaced two validation errors from its own `validate` stage and exited 1, while this gate sat
    in the same report saying `PASS errors=0`. Two code paths in one report disagreeing about
    whether the design was valid.

    The pre-populated key is still honoured, so a caller that has already validated does not pay
    for it twice; the difference is that its absence now means "validate now", not "assume clean".
    """
    if not isinstance(design, Mapping):
        return _result("ir_valid", False, 1, 0, "errors=1 threshold=0 (design is not a mapping)")
    errors = design.get("_errors")
    source = "design._errors"
    if errors is None:
        source = "validated in-gate"
        try:
            from phimech.ir import ir as _ir  # local: keeps module import cheap and cycle-free

            errors = list(_ir.validate(design))
        except Exception as exc:  # a validator that cannot run must not silently pass the gate
            return _result(
                "ir_valid", False, 1, 0,
                "errors=1 threshold=0 (could not validate: {}: {})".format(
                    type(exc).__name__, exc
                ),
            )
    count = len(errors)
    detail = "" if count == 0 else " first={}".format(str(errors[0])[:120])
    return _result(
        "ir_valid", count == 0, count, 0,
        "errors={} threshold=0 ({}){}".format(count, source, detail),
    )


def _placed_count(placement: Any) -> Optional[int]:
    """How many components were placed, or None when the placement carries no item list."""
    items = _field(placement, "items", None)
    if items is None:
        return None
    try:
        return len(list(items() if callable(items) else items))
    except TypeError:
        return None


def _gate_placement_no_overlap(placement: Any) -> GateResult:
    if placement is None or _placed_count(placement) == 0:
        # A freeform part with no components has nothing to overlap; saying "pass" here would
        # let a sketch-lane design collect component evidence it never earned.
        return _not_applicable("placement_no_overlap", "no placed components", 0)
    errors = getattr(placement, "errors", [])
    errors = errors or []
    count = len(errors)
    return _result(
        "placement_no_overlap",
        count == 0,
        count,
        0,
        "overlap_errors={} threshold=0".format(count),
    )


def _bbox_pair(value: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        minimum = value.get("min")
        maximum = value.get("max")
    else:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("bbox must contain min and max")
        minimum, maximum = value
    if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
        raise ValueError("bbox bounds must be vectors")
    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("bbox bounds must have three coordinates")
    return (
        (float(minimum[0]), float(minimum[1]), float(minimum[2])),
        (float(maximum[0]), float(maximum[1]), float(maximum[2])),
    )


def _world_bbox(value: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    method = getattr(value, "world_bbox", _MISSING)
    if method is not _MISSING:
        return _bbox_pair(method)
    value_bbox = getattr(value, "bbox", _MISSING)
    if value_bbox is not _MISSING:
        return _bbox_pair(value_bbox)
    if isinstance(value, Mapping):
        return _bbox_pair(value.get("world_bbox", value.get("bbox")))
    raise ValueError("placed item has no bbox")


def _placement_boxes(placement: Any) -> List[Any]:
    keepouts = getattr(placement, "keepouts", _MISSING)
    if keepouts is not _MISSING:
        if callable(keepouts):
            keepouts = keepouts()
        if keepouts:
            return list(keepouts)
    by_ref = getattr(placement, "keepout_by_ref", _MISSING)
    if by_ref is not _MISSING and by_ref:
        return list(by_ref.values())
    items = getattr(placement, "items", []) or []
    return list(items)


def _world_offset(parts: Any) -> Tuple[float, float, float]:
    """The shift `synthesize()` applied to put the enclosure floor at Z = 0.

    `EnclosureParts.inner_cavity` is in **enclosure** coordinates while `Placed.world_bbox()`
    is in **placement** coordinates; the two differ by this offset. Comparing them without it
    reported a -3 mm cavity margin on a design whose real margins were all >= 1.5 mm.
    """
    value = _field(parts, "world_offset", None)
    if value is None:
        value = _part_summary(parts).get("world_offset")
    if value is None:
        return (0.0, 0.0, 0.0)
    try:
        seq = [float(v) for v in value]
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    if len(seq) != 3:
        return (0.0, 0.0, 0.0)
    return (seq[0], seq[1], seq[2])


def _gate_clearance_positive(placement: Any, parts: Any) -> GateResult:
    cavity = getattr(parts, "inner_cavity", _MISSING)
    if cavity is _MISSING or cavity is None:
        raise ValueError("enclosure_parts.inner_cavity is missing")
    cavity_min, cavity_max = _bbox_pair(cavity)
    boxes = _placement_boxes(placement)
    if not boxes:
        return _not_applicable("clearance_positive", "no placed components", 0.0)
    offset = _world_offset(parts)
    margin = None
    for item in boxes:
        raw_min, raw_max = _world_bbox(item)
        box_min = tuple(raw_min[axis] + offset[axis] for axis in range(3))
        box_max = tuple(raw_max[axis] + offset[axis] for axis in range(3))
        item_margin = min(
            box_min[axis] - cavity_min[axis]
            for axis in range(3)
        )
        item_margin = min(
            item_margin,
            *(cavity_max[axis] - box_max[axis] for axis in range(3))
        )
        margin = item_margin if margin is None else min(margin, item_margin)
    assert margin is not None
    return _result(
        "clearance_positive",
        margin >= 0.0,
        margin,
        0.0,
        "minimum_cavity_margin={:.6g} threshold=0".format(margin),
    )


def _gate_wall_thickness(design: Any, parts: Any) -> GateResult:
    spec = _enclosure_spec(design)
    summary = _part_summary(parts)
    wall_value = _part_value(parts, summary, spec, "wall")
    if wall_value is None:
        raise ValueError("enclosure wall is missing")
    wall = _number(wall_value, "enclosure.wall")
    minimum = _mfg_number(design, "min_wall", _DEFAULT_MIN_WALL)
    nozzle = _mfg_number(design, "nozzle", _DEFAULT_NOZZLE)
    nozzle_delta = 0.0
    if nozzle > 0.0:
        nozzle_delta = abs(wall - round(wall / nozzle) * nozzle)
    if wall < minimum:
        return _result(
            "wall_thickness",
            False,
            wall,
            minimum,
            "wall={:.6g} min_wall={:.6g}".format(wall, minimum),
            "error",
        )
    if nozzle > 0.0 and nozzle_delta > 0.05:
        return _result(
            "wall_thickness",
            False,
            wall,
            minimum,
            "wall={:.6g} nozzle_multiple_delta={:.6g} threshold=0.05".format(
                wall, nozzle_delta
            ),
            "warn",
        )
    return _result(
        "wall_thickness",
        True,
        wall,
        minimum,
        "wall={:.6g} min_wall={:.6g} nozzle_delta={:.6g}".format(wall, minimum, nozzle_delta),
    )


def _gate_floor_thickness(design: Any, parts: Any) -> GateResult:
    spec = _enclosure_spec(design)
    summary = _part_summary(parts)
    wall_value = _part_value(parts, summary, spec, "wall")
    floor_value = _part_value(parts, summary, spec, "floor")
    if floor_value is None:
        floor_value = wall_value
    if floor_value is None:
        raise ValueError("enclosure floor is missing")
    floor = _number(floor_value, "enclosure.floor")
    minimum = _mfg_number(design, "min_wall", _DEFAULT_MIN_WALL)
    return _result(
        "floor_thickness",
        floor >= minimum,
        floor,
        minimum,
        "floor={:.6g} min_wall={:.6g}".format(floor, minimum),
    )


def _measurement_bbox_size(measurement: Any) -> Tuple[float, float, float]:
    bbox_value = getattr(measurement, "bbox", _MISSING)
    if bbox_value is _MISSING:
        raise ValueError("measurement has no bbox")
    size = getattr(bbox_value, "size", _MISSING)
    if size is _MISSING:
        minimum, maximum = _bbox_pair(bbox_value)
        size = tuple(maximum[index] - minimum[index] for index in range(3))
    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ValueError("measurement bbox size is not a vec3")
    return (float(size[0]), float(size[1]), float(size[2]))


def _measurement_items(measurements: Any) -> List[Tuple[str, Any]]:
    if measurements is None:
        return []
    if not isinstance(measurements, Mapping):
        raise ValueError("measurements is not a mapping")
    return list(measurements.items())


def _gate_fits_bed(design: Any, measurements: Any) -> GateResult:
    bed_value = _manufacturing(design).get("printer_bed")
    if bed_value is None:
        return _skipped("fits_bed", "skipped: no printer_bed")
    if not isinstance(bed_value, (list, tuple)) or len(bed_value) != 3:
        raise ValueError("manufacturing.printer_bed must be a vec3")
    bed = tuple(float(value) for value in bed_value)
    entries = _measurement_items(measurements)
    if not entries:
        return _skipped("fits_bed", "skipped: no render")
    margin = min(
        bed[axis] - _measurement_bbox_size(measurement)[axis]
        for _, measurement in entries
        for axis in range(3)
    )
    return _result(
        "fits_bed",
        margin >= 0.0,
        margin,
        0.0,
        "minimum_bed_margin={:.6g} threshold=0".format(margin),
    )


def _gate_part_volume_positive(measurements: Any) -> GateResult:
    entries = _measurement_items(measurements)
    if not entries:
        return _skipped("part_volume_positive", "skipped: no render")
    volumes = [float(getattr(measurement, "volume")) for _, measurement in entries]
    minimum = min(volumes)
    return _result(
        "part_volume_positive",
        minimum > 0.0,
        minimum,
        0.0,
        "minimum_volume_mm3={:.6g} threshold>0".format(minimum),
    )


def _measurement_stl(measurement: Any) -> Optional[Any]:
    """The STL path recorded alongside a measurement, from an object OR a plain dict.

    `getattr` alone only works for the live Measurement dataclass; a measurement re-read from
    `report.json` is a dict, so every mesh-backed gate (manifold, overhang, and the optional machine
    CG/tape gates) silently SKIPPED when driven from a saved report. `_field` handles both.
    """
    extra = _field(measurement, "extra", None) or {}
    if not isinstance(extra, Mapping):
        return None
    return extra.get("stl")


def _gate_manifold(measurements: Any) -> GateResult:
    entries = _measurement_items(measurements)
    if not entries:
        return _skipped("manifold", "skipped: no render", 1)
    states: List[int] = []
    details: List[str] = []
    for name, measurement in entries:
        solids = int(getattr(measurement, "solids"))
        state = 1 if solids == 1 else 0
        stl = _measurement_stl(measurement)
        if stl is not None:
            from phimech.check.mesh import count_bodies, is_watertight, read_stl

            tris = read_stl(str(stl))
            watertight, open_edges = is_watertight(tris)
            # A kernel can report one solid for a compound of disjoint bodies (an open_frame's
            # floating wall-mount tabs measured solids=1); the exported mesh cannot hide them.
            bodies = count_bodies(tris)
            state = 1 if state and watertight and bodies == 1 else 0
            details.append("{}:solids={},bodies={},open_edges={}".format(
                name, solids, bodies, open_edges))
        else:
            details.append("{}:solids={}".format(name, solids))
        states.append(state)
    value = min(states)
    return _result(
        "manifold",
        value == 1,
        value,
        1,
        "; ".join(details) + " threshold=1",
    )


#: Gates that read `measurements`; the pipeline re-states these as `undetermined` when a render
#: was attempted and did not produce a measurement for every printable part.
MEASUREMENT_GATES: Tuple[str, ...] = (
    "fits_bed", "part_volume_positive", "manifold", "boolean_volume_sane", "overhang",
    "material_estimate", "cutout_through",
)


def _boolean_reference(measurement: Any) -> Optional[Mapping[str, Any]]:
    """The `extra["boolean_reference"]` block the pipeline's measure stage records per part."""
    extra = getattr(measurement, "extra", None)
    if extra is None and isinstance(measurement, Mapping):
        extra = measurement.get("extra")
    if not isinstance(extra, Mapping):
        return None
    reference = extra.get("boolean_reference")
    return reference if isinstance(reference, Mapping) else None


def _boolean_ratio(kind: str, result_volume: float, reference_volume: float) -> float:
    """One "> 1 is impossible" number for either boolean, so one threshold serves both:
    result/base for a cut, child/result for a union."""
    numerator, denominator = result_volume, reference_volume
    if kind == "union":
        numerator, denominator = reference_volume, result_volume
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _gate_boolean_volume_sane(measurements: Any) -> GateResult:
    """A cut may not grow a part; a union may not shrink below its largest child.

    B-rep booleans fail *silently*: OCC's `cut` on a bad tool can hand back the base unchanged
    or with the tool fused on, and every downstream gate then measures a perfectly manifold,
    perfectly wrong solid. The pipeline's measure stage walks each part's boolean spine and
    records `extra["boolean_reference"]["checks"]`: for every level the result volume and the
    reference volume (the base of a `difference`, the first child of a `union`), measured by the
    same backend as the part. This gate turns each check into a ratio and takes the worst.

    Both numbers of a check come from the *same kernel* deliberately. Comparing an exact B-rep
    reference against the tessellated STL would report the tessellation deficit of every
    cylinder as a boolean defect. `BOOLEAN_VOLUME_TOLERANCE` (0.5 %) covers the noise between
    two renders of curved faces; a real failed cut is tens of percent.
    """
    limit = 1.0 + BOOLEAN_VOLUME_TOLERANCE
    entries = _measurement_items(measurements)
    if not entries:
        return _skipped("boolean_volume_sane", "skipped: no render", limit)
    worst: Optional[float] = None
    ok = True
    details: List[str] = []
    for name, measurement in entries:
        reference = _boolean_reference(measurement)
        if reference is None:
            continue
        checks = reference.get("checks")
        checks = list(checks) if isinstance(checks, (list, tuple)) else []
        if reference.get("error") or not checks:
            details.append("{}: reference unmeasured ({})".format(
                name, str(reference.get("error") or "no checks recorded")[:80]))
            continue
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            kind = str(check.get("kind") or "?")
            result_volume = float(check.get("result_volume_mm3") or 0.0)
            reference_volume = float(check.get("reference_volume_mm3") or 0.0)
            ratio = _boolean_ratio(kind, result_volume, reference_volume)
            worst = ratio if worst is None else max(worst, ratio)
            if ratio > limit:
                ok = False
            details.append("{}:{} {} result={:.6g} reference={:.6g} ratio={:.4f}".format(
                name, str(check.get("node") or "part"), kind, result_volume, reference_volume,
                ratio))
    if worst is None:
        return _not_applicable(
            "boolean_volume_sane",
            "no part has a top-level boolean with a measured reference"
            + (" (" + "; ".join(details) + ")" if details else ""),
            limit,
        )
    return _result(
        "boolean_volume_sane", ok, worst, limit,
        "; ".join(details) + " threshold={:.3f}".format(limit),
    )


def _standoff_entries(parts: Any) -> List[Any]:
    """The per-standoff dicts, whatever the producer called the field.

    `EnclosureParts.summary()['standoffs']` is a **count**; the geometry lives in
    `standoff_positions` (`at`, `od`, `hole_d`, `fastener`, ...). Preferring the count made
    this gate crash on every real design, so list-valued names are tried first and a bare
    number means "no per-standoff detail available".
    """
    summary = _part_summary(parts)
    entries = None
    for name in ("standoff_positions", "standoffs", "standoff_entries"):
        candidate = _field(parts, name, None)
        if candidate is None:
            candidate = summary.get(name)
        if isinstance(candidate, (list, tuple, Mapping)):
            entries = candidate
            break
    if entries is None:
        return []
    if isinstance(entries, Mapping):
        if any(key in entries for key in ("od", "outer_d", "outer_diameter", "hole", "hole_d")):
            return [entries]
        return list(entries.values())
    if isinstance(entries, (list, tuple)):
        return list(entries)
    return [entries]


def _standoff_dimension(entry: Any, names: Sequence[str]) -> Optional[float]:
    for name in names:
        value = _field(entry, name, _MISSING)
        if value is not _MISSING and value is not None:
            return _number(value, name)
    return None


def _gate_min_feature(design: Any, parts: Any) -> GateResult:
    entries = _standoff_entries(parts)
    minimum = _mfg_number(design, "min_feature", _DEFAULT_MIN_FEATURE)
    if not entries:
        return _not_applicable("min_feature", "no standoffs", minimum)
    features: List[float] = []
    for entry in entries:
        outer = _standoff_dimension(entry, ("od", "outer_d", "outer_diameter", "diameter"))
        hole = _standoff_dimension(entry, ("hole", "hole_d", "hole_diameter", "id", "inner_d"))
        if outer is None or hole is None:
            raise ValueError("standoff entry lacks od and hole")
        features.append((outer - hole) / 2.0)
    value = min(features)
    return _result(
        "min_feature",
        value >= minimum,
        value,
        minimum,
        "minimum_standoff_wall={:.6g} min_feature={:.6g}".format(value, minimum),
    )


def _screw_count(design: Any, parts: Any) -> Tuple[Optional[float], bool]:
    spec = _enclosure_spec(design)
    if not spec:
        return None, False
    lid = spec.get("lid", {})
    if not isinstance(lid, Mapping):
        lid = {}
    fastening = lid.get("fastening", "screws")
    if fastening != "screws":
        return 0.0, False
    summary = _part_summary(parts)
    candidates: List[Any] = []
    for source in (parts, summary):
        for key in ("screw_count", "lid_screw_count", "lid_screws", "screws"):
            value = _field(source, key, _MISSING)
            if value is not _MISSING and value is not None:
                candidates.append(value)
        lid_value = _field(source, "lid", _MISSING)
        if isinstance(lid_value, Mapping):
            for key in ("screw_count", "lid_screws", "screws"):
                if key in lid_value:
                    candidates.append(lid_value[key])
    value = candidates[0] if candidates else lid.get("screw_count", 4)
    if isinstance(value, (list, tuple, dict)):
        value = len(value)
    return _number(value, "lid screw count"), True


def _gate_screw_count_sane(design: Any, parts: Any) -> GateResult:
    count, applicable = _screw_count(design, parts)
    if not applicable:
        return _not_applicable("screw_count_sane", "lid is not screw-fastened", 2)
    assert count is not None
    if count < 2.0:
        return _result(
            "screw_count_sane",
            False,
            count,
            2,
            "lid_screws={:.6g} minimum=2".format(count),
            "error",

        )
    if count > 8.0:
        return _result(
            "screw_count_sane", False, count, 8, "lid_screws={:.6g} maximum=8".format(count), "warn"
        )
    return _result("screw_count_sane", True, count, 2, "lid_screws={:.6g} range=2..8".format(count))


def _rect(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if all(key in value for key in ("min", "max")):
            return _rect((value["min"][0], value["min"][1], value["max"][0], value["max"][1]))
        if all(key in value for key in ("x", "y")):
            width = value.get("width", value.get("w"))
            height = value.get("height", value.get("h"))
            if width is not None and height is not None:
                return _rect(
                    (
                        value["x"],
                        value["y"],
                        float(value["x"]) + float(width),
                        float(value["y"]) + float(height),
                    )
                )
        if "at" in value and "size" in value:
            at = value["at"]
            size = value["size"]
            return (
                float(at[0]) - float(size[0]) / 2.0,
                float(at[1]) - float(size[1]) / 2.0,
                float(at[0]) + float(size[0]) / 2.0,
                float(at[1]) + float(size[1]) / 2.0,
            )
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = tuple(float(item) for item in value)
    return (values[0], values[1], values[2], values[3])


def _cutout_rect(cutout: Any) -> Optional[Tuple[float, float, float, float]]:
    value = _field(cutout, "rect", None)
    if value is not None:
        return _rect(value)
    direct = _rect(cutout)
    if direct is not None:
        return direct
    return _cutout_rect_from_center(cutout)


def _cutout_rect_from_center(cutout: Any) -> Optional[Tuple[float, float, float, float]]:
    """Project a `{face, center (3D world), size (2D)}` cutout onto its face's two axes.

    This is the shape `synth/enclosure.py` actually emits. `_rect()` only understands 2D
    `at`+`size`, so without this projection every design with a cutout crashed the gate. The
    tolerance is included because the cut is made oversized by it on each side.
    """
    face = _field(cutout, "face", None)
    center = _field(cutout, "center", None)
    size = _field(cutout, "size", None)
    if not isinstance(face, str) or center is None or size is None:
        return None
    axes = _FACE_AXES.get(face.lower())
    if axes is None:
        return None
    try:
        c = [float(v) for v in center]
        w, h = float(size[0]), float(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if len(c) < 3:
        return None
    try:
        tol = float(_field(cutout, "tolerance", 0.0) or 0.0)
    except (TypeError, ValueError):
        tol = 0.0
    a, b = axes
    half_u = w / 2.0 + tol
    half_v = h / 2.0 + tol
    return (c[a] - half_u, c[b] - half_v, c[a] + half_u, c[b] + half_v)


def _face_rect(
    cutout: Any, parts: Any, summary: Mapping[str, Any]
) -> Optional[Tuple[float, float, float, float]]:
    for key in ("face_rect", "face_bounds", "bounds"):
        value = _field(cutout, key, None)
        result = _rect(value)
        if result is not None:
            return result
    face = _field(cutout, "face", None)
    direct_face = _rect(face)
    if direct_face is not None:
        return direct_face
    for source in (parts, summary):
        for key in ("face_rects", "face_bounds", "faces"):
            mapping = _field(source, key, None)
            if (
                isinstance(mapping, Mapping)
                and isinstance(face, (str, int, float))
                and face in mapping
            ):
                value = mapping[face]
                result = _rect(value)
                if result is None and isinstance(value, Mapping):
                    result = _rect(value.get("rect"))
                if result is not None:
                    return result
    if isinstance(face, str):
        derived = _face_rect_from_bbox(face, parts, summary)
        if derived is not None:
            return derived
    return None


# Which world axes span each face of an axis-aligned box, matching the (u, v) convention in
# phimech/synth/features.py `face_frame()`. Only the *extent* matters here, so the sign of
# u/v is irrelevant and the rect is always stored min-first.
_FACE_AXES = {
    "top": (0, 1),
    "bottom": (0, 1),
    "front": (0, 2),
    "back": (0, 2),
    "left": (1, 2),
    "right": (1, 2),
}


def _face_rect_from_bbox(
    face: str, parts: Any, summary: Mapping[str, Any]
) -> Optional[Tuple[float, float, float, float]]:
    """Derive a face rect from the enclosure's outer bbox.

    Producers emit cutouts as `{face, center, size}` and never a `face_rects` table, so
    without this the gate crashed on every design that had a cutout. The face rect is the
    outer box's extent on the two axes that span `face`.
    """
    box = _field(parts, "outer_bbox", None)
    if box is None:
        box = summary.get("outer_bbox")
    if box is None:
        return None
    try:
        box_min, box_max = _bbox_pair(box)
    except (ValueError, TypeError):
        return None
    axes = _FACE_AXES.get(face.lower())
    if axes is None:
        return None
    a, b = axes
    return (box_min[a], box_min[b], box_max[a], box_max[b])


def _cutout_list(parts: Any) -> List[Any]:
    """The per-cutout records, from live `EnclosureParts` or a summary-shaped mapping."""
    cutouts = getattr(parts, "cutouts", _MISSING)
    if cutouts is _MISSING:
        cutouts = _part_summary(parts).get("cutouts", [])
    if cutouts is None:
        cutouts = []
    if isinstance(cutouts, Mapping):
        cutouts = [cutouts] if "rect" in cutouts or "face" in cutouts else list(cutouts.values())
    return list(cutouts)


def _gate_cutout_within_face(parts: Any) -> GateResult:
    cutouts = _cutout_list(parts)
    if not cutouts:
        return _not_applicable("cutout_within_face", "no cutouts", 0.0)
    summary = _part_summary(parts)
    margin: Optional[float] = None
    for cutout in cutouts:
        cutout_box = _cutout_rect(cutout)
        face_box = _face_rect(cutout, parts, summary)
        if cutout_box is None or face_box is None:
            raise ValueError("cutout or face rect is missing")
        current = min(
            cutout_box[0] - face_box[0],
            cutout_box[1] - face_box[1],
            face_box[2] - cutout_box[2],
            face_box[3] - cutout_box[3],
        )
        margin = current if margin is None else min(margin, current)
    assert margin is not None
    return _result(
        "cutout_within_face",
        margin >= 0.0,
        margin,
        0.0,
        "minimum_face_margin={:.6g} threshold=0".format(margin),
    )


#: The ray that proves a through-cutout is open starts this far outside the outer surface.
_THROUGH_PROBE_OUTSIDE = 1.0
#: Clear depth may fall short of the wall by this much and still count as through: well below
#: one STL float ulp-scale facet error, and two orders below any skin a printer could lay down.
_THROUGH_TOL = 0.01


def _gate_cutout_through(parts: Any, measurements: Any) -> GateResult:
    """Every cutout the synth declared as an opening is open, measured on the exported mesh.

    Synth records each component access opening and each `cutout` / `cable_gland` feature with
    `through: true`, the point where its axis meets the outer surface (`surface_center`) and the
    thickness of the material it has to pierce (`wall_t`). A ray from just outside that point,
    pointing inward along the face normal, must meet no surface before it has travelled
    `wall_t`. Faces parallel to the ray (the bore) are never hits, so an open hole reports no
    hit and a blind one reports its floor.

    Written against a cable gland on a 2.8 mm open_frame floor whose cutter only reached
    (wall + 2) / 2 = 2.0 mm deep: a blind 2.0 mm pocket the drawing called out as `↓2.01`, with
    every other gate green. Vents are not judged: their recorded centre can fall between slots.
    """
    cutouts = [
        c for c in _cutout_list(parts)
        if _field(c, "through") and _field(c, "wall_t") is not None
        and _field(c, "surface_center") is not None
    ]
    if not cutouts:
        return _not_applicable("cutout_through", "no declared through-cutouts", 0.0)
    entries = dict(_measurement_items(measurements))
    if not entries:
        return _skipped("cutout_through", "skipped: no render", 0.0)

    from phimech.check.mesh import first_hit, read_stl
    from phimech.synth.features import face_axes

    meshes: Dict[str, Any] = {}
    worst: Optional[float] = None
    judged = 0
    blind: List[str] = []
    unmeasured: List[str] = []
    for cutout in cutouts:
        face = str(_field(cutout, "face") or "")
        wall_t = _number(_field(cutout, "wall_t"), "cutout.wall_t")
        surface = [float(v) for v in _field(cutout, "surface_center")]
        normal = face_axes(face)[2]
        label = "{}@{}({})".format(
            _field(cutout, "name") or _field(cutout, "kind") or "cutout", face,
            ",".join("{:.2f}".format(v) for v in surface),
        )
        for part in list(_field(cutout, "parts") or []):
            measurement = entries.get(part)
            stl = _measurement_stl(measurement) if measurement is not None else None
            if stl is None:
                unmeasured.append("{}:{}".format(part, label))
                continue
            if part not in meshes:
                meshes[part] = read_stl(str(stl))
            origin = tuple(surface[i] + normal[i] * _THROUGH_PROBE_OUTSIDE for i in range(3))
            inward = (-normal[0], -normal[1], -normal[2])
            hit = first_hit(meshes[part], origin, inward, _THROUGH_PROBE_OUTSIDE + wall_t)
            clear = wall_t if hit is None else max(0.0, hit - _THROUGH_PROBE_OUTSIDE)
            margin = clear - wall_t
            judged += 1
            worst = margin if worst is None else min(worst, margin)
            if margin < -_THROUGH_TOL:
                blind.append("{}:{} clear {:.3f} of {:.3f} mm (BLIND)".format(
                    part, label, clear, wall_t))
    if not judged:
        return _skipped(
            "cutout_through",
            "skipped: no stl for " + ", ".join(unmeasured[:6]),
            0.0,
        )
    assert worst is not None
    message = "minimum_clear_minus_wall={:.6g} over {} opening(s) threshold=0".format(
        worst, judged)
    if blind:
        message += "; " + "; ".join(blind[:6])
    if unmeasured:
        message += "; unmeasured: " + ", ".join(unmeasured[:6])
    return _result("cutout_through", not blind, round(worst, 6), 0.0, message)


def _text_sizes(design: Any, parts: Any) -> List[float]:
    sizes: List[float] = []
    spec = _enclosure_spec(design)
    text = spec.get("text")
    if isinstance(text, Mapping):
        if text.get("size") is not None:
            sizes.append(_number(text["size"], "enclosure.text.size"))
    features = spec.get("features", [])
    if isinstance(features, list):
        for feature in features:
            if isinstance(feature, Mapping) and feature.get("kind") == "text":
                value = feature.get("size")
                if value is None and isinstance(feature.get("params"), Mapping):
                    value = feature["params"].get("size")
                if value is not None:
                    sizes.append(_number(value, "text size"))
    summary = _part_summary(parts)
    for source in (parts, summary):
        for key in ("text_size", "text_sizes", "text", "texts"):
            value = _field(source, key, None)
            if isinstance(value, Mapping):
                value = value.get("size")
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Mapping):
                        item = item.get("size")
                    if item is not None:
                        sizes.append(_number(item, "text size"))
            elif value is not None:
                sizes.append(_number(value, "text size"))
    return sizes


def _labels_cut_away(parts: Any) -> List[str]:
    """Labels whose footprint overlaps a through-cutout on the same part.

    Cutouts are cut before text, so a label under a hole is removed with it while the report
    still lists its size: a "FAN" label under a d35 fan hole passed this gate at 6 mm with no
    glyph left in the mesh. Such a label's legible size is 0.
    """
    labels = _field(parts, "labels", None)
    if labels is None:
        labels = _part_summary(parts).get("labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        return []
    from phimech.synth.enclosure import through_rects_under_top

    style = str(_field(parts, "style", None) or _part_summary(parts).get("style") or "")
    cutouts = [dict(c) for c in _cutout_list(parts) if isinstance(c, Mapping) and "size" in c
               and "center" in c]
    lost: List[str] = []
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        if label.get("skipped"):
            lost.append("{!r} was not placed: no spot clear of the cutouts".format(
                label.get("label")))
            continue
        if label.get("rect") is None:
            continue
        x0, y0, x1, y1 = (float(v) for v in label["rect"])
        for r in through_rects_under_top(cutouts, str(label.get("part")), style):
            overlap = (min(x1, r[2]) - max(x0, r[0])) * (min(y1, r[3]) - max(y0, r[1]))
            if min(x1, r[2]) > max(x0, r[0]) and min(y1, r[3]) > max(y0, r[1]):
                lost.append("{!r} overlaps a cutout by {:.1f} mm2".format(
                    label.get("label"), overlap))
                break
    return lost


def _gate_text_legible(design: Any, parts: Any) -> GateResult:
    sizes = _text_sizes(design, parts)
    lost = _labels_cut_away(parts)
    if lost:
        sizes.append(0.0)
    if not sizes:
        return _not_applicable("text_legible", "no text", _MIN_TEXT_SIZE, "warn")
    value = min(sizes)
    message = "minimum_text_size_mm={:.6g} threshold=4".format(value)
    if lost:
        message += "; cut away (size 0): " + "; ".join(lost)
    return _result(
        "text_legible",
        value >= _MIN_TEXT_SIZE,
        value,
        _MIN_TEXT_SIZE,
        message,
        "warn" if value < _MIN_TEXT_SIZE else "error",
    )


def _gate_overhang(design: Any, measurements: Any) -> GateResult:
    entries = _measurement_items(measurements)
    if not entries:
        return _skipped("overhang", "skipped: no render", _MAX_OVERHANG_FRACTION, "warn")
    stls = [(name, _measurement_stl(measurement)) for name, measurement in entries]
    stls = [(name, path) for name, path in stls if path is not None]
    if not stls:
        return _skipped("overhang", "skipped: no stl", _MAX_OVERHANG_FRACTION, "warn")
    from phimech.check.mesh import overhang_fraction, read_stl

    limit_angle = _mfg_number(design, "max_overhang_deg", _DEFAULT_MAX_OVERHANG)
    fractions = [overhang_fraction(read_stl(str(path)), limit_angle) for _, path in stls]
    value = max(fractions)
    return _result(
        "overhang",
        value <= _MAX_OVERHANG_FRACTION,
        value,
        _MAX_OVERHANG_FRACTION,
        "maximum_overhang_fraction={:.6g} threshold=0.35 angle_deg={:.6g}".format(
            value, limit_angle
        ),
        "warn",
    )


def _gate_material_estimate(measurements: Any) -> GateResult:
    entries = _measurement_items(measurements)
    if not entries:
        # 0 g of material is not an estimate, it is the absence of one.
        return _skipped("material_estimate", "skipped: no render", 0.0, "warn")
    volume_mm3 = sum(float(getattr(measurement, "volume")) for _, measurement in entries)
    grams = volume_mm3 / 1000.0 * _PLA_DENSITY
    return _result(
        "material_estimate",
        True,
        grams,
        0.0,
        "material_g={:.6g} density_g_cm3=1.24 scale=1.0".format(grams),
        "warn",
    )


# --------------------------------------------------------------------------------------
# assembly-interference gates
#
# The gates above answer "is this printable": walls, floors, feature sizes, bed fit. They do
# not answer "does the thing that was designed actually go together". These three do, and each
# was written against a design that the pre-existing 14 gates passed clean:
#
#  * `standoff_within_cavity` — a mount hole at (900, 900) on a 60 x 40 mm part produced a
#    standoff boss 830 mm outside the enclosure, floating in space. PASS, 0 failed.
#  * `fastener_clears_components` — `lid.screw_count: 6` puts two posts at the mid-points of
#    the long edges. `_reserve_screw_corners()` only reserves room for the four *corner* posts,
#    so on an 80 x 40 mm component the mid-edge posts ran straight through it. PASS, 0 failed.
#  * `standoff_lands_on_hole` — nothing checked that a synthesised boss corresponds to a hole
#    that the referenced component actually has.
#
# All three work off recorded data (`standoff_positions`, `screw_positions`, the placement's
# own boxes) rather than re-deriving geometry, and all three are numeric with a 0 mm threshold.
# --------------------------------------------------------------------------------------

def _feature_cylinder(entry: Any) -> Optional[Tuple[float, float, float, float, float]]:
    """One recorded boss/post as ``(x, y, r, z0, z1)``, or None if it lacks geometry.

    A boss is a cylinder in both `features.boss_for_hole()` and `features.lid_screw_post()`, so
    it is modelled as one here. Substituting the circumscribing square prism would be simpler
    but wrong in the expensive direction: it reports a corner post as colliding with a component
    whose corner is 2 mm diagonally away, and a gate that cries wolf gets switched off.
    """
    at = _field(entry, "at", None)
    if at is None:
        return None
    try:
        x = float(at[0])
        y = float(at[1])
    except (TypeError, ValueError, IndexError):
        return None
    od = _standoff_dimension(entry, ("od", "outer_d", "outer_diameter", "diameter"))
    z0 = _standoff_dimension(entry, ("z0", "z", "base_z"))
    height = _standoff_dimension(entry, ("h", "height", "length"))
    if od is None or z0 is None or height is None:
        return None
    return (x, y, od / 2.0, z0, z0 + height)


def _signed_distance_to_rect(
    x: float, y: float, rect: Tuple[float, float, float, float]
) -> float:
    """Distance from ``(x, y)`` to an axis-aligned rectangle; negative inside.

    Exact in both directions, which is what makes the cylinder tests below exact rather than
    conservative: outside it is the true Euclidean distance to the boundary, inside it is minus
    the distance to the nearest edge.
    """
    x0, y0, x1, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    if dx > 0.0 or dy > 0.0:
        return (dx * dx + dy * dy) ** 0.5
    return -min(x - x0, x1 - x, y - y0, y1 - y)


def _gate_standoff_within_cavity(parts: Any) -> GateResult:
    """Every synthesised boss and lid post must stand on the enclosure cavity floor.

    The decision is on the boss *axis*, not its outside diameter, and only in X/Y:

    * A boss whose axis is outside the cavity footprint is not a mounting feature. Its bore has
      no floor under it, so it is either unsupported geometry floating in the print or a blind
      lump buried in a wall, and in both cases the component hole it claims to serve is not over
      the floor either.
    * A boss whose axis is inside but whose outside diameter overlaps a wall is *correct and
      common* — a PCB mounting hole sits a few millimetres from the board edge, so its boss
      fuses into the wall. Judging the outside diameter would fail every real design; the repo's
      example enclosure designs both have bosses that touch a wall.
    * Z is not judged: a lid post is *meant* to reach the cavity ceiling.

    The value is the smallest axis-to-cavity-edge distance in mm, negative when outside.
    """
    cavity = _field(parts, "inner_cavity", None)
    if cavity is None:
        raise ValueError("enclosure_parts.inner_cavity is missing")
    cavity_min, cavity_max = _bbox_pair(cavity)
    rect = (cavity_min[0], cavity_min[1], cavity_max[0], cavity_max[1])
    entries: List[Tuple[str, Any]] = []
    for label, name in (("standoff", "standoff_positions"), ("post", "screw_positions")):
        for entry in _field(parts, name, None) or ():
            entries.append((label, entry))
    if not entries:
        return _not_applicable("standoff_within_cavity", "no standoffs or lid posts", 0.0)
    margin: Optional[float] = None
    worst = ""
    for label, entry in entries:
        cylinder = _feature_cylinder(entry)
        if cylinder is None:
            raise ValueError("{} entry lacks at/od/z0/h".format(label))
        x, y = cylinder[0], cylinder[1]
        # Negated: signed distance is negative inside, and a margin is positive inside.
        current = -_signed_distance_to_rect(x, y, rect)
        if margin is None or current < margin:
            margin = current
            worst = "{} at ({:.2f}, {:.2f})".format(label, x, y)
    assert margin is not None
    return _result(
        "standoff_within_cavity",
        margin >= 0.0,
        margin,
        0.0,
        "minimum_axis_to_cavity_edge_mm={:.6g} threshold=0 worst={}".format(margin, worst),
    )


def _gate_fastener_clears_components(placement: Any, parts: Any) -> GateResult:
    """Lid screw posts must not interpenetrate any placed component keep-out.

    `synth/enclosure.py` reserves cavity width for the four *corner* posts only. Any other post
    — a mid-edge post from `lid.screw_count > 4`, a magnet post — is placed with no clearance
    analysis at all, and the enclosure's own note says so ("posts run the full cavity height —
    check them against components"). This is that check.

    The value is the deepest interpenetration in mm, so 0 means touching and positive means
    overlapping; the threshold is 0 and the comparison is `<= 0`.
    """
    entries = list(_field(parts, "screw_positions", None) or ())
    if not entries:
        return _not_applicable("fastener_clears_components", "no lid posts", 0.0)
    boxes = _placement_boxes(placement)
    if not boxes:
        return _not_applicable("fastener_clears_components", "no placed components", 0.0)
    offset = _world_offset(parts)
    deepest = 0.0
    worst = ""
    for entry in entries:
        cylinder = _feature_cylinder(entry)
        if cylinder is None:
            raise ValueError("lid post entry lacks at/od/z0/h")
        x, y, radius, post_z0, post_z1 = cylinder
        for item in boxes:
            raw_min, raw_max = _world_bbox(item)
            box_min = tuple(raw_min[axis] + offset[axis] for axis in range(3))
            box_max = tuple(raw_max[axis] + offset[axis] for axis in range(3))
            z_overlap = min(post_z1, box_max[2]) - max(post_z0, box_min[2])
            if z_overlap <= 0.0:
                continue
            # Exact minimum translation in XY to separate a disc from a rectangle.
            xy_overlap = radius - _signed_distance_to_rect(
                x, y, (box_min[0], box_min[1], box_max[0], box_max[1])
            )
            if xy_overlap <= 0.0:
                continue
            # Interference exists only when both XY and Z overlap; the depth reported is the
            # smaller of the two, i.e. the shortest distance the post would have to move.
            depth = min(xy_overlap, z_overlap)
            if depth > deepest:
                deepest = depth
                worst = "post at ({:.2f}, {:.2f}) into {}".format(
                    x, y, _field(item, "ref", "?")
                )
    detail = " worst={}".format(worst) if worst else ""
    return _result(
        "fastener_clears_components",
        deepest <= _TOUCH_EPS,
        deepest,
        0.0,
        "maximum_interpenetration_mm={:.6g} threshold=0{}".format(deepest, detail),
    )


def _component_hole_positions(placement: Any) -> Dict[str, List[Tuple[float, float]]]:
    """World XY of every mounting hole the placed components actually declare, per ref.

    Read through `synth.features.hole_to_world()` so this gate uses the same local-to-world
    transform the synthesiser used, rather than a second copy of the rotation maths.
    """
    from phimech.synth.features import hole_to_world

    out: Dict[str, List[Tuple[float, float]]] = {}
    items = _field(placement, "items", None) or ()
    if callable(items):
        items = items()
    for placed in items:
        envelope = _field(placed, "envelope", None)
        if envelope is None:
            continue
        ref = str(_field(placed, "ref", "") or "")
        positions: List[Tuple[float, float]] = []
        for hole in _field(envelope, "holes", None) or ():
            world = hole_to_world(placed, hole)
            positions.append((float(world[0]), float(world[1])))
        out[ref] = positions
    return out


def _gate_standoff_lands_on_hole(placement: Any, parts: Any) -> GateResult:
    """Every component standoff must sit under a mounting hole that component really has.

    This is the referencing half of "does it fit together": a boss is only a mount if the part
    it holds has a hole there. The value is the largest centre-to-centre distance between a
    synthesised boss and its nearest declared hole, in mm; the threshold is a tight 0.05 mm
    because both numbers come from the same transform and should agree to floating-point noise.
    A boss whose recorded `ref` names no placed component fails outright.
    """
    entries = list(_field(parts, "standoff_positions", None) or ())
    if not entries:
        return _not_applicable("standoff_lands_on_hole", "no standoffs", _HOLE_MATCH_TOL)
    holes_by_ref = _component_hole_positions(placement)
    if not holes_by_ref:
        return _skipped(
            "standoff_lands_on_hole", "skipped: no placement", _HOLE_MATCH_TOL
        )
    offset = _world_offset(parts)
    worst_distance = 0.0
    worst = ""
    for entry in entries:
        at = _field(entry, "at", None)
        ref = _field(entry, "ref", None)
        if at is None:
            raise ValueError("standoff entry lacks at")
        try:
            bx = float(at[0])
            by = float(at[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError("standoff at is not a 2D point: {!r}".format(at)) from None
        if ref is None or str(ref) not in holes_by_ref:
            return _result(
                "standoff_lands_on_hole",
                False,
                float("inf"),
                _HOLE_MATCH_TOL,
                "standoff at ({:.2f}, {:.2f}) names component {!r}, which is not placed".format(
                    bx, by, ref
                ),
            )
        candidates = holes_by_ref[str(ref)]
        if not candidates:
            return _result(
                "standoff_lands_on_hole",
                False,
                float("inf"),
                _HOLE_MATCH_TOL,
                "standoff at ({:.2f}, {:.2f}) claims {} but that component declares no holes".format(
                    bx, by, ref
                ),
            )
        # Boss records live in the enclosure datum frame, hole positions in the placement frame.
        distance = min(
            (((bx - (hx + offset[0])) ** 2 + (by - (hy + offset[1])) ** 2) ** 0.5)
            for hx, hy in candidates
        )
        if distance > worst_distance:
            worst_distance = distance
            worst = "{} boss at ({:.2f}, {:.2f})".format(ref, bx, by)
    detail = " worst={}".format(worst) if worst else ""
    return _result(
        "standoff_lands_on_hole",
        worst_distance <= _HOLE_MATCH_TOL,
        worst_distance,
        _HOLE_MATCH_TOL,
        "maximum_boss_to_hole_mm={:.6g} threshold={:g}{}".format(
            worst_distance, _HOLE_MATCH_TOL, detail
        ),
    )


def _gate_catalog_entry(name: str, severity: str, description: str, source: str) -> Dict[str, str]:
    return {
        "name": name,
        "severity": severity,
        "description": description,
        "threshold_source": source,
    }


def gate_catalog() -> List[Dict[str, str]]:
    """Describe the stable gate names for docs and MCP discovery."""
    return [
        _gate_catalog_entry(
            "ir_valid", "error", "IR has zero validation errors.", "design._errors; 0"
        ),
        _gate_catalog_entry(
            "placement_no_overlap",
            "error",
            "Placed keep-out boxes do not overlap.",
            "placement.errors; 0",
        ),
        _gate_catalog_entry(
            "clearance_positive",
            "error",
            "Every grown box is inside the cavity.",
            "inner_cavity; 0 mm",
        ),
        _gate_catalog_entry(
            "wall_thickness",
            "error",
            "Wall meets minimum and is nozzle-friendly; nozzle mismatch warns.",
            "manufacturing.min_wall; nozzle ±0.05 mm",
        ),
        _gate_catalog_entry(
            "floor_thickness",
            "error",
            "Floor meets the minimum wall threshold.",
            "manufacturing.min_wall; mm",
        ),
        _gate_catalog_entry(
            "fits_bed",
            "error",
            "Rendered parts fit the printer bed.",
            "manufacturing.printer_bed; 0 mm margin",
        ),
        _gate_catalog_entry(
            "part_volume_positive",
            "error",
            "Every rendered part has positive solid volume.",
            "measurement.volume; >0 mm3",
        ),
        _gate_catalog_entry(
            "manifold",
            "error",
            "Parts are one solid and optional STL meshes are watertight.",
            "measurement.solids; 1 and mesh edges ×2",
        ),
        _gate_catalog_entry(
            "boolean_volume_sane",
            "error",
            "A part's top-level cut did not grow it and its union did not shrink below its "
            "largest child (same-kernel volumes).",
            "measurement.extra.boolean_reference; ratio <= 1.005",
        ),
        _gate_catalog_entry(
            "min_feature",
            "error",
            "Standoff annular wall meets minimum feature.",
            "manufacturing.min_feature; mm",
        ),
        _gate_catalog_entry(
            "screw_count_sane",
            "error",
            "Screw lids have 2–8 screws; excess warns.",
            "lid.screw_count; 2 minimum, 8 warning maximum",
        ),
        _gate_catalog_entry(
            "cutout_within_face",
            "error",
            "Every cutout rectangle stays within its face rectangle.",
            "cutout/face rects; 0 mm margin",
        ),
        _gate_catalog_entry(
            "cutout_through",
            "error",
            "Every declared through-opening (access, cutout, cable_gland) pierces its whole "
            "wall, floor or lid on the exported mesh.",
            "cutouts[].wall_t vs STL ray clear depth; 0 mm (0.01 tol)",
        ),
        _gate_catalog_entry(
            "standoff_within_cavity",
            "error",
            "Every standoff boss and lid post stands inside the cavity footprint.",
            "standoff/screw_positions vs inner_cavity; 0 mm margin",
        ),
        _gate_catalog_entry(
            "fastener_clears_components",
            "error",
            "No lid screw post interpenetrates a placed component keep-out.",
            "screw_positions vs placement keep-outs; 0 mm overlap",
        ),
        _gate_catalog_entry(
            "standoff_lands_on_hole",
            "error",
            "Every standoff sits under a mounting hole its component really declares.",
            "standoff_positions vs envelope holes; 0.05 mm",
        ),
        _gate_catalog_entry(
            "text_legible",
            "warn",
            "Text is at least 4 mm high and no label is cut away by a through-cutout.",
            "fixed printability threshold; 4 mm",
        ),
        _gate_catalog_entry(
            "overhang",
            "warn",
            "Support-demanding overhang area is limited.",
            "manufacturing.max_overhang_deg; fraction ≤0.35",
        ),
        _gate_catalog_entry(
            "material_estimate",
            "warn",
            "Informational PLA mass estimate.",
            "1.24 g/cm3 × volume × 1.0",
        ),
    ]


def run_gates(
    design: Dict[str, Any],
    placement: Any,
    enclosure_parts: Any,
    measurements: Dict[str, Measurement],
) -> GateReport:
    """Run every gate and turn any individual crash into a failed result."""
    checks: List[Tuple[str, Callable[[], GateResult], str]] = [
        ("ir_valid", lambda: _gate_ir_valid(design), "error"),
        (
            "placement_no_overlap",
            lambda: _gate_placement_no_overlap(placement),
            "error",
        ),        (
            "clearance_positive",
            lambda: _gate_clearance_positive(placement, enclosure_parts),
            "error",
        ),
        ("wall_thickness", lambda: _gate_wall_thickness(design, enclosure_parts), "error"),
        ("floor_thickness", lambda: _gate_floor_thickness(design, enclosure_parts), "error"),
        ("fits_bed", lambda: _gate_fits_bed(design, measurements), "error"),
        ("part_volume_positive", lambda: _gate_part_volume_positive(measurements), "error"),
        ("manifold", lambda: _gate_manifold(measurements), "error"),
        ("boolean_volume_sane", lambda: _gate_boolean_volume_sane(measurements), "error"),
        ("min_feature", lambda: _gate_min_feature(design, enclosure_parts), "error"),
        ("screw_count_sane", lambda: _gate_screw_count_sane(design, enclosure_parts), "error"),
        ("cutout_within_face", lambda: _gate_cutout_within_face(enclosure_parts), "error"),
        (
            "cutout_through",
            lambda: _gate_cutout_through(enclosure_parts, measurements),
            "error",
        ),
        (
            "standoff_within_cavity",
            lambda: _gate_standoff_within_cavity(enclosure_parts),
            "error",
        ),
        (
            "fastener_clears_components",
            lambda: _gate_fastener_clears_components(placement, enclosure_parts),
            "error",
        ),
        (
            "standoff_lands_on_hole",
            lambda: _gate_standoff_lands_on_hole(placement, enclosure_parts),
            "error",
        ),
        ("text_legible", lambda: _gate_text_legible(design, enclosure_parts), "warn"),
        ("overhang", lambda: _gate_overhang(design, measurements), "warn"),
        ("material_estimate", lambda: _gate_material_estimate(measurements), "warn"),
    ]
    results: List[GateResult] = []
    for name, check, severity in checks:
        try:
            result = check()
            if result.name != name:
                result = GateResult(
                    name,
                    result.passed,
                    result.value,
                    result.threshold,
                    result.message,
                    result.severity,
                    result.status,
                    result.layer,
                )
        except Exception as exc:  # one malformed optional surface must not abort all gates
            # A crash is `undetermined`, not `fail`: the gate never reached its predicate. It
            # still blocks (`passed=False`), so nobody mistakes a broken gate for a green one.
            result = _undetermined(
                name, "gate crashed: {}".format(exc), 0.0, severity, layer="gates"
            )
        results.append(result)
    return GateReport(results)


__all__ = [
    "BOOLEAN_VOLUME_TOLERANCE",
    "GateReport",
    "GateResult",
    "MEASUREMENT_GATES",
    "STATUSES",
    "as_undetermined",
    "derive_status",
    "gate_catalog",
    "passed_for_status",
    "result_status",
    "run_gates",
    "summarize",
]
