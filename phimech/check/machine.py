"""Gates that judge a design against the **actual printer**, queried read-only.

Every other gate in `phimech/check/` judges a design against numbers written in the design
itself: `manufacturing.printer_bed`, `min_wall`, `nozzle`. Those are author intent. This module
judges it against what the machine on the network actually reports — its levelled bed area, its
Z travel, its nozzle diameter, its heater ceilings, whether it has a heated chamber at all —
so that "will this print on *our* printer" is answered by the printer rather than by a number
somebody typed into a JSON file months ago.

Read-only, and structurally so
------------------------------
This module reaches the machine through exactly one function: `phimech.print.printer.printer_info()`.
That module carries the repository's no-print-control safety contract, machine-checked by
`tests/test_print.py` against its own source (no print/pause/cancel endpoint, no G-code command,
no heater or motion helper, exactly one POST and that is the file upload). `tests/test_print.py`
runs the same scan over *this* module. Nothing here starts a print, moves an axis, heats
anything, loads filament, or writes config; starting a print belongs to the separate `qidi-mcp`
server behind its own two-step confirm token, where the interlocks and the audit log live.

Fail closed
-----------
An unreachable printer is `unknown`, and `unknown` fails. It is not skipped, and it does not fall
back to the compiled-in envelope constant.

That is the opposite of the convention used by the render-dependent gates (`skipped: no render`
→ `passed=None` → does not block), and the difference is deliberate. A missing render means a
question was not asked yet. A missing printer means the question *was* asked and could not be
answered — and the failure mode is a part that does not fit a machine somebody is about to send
it to. `printer_info()` deliberately still fills in the documented envelope so a build can be
gated with the printer switched off, so these gates check `build_volume_source` and refuse to
treat a compiled-in constant as a machine reading.

Material feasibility
--------------------
`MATERIAL_REQUIREMENTS` is published-typical process temperatures per family, not measurements
taken on this machine, and it says so in every message it produces. Where the OrcaSlicer profile
pack is installed the real flattened profile is preferred and the report says
`temps_source=profile`; otherwise the table is used and the report says `temps_source=table`.
The load-bearing entries are the chamber requirements: ABS, ASA, PC, PA and their filled variants
warp or delaminate without a heated chamber, which is exactly the fact a design should be gated
on before a human commits to a 9-hour print.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from phimech.check.gates import GateReport, GateResult

#: Published-typical process requirements per material family. NOT measurements from this
#: machine, and never presented as such: every message naming these numbers also names its
#: source. `chamber_c` is `None` when the family does not need a heated chamber, and a
#: temperature when it does.
MATERIAL_REQUIREMENTS: Dict[str, Dict[str, Optional[float]]] = {
    "pla": {"nozzle_c": 215.0, "bed_c": 60.0, "chamber_c": None},
    "petg": {"nozzle_c": 245.0, "bed_c": 80.0, "chamber_c": None},
    "tpu": {"nozzle_c": 230.0, "bed_c": 45.0, "chamber_c": None},
    "abs": {"nozzle_c": 260.0, "bed_c": 100.0, "chamber_c": 50.0},
    "asa": {"nozzle_c": 265.0, "bed_c": 100.0, "chamber_c": 55.0},
    "pc": {"nozzle_c": 290.0, "bed_c": 110.0, "chamber_c": 60.0},
    "pa": {"nozzle_c": 290.0, "bed_c": 100.0, "chamber_c": 55.0},
    "pet": {"nozzle_c": 290.0, "bed_c": 100.0, "chamber_c": 55.0},
    "pps": {"nozzle_c": 320.0, "bed_c": 120.0, "chamber_c": 60.0},
    "peba": {"nozzle_c": 240.0, "bed_c": 60.0, "chamber_c": None},
}

#: Longest first, so `petg-cf` resolves to `petg` and not to `pet`.
_FAMILY_ORDER = sorted(MATERIAL_REQUIREMENTS, key=len, reverse=True)

#: How far the design's declared nozzle may differ from the machine's fitted nozzle, mm.
#: Tight: a nozzle is a discrete part, so the two either name the same one or they do not.
_NOZZLE_TOL = 0.001

_MACHINE_GATES = (
    "machine_reachable",
    "machine_fits_build_volume",
    "machine_nozzle_matches",
    "machine_material_feasible",
)


def _result(
    name: str,
    passed: Optional[bool],
    value: Any,
    threshold: Any,
    message: str,
    severity: str = "error",
) -> GateResult:
    return GateResult(name, passed, value, threshold, message, severity)


def _unknown(name: str, threshold: Any, why: str) -> GateResult:
    """A fact the machine could not supply. Fails, and says `unknown` rather than a number."""
    return _result(name, False, "unknown", threshold, "unknown: {}".format(why))


def query_machine(base_url: str = "") -> Dict[str, Any]:
    """Read this printer's capabilities. Never raises; never changes machine state.

    Returns `printer_info()`'s dict unchanged plus nothing of its own, so a caller and a gate
    read the same facts. On failure the dict carries `reachable: False`, an `error` string, and
    `build_volume_source: "documented"`.
    """
    from phimech.print import printer

    return dict(printer.printer_info(base_url=base_url or None))


def material_family(material: str) -> Optional[str]:
    """The `MATERIAL_REQUIREMENTS` family a material name belongs to, or None if unrecognised.

    Matches on substring so vendor and filled names resolve: `QIDI ABS Rapido` -> `abs`,
    `PETG-CF` -> `petg`, `PAHT-CF` -> `pa`. An unrecognised name is None rather than a guessed
    default, because guessing PLA for an unknown material is exactly how a design gets gated
    against a chamber requirement it does not have.
    """
    text = str(material or "").strip().lower()
    if not text:
        return None
    for family in _FAMILY_ORDER:
        if family in text:
            return family
    return None


def _profile_temps(material: str) -> Optional[Dict[str, Optional[float]]]:
    """Real nozzle/bed temperatures from the installed OrcaSlicer pack, if it is installed.

    Preferred over the table because it is the number this machine's own profile would actually
    slice with. Returns None whenever the pack is absent or the name does not resolve, which is
    the normal case on a machine that only has PhiMech checked out.
    """
    try:
        from phimech.print import profiles

        resolved = profiles.resolve(material=material, quality="standard")
    except Exception:  # a missing/incomplete profile pack must not fail the gate
        return None
    nozzle_c = getattr(resolved, "nozzle_temp_c", None)
    bed_c = getattr(resolved, "bed_temp_c", None)
    if nozzle_c is None and bed_c is None:
        return None
    family = material_family(material)
    table = MATERIAL_REQUIREMENTS.get(family or "", {})
    return {
        "nozzle_c": float(nozzle_c) if nozzle_c is not None else table.get("nozzle_c"),
        "bed_c": float(bed_c) if bed_c is not None else table.get("bed_c"),
        # The profile pack has no "needs a chamber" field, so the chamber requirement always
        # comes from the table. This is the one number the gate leans on hardest, so where it
        # comes from must not be ambiguous.
        "chamber_c": table.get("chamber_c"),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _manufacturing(design: Any) -> Mapping[str, Any]:
    return _mapping(_mapping(design).get("manufacturing"))


def _reachable(machine: Mapping[str, Any]) -> bool:
    return bool(machine.get("reachable"))


def _queried_envelope(machine: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    """The machine's own build volume, or None when the value is the compiled-in constant.

    The provenance check is the whole point: `printer_info()` fills in the documented envelope
    even when the printer is off, so a gate that read `build_volume_mm` alone would report a
    constant as a measurement.
    """
    if machine.get("build_volume_source") != "queried":
        return None
    volume = machine.get("build_volume_mm")
    if not isinstance(volume, (list, tuple)) or len(volume) != 3:
        return None
    try:
        values = tuple(float(v) for v in volume)
    except (TypeError, ValueError):
        return None
    if any(v <= 0.0 for v in values):
        return None
    return (values[0], values[1], values[2])


def _measurement_sizes(measurements: Any) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Rendered part bounding sizes, from live `Measurement` objects or a re-read report dict."""
    out: List[Tuple[str, Tuple[float, float, float]]] = []
    if not isinstance(measurements, Mapping):
        return out
    for name, measurement in measurements.items():
        bbox = getattr(measurement, "bbox", None)
        if bbox is None and isinstance(measurement, Mapping):
            bbox = measurement.get("bbox")
        if bbox is None:
            continue
        size = getattr(bbox, "size", None)
        if size is None and isinstance(bbox, Mapping):
            if "size" in bbox:
                size = bbox["size"]
            elif "min" in bbox and "max" in bbox:
                try:
                    size = [
                        float(bbox["max"][axis]) - float(bbox["min"][axis]) for axis in range(3)
                    ]
                except (TypeError, ValueError, IndexError, KeyError):
                    size = None
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            continue
        try:
            out.append((str(name), (float(size[0]), float(size[1]), float(size[2]))))
        except (TypeError, ValueError):
            continue
    return out


def _enclosure_outer_size(enclosure_parts: Any) -> Optional[Tuple[float, float, float]]:
    """Outer size of the synthesised enclosure, so the gate works before any render exists.

    A render is the better evidence and is preferred when present, but a bbox from synthesis is
    enough to catch the case this gate exists for — a design that cannot physically fit the
    machine — and catching it in `synth check` (milliseconds) rather than after a render is the
    difference between a useful gate and one that runs too late to matter.
    """
    for attribute in ("outer_size", "outer_bbox"):
        value = getattr(enclosure_parts, attribute, None)
        if value is None:
            continue
        if attribute == "outer_size":
            if isinstance(value, (list, tuple)) and len(value) == 3:
                try:
                    return (float(value[0]), float(value[1]), float(value[2]))
                except (TypeError, ValueError):
                    continue
            continue
        try:
            low, high = value
            return (
                float(high[0]) - float(low[0]),
                float(high[1]) - float(low[1]),
                float(high[2]) - float(low[2]),
            )
        except (TypeError, ValueError, IndexError):
            continue
    summary = getattr(enclosure_parts, "summary", None)
    if callable(summary):
        try:
            size = summary().get("outer_size_mm")
        except Exception:
            return None
        if isinstance(size, (list, tuple)) and len(size) == 3:
            try:
                return (float(size[0]), float(size[1]), float(size[2]))
            except (TypeError, ValueError):
                return None
    return None


def _gate_machine_reachable(machine: Mapping[str, Any]) -> GateResult:
    """Did the machine answer, and are the numbers below its own rather than compiled in?

    Value is the provenance string, not a temperature or a length: this gate exists so that a
    reader can tell at a glance whether the three gates after it mean anything.
    """
    url = machine.get("url") or "?"
    if not _reachable(machine):
        return _result(
            "machine_reachable",
            False,
            "unreachable",
            "queried",
            "unknown: printer at {} did not answer ({}); machine gates fail closed rather "
            "than assume the documented envelope".format(url, machine.get("error") or "no error"),
        )
    source = machine.get("build_volume_source")
    if source != "queried":
        return _result(
            "machine_reachable",
            False,
            str(source),
            "queried",
            "unknown: printer at {} answered but supplied no bed mesh / Z limit, so the "
            "envelope is the compiled-in constant rather than a machine reading".format(url),
        )
    state = machine.get("klippy_state")
    return _result(
        "machine_reachable",
        True,
        "queried",
        "queried",
        "printer={} state={} envelope_source=queried".format(
            machine.get("hostname") or url, state
        ),
    )


def _gate_machine_fits_build_volume(
    machine: Mapping[str, Any], enclosure_parts: Any, measurements: Any
) -> GateResult:
    """Does every printable part fit the machine's own levelled build volume?

    Uses `printer.fits_build_volume()` so the 90-degree-about-Z retry and the wording of the
    verdict are the ones already proven in `tests/test_print.py`, not a second implementation.
    The value is the smallest per-axis slack in mm over the best orientation of the worst part.
    """
    envelope = _queried_envelope(machine)
    if envelope is None:
        return _unknown(
            "machine_fits_build_volume",
            0.0,
            "no queried build volume; refusing to gate against the compiled-in envelope",
        )
    parts = _measurement_sizes(measurements)
    basis = "render"
    if not parts:
        outer = _enclosure_outer_size(enclosure_parts)
        if outer is None:
            return _unknown(
                "machine_fits_build_volume", 0.0, "no rendered part and no synthesised enclosure"
            )
        parts = [("enclosure_bbox", outer)]
        basis = "synth bbox"

    from phimech.print import printer

    worst_margin: Optional[float] = None
    worst_name = ""
    worst_why = ""
    for name, size in parts:
        fits, why = printer.fits_build_volume(size, envelope=envelope)
        # Slack of the better of the two Z orientations, matching what fits_build_volume allows.
        margin = max(
            min(envelope[0] - size[0], envelope[1] - size[1], envelope[2] - size[2]),
            min(envelope[0] - size[1], envelope[1] - size[0], envelope[2] - size[2]),
        )
        if not fits and margin >= 0.0:
            # Keep verdict and number consistent: fits_build_volume is authoritative.
            margin = -0.0
        if worst_margin is None or margin < worst_margin:
            worst_margin, worst_name, worst_why = margin, name, why
    assert worst_margin is not None
    return _result(
        "machine_fits_build_volume",
        worst_margin >= 0.0,
        worst_margin,
        0.0,
        "minimum_machine_margin_mm={:.6g} threshold=0 envelope={:g}x{:g}x{:g} "
        "source=queried basis={} worst={} ({})".format(
            worst_margin, envelope[0], envelope[1], envelope[2], basis, worst_name, worst_why
        ),
    )


def _gate_machine_nozzle_matches(machine: Mapping[str, Any], design: Any) -> GateResult:
    """Is the nozzle the design's wall arithmetic assumed the nozzle actually fitted?

    Warning severity, not error: a 0.4 mm design on a 0.6 mm nozzle still prints, it just does
    not have the wall/nozzle relationship `wall_thickness` believed it had. The number is what
    matters, so the gate reports both diameters.
    """
    if not _reachable(machine):
        return _result(
            "machine_nozzle_matches",
            False,
            "unknown",
            _NOZZLE_TOL,
            "unknown: printer unreachable, cannot confirm the fitted nozzle",
            "warn",
        )
    fitted = machine.get("nozzle_mm")
    if fitted is None:
        return _result(
            "machine_nozzle_matches",
            False,
            "unknown",
            _NOZZLE_TOL,
            "unknown: printer reported no nozzle_diameter",
            "warn",
        )
    declared = _manufacturing(design).get("nozzle")
    if declared is None:
        return _result(
            "machine_nozzle_matches",
            True,
            float(fitted),
            _NOZZLE_TOL,
            "design declares no nozzle; machine has {:g} mm".format(float(fitted)),
            "warn",
        )
    try:
        delta = abs(float(declared) - float(fitted))
    except (TypeError, ValueError):
        return _result(
            "machine_nozzle_matches",
            False,
            "unknown",
            _NOZZLE_TOL,
            "manufacturing.nozzle is not numeric: {!r}".format(declared),
            "warn",
        )
    return _result(
        "machine_nozzle_matches",
        delta <= _NOZZLE_TOL,
        delta,
        _NOZZLE_TOL,
        "declared={:g} mm fitted={:g} mm delta={:.6g}".format(
            float(declared), float(fitted), delta
        ),
        "warn",
    )


def _gate_machine_material_feasible(machine: Mapping[str, Any], design: Any) -> GateResult:
    """Can this machine run the material the design asks for?

    Three ways to fail, and each names the number it failed on:

    * the machine cannot reach the material's nozzle or bed temperature;
    * the material needs a heated chamber and the machine has none;
    * the material needs a chamber hotter than the machine's chamber ceiling.

    A design that declares no material is not judged: `passed=None`, `skipped`. That is a real
    skip rather than a fail-closed `unknown`, because nothing was claimed \u2014 unlike an
    unreachable printer, where a question was asked and could not be answered.
    """
    material = _manufacturing(design).get("material")
    if material is None or not str(material).strip():
        return _result(
            "machine_material_feasible",
            None,
            None,
            0.0,
            "skipped: design declares no manufacturing.material",
        )
    material = str(material).strip()
    if not _reachable(machine):
        return _unknown(
            "machine_material_feasible",
            0.0,
            "printer unreachable, cannot confirm heater ceilings for {!r}".format(material),
        )
    family = material_family(material)
    if family is None:
        return _result(
            "machine_material_feasible",
            False,
            "unknown",
            0.0,
            "unknown: material {!r} matches no family in MATERIAL_REQUIREMENTS ({}); add it "
            "rather than assuming it prints".format(material, ", ".join(sorted(MATERIAL_REQUIREMENTS))),
        )
    requirements = _profile_temps(material)
    temps_source = "profile"
    if requirements is None:
        requirements = dict(MATERIAL_REQUIREMENTS[family])
        temps_source = "table"

    ceilings = _mapping(machine.get("max_temps"))
    problems: List[str] = []
    margins: List[float] = []

    for key, ceiling_key, label in (
        ("nozzle_c", "extruder_c", "nozzle"),
        ("bed_c", "bed_c", "bed"),
    ):
        need = requirements.get(key)
        have = ceilings.get(ceiling_key)
        if need is None:
            continue
        if have is None:
            return _unknown(
                "machine_material_feasible",
                0.0,
                "printer reported no {} max_temp".format(label),
            )
        margin = float(have) - float(need)
        margins.append(margin)
        if margin < 0.0:
            problems.append(
                "{} needs {:g} C, machine ceiling {:g} C".format(label, float(need), float(have))
            )

    chamber_need = requirements.get("chamber_c")
    if chamber_need is not None:
        chamber_have = ceilings.get("chamber_c") if machine.get("heated_chamber") else None
        if chamber_have is None:
            problems.append(
                "{} needs a {:g} C heated chamber; this machine reports none".format(
                    family.upper(), float(chamber_need)
                )
            )
            margins.append(-float(chamber_need))
        else:
            margin = float(chamber_have) - float(chamber_need)
            margins.append(margin)
            if margin < 0.0:
                problems.append(
                    "chamber needs {:g} C, machine ceiling {:g} C".format(
                        float(chamber_need), float(chamber_have)
                    )
                )

    worst = min(margins) if margins else 0.0
    chamber_text = (
        "none required" if chamber_need is None else "{:g} C".format(float(chamber_need))
    )
    detail = "; ".join(problems)
    return _result(
        "machine_material_feasible",
        not problems,
        worst,
        0.0,
        "material={} family={} minimum_thermal_margin_C={:.6g} threshold=0 chamber={} "
        "temps_source={} (published-typical, not measured on this machine){}".format(
            material, family, worst, chamber_text, temps_source,
            "" if not detail else " | " + detail,
        ),
    )


def machine_gate_catalog() -> List[Dict[str, str]]:
    """Describe the machine gates for docs and MCP discovery, like `gates.gate_catalog()`."""
    return [
        {
            "name": "machine_reachable",
            "severity": "error",
            "description": (
                "The printer answered and supplied its own envelope; a compiled-in constant "
                "does not count. Fails closed when unknown."
            ),
            "threshold_source": "printer_info().build_volume_source; 'queried'",
        },
        {
            "name": "machine_fits_build_volume",
            "severity": "error",
            "description": "Every printable part fits the machine's queried levelled build volume.",
            "threshold_source": "queried bed mesh + Z axis; 0 mm margin",
        },
        {
            "name": "machine_nozzle_matches",
            "severity": "warn",
            "description": "The fitted nozzle is the one the design's wall arithmetic assumed.",
            "threshold_source": "extruder.nozzle_diameter; 0.001 mm",
        },
        {
            "name": "machine_material_feasible",
            "severity": "error",
            "description": (
                "The machine's heater ceilings and chamber capability can run the declared "
                "material."
            ),
            "threshold_source": "queried max_temps vs MATERIAL_REQUIREMENTS; 0 C margin",
        },
    ]


def run_machine_gates(
    design: Dict[str, Any],
    placement: Any,
    enclosure_parts: Any,
    measurements: Any,
    *,
    machine: Optional[Mapping[str, Any]] = None,
    base_url: str = "",
) -> GateReport:
    """Gate a design against the real machine, querying it read-only unless `machine` is given.

    Signature mirrors `gates.run_gates()`
    so all four compose: `GateReport(run_gates(...).results + run_machine_gates(...).results)`.
    `placement` is accepted for that symmetry and is not read \u2014 machine feasibility is a
    property of the printable parts and the machine, not of where components sit inside them.

    `machine` exists so a caller that already queried the printer does not query it twice, and
    so the tests can supply a machine that is off, ancient, or chamber-less without one.
    """
    del placement  # signature symmetry; see docstring
    facts: Mapping[str, Any] = machine if machine is not None else query_machine(base_url)
    checks = (
        ("machine_reachable", lambda: _gate_machine_reachable(facts), "error"),
        (
            "machine_fits_build_volume",
            lambda: _gate_machine_fits_build_volume(facts, enclosure_parts, measurements),
            "error",
        ),
        ("machine_nozzle_matches", lambda: _gate_machine_nozzle_matches(facts, design), "warn"),
        (
            "machine_material_feasible",
            lambda: _gate_machine_material_feasible(facts, design),
            "error",
        ),
    )
    results: List[GateResult] = []
    for name, check, severity in checks:
        try:
            result = check()
            if result.name != name:
                result = GateResult(
                    name, result.passed, result.value, result.threshold, result.message,
                    result.severity,
                )
        except Exception as exc:  # a crashed machine gate must fail, never silently pass
            result = _result(
                name, False, None, 0.0, "gate crashed: {}: {}".format(type(exc).__name__, exc),
                severity,
            )
        results.append(result)
    return GateReport(results)


def machine_gate_names() -> Sequence[str]:
    """The stable gate names this module contributes, for the pipeline and the MCP surface."""
    return _MACHINE_GATES


__all__ = [
    "MATERIAL_REQUIREMENTS",
    "machine_gate_catalog",
    "machine_gate_names",
    "material_family",
    "query_machine",
    "run_machine_gates",
]
