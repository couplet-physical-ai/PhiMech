"""Tests for the gates that judge a design against the real printer.

Runs both ways, like tests/test_print.py: ``/usr/bin/python3 tests/test_machine_gates.py`` and
under pytest.  Almost everything here runs with **no printer at all**, because the gates take
their machine facts as a plain dict: the fixtures below are the machine states that matter
(off, chamber-less, small-bed, wrong nozzle) and none of them can be produced on demand from
the one real printer on the bench.  The two tests that do touch the live machine skip when it
is unreachable, and they only ever read.

Nothing in this file starts a print, moves an axis, heats anything, loads filament, or writes
config, and `test_machine_module_has_no_print_control` proves it by scanning the module source.
"""

from __future__ import annotations

import ast
import os as _os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by the system-python runner
    pytest = None  # type: ignore[assignment]

from phimech.check import machine as machine_check
from phimech.check.machine import (
    MATERIAL_REQUIREMENTS,
    machine_gate_catalog,
    machine_gate_names,
    material_family,
    run_machine_gates,
)


class _Skipped(Exception):
    pass


def _skip(message: str):
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _gate(report, name):
    for result in report.results:
        if result.name == name:
            return result
    raise AssertionError(
        "no gate named {!r}; have {}".format(name, [r.name for r in report.results])
    )


# --------------------------------------------------------------------------------------
# machine fixtures: the states that matter, none of which the bench printer can be put into
# --------------------------------------------------------------------------------------


def _max4(**overrides):
    """The lab QIDI Max 4 as `printer_info()` actually reports it (verified live)."""
    facts = {
        "url": "http://printer.invalid:7125",
        "reachable": True,
        "klippy_state": "ready",
        "hostname": "qidi-max4",
        "build_volume_mm": [360.0, 360.0, 342.0],
        "build_volume_source": "queried",
        "bed_mesh": {"min": [15.0, 15.0], "max": [375.0, 375.0]},
        "nozzle_mm": 0.4,
        "heated_chamber": True,
        "max_temps": {
            "extruder_c": 375.0,
            "min_extrude_c": 135.0,
            "bed_c": 125.0,
            "chamber_c": 68.0,
        },
        "error": None,
    }
    facts.update(overrides)
    return facts


def _offline():
    """What `printer_info()` returns for a printer that did not answer.

    Note `build_volume_mm` is still populated \u2014 that is documented behaviour so a build can be
    gated with the printer off \u2014 which is exactly the trap `build_volume_source` exists to close.
    """
    return {
        "url": "http://printer.invalid:7125",
        "reachable": False,
        "build_volume_mm": [360.0, 360.0, 340.0],
        "build_volume_source": "documented",
        "nozzle_mm": None,
        "max_temps": {},
        "error": "[Errno 61] Connection refused",
    }


def _parts(size=(67.0, 88.4, 24.1)):
    """An enclosure-shaped object exposing only what the size gate reads."""
    return SimpleNamespace(outer_size=tuple(float(v) for v in size))


def _design(**manufacturing):
    return {"project": "machine-test", "manufacturing": dict(manufacturing)}


# --------------------------------------------------------------------------------------
# material family resolution
# --------------------------------------------------------------------------------------


def test_material_family_resolves_vendor_and_filled_names() -> None:
    assert material_family("PLA") == "pla"
    assert material_family("QIDI ABS Rapido Metal") == "abs"
    assert material_family("ASA-CF") == "asa"
    assert material_family("PAHT-CF") == "pa"
    assert material_family("PPS-GF") == "pps"


def test_petg_does_not_resolve_to_pet() -> None:
    """Longest-first matching: both `petg` and `pet` are families and PETG must win.

    Getting this backwards would gate every PETG design against PET's 290 C nozzle and 55 C
    chamber requirement, i.e. would declare PETG unprintable without a heated chamber.
    """
    assert material_family("PETG") == "petg"
    assert material_family("PETG-CF") == "petg"
    assert material_family("QIDI PETG Tough") == "petg"
    assert material_family("PET-CF") == "pet"


def test_unknown_material_is_none_not_a_guess() -> None:
    assert material_family("unobtainium") is None
    assert material_family("") is None
    assert material_family(None) is None


def test_chamber_requirements_are_set_for_the_warping_families() -> None:
    """The load-bearing content of the table: which families need a heated chamber."""
    for family in ("abs", "asa", "pc", "pa", "pps"):
        assert MATERIAL_REQUIREMENTS[family]["chamber_c"], family
    for family in ("pla", "petg", "tpu"):
        assert MATERIAL_REQUIREMENTS[family]["chamber_c"] is None, family


# --------------------------------------------------------------------------------------
# fail closed
# --------------------------------------------------------------------------------------


def test_unreachable_printer_fails_every_gate_it_cannot_answer() -> None:
    """An unreachable printer is `unknown`, and `unknown` FAILS. It must not skip.

    This is the opposite of the render-dependent gate convention (`skipped: no render` ->
    `passed=None` -> does not block) and the difference is the point: a missing render means a
    question was not asked yet, a missing printer means it was asked and could not be answered.
    """
    report = run_machine_gates(
        _design(nozzle=0.4, material="ABS"), None, _parts(), {}, machine=_offline()
    )
    assert report.passed is False
    for name in machine_gate_names():
        gate = _gate(report, name)
        assert gate.passed is False, (name, gate.message)
        assert "unknown" in gate.message, (name, gate.message)


def test_unreachable_printer_never_gates_against_the_documented_envelope() -> None:
    """`printer_info()` still fills in `build_volume_mm` when the printer is off.

    A gate reading that field alone would report a compiled-in constant as a machine reading and
    happily PASS a part that fits it, so the size gate must refuse on provenance. The part here
    fits the documented 360 x 360 x 340 envelope comfortably.
    """
    facts = _offline()
    assert facts["build_volume_mm"] == [360.0, 360.0, 340.0], "fixture must keep the trap intact"
    gate = _gate(
        run_machine_gates(_design(), None, _parts((50.0, 50.0, 50.0)), {}, machine=facts),
        "machine_fits_build_volume",
    )
    assert gate.passed is False, gate.message
    assert "compiled-in" in gate.message, gate.message


def test_reachable_printer_with_no_bed_mesh_is_also_unknown() -> None:
    """Reachable is not enough: the envelope has to be the machine's own.

    A printer that answers `/printer/info` but exposes no bed mesh leaves the documented
    constant in place, and that must fail exactly like an unreachable one.
    """
    facts = _max4(build_volume_source="documented")
    report = run_machine_gates(_design(), None, _parts(), {}, machine=facts)
    reachable = _gate(report, "machine_reachable")
    assert reachable.passed is False, reachable.message
    assert "compiled-in constant" in reachable.message, reachable.message
    assert _gate(report, "machine_fits_build_volume").passed is False


# --------------------------------------------------------------------------------------
# build volume
# --------------------------------------------------------------------------------------


def test_part_that_fits_the_queried_envelope_passes() -> None:
    report = run_machine_gates(_design(nozzle=0.4), None, _parts(), {}, machine=_max4())
    gate = _gate(report, "machine_fits_build_volume")
    assert gate.passed is True, gate.message
    assert gate.value > 200.0, gate.message
    assert "source=queried" in gate.message, gate.message


def test_part_too_big_for_the_queried_envelope_fails_with_the_numbers() -> None:
    gate = _gate(
        run_machine_gates(_design(), None, _parts((400.0, 100.0, 50.0)), {}, machine=_max4()),
        "machine_fits_build_volume",
    )
    assert gate.passed is False, gate.message
    assert gate.value < 0.0, gate.message
    assert "400" in gate.message and "360" in gate.message, (
        "the message must name both numbers: {}".format(gate.message)
    )


def test_part_too_tall_for_the_queried_envelope_fails() -> None:
    gate = _gate(
        run_machine_gates(_design(), None, _parts((100.0, 100.0, 400.0)), {}, machine=_max4()),
        "machine_fits_build_volume",
    )
    assert gate.passed is False, gate.message
    assert "too tall" in gate.message, gate.message


def test_a_part_that_only_fits_rotated_is_accepted_on_a_rectangular_bed() -> None:
    """Delegating to `printer.fits_build_volume()` must preserve its 90-degrees-about-Z retry.

    Proven on a rectangular envelope because a rotation cannot change the verdict on the Max 4's
    square bed, which is the same reasoning tests/test_print.py uses.
    """
    facts = _max4(build_volume_mm=[300.0, 400.0, 340.0])
    gate = _gate(
        run_machine_gates(_design(), None, _parts((380.0, 250.0, 50.0)), {}, machine=facts),
        "machine_fits_build_volume",
    )
    assert gate.passed is True, gate.message
    assert gate.value >= 0.0, gate.message


def test_size_gate_prefers_a_render_over_the_synth_bbox() -> None:
    """A rendered measurement is better evidence than a bbox, so it must win when present.

    The enclosure bbox here fits and the rendered part does not; the gate has to report the
    render's verdict, otherwise a real oversized STL would pass behind a small bbox.
    """
    from phimech.backends.base import BBox, Measurement

    measurement = Measurement(
        bbox=BBox((0.0, 0.0, 0.0), (400.0, 100.0, 50.0)), volume=1000.0, area=100.0
    )
    gate = _gate(
        run_machine_gates(
            _design(), None, _parts((50.0, 50.0, 50.0)), {"base": measurement}, machine=_max4()
        ),
        "machine_fits_build_volume",
    )
    assert gate.passed is False, gate.message
    assert "basis=render" in gate.message, gate.message


def test_size_gate_reads_a_measurement_re_read_from_a_report() -> None:
    """A measurement from `report.json` is a dict, not a Measurement; both must work.

    The repo has already been bitten by exactly this (see `_measurement_stl` in gates.py, where
    plain `getattr` made every mesh-backed gate silently skip when driven from a saved report).
    """
    as_dict = {"bbox": {"min": [0.0, 0.0, 0.0], "max": [400.0, 100.0, 50.0]}, "volume_mm3": 1.0}
    gate = _gate(
        run_machine_gates(_design(), None, None, {"base": as_dict}, machine=_max4()),
        "machine_fits_build_volume",
    )
    assert gate.passed is False, gate.message
    assert "basis=render" in gate.message, gate.message


def test_no_geometry_at_all_is_unknown_not_pass() -> None:
    gate = _gate(
        run_machine_gates(_design(), None, None, {}, machine=_max4()),
        "machine_fits_build_volume",
    )
    assert gate.passed is False, gate.message
    assert "unknown" in gate.message, gate.message


# --------------------------------------------------------------------------------------
# nozzle
# --------------------------------------------------------------------------------------


def test_nozzle_mismatch_warns_with_both_diameters() -> None:
    gate = _gate(
        run_machine_gates(_design(nozzle=0.4), None, _parts(), {}, machine=_max4(nozzle_mm=0.6)),
        "machine_nozzle_matches",
    )
    assert gate.passed is False, gate.message
    assert gate.severity == "warn", "a printable-but-different nozzle must not block"
    assert "0.4" in gate.message and "0.6" in gate.message, gate.message


def test_nozzle_mismatch_does_not_block_the_report() -> None:
    report = run_machine_gates(
        _design(nozzle=0.4, material="PLA"), None, _parts(), {}, machine=_max4(nozzle_mm=0.6)
    )
    assert _gate(report, "machine_nozzle_matches").passed is False
    assert report.passed is True, "only warn severity, so the report must still pass"


def test_matching_nozzle_passes() -> None:
    gate = _gate(
        run_machine_gates(_design(nozzle=0.4), None, _parts(), {}, machine=_max4()),
        "machine_nozzle_matches",
    )
    assert gate.passed is True, gate.message


# --------------------------------------------------------------------------------------
# material feasibility
# --------------------------------------------------------------------------------------


def test_abs_is_feasible_on_the_heated_chamber_machine() -> None:
    gate = _gate(
        run_machine_gates(_design(material="ABS"), None, _parts(), {}, machine=_max4()),
        "machine_material_feasible",
    )
    assert gate.passed is True, gate.message
    assert "chamber=50 C" in gate.message, gate.message
    assert "not measured on this machine" in gate.message, (
        "the message must disclose that the requirement numbers are published-typical: "
        "{}".format(gate.message)
    )


def test_abs_fails_on_a_machine_with_no_heated_chamber() -> None:
    """The fact this whole gate exists for. ABS warps without a chamber.

    The machine here can reach ABS's nozzle and bed temperatures, so only the chamber
    requirement can fail it \u2014 which is the case a temperature-ceiling-only check would miss.
    """
    facts = _max4(
        heated_chamber=False,
        max_temps={"extruder_c": 300.0, "bed_c": 110.0},
        hostname="chamberless",
    )
    gate = _gate(
        run_machine_gates(_design(material="ABS"), None, _parts(), {}, machine=facts),
        "machine_material_feasible",
    )
    assert gate.passed is False, gate.message
    assert "heated chamber" in gate.message, gate.message
    assert gate.value < 0.0, gate.message


def test_pla_passes_on_the_same_chamberless_machine() -> None:
    """The other half: the chamber rule must not condemn a material that never needed one."""
    facts = _max4(heated_chamber=False, max_temps={"extruder_c": 260.0, "bed_c": 100.0})
    gate = _gate(
        run_machine_gates(_design(material="PLA"), None, _parts(), {}, machine=facts),
        "machine_material_feasible",
    )
    assert gate.passed is True, gate.message
    assert "none required" in gate.message, gate.message


def test_material_needing_more_nozzle_heat_than_the_machine_has_fails() -> None:
    facts = _max4(max_temps={"extruder_c": 260.0, "bed_c": 125.0, "chamber_c": 68.0})
    gate = _gate(
        run_machine_gates(_design(material="PPS-CF"), None, _parts(), {}, machine=facts),
        "machine_material_feasible",
    )
    assert gate.passed is False, gate.message
    assert "nozzle needs" in gate.message, gate.message


def test_unknown_material_fails_rather_than_assuming_it_prints() -> None:
    gate = _gate(
        run_machine_gates(_design(material="unobtainium"), None, _parts(), {}, machine=_max4()),
        "machine_material_feasible",
    )
    assert gate.passed is False, gate.message
    assert "unobtainium" in gate.message, gate.message


def test_no_declared_material_is_skipped_not_failed() -> None:
    """Nothing was claimed, so nothing is judged \u2014 the one honest skip in this module."""
    gate = _gate(
        run_machine_gates(_design(nozzle=0.4), None, _parts(), {}, machine=_max4()),
        "machine_material_feasible",
    )
    assert gate.passed is None, gate.message
    assert gate.message.startswith("skipped:"), gate.message


# --------------------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------------------


def test_catalog_matches_the_gates_that_actually_run() -> None:
    catalog = [entry["name"] for entry in machine_gate_catalog()]
    executed = [
        result.name
        for result in run_machine_gates(_design(), None, _parts(), {}, machine=_max4()).results
    ]
    assert catalog == executed, (catalog, executed)
    assert catalog == list(machine_gate_names())


def test_a_crashing_gate_fails_rather_than_vanishing() -> None:
    """A machine gate that raises must FAIL. Swallowing it would turn unknown into PASS."""

    class Exploding(dict):
        def get(self, key, default=None):  # noqa: D102
            raise RuntimeError("boom")

    report = run_machine_gates(_design(material="ABS"), None, _parts(), {}, machine=Exploding())
    assert report.passed is False
    assert any("gate crashed" in result.message for result in report.results), [
        result.message for result in report.results
    ]


def test_gates_compose_with_the_standard_gate_report() -> None:
    """The documented composition: `GateReport(run_gates(...) + run_machine_gates(...))`."""
    from phimech.check.gates import GateReport

    machine_report = run_machine_gates(_design(), None, _parts(), {}, machine=_max4())
    combined = GateReport(list(machine_report.results))
    assert combined.summary_line().startswith(("PASS", "FAIL"))
    assert len(combined.results) == len(machine_gate_names())


# --------------------------------------------------------------------------------------
# safety: this module cannot move the machine either
# --------------------------------------------------------------------------------------


def test_machine_module_has_no_print_control() -> None:
    """The same source scan tests/test_print.py runs over printer.py, run over machine.py.

    The gate module is a new place a start-print call could be added \u201cfor convenience\u201d, and it
    is the module most tempted to do so, since it already talks to the printer.
    """
    from tests.test_print import _FORBIDDEN_NAMES, _FORBIDDEN_TOKENS, _code_strings

    source = Path(machine_check.__file__).read_text(encoding="utf-8")
    for value in _code_strings(source):
        for token in _FORBIDDEN_TOKENS:
            assert token not in value, "print-control token {!r} in {!r}".format(token, value)
    for name in [n for n in dir(machine_check) if not n.startswith("_")]:
        assert name.lower() not in _FORBIDDEN_NAMES, "{} must not exist in machine.py".format(name)
    for verb in ('method="POST"', 'method="DELETE"', 'method="PUT"', 'method="PATCH"'):
        assert verb not in source, "{} must not appear: this module only reads".format(verb)


def test_machine_module_reaches_the_printer_only_through_printer_info() -> None:
    """One door to the machine, and it is the module carrying the no-print-control contract.

    Asserted structurally rather than by eye: any other attribute pulled off `phimech.print.printer`
    is a new capability that has not been through that contract's source scan.
    """
    source = Path(machine_check.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "printer"
    }
    assert used <= {"printer_info", "fits_build_volume"}, (
        "machine.py may only call read-only printer helpers, found {}".format(sorted(used))
    )


# --------------------------------------------------------------------------------------
# live machine: read-only, skipped when it is not on the bench
# --------------------------------------------------------------------------------------


def test_live_printer_supplies_a_queried_envelope() -> None:
    facts = machine_check.query_machine()
    if not facts.get("reachable"):
        _skip("printer unreachable: {}".format(facts.get("error")))
    assert facts["build_volume_source"] == "queried", (
        "a reachable printer must supply its own envelope, got {}".format(facts)
    )
    volume = facts["build_volume_mm"]
    assert all(value > 0.0 for value in volume), volume
    # The mesh area, not the axis maxima: X reaches 392 and Y 410 on the Max 4, but a part out
    # there is printed on unmeasured bed.
    mesh = facts["bed_mesh"]
    assert abs(volume[0] - (mesh["max"][0] - mesh["min"][0])) < 1e-6, (volume, mesh)
    assert abs(volume[1] - (mesh["max"][1] - mesh["min"][1])) < 1e-6, (volume, mesh)


def test_live_printer_gates_a_real_design() -> None:
    facts = machine_check.query_machine()
    if not facts.get("reachable"):
        _skip("printer unreachable: {}".format(facts.get("error")))
    report = run_machine_gates(
        _design(nozzle=facts.get("nozzle_mm") or 0.4), None, _parts(), {}, machine=facts
    )
    assert _gate(report, "machine_reachable").passed is True
    assert _gate(report, "machine_fits_build_volume").passed is True
    # And an absurd part must fail against the same live facts.
    oversized = run_machine_gates(
        _design(), None, _parts((900.0, 900.0, 900.0)), {}, machine=facts
    )
    assert _gate(oversized, "machine_fits_build_volume").passed is False


def _run_direct() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(test.__name__, exc))
        except Exception as exc:  # pragma: no cover - only used by direct invocation
            failures += 1
            print("FAIL {}: {}".format(test.__name__, exc))
        else:
            print("PASS {}".format(test.__name__))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
