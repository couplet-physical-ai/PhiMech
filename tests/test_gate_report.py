"""Gate status vocabulary, report summary, reloaded-artifact gating and the boolean guard.

Runnable with pytest or `/usr/bin/python3 tests/test_gate_report.py`. Backends are stubbed so
these tests need no OpenSCAD or FreeCAD; the real-toolchain path is covered by the examples.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phimech.backends.base import BackendError, BBox, Measurement, RenderResult  # noqa: E402
from phimech.check import gates  # noqa: E402
from phimech.csg import ops  # noqa: E402
from phimech.synth import pipeline  # noqa: E402
from phimech.synth import report as report_module  # noqa: E402

BOX_IR = {
    "project": "gate-report-box",
    "title": "Two boxes, no enclosure",
    "components": [
        {"ref": "A", "name": "Block A", "source": {"kind": "box", "size": [20, 10, 5]}},
        {"ref": "B", "name": "Block B", "source": {"kind": "box", "size": [8, 8, 8]}},
    ],
    "outputs": {"formats": ["stl"]},
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _cube_triangles(size: float = 2.0):
    """A closed axis-aligned cube at the origin: 12 triangles, volume size**3."""
    s = float(size)
    a, b, c, d = (0.0, 0.0, 0.0), (s, 0.0, 0.0), (s, s, 0.0), (0.0, s, 0.0)
    e, f, g, h = (0.0, 0.0, s), (s, 0.0, s), (s, s, s), (0.0, s, s)
    return [
        (a, c, b), (a, d, c), (e, f, g), (e, g, h), (a, b, f), (a, f, e),
        (b, c, g), (b, g, f), (c, d, h), (c, h, g), (d, a, e), (d, e, h),
    ]


def _ascii_stl(triangles) -> str:
    lines = ["solid stub"]
    for first, second, third in triangles:
        lines += [
            " facet normal 0 0 0", "  outer loop",
            "   vertex {} {} {}".format(*first),
            "   vertex {} {} {}".format(*second),
            "   vertex {} {} {}".format(*third),
            "  endloop", " endfacet",
        ]
    lines.append("endsolid stub")
    return "\n".join(lines) + "\n"


class _StubBackend:
    """Backend-shaped: writes a real 2 mm cube STL but *reports* a 20 x 10 x 5 block of
    1000 mm3, so the reloaded file and the tree measurement are told apart by their numbers."""

    name = "stub"
    supported_formats = ("stl", "step", "scad", "png")

    def __init__(self, stl_text: str = _ascii_stl(_cube_triangles(2.0))) -> None:
        self.stl_text = stl_text
        self.measured = []

    def available(self) -> bool:
        return True

    def supports(self, fmt) -> bool:
        return str(fmt).lower().lstrip(".") in self.supported_formats

    def _measurement(self) -> Measurement:
        return Measurement(
            bbox=BBox(min=(0.0, 0.0, 0.0), max=(20.0, 10.0, 5.0)),
            volume=1000.0, area=700.0, solids=1,
        )

    def render(self, node, out_path, fmt=None, *, timeout=600.0):
        resolved = str(fmt or Path(out_path).suffix.lstrip(".")).lower()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.stl_text if resolved == "stl" else "", encoding="utf-8")
        return RenderResult(path=path, fmt=resolved, backend=self.name, seconds=0.01,
                            bytes=path.stat().st_size)

    def measure(self, node, *, timeout=600.0):
        self.measured.append(node)
        return self._measurement()


class _TimingOut(_StubBackend):
    """What a wedged kernel looks like from the pipeline: every call raises a timeout."""

    def render(self, node, out_path, fmt=None, *, timeout=600.0):
        raise BackendError("openscad timed out after 0.01 s")

    def measure(self, node, *, timeout=600.0):
        raise BackendError("openscad timed out after 0.01 s")


def _build(tmp: Path, backend, design=BOX_IR, **kwargs):
    from phimech.backends import base

    original_for_format, original_get = base.for_format, base.get
    base.for_format = lambda fmt, prefer=("openscad", "freecad"): backend  # type: ignore
    base.get = lambda name: backend  # type: ignore[assignment]
    try:
        ir_path = tmp / "design.json"
        ir_path.write_text(json.dumps(design), encoding="utf-8")
        return pipeline.build(str(ir_path), tmp / "out", formats=["stl"], **kwargs)
    finally:
        base.for_format, base.get = original_for_format, original_get  # type: ignore


def _gate(report: dict, name: str) -> dict:
    for result in report["gates"]["results"]:
        if result["name"] == name:
            return result
    raise AssertionError("no gate {!r} in {}".format(
        name, [r["name"] for r in report["gates"]["results"]]))


def _objects_without_components():
    """A design and placement with nothing placed: the sketch-lane shape."""
    placement = SimpleNamespace(items=[], errors=[], keepouts=[])
    parts = SimpleNamespace(
        inner_cavity=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), wall=1.6, floor=1.6, cutouts=[],
        summary=lambda: {},
    )
    design = {
        "project": "freeform", "_errors": [],
        "manufacturing": {"min_wall": 1.2},
        "enclosure": {"wall": 1.6, "floor": 1.6, "lid": {"fastening": "snap"}},
    }
    return design, placement, parts


# --------------------------------------------------------------------------------------
# 1. the status vocabulary
# --------------------------------------------------------------------------------------


def test_every_gate_result_carries_a_status_from_the_vocabulary() -> None:
    design, placement, parts = _objects_without_components()
    report = gates.run_gates(design, placement, parts, {})
    payload = report.to_dict()
    assert payload["results"], "no gates ran"
    for entry in payload["results"]:
        assert entry["status"] in gates.STATUSES, entry
        # `passed` must stay the boolean projection of `status`, never something else.
        assert entry["passed"] is gates.passed_for_status(entry["status"]), entry
    json.dumps(payload)


def test_status_is_derived_for_results_built_positionally() -> None:
    """Machine gates and legacy fixtures construct results without a status."""
    assert gates.GateResult("g", True, 1, 1, "ok").effective_status == "pass"
    assert gates.GateResult("g", False, 0, 1, "too small").effective_status == "fail"
    assert gates.GateResult("g", None, None, 1, "skipped: no render").effective_status == "skipped"
    assert gates.GateResult("g", None, None, 1, "not applicable: no text").effective_status == (
        "not_applicable"
    )
    crashed = gates.GateResult("g", False, None, 0, "gate crashed: KeyError('x')")
    assert crashed.effective_status == "undetermined"
    assert crashed.passed is False, "a crashed gate must still block"


def test_a_status_that_contradicts_passed_is_refused() -> None:
    try:
        gates.GateResult("g", True, 1, 1, "ok", status="undetermined")
    except ValueError as exc:
        assert "contradicts" in str(exc)
    else:
        raise AssertionError("status=undetermined with passed=True was accepted")


def test_component_gates_are_not_applicable_without_components() -> None:
    """A freeform part has nothing to overlap, clear or land on; saying PASS would be evidence
    the design never earned. `not_applicable` does not block, and is never a pass."""
    design, placement, parts = _objects_without_components()
    report = gates.run_gates(design, placement, parts, {})
    for name in (
        "placement_no_overlap", "clearance_positive", "min_feature", "screw_count_sane",
        "cutout_within_face", "standoff_within_cavity", "fastener_clears_components",
        "standoff_lands_on_hole", "text_legible",
    ):
        entry = _gate({"gates": report.to_dict()}, name)
        assert entry["status"] == "not_applicable", (name, entry["message"])
        assert entry["passed"] is None, (name, entry)
        assert entry["message"].startswith("not applicable"), entry["message"]
    summary = report.summary()
    assert summary["is_valid"] is True
    assert summary["counts"]["not_applicable"] >= 9
    assert summary["counts"]["pass"] < summary["counts"]["total"]


# --------------------------------------------------------------------------------------
# 2. the report-level summary
# --------------------------------------------------------------------------------------


def test_summary_names_the_first_blocking_failure() -> None:
    results = [
        gates.GateResult("text_legible", False, 2.0, 4.0, "too small", "warn"),
        gates.GateResult("wall_thickness", True, 2.0, 1.2, "ok"),
        gates.GateResult("manifold", False, 0, 1, "solids=3"),
        gates.GateResult("fits_bed", False, 400.0, 360.0, "too big"),
        gates.GateResult("overhang", None, None, 0.35, "skipped: no render", "warn"),
    ]
    summary = gates.GateReport(results).summary()
    assert summary["is_valid"] is False
    # The advisory failure comes first in order, but what *blocks* is what is named.
    assert summary["first_failure"] == "manifold"
    assert summary["undetermined_layer"] is None and summary["first_undetermined"] is None
    counts = summary["counts"]
    assert counts == {"pass": 1, "fail": 3, "skipped": 1, "not_applicable": 0,
                      "undetermined": 0, "total": 5}
    only_advisory = gates.GateReport(results[:2]).summary()
    assert only_advisory["is_valid"] is True
    assert only_advisory["first_failure"] == "text_legible"


def test_summary_works_on_a_legacy_report_dict_without_status() -> None:
    """report.json files written before this field existed still summarise correctly."""
    legacy = [
        {"name": "wall_thickness", "passed": True, "message": "wall=2", "severity": "error"},
        {"name": "manifold", "passed": False, "message": "solids=3", "severity": "error"},
        {"name": "fits_bed", "passed": None, "message": "skipped: no printer_bed",
         "severity": "error"},
    ]
    summary = gates.summarize(legacy)
    assert summary["first_failure"] == "manifold" and summary["is_valid"] is False
    assert summary["counts"]["skipped"] == 1 and summary["counts"]["pass"] == 1
    assert report_module.verdict_line({"ok": False, "gates": {"results": legacy}}) == (
        "FAIL \u2014 1 gate passed, 1 failed, 1 skipped"
    )


def test_summary_line_reports_the_new_words_only_when_present() -> None:
    plain = gates.GateReport([gates.GateResult("a", True, 1, 1, "ok")])
    assert plain.summary_line() == "PASS: 1 gates, 0 failed, 0 skipped"
    mixed = gates.GateReport([
        gates.GateResult("a", True, 1, 1, "ok"),
        gates.GateResult("b", None, None, 1, "not applicable: nothing"),
        gates.GateResult("c", False, None, 1, "gate crashed: boom"),
    ])
    assert mixed.summary_line() == (
        "FAIL: 3 gates, 0 failed, 0 skipped, 1 not applicable, 1 undetermined"
    )


# --------------------------------------------------------------------------------------
# 3. a timed-out backend is undetermined, never a pass
# --------------------------------------------------------------------------------------


def test_forced_timeout_yields_undetermined_and_invalid() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-gate-timeout-") as tmp:
        report = _build(Path(tmp), _TimingOut())
    if not any(stage["stage"] == "place" and stage["ok"] for stage in report["stages"]):
        return  # providers absent in this checkout: nothing to render, nothing to judge
    assert report["ok"] is False
    assert any("timed out" in error for error in report["errors"]), report["errors"]
    summary = report["gate_summary"]
    assert summary["is_valid"] is False
    assert summary["undetermined_layer"] == "render", summary
    assert summary["first_undetermined"] in gates.MEASUREMENT_GATES, summary
    assert summary["counts"]["undetermined"] >= len(gates.MEASUREMENT_GATES) - 1
    assert report["gates"]["summary"] == {k: v for k, v in summary.items() if k != "note"}
    for name in gates.MEASUREMENT_GATES:
        entry = _gate(report, name)
        if entry["status"] == "not_applicable":
            continue  # manifold does not apply to a multi-body bare assembly
        assert entry["status"] == "undetermined", (name, entry)
        assert entry["passed"] is False, "undetermined must never project to a pass"
        assert entry["layer"] == "render", entry
        assert "timed out" in entry["message"], entry["message"]
    # The legacy gate verdict agrees: a timeout blocks.
    assert report["gates"]["passed"] is False


def test_skipped_before_any_render_stays_skipped() -> None:
    """`synth check` never renders: measurement gates are skipped there, not undetermined."""
    with tempfile.TemporaryDirectory(prefix="phimech-gate-check-") as tmp:
        ir_path = Path(tmp) / "design.json"
        ir_path.write_text(json.dumps(BOX_IR), encoding="utf-8")
        report = pipeline.check(str(ir_path))
    if "gates" not in report:
        return
    entry = _gate(report, "part_volume_positive")
    assert entry["status"] == "skipped" and entry["passed"] is None, entry
    assert report["gate_summary"]["undetermined_layer"] is None


# --------------------------------------------------------------------------------------
# 4. the impossible boolean
# --------------------------------------------------------------------------------------


def test_an_impossible_cut_raises_with_the_numbers() -> None:
    target, cutter = ops.box(10, 10, 10), ops.box(2, 2, 2)
    seen = []

    def broken_kernel(node) -> float:
        # A "cut" that returns the base with the tool fused on: bigger than it started.
        seen.append(node.kind)
        return 1200.0 if node.kind == "difference" else 1000.0

    try:
        ops.safe_cut(target, cutter, measure=broken_kernel)
    except ops.ImpossibleBoolean as exc:
        assert exc.kind == "difference"
        assert exc.result_volume == 1200.0 and exc.reference_volume == 1000.0
        assert "1200" in str(exc) and "1000" in str(exc)
    else:
        raise AssertionError("a cut that grew the part was accepted")
    assert seen == ["difference", "cube"], seen


def test_a_sane_cut_passes_within_tolerance_and_a_shrunken_union_raises() -> None:
    node = ops.safe_cut(ops.box(10, 10, 10), ops.box(2, 2, 2),
                        measure=lambda n: 1004.0 if n.kind == "difference" else 1000.0)
    assert node.kind == "difference"
    verdict = ops.check_boolean_volume("difference", 992.0, base_volume=1000.0)
    assert verdict["ratio"] == 0.992 and verdict["limit"] == 1.005
    try:
        ops.safe_union([ops.box(1, 1, 1), ops.box(5, 5, 5)],
                       measure=lambda n: 20.0 if n.kind == "union" else 125.0)
    except ops.ImpossibleBoolean as exc:
        assert exc.kind == "union" and exc.reference_volume == 125.0
    else:
        raise AssertionError("a union smaller than its largest child was accepted")


def test_boolean_gate_reads_the_recorded_reference() -> None:
    def measurement(checks):
        return Measurement(
            bbox=BBox((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), volume=900.0, area=600.0,
            extra={"boolean_reference": {"backend": "stub", "checks": checks}},
        )

    sane = {"kind": "difference", "node": "part", "result_volume_mm3": 900.0,
            "reference_volume_mm3": 1000.0}
    grew = dict(sane, result_volume_mm3=1100.0)
    design, placement, parts = _objects_without_components()
    ok = gates.run_gates(design, placement, parts, {"p": measurement([sane])})
    bad = gates.run_gates(design, placement, parts, {"p": measurement([sane, grew])})
    assert _gate({"gates": ok.to_dict()}, "boolean_volume_sane")["status"] == "pass"
    failed = _gate({"gates": bad.to_dict()}, "boolean_volume_sane")
    assert failed["status"] == "fail" and failed["threshold"] == 1.005
    assert abs(failed["value"] - 1.1) < 1e-9, failed
    unmeasured = gates.run_gates(design, placement, parts, {"p": Measurement(
        bbox=BBox((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), volume=900.0, area=600.0,
        extra={"boolean_reference": {"backend": "stub", "checks": [], "error": "timed out"}},
    )})
    entry = _gate({"gates": unmeasured.to_dict()}, "boolean_volume_sane")
    assert entry["status"] == "not_applicable" and "unmeasured" in entry["message"], entry


# --------------------------------------------------------------------------------------
# 5. the gates judge the reloaded file, and the round-trip is reported
# --------------------------------------------------------------------------------------


def test_gates_read_the_reloaded_stl_not_the_backend_claim() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-gate-reload-") as tmp:
        stub = _StubBackend()
        report = _build(Path(tmp), stub)
        if not any(stage["stage"] == "place" and stage["ok"] for stage in report["stages"]):
            return
        measured = report["measurements"]["assembly"]
        # The stub *claimed* 1000 mm3; the file it wrote is a 2 mm cube.
        assert abs(measured["volume_mm3"] - 8.0) < 1e-9, measured
        assert measured["bbox"]["size"] == [2.0, 2.0, 2.0]
        assert measured["extra"]["tree"]["volume_mm3"] == 1000.0
        assert measured["extra"]["source"].startswith("reloaded ")
        assert abs(_gate(report, "part_volume_positive")["value"] - 8.0) < 1e-9
        roundtrip = report["export_roundtrip"]["assembly"]
        assert roundtrip["format"] == "stl" and roundtrip["artifact"].endswith(".stl")
        assert abs(roundtrip["volume_delta_mm3"] - (8.0 - 1000.0)) < 1e-9
        assert abs(roundtrip["bbox_delta_mm"] - 18.0) < 1e-9  # 20 mm claimed vs 2 mm on X
        assert roundtrip["faces_after"] == 12 and roundtrip["faces_before"] is None
        # A diagnostic, not a gate: a 99 % delta must not fail anything on its own.
        assert not any(
            "roundtrip" in r["name"] or "round_trip" in r["name"]
            for r in report["gates"]["results"]
        )
        # The union reference was measured by the same stub, once per boolean level.
        reference = measured["extra"]["boolean_reference"]
        assert reference["checks"][0]["kind"] == "union"
        assert reference["checks"][0]["result_volume_mm3"] == 1000.0  # tree vs tree, same kernel
        assert len(stub.measured) >= 2
        markdown = (Path(tmp) / "out" / "REPORT.md").read_text(encoding="utf-8")
        assert "## Export round-trip" in markdown and "| status |" in markdown
        assert "- valid: **" in markdown


def test_manifold_gate_sees_the_loader_output() -> None:
    """Monkeypatch the STL loader: a leaky mesh in the file must fail `manifold` even though
    the backend swore `solids=1`."""
    from phimech.check import mesh

    leaky = _cube_triangles(2.0)[:-1]
    original = mesh.read_stl
    mesh.read_stl = lambda path: list(leaky)  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory(prefix="phimech-gate-leaky-") as tmp:
            design = dict(BOX_IR, components=BOX_IR["components"][:1])  # one body: manifold applies
            design["enclosure"] = {"wall": 2.0, "floor": 2.0, "lid": {"fastening": "snap"}}
            report = _build(Path(tmp), _StubBackend(), design=design)
    finally:
        mesh.read_stl = original  # type: ignore[assignment]
    if not any(stage["stage"] == "place" and stage["ok"] for stage in report["stages"]):
        return
    manifold = _gate(report, "manifold")
    assert manifold["status"] == "fail", manifold
    assert "open_edges=3" in manifold["message"], manifold["message"]
    assert report["gate_summary"]["first_failure"] == "manifold"


def test_roundtrip_is_reported_for_every_build_and_reload_can_be_turned_off() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-gate-noreload-") as tmp:
        report = _build(Path(tmp), _StubBackend(), reload=False, boolean_check=False)
    if not any(stage["stage"] == "place" and stage["ok"] for stage in report["stages"]):
        return
    assert "export_roundtrip" in report and report["export_roundtrip"] == {}
    assert report["measurements"]["assembly"]["volume_mm3"] == 1000.0
    assert _gate(report, "boolean_volume_sane")["status"] == "not_applicable"


def test_unreadable_export_is_an_error_not_a_green_build() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-gate-empty-") as tmp:
        report = _build(Path(tmp), _StubBackend(stl_text=""))
    if not any(stage["stage"] == "place" and stage["ok"] for stage in report["stages"]):
        return
    assert report["ok"] is False
    assert any("could not be reloaded" in error for error in report["errors"]), report["errors"]
    assert "error" in report["export_roundtrip"]["assembly"]
    # The tree measurement is kept so the numbers stay visible.
    assert report["measurements"]["assembly"]["volume_mm3"] == 1000.0


# --------------------------------------------------------------------------------------
# 6. the viewer contract keeps working
# --------------------------------------------------------------------------------------


def test_viewer_manifest_still_reads_the_report() -> None:
    try:
        from phimech.viewer.server import build_manifest
    except ImportError:
        return
    with tempfile.TemporaryDirectory(prefix="phimech-gate-viewer-") as tmp:
        report = _build(Path(tmp), _StubBackend())
        if not any(stage["stage"] == "place" and stage["ok"] for stage in report["stages"]):
            return
        manifest = build_manifest(Path(tmp) / "out")
    assert manifest["gates"], "manifest lost its gates"
    by_name = {gate["name"]: gate for gate in manifest["gates"]}
    for entry in report["gates"]["results"]:
        assert by_name[entry["name"]]["passed"] is entry["passed"]
        assert by_name[entry["name"]]["passed"] in (True, False, None)
    summary = manifest["gates_summary"]
    counts = report["gate_summary"]["counts"]
    assert summary["total"] == counts["total"]
    assert summary["passed"] == counts["pass"]
    assert summary["failed"] == counts["fail"] + counts["undetermined"]
    assert summary["skipped"] == counts["skipped"] + counts["not_applicable"]


def test_every_report_carries_a_gate_summary() -> None:
    absent = pipeline.check("/nonexistent/never-here.json")
    summary = absent["gate_summary"]
    assert summary["is_valid"] is False and summary["first_failure"] is None
    assert summary["note"] == "gates did not run"
    with tempfile.TemporaryDirectory(prefix="phimech-gate-nogates-") as tmp:
        report = _build(Path(tmp), _StubBackend(), gates=False)
    assert report["gate_summary"]["note"] == "gates disabled"
    assert report["gate_summary"]["is_valid"] is False
    assert "first_failure" in report["gate_summary"]


def _run_without_pytest() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print("{} gate-report tests passed".format(len(tests)))


if __name__ == "__main__":
    _run_without_pytest()
