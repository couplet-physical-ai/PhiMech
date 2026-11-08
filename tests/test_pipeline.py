"""Tests for the synth pipeline orchestrator and the report writer.

Runnable both ways: `pytest tests/test_pipeline.py` and `/usr/bin/python3 tests/test_pipeline.py`.
The pipeline depends on nearly every other slice, so each test states which slices it needs and
degrades to asserting the *fault-tolerance contract* (a report dict, never an exception) when a
slice is missing from the checkout.
"""

from __future__ import annotations

import os as _os

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phimech.synth import pipeline, report as report_module  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


BOX_IR = {
    "project": "unit-box",
    "title": "One box, no enclosure",
    "components": [
        {"ref": "BOX1", "name": "Block", "source": {"kind": "box", "size": [20, 10, 5]}}
    ],
    "outputs": {"formats": ["stl"]},
}


def _write_ir(directory: Path, design: dict, name: str = "design.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    return path


def _stage(report: dict, name: str) -> dict:
    for record in report["stages"]:
        if record["stage"] == name:
            return record
    raise AssertionError("no stage {!r} in {}".format(name, [s["stage"] for s in report["stages"]]))


def _report_shape(report: dict) -> None:
    """Every report — successful, partial, or wholly failed — must have this shape."""
    assert isinstance(report, dict)
    for key in ("ok", "project", "stages", "errors", "elapsed_s"):
        assert key in report, "report is missing {!r}".format(key)
    assert [record["stage"] for record in report["stages"]] == pipeline.STAGES
    assert isinstance(report["ok"], bool)


# --------------------------------------------------------------------------------------
# 1. a missing file must not raise
# --------------------------------------------------------------------------------------


def test_check_missing_file_reports_load_error() -> None:
    report = pipeline.check("/nonexistent/definitely-not-here.json")
    _report_shape(report)
    assert report["ok"] is False
    load = _stage(report, "load")
    assert load["ok"] is False
    assert "error" in load and load["error"]
    assert isinstance(load["traceback"], list) and 1 <= len(load["traceback"]) <= 5
    assert any(message.startswith("load:") for message in report["errors"])


# --------------------------------------------------------------------------------------
# 2. a minimal inline IR runs as far as the checkout allows
# --------------------------------------------------------------------------------------


def test_check_minimal_box_ir() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-pipeline-") as tmp:
        path = _write_ir(Path(tmp), BOX_IR)
        report = pipeline.check(str(path))
    _report_shape(report)
    assert report["project"] == "unit-box"
    assert _stage(report, "load")["ok"] is True
    envelopes = _stage(report, "envelopes")
    if not envelopes["ok"]:  # providers slice absent: the error must be recorded, not raised
        assert "error" in envelopes
        return
    place = _stage(report, "place")
    if place["ok"]:
        assert report["placement"]["items"][0]["ref"] == "BOX1"
        assert report["components"][0]["size"] == [20.0, 10.0, 5.0]
    else:
        assert "error" in place
    # A design with no `enclosure` key skips that stage rather than failing it.
    enclosure = _stage(report, "enclosure")
    assert enclosure.get("skipped") is True
    assert "enclosure" not in report


def test_check_auto_arranges_when_placement_is_absent() -> None:
    design = {
        "project": "unit-auto",
        "components": [
            {"ref": "A", "source": {"kind": "box", "size": [10, 10, 10]}},
            {"ref": "B", "source": {"kind": "box", "size": [5, 5, 5]}},
        ],
    }
    with tempfile.TemporaryDirectory(prefix="phimech-pipeline-") as tmp:
        report = pipeline.check(str(_write_ir(Path(tmp), design)))
    _report_shape(report)
    place = _stage(report, "place")
    if place["ok"]:
        assert place.get("auto_arranged") is True
        assert any("auto-arranged" in note for note in report["notes"])


# --------------------------------------------------------------------------------------
# 3. the report writer stands alone
# --------------------------------------------------------------------------------------


SYNTHETIC = {
    "ok": False,
    "project": "synthetic",
    "title": "Synthetic report",
    "rev": "B",
    "company": "Example Co",
    "phimech_version": "0.1.0",
    "ir_path": "/tmp/synthetic.json",
    "out_dir": "/tmp/synthetic-out",
    "elapsed_s": 1.25,
    "stages": [
        {"stage": name, "ok": name != "render", "seconds": 0.01, "note": "note for " + name}
        for name in pipeline.STAGES
    ],
    "components": [
        {
            "ref": "PCB1",
            "name": "Board",
            "source": "kicad_pcb",
            "size": [60, 40, 12],
            "position": [0, 0, 0],
            "rotation": [0, 0, 90],
        }
    ],
    "enclosure": {
        "style": "tray_lid",
        "parts": ["base", "lid"],
        "outer_size_mm": [66, 46, 20],
        "cavity_size_mm": [62, 42, 16],
        "wall_mm": 2.0,
        "floor_mm": 2.0,
        "fastening": "screws",
        "standoffs": 4,
        "cutouts": 2,
        "screws": 4,
        "estimated_material_mm3": 31892.0,
        "notes": ["label debossed on the lid"],
    },
    "outputs": [
        {
            "part": "base",
            "fmt": "stl",
            "path": "/tmp/synthetic-out/synthetic-base.stl",
            "bytes": 2048,
            "backend": "openscad",
            "seconds": 0.2,
        }
    ],
    "measurements": {
        "base": {
            "bbox": {"min": [0, 0, 0], "max": [66, 46, 20], "size": [66, 46, 20]},
            "volume_mm3": 12345.0,
            "area_mm2": 6789.0,
            "solids": 1,
        }
    },
    "gates": {
        "passed": False,
        "results": [
            {"name": "wall_thickness", "passed": True, "value": 2.0, "threshold": 1.2,
             "message": "wall=2", "severity": "error"},
            {"name": "manifold", "passed": False, "value": 0, "threshold": 1,
             "message": "solids=3", "severity": "error"},
            {"name": "fits_bed", "passed": None, "value": None, "threshold": 0.0,
             "message": "skipped: no printer_bed", "severity": "error"},
        ],
    },
    "errors": ["render: base .step: BackendError: freecadcmd not found"],
    "notes": ["placement was auto-arranged"],
    "command": "bin/phimech synth build /tmp/synthetic.json -o /tmp/synthetic-out",
}


def test_write_report_writes_both_files() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-report-") as tmp:
        out = Path(tmp)
        json_path = report_module.write_report(out, dict(SYNTHETIC))
        assert json_path == out / "report.json"
        assert json_path.is_file()
        markdown = (out / "REPORT.md").read_text(encoding="utf-8")
        round_tripped = json.loads(json_path.read_text(encoding="utf-8"))

    assert round_tripped["project"] == "synthetic"
    assert round_tripped["gates"]["results"][1]["name"] == "manifold"
    assert round_tripped["timestamp"]

    assert "FAIL \u2014 1 gate passed, 1 failed, 1 skipped, 1 stage error" in markdown
    for stage in pipeline.STAGES:
        assert stage in markdown, "REPORT.md never mentions stage {!r}".format(stage)
    for expected in ("PCB1", "tray_lid", "synthetic-base.stl", "manifold", "How to reproduce",
                     "bin/phimech synth build"):
        assert expected in markdown
    assert "\u2717" in markdown and "\u2713" in markdown and "\u2013" in markdown


def test_verdict_and_summary_lines() -> None:
    assert report_module.verdict_line(SYNTHETIC).startswith("FAIL \u2014 ")
    assert report_module.verdict_line({"ok": True, "gates": {"results": []}}).startswith("PASS")
    summary = report_module.summary_line(SYNTHETIC)
    assert summary.startswith("FAIL synthetic")
    assert "\n" not in summary and len(summary) < 200


def test_render_markdown_survives_a_nearly_empty_report() -> None:
    markdown = report_module.render_markdown({"project": "x", "stages": [], "errors": []})
    assert "PhiMech report" in markdown
    assert "_none_" in markdown


# --------------------------------------------------------------------------------------
# 4. build() with a stub backend
# --------------------------------------------------------------------------------------


class _StubBackend:
    """A Backend-shaped stub: writes a tiny file and reports a known measurement."""

    name = "stub"
    supported_formats = ("stl", "step", "scad", "png", "3mf")
    UNSUPPORTED = frozenset()

    def __init__(self) -> None:
        self.calls = []

    def available(self) -> bool:
        return True

    def version(self):
        return "stub-1"

    def supports(self, fmt) -> bool:
        return str(fmt).lower().lstrip(".") in self.supported_formats

    def _measurement(self):
        from phimech.backends.base import BBox, Measurement

        return Measurement(bbox=BBox(min=(0.0, 0.0, 0.0), max=(20.0, 10.0, 5.0)),
                           volume=1000.0, area=700.0, solids=1)

    def render(self, node, out_path, fmt=None, *, timeout=600.0):
        from phimech.backends.base import RenderResult

        resolved = str(fmt or Path(out_path).suffix.lstrip(".")).lower()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        self.calls.append((resolved, path.name))
        return RenderResult(path=path, fmt=resolved, backend=self.name, seconds=0.01,
                            bytes=0, measurement=self._measurement())

    def measure(self, node, *, timeout=600.0):
        return self._measurement()


def _build_with_stub(tmp: Path, design: dict, formats):
    """Run build() with every format routed to the stub backend."""
    from phimech.backends import base

    stub = _StubBackend()
    original_for_format = base.for_format
    original_get = base.get
    base.for_format = lambda fmt, prefer=("openscad", "freecad"): stub  # type: ignore[assignment]
    base.get = lambda name: stub  # type: ignore[assignment]
    try:
        ir_path = _write_ir(tmp, design, "stub-design.json")
        out_dir = tmp / "out"
        return stub, pipeline.build(str(ir_path), out_dir, formats=formats, backend="auto")
    finally:
        base.for_format = original_for_format  # type: ignore[assignment]
        base.get = original_get  # type: ignore[assignment]


def test_build_with_stub_backend() -> None:
    with tempfile.TemporaryDirectory(prefix="phimech-pipeline-build-") as tmp:
        directory = Path(tmp)
        stub, report = _build_with_stub(directory, BOX_IR, ["stl", "scad"])
        out_dir = directory / "out"

        _report_shape(report)
        if not _stage(report, "place")["ok"]:  # place/providers absent: contract still holds
            assert report["ok"] is False
            return

        assert report["outputs"], "no outputs were recorded"
        by_fmt = {output["fmt"]: output for output in report["outputs"]}
        assert set(by_fmt) == {"stl", "scad"}
        for output in report["outputs"]:
            assert output["part"] == "assembly"
            assert Path(output["path"]).is_file()
            assert output["backend"] == "stub"

        ast_path = out_dir / "unit-box-assembly.ast.json"
        assert ast_path.is_file()
        from phimech.csg.ast import from_dict

        node = from_dict(json.loads(ast_path.read_text(encoding="utf-8")))
        assert node.kind == "group"

        assert (out_dir / "report.json").is_file()
        assert (out_dir / "REPORT.md").is_file()
        assert (out_dir / "design.json").is_file()
        # The pipeline resolves out_dir (symlinked /tmp on macOS), so compare resolved paths.
        assert Path(report["report_path"]).resolve() == (out_dir / "report.json").resolve()
        assert report["measurements"]["assembly"]["volume_mm3"] == 1000.0
        assert report["measurements"]["assembly"]["extra"]["stl"].endswith(".stl")

        gates = report.get("gates")
        if gates is not None:
            assert report["ok"] is bool(gates["passed"]) or report["errors"]
        assert stub.calls


def test_build_records_render_failure_without_raising() -> None:
    class _Exploding(_StubBackend):
        def render(self, node, out_path, fmt=None, *, timeout=600.0):
            raise RuntimeError("stub refuses to render")

    from phimech.backends import base

    exploding = _Exploding()
    original = base.for_format
    base.for_format = lambda fmt, prefer=("openscad", "freecad"): exploding  # type: ignore
    try:
        with tempfile.TemporaryDirectory(prefix="phimech-pipeline-fail-") as tmp:
            directory = Path(tmp)
            out_dir = directory / "out"
            report = pipeline.build(str(_write_ir(directory, BOX_IR)), out_dir, formats=["stl"])
            _report_shape(report)
            assert report["ok"] is False
            # A failed render still writes the report: that is how the agent learns why.
            assert (out_dir / "REPORT.md").is_file()
            if _stage(report, "place")["ok"]:
                assert any("stub refuses to render" in msg for msg in report["errors"])
                assert _stage(report, "render")["ok"] is False
    finally:
        base.for_format = original  # type: ignore[assignment]


def test_scad_is_emitted_without_any_usable_backend() -> None:
    """`.scad` needs no binary (`scad_emit` is pure text), so it survives a bare machine."""
    from phimech.backends import base

    base._ensure_loaded()
    backends = base.all_backends()
    originals = [(backend, backend.available) for backend in backends]
    for backend in backends:
        backend.available = lambda: False  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory(prefix="phimech-pipeline-scad-") as tmp:
            directory = Path(tmp)
            out_dir = directory / "out"
            report = pipeline.build(
                str(_write_ir(directory, BOX_IR)), out_dir, formats=["scad", "stl"]
            )
            _report_shape(report)
            if not _stage(report, "place")["ok"]:
                return
            scad = out_dir / "unit-box-assembly.scad"
            assert scad.is_file() and scad.stat().st_size > 0
            assert [output["fmt"] for output in report["outputs"]] == ["scad"]
            assert any(".stl" in note for note in report["notes"])
    finally:
        for backend, available in originals:
            backend.available = available  # type: ignore[assignment]


def test_step_import_forces_the_freecad_route() -> None:
    """OpenSCAD cannot read STEP, so a part containing such an import must not be routed there."""
    from phimech.csg.ast import Group, Import, Translate

    step_node = Group([Translate((1.0, 0.0, 0.0), [Import("/tmp/vendor-part.step")])])
    stl_node = Group([Import("/tmp/vendor-part.stl")])
    assert pipeline._has_step_import(step_node) is True
    assert pipeline._has_step_import(stl_node) is False
    assert pipeline._prefer_order("openscad", "stl", True)[0] == "freecad"
    assert pipeline._prefer_order("auto", "step", False)[0] == "freecad"
    assert pipeline._prefer_order("auto", "png", False)[0] == "openscad"


# --------------------------------------------------------------------------------------
# 5. the real OpenSCAD path, when the binary is present
# --------------------------------------------------------------------------------------


def _openscad_available() -> bool:
    try:
        from phimech.backends.openscad import OpenSCADBackend
    except ImportError:
        return False
    return OpenSCADBackend().available()


def test_build_box_for_real_with_openscad() -> None:
    if not _openscad_available():
        if _os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import pytest

                pytest.skip("openscad backend unavailable")
            except ImportError:
                pass
        print("skip: openscad backend unavailable")
        return
    with tempfile.TemporaryDirectory(prefix="phimech-pipeline-real-") as tmp:
        directory = Path(tmp)
        out_dir = directory / "out"
        report = pipeline.build(
            str(_write_ir(directory, BOX_IR)), out_dir, formats=["stl", "scad"], backend="openscad"
        )
        _report_shape(report)
        stl = out_dir / "unit-box-assembly.stl"
        scad = out_dir / "unit-box-assembly.scad"
        assert stl.is_file() and stl.stat().st_size > 0, report["errors"]
        assert scad.is_file() and "cube" in scad.read_text(encoding="utf-8")
        volume = report["measurements"]["assembly"]["volume_mm3"]
        assert abs(volume - 20 * 10 * 5) < 1.0, "measured {} for a 1000 mm3 box".format(volume)
        assert (out_dir / "REPORT.md").is_file()


def test_build_drawing_is_opt_in_and_never_fails_the_build() -> None:
    """`drawing=False` records a skipped stage; `drawing=True` with no OpenSCAD records an
    `unavailable` skip and a note - the build stays green either way (evidence, not a gate)."""
    from phimech.backends import base
    from phimech.synth import drawing as drawing_module

    with tempfile.TemporaryDirectory(prefix="phimech-pipeline-drawing-") as tmp:
        _, report = _build_with_stub(Path(tmp), BOX_IR, ["stl", "scad"])
        stage = _stage(report, "drawing")
        assert stage["skipped"] and "not requested" in stage["note"]
        assert report["drawing"] is None
        assert report["stages"][-1]["stage"] == "report"

    def _no_openscad(*_args, **_kwargs):
        raise RuntimeError("OpenSCAD not found (test)")

    stub = _StubBackend()
    saved = (base.for_format, base.get, drawing_module.for_build)
    base.for_format = lambda fmt, prefer=("openscad", "freecad"): stub  # type: ignore[assignment]
    base.get = lambda name: stub  # type: ignore[assignment]
    drawing_module.for_build = _no_openscad  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory(prefix="phimech-pipeline-drawing-") as tmp:
            directory = Path(tmp)
            report = pipeline.build(str(_write_ir(directory, BOX_IR)), directory / "out",
                                    formats=["stl"], backend="auto", drawing=True)
    finally:
        base.for_format, base.get, drawing_module.for_build = saved  # type: ignore[assignment]
    if not _stage(report, "place")["ok"]:
        return
    stage = _stage(report, "drawing")
    assert stage["ok"] and stage["skipped"] and "OpenSCAD not found (test)" in stage["note"]
    assert not any(e.startswith("drawing") for e in report["errors"]), report["errors"]
    assert report["drawing"] is None
    assert any(n.startswith("drawing: not rendered") for n in report["notes"])
    assert report["stages"][-1]["stage"] == "report"


def test_build_drawing_for_real_with_openscad() -> None:
    """A build with drawing=True writes the sheet with the design as its manifest (revision and
    material stated, not UNSPECIFIED), runs the deterministic sheet checks, and files both in
    report.json - and a native build is not judged as a failed STEP translation."""
    if not _openscad_available():
        print("skip: openscad backend unavailable")
        return
    design = dict(BOX_IR)
    design.update({"rev": "B", "company": "Example Co",
                   "manufacturing": {"process": "fdm", "material": "PETG"}})
    with tempfile.TemporaryDirectory(prefix="phimech-pipeline-drawing-real-") as tmp:
        directory = Path(tmp)
        out_dir = directory / "out"
        report = pipeline.build(str(_write_ir(directory, design)), out_dir, formats=["stl"],
                                backend="openscad", preview=False, drawing=True)
        assert report["ok"], report["errors"]
        stage = _stage(report, "drawing")
        assert stage["ok"] and not stage.get("skipped"), stage
        record = report["drawing"]
        assert record and Path(record["path"]).name == "drawing.svg"
        assert Path(record["path"]).is_file()
        assert record["revision"] == "B" and record["material"] == "PETG"
        checks = record["checks"]
        by_name = {c["name"]: c for c in checks["checks"]}
        assert by_name["views_readable"]["status"] == "pass", by_name["views_readable"]
        assert by_name["dimension_traceability"]["status"] == "pass", by_name["dimension_traceability"]
        assert by_name["source_translation_clean"]["status"] == "not_applicable"
        assert checks["summary"]["status"] in ("pass", "needs_review"), checks["summary"]
        drawn = [o for o in report["outputs"] if o.get("role") == "drawing"]
        assert {o["fmt"] for o in drawn} >= {"svg"}
        assert all(Path(o["path"]).is_file() and o["bytes"] > 0 for o in drawn)
        if record["pdf"]:
            assert ("pdf", True) in {(o["fmt"], Path(o["path"]).is_file()) for o in drawn}
        assert report["stages"][-1]["stage"] == "report"


def test_check_examples_never_raise() -> None:
    examples = sorted((REPO / "examples").glob("*.json"))
    if not examples:
        return
    for example in examples:
        report = pipeline.check(str(example))
        _report_shape(report)
        assert report["elapsed_s"] < 5.0, "{} took {}s".format(example.name, report["elapsed_s"])
        for record in report["stages"]:
            assert "traceback" not in record or record["stage"] != "load", report["errors"]


# --------------------------------------------------------------------------------------
# runner for `python3 tests/test_pipeline.py`
# --------------------------------------------------------------------------------------


def _run_without_pytest() -> None:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for name, test in tests:
        test()
        print("ok   {}".format(name))
    print("{} pipeline tests passed".format(len(tests)))


if __name__ == "__main__":
    _run_without_pytest()
