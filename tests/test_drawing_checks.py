"""Tests for the deterministic drawing gates (``phimech/check/drawing_checks.py``).

Runs both ways, per house rule::

    /usr/bin/python3 tests/test_drawing_checks.py
    pytest tests/test_drawing_checks.py

The gate that matters is the fixture sweep: ``fixtures/drawing/`` holds one correct sheet and
ten sheets each carrying exactly one planted defect, and every defective one must fail
**exactly** the check it was built to fail.  A check that fires on the wrong fixture is as
broken as one that never fires.

Nothing here needs OpenSCAD, FreeCAD, a model or a network; the one end-to-end test that
renders a real sheet skips cleanly when OpenSCAD is absent, and exists only to prove the SVG
dialect the parser reads is still the dialect the drawing module writes.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from phimech.check import drawing_checks as dc  # noqa: E402

FIXTURES = REPO / "fixtures" / "drawing"
BRACKET = REPO / "examples" / "brep" / "bracket.json"


def _fixture_dirs() -> List[Path]:
    return sorted(p for p in FIXTURES.iterdir() if p.is_dir() and (p / "expect.json").is_file())


def _run_fixture(directory: Path) -> Dict[str, Any]:
    expect = json.loads((directory / "expect.json").read_text(encoding="utf-8"))
    return dc.run_checks(
        json.loads((directory / expect["drawing"]).read_text(encoding="utf-8")),
        svg_path=directory / expect["svg"],
        source_manifest=json.loads(
            (directory / expect["source_manifest"]).read_text(encoding="utf-8")),
        stl_path=directory / expect["stl"],
        declared_intent=json.loads(
            (directory / expect["declared_intent"]).read_text(encoding="utf-8")),
    )


def _drawing(**overrides: Any) -> Dict[str, Any]:
    """A minimal measured result; the sheet is supplied separately per test."""
    record: Dict[str, Any] = {
        "path": "drawing.svg", "scale": 2.0, "sheet_mm": [420.0, 297.0],
        "bbox_mm": {"min": [0.0, 0.0, 0.0], "max": [60.0, 40.0, 9.0]},
        "levels_z": [0.0, 9.0], "groups": [],
        "views": {"top": {"bbox": [0.0, 0.0, 60.0, 40.0]},
                  "front": {"bbox": [0.0, 0.0, 60.0, 9.0]},
                  "right": {"bbox": [0.0, 0.0, 40.0, 9.0]}},
    }
    record.update(overrides)
    return record


def _named(report: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(check for check in report["checks"] if check["name"] == name)


class FixtureSweep(unittest.TestCase):
    """The exit gate: one correct sheet, ten defects, each failing exactly its own check."""

    def test_fixture_set_meets_the_proposal_minimum(self) -> None:
        directories = _fixture_dirs()
        defective = [d for d in directories
                     if json.loads((d / "expect.json").read_text(encoding="utf-8"))["fails"]]
        self.assertGreaterEqual(len(directories) - len(defective), 1, "need a correct fixture")
        self.assertGreaterEqual(len(defective), 4, "need >= 4 defective fixtures")

    def test_every_defective_fixture_fails_exactly_its_target(self) -> None:
        for directory in _fixture_dirs():
            with self.subTest(fixture=directory.name):
                expect = json.loads((directory / "expect.json").read_text(encoding="utf-8"))
                report = _run_fixture(directory)
                failed = [c["name"] for c in report["checks"] if c["status"] == "fail"]
                self.assertEqual(failed, expect["fails"], "{}: {}".format(
                    directory.name,
                    [(c["name"], c["evidence"]["note"]) for c in report["checks"]
                     if c["status"] == "fail"]))
                skipped = [c["name"] for c in report["checks"] if c["status"] == "skipped"]
                self.assertEqual(skipped, [], "a fixture must evaluate every check")

    def test_the_correct_fixture_passes_everything(self) -> None:
        report = _run_fixture(FIXTURES / "correct")
        self.assertEqual(report["summary"]["status"], "pass")
        self.assertIsNone(report["summary"]["first_failure"])
        self.assertEqual(report["summary"]["counts"]["fail"], 0)
        self.assertEqual(report["summary"]["counts"]["skipped"], 0)

    def test_every_check_is_present_and_shaped(self) -> None:
        report = _run_fixture(FIXTURES / "correct")
        self.assertEqual([c["name"] for c in report["checks"]], list(dc.CHECK_NAMES))
        for check in report["checks"]:
            with self.subTest(check=check["name"]):
                self.assertEqual(sorted(check),
                                 ["evidence", "name", "status", "threshold", "value"])
                self.assertIn(check["status"], dc.STATUSES)
                self.assertIn("note", check["evidence"])
                self.assertTrue(check["evidence"]["note"])

    def test_fixture_stl_is_binary_and_small(self) -> None:
        """Same rule as the viewer fixtures: binary, and exactly sized, so it cannot bloat."""
        stl = FIXTURES / "plate.stl"
        data = stl.read_bytes()
        self.assertNotEqual(data[:5], b"solid", "commit binary STL, not ASCII")
        count = struct.unpack("<I", data[80:84])[0]
        self.assertEqual(len(data), 84 + 50 * count)
        self.assertLessEqual(len(data), 64 * 1024)

    def test_fixtures_leak_no_absolute_paths(self) -> None:
        for directory in _fixture_dirs():
            for path in sorted(directory.glob("*.json")):
                blob = path.read_text(encoding="utf-8")
                for needle in ("/Users/", "/home/", "/private/", "C:\\"):
                    self.assertNotIn(needle, blob, "{} leaks a local path".format(path))


class IntentVsMeasured(unittest.TestCase):
    """The check that sees a feature that was asked for and never built."""

    HOLES = [[20.0, 10.0], [20.0, 30.0], [45.0, 20.0]]
    PILOTS = [[4.0, 12.0], [4.0, 28.0]]

    def _bracket_measurement(self, with_counterbores: bool) -> Dict[str, Any]:
        groups = [
            {"id": "H1", "kind": "circle", "hole": True, "thru": True, "size": [5.2, 5.2],
             "corner_r": 0.0, "depth": 8.0, "view": "top", "positions": self.HOLES},
            {"id": "H2", "kind": "circle", "hole": True, "thru": False,
             "size": [3.24225, 3.24225], "corner_r": 0.0, "depth": 10.0, "view": "top",
             "positions": self.PILOTS},
        ]
        if with_counterbores:
            groups.insert(0, {"id": "H0", "kind": "circle", "hole": True, "thru": False,
                              "size": [9.5, 9.5], "corner_r": 0.0, "depth": 4.0, "view": "top",
                              "positions": self.HOLES})
        return _drawing(groups=groups)

    def _check(self, drawing: Dict[str, Any]) -> dc.Check:
        program = json.loads(BRACKET.read_text(encoding="utf-8"))
        sheet = dc._Sheet(drawing, [], [], None, None, "inspection", None, program)
        return dc.check_intent_vs_measured(sheet)

    def test_todays_bracket_reports_the_missing_counterbores(self) -> None:
        """The defect another lane is fixing upstream: three declared, none in the solid."""
        check = self._check(self._bracket_measurement(with_counterbores=False))
        self.assertEqual(check.status, dc.FAIL)
        self.assertEqual(check.value, 3)
        self.assertEqual(check.evidence["note"], "3 declared counterbores, 0 found")
        self.assertEqual(check.evidence["by_kind"]["hole"], {"declared": 3, "found": 3})
        self.assertEqual(check.evidence["by_kind"]["tapped_hole"], {"declared": 2, "found": 2})

    def test_the_fixed_bracket_matches_all_eight_features(self) -> None:
        check = self._check(self._bracket_measurement(with_counterbores=True))
        self.assertEqual(check.status, dc.PASS)
        self.assertEqual(check.evidence["declared"], 8)
        self.assertEqual(check.evidence["matched"], 8)
        self.assertEqual(check.evidence["tolerance_mm"], 0.05)

    def test_a_hole_off_by_more_than_the_tolerance_is_not_a_match(self) -> None:
        drawing = self._bracket_measurement(with_counterbores=True)
        drawing["groups"][1]["positions"] = [[20.0, 10.0], [20.0, 30.0], [45.0, 20.06]]
        check = self._check(drawing)
        self.assertEqual(check.status, dc.FAIL)
        self.assertEqual(check.value, 1)

    def test_no_declared_intent_is_not_applicable_never_invented(self) -> None:
        sheet = dc._Sheet(_drawing(), [], [], None, None, "inspection", None, None)
        self.assertEqual(dc.check_intent_vs_measured(sheet).status, dc.NOT_APPLICABLE)

    def test_source_json_features_block_is_understood(self) -> None:
        declared = dc.declared_features(
            {"features": [{"kind": "counterbore", "count": 2, "diameter": 9.5, "depth": 4}]})
        self.assertEqual(len(declared), 2)
        self.assertEqual(declared[0]["diameter"], 9.5)
        self.assertIsNone(declared[0]["position"])


class LevelsAreFaces(unittest.TestCase):
    """The check ``dimension_traceability`` structurally cannot be."""

    def test_a_phantom_level_traces_perfectly_and_still_fails(self) -> None:
        report = _run_fixture(FIXTURES / "phantom-level")
        self.assertEqual(_named(report, "dimension_traceability")["status"], dc.PASS,
                         "the phantom level is a 'measured' value: that is the whole point")
        phantom = _named(report, "levels_are_faces")
        self.assertEqual(phantom["status"], dc.FAIL)
        self.assertEqual(phantom["evidence"]["levels"], [4.33])
        self.assertEqual(phantom["evidence"]["phantom"][0]["face_area_mm2"], 0.0)

    def test_without_a_mesh_it_skips_rather_than_passes(self) -> None:
        sheet = dc._Sheet(_drawing(), [], [], None, None, "inspection", None, None)
        check = dc.check_levels_are_faces(sheet)
        self.assertEqual(check.status, dc.SKIPPED)
        self.assertIsNone(check.value)

    def test_a_precomputed_level_faces_block_is_accepted(self) -> None:
        drawing = _drawing(levels_z=[0.0, 3.0],
                           level_faces=[{"z": 0.0, "area_mm2": 2400.0},
                                        {"z": 3.0, "area_mm2": 0.2}])
        sheet = dc._Sheet(drawing, [], [], None, None, "inspection", None, None)
        check = dc.check_levels_are_faces(sheet)
        self.assertEqual(check.status, dc.FAIL)
        self.assertEqual(check.value, 1)

    def test_face_area_of_a_horizontal_triangle_pair(self) -> None:
        square = [((0.0, 0.0, 2.0), (4.0, 0.0, 2.0), (4.0, 3.0, 2.0)),
                  ((0.0, 0.0, 2.0), (4.0, 3.0, 2.0), (0.0, 3.0, 2.0))]
        self.assertAlmostEqual(dc.horizontal_face_area(square, 2.0), 12.0)
        self.assertEqual(dc.horizontal_face_area(square, 5.0), 0.0)


class MemberCoverage(unittest.TestCase):
    """Coverage is per member: a group-level pass hides an unlocatable hole."""

    def test_an_unlocatable_member_fails_while_its_group_looks_covered(self) -> None:
        report = _run_fixture(FIXTURES / "unlocated-member")
        coverage = _named(report, "feature_coverage")
        self.assertEqual(coverage["status"], dc.FAIL)
        self.assertEqual(coverage["value"], 1)
        self.assertEqual(coverage["evidence"]["member_coverage"], "2/3")
        self.assertEqual(coverage["evidence"]["members"], ["H2#1"])
        detail = coverage["evidence"]["detail"][0]
        self.assertTrue(detail["callout"], "the group is called out; only the member is lost")
        self.assertFalse(detail["located_x"])
        self.assertTrue(detail["located_y"])

    def test_locatable_follows_a_chain_of_printed_pitches(self) -> None:
        # 10 from the datum, then 35 to its neighbour: both located, the stray one is not.
        self.assertEqual(dc.locatable([10.0, 45.0, 52.0], [10.0, 35.0]), {0, 1})
        self.assertEqual(dc.locatable([10.0, 45.0], [10.0]), {0})
        self.assertEqual(dc.locatable([10.0, 45.0], []), set())


class SheetParsing(unittest.TestCase):
    """The sheet is read back from its own text positions; nothing is assumed."""

    SVG = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" '
        'viewBox="0 0 420 297">\n'
        '<rect width="420" height="297" fill="white"/>\n'
        '<rect x="292.000" y="241.000" width="118.000" height="42.000" fill="none" '
        'stroke="#111" stroke-width="0.50"/>\n'
        '<text x="50.000" y="100.000" font-family="Helvetica, Arial, sans-serif" '
        'font-size="2.60" text-anchor="middle" fill="#1d4ed8" dominant-baseline="auto">30'
        '</text>\n'
        '<text x="33.000" y="150.000" font-family="Helvetica, Arial, sans-serif" '
        'font-size="2.60" text-anchor="middle" fill="#1d4ed8" dominant-baseline="auto" '
        'transform="rotate(-90.0 33.000 150.000)">20</text>\n'
        '<text x="172.800" y="102.000" font-family="Helvetica, Arial, sans-serif" '
        'font-size="2.60" text-anchor="start" fill="#111" dominant-baseline="middle">'
        'H1  10 x 6 THRU</text>\n'
        '</svg>\n'
    )

    def test_anchor_baseline_and_rotation_become_paper_boxes(self) -> None:
        texts = dc.parse_texts(self.SVG)
        self.assertEqual([t.content for t in texts], ["30", "20", "H1  10 x 6 THRU"])
        middle = texts[0]
        self.assertAlmostEqual((middle.bbox[0] + middle.bbox[2]) / 2.0, 50.0, places=6)
        rotated = texts[1]
        self.assertLess(rotated.bbox[2] - rotated.bbox[0], rotated.bbox[3] - rotated.bbox[1],
                        "a -90 rotated label is taller than it is wide")
        self.assertTrue(texts[0].is_dimension)
        self.assertFalse(texts[2].is_dimension)
        self.assertEqual(texts[2].callout_group, "H1")

    def test_the_title_block_is_found_not_assumed(self) -> None:
        block = dc.find_title_block(dc.parse_rects(self.SVG), (420.0, 297.0))
        self.assertIsNotNone(block)
        self.assertEqual((block.x, block.w), (292.0, 118.0))
        self.assertIsNone(dc.find_title_block([], (420.0, 297.0)))

    def test_a_callouts_id_and_quantity_are_not_dimensions(self) -> None:
        self.assertEqual(dc.callout_numbers("H1  10 x 6 THRU"), [10.0, 6.0])
        self.assertEqual(dc.callout_numbers("H2  2x 6 x 6 THRU"), [6.0, 6.0])
        self.assertEqual(dc.callout_numbers("B1  BOSS \u23008 \u21916"), [8.0, 6.0])

    def test_an_unreadable_sheet_skips_and_never_passes(self) -> None:
        report = dc.run_checks(_drawing(path="/nonexistent/drawing.svg"))
        self.assertEqual(report["summary"]["status"], "needs_review")
        for name in ("text_no_collision", "text_not_clipped", "dimension_traceability"):
            self.assertEqual(_named(report, name)["status"], dc.SKIPPED)


class ContinuationSheets(unittest.TestCase):
    """A feature table that continues on sheet 2 is still part of the drawing."""

    @staticmethod
    def _sheet(body: str) -> str:
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" '
                'viewBox="0 0 420 297">\n' + body + '</svg>\n')

    def test_a_callout_on_sheet_2_covers_a_feature_dimensioned_on_sheet_1(self) -> None:
        dims = ('<text x="50.000" y="100.000" font-size="2.60" text-anchor="middle" '
                'fill="#1d4ed8">30</text>\n'
                '<text x="80.000" y="100.000" font-size="2.60" text-anchor="middle" '
                'fill="#1d4ed8">20</text>\n')
        # drawn where sheet 1 has its "30": different paper, so not a collision
        row = ('<text x="49.000" y="100.000" font-size="2.30" text-anchor="start" '
               'fill="#111">H1  10 x 6 THRU</text>\n')
        groups = [{"id": "H1", "view": "top", "positions": [[30.0, 20.0]]}]
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / "d.svg", Path(tmp) / "d-sheet2.svg"
            one.write_text(self._sheet(dims), encoding="utf-8")
            two.write_text(self._sheet(row), encoding="utf-8")
            single = dc.run_checks(_drawing(path=str(one), groups=groups))
            self.assertEqual(_named(single, "feature_coverage")["status"], dc.FAIL)
            both = dc.run_checks(_drawing(
                path=str(one), groups=groups,
                sheets=[{"sheet": 1, "svg": str(one)}, {"sheet": 2, "svg": str(two)}]))
            self.assertEqual(_named(both, "feature_coverage")["status"], dc.PASS)
            self.assertEqual(_named(both, "text_no_collision")["status"], dc.PASS)
            self.assertIn("H1", dc.printed_inventory(_drawing(
                path=str(one), groups=[],
                sheets=[{"svg": str(one)}, {"svg": str(two)}]))["feature_ids"])
            missing = dc.run_checks(_drawing(
                path=str(one), groups=groups,
                sheets=[{"svg": str(one)}, {"svg": str(Path(tmp) / "gone.svg")}]))
            self.assertEqual(_named(missing, "feature_coverage")["status"], dc.SKIPPED)


class SimpleGates(unittest.TestCase):
    """The remaining thresholds, exercised where a fixture would be overkill."""

    def _sheet(self, drawing: Dict[str, Any], intent: str = "inspection") -> dc._Sheet:
        return dc._Sheet(drawing, [], [], None, None, intent, None, None)

    def test_a_dot_sized_view_is_not_readable(self) -> None:
        drawing = _drawing(scale=0.1, views={
            "top": {"bbox": [0.0, 0.0, 2400.0, 8.0]},
            "front": {"bbox": [0.0, 0.0, 2400.0, 8.0]},
            "right": {"bbox": [0.0, 0.0, 8.0, 8.0]}})
        check = dc.check_views_readable(self._sheet(drawing))
        self.assertEqual(check.status, dc.FAIL)
        self.assertAlmostEqual(check.value, 0.8)
        self.assertEqual(check.evidence["views"], ["right"])

    def test_an_off_series_scale_is_refused(self) -> None:
        check = dc.check_views_readable(self._sheet(_drawing(scale=0.9)))
        self.assertEqual(check.status, dc.FAIL)
        self.assertTrue(check.evidence["off_series"])

    def test_a_standard_scale_with_big_views_passes(self) -> None:
        self.assertEqual(dc.check_views_readable(self._sheet(_drawing())).status, dc.PASS)

    def test_translation_is_not_applicable_without_a_manifest(self) -> None:
        self.assertEqual(dc.check_source_translation_clean(self._sheet(_drawing())).status,
                         dc.NOT_APPLICABLE)

    def test_translation_is_not_applicable_for_a_design_ir_manifest(self) -> None:
        """A build's design.json is a manifest with no `translation` block: nothing was
        translated, so the check is N/A - not a failure of a sheet that came straight from CSG."""
        design = {"project": "unit-box", "units": "mm", "components": [], "rev": "A"}
        sheet = dc._Sheet(_drawing(), [], [], None, design, "inspection", None, None)
        self.assertEqual(dc.check_source_translation_clean(sheet).status, dc.NOT_APPLICABLE)

    def test_a_source_json_without_a_translation_block_still_fails(self) -> None:
        manifest = {"source": {"kind": "solidworks", "path": "x.sldprt"}, "step": {"path": "x.step"}}
        sheet = dc._Sheet(_drawing(), [], [], None, manifest, "inspection", None, None)
        self.assertEqual(dc.check_source_translation_clean(sheet).status, dc.FAIL)

    def test_a_completed_export_with_an_error_diagnostic_still_fails(self) -> None:
        manifest = {"translation": {"ok": True, "diagnostics": [
            {"severity": "warning", "message": "a face was rebuilt"},
            {"severity": "error", "message": "entity #1842 was dropped"}]}}
        sheet = dc._Sheet(_drawing(), [], [], None, manifest, "inspection", None, None)
        check = dc.check_source_translation_clean(sheet)
        self.assertEqual(check.status, dc.FAIL)
        self.assertEqual(check.value, 1)

    def test_manufacturing_intent_is_advisory_for_inspection_and_a_gate_otherwise(self) -> None:
        advisory = dc.check_manufacturing_intent_present(self._sheet(_drawing()))
        self.assertEqual(advisory.status, dc.NOT_APPLICABLE)
        self.assertIn("material", advisory.value)
        gate = dc.check_manufacturing_intent_present(self._sheet(_drawing(), "manufacturing"))
        self.assertEqual(gate.status, dc.FAIL)
        self.assertIn("tolerance", gate.evidence["absent"])

    def test_summary_is_fail_on_any_fail_and_needs_review_on_any_skip(self) -> None:
        checks = [dc.Check("a", dc.PASS, 1, 1), dc.Check("b", dc.SKIPPED, None, 1)]
        self.assertEqual(dc.summarize(checks)["status"], "needs_review")
        checks.append(dc.Check("c", dc.FAIL, 2, 1))
        summary = dc.summarize(checks)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["first_failure"], "c")
        self.assertEqual(summary["counts"], {"pass": 1, "fail": 1, "skipped": 1,
                                             "not_applicable": 0})


def _openscad() -> bool:
    try:
        from phimech.backends.openscad import OpenSCADBackend

        return bool(OpenSCADBackend().available())
    except Exception:       # noqa: BLE001 - a missing backend is a skip, not a failure
        return False


class RealRender(unittest.TestCase):
    """One live render: a canary for the sheet dialect, deliberately not a gate.

    The committed fixtures are what gate the checker - they are frozen so a change to the
    drawing module's measurement internals cannot make this suite lie.  This test exists only
    to notice if the *dialect* (the attributes the parser reads off ``<text>`` and ``<rect>``)
    ever stops matching what the drawing module writes.  A render that fails for reasons of
    its own is that module's business and is skipped here, not failed.
    """

    @unittest.skipUnless(_openscad(), "openscad is not installed")
    def test_checks_read_a_freshly_rendered_sheet(self) -> None:
        from phimech.synth import drawing

        part = ("difference() { cube([60,40,9]); "
                "translate([10,20,-1]) cube([6,6,11], center=false); }")
        with tempfile.TemporaryDirectory() as tmp:
            scad = Path(tmp) / "part.scad"
            scad.write_text(part, encoding="utf-8")
            try:
                result = drawing.render_drawing(scad, Path(tmp) / "part.svg", png=False)
            except Exception as exc:        # noqa: BLE001 - not this suite's failure to own
                raise unittest.SkipTest("the drawing module could not render: {}".format(exc))
            report = dc.run_checks(result, svg_path=result.path)
            # Read inside the temp dir: the sheet is gone once the context exits.
            texts = dc.parse_texts(Path(result.path).read_text(encoding="utf-8"))
        self.assertEqual([check["name"] for check in report["checks"]], list(dc.CHECK_NAMES))
        # The parser must find real labels on a real sheet; whether they all pass is the
        # drawing module's business, not this test's.
        self.assertGreater(len(texts), 10, "no labels parsed from a real sheet")
        self.assertTrue(any(t.is_dimension for t in texts), "no dimension labels parsed")
        self.assertTrue(any(t.callout_group for t in texts), "no leader callouts parsed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
