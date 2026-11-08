"""Tests for the DXF writer (``phimech/synth/dxf.py``).

Three halves, deliberately:

* ``TestGeometry`` and ``TestDocument`` are stdlib-only and always run. They are the important
  ones: a DXF that "looks written" proves nothing, so every assertion here is a *number* read
  back out of the file — a fitted circle's radius, an arc's sweep, a dimension's measurement
  against its own definition points, the group-code stream's well-formedness.
* ``TestEzdxf`` self-skips unless an interpreter with ezdxf is available (``build/brep-venv``,
  or ``PHIMECH_BREP_PYTHON``). ezdxf's ``audit()`` is an independent R2000 structure checker.
* ``TestFreeCAD`` self-skips unless ``freecadcmd`` is installed. It is the one check that a real
  CAD kernel reads our circles back as *analytic circles* rather than polygon approximations,
  which is the entire point of the module.

Run both ways:
    /usr/bin/python3 tests/test_dxf.py
    pytest tests/test_dxf.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phimech.synth import drawing as d  # noqa: E402
from phimech.synth import dxf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# synthetic geometry: the same vocabulary drawing.py's tests use
# --------------------------------------------------------------------------------------


def _circle(cx: float, cy: float, dia: float, n: int = 48) -> List[Tuple[float, float]]:
    """A tessellated circle with every vertex exactly on the circle, as OpenSCAD emits."""
    r = dia / 2.0
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def _rect(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _rounded_rect(x0: float, y0: float, x1: float, y1: float, r: float,
                  n: int = 12) -> List[Tuple[float, float]]:
    """Vertices only on the corner arcs, exactly as OpenSCAD's offset()/minkowski emit them."""
    points: List[Tuple[float, float]] = []
    for (cx, cy), start in (((x1 - r, y1 - r), 0.0), ((x0 + r, y1 - r), 90.0),
                            ((x0 + r, y0 + r), 180.0), ((x1 - r, y0 + r), 270.0)):
        for i in range(n + 1):
            angle = math.radians(start + 90.0 * i / n)
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


#: A 60 x 40 x 9 plate carrying one of everything the classifier can produce:
#: a thru circle, a thru rectangle, a blind obround slot, two bosses and two blind bores.
#: The awkward cases are deliberate — thru vs blind, and a slot whose ends are true semicircles.
PLATE_LEVELS = [0.0, 3.0, 4.0, 9.0]


#: What ``measure_features`` finds for the fixture: the wall span of each feature, keyed by
#: (kind, rounded diameter/width). The fixture has no mesh, so the measurement is supplied
#: here rather than faked inside the writer.
PLATE_WALLS = {("circle", 6.0): (0.0, 3.0), ("rect", 10.0): (0.0, 3.0),
               ("slot", 20.0): (0.0, 3.0), ("circle", 8.0): (3.0, 9.0),
               ("circle", 2.7): (4.0, 9.0)}


def _apply_walls(features: Sequence["d.Feature"]) -> None:
    """Stand in for ``drawing.measure_features``: give each feature its own wall span.

    Real output is measured, so the fixture must be too — an unmeasured group is marked
    ``(EST)`` on the sheet and that is a different case, covered by its own test."""
    for feature in features:
        key = (feature.shape.kind, round(feature.shape.size[0], 1))
        span = PLATE_WALLS.get(key)
        if span is None:
            continue
        feature.depth_from, feature.depth_to = span
        feature.segments = [span]
        feature.measured = True


def _plate_views(measured: bool = True) -> Tuple[Dict[str, "d.ViewData"], List["d.Group"]]:
    plate = _rect(0, 0, 60, 40)
    hole = _circle(15, 20, 6)
    window = _rect(25, 8, 35, 16)
    slot = _rounded_rect(4, 4, 24, 12, 4)
    boss_a, boss_b = _circle(45, 30, 8), _circle(45, 10, 8)
    bore_a, bore_b = _circle(45, 30, 2.7), _circle(45, 10, 2.7)

    silhouette = [plate, hole, window]
    slices = [
        (1.5, 0.0, 3.0, [plate, hole, window, slot]),
        (3.5, 3.0, 4.0, [boss_a, boss_b]),
        (6.5, 4.0, 9.0, [boss_a, boss_b, bore_a, bore_b]),
    ]
    features = d.aggregate_features("top", silhouette, slices)
    d._finish_depths(features, PLATE_LEVELS)
    if measured:
        _apply_walls(features)
    top = d.ViewData("top", silhouette, slices, features, (0, 0, 60, 40), list(PLATE_LEVELS))
    front = d.ViewData("front", [_rect(0, 0, 60, 9)], [], [], (0, 0, 60, 9), [])
    right = d.ViewData("right", [_rect(0, 0, 40, 9)], [], [], (0, 0, 40, 9), [])
    groups = d.group_features(features)
    return {"top": top, "front": front, "right": right}, groups


def _many_feature_views(count: int = 24) -> Tuple[Dict[str, "d.ViewData"], List["d.Group"]]:
    """A 200 x 120 plate with ``count`` thru holes of *distinct* diameters: one group each, so
    the table cap, the callout cap and the label cap are all exercised at once."""
    plate = _rect(0, 0, 200, 120)
    holes = []
    for index in range(count):
        column, row = index % 6, index // 6
        holes.append(_circle(15.0 + 30.0 * column, 20.0 + 25.0 * row, 1.0 + 0.5 * index))
    silhouette = [plate] + holes
    slices = [(3.0, 0.0, 6.0, [plate] + holes)]
    features = d.aggregate_features("top", silhouette, slices)
    d._finish_depths(features, [0.0, 6.0])
    for feature in features:
        feature.segments = [(0.0, 6.0)]
        feature.measured = True
    top = d.ViewData("top", silhouette, slices, features, (0, 0, 200, 120), [0.0, 6.0])
    front = d.ViewData("front", [_rect(0, 0, 200, 6)], [], [], (0, 0, 200, 6), [])
    right = d.ViewData("right", [_rect(0, 0, 120, 6)], [], [], (0, 0, 120, 6), [])
    return {"top": top, "front": front, "right": right}, d.group_features(features)


def _two_slab_views() -> Tuple[Dict[str, "d.ViewData"], List["d.Group"]]:
    """A hole bored through two slabs with a 9 mm gap, plus one whose second run is in the gap.

    This is the case a single prism gets wrong: drawn as one 0->15 box it puts a hidden edge
    across empty space, which a machinist reads as "there is material here"."""
    plate, hole, ghost = _rect(0, 0, 60, 40), _circle(15, 20, 6), _circle(45, 20, 4)
    top = d.ViewData("top", [plate, hole, ghost], [], [], (0, 0, 60, 40), [0.0, 3.0, 12.0, 15.0])
    top.features.append(d.Feature(d.classify_loop(hole, True), "top", 0.0, 15.0, True,
                                  segments=[(0.0, 3.0), (12.0, 15.0)], measured=True))
    top.features.append(d.Feature(d.classify_loop(ghost, True), "top", 0.0, 9.0, False,
                                  segments=[(0.0, 3.0), (6.0, 9.0)], measured=True))
    front = d.ViewData("front", [_rect(0, 0, 60, 3), _rect(0, 12, 60, 15)], [], [],
                       (0, 0, 60, 15), [])
    right = d.ViewData("right", [_rect(0, 0, 40, 3), _rect(0, 12, 40, 15)], [], [],
                       (0, 0, 40, 15), [])
    return {"top": top, "front": front, "right": right}, d.group_features(top.features)


def _write(views, groups, directory: Path, name: str = "part.dxf", **kwargs):
    kwargs.setdefault("title", "PLATE")
    kwargs.setdefault("source_label", "plate.scad")
    kwargs.setdefault("part", "plate")
    kwargs.setdefault("volume_mm3", 20000.0)
    kwargs.setdefault("date", "2026-09-24")
    target = directory / name
    return dxf.write_dxf(views, groups, target, **kwargs), target


# --------------------------------------------------------------------------------------
# geometry: analytic reconstruction, with no file involved
# --------------------------------------------------------------------------------------


class TestGeometry(unittest.TestCase):
    def test_choose_text_height_snaps_to_the_iso_series(self):
        self.assertEqual(dxf.choose_text_height(100.0), 2.5)     # 100/40 = 2.5
        self.assertEqual(dxf.choose_text_height(140.0), 3.5)
        # below the smallest height in the series: the floor, never zero
        self.assertEqual(dxf.choose_text_height(4.0), dxf.TEXT_HEIGHTS[0])
        self.assertIn(dxf.choose_text_height(2000.0), dxf.TEXT_HEIGHTS)

    def test_choose_sheet_picks_the_smallest_that_fits_at_1_to_1(self):
        self.assertEqual(dxf.choose_sheet(200.0, 150.0)[0], "A4")
        self.assertEqual(dxf.choose_sheet(300.0, 150.0)[0], "A3")
        self.assertEqual(dxf.choose_sheet(500.0, 300.0)[0], "A2")
        self.assertIsNone(dxf.choose_sheet(2000.0, 900.0),
                          "larger than A0 must refuse a sheet rather than scale the geometry")

    def test_decompose_loop_recovers_four_lines_and_four_arcs(self):
        loop = _rounded_rect(0, 0, 40, 20, 5, n=16)
        segments = dxf.decompose_loop(loop, tol=0.01)
        arcs = [s for s in segments if s.kind == "arc"]
        lines = [s for s in segments if s.kind == "line"]
        self.assertEqual(len(arcs), 4, "one arc per corner")
        self.assertEqual(len(lines), 4, "one line per side")
        for arc in arcs:
            self.assertAlmostEqual(arc.radius, 5.0, places=6)
            self.assertAlmostEqual(arc.sweep, 90.0, places=4)

    def test_decompose_loop_leaves_a_polygon_as_lines(self):
        segments = dxf.decompose_loop(_rect(0, 0, 10, 8), tol=0.01)
        self.assertEqual([s.kind for s in segments], ["line"] * 4)

    def test_decompose_loop_refuses_an_arc_that_is_not_there(self):
        """A run of points wandering by more than the tolerance must stay lines, not become a
        smooth arc that the part does not have."""
        noisy = [(i * 1.0, (0.4 if i % 2 else -0.4)) for i in range(12)]
        segments = dxf.decompose_loop(noisy, tol=0.02)
        self.assertEqual([s.kind for s in segments].count("arc"), 0)

    def test_rounded_rect_deviation_is_zero_for_a_true_rounded_rect(self):
        loop = _rounded_rect(0, 0, 40, 20, 5, n=24)
        self.assertLess(dxf.rounded_rect_deviation(loop, (0, 0, 40, 20), 5.0), 1e-9)

    def test_rounded_rect_deviation_catches_a_wrong_radius(self):
        loop = _rounded_rect(0, 0, 40, 20, 5, n=24)
        self.assertGreater(dxf.rounded_rect_deviation(loop, (0, 0, 40, 20), 1.0), 1.0)

    def test_rounded_rect_segments_drop_zero_length_sides_of_an_obround(self):
        segments = dxf.rounded_rect_segments((0, 0, 20, 8), 4.0)
        self.assertEqual(sum(1 for s in segments if s.kind == "arc"), 4)
        self.assertEqual(sum(1 for s in segments if s.kind == "line"), 2,
                         "an obround has no straight ends")

    def test_view_origins_keep_the_top_view_at_true_model_coordinates(self):
        views, _ = _plate_views()
        origins = dxf.view_origins(views, gap=20.0)
        self.assertEqual(origins["top"], (0.0, 0.0))
        self.assertEqual(origins["front"][0], 0.0)
        self.assertLess(origins["front"][1], 0.0, "the front view sits below the plan")
        self.assertGreater(origins["right"][0], 0.0)
        self.assertEqual(origins["right"][1], origins["front"][1], "z must stay aligned")


# --------------------------------------------------------------------------------------
# the written document
# --------------------------------------------------------------------------------------


class TestDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="phimech-dxf-test-")
        cls.tmp = Path(cls._tmp)
        cls.views, cls.groups = _plate_views()
        cls.result, cls.path = _write(cls.views, cls.groups, cls.tmp)
        cls.text = cls.path.read_text(encoding="utf-8")
        cls.facts = dxf.inspect_dxf(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    # -- structure -----------------------------------------------------------------

    def test_group_code_stream_is_well_formed(self):
        tags = dxf.read_tags(self.text)
        self.assertTrue(tags)
        self.assertEqual(tags[0], (0, "SECTION"))
        self.assertEqual(tags[-1], (0, "EOF"))
        self.assertEqual(len(self.text.splitlines()) % 2, 0, "every code needs a value line")
        sections = [value for (code, value), (next_code, _) in zip(tags, tags[1:])
                    if code == 0 and value == "SECTION" and next_code == 2]
        self.assertEqual(len(sections), 6)
        names = [tags[i + 1][1] for i, (code, value) in enumerate(tags)
                 if code == 0 and value == "SECTION"]
        self.assertEqual(names, ["HEADER", "CLASSES", "TABLES", "BLOCKS", "ENTITIES", "OBJECTS"])

    def test_header_declares_r2000_millimetres_and_iso_dimension_text(self):
        self.assertEqual(self.facts.version, "AC1015")
        self.assertEqual(self.facts.insunits, 4)
        self.assertEqual(self.facts.measurement, 1)
        self.assertEqual(self.facts.dimlunit, 2)
        self.assertEqual(self.facts.dimdec, 2)
        self.assertEqual(self.facts.dimzin, 8)
        self.assertEqual(self.facts.dimdsep, ord("."))
        self.assertIn("$DIMLFAC\n40\n1.0", self.text, "1:1 means the value is never multiplied")

    def test_tables_define_exactly_the_documented_layers_and_every_linetype_used(self):
        self.assertEqual(sorted(self.facts.layers), sorted(dxf.REQUIRED_LAYERS))
        for name in self.facts.layers_used:
            self.assertIn(name, self.facts.layers, "entity on an undefined layer")
        for name, (_, linetype) in self.facts.layers.items():
            self.assertIn(linetype, self.facts.linetypes,
                          "layer {} references an undefined linetype".format(name))
        for name, (color, linetype) in self.facts.layers.items():
            spec = dxf.LAYER_BY_NAME[name]
            self.assertEqual((color, linetype), (spec.color, spec.linetype),
                             "layer {} drifted from the documented scheme".format(name))
        self.assertEqual(self.facts.layers["HIDDEN"][1], "HIDDEN")
        self.assertEqual(self.facts.layers["CENTER"][1], "CENTER")
        self.assertIn(dxf.DIMSTYLE_NAME, self.facts.dimstyles)
        self.assertIn("txt.shx", self.text, "the text style must name a real .shx font")

    def test_the_sheet_frame_fits_the_1_to_1_content(self):
        self.assertEqual(self.result.scale, "1:1")
        # A3, not A4: the 60 x 40 plate is tiny, but the hole table and the title block beside
        # it are sized from the annotation, and the sheet is chosen to fit the *content*.
        self.assertEqual(self.result.sheet, "A3")
        self.assertEqual(self.result.sheet_mm, [420.0, 297.0])
        width = self.result.extents["max"][0] - self.result.extents["min"][0]
        height = self.result.extents["max"][1] - self.result.extents["min"][1]
        self.assertAlmostEqual(width, 420.0, places=6)
        self.assertAlmostEqual(height, 297.0, places=6)

    # -- the hard requirement: true entities ---------------------------------------

    def test_every_fitted_circle_is_a_circle_entity_with_the_measured_radius(self):
        radii = sorted(round(radius, 6) for _, _, _, radius in self.facts.circles)
        self.assertEqual(radii, [1.35, 1.35, 3.0, 4.0, 4.0],
                         "Ø6 thru, 2x Ø8 boss, 2x Ø2.7 bore")
        self.assertEqual(self.result.circles, 5)
        thru = [c for c in self.facts.circles if c[0] == "VISIBLE"]
        self.assertEqual(len(thru), 1)
        layer, x, y, radius = thru[0]
        self.assertAlmostEqual(x, 15.0, delta=1e-9)
        self.assertAlmostEqual(y, 20.0, delta=1e-9)
        self.assertAlmostEqual(radius, 3.0, delta=0.001)

    def test_no_hole_degraded_into_a_tessellated_polyline(self):
        """The whole reason this module exists: the source loops are 48-gons, and not one of
        them may reach the file as a polyline."""
        self.assertTrue(self.facts.polylines)
        worst = max(vertices for _, vertices, _ in self.facts.polylines)
        self.assertLessEqual(worst, 4, "every polyline here is a rectangle: {}".format(
            self.facts.polylines))

    def test_the_slot_is_four_arcs_and_two_lines_not_a_polygon(self):
        slot_arcs = [a for a in self.facts.arcs if abs(a[3] - 4.0) < 1e-9]
        self.assertEqual(len(slot_arcs), 4)
        for _, _, _, radius, start, end in slot_arcs:
            self.assertAlmostEqual(radius, 4.0, places=9)
            self.assertAlmostEqual((end - start) % 360.0, 90.0, places=6)

    def test_centre_marks_exist_for_every_circular_feature(self):
        self.assertEqual(self.facts.layers_used.get("CENTER"), 2 * self.result.circles,
                         "two crossing centre lines per circle")

    # -- dimensions ----------------------------------------------------------------

    def test_dimension_count_matches_the_placed_dimensions(self):
        # 4 located groups x (x, y) + overall x + overall y on the plan  = 10
        # + one overall width on each of the front and right views       = 12
        # + the z level chain (levels 3 and 4, plus the overall height)  = 15
        self.assertEqual(self.result.dimensions, 15)
        self.assertEqual(len(self.facts.dimensions), 15)
        self.assertEqual(self.facts.entity_counts["DIMENSION"], 15)

    def test_every_dimension_is_editable_with_no_text_override(self):
        for dimension in self.facts.dimensions:
            self.assertIsNone(dimension["override"],
                              "a text override would let the sheet disagree with the geometry")
            self.assertEqual(dimension["style"], dxf.DIMSTYLE_NAME)
            self.assertIn(dimension["block"], self.facts.blocks,
                          "every DIMENSION needs its rendered geometry block")

    def test_every_dimension_measures_its_own_definition_points(self):
        for dimension in self.facts.dimensions:
            p1, p2 = dimension["p1"], dimension["p2"]
            span = abs(p2[1] - p1[1]) if dimension["rotation"] == 90.0 else abs(p2[0] - p1[0])
            self.assertAlmostEqual(span, dimension["measurement"], delta=0.001)

    def test_every_dimension_value_traces_to_a_measured_source_value(self):
        legal = dxf.source_distances(self.views, self.groups)
        for dimension in self.facts.dimensions:
            self.assertTrue(
                any(abs(dimension["measurement"] - value) <= 0.01 for value in legal),
                "{} mm is not a distance the model measures".format(dimension["measurement"]))

    def test_the_plan_dimensions_are_true_model_distances(self):
        values = sorted(round(value, 3) for value in self.result.dimension_values)
        for expected in (60.0, 40.0, 15.0, 20.0, 9.0, 3.0, 4.0):
            self.assertIn(expected, values, "expected a {} mm dimension".format(expected))

    # -- annotation ----------------------------------------------------------------

    def test_callouts_distinguish_thru_from_blind_and_name_the_quantity(self):
        by_id = {group.group_id: group for group in self.groups}
        self.assertEqual(dxf.group_callout(by_id["H3"]), "H3  %%c6 THRU")
        self.assertEqual(dxf.group_callout(by_id["H4"]), "H4  2x %%c2.7 DEPTH 5")
        self.assertEqual(dxf.group_callout(by_id["B1"]), "B1  2x BOSS %%c8 DEPTH 6")
        self.assertIn("H3  %%c6 THRU", self.text)
        self.assertIn("H4  2x %%c2.7 DEPTH 5", self.text)

    def test_callout_size_is_the_group_representative_and_order_independent(self):
        """``Group.size`` is the mean of its members' measured sizes and the only value the
        group prints. Reading ``members[0]`` instead is order-dependent: a 7.28/7.32 pair would
        print whichever was found first. Shuffling the members must change nothing."""
        group = [g for g in self.groups if g.group_id == "H4"][0]
        before = dxf.group_size_text(group)
        self.assertEqual(before, "%%c{}".format(dxf._mm(group.size[0])))
        group.members.reverse()
        try:
            self.assertEqual(dxf.group_size_text(group), before)
            self.assertEqual(dxf.group_callout(group), "H4  2x %%c2.7 DEPTH 5")
        finally:
            group.members.reverse()

    def test_group_text_matches_the_svg_sheet_word_for_word(self):
        """One label function, so the DXF callout and the SVG row cannot disagree; only the
        diameter sign differs, because DXF annotation is ASCII."""
        for group in self.groups:
            self.assertEqual(dxf.group_size_text(group),
                             group.label.replace("\u2300", "%%c"))

    def test_the_title_block_is_a_block_insert_with_only_known_facts(self):
        self.assertIn(dxf.TITLEBLOCK_BLOCK, self.facts.blocks)
        self.assertEqual(self.facts.entity_counts.get("INSERT"), 1)
        self.assertIn("2026-09-24", self.text)
        self.assertIn("mm, 1:1 true size", self.text)
        for invented in ("TOLERANCE:", "MATERIAL:", "FINISH:", "Ra ", "ISO 2768"):
            self.assertNotIn(invented, self.text,
                             "the model carries no {}, so the sheet must not claim one".format(
                                 invented))

    def test_notes_state_what_the_drawing_does_not_specify(self):
        self.assertIn("NO TOLERANCE, THREAD, MATERIAL OR FINISH IS SPECIFIED", self.text)
        self.assertIn("TRUE MODEL COORDINATES", self.text)

    # -- the gate ------------------------------------------------------------------

    def test_check_fidelity_passes_on_the_written_file(self):
        self.assertEqual(dxf.check_fidelity(self.result, self.views, self.groups), [])

    def test_check_fidelity_catches_a_falsified_circle(self):
        corrupted = self.tmp / "corrupt-circle.dxf"
        corrupted.write_text(_bump_first_circle(self.text, 0.75), encoding="utf-8")
        broken = dxf.DxfResult(**dict(self.result.__dict__, path=str(corrupted)))
        problems = dxf.check_fidelity(broken, self.views, self.groups)
        self.assertTrue(any("is not a CIRCLE in the file" in problem for problem in problems),
                        problems)

    def test_check_fidelity_catches_wrong_units_in_the_header(self):
        corrupted = self.tmp / "corrupt-units.dxf"
        corrupted.write_text(self.text.replace("$INSUNITS\n70\n4\n", "$INSUNITS\n70\n1\n"),
                             encoding="utf-8")
        broken = dxf.DxfResult(**dict(self.result.__dict__, path=str(corrupted)))
        problems = dxf.check_fidelity(broken, self.views, self.groups)
        self.assertTrue(any("$INSUNITS" in problem for problem in problems), problems)

    def test_extents_cover_every_entity(self):
        self.assertIsNotNone(self.facts.content_bbox)
        x0, y0, x1, y1 = self.facts.content_bbox
        self.assertLessEqual(self.facts.extmin[0], x0 + 1e-6)
        self.assertLessEqual(self.facts.extmin[1], y0 + 1e-6)
        self.assertGreaterEqual(self.facts.extmax[0], x1 - 1e-6)
        self.assertGreaterEqual(self.facts.extmax[1], y1 - 1e-6)

    def test_result_is_json_serialisable(self):
        json.dumps(self.result.to_dict())


def _bump_first_circle(text: str, delta: float) -> str:
    """Change the radius of the first CIRCLE entity — a falsified file for the negative test."""
    index = text.index("0\nCIRCLE\n")
    tail = text[index:]
    marker = "\n40\n"
    start = tail.index(marker) + len(marker)
    end = tail.index("\n", start)
    radius = float(tail[start:end]) + delta
    return text[:index] + tail[:start] + repr(radius) + tail[end:]


class TestManyFeatures(unittest.TestCase):
    """The stress case: 24 distinct hole sizes, which overflows the table and the callout stack."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="phimech-dxf-many-")
        cls.tmp = Path(cls._tmp)
        cls.views, cls.groups = _many_feature_views(24)
        cls.result, cls.path = _write(cls.views, cls.groups, cls.tmp, name="many.dxf")
        cls.facts = dxf.inspect_dxf(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_every_hole_is_a_circle_entity(self):
        self.assertEqual(len(self.groups), 24)
        self.assertEqual(self.result.circles, 24)
        self.assertEqual(len(self.facts.circles), 24)
        expected = sorted(round((1.0 + 0.5 * i) / 2.0, 6) for i in range(24))
        self.assertEqual(sorted(round(c[3], 6) for c in self.facts.circles), expected)

    def test_caps_hold_and_the_overflow_is_reported_not_hidden(self):
        self.assertEqual(self.result.groups, 24)
        self.assertEqual(self.result.feature_labels, 24, "under the label cap: all 24 labelled")
        self.assertGreater(self.result.callouts, 0)
        self.assertLessEqual(self.result.callouts, dxf.MAX_CALLOUTS)
        self.assertIn("+6 more groups", self.path.read_text(encoding="utf-8"),
                      "the table caps at 18 rows and says how many it dropped")
        self.assertTrue(any("feature table" in warning for warning in self.result.warnings),
                        self.result.warnings)

    def test_the_gate_still_passes(self):
        self.assertEqual(dxf.check_fidelity(self.result, self.views, self.groups), [])

    def test_a_bigger_part_gets_a_bigger_sheet(self):
        self.assertIn(self.result.sheet, ("A2", "A1", "A0"))


class TestHiddenEdges(unittest.TestCase):
    """Hidden edges follow the material, not the feature's overall span."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="phimech-dxf-slabs-")
        cls.tmp = Path(cls._tmp)
        cls.views, cls.groups = _two_slab_views()
        cls.result, cls.path = _write(cls.views, cls.groups, cls.tmp, name="slabs.dxf")
        cls.facts = dxf.inspect_dxf(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _hidden_boxes(self, view: str):
        """HIDDEN boxes of one view, mapped back into that view's own model coordinates.

        The views sit at known offsets, so a box belongs to the view whose bbox contains it
        once that view's offset is removed; the offsets are far enough apart that no box can
        belong to two."""
        dx, dy = self.result.origins[view]
        vx0, vy0, vx1, vy1 = self.views[view].bbox
        out = []
        for layer, _, (x0, y0, x1, y1) in self.facts.polylines:
            if layer != "HIDDEN":
                continue
            box = (x0 - dx, y0 - dy, x1 - dx, y1 - dy)
            if vx0 - 0.01 <= box[0] and box[2] <= vx1 + 0.01 and \
                    vy0 - 0.01 <= box[1] and box[3] <= vy1 + 0.01:
                out.append(box)
        return out

    def test_one_box_per_material_segment_never_one_through_the_gap(self):
        for view in ("front", "right"):
            boxes = self._hidden_boxes(view)
            self.assertEqual(len(boxes), 3, "{}: {}".format(view, boxes))
            for x0, y0, x1, y1 in boxes:
                inside_lower = y1 <= 3.0 + 1e-6
                inside_upper = y0 >= 12.0 - 1e-6
                self.assertTrue(inside_lower or inside_upper,
                                "box {} spans the gap between the slabs".format((x0, y0, x1, y1)))

    def test_a_box_landing_in_empty_air_is_dropped_and_counted(self):
        # the ghost hole's second run (z 6..9) is in the gap: one box per side view refused
        self.assertEqual(self.result.hidden_edges_dropped, 2)

    def test_the_gate_still_passes(self):
        self.assertEqual(dxf.check_fidelity(self.result, self.views, self.groups), [])


class TestUnmeasured(unittest.TestCase):
    """A group whose own wall was not found must not read like a caliper reading."""

    def test_estimated_extents_are_marked_on_the_sheet(self):
        views, groups = _plate_views(measured=False)
        with tempfile.TemporaryDirectory() as tmp:
            result, path = _write(views, groups, Path(tmp), name="est.dxf")
            text = path.read_text(encoding="utf-8")
        self.assertEqual(result.unmeasured_groups, len(groups))
        self.assertTrue(all(dxf.group_callout(g).endswith("(EST)") for g in groups))
        self.assertIn("(est)", text)
        self.assertIn("(EST) MARKS AN EXTENT TAKEN FROM SLICE LEVELS", text)

    def test_measured_groups_carry_no_marker(self):
        _, groups = _plate_views(measured=True)
        self.assertEqual([g for g in groups if not g.measured], [])
        self.assertFalse(any("(EST)" in dxf.group_callout(g) for g in groups))


class TestCounterbore(unittest.TestCase):
    """A counterbore is a measured coaxial wall, so it belongs on the callout — when shared."""

    def _group(self, *records):
        _, groups = _plate_views()
        group = [g for g in groups if g.group_id == "H4"][0]
        for member, record in zip(group.members, records):
            member.counterbore = record
        return group

    def test_a_shared_counterbore_is_printed(self):
        record = {"size": [9.5, 9.5], "depth": 4.0, "from": 5.0, "to": 9.0, "diameter": 9.5}
        group = self._group(dict(record), dict(record))
        self.assertEqual(dxf.counterbore_text(record), "C'BORE %%c9.5 DEPTH 4")
        self.assertIn("C'BORE %%c9.5 DEPTH 4", dxf.group_callout(group))

    def test_members_that_disagree_print_nothing(self):
        wide = {"size": [9.5, 9.5], "depth": 4.0, "from": 5.0, "to": 9.0, "diameter": 9.5}
        narrow = dict(wide, size=[6.0, 6.0], diameter=6.0)
        group = self._group(wide, narrow)
        self.assertIsNone(dxf.group_counterbore(group))
        self.assertNotIn("C'BORE", dxf.group_callout(group))

    def test_one_member_without_one_prints_nothing(self):
        record = {"size": [9.5, 9.5], "depth": 4.0, "from": 5.0, "to": 9.0, "diameter": 9.5}
        group = self._group(dict(record), None)
        self.assertIsNone(dxf.group_counterbore(group))


class TestWiringSeam(unittest.TestCase):
    """``write_dxf_for_result`` is how drawing.py will call this; its defaults must be right."""

    def test_metadata_comes_from_the_drawing_result(self):
        views, groups = _plate_views()
        with tempfile.TemporaryDirectory() as tmp:
            svg = Path(tmp) / "drawing.svg"
            svg.write_text("<svg/>", encoding="utf-8")
            result = d.DrawingResult(
                path=str(svg), png=None, png_tool=None, scale=2.0, sheet_mm=[420.0, 297.0],
                bbox_mm={"min": [0, 0, 0], "max": [60, 40, 9]}, volume_mm3=20000.0,
                levels_z=PLATE_LEVELS, groups=[], views={}, render_s=1.0,
                source="plate.scad", part="plate")
            written = dxf.write_dxf_for_result(result, views, groups, date="2026-09-24")
        self.assertTrue(written.path.endswith("drawing.dxf"))
        self.assertEqual(written.units, "mm")
        self.assertEqual(written.scale, "1:1")


# --------------------------------------------------------------------------------------
# independent readers
# --------------------------------------------------------------------------------------


def _python_with_ezdxf() -> str:
    """An interpreter that can import ezdxf, or "" — ezdxf is never a PhiMech dependency."""
    candidates = [os.environ.get("PHIMECH_EZDXF_PYTHON"),
                  os.environ.get("PHIMECH_BREP_PYTHON"),
                  str(REPO / "build" / "brep-venv" / "bin" / "python"),
                  str(REPO / "build" / "venv" / "bin" / "python")]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run([candidate, "-c", "import ezdxf"], capture_output=True)
        if probe.returncode == 0:
            return candidate
    return ""


def _freecadcmd() -> str:
    explicit = os.environ.get("PHIMECH_FREECADCMD")
    if explicit and Path(explicit).exists():
        return explicit
    default = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"
    return default if Path(default).exists() else (shutil.which("freecadcmd") or "")


_EZDXF_SCRIPT = r"""
import json, sys
import ezdxf
doc = ezdxf.readfile(sys.argv[1])
auditor = doc.audit()
msp = doc.modelspace()
counts = {}
for entity in msp:
    counts[entity.dxftype()] = counts.get(entity.dxftype(), 0) + 1
print("PMDXF:" + json.dumps({
    "version": doc.dxfversion,
    "insunits": doc.header.get("$INSUNITS"),
    "measurement": doc.header.get("$MEASUREMENT"),
    "errors": [str(e) for e in auditor.errors],
    "fixes": [str(f) for f in auditor.fixes],
    "counts": counts,
    "circles": sorted((round(c.dxf.center.x, 6), round(c.dxf.center.y, 6),
                       round(c.dxf.radius, 6)) for c in msp.query("CIRCLE")),
    "dimensions": len(msp.query("DIMENSION")),
    "blocks": sorted(b.name for b in doc.blocks),
}))
"""

_FREECAD_SCRIPT = r"""
import json, math, os, sys
import FreeCAD, Part, importDXF

path = os.environ["PMDXF_PATH"]
doc = FreeCAD.newDocument("rt")
importDXF.insert(path, doc.Name)
doc.recompute()
circles, arcs, lines = [], 0, 0
for obj in doc.Objects:
    shape = getattr(obj, "Shape", None)
    if shape is None:
        continue
    for edge in shape.Edges:
        curve = edge.Curve
        if isinstance(curve, Part.Circle):
            if abs(edge.Length - 2 * math.pi * curve.Radius) < 1e-6:
                circles.append((round(curve.Center.x, 6), round(curve.Center.y, 6),
                                round(curve.Radius, 6)))
            else:
                arcs += 1
        elif isinstance(curve, Part.Line):
            lines += 1
print("PMDXF:" + json.dumps({"circles": sorted(circles), "arcs": arcs, "lines": lines}))
"""


def _marker_json(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("PMDXF:"):
            return json.loads(line[len("PMDXF:"):])
    raise AssertionError("no PMDXF: line in output:\n" + output[-2000:])


@unittest.skipUnless(_python_with_ezdxf(), "no interpreter with ezdxf (build/brep-venv)")
class TestEzdxf(unittest.TestCase):
    """ezdxf is an independent, strict R2000 implementation: its audit is the structure gate."""

    def test_audit_reports_no_errors_and_the_same_entities(self):
        views, groups = _plate_views()
        with tempfile.TemporaryDirectory() as tmp:
            result, path = _write(views, groups, Path(tmp))
            probe = subprocess.run([_python_with_ezdxf(), "-c", _EZDXF_SCRIPT, str(path)],
                                   capture_output=True, text=True, timeout=180)
        self.assertEqual(probe.returncode, 0, probe.stderr[-2000:])
        report = _marker_json(probe.stdout)
        self.assertEqual(report["version"], "AC1015")
        self.assertEqual(report["insunits"], 4)
        self.assertEqual(report["measurement"], 1)
        self.assertEqual(report["errors"], [], "ezdxf audit must find nothing to complain about")
        self.assertEqual(report["fixes"], [], "ezdxf audit must find nothing to repair")
        self.assertEqual(report["counts"], result.entity_counts)
        self.assertEqual(report["dimensions"], result.dimensions)
        self.assertEqual(len(report["circles"]), result.circles)
        self.assertIn(dxf.TITLEBLOCK_BLOCK, report["blocks"])


@unittest.skipUnless(_freecadcmd(), "freecadcmd is not installed")
class TestFreeCAD(unittest.TestCase):
    """A real CAD kernel must read our holes back as analytic circles, not as polygons."""

    def test_round_trip_preserves_circles_and_arcs(self):
        views, groups = _plate_views()
        with tempfile.TemporaryDirectory() as tmp:
            result, path = _write(views, groups, Path(tmp))
            script = Path(tmp) / "roundtrip.py"
            script.write_text(_FREECAD_SCRIPT, encoding="utf-8")
            environment = dict(os.environ, PMDXF_PATH=str(path), QT_QPA_PLATFORM="offscreen")
            for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY",
                         "http_proxy"):
                environment.pop(name, None)
            probe = subprocess.run([_freecadcmd(), str(script)], capture_output=True, text=True,
                                   timeout=300, env=environment)
            facts = dxf.inspect_dxf(path)
        self.assertEqual(probe.returncode, 0, probe.stderr[-2000:])
        report = _marker_json(probe.stdout)
        self.assertEqual(len(report["circles"]), result.circles,
                         "a CIRCLE must survive as a full analytic circle")
        self.assertEqual(report["arcs"], len(facts.arcs))
        expected = sorted((round(x, 6), round(y, 6), round(radius, 6))
                          for _, x, y, radius in facts.circles)
        for (ax, ay, ar), (bx, by, br) in zip(expected, report["circles"]):
            self.assertAlmostEqual(ax, bx, delta=1e-4)
            self.assertAlmostEqual(ay, by, delta=1e-4)
            self.assertAlmostEqual(ar, br, delta=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
