"""Tests for `phimech.interop.step` — the SolidWorks-readiness parse.

Runs two ways, per house rule:
    /usr/bin/python3 tests/test_interop.py
    pytest tests/test_interop.py

The interesting cases are the *failures*: a tessellated STEP, an inch STEP and a
surfaces-only STEP all "open fine" in any CAD tool, and all three are useless as solids.
A checker that only proves the happy path would not have caught any of them.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phimech.interop import inspect, readiness  # noqa: E402


def _step(entities: str, *, schema: str = "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }",
          unit: str = "SI_UNIT(.MILLI.,.METRE.)", system: str = "'PhiMech'") -> str:
    """Assemble a minimal but structurally honest STEP file."""
    return f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test model'),'2;1');
FILE_NAME('t','2026-01-01T00:00:00',('a'),('o'),'preproc 1.0',{system},'none');
FILE_SCHEMA(('{schema}'));
ENDSEC;
DATA;
#1 = ( LENGTH_UNIT() NAMED_UNIT(*) {unit} );
{entities}
ENDSEC;
END-ISO-10303-21;
"""


SOLID_BODY = """#10 = ADVANCED_FACE('',(#11),#12,.T.);
#11 = FACE_OUTER_BOUND('',#13,.T.);
#12 = PLANE('',#14);
#15 = CLOSED_SHELL('',(#10));
#16 = MANIFOLD_SOLID_BREP('body',#15);
#17 = EDGE_CURVE('',#18,#19,#20,.T.);
"""

MESH_BODY = """#10 = TRIANGULATED_FACE_SET('',(#11),(#12),(#13),(#14));
#11 = CARTESIAN_POINT('',(0.,0.,0.));
"""

SURFACES_ONLY = """#10 = ADVANCED_FACE('',(#11),#12,.T.);
#12 = PLANE('',#14);
#13 = ADVANCED_FACE('',(#15),#16,.T.);
#16 = PLANE('',#17);
"""


class TestInspect(unittest.TestCase):
    def _write(self, text: str) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "m.step"
        tmp.write_text(text)
        return tmp

    def test_solid_is_recognised(self):
        f = inspect(self._write(_step(SOLID_BODY)))
        self.assertEqual(f.protocol, "214")
        self.assertEqual(f.body_kind, "solid")
        self.assertEqual(f.solids, 1)
        self.assertEqual(f.closed_shells, 1)
        self.assertEqual(f.faces, 1)
        self.assertEqual(f.edges, 1)
        self.assertEqual(f.length_unit, "mm")
        self.assertEqual(f.mm_per_unit, 1.0)
        self.assertEqual(f.originating_system, "PhiMech")
        self.assertIsNone(f.parse_error)

    def test_readiness_passes_a_millimetre_solid(self):
        r = readiness(inspect(self._write(_step(SOLID_BODY))))
        self.assertIs(r["ready"], True)
        self.assertIn("solid body", r["summary"])
        # The caveat must always ride along: this is not a topology check.
        self.assertIn("not a topology check", r["caveat"])

    def test_tessellated_step_is_refused(self):
        """A mesh STEP opens fine and is useless as a solid — the check must say so."""
        f = inspect(self._write(_step(MESH_BODY)))
        self.assertEqual(f.body_kind, "mesh")
        r = readiness(f)
        self.assertIs(r["ready"], False)
        finding = [x for x in r["findings"] if x["name"] == "imports_as_solid"][0]
        self.assertIs(finding["ok"], False)
        self.assertIn("graphics/mesh body", finding["detail"])

    def test_surfaces_without_a_closed_shell_are_refused(self):
        f = inspect(self._write(_step(SURFACES_ONLY)))
        self.assertEqual(f.body_kind, "surfaces")
        self.assertEqual(f.faces, 2)
        self.assertIs(readiness(f)["ready"], False)

    def test_empty_geometry_is_refused(self):
        f = inspect(self._write(_step("#10 = CARTESIAN_POINT('',(0.,0.,0.));")))
        self.assertEqual(f.body_kind, "empty")
        self.assertIs(readiness(f)["ready"], False)

    def test_inch_units_fail_loudly(self):
        """The classic silent 25.4x error must be a hard fail, not a warning."""
        text = _step(SOLID_BODY, unit="CONVERSION_BASED_UNIT('INCH',#99)")
        f = inspect(self._write(text))
        self.assertEqual(f.length_unit, "in")
        self.assertEqual(f.mm_per_unit, 25.4)
        r = readiness(f)
        self.assertIs(r["ready"], False)
        detail = [x for x in r["findings"] if x["name"] == "units_declared"][0]["detail"]
        self.assertIn("25.4", detail)

    def test_metre_units_are_detected(self):
        f = inspect(self._write(_step(SOLID_BODY, unit="SI_UNIT($,.METRE.)")))
        self.assertEqual(f.length_unit, "m")
        self.assertEqual(f.mm_per_unit, 1000.0)

    def test_unknown_protocol_fails(self):
        f = inspect(self._write(_step(SOLID_BODY, schema="STRUCTURAL_FRAME_SCHEMA")))
        self.assertIsNone(f.protocol)
        r = readiness(f)
        # No schema recognised is "unknown", which must not render as ready.
        self.assertIsNot(r["ready"], True)

    def test_ap242_is_supported(self):
        f = inspect(self._write(_step(SOLID_BODY, schema="AP242_MANAGED_MODEL_BASED_3D_ENGINEERING "
                                                        "{ 1 0 10303 442 1 1 4 }")))
        self.assertEqual(f.protocol, "442")  # the schema string's own number wins

    def test_ready_is_tri_state_not_boolean(self):
        """An undeterminable fact must yield None, never True. Same rule as the gates."""
        text = _step(SOLID_BODY).replace("FILE_SCHEMA(('AUTOMOTIVE_DESIGN "
                                         "{ 1 0 10303 214 1 1 1 1 }'));", "")
        r = readiness(inspect(self._write(text)))
        self.assertIsNone(r["ready"])
        self.assertIn("could not confirm", r["summary"])

    def test_entity_names_in_string_literals_do_not_count(self):
        """A part named 'TRIANGULATED_BRACKET' must not make a solid look tessellated."""
        body = SOLID_BODY + "#20 = PRODUCT('TRIANGULATED_FACE_SET','desc',(#21));\n"
        f = inspect(self._write(_step(body)))
        self.assertEqual(f.mesh_entities, 0)
        self.assertEqual(f.body_kind, "solid")
        self.assertIs(readiness(f)["ready"], True)

    def test_comments_do_not_count(self):
        body = SOLID_BODY + "/* #99 = TRIANGULATED_FACE_SET('',(#1)); */\n"
        f = inspect(self._write(_step(body)))
        self.assertEqual(f.mesh_entities, 0)

    def test_missing_file_reports_cleanly(self):
        f = inspect(Path("/nonexistent/nope.step"))
        self.assertIsNotNone(f.parse_error)
        r = readiness(f)
        self.assertIs(r["ready"], False)
        self.assertIn("not a readable STEP", r["summary"])

    def test_garbage_file_does_not_raise(self):
        p = Path(tempfile.mkdtemp()) / "junk.step"
        p.write_bytes(b"\x00\x01\x02 not step at all \xff")
        f = inspect(p)                      # must not raise
        self.assertEqual(f.body_kind, "empty")
        self.assertIs(readiness(f)["ready"], False)

    def test_to_dict_is_json_serialisable(self):
        import json
        f = inspect(self._write(_step(SOLID_BODY)))
        json.dumps(f.to_dict())             # must not raise
        json.dumps(readiness(f))

    def test_analytic_and_spline_surfaces_are_separated(self):
        body = SOLID_BODY + "#30 = CYLINDRICAL_SURFACE('',#31,3.0);\n" \
                            "#32 = B_SPLINE_SURFACE_WITH_KNOTS('',3,3,((#33)),.UNSPECIFIED.," \
                            ".F.,.F.,.F.,(4),(4),(0.),(1.),.UNSPECIFIED.);\n"
        f = inspect(self._write(_step(body)))
        self.assertEqual(f.analytic_surfaces, 2)   # PLANE + CYLINDRICAL_SURFACE
        self.assertEqual(f.spline_surfaces, 1)
        r = readiness(f)
        names = [x["name"] for x in r["findings"]]
        self.assertIn("analytic_surfaces", names)

    def test_spline_curves_are_not_counted_as_surfaces(self):
        """B_SPLINE_CURVE is an edge, not a surface. Grep conflates them; this must not."""
        body = SOLID_BODY + "#40 = B_SPLINE_CURVE_WITH_KNOTS('',3,(#41),.UNSPECIFIED.,.F.,.F.);\n"
        f = inspect(self._write(_step(body)))
        self.assertEqual(f.spline_surfaces, 0)


def _tessellated_solid(n_faces: int, *, planar: int = None, edges: int = None) -> str:
    """A closed shell of `n_faces` triangles, as a mesh->B-rep converter would emit it."""
    planar = n_faces if planar is None else planar
    edges = int(n_faces * 1.5) if edges is None else edges   # closed triangulation: E = 1.5F
    lines = ["#1 = CLOSED_SHELL('',(#2));", "#2 = MANIFOLD_SOLID_BREP('m',#1);"]
    for i in range(n_faces):
        lines.append(f"#{100 + i} = ADVANCED_FACE('',(#1),#2,.T.);")
    for i in range(planar):
        lines.append(f"#{200000 + i} = PLANE('',#3);")
    for i in range(edges):
        lines.append(f"#{400000 + i} = EDGE_CURVE('',#4,#5,#6,.T.);")
    return "\n".join(lines) + "\n"


class TestTessellationDetection(unittest.TestCase):
    """A mesh converted face-by-face into a solid is the trap this must catch.

    It is a genuine MANIFOLD_SOLID_BREP, so it imports without error and `body_kind` says
    `solid`. What fails is every CAD operation afterwards.
    """

    def _facts(self, text: str):
        tmp = Path(tempfile.mkdtemp()) / "m.step"
        tmp.write_text(text)
        return inspect(tmp)

    def test_large_all_planar_solid_is_flagged(self):
        f = self._facts(_step(_tessellated_solid(9174)))
        self.assertEqual(f.body_kind, "solid")        # it really is a solid
        self.assertTrue(f.looks_tessellated)          # and it is really a mesh
        self.assertAlmostEqual(f.planar_fraction, 1.0, places=3)
        r = readiness(f)
        finding = [x for x in r["findings"] if x["name"] == "true_brep"][0]
        self.assertIs(finding["ok"], False)
        self.assertEqual(finding["severity"], "warn")
        self.assertIn("FACETED", r["summary"])

    def test_faceted_file_is_still_transferable(self):
        """It imports, so `ready` stays True -- the warning must not become a hard fail."""
        r = readiness(self._facts(_step(_tessellated_solid(9174))))
        self.assertIs(r["ready"], True)

    def test_small_prismatic_part_is_not_flagged(self):
        """The false-positive guard: a real machined part is all-planar with few faces."""
        f = self._facts(_step(_tessellated_solid(12, edges=24)))
        self.assertFalse(f.looks_tessellated)
        r = readiness(f)
        finding = [x for x in r["findings"] if x["name"] == "true_brep"][0]
        self.assertIs(finding["ok"], True)

    def test_curved_part_with_many_faces_is_not_flagged(self):
        """Many faces alone is not tessellation -- curvature means it was modelled."""
        body = _tessellated_solid(600, planar=100)
        body += "".join(f"#{700000+i} = CYLINDRICAL_SURFACE('',#9,2.0);\n" for i in range(500))
        f = self._facts(_step(body))
        self.assertLess(f.planar_fraction, 0.95)
        self.assertFalse(f.looks_tessellated)

    def test_face_edge_ratio_outside_the_triangulation_band_is_not_flagged(self):
        """A prismatic part with many planar faces but no 2/3 ratio is genuine geometry."""
        f = self._facts(_step(_tessellated_solid(400, edges=1600)))   # ratio 0.25
        self.assertFalse(f.looks_tessellated)

    def test_real_faceted_build_output_if_present(self):
        """If a synth build is lying around, assert the detector fires on the real thing."""
        build_dir = Path("/tmp/sw-build")
        candidates = sorted(build_dir.glob("*.step")) if build_dir.is_dir() else []
        if not candidates:
            self.skipTest("no /tmp/sw-build output to check")
        f = inspect(candidates[0])
        self.assertTrue(f.looks_tessellated, f"{candidates[0]} should read as faceted")


class TestRealFiles(unittest.TestCase):
    """If a real STEP happens to be lying around from a build, assert on it too."""

    def test_repo_fixture_if_present(self):
        candidates = sorted(Path(__file__).resolve().parent.parent.glob("fixtures/**/*.step"))
        if not candidates:
            self.skipTest("no committed .step fixture yet")
        for p in candidates:
            f = inspect(p)
            self.assertIsNone(f.parse_error, f"{p}: {f.parse_error}")
            self.assertEqual(f.body_kind, "solid", f"{p} is {f.body_kind}, not a solid")
            self.assertEqual(f.length_unit, "mm", f"{p} is in {f.length_unit}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
