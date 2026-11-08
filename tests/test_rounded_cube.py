"""The `rounded_cube` node: one AST node, two deliberately different renderings.

The node exists because rounding an edge is a *fillet*, and the previous always-hull form threw
the curvature away — a 60x40x25 rounded box exported as 130 planar faces with **zero**
cylindrical surfaces, and a real enclosure as 9,174 faces in a 9.7 MB STEP nobody can edit.

Two properties matter and both are pinned here:

* **OpenSCAD output must not change.** It has no fillet, so it desugars to the historical hull.
  A regression would silently alter every existing design, so the desugar is compared against
  the hull construction written out longhand rather than against itself.
* **FreeCAD must produce real curved faces.** Asserted on the STEP entity counts, because
  "it rendered" says nothing about whether the corners are round or faceted.

Run both ways:
    /usr/bin/python3 tests/test_rounded_cube.py
    pytest tests/test_rounded_cube.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phimech.backends import scad_emit  # noqa: E402
from phimech.csg import ops  # noqa: E402
from phimech.csg.ast import Hull, LinearExtrude, RoundedCube, RoundedSquare, from_dict  # noqa: E402
from phimech.csg.validate import validate  # noqa: E402


def _freecad_available() -> bool:
    import os

    explicit = os.environ.get("PHIMECH_FREECADCMD")
    if explicit:
        return Path(explicit).exists()
    default = Path("/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd")
    return default.exists() or bool(shutil.which("freecadcmd"))


class TestNode(unittest.TestCase):
    def test_rounded_box_returns_a_rounded_cube(self):
        node = ops.rounded_box(60, 40, 25, 3)
        self.assertEqual(node.kind, "rounded_cube")
        self.assertEqual(node.size, (60.0, 40.0, 25.0))
        self.assertEqual(node.r, 3.0)

    def test_zero_radius_stays_a_plain_cube(self):
        """No fillet means no reason to pay for one."""
        self.assertEqual(ops.rounded_box(10, 10, 10, 0).kind, "cube")
        self.assertEqual(ops.rounded_box(10, 10, 10, 0, edge_r=0).kind, "cube")

    def test_round_trips_through_to_dict(self):
        node = ops.rounded_box(60, 40, 25, 3, edge_r=1.5)
        self.assertEqual(from_dict(node.to_dict()), node)

    def test_validate_accepts_it(self):
        """A new kind must be added to the validator too, or every render refuses it."""
        self.assertEqual(validate(ops.rounded_box(60, 40, 25, 3)), [])

    def test_radii_are_clamped_to_what_the_box_can_take(self):
        """An over-large radius must degrade, not fail a whole build."""
        vertical, horizontal = RoundedCube(size=(10, 10, 10), r=99).radii()
        self.assertEqual(vertical, 5.0)
        self.assertEqual(horizontal, 0.0)

    def test_edge_r_alone_rounds_every_edge(self):
        """With no `r`, the horizontal radius applies to the vertical edges as well.

        This mirrors what the old hull did: with `edge_r` set and `r` zero it emitted corner
        spheres and no vertical cylinders, which rounds all twelve edges by `edge_r`.
        """
        vertical, horizontal = RoundedCube(size=(60, 40, 25), r=0, edge_r=2).radii()
        self.assertEqual(vertical, 2.0)
        self.assertEqual(horizontal, 2.0)

    def test_negative_radius_is_treated_as_zero(self):
        vertical, horizontal = RoundedCube(size=(60, 40, 25), r=-5, edge_r=-1).radii()
        self.assertEqual((vertical, horizontal), (0.0, 0.0))


class TestOpenScadIsUnchanged(unittest.TestCase):
    """The no-regression half. OpenSCAD has no fillet, so it must emit exactly what it did."""

    def test_desugar_matches_the_historical_hull_exactly(self):
        node = ops.rounded_box(60, 40, 25, 3)
        expected = Hull([ops.cyl(6.0, 25.0, at=(x, y, 0.0))
                         for x, y in ((3, 3), (57, 3), (57, 37), (3, 37))])
        self.assertEqual(ops.rounded_box_as_hull(node).to_dict(), expected.to_dict())

    def test_scad_emit_of_the_node_equals_scad_emit_of_the_hull(self):
        node = ops.rounded_box(60, 40, 25, 3)
        hull = ops.rounded_box_as_hull(node)
        self.assertEqual(scad_emit.emit(node), scad_emit.emit(hull))

    def test_scad_emit_contains_a_hull_not_a_fillet(self):
        text = scad_emit.emit(ops.rounded_box(60, 40, 25, 3))
        self.assertIn("hull()", text)
        self.assertNotIn("fillet", text)

    def test_edge_rounded_variant_also_desugars(self):
        node = ops.rounded_box(60, 40, 25, 3, edge_r=2)
        hull = ops.rounded_box_as_hull(node)
        self.assertEqual(hull.kind, "hull")
        self.assertGreater(len(hull.children), 8)      # spheres at the corners, plus edges
        self.assertIn("hull()", scad_emit.emit(node))


@unittest.skipUnless(_freecad_available(), "freecadcmd not installed")
class TestFreeCadFillets(unittest.TestCase):
    """The half that justifies the node existing at all."""

    def _render(self, node, name="rc.step"):
        from phimech.backends import base

        out = Path(tempfile.mkdtemp()) / name
        base.get("freecad").render(node, out)
        from phimech.interop import inspect

        return inspect(out)

    def test_rounded_box_has_real_cylindrical_faces(self):
        facts = self._render(ops.rounded_box(60, 40, 25, 3))
        self.assertEqual(facts.faces, 10, "a rounded box is 6 planes + 4 cylinders")
        self.assertEqual(facts.entity_counts.get("CYLINDRICAL_SURFACE"), 4)
        self.assertEqual(facts.entity_counts.get("PLANE"), 6)
        self.assertFalse(facts.looks_tessellated)

    def test_it_is_far_smaller_than_the_hull_form(self):
        """The hull form of this exact box measured 192,647 bytes and 130 faces."""
        facts = self._render(ops.rounded_box(60, 40, 25, 3))
        self.assertLess(facts.size_bytes, 60_000)
        self.assertLess(facts.faces, 20)

    def test_fully_rounded_box_produces_curved_corner_blends(self):
        """Where a vertical fillet meets a horizontal one there must be a curved blend.

        Measured: 6 planes + 12 cylinders + 8 SURFACE_OF_REVOLUTION. OCC emits a
        TOROIDAL_SURFACE only when the two radii are *equal*; with r=3 and edge_r=2 the blend is
        variable-radius and becomes a surface of revolution. Both are genuine curved B-rep, so
        this asserts on `curved_surfaces` rather than on one entity name -- an earlier version of
        this test demanded a torus and failed on perfectly correct geometry.
        """
        facts = self._render(ops.rounded_box(60, 40, 25, 3, edge_r=2), name="rc2.step")
        self.assertEqual(facts.entity_counts.get("PLANE"), 6)
        self.assertEqual(facts.entity_counts.get("CYLINDRICAL_SURFACE"), 12)
        self.assertEqual(facts.revolved_surfaces, 8)
        self.assertEqual(facts.curved_surfaces, 20)
        self.assertFalse(facts.looks_tessellated)

    def test_equal_radii_give_spherical_corners(self):
        """With one radius everywhere the eight corners are exact sphere octants.

        This is the geometrically correct answer and worth pinning, because the intuitive guess
        is a torus: three fillets of *equal* radius meeting at a corner form part of a sphere,
        while unequal radii give a surface of revolution. Measured: 6 planes + 12 cylinders + 8
        spherical surfaces, and not one facet.
        """
        facts = self._render(ops.rounded_box(60, 40, 25, 2, edge_r=2), name="rc4.step")
        self.assertEqual(facts.entity_counts.get("SPHERICAL_SURFACE"), 8)
        self.assertEqual(facts.entity_counts.get("PLANE"), 6)
        self.assertEqual(facts.curved_surfaces, 20)
        self.assertFalse(facts.looks_tessellated)

    def test_volume_matches_the_analytic_answer(self):
        """A filleted box has a known volume; a faceted approximation does not hit it."""
        from phimech.backends import base

        node = ops.rounded_box(60, 40, 25, 3)
        measurement = base.get("freecad").measure(node)
        # Box minus the four corner offcuts: each is r^2*(1 - pi/4) of cross-section, times h.
        expected = 60 * 40 * 25 - 4 * (3.0 ** 2) * (1 - 3.141592653589793 / 4) * 25
        self.assertAlmostEqual(measurement.volume, expected, delta=0.5)

    def test_lane_agreement_and_the_one_place_they_diverge(self):
        """Pin how far the two lanes may disagree, per case, with the numbers.

        OpenSCAD has no fillet, so its lane approximates a rounded box with a convex hull. That
        approximation is exact enough to ignore in two cases and measurably wrong in a third:

            rounded_box(60,40,25, r, edge_r)   freecad      openscad     delta
            r=3, edge_r=0                     59806.858    59805.723    0.002%
            r=2, edge_r=2  (equal radii)      59581.510    59580.496    0.002%
            r=3, edge_r=1  (MIXED radii)      59723.831    59893.622   -0.283%

        The fillet is the authoritative shape, not a regression: it matches the analytic volume
        exactly for edge_r=0, and **build123d -- a second, independent OCC binding -- returns
        59723.831 for the mixed case too**, agreeing with FreeCAD to the last decimal. The hull
        is 0.283% oversized because the convex hull spanning a radius-3 vertical cylinder and a
        radius-1 corner sphere is a taper, not a fillet.

        Before the `rounded_cube` node both lanes went through the hull, so they agreed *on the
        approximation*. The consequence to know: for a mixed-radius rounded box the printed STL
        and the exported STEP now differ by ~0.3%. The print is unchanged; the CAD hand-off got
        more exact. This test exists so that gap can never widen unnoticed.
        """
        from phimech.backends import base

        openscad = base.get("openscad")
        if not getattr(openscad, "available", lambda: False)():
            self.skipTest("openscad not installed")
        for (r, edge_r), tolerance_pct in ((( 3, 0.0), 0.01), ((2, 2.0), 0.01), ((3, 1.0), 0.30)):
            with self.subTest(r=r, edge_r=edge_r):
                node = ops.rounded_box(60, 40, 25, r, edge_r=edge_r)
                fillet = base.get("freecad").measure(node).volume
                hull = openscad.measure(node).volume
                delta_pct = abs(fillet - hull) / hull * 100.0
                self.assertLessEqual(
                    delta_pct, tolerance_pct,
                    "r={} edge_r={}: lanes differ by {:.3f}% (fillet {:.3f}, hull {:.3f})".format(
                        r, edge_r, delta_pct, fillet, hull))

    def test_fillet_matches_a_second_independent_kernel(self):
        """The mixed-radius volume is not a FreeCAD quirk.

        Skipped unless the optional B-rep venv is present, because the whole point is to compare
        two different OCC bindings rather than to re-measure one.
        """
        from phimech.brep import kernel as brep_kernel

        if not brep_kernel.available():
            self.skipTest("no build123d; nothing independent to compare against")
        from phimech.backends import base

        node = ops.rounded_box(60, 40, 25, 3, edge_r=1.0)
        freecad_volume = base.get("freecad").measure(node).volume
        program = {"ops": [{"op": "box", "size": [60, 40, 25]},
                           {"op": "fillet", "select": "vertical", "radius": 3},
                           {"op": "fillet", "select": "all", "radius": 1}]}
        import tempfile

        out = Path(tempfile.mkdtemp()) / "mixed.step"
        result = brep_kernel.build(program, {"step": str(out)})
        self.assertAlmostEqual(result.measurements["volume_mm3"], freecad_volume, delta=0.01)

    def test_a_radius_too_large_degrades_instead_of_failing(self):
        """Losing one part's corners beats losing the whole build."""
        facts = self._render(RoundedCube(size=(10, 10, 40), r=5.0), name="rc3.step")
        self.assertEqual(facts.body_kind, "solid")
        self.assertGreater(facts.faces, 0)


class TestRoundedSquare(unittest.TestCase):
    """The 2D sibling, which fixed a worse bug than `rounded_cube` did.

    `Hull([circles])` is not merely faceted in FreeCAD -- it is unbuildable, because a hull of
    coplanar points has no volume and qhull reports a flat initial simplex. So one `rounded_rect`
    anywhere pushed the ENTIRE part onto the mesh-to-solid fallback: `rpi-zero-case-base` came out
    as 13,758 faces in a 13 MB file, and is now 261 faces in 354 KB.
    """

    def test_rounded_rect_returns_a_rounded_square(self):
        node = ops.rounded_rect(40, 20, 5)
        self.assertEqual(node.kind, "rounded_square")
        self.assertEqual(node.size, (40.0, 20.0))

    def test_zero_radius_stays_a_plain_square(self):
        self.assertEqual(ops.rounded_rect(40, 20, 0).kind, "square")

    def test_radius_is_clamped(self):
        self.assertEqual(RoundedSquare(size=(40, 20), r=99).radius(), 10.0)

    def test_round_trips_and_validates(self):
        node = ops.rounded_rect(40, 20, 5)
        self.assertEqual(from_dict(node.to_dict()), node)
        self.assertEqual(validate(LinearExtrude(10.0, [node])), [])

    def test_desugar_matches_the_historical_hull(self):
        node = ops.rounded_rect(40, 20, 5)
        from phimech.csg.ast import Circle, Translate

        expected = Hull([Translate((x, y, 0.0), [Circle(5.0)])
                         for x, y in ((5, 5), (35, 5), (35, 15), (5, 15))])
        self.assertEqual(ops.rounded_rect_as_hull(node).to_dict(), expected.to_dict())

    def test_scad_emit_is_unchanged(self):
        node = ops.rounded_rect(40, 20, 5)
        self.assertEqual(scad_emit.emit(node), scad_emit.emit(ops.rounded_rect_as_hull(node)))

    def test_slot_uses_it(self):
        """`slot()` is the most common caller and must benefit without being touched."""
        node = ops.slot(30, 8, 10)
        self.assertEqual(node.children[0].kind, "rounded_square")


@unittest.skipUnless(_freecad_available(), "freecadcmd not installed")
class TestFreeCadRoundedSquare(unittest.TestCase):
    def test_extruded_rounded_rect_has_cylindrical_sides(self):
        """The whole point: an extruded rounded profile is a prism with real curved sides."""
        from phimech.backends import base
        from phimech.interop import inspect

        out = Path(tempfile.mkdtemp()) / "rr.step"
        base.get("freecad").render(LinearExtrude(10.0, [ops.rounded_rect(40, 20, 5)]), out)
        facts = inspect(out)
        self.assertEqual(facts.faces, 10)
        self.assertEqual(facts.entity_counts.get("CYLINDRICAL_SURFACE"), 4)
        self.assertFalse(facts.looks_tessellated)

    def test_a_part_containing_one_no_longer_falls_back_to_mesh(self):
        """Regression on the real failure: a planar hull used to poison the whole part."""
        from phimech.backends import base
        from phimech.csg.ast import Difference
        from phimech.interop import inspect

        node = Difference([
            ops.rounded_box(60, 40, 20, 3),
            ops.slot(30, 8, 30),
        ])
        out = Path(tempfile.mkdtemp()) / "mixed.step"
        base.get("freecad").render(node, out)
        facts = inspect(out)
        self.assertFalse(facts.looks_tessellated)
        self.assertLess(facts.faces, 100)
        self.assertGreater(facts.curved_surfaces, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
