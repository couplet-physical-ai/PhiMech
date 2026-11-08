"""Tests for the B-rep lane.

Two halves, deliberately:

* `TestProgram` is stdlib-only and always runs, including in CI where no kernel is installed.
  It is the important half: validation is what stops a malformed request from ever reaching a
  subprocess, so every rejection path is pinned here.
* `TestKernel` self-skips unless an interpreter with `build123d` is available. When it does run
  it asserts on *numbers* — face counts and volumes — because "it built something" is not
  evidence that the something is right.

Run both ways:
    /usr/bin/python3 tests/test_brep.py
    pytest tests/test_brep.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phimech.brep import ProgramError, describe, validate  # noqa: E402
from phimech.brep import kernel as brep_kernel  # noqa: E402
from phimech.brep.program import THREAD_PITCH  # noqa: E402

def _example_programs():
    """Shipped feature programs.

    They live in `examples/brep/`, not `examples/`, because `examples/*.json` is validated as
    design IR by `bin/phimech check` and a feature program is a different schema entirely.
    Putting them in the same directory made the CAD gate fail, which is the gate working.
    """
    return sorted((Path(__file__).resolve().parent.parent / "examples" / "brep").glob("*.json"))


ROUNDED_BOX = {
    "version": 1, "units": "mm",
    "ops": [{"op": "box", "size": [60, 40, 25]},
            {"op": "fillet", "select": "vertical", "radius": 3}],
}

#: The three counterbored holes shared by the L-bracket and the flat plate below. Identical
#: numbers on purpose: the flat plate is the no-regression control for the L-bracket's fix.
COUNTERBORED_HOLES = {
    "op": "hole", "at": [[20, 10], [20, 30], [45, 20]], "diameter": 5.2,
    "counterbore": {"diameter": 9.5, "depth": 4},
}

#: The same 60x40x8 plate the L-bracket is built on, on its own. Every feature here IS on the
#: topmost face of the whole solid, so it is the case the old bounding-box anchoring got right
#: by luck — and therefore the case that must not move.
FLAT_PLATE = {
    "version": 1, "units": "mm",
    "ops": [{"op": "box", "size": [60, 40, 8]}, dict(COUNTERBORED_HOLES)],
}


def _l_bracket(*features):
    """A 60x40x8 plate under a 8x40x45 upstand, plus whatever features are asked for.

    The shape that exposed the bug: the solid's bounding-box z_max is 45, at the top of the
    upstand, while every feature in the plate enters a face at z = 8.
    """
    return {"version": 1, "units": "mm", "ops": [
        {"op": "box", "size": [60, 40, 8]},
        {"op": "box", "size": [8, 40, 45], "at": [0, 0, 0]},
    ] + [dict(f) for f in features]}


def _example(name):
    return json.loads(
        (Path(__file__).resolve().parent.parent / "examples" / "brep" / name).read_text()
    )


SHELL_BOX = {"ops": [{"op": "box", "size": [50, 50, 20]},
                     {"op": "shell", "thickness": 2, "open_face": "top"}]}
CB_PLATE = {"ops": [{"op": "box", "size": [60, 40, 8]},
                    {"op": "hole", "at": [[10, 10], [50, 30]], "diameter": 3.2,
                     "counterbore": {"diameter": 6, "depth": 3}}]}
CSK_PLATE = {"ops": [{"op": "box", "size": [40, 40, 10]},
                     {"op": "hole", "at": [[20, 20]], "diameter": 4,
                      "countersink": {"diameter": 10, "depth": 3}}]}
CSK_BREACH = {"ops": [{"op": "box", "size": [40, 40, 4]},
                      {"op": "hole", "at": [[20, 20]], "diameter": 4,
                       "countersink": {"diameter": 10, "depth": 5}}]}
CUT_IN_AIR = {"ops": [{"op": "box", "size": [60, 40, 8]},
                      {"op": "cut_box", "size": [10, 10, 10], "at": [100, 0, 0]}]}
TAPPED_PLATE = {"ops": [{"op": "box", "size": [40, 30, 10]},
                        {"op": "tapped_hole", "at": [[10, 15], [30, 15]], "size": "M3",
                         "depth": 8}]}
BAD_FILLET = {"ops": [{"op": "box", "size": [10, 10, 10]},
                      {"op": "fillet", "select": "vertical", "radius": 50}]}

L_COUNTERBORED = _l_bracket(COUNTERBORED_HOLES)
L_PLAIN = _l_bracket({k: v for k, v in COUNTERBORED_HOLES.items() if k != "counterbore"})
L_TAPPED = _l_bracket({"op": "tapped_hole", "at": [[30, 20]], "size": "M4", "depth": 5})
#: The third point is off the part entirely, so its axis finds no face to enter.
L_HOLE_IN_AIR = _l_bracket(dict(COUNTERBORED_HOLES, at=[[20, 10], [20, 30], [45, 70]]))
#: A 4 mm counterbore is legal in the 8 mm plate and illegal in this 2 mm flange. Both are on
#: the same solid, so a gate reading the part's 45 mm Z extent would wave both through.
L_FLANGE_BREACH = _l_bracket({"op": "box", "size": [60, 8, 2], "at": [0, 40, 0]},
                             dict(COUNTERBORED_HOLES, at=[[30, 44]]))

#: Deliberately NOT batched. What these two pin is the code that batching would move out of
#: the way: the single-program request shape and its stdout contract, and the host mapping
#: from a worker gate refusal to `ProgramError`. Each costs its own kernel start, on purpose.
SINGLE_BOX = {"version": 1, "units": "mm", "ops": [{"op": "box", "size": [12, 10, 8]}]}


def _enclosure_breach():
    """`examples/brep/enclosure.json` with the defect put back: 3 mm into a 2 mm floor."""
    program = _example("enclosure.json")
    program["ops"][3]["counterbore"]["depth"] = 3
    return program


def _enclosure_plain():
    program = _example("enclosure.json")
    program["ops"][3].pop("counterbore")
    return program


def _program_key(program):
    return hashlib.sha256(json.dumps(program, sort_keys=True).encode("utf-8")).hexdigest()


def _batched_programs():
    """Every program `TestKernel` builds, so one kernel start can serve all of them.

    `import build123d` is ~3.2 s and a build is typically under one, so a subprocess per
    assertion spent its whole life starting up: 26 kernel starts for 47 tests. Membership here
    is a *speed* property only — `_build()` falls back to its own invocation for anything not
    listed, so a new test is never wrong, only slower, and this is where to make it fast again.
    """
    return [
        ROUNDED_BOX, SHELL_BOX, CB_PLATE, FLAT_PLATE, CSK_PLATE, CSK_BREACH, CUT_IN_AIR,
        TAPPED_PLATE, BAD_FILLET, L_COUNTERBORED, L_PLAIN, L_TAPPED, L_HOLE_IN_AIR,
        L_FLANGE_BREACH, _enclosure_plain(),
    ] + [json.loads(path.read_text()) for path in _example_programs()]


class TestProgram(unittest.TestCase):
    def test_minimal_program_validates(self):
        ops = validate(ROUNDED_BOX)
        self.assertEqual(len(ops), 2)

    def test_version_and_units_defaults(self):
        validate({"ops": [{"op": "box", "size": [1, 2, 3]}]})

    def test_non_mm_units_refused(self):
        """PhiMech is millimetres throughout; a unit field is not an invitation to convert."""
        with self.assertRaises(ProgramError) as cm:
            validate({"units": "in", "ops": [{"op": "box", "size": [1, 1, 1]}]})
        self.assertIn("millimetres", str(cm.exception))

    def test_future_version_refused(self):
        with self.assertRaises(ProgramError):
            validate({"version": 99, "ops": [{"op": "box", "size": [1, 1, 1]}]})

    def test_empty_ops_refused(self):
        for bad in ({"ops": []}, {"ops": "box"}, {}, [], "nope"):
            with self.assertRaises(ProgramError):
                validate(bad)

    def test_unknown_op_lists_the_known_ones(self):
        """A typo must be a named error, not an empty result."""
        with self.assertRaises(ProgramError) as cm:
            validate({"ops": [{"op": "filet", "radius": 1}]})
        message = str(cm.exception)
        self.assertIn("unknown op", message)
        self.assertIn("fillet", message)      # the correct spelling is offered

    def test_program_cannot_start_with_a_modifier(self):
        for op in ("fillet", "chamfer", "shell", "hole", "tapped_hole", "cut_box"):
            with self.assertRaises(ProgramError) as cm:
                validate({"ops": [{"op": op, "radius": 1, "length": 1, "thickness": 1,
                                   "diameter": 1, "size": "M3", "depth": 1, "at": [[0, 0]]}]})
            self.assertIn("no solid yet", str(cm.exception))

    def test_op_index_is_named_in_the_error(self):
        """A model fixing one op needs to know which one."""
        with self.assertRaises(ProgramError) as cm:
            validate({"ops": [{"op": "box", "size": [10, 10, 10]},
                              {"op": "fillet", "select": "sideways", "radius": 1}]})
        self.assertIn("ops[1]", str(cm.exception))

    def test_bad_selector_refused(self):
        with self.assertRaises(ProgramError) as cm:
            validate({"ops": [{"op": "box", "size": [10, 10, 10]},
                              {"op": "fillet", "select": "diagonal", "radius": 1}]})
        self.assertIn("vertical", str(cm.exception))

    def test_negative_and_zero_dimensions_refused(self):
        for size in ([0, 10, 10], [-5, 10, 10], [10, 10, 0]):
            with self.assertRaises(ProgramError):
                validate({"ops": [{"op": "box", "size": size}]})

    def test_non_numeric_dimension_refused(self):
        with self.assertRaises(ProgramError):
            validate({"ops": [{"op": "box", "size": ["60", 40, 25]}]})

    def test_booleans_are_not_numbers(self):
        """True == 1 in Python; a size of True is a bug, not a 1 mm box."""
        with self.assertRaises(ProgramError):
            validate({"ops": [{"op": "box", "size": [True, 10, 10]}]})

    def test_wrong_length_vector_refused(self):
        with self.assertRaises(ProgramError):
            validate({"ops": [{"op": "box", "size": [60, 40]}]})

    def test_counterbore_must_be_wider_than_the_hole(self):
        """A counterbore no wider than its hole cuts nothing — catch it before the kernel."""
        with self.assertRaises(ProgramError) as cm:
            validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                              {"op": "hole", "at": [[10, 10]], "diameter": 5,
                               "counterbore": {"diameter": 5, "depth": 2}}]})
        self.assertIn("cuts nothing", str(cm.exception))

    def test_countersink_narrower_than_hole_refused(self):
        with self.assertRaises(ProgramError):
            validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                              {"op": "hole", "at": [[10, 10]], "diameter": 6,
                               "countersink": {"diameter": 4, "depth": 2}}]})

    def test_counterbore_no_shallower_than_its_blind_hole_refused(self):
        """Both depths run down from the same face, so an equal counterbore swallows the hole."""
        for extra in ("counterbore", "countersink"):
            with self.assertRaises(ProgramError) as cm:
                validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                                  {"op": "hole", "at": [[10, 10]], "diameter": 5, "depth": 4,
                                   extra: {"diameter": 9, "depth": 4}}]})
            self.assertIn("must be less than the hole depth", str(cm.exception))

    def test_counterbore_shallower_than_its_blind_hole_is_fine(self):
        validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                          {"op": "hole", "at": [[10, 10]], "diameter": 5, "depth": 4,
                           "counterbore": {"diameter": 9, "depth": 2}}]})

    def test_a_through_hole_places_no_depth_constraint(self):
        """No `depth` means through, and nothing static can then judge the counterbore depth."""
        validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                          {"op": "hole", "at": [[10, 10]], "diameter": 5,
                           "counterbore": {"diameter": 9, "depth": 8}}]})

    def test_hole_needs_points(self):
        for at in ([], None, [[0]], [[0, 0, 0]], "0,0"):
            with self.assertRaises(ProgramError):
                validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                                  {"op": "hole", "at": at, "diameter": 3}]})

    def test_unknown_thread_size_refused(self):
        with self.assertRaises(ProgramError) as cm:
            validate({"ops": [{"op": "box", "size": [40, 40, 10]},
                              {"op": "tapped_hole", "at": [[10, 10]], "size": "M7",
                               "depth": 5}]})
        self.assertIn("M3", str(cm.exception))     # the supported set is offered

    def test_thread_pitches_are_standard(self):
        """Wrong pitch means a wrong pilot diameter and a stripped hole."""
        self.assertEqual(THREAD_PITCH["M3"], 0.5)
        self.assertEqual(THREAD_PITCH["M4"], 0.7)
        self.assertEqual(THREAD_PITCH["M5"], 0.8)

    def test_describe_is_human_readable(self):
        text = describe(ROUNDED_BOX)
        self.assertIn("box", text)
        self.assertIn("r3 on vertical edges", text)

    def test_relative_output_paths_are_rejected(self):
        """Rejected, not resolved: the worker's cwd is not the caller's."""
        with self.assertRaises(brep_kernel.BrepError) as cm:
            brep_kernel.build(ROUNDED_BOX, {"step": "out/relative.step"})
        self.assertIn("must be absolute", str(cm.exception))

    def test_invalid_program_fails_before_any_subprocess(self):
        """Validation must precede process launch, so a typo costs milliseconds."""
        with self.assertRaises(ProgramError):
            brep_kernel.build({"ops": [{"op": "nonsense"}]},
                              {"step": "/tmp/never-written.step"})
        self.assertFalse(Path("/tmp/never-written.step").exists())

    def test_shipped_examples_all_validate(self):
        """CI has no kernel, but it can still prove every shipped program is well-formed.

        This is the cheap half of "examples must always build": validation catches a typo or a
        renamed op the moment it lands, on a machine with no 500 MB kernel installed.
        """
        examples = _example_programs()
        self.assertTrue(examples, "no examples/brep/*.json programs found")
        for path in examples:
            with self.subTest(example=path.name):
                program = json.loads(path.read_text())
                validate(program)
                self.assertTrue(describe(program).strip())

    def test_doctor_never_raises(self):
        info = brep_kernel.doctor()
        self.assertIn("available", info)
        self.assertIn("source", info)
        json.dumps(info)


@unittest.skipUnless(brep_kernel.available(), "no interpreter with build123d; run `brep setup`")
class TestKernel(unittest.TestCase):
    """Numbers, not just "it ran"."""

    @classmethod
    def setUpClass(cls):
        """Build every declared program in ONE kernel start, and cache by program content.

        Nothing here weakens what is checked: each test still asserts against a real result
        from a real kernel, refusals included, and `raise_for()` is the same production code
        `build()` ends with, so a batched refusal surfaces as the identical exception. What
        goes away is only the ~3.2 s `import build123d` that each of 26 invocations was paying.
        """
        directory = Path(tempfile.mkdtemp(prefix="phimech-brep-batch-"))
        seen = {}
        for index, program in enumerate(_batched_programs()):
            key = _program_key(program)
            if key in seen:
                continue
            seen[key] = (program, directory / "p{:02d}.step".format(index))
        items = [{"program": program, "outputs": {"step": str(out)}}
                 for program, out in seen.values()]
        results = brep_kernel.build_many(items, timeout=900.0)
        cls._batch = {key: (result, out)
                      for key, result, (_, out) in zip(seen, results, seen.values())}

    def _build(self, program, name="out.step"):
        """This program's kernel result — from the class batch when it was declared there.

        Falls back to its own invocation otherwise, so membership of `_batched_programs()` can
        only ever cost time, never correctness.
        """
        cached = self._batch.get(_program_key(program))
        if cached is None:
            out = Path(tempfile.mkdtemp()) / name
            return brep_kernel.build(program, {"step": str(out)}), out
        result, out = cached
        return brep_kernel.raise_for(result), out

    def test_rounded_box_is_ten_faces_not_a_hull(self):
        """The whole reason this lane exists.

        A rounded box is 6 planes + 4 cylinders. `ops.rounded_box()` renders it as a convex
        hull, which measured 130 planar faces and zero cylindrical surfaces.
        """
        result, out = self._build(ROUNDED_BOX)
        self.assertTrue(result.ok)
        self.assertEqual(result.measurements["faces"], 10)
        self.assertEqual(result.measurements["solids"], 1)
        self.assertTrue(out.exists())

        from phimech.interop import inspect as step_inspect
        facts = step_inspect(out)
        self.assertEqual(facts.entity_counts.get("CYLINDRICAL_SURFACE"), 4)
        self.assertEqual(facts.entity_counts.get("PLANE"), 6)
        self.assertFalse(facts.looks_tessellated)
        self.assertEqual(facts.length_unit, "mm")

    def test_fillet_volume_is_less_than_the_box(self):
        result, _ = self._build(ROUNDED_BOX)
        self.assertLess(result.measurements["volume_mm3"], 60 * 40 * 25)
        self.assertGreater(result.measurements["volume_mm3"], 59000)

    def test_shell_hollows_the_solid(self):
        result, _ = self._build(SHELL_BOX)
        # A 2 mm shell of a 50x50x20 box is far lighter than the solid it came from.
        self.assertLess(result.measurements["volume_mm3"], 0.4 * 50 * 50 * 20)
        self.assertGreater(result.measurements["volume_mm3"], 0)

    def test_counterbored_holes_remove_material_and_keep_one_solid(self):
        result, _ = self._build(CB_PLATE)
        self.assertEqual(result.measurements["solids"], 1)
        self.assertLess(result.measurements["volume_mm3"], 60 * 40 * 8)

    # ----------------------------------------------------------------------------------
    # Where a blind feature lands. A flat plate cannot tell these apart, which is exactly
    # why counterbores were cut 33 mm above the L-bracket for as long as they were: every
    # kernel test here used a plate, and on a plate the top face IS the bounding-box top.
    # ----------------------------------------------------------------------------------

    def test_flat_plate_counterbores_are_unchanged(self):
        """The no-regression control: on a plate the feature really is on the topmost face.

        60*40*8 - 3*pi/4*5.2^2*8 - 3*pi/4*(9.5^2-5.2^2)*4 = 18094.57 mm3, analytically.
        """
        import math
        result, _ = self._build(FLAT_PLATE)
        expected = (60 * 40 * 8
                    - 3 * math.pi / 4 * 5.2 ** 2 * 8
                    - 3 * math.pi / 4 * (9.5 ** 2 - 5.2 ** 2) * 4)
        self.assertAlmostEqual(expected, 18094.57, delta=0.01)      # the arithmetic, pinned
        self.assertAlmostEqual(result.measurements["volume_mm3"], expected, delta=0.5)
        self.assertEqual(result.measurements["solids"], 1)

    def test_counterbore_below_the_top_of_the_part_actually_cuts(self):
        """The bug. The pockets belong in the 8 mm plate; the solid's z_max is 45.

        Anchored on the bounding box they landed at z 41..45, in free air, and the part
        measured *identically* with and without them — 29973.1504 mm3 either way.
        """
        import math
        annulus = 3 * math.pi / 4 * (9.5 ** 2 - 5.2 ** 2) * 4
        with_cb, _ = self._build(L_COUNTERBORED, name="cb.step")
        without, _ = self._build(L_PLAIN, name="nocb.step")
        removed = without.measurements["volume_mm3"] - with_cb.measurements["volume_mm3"]
        self.assertAlmostEqual(removed, annulus, delta=0.5,
                               msg="the counterbore removed {:.4f} mm3, not {:.4f}".format(
                                   removed, annulus))
        # Two new faces per hole: the pocket wall and its annular floor.
        self.assertEqual(with_cb.measurements["faces"],
                         without.measurements["faces"] + 6)

    def test_shipped_bracket_measures_what_its_features_declare(self):
        """The whole shipped program, pinned: 29973.1504 - 595.7402 = 29377.4102 mm3."""
        result, _ = self._build(_example("bracket.json"), name="bracket.step")
        self.assertAlmostEqual(result.measurements["volume_mm3"], 29377.5, delta=1.0)
        self.assertEqual(result.measurements["faces"], 35)
        self.assertEqual(result.measurements["solids"], 1)

    def test_tapped_hole_below_the_top_of_the_part_actually_cuts(self):
        """An M4 pilot in the plate, not the upstand. Anchored on z_max it cut nothing.

        31040 mm3 of L minus pi*(3.24225/2)^2*5 of pilot.
        """
        import math
        result, _ = self._build(L_TAPPED, name="tapped.step")
        pilot = 4.0 - 1.0825 * 0.7
        expected = 31040.0 - math.pi * (pilot / 2) ** 2 * 5
        self.assertAlmostEqual(result.measurements["volume_mm3"], expected, delta=0.5)

    def test_countersink_opens_at_the_face_and_narrows_downwards(self):
        """A countersink is a screw-head seat, not an undercut.

        `Cone(bottom_radius=...)` puts the wide end at the *bottom*, so the obvious spelling
        flared below the surface. The flipped cone has the same volume — which is why this
        asserts on faces and area instead: correct is 8 faces / 4890.1589 mm2, flipped was
        9 faces / 5022.1058 mm2, the ninth face being the annular ledge where the inverted
        cone meets the drill.
        """
        result, _ = self._build(CSK_PLATE, name="csk.step")
        self.assertEqual(result.measurements["faces"], 8)
        self.assertAlmostEqual(result.measurements["area_mm2"], 4890.1589, delta=0.5)

    def test_feature_with_no_face_to_enter_is_refused_with_the_op_index(self):
        """A cutter in free space must be loud; silence is what let the bug live."""
        with self.assertRaises(ProgramError) as cm:
            self._build(L_HOLE_IN_AIR, name="air.step")
        message = str(cm.exception)
        self.assertIn("ops[2]", message)
        self.assertIn("(45, 70)", message)

    # ----------------------------------------------------------------------------------
    # The opposite failure: a recess deeper than the material it enters is not a recess.
    # The shipped enclosure declared one (shell 2 mm, counterbore 3 mm) and got away with
    # it only while the pocket was being cut in mid-air.
    # ----------------------------------------------------------------------------------

    def test_shipped_enclosure_counterbores_are_real_recesses(self):
        """4*pi/4*(6^2-3.2^2)*1 = 80.9274 mm3 of recess, and two new faces per hole."""
        import math
        program = _example("enclosure.json")
        self.assertEqual(program["ops"][3]["counterbore"]["depth"], 1,
                         "a counterbore deeper than the 2 mm shell floor is a through hole")
        result, _ = self._build(program, name="enclosure.step")
        bare, _ = self._build(_enclosure_plain(), name="enclosure-plain.step")
        removed = bare.measurements["volume_mm3"] - result.measurements["volume_mm3"]
        self.assertAlmostEqual(removed, 4 * math.pi / 4 * (6 ** 2 - 3.2 ** 2) * 1, delta=0.5)
        self.assertEqual(result.measurements["faces"], bare.measurements["faces"] + 8)

    def test_the_enclosures_old_counterbore_depth_is_refused(self):
        """The defect, pinned against the real file so it cannot come back.

        `depth: 3` into the 2 mm floor left by `shell thickness 2` would turn four M3
        mounting holes into 6 mm clearance holes. It only ever built because the pocket was
        anchored 23 mm above the floor and removed nothing at all.
        """
        with self.assertRaises(ProgramError) as cm:
            self._build(_enclosure_breach(), name="breach.step")
        message = str(cm.exception)
        self.assertIn("ops[3]", message)
        self.assertIn("2.0000 mm of material", message)
        self.assertIn("-1.0000 mm of wall", message)

    def test_countersink_that_breaches_the_far_face_is_refused(self):
        """Same gate, other recess: a 5 mm countersink cannot sit in a 4 mm plate."""
        with self.assertRaises(ProgramError) as cm:
            self._build(CSK_BREACH, name="csk-breach.step")
        message = str(cm.exception)
        self.assertIn("ops[1]", message)
        self.assertIn("-1.0000 mm of wall", message)

    def test_a_recess_that_stops_inside_the_material_is_fine(self):
        """The gate is about breaching, not about depth: 4 mm into 8 mm leaves 4 mm."""
        result, _ = self._build(FLAT_PLATE, name="recess-ok.step")
        self.assertEqual(result.measurements["solids"], 1)

    def test_breach_is_measured_against_the_run_entered_not_the_whole_part(self):
        """A 4 mm counterbore is legal in the 8 mm plate and illegal in a 2 mm lip.

        Both live on the same solid, so a gate reading the part's overall Z extent — 45 mm —
        would wave both through. Only the contiguous run under the axis is the right ruler.
        """
        with self.assertRaises(ProgramError) as cm:
            self._build(L_FLANGE_BREACH, name="lip.step")
        self.assertIn("2.0000 mm of material", str(cm.exception))

    def test_cutter_that_meets_no_material_is_refused_with_its_measurement(self):
        """The numeric half of the same gate: under 0.5 mm3 removed is an error, with a number."""
        with self.assertRaises(ProgramError) as cm:
            self._build(CUT_IN_AIR, name="nocut.step")
        message = str(cm.exception)
        self.assertIn("ops[1]", message)
        self.assertIn("0.0000 mm3", message)
        self.assertIn("0.5 mm3", message)

    def test_tapped_hole_uses_the_standard_pilot_diameter(self):
        """M3 pilot = 3 - 1.0825*0.5 = 2.459 mm. A wrong pilot strips the hole."""
        import math
        result, _ = self._build(TAPPED_PLATE)
        pilot = 3.0 - 1.0825 * 0.5
        expected = 40 * 30 * 10 - 2 * math.pi * (pilot / 2) ** 2 * 8
        self.assertAlmostEqual(result.measurements["volume_mm3"], expected, delta=1.0)

    def test_bad_fillet_radius_fails_with_the_op_index(self):
        """A radius larger than the material cannot be built; the error must say which op."""
        with self.assertRaises(brep_kernel.BrepError) as cm:
            self._build(BAD_FILLET)
        self.assertIn("ops[1]", str(cm.exception))

    def test_stdout_stays_parseable_json(self):
        """The fd-1 hazard: OCCT writes to file descriptor 1 and would corrupt the contract.

        `SINGLE_BOX` is kept out of `_batched_programs()` so this goes through a real
        single-program invocation — batching the test that pins the single-program reply
        would move the thing under test out of the way.
        """
        result, out = self._build(SINGLE_BOX, name="single.step")
        self.assertTrue(result.ok)
        self.assertEqual(result.kernel, "build123d")
        self.assertIsNotNone(result.kernel_version)
        self.assertTrue(out.exists())
        self.assertEqual(result.measurements["volume_mm3"], 960.0)

    def test_measurements_are_json_serialisable(self):
        result, _ = self._build(ROUNDED_BOX)
        json.dumps(result.to_dict())

    def test_shipped_examples_actually_build(self):
        """The expensive half: a program that validates can still be geometrically impossible."""
        examples = _example_programs()
        self.assertTrue(examples)
        for path in examples:
            with self.subTest(example=path.name):
                result, out = self._build(json.loads(path.read_text()),
                                          name=path.stem + ".step")
                self.assertTrue(result.ok)
                self.assertEqual(result.measurements["solids"], 1,
                                 "{} should be one solid body".format(path.name))
                self.assertGreater(result.measurements["volume_mm3"], 0)
                from phimech.interop import inspect as step_inspect
                facts = step_inspect(out)
                # The point of the lane: shipped examples must not come out faceted.
                self.assertFalse(facts.looks_tessellated, "{} exported faceted".format(path.name))
                self.assertEqual(facts.length_unit, "mm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
