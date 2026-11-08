"""Tests for STEP as a drawing/preview source — the SolidWorks entry point.

Runs two ways, per house rule:
    /usr/bin/python3 tests/test_step_source.py
    pytest tests/test_step_source.py

SolidWorks hands PhiMech a STEP file; every PhiMech surface that measures geometry measures a
*mesh*. The interesting cases are therefore the ones where tessellation succeeds and the mesh is
still the wrong thing: an inch STEP (valid, 25.4x too big, passes every geometric gate), a mesh
STEP masquerading as a solid, and a translation that silently dropped a body. Those refusals are
the point of this module, so they are what is tested here; they run with no CAD installed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phimech.synth import preview  # noqa: E402
from tests.test_interop import MESH_BODY, SOLID_BODY, _step  # noqa: E402



def _write(text: str, name: str = "m.step") -> Path:
    tmp = Path(tempfile.mkdtemp()) / name
    tmp.write_text(text, encoding="utf-8")
    return tmp


class TestStepRefusals(unittest.TestCase):
    """Every refusal here describes a file that opens fine in CAD and is still unusable."""

    def _resolve(self, path: Path) -> None:
        workdir = Path(tempfile.mkdtemp())
        preview._resolve_source(str(path), workdir, timeout=60.0)

    def test_inch_step_is_refused_not_converted(self):
        """Rule 6: mm throughout. A converted inch file would pass every downstream gate."""
        path = _write(_step(SOLID_BODY, unit="( CONVERSION_BASED_UNIT('INCH',#9) )"))
        with self.assertRaises(ValueError) as ctx:
            self._resolve(path)
        self.assertIn("millimetres", str(ctx.exception))

    def test_tessellated_step_is_refused(self):
        """A mesh-in-STEP has no faces to measure; its 'edges' are triangle boundaries."""
        path = _write(_step(MESH_BODY))
        with self.assertRaises(ValueError) as ctx:
            self._resolve(path)
        self.assertIn("not a solid STEP", str(ctx.exception))

    def test_surfaces_only_step_is_refused(self):
        """No closed shell means no solid, however cleanly it opens."""
        path = _write(_step("#10 = ADVANCED_FACE('',(#11),#12,.T.);\n#12 = PLANE('',#14);\n"))
        with self.assertRaises(ValueError) as ctx:
            self._resolve(path)
        self.assertIn("not a solid STEP", str(ctx.exception))

    def test_unparseable_step_names_the_parse_error(self):
        path = _write("this is not a STEP file at all\n")
        with self.assertRaises(ValueError):
            self._resolve(path)

    def test_stp_suffix_is_accepted_as_a_step(self):
        """SolidWorks writes .stp as often as .step; the refusal proves it reached the parser."""
        path = _write(_step(MESH_BODY), name="m.stp")
        with self.assertRaises(ValueError) as ctx:
            self._resolve(path)
        self.assertIn("not a solid STEP", str(ctx.exception))

    def test_unsupported_suffix_mentions_step(self):
        path = _write("x", name="m.sldprt")
        with self.assertRaises(ValueError) as ctx:
            self._resolve(path)
        self.assertIn(".step", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
