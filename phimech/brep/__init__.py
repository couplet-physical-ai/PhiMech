"""The B-rep lane — real fillets, chamfers, threads and non-faceted STEP.

The existing OpenSCAD/FreeCAD lane is mesh-first: `ops.rounded_box()` builds rounded corners as
a convex hull, and a hull has no curved faces. Measured, that exports a 60x40x25 rounded box as
130 planar faces with zero cylindrical surfaces, and a real enclosure as 9,174. This lane builds
the same shapes in a B-rep kernel, where rounding an edge is a fillet.

`program.py` is the declarative op list (stdlib, validated before anything runs), `kernel.py`
drives a subprocess, and `worker.py` is the only file that imports the kernel.
"""

from __future__ import annotations

from phimech.brep.program import OPS, ProgramError, describe, validate

__all__ = ["OPS", "ProgramError", "describe", "validate"]
