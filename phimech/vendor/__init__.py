"""Third-party OpenSCAD libraries redistributed with PhiMech.

Each library lives in its own subdirectory (`BOSL2/`), which is also the name OpenSCAD sees in
`include <BOSL2/std.scad>`: the OpenSCAD backend puts this directory on `OPENSCADPATH`. The
Python here is glue only — path resolution, the pinned revision, a curated module table — and
stays stdlib-only so `phimech.csg.validate` can import it under python 3.9.

Licences and attribution: `THIRD_PARTY_NOTICES.md` at the repository root, and `VENDORED.md`
inside each library directory for the pin and refresh procedure.
"""

from __future__ import annotations

from pathlib import Path

#: The directory OpenSCAD searches: `include <BOSL2/std.scad>` resolves to `VENDOR_DIR/BOSL2/std.scad`.
VENDOR_DIR: Path = Path(__file__).resolve().parent

__all__ = ["VENDOR_DIR"]
