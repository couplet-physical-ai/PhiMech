# `phimech/vendor` — redistributed OpenSCAD libraries

| Library | Directory | Licence | Pin | Size |
| --- | --- | --- | --- | --- |
| BOSL2 | [`BOSL2/`](BOSL2/) | BSD-2-Clause ([`BOSL2/LICENSE`](BOSL2/LICENSE), [`/THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)) | [`BOSL2/VENDORED.md`](BOSL2/VENDORED.md) | 4.4 MB, 56 `.scad` files |

The directory name a library sits under is the name OpenSCAD sees: the OpenSCAD backend puts
`phimech/vendor` on `OPENSCADPATH`, so `include <BOSL2/std.scad>` resolves to `BOSL2/std.scad`
here. Renaming a directory breaks every include line.

The Python in this package is glue, not geometry, and is stdlib-only (imports under python 3.9):

| Module | What it is for |
| --- | --- |
| [`__init__.py`](__init__.py) | `VENDOR_DIR` — the `OPENSCADPATH` entry |
| [`bosl2.py`](bosl2.py) | `resolve_include()` (refuses anything outside this directory), `required_includes()` (`std.scad` first), `openscadpath()`, `include_files()`, `pinned()`, `defines_module()`, `CURATED_MODULES`, `doctor_section()` |

How the node, the emitter, the backends and the gate fit together is documented in
[`phimech/backends/README.md`](../backends/README.md#bosl2-openscad-library-calls) and
[`phimech/csg/README.md`](../csg/README.md#library-calls-scadmodule).

## Distribution

The OpenSCAD backend discovers BOSL2 through `OPENSCADPATH`. The built Python wheel includes the SCAD files and BSD license; `phimech doctor` checks their availability. See `pyproject.toml` for the included package-data patterns.

## Adding another library

Same shape: `phimech/vendor/<Name>/` with the upstream library files, its `LICENSE`, a
`VENDORED.md` pin, an entry in `THIRD_PARTY_NOTICES.md`, and a `required_includes()` rule if the
library has a prerequisite file. Take it from its upstream repository, never from a copy inside
another project — the licence that travels with a copy is the licence of the repository it came
from.
