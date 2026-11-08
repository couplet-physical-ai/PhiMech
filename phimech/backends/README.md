# PhiMech CAD backends

The [backend contract](base.py) selects subprocess backends using format support. Both consume
the CSG AST; neither reads design IR. Read the actual render result and build manifest rather
than assuming every requested format was produced.

| Backend | Current implementation | Limits |
| --- | --- | --- |
| OpenSCAD | [openscad.py](openscad.py), [scad_emit.py](scad_emit.py): AST emission and CLI rendering; the only backend for `scad_module` (BOSL2) nodes | Cannot import STEP; PNG execution inherits the process environment and needs host qualification |
| FreeCAD | [freecad.py](freecad.py), [freecad_worker.py](freecad_worker.py): `freecadcmd` worker, measurements and solid/mesh exports | No headless PNG/SVG/DXF export; unsupported AST operations (`minkowski`, `scad_module`, twisted `linear_extrude`) are reported and refused before launch |

FreeCAD sets `QT_QPA_PLATFORM=offscreen` and closes stdin. Calls have explicit timeouts and
return subprocess diagnostics. Both backend entry points first run the dependency-free CSG
validator and report path-qualified invalid-AST errors without launching a kernel; optional
OpenSCAD or FreeCAD absence remains an explicit error for valid input. Kernel installation
paths may refer to application bundles; that does not mean the desktop application must be
opened. Never replace a failed headless call with desktop automation.

The pipeline may convert OpenSCAD STL to FreeCAD STEP when exact FreeCAD construction is
unsupported. That conversion preserves a tessellated shape, not the original analytic model.
STEP imports cannot simply be routed to OpenSCAD for a PNG; unsupported formats are recorded
as skipped. See [pipeline.py](../synth/pipeline.py) for selection and fallback behavior.

See [the public documentation](../../docs/README.md) for the supported workflow.

## BOSL2: OpenSCAD library calls

**What it is.** [BOSL2](https://github.com/BelfrySCAD/BOSL2) (the Belfry OpenSCAD Library v2) is
the de-facto standard library for OpenSCAD: ISO/UTS threads, screws and nuts from spec strings,
path and offset sweeps, lofts, continuous-curvature rounding, attachable primitives. PhiMech
vendors its library files under [`phimech/vendor/BOSL2/`](../vendor/BOSL2/) so an agent can call
them through the `scad_module` CSG node instead of hand-rolling a helix out of cylinders. This is
the generic OpenSCAD backend.

**Licence.** BSD-2-Clause, taken from upstream only (never from another project's vendored copy,
which could arrive with GPL paperwork). The attribution and full licence text are in
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md); the pinned commit, what was omitted
(15 MB of tests, examples, images and tutorials) and the refresh procedure are in
[`phimech/vendor/BOSL2/VENDORED.md`](../vendor/BOSL2/VENDORED.md). The copy is 4.4 MB.

**How a node becomes a render.**

1. The AST carries the *call*: `ScadModule(module="threaded_rod", args={"d": 3, "l": 12,
   "pitch": 0.5, "$fn": 48}, include="BOSL2/threading.scad")`. Named arguments only, JSON values
   only. `phimech.csg.validate` requires the module and argument names to be identifiers and the
   `include` to resolve to an existing file **inside** `phimech/vendor` (`phimech/vendor/bosl2.py`
   is the resolver; `..`, absolute paths and unknown files are rejected).
2. [scad_emit.py](scad_emit.py) writes `include <BOSL2/std.scad>` (always first: BOSL2 requires it)
   and `include <BOSL2/threading.scad>` directly under the header, deduplicated, and the call as
   `threaded_rod(d=3, l=12, pitch=0.5, $fn=48);` — booleans as `true`/`false`, `None` as `undef`,
   strings JSON-quoted, lists nested. Trees without a library call emit byte-identical source to
   before; the golden renders depend on that.
3. [openscad.py](openscad.py) runs the binary with `OPENSCADPATH=<repo>/phimech/vendor` (ours
   first, the caller's own entries kept behind it), which is how `include <BOSL2/…>` resolves. It
   is set on every render, so a `.scad` export re-run by hand resolves the same way.
   `OpenSCADBackend.doctor()` reports the path and the library pin.

**Routing: OpenSCAD-only, by design.** FreeCAD has no OpenSCAD interpreter, so `scad_module` is in
`FreeCADBackend.UNSUPPORTED` next to `minkowski` (visible in `bin/phimech doctor --json`), and
`can_render()` returns a reason naming the module and include. `_run_job()` checks that reason
**before** looking for `freecadcmd`, so a BOSL2 design on a machine without FreeCAD hears "render
it with the openscad backend", never "install freecad". The synth pipeline's router consults the
same two things and sends such parts to OpenSCAD; a direct `render … -o part.step` fails with that
message. The consequence to state plainly: **no STEP or FCStd from this lane** — STL/3MF/OBJ/PNG
only, and the pipeline's STL→STEP fallback would give a tessellated STEP, not a B-rep.

**The gate** ([`tests/test_bosl2.py`](../../tests/test_bosl2.py), run by `bin/phimech check`):
the M3 rod example must measure within ±0.1 mm of Ø3.0 × 12.0 and be watertight (measured
3.0000 × 3.0000 × 12.0000 mm, 68.302 mm³, 0 open edges, 1 shell); the swept handle must be
watertight with volume > 0 (1620.724 mm³, 1 shell). Both live in
[`examples/csg/`](../../examples/csg/README.md). The test also refuses a path-escaping include,
proves FreeCAD refuses the node with the binary hidden and `subprocess.run` patched to fail, and
checks every curated MCP module against a `module <name>(` line in the vendored source.

**Refreshing the vendor copy** is a five-line procedure in `VENDORED.md`: clone upstream with the
proxy stripped, replace `*.scad` + `LICENSE`, update the pin, run `bin/phimech check`. If the rod
moves outside tolerance after a refresh, that is upstream changing under us — read its changelog
before loosening the gate.

**Discovery for agents.** `phimech/mcp/tools_bosl2.py` exposes `bosl2_modules(query)`: the
include files, a curated table of ~15 modules with key arguments, the pin, a copy-ready example
node and the writing rules (profiles and paths are point lists, never `circle()`/`arc()` calls;
`$fn` is an ordinary named argument).
