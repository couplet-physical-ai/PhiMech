# Raw CSG AST examples

These are **CSG ASTs** (`phimech/csg/ast.py` in JSON form), not design IRs. They live here and not
in `examples/` because `bin/phimech check` validates every `examples/*.json` as a design IR and CI
runs `synth build` on each one — a raw AST is a different schema and would fail both, exactly as
`examples/brep/` explains for feature programs. They are still gated: `tests/test_bosl2.py`
renders and measures each of them, and `bin/phimech check` runs that test.

Use them with the AST-level commands:

```bash
bin/phimech emit    examples/csg/bosl2-m3-threaded-rod.json                 # the .scad source
bin/phimech measure examples/csg/bosl2-m3-threaded-rod.json                 # bbox / volume / area
bin/phimech render  examples/csg/bosl2-swept-handle.json -o out/handle.stl  # or .3mf / .png
```

## The examples

| Example | What it exercises | Gate (`tests/test_bosl2.py`) | Measured |
| --- | --- | --- | --- |
| [`bosl2-m3-threaded-rod.json`](bosl2-m3-threaded-rod.json) | `scad_module` calling BOSL2 `threaded_rod(d=3, l=12, pitch=0.5, $fn=48)` from `BOSL2/threading.scad` | bbox X and Y within ±0.1 mm of **3.0** (M3 major diameter), Z within ±0.1 mm of **12.0**; watertight, one shell | 3.0 × 3.0 × 12.0 mm, 68.30 mm³ |
| [`bosl2-swept-handle.json`](bosl2-swept-handle.json) | BOSL2 `path_sweep` of a Ø5 circle (given as 32 points) along a 30 mm U-path (29 points), composed with core `Group`/`Translate` nodes and two `cuboid(rounding=1)` feet | watertight, exactly one shell, volume > 0 | 38.0 × 29.5 × 8.0 mm, 1620.7 mm³ |

Both are **OpenSCAD-only**: a `scad_module` node is a call into vendored OpenSCAD library source,
so the FreeCAD backend refuses the tree by name and no STEP can be produced from it
(`bin/phimech render … -o part.step` tells you so before looking for `freecadcmd`). See
[`phimech/backends/README.md`](../../phimech/backends/README.md#bosl2-openscad-library-calls).

Two things about how the arguments are written, because they are the parts an agent gets wrong:

* **Profiles and paths are point lists, not calls.** `args` may hold JSON scalars and arrays only,
  so `circle(d=5)` and `arc(r=15, …)` cannot appear inside an argument; the JSON carries the
  points themselves. This is deliberate — the emitter formats literals and never interpolates
  source text.
* **`$fn` is an ordinary named argument** (`"$fn": 48`). It is how the rod gets 48 segments per
  turn instead of OpenSCAD's coarse default, and it is why the measured diameter is exactly 3.0
  rather than a flat-to-flat polygon width.
