# PhiMech CSG

The CSG package is the geometry currency shared by PhiMech's synth layer and CAD backends.
It is deliberately small, immutable, serialisable, and independent of OpenSCAD or FreeCAD.

## AST philosophy

`phimech.csg.ast` defines plain frozen dataclasses, not a modelling-kernel wrapper.
Every geometry-producing layer emits a `Node`; every renderer consumes a `Node`.
Nothing across that boundary is backend text, a FreeCAD `Part.Shape`, or an IR dictionary.

The primitive set follows the smallest useful OpenSCAD vocabulary that FreeCAD can also
reproduce: boxes, cylinders, spheres, explicit polyhedra, and file imports.
Two-dimensional square, circle, and polygon nodes become solids only through an extrusion.
Booleans and transforms wrap children rather than mutating them. The one node outside that
both-backends rule is `ScadModule`, a call into a vendored OpenSCAD library (below).

Units are millimetres and degrees.  A node is a value: `to_dict()` and `from_dict()` make
it suitable for logs, cache keys, MCP messages, and deterministic tests.
The AST intentionally describes *what* shape is wanted; a backend decides *how* to render it.

Use `validate()` before handing an AST to a backend.  It checks dimensions, child shape
contracts, import paths, indices, finite numbers, transform factors, and whether each implicit
or explicit polygon loop encloses nonzero area. Diagnostics use deterministic tree paths so an
invalid nested loop can be traced to its source. Backend render and measure entry points run
this dependency-free validation before consulting an optional CAD executable; a valid AST still
reports a missing OpenSCAD or FreeCAD installation explicitly. `stats()` gives a small tree
summary without evaluating geometry.

## Builder catalogue

`box(w, d, h, center=False, at=...)` — axis-aligned Cube with optional translation.
`cyl(d, h, at=..., axis=..., segments=...)` — a cylinder along X, Y, or Z.
`tube(d_outer, d_inner, h, **kw)` — a hollow cylinder made with a through cut.
`rounded_box(w, d, h, r, edge_r=...)` — a hull of cylinders, or all-edge hull features.
`rounded_rect(w, d, r)` — a 2D rounded rectangle for use in an extrusion.
`slot(length, width, h)` — a rounded-ended slot extruded along +Z.
`countersunk_hole(d, h, head_d, head_depth)` — shaft plus a top conical countersink.
`screw_boss(od, id, h, fillet_r=...)` — hollow boss with an optional base transition.
`standoff(h, od, hole_d, base_d=...)` — hollow mounting standoff and optional flange.
`heat_set_boss(insert, h)` — boss sized from the standard M2–M4 insert table.
`vent_slots(face_w, face_h, slot_w, pitch, depth, margin=...)` — repeated XY slot cutters.
`cut_through(target, cutter)` — Difference with a tiny in-place over-cut.
`union_all(nodes)` — non-empty Union constructor.
`diff_all(base, cutters)` — subtract all cutters, or return base if there are none.
`place(node, at, rot=...)` — rotation followed by translation.
`text_emboss(text, size, depth, ...)` — backend-rendered text intent, without a Text node.

Polygon helpers are also available: `polygon_area`, `polygon_bbox`, `is_ccw`,
`ensure_ccw`, and `offset_polygon`.  The offsetter handles simple CCW polygons with miter
joins and uses only the standard library.

## Library calls: `ScadModule`

`ScadModule(module, args, include, children=())`, kind `scad_module`, is a call to a module from
a vendored OpenSCAD library — today [BOSL2](../vendor/BOSL2/VENDORED.md) (BSD-2-Clause):

```json
{"kind": "scad_module", "module": "threaded_rod",
 "args": {"d": 3, "l": 12, "pitch": 0.5, "$fn": 48},
 "include": "BOSL2/threading.scad", "children": []}
```

It exists because hand-rolling an ISO thread, a swept handle or a continuous-curvature rounding
from cubes and cylinders is exactly where agents fail, and BOSL2 already does each correctly.
The AST carries the *call* rather than mirroring the library primitive by primitive.

* **Named arguments only**, with JSON scalar or array values: `[[0, 0], [10, 0]]` is a path,
  `"hex"` a string, `null` becomes `undef`, `"$fn"` is a legal name. Profiles and paths are
  therefore point lists — a function call such as `circle(d=5)` cannot appear in an argument, by
  design: the emitter formats literals and never interpolates source text.
* **`include`** is relative to `phimech/vendor/` and must resolve to an existing file inside it.
  `validate()` rejects `..`, absolute paths, stray characters and unknown files; `BOSL2/std.scad`
  is always emitted first because the library requires it.
* **`children`** become the module's child block, which BOSL2 uses for attachments.
* **OpenSCAD-only.** The FreeCAD backend lists `scad_module` under `UNSUPPORTED` and refuses a tree
  containing one before launching anything, so there is no STEP from this lane; the pipeline
  routes such parts to OpenSCAD (STL/3MF/OBJ/PNG). See
  [`phimech/backends/README.md`](../backends/README.md#bosl2-openscad-library-calls).
* **Dimensionality is not inferred.** `is_2d()` treats a `ScadModule` as 3D, so a BOSL2 2D shape
  cannot be a `LinearExtrude` child; use BOSL2's `linear_sweep(region=<points>, height=…)` or the
  core `RoundedSquare` instead.

`validate()` checks all of the above; the two shipped examples and their measured numbers are in
[`examples/csg/`](../../examples/csg/README.md), and `bosl2_modules()` (MCP) lists the curated
modules with their key arguments.

## Text convention

The AST has no Text primitive by design.  `text_emboss()` returns an empty `Group` whose
name is exactly `text:{json}`.  The JSON object has these keys: `text`, `size`, `depth`,
`font`, `halign`, and `valign`; `valign` is currently `center`.
Backends render Group names beginning with `text:` as text rather than as ordinary geometry.
Keeping the payload in the name lets the AST remain unchanged while text stays serialisable.

## Over-cutting

`EPS = 0.01` mm is the shared over-cut constant.  Coincident faces are numerically fragile:
a nominally through-going cutter can leave a thin skin after tessellation or B-rep boolean
classification.  `cut_through()` Minkowski-extends the cutter by EPS in place before making
the Difference.  In-place extension matters because a cutter may already be translated;
a naive scale wrapper would also scale that translation and move the feature.

The extension is intentionally tiny relative to normal mechanical dimensions, but explicit
and deterministic.  It is not a substitute for the printability and manifold gates.
