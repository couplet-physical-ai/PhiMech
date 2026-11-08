# The B-rep lane

Real fillets, chamfers, counterbores and non-faceted STEP, built in an OpenCascade kernel that
runs as a subprocess.

## Why this exists

`ops.rounded_box()` builds rounded corners as `Hull([cylinders])`, and a convex hull has no
curved faces. Measured on the same 60 × 40 × 25 rounded box:

| | faces | `PLANE` | `CYLINDRICAL_SURFACE` | bytes |
| --- | --- | --- | --- | --- |
| hull lane (OpenSCAD/FreeCAD) | 130 | 130 | **0** | 192,647 |
| B-rep lane | **10** | 6 | **4** | 31,913 |

Ten is the right answer: a rounded box is six planes and four cylinders. Rounding an edge is a
*fillet*, and a fillet needs a kernel that knows what an edge is.

**Note that `rounded_box` itself no longer needs this lane.** FreeCAD is OpenCascade underneath,
so it now fillets the `rounded_cube` AST node natively, which took the shipped enclosure from
9,174 faces to 75 with no new dependency — see
[the public documentation](../../docs/README.md). This lane is for what the CSG vocabulary
cannot express at all: counterbores, countersinks, chamfers, shelling and tapped pilot holes.

It does not replace the existing pipeline. OpenSCAD stays the fast path for CSG and preview;
this is the path for a part someone else has to open and edit.

## Install (optional)

```bash
bin/phimech brep setup          # creates build/brep-venv, ~518 MB, a few minutes
bin/phimech doctor              # brep ✓ build123d 0.11.1 (build/brep-venv)
```

**Absence is not an error.** `doctor` prints a hint and `overall` stays green, because a
half-gigabyte dependency that turns the doctor red just trains people to ignore a red doctor.
Everything that built before still builds without it.

Interpreter resolution, first hit wins:

1. `PHIMECH_BREP_PYTHON` — explicit override, and how CI would pin it.
2. `build/brep-venv/bin/python` — what `brep setup` creates.
3. `sys.executable`, but only if it already imports `build123d`.

`build123d` requires python `>=3.10,<3.13`, so `setup` defaults to `python3.12`. Verified on
0.10.0 and 0.11.1, which agree to the last decimal on volume.

## Use

```bash
bin/phimech brep ops                                            # the vocabulary
bin/phimech brep build examples/brep/enclosure.json -o out/e.step -o out/e.stl
bin/phimech interop step out/e.step                             # confirm it is not faceted
```

A feature program is a JSON object with an ordered `ops` list, applied to one evolving solid:

```json
{"version": 1, "units": "mm", "ops": [
  {"op": "box", "size": [80, 60, 25]},
  {"op": "fillet", "select": "vertical", "radius": 4},
  {"op": "shell", "thickness": 2, "open_face": "top"},
  {"op": "hole", "at": [[8, 8], [72, 8]], "diameter": 3.2,
   "counterbore": {"diameter": 6, "depth": 1}}
]}
```

(`depth: 1`, not 3: `shell thickness 2` leaves a 2 mm floor, and a 3 mm counterbore in it is
not a recess but a 6 mm through hole. The gate below refuses it.)

| op | what it does |
| --- | --- |
| `box`, `cylinder` | start the solid, or add to it |
| `cut_box`, `cut_cylinder` | subtract a prism or a cylinder |
| `fillet` | round selected edges by `radius` |
| `chamfer` | bevel selected edges by `length` |
| `shell` | hollow the solid to `thickness`, leaving `open_face` open |
| `hole` | drill at a list of `[x, y]` points; optional `counterbore` / `countersink` |
| `tapped_hole` | pilot hole at the standard tapping diameter for `size` |

Edge selectors: `vertical` (Z-parallel — on a box these are the four **corner** edges, *not* the
top and bottom rims), `top`, `bottom`, `top_and_bottom`, `all`.
Face selectors: `top`, `bottom`, `front`, `back`, `left`, `right`.

### Where a blind feature lands, and what it must remove

`hole` omits `depth` to cut through. Everything else is blind — a `depth`-limited hole, a
counterbore pocket, a countersink cone — and its depth is measured down from **the highest
material face on that feature's own axis**, found by probing the solid at `(x, y)` before
anything is cut there.

Not from the top of the part. That shortcut is right only when the feature happens to sit on
the topmost face of the whole solid, and on the shipped L-bracket — a 60 × 40 × 8 plate under a
45 mm upstand — it put all three counterbore pockets at z 41…45, 33 mm above the plate, in free
air. They removed nothing, and the bracket measured **identically** with and without them.
Nothing noticed, because a boolean whose result is empty is not an error.

So it is one now, with a number on it (rule 7): **every individual cutter must remove at least
0.5 mm³**, or the program is refused with exit 2 and the op index. Per *cutter*, not per op —
the bug above hid inside a `hole` op that removed 510 mm³ of through-holes while its three
counterbores removed exactly zero.

The opposite failure has a number on it too. A counterbore or countersink must leave at least
`MIN_REMAINING_WALL_MM` between its floor and **the far side of the run it enters** — the
contiguous slab of material under the axis, not the part's overall Z extent. A recess deeper
than that is a wider through hole, and the refusal reports the measured remaining wall:

```
ops[3] (hole): counterbore at (8, 8) is 3 mm deep into the 2.0000 mm of material it enters
at z=2, leaving -1.0000 mm of wall at z=0: it reaches the far face, so it is a wider through
hole and not a recess
```

That is not hypothetical: `examples/brep/enclosure.json` shipped with exactly that spelling and
built cleanly, because the pocket was being cut 23 mm above the floor it belonged in. The
moment the anchoring was fixed, four M3 mounting holes would have quietly become 6 mm clearance
holes. **The breach gate is kernel-side, not in `program.py`, and deliberately so:** how much
material sits under a feature is a property of the accumulated solid — shell thickness,
fillets, which face the axis enters, whatever earlier ops cut away — not of the declared
numbers. Deriving it from `shell.thickness` would be right for a flat uniform floor and
silently wrong for a boss, a rib, a corner or anything already cut. `program.py` keeps the
checks the numbers alone can settle (a counterbore no wider than its hole, or no shallower than
the blind hole it sits on top of); the kernel measures the rest.

A blind `hole` whose `depth` exceeds the material is *not* gated — that is a through hole
spelled out the long way, which is legible rather than wrong.

A countersink is a screw-head seat: wide at the face, narrowing down to the hole.
(`build123d`'s `Cone(bottom_radius=…)` puts the wide end at the bottom, so the obvious spelling
built an undercut that flared *below* the surface. Same volume either way, which is why it went
unseen; it shows up as an extra annular face — 9 instead of 8 on a countersunk plate.)

### `tapped_hole` models drill depth, not thread depth

`tapped_hole` cuts the pilot flat-bottomed to exactly `depth`, so as modelled the drill and the
usable thread are the same length. No real tapped hole is: a tap cannot form a full thread in
the last few pitches of a flat-bottomed hole, so a shop drills roughly one to two diameters
deeper than the thread it needs, plus the drill point. Splitting the two needs a second number
in the vocabulary (`thread_depth` beside `depth`, or a `pilot_extra`) — a reviewed change to the
op table, not a default the worker should invent. **Until it is made, read `depth` as drill
depth and specify a shorter usable thread than the number says.**

### Many programs, one kernel start

`import build123d` costs ~3.2 s — longer than most of our programs take to build — so anything
that builds N variants should not pay it N times. `kernel.build_many()` sends a list of
`{program, outputs}` in one invocation and returns one `BrepResult` per program, in order:

```python
from phimech.brep import kernel
results = kernel.build_many(
    [{"program": p, "outputs": {"step": "/abs/out/{}.step".format(i)}}
     for i, p in enumerate(variants)], timeout=900.0)
for r in results:
    kernel.raise_for(r)          # same exception `build()` would have raised, or the result
```

It does **not** raise on a failed program — one program's refusal is not another's verdict —
so inspect `result.ok` / `result.gate`, or pass the one you care about through `raise_for()`.

This is still a subprocess with an explicit timeout that exits when it is done. Rule 5 bans a
kernel in *our* interpreter and bans a daemon; it does not ask each program to buy its own
process. Measured on the test suite: 26 kernel starts became 5, and 45 s became 15 s.

## Three design decisions worth knowing

**The program is data, not code.** The obvious way to drive a B-rep kernel is to let a model
write kernel code and `exec` it, which is what [agentcad](https://github.com/jdilla1277/agentcad)
does. We do not: rule 4 says the agent writes intent and *we* compute numbers; an op list can be
fully validated **before** a subprocess starts, so a typo costs milliseconds instead of a kernel
timeout; and `exec` of model-written code has no sandbox. The cost is expressiveness — anything
not in the table above cannot be built, and the vocabulary grows one reviewed op at a time.

**The kernel is a subprocess, never an import.** `build123d` pulls in `cadquery-ocp`, ~518 MB of
compiled OpenCascade. Importing that into the MCP server's interpreter is how you get the
ABI conflicts that the core boundary exists to prevent. `worker.py` is the only file that imports
a kernel; a kernel crash kills a child process, not a build.

**`tapped_hole` does not model a helix.** It cuts the pilot to `d - 1.0825 × pitch`, the standard
tapping-drill approximation, because build123d 0.10/0.11 have no `IsoThread` and sweeping a
profile along a `Helix` is slow and fragile — and a printed part is tapped or takes a heat-set
insert anyway. The op is named for what it does rather than for what it might be mistaken for.

## Failure modes

| Symptom | Cause |
| --- | --- |
| `ops[N] (fillet) failed` | radius exceeds the material at that edge — the op index is always named |
| `ops[N] (hole): counterbore … removed 0.0000 mm3` | the cutter meets no material; check the face it is meant to enter exists |
| `ops[N] (hole): nothing to drill at (x, y)` | the hole axis misses the solid entirely |
| `ops[N] (hole): counterbore … leaving -1.0000 mm of wall` | the recess is deeper than the material it enters; it is a through hole |
| `selector 'vertical' matched no edges` | the solid has no Z-parallel edges (e.g. a bare cylinder) |
| `worker stdout was not JSON` | something wrote to fd 1 around a kernel call; see the note below |
| `no interpreter with build123d found` | run `bin/phimech brep setup` |
| `output path must be absolute` | rejected rather than resolved: the worker's cwd is not yours |

**The fd-1 hazard.** OCCT's `Message_DefaultMessenger` writes to file descriptor 1 *directly*,
bypassing `sys.stdout`, so a single kernel warning would land inside the JSON contract and break
the parse. `worker.py` redirects fd 1 to fd 2 around every kernel call. The idea is from
agentcad's `native_io.py` (Copyright James Dillard, Apache-2.0); this is an independent
implementation.

Shipped programs live in `examples/brep/`, deliberately *not* in `examples/` — that directory is
validated as design IR by `bin/phimech check`, and a feature program is a different schema.

## Tests

`tests/test_brep.py`, 47 tests in two halves. The validation half is stdlib-only and always
runs, including in CI where no kernel is installed — that is the half that stops a malformed
request reaching a subprocess. The kernel half self-skips, and when it runs it asserts on
*numbers* (10 faces, 4 cylindrical surfaces, the M3 pilot volume to within 1 mm³), because
"it built something" is not evidence that the something is right.

Blind features are tested on an **L-shaped** part as well as a flat plate, deliberately. Every
kernel test used to use a plate, and on a plate the top face *is* the bounding-box top — which
is precisely why a counterbore cut 33 mm above the material passed for as long as it did.

The kernel half runs in **one batched invocation**: `_batched_programs()` declares every
program the class builds and `setUpClass` builds them all at once, keyed by program content so
duplicates cost nothing. Membership is a speed property only — `_build()` falls back to its own
invocation for anything unlisted, so a new test can be slow but never wrong. Two programs are
kept *out* of the batch on purpose, because what they pin is the code batching would bypass:
the single-program request shape, and the host's mapping of a worker gate refusal onto
`ProgramError`. Whole file: **15 s wall, 138 assertions**, down from 45 s for the same 138.
