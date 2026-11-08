# `fixtures/viewer` — a committed build so the viewer works with no CAD installed

A real, complete pipeline output, checked in. It exists for one reason: **without it, a clone with
no OpenSCAD/FreeCAD renders nothing and every geometry test in `tests/test_viewer.py` skips** — a
green run that asserted nothing about geometry. That is a bad first hour for anyone building on the
viewer, and it is invisible in CI.

```bash
bin/phimech view fixtures/viewer --no-open   # renders immediately, no CAD toolchain required
/usr/bin/python3 tests/test_viewer.py        # 28 pass, 0 skip
```

A generated build under `out/` still wins when one exists, so developer machines keep validating
real pipeline output. The fixture is the floor, not the preference.

## What is in here

| File | Why |
| --- | --- |
| `report.json` | 17 gates, measurements, stages — the manifest is built from this |
| `design.json` | material and manufacturing context the manifest reads |
| `REPORT.md` | so `files.report_md` is not null |
| `viewer-fixture-base.stl`, `-lid.stl` | **binary** STL, 5192 + 7164 triangles, 2 parts |
| `viewer-fixture-assembly.png`, `-exploded.png` | so `previews[]` is not all null |

~760 KB total. `tests/test_viewer.py` enforces a 1 MB ceiling on the STLs, because a "nicer"
fixture is the easy way to add megabytes to every clone.

## Deliberate properties — do not "fix" these

**It fails a gate on purpose.** `text_legible` fails at `minimum_text_size_mm=2` against a
threshold of 4, because the label inherits the design title and is shrunk to fit a small lid. That
failure is the point: **no bundled example produces a failing gate**, so without it a UI could
never be developed or reviewed against a red gate — the state it most needs to get right. Between
this fixture's `text_legible` (fail), the other 15 (pass) and `fits_bed` (skipped, no
`manufacturing.printer_bed`), all three tri-states are present in one payload.

**`ok` is `true` even though a gate failed.** `text_legible` is *warn* severity, and the verdict is
`stage_ok and gates_ok and not run.errors` where only error-severity gates count. A UI that renders
`ok` as "all gates passed" is wrong, and `test_fixture_failing_gate_does_not_flip_the_verdict`
catches it.

**Low-poly by design.** `corner_radius: 0` and a plain box stand-in keep the mesh at 12356
triangles. The remaining count is spheres and cylinders in the lid lip.

**Binary STL, and no absolute paths.** ASCII is ~5x larger for identical geometry. Every build path
in `report.json` was rewritten to `fixtures/viewer/...`; a committed fixture must not leak the
authoring machine's layout. Both are asserted by tests.

## Regenerating

```bash
bin/phimech synth build fixtures/viewer/design.json -o /tmp/fixture-build
```

Then, from the repo root: convert each `*.stl` to binary, copy `report.json`, `design.json`,
`REPORT.md` and the two PNGs, and rewrite every absolute path in `report.json` to
`fixtures/viewer/<name>`. Re-run `/usr/bin/python3 tests/test_viewer.py`; the five `test_fixture_*`
tests check the size ceiling, binary encoding, path hygiene, the three gate states, and the
warn-failure verdict.

Keep the failing gate. If a pipeline change makes `text_legible` pass, introduce a different
warn-severity failure rather than shipping an all-green fixture.
