# BOSL2 — vendored copy

| | |
| --- | --- |
| Upstream | <https://github.com/BelfrySCAD/BOSL2> (the **only** acceptable source — see below) |
| Commit | `9f1fa5c739018f58e9c7bd92585427a6c6996e46` |
| Commit date | 2026-09-22T02:22:02-07:00 |
| Library version | `BOSL_VERSION = [2,0,757]` (`version.scad`) |
| Licence | BSD-2-Clause — [`LICENSE`](LICENSE), attributed in [`/THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) |
| Vendored | the 56 `*.scad` library files at the upstream repository root, plus `LICENSE` (4.4 MB) |
| Omitted | `tests/`, `examples/`, `images/`, `tutorials/`, `resources/`, `scripts/`, `.github/`, and the upstream markdown (`README.md`, `CONTRIBUTING.md`, `TESTING.md`, `WRITING_DOCS.md`, `CLAUDE.md`) — none of it is needed to *use* the library, and it is 15 MB of pictures and docs |

## Why it is here

PhiMech's CSG AST gained a `scad_module` node (`phimech/csg/ast.py`) so an agent can reach for
BOSL2's threads, screws, sweeps and rounding instead of hand-rolling them. The OpenSCAD backend
resolves `include <BOSL2/…>` by pointing `OPENSCADPATH` at the **parent** of this directory
(`phimech/vendor`), so the directory name `BOSL2` is part of the contract: renaming it breaks
every include line.

## Take it from upstream only

Take BOSL2 from `BelfrySCAD/BOSL2` directly, never from another project's vendored copy (CADAM
ships one inside a GPL-3.0 repository; a copy pulled out of there arrives with GPL paperwork,
while upstream is BSD-2-Clause on its own terms). This is recorded in
`docs/README.md`.

## How to refresh

```bash
# 1. clone upstream using your normal network configuration
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
    git clone --depth 1 https://github.com/BelfrySCAD/BOSL2 /tmp/bosl2-upstream

# 2. replace the library files, and nothing else
rm phimech/vendor/BOSL2/*.scad
cp /tmp/bosl2-upstream/*.scad /tmp/bosl2-upstream/LICENSE phimech/vendor/BOSL2/

# 3. pin the new commit here (sha, date, BOSL_VERSION) and re-run the gate
(cd /tmp/bosl2-upstream && git log -1 --format='%H %cI' && grep BOSL_VERSION version.scad)
bin/phimech check                       # tests/test_bosl2.py re-measures the M3 rod and the sweep
```

If a refresh moves the measured M3 rod outside ±0.1 mm of Ø3.0 × 12.0, or the swept example stops
being watertight, that is upstream behaviour changing under us: read the upstream changelog before
loosening anything.

If the copy ever exceeds 6 MB, drop the largest rarely-needed files (candidates in size order:
`nurbs.scad`, `isosurface.scad`, `gears.scad`) and list them under *Omitted* above.
