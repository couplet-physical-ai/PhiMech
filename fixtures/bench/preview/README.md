# Wrong-shape suite — recorded run, 2026-09-22

Eight deliberately wrong parts with the **right dimensions**: every candidate shares its
reference's bounding box, is manifold and has a positive volume. The question the run answers is
how many of them the numeric gates notice, and how many the preview sheet's silhouette IoU notices.

```bash
bin/phimech preview bench                   # reproduce (exit 0 iff the gate holds)
bin/phimech preview bench --record          # rewrite results.json
```

## Score: numeric gates 0 / 8 caught, silhouette IoU 7 / 8 caught

Threshold 0.92, 160 px tiles, OpenSCAD 2026.09.07. Gate: numeric ≤ 1/6, preview ≥ 5/6 → **OK**.

| Case | What is wrong | Numeric gates | min IoU | Worst view | Preview |
| --- | --- | --- | --- | --- | --- |
| `spacer-box-for-cylinder` | a 30 × 30 × 40 box instead of a cylinder | passed | 0.761 | iso | caught |
| `funnel-straight-for-tapered` | a straight Ø30 cylinder instead of a Ø30 → Ø16 cone | passed | 0.739 | iso | caught |
| `lid-no-cutout` | a plain plate: the display window is missing | passed | 0.799 | top | caught |
| `stand-wrong-angle` | the back rest leans 45° instead of 15° | passed | 0.415 | left | caught |
| `bracket-holes-wrong-leg` | the two holes go through the horizontal leg, not the vertical one | passed | 0.870 | front | caught |
| `lid-bezel-missing` | the LED bezel boss is missing | passed | 0.763 | front | caught |
| `plate-tab-mirrored` | the tab is on the −X edge: a mirror image | passed | 0.690 | top | caught |
| **`tray-open-front`** | open on the front face with a closed top, instead of open on top | passed | **1.000** | — | **blind** |

"Numeric gates" are `builds`, `manifold`, `part_volume_positive` and `outer_size_mm` (±0.5 mm),
recomputed on the candidate STL with the check layer's own mesh functions at the same thresholds
the pipeline uses. All 32 checks passed on all eight wrong parts.

## Same-shape controls

The threshold has to be tolerant as well as strict, or every tolerance discussion would be
flagged as a wrong shape:

| Control | min IoU | Verdict |
| --- | --- | --- |
| the reference re-rendered | 1.000 | same shape (renders are pixel-deterministic on one machine) |
| the reference 2 mm (5 %) taller | 0.954 | same shape — a size change is the numeric gates' job |

Closest wrong shape 0.870, closest same shape 0.954: **0.92** sits between with ≥ 0.03 of clearance
each way. Anti-aliasing differences between OpenSCAD builds move IoU by an order of magnitude less.

## The miss is kept on purpose

`tray-open-front` scores 1.000 in all seven views for a part that is plainly wrong. An opening
that never reaches the outline is invisible to a silhouette; from every axis, an open-front tray
with a closed top and an open-top tray are the same rectangle, and in the iso view the cavity is
model surface, not background. This is the concrete limit of the proxy — and the reason the sheet
itself, looked at, is what closes the loop. The proxy is a floor for "looks right", not a proof.

## Files

- `<case-id>.png` — the reference sheets, 640 × 320 (7 views + bbox/volume tile), rendered from
  the reference geometry in `phimech/bench/wrong_shape.py`.
- `wrong/<case-id>.json` — the deliberately wrong candidates as CSG AST JSON, the recorded
  submission. `tests/test_preview.py` fails if they drift from the case definitions.
- `results.json` — the numbers above, per view. `tests/test_preview.py` asserts the gate from it,
  so the evidence cannot silently rot.
