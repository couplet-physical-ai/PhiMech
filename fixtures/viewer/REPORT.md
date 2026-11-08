# PhiMech report — Viewer fixture — low-poly two-part enclosure

**PASS — 15 gates passed, 1 failed, 1 skipped**

| field | value |
| --- | --- |
| project | viewer-fixture |
| title | Viewer fixture — low-poly two-part enclosure |
| rev | A |
| company | Example Co |
| generated | 2026-09-14T15:34:16 |
| phimech | 0.1.0 |
| design IR | /tmp/fixgen/v3.json |
| out dir | /private/tmp/fixgen/out3 |
| elapsed | 1.772 s |

## Stages

|  | stage | seconds | note |
| --- | --- | --- | --- |
| ✓ | load | 0.002 | - |
| ✓ | validate | 0.065 | - |
| ✓ | envelopes | 0.012 | 1 envelope(s) |
| ✓ | place | 0.001 | - |
| ✓ | enclosure | 0.003 | parts: base, lid |
| ✓ | gates_static | 0.004 | PASS: 17 gates, 1 failed, 4 skipped |
| ✓ | render | 1.175 | 4 file(s) for 4 part(s) |
| ✓ | measure | 0.318 | 2 part(s) measured |
| ✓ | gates_full | 0.191 | PASS: 17 gates, 1 failed, 1 skipped |
| ✓ | report | 0.000 | report.json + REPORT.md |

## Components

| ref | name | source | size mm | position mm | rotation ° |
| --- | --- | --- | --- | --- | --- |
| BLK1 | Stand-in board | box | 24 × 30 × 6 | 0 × 0 × 0 | 0 × 0 × 0 |

## Enclosure

| field | value |
| --- | --- |
| style | tray_lid |
| parts | base, lid |
| outer size mm | 33 × 39 × 16.5 |
| cavity size mm | 29 × 35 × 12.5 |
| wall / floor mm | 2 / 2 |
| fastening | snap |
| standoffs | 0 |
| cutouts | 0 |
| screws | 0 |
| est. material g | 11.15 |

Enclosure notes:

- 4 snap hooks on the base rim with matching recesses in the lid lip
- label text scaled down to 2.0 mm to fit the lid
- label 'Viewer fixture — low-poly two-part enclosure' debossed on the lid top face

## Outputs

| part | fmt | size KB | backend | seconds | file |
| --- | --- | --- | --- | --- | --- |
| base | stl | 1529.7 | openscad | 0.195 | viewer-fixture-base.stl |
| lid | stl | 2003.9 | openscad | 0.176 | viewer-fixture-lid.stl |
| assembly | png | 63.8 | openscad | 0.326 | viewer-fixture-assembly.png |
| exploded | png | 62.2 | openscad | 0.258 | viewer-fixture-exploded.png |

## Measurements

| part | bbox mm | volume mm³ | area mm² | solids |
| --- | --- | --- | --- | --- |
| base | 33 × 39 × 14.5 | 6375.4 | 6525.6 | 1 |
| lid | 33 × 39 × 3.5 | 2616.2 | 3130.5 | 1 |

## Gates

|  | gate | value | threshold | severity | message |
| --- | --- | --- | --- | --- | --- |
| ✓ | ir_valid | 0 | 0 | error | errors=0 threshold=0 (validated in-gate) |
| ✓ | placement_no_overlap | 0 | 0 | error | overlap_errors=0 threshold=0 |
| ✓ | clearance_positive | 1.5 | 0.0 | error | minimum_cavity_margin=1.5 threshold=0 |
| ✓ | wall_thickness | 2.0 | 1.2 | error | wall=2 min_wall=1.2 nozzle_delta=0 |
| ✓ | floor_thickness | 2.0 | 1.2 | error | floor=2 min_wall=1.2 |
| – | fits_bed | - | 0.0 | error | skipped: no printer_bed |
| ✓ | part_volume_positive | 2616.1505020281347 | 0.0 | error | minimum_volume_mm3=2616.15 threshold>0 |
| ✓ | manifold | 1 | 1 | error | base:solids=1,open_edges=0; lid:solids=1,open_edges=0 threshold=1 |
| ✓ | min_feature | 0.8 | 0.8 | error | no standoffs |
| ✓ | screw_count_sane | 0.0 | 2 | error | not applicable |
| ✓ | cutout_within_face | 0.0 | 0.0 | error | no cutouts |
| ✓ | standoff_within_cavity | 0.0 | 0.0 | error | no standoffs or lid posts |
| ✓ | fastener_clears_components | 0.0 | 0.0 | error | no lid posts |
| ✓ | standoff_lands_on_hole | 0.0 | 0.05 | error | no standoffs |
| ✗ | text_legible | 2.0 | 4.0 | warn | minimum_text_size_mm=2 threshold=4 |
| ✓ | overhang | 0.0355747651933221 | 0.35 | warn | maximum_overhang_fraction=0.0355748 threshold=0.35 angle_deg=45 |
| ✓ | material_estimate | 11.14947493020164 | 0.0 | warn | material_g=11.1495 density_g_cm3=1.24 scale=1.0 |

## Errors and notes

Notes:

- enclosure: 4 snap hooks on the base rim with matching recesses in the lid lip
- enclosure: label text scaled down to 2.0 mm to fit the lid
- enclosure: label 'Viewer fixture — low-poly two-part enclosure' debossed on the lid top face

## How to reproduce

```bash
bin/phimech synth build /tmp/fixgen/v3.json -o /private/tmp/fixgen/out3 --formats stl,png
```
