"""Tests for the inspection drawing (``phimech/synth/drawing.py``).

Everything about parsing, classification, grouping and sheet layout runs without OpenSCAD on
synthetic loops; the one end-to-end test renders a small part and skips cleanly when OpenSCAD
is absent (same pattern as test_preview.py).
"""

from __future__ import annotations

import math
import os as _os
import re
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - direct runner
    pytest = None  # type: ignore[assignment]

from phimech.synth import drawing as d


class _Skipped(Exception):
    pass


def _skip(message: str) -> None:
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _require_openscad() -> None:
    from phimech.backends.openscad import OpenSCADBackend

    if not OpenSCADBackend().available():
        _skip("openscad is not installed")


# --------------------------------------------------------------------------------------
# synthetic loops
# --------------------------------------------------------------------------------------


def _circle(cx: float, cy: float, dia: float, n: int = 48) -> d.Loop:
    r = dia / 2.0
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def _rect(x0: float, y0: float, x1: float, y1: float) -> d.Loop:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _rounded_rect(x0: float, y0: float, x1: float, y1: float, r: float, n: int = 12) -> d.Loop:
    """Vertices only on the corner arcs, exactly as OpenSCAD's offset()/minkowski emit them."""
    pts: d.Loop = []
    corners = [((x1 - r, y1 - r), 0.0), ((x0 + r, y1 - r), 90.0),
               ((x0 + r, y0 + r), 180.0), ((x1 - r, y0 + r), 270.0)]
    for (cx, cy), start in corners:
        for i in range(n + 1):
            a = math.radians(start + 90.0 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


SAMPLE_SVG = (
    '<svg><path d="\nM 40,-20 L 0,-20 L 0,-0 L 40,-0 z\n'
    'M 25,-15 L 33,-15 L 33,-5 L 25,-5 z\n'
    'M 1040,-20 L 1000,-20 L 1000,-0 L 1040,-0 z\n"/></svg>'
)


def test_parse_openscad_svg_unnegates_y_and_splits_subpaths() -> None:
    loops = d.parse_openscad_svg(SAMPLE_SVG)
    assert len(loops) == 3
    assert d.loop_bbox(loops[0]) == (0.0, 0.0, 40.0, 20.0)
    assert d.loop_bbox(loops[1]) == (25.0, 5.0, 33.0, 15.0)


def test_parse_openscad_svg_refuses_curves() -> None:
    try:
        d.parse_openscad_svg('<svg><path d="M 0,0 C 1,1 2,2 3,3 z"/></svg>')
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("curve commands must be refused, not guessed")


def test_unpack_offsets_bins_by_centroid_and_shifts_back() -> None:
    loops = d.parse_openscad_svg(SAMPLE_SVG)
    groups = d.unpack_offsets(loops, offset=1000.0, count=1)
    assert len(groups) == 2 and len(groups[0]) == 2 and len(groups[1]) == 1
    assert d.loop_bbox(groups[1][0]) == (0.0, 0.0, 40.0, 20.0)


def test_classify_circle_rect_slot_and_rounded_rect_is_not_a_circle() -> None:
    c = d.classify_loop(_circle(10, 20, 6), is_hole=True)
    assert c.kind == "circle" and abs(c.size[0] - 6.0) < 1e-6
    assert abs(c.center[0] - 10) < 1e-6 and abs(c.center[1] - 20) < 1e-6

    r = d.classify_loop(_rect(0, 0, 10, 8), is_hole=True)
    assert r.kind == "rect" and r.size == (10.0, 8.0) and r.center == (5.0, 4.0)

    s = d.classify_loop(_rounded_rect(0, 0, 10, 8, 2), is_hole=True)
    assert s.kind == "slot" and abs(s.corner_r - 2.0) < 0.1

    # the enclosure case: a big rounded rectangle whose vertices are all on its corner arcs
    wall = d.classify_loop(_rounded_rect(0, 0, 67, 88.4, 2, n=8), is_hole=False)
    assert wall.kind != "circle"
    assert wall.kind == "slot" and abs(wall.size[0] - 67) < 1e-6


def test_classify_slice_hole_parity() -> None:
    shapes = d.classify_slice([_rect(0, 0, 40, 20), _circle(10, 10, 6), _rect(25, 5, 33, 15)])
    assert [s.is_hole for s in shapes] == [False, True, True]


def test_cluster_levels_merges_noise_and_caps_count() -> None:
    assert d.cluster_levels([0, 3, 3.001, 4, 9], 9.0) == [0.0, 3.0, 4.0, 9.0]
    many = [i * 0.1 for i in range(201)]  # a fillet: 201 levels over 20 mm
    levels = d.cluster_levels(many, 20.0, max_slices=6)
    assert len(levels) <= 7 and levels[0] == 0.0 and levels[-1] == 20.0
    assert d.slice_positions([0.0, 3.0, 9.0]) == [(1.5, 0.0, 3.0), (6.0, 3.0, 9.0)]


# --------------------------------------------------------------------------------------
# synthetic meshes: levels and wall bands are measured, not clustered
# --------------------------------------------------------------------------------------


def _ring(cx: float, cy: float, r: float, z: float, n: int) -> list:
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n), z)
            for i in range(n)]


def _tube(cx: float, cy: float, r0: float, z0: float, r1: float, z1: float, n: int = 32,
          outward: bool = True) -> list:
    """Wall facets of a cylinder (r0 == r1) or cone, wound so the normals face ``outward``.

    Outward normals are what tells a boss from a bore, so the winding is the point of this
    helper, not an incidental detail."""
    lower, upper = _ring(cx, cy, r0, z0, n), _ring(cx, cy, r1, z1, n)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        quad = [(lower[i], lower[j], upper[j]), (lower[i], upper[j], upper[i])]
        tris.extend(quad if outward else [(t[0], t[2], t[1]) for t in quad])
    return tris


def _annulus(cx: float, cy: float, r_in: float, r_out: float, z: float, n: int = 32) -> list:
    inner, outer = _ring(cx, cy, r_in, z, n), _ring(cx, cy, r_out, z, n)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.extend([(inner[i], outer[i], outer[j]), (inner[i], outer[j], inner[j])])
    return tris


def _slab(x0: float, y0: float, x1: float, y1: float, z: float) -> list:
    return [((x0, y0, z), (x1, y0, z), (x1, y1, z)), ((x0, y0, z), (x1, y1, z), (x0, y1, z))]


def _boss_mesh() -> list:
    """A plate 0..2 carrying a d7 boss whose wall is 3..6 over a d8.6 conical skirt 2..3.

    This is a synthetic standoff in miniature and the whole bug in one part: the only vertex z
    between the two real faces is 3.0, where the cone meets the cylinder and no face exists."""
    tris = []
    tris += _slab(0, 0, 20, 20, 0.0)                       # plate bottom, a real face
    tris += _annulus(10, 10, 4.3, 12.0, 2.0)               # plate top around the skirt
    tris += _tube(10, 10, 4.3, 2.0, 3.5, 3.0)              # the conical skirt: no face on top
    tris += _tube(10, 10, 3.5, 3.0, 3.5, 6.0)              # the boss wall, the real diameter
    tris += _annulus(10, 10, 0.0, 3.5, 6.0)                # boss top, a real face
    return tris


def test_face_levels_keep_faces_and_drop_the_ring_a_cone_leaves() -> None:
    mesh = _boss_mesh()
    levels = d.face_levels(mesh, 2)
    assert [round(lv.value, 3) for lv in levels] == [0.0, 2.0, 6.0]
    assert all(lv.face_area >= d.MIN_LEVEL_FACE_AREA for lv in levels)
    # the old rule - distinct vertex coordinates - promoted the cone/cylinder ring to a level,
    # and every depth quoted against z=3 was then a property of the tessellation
    assert 3.0 in d.cluster_levels([v[2] for tri in mesh for v in tri], 6.0)
    assert 3.0 not in [lv.value for lv in levels]


def test_face_levels_always_keep_the_extremes_even_without_a_face() -> None:
    cone = _tube(0, 0, 4.0, 0.0, 0.0, 5.0)                 # a spike: no horizontal face at all
    levels = d.face_levels(cone, 2)
    assert [round(lv.value, 3) for lv in levels] == [0.0, 5.0]
    assert all(lv.extreme for lv in levels)


def test_face_levels_cap_keeps_the_biggest_faces() -> None:
    mesh = _slab(0, 0, 40, 40, 0.0) + _slab(0, 0, 1, 1, 9.0)
    for i in range(1, 9):                                  # eight small intermediate faces
        mesh += _slab(0, 0, 3, 3, float(i))
    levels = d.face_levels(mesh, 2, max_levels=4)
    assert [lv.value for lv in levels[:1]] == [0.0] and levels[-1].value == 9.0
    assert len(levels) == 4


def test_section_cuts_mark_a_sloped_band_as_a_transition() -> None:
    levels = [d.Level(0.0, 100.0, True), d.Level(10.0, 100.0, True)]
    coords = [0.0, 10.0] + [1.0 + 0.5 * i for i in range(9)]   # a tessellated run 1..5
    cuts = d.section_cuts(levels, coords)
    assert [c.position for c in cuts if not c.transition] == [5.0]
    transitions = [c for c in cuts if c.transition]
    assert len(transitions) == 1 and 1.0 < transitions[0].position < 5.0
    # a prismatic band earns no extra cut
    assert [c.transition for c in d.section_cuts(levels, [0.0, 10.0])] == [False]


def test_wall_bands_measure_diameter_extent_parity_and_the_skirt() -> None:
    bands = d.wall_bands(_boss_mesh(), 2)
    boss = [b for b in bands if b.circular and not b.is_hole]
    assert len(boss) == 1
    band = boss[0]
    assert abs(2 * band.radius - 7.0) < 0.02        # the wall, never the 8.6 mm skirt
    assert (round(band.lo, 3), round(band.hi, 3)) == (3.0, 6.0)
    assert abs(band.center[0] - 10.0) < 0.01 and abs(band.center[1] - 10.0) < 0.01
    assert len(band.skirts) == 1
    skirt = band.skirts[0]
    assert abs(skirt.at - 3.0) < 1e-6
    assert abs(skirt.outward - 0.8) < 0.02 and abs(skirt.run - 1.0) < 0.02


def test_wall_bands_read_parity_from_the_facet_normals() -> None:
    plate = _slab(0, 0, 20, 20, 0.0) + _slab(0, 0, 20, 20, 5.0)
    bore = _tube(10, 10, 2.0, 0.0, 2.0, 5.0, outward=False)
    bands = [b for b in d.wall_bands(plate + bore, 2) if b.circular]
    assert len(bands) == 1 and bands[0].is_hole
    assert abs(2 * bands[0].radius - 4.0) < 0.02


def test_wall_bands_do_not_call_a_rounded_rectangle_a_circle() -> None:
    """A tray outline's vertices all sit on its corner arcs, so they are equidistant too."""
    corners = [(58.0, 68.0), (5.0, 68.0), (5.0, 5.0), (58.0, 5.0)]
    lower = [pt for c in corners for pt in
             [(c[0] + 2 * math.cos(math.radians(a)), c[1] + 2 * math.sin(math.radians(a)), 0.0)
              for a in range(0, 360, 20)]]
    tris = []
    for i in range(len(lower)):
        j = (i + 1) % len(lower)
        a, b = lower[i], lower[j]
        tris.extend([(a, b, (b[0], b[1], 4.0)), (a, (b[0], b[1], 4.0), (a[0], a[1], 4.0))])
    assert not any(band.circular for band in d.wall_bands(tris, 2))


def test_measure_features_takes_the_wall_and_not_the_skirt_section() -> None:
    """The reported diameter is the boss's wall; a cut through its base fillet is not it."""
    skirt = d.classify_loop(_circle(3.5, 3.5, 7.29), is_hole=False)
    f = d.Feature(skirt, "top", 2.0, 6.0, False)
    f.sections = [(skirt, 4.0, True)]                     # only ever seen in a transition cut
    band = d.WallBand(2, (3.5, 3.5), (7.0, 7.0), 3.5, True, 3.0, 6.0, False, 66.0,
                      (d.Skirt(3.0, 0.8, 1.0),))
    d.measure_features([f], [band])
    assert f.measured and abs(f.shape.size[0] - 7.0) < 0.02
    assert (f.depth_from, f.depth_to) == (3.0, 6.0)
    assert f.segments == [(3.0, 6.0)]
    assert f.skirts and abs(f.skirts[0].outward - 0.8) < 1e-9
    assert f.to_dict()["skirts"][0]["run"] == 1.0


def test_measure_features_thru_extent_is_the_material_at_that_xy() -> None:
    """A hole through a 8 mm plate is 8 deep even when the part is 45 tall."""
    shape = d.classify_loop(_circle(20, 10, 5.2), is_hole=True)
    f = d.Feature(shape, "top", math.inf, -math.inf, True)
    band = d.WallBand(2, (20.0, 10.0), (5.2, 5.2), 2.6, True, 0.0, 8.0, True, 130.0)
    d.measure_features([f], [band])
    d._finish_depths([f], [0.0, 8.0, 35.0, 45.0])
    assert (f.depth_from, f.depth_to) == (0.0, 8.0)
    assert f.segments == [(0.0, 8.0)]


def test_measure_features_reports_each_run_of_material_separately() -> None:
    """Two slabs with a gap: two segments, so no hidden prism is drawn through the gap."""
    shape = d.classify_loop(_circle(5, 5, 4.0), is_hole=True)
    f = d.Feature(shape, "top", math.inf, -math.inf, True)
    bands = [d.WallBand(2, (5.0, 5.0), (4.0, 4.0), 2.0, True, 0.0, 3.0, True, 40.0),
             d.WallBand(2, (5.0, 5.0), (4.0, 4.0), 2.0, True, 10.0, 13.0, True, 40.0)]
    d.measure_features([f], bands)
    assert f.segments == [(0.0, 3.0), (10.0, 13.0)]
    assert (f.depth_from, f.depth_to) == (0.0, 13.0)
    boxes = d.feature_extent_boxes(f, "front")
    assert boxes == [(3.0, 0.0, 7.0, 3.0), (3.0, 10.0, 7.0, 13.0)]


def test_measure_features_recognises_a_counterbore_as_a_coaxial_wall() -> None:
    shape = d.classify_loop(_circle(20, 10, 5.2), is_hole=True)
    f = d.Feature(shape, "top", 0.0, 8.0, True)
    f.sections = [(shape, 8.0, False)]
    bands = [d.WallBand(2, (20.0, 10.0), (5.2, 5.2), 2.6, True, 0.0, 4.0, True, 65.0),
             d.WallBand(2, (20.0, 10.0), (9.5, 9.5), 4.75, True, 4.0, 8.0, True, 119.0)]
    d.measure_features([f], bands)
    assert f.counterbore is not None
    assert f.counterbore["diameter"] == 9.5 and f.counterbore["depth"] == 4.0


def test_measure_features_leaves_a_wall_it_cannot_find_alone() -> None:
    """A step fused to the part outline has no wall of its own; it says so rather than guess."""
    shape = d.classify_loop(_rounded_rect(0, 0, 8, 40, 3), is_hole=False)
    f = d.Feature(shape, "top", 8.0, 45.0, False)
    f.sections = [(shape, 37.0, False)]
    outline = d.WallBand(2, (30.0, 20.0), (60.0, 40.0), 0.0, False, 0.0, 44.0, False, 4829.0)
    d.measure_features([f], [outline])
    assert not f.measured
    d._finish_depths([f], [0.0, 8.0, 35.0, 45.0])
    assert (f.depth_from, f.depth_to) == (8.0, 45.0)      # bounded by real faces even so
    assert f.segments == [(8.0, 45.0)]


def test_split_slivers_by_width_and_by_size() -> None:
    crevice = d.classify_loop(
        [(0.0, 0.0), (6.0, 0.0), (6.0, 0.2), (0.0, 0.2)], is_hole=True)   # 0.2 mm wide
    kept, slivers = d.split_slivers([d.Feature(crevice, "top", 2.0, 22.1, False)])
    assert kept == [] and len(slivers) == 1
    assert slivers[0]["why"].startswith("narrower")
    assert slivers[0]["width_mm"] < d.SLIVER_WIDTH_MM

    nick = d.classify_loop(_circle(0, 0, 1.2), is_hole=True)              # 1.13 mm2, 0.5 deep
    kept, slivers = d.split_slivers([d.Feature(nick, "top", 0.0, 0.5, False)])
    assert kept == [] and slivers[0]["why"].startswith("under")

    real = d.classify_loop(_circle(0, 0, 1.6), is_hole=True)              # a real small hole
    kept, slivers = d.split_slivers([d.Feature(real, "top", 0.0, 5.0, False)])
    assert len(kept) == 1 and slivers == []


def test_box_in_silhouette_rejects_a_prism_that_leaves_the_part() -> None:
    plate = _rect(0, 0, 40, 8)
    assert d.box_in_silhouette((10.0, 1.0, 20.0, 7.0), [plate])
    assert not d.box_in_silhouette((10.0, 1.0, 20.0, 12.0), [plate])
    # a hole in the silhouette is a loop too, so parity keeps a box out of it
    assert not d.box_in_silhouette((18.0, 3.0, 22.0, 5.0), [plate, _rect(15, 2, 25, 6)])


def test_aggregate_features_depth_thru_and_taper_merge() -> None:
    plate = _rect(0, 0, 60, 40)
    thru = _circle(15, 20, 6)
    boss = _circle(45, 30, 8)
    bore = _circle(45, 30, 2.7)
    fillet_ring = _circle(45, 30, 8.3)   # boss base fillet seen in the first slice above the plate
    silhouette = [plate, thru]
    slices = [                       # above z=3 only the boss exists, as a real cut shows
        (1.5, 0.0, 3.0, [plate, thru]),
        (3.2, 3.0, 3.4, [fillet_ring]),
        (3.7, 3.4, 4.0, [boss]),
        (6.5, 4.0, 9.0, [boss, bore]),
    ]
    feats = d.aggregate_features("top", silhouette, slices)
    d._finish_depths(feats, [0.0, 3.0, 3.4, 4.0, 9.0])
    by = {(f.shape.kind, f.shape.is_hole, round(f.shape.size[0], 2)): f for f in feats}
    assert by[("circle", True, 6.0)].thru
    assert (by[("circle", True, 6.0)].depth_from, by[("circle", True, 6.0)].depth_to) == (0.0, 9.0)
    b = by[("circle", False, 8.0)]            # named by its widest span, not the fillet ring
    assert (b.depth_from, b.depth_to) == (3.0, 9.0) and not b.thru
    assert ("circle", False, 8.3) not in by
    bore_f = by[("circle", True, 2.7)]
    assert (bore_f.depth_from, bore_f.depth_to) == (4.0, 9.0)


def test_group_features_ids_and_quantities() -> None:
    def feat(kind: str, hole: bool, dia: float, cx: float, cy: float, z0: float, z1: float,
             thru: bool) -> d.Feature:
        shape = d.classify_loop(_circle(cx, cy, dia), is_hole=hole)
        return d.Feature(shape, "top", z0, z1, thru)

    feats = [feat("circle", True, 2.7, 45, 30, 4, 9, False), feat("circle", True, 2.7, 45, 10, 4, 9, False),
             feat("circle", True, 6, 15, 20, 0, 9, True), feat("circle", False, 8, 45, 30, 3, 9, False)]
    groups = d.group_features(feats)
    ids = [(g.group_id, len(g.members), g.thru) for g in groups]
    assert ids == [("H1", 1, True), ("H2", 2, False), ("B1", 1, False)]
    assert groups[1].depth == 5.0
    assert all(m.group_id == "H2" for m in groups[1].members)


def test_group_prints_its_representative_not_whichever_member_came_first() -> None:
    """Grouping quantises to 0.05 mm, so both of these share a bucket; the row must not depend
    on which one the bucket saw first."""
    wide = d.Feature(d.classify_loop(_circle(10, 10, 7.32), is_hole=False), "top", 3, 6, False)
    narrow = d.Feature(d.classify_loop(_circle(30, 10, 7.28), is_hole=False), "top", 3, 6, False)
    forwards = d.group_features([wide, narrow])
    backwards = d.group_features([narrow, wide])
    assert len(forwards) == len(backwards) == 1
    assert forwards[0].label == backwards[0].label == "⌀7.3"
    assert forwards[0].to_dict()["size"] == backwards[0].to_dict()["size"] == [7.3, 7.3]
    assert (forwards[0].depth_from, forwards[0].depth_to) == (3.0, 6.0)
    # the callout, the table row and the machine-readable record all say the same thing
    assert d._group_callout(forwards[0]) == "B1  2X BOSS ⌀7.3, 3 HIGH"
    assert d._depth_text(forwards[0]) == "Z 3 TO 6 (3)"
    assert forwards[0].to_dict()["depth"] == 3.0
    assert {m.shape.label for m in forwards[0].members} == {"⌀7.32", "⌀7.28"}


def test_feature_extents_projects_a_prism_into_the_side_views() -> None:
    shape = d.classify_loop(_circle(45, 30, 8), is_hole=False)
    f = d.Feature(shape, "top", 3.0, 9.0, False)
    assert d.feature_extents(f, "front") == (41.0, 3.0, 49.0, 9.0)   # x by z
    assert d.feature_extents(f, "right") == (26.0, 3.0, 34.0, 9.0)   # y by z


def test_choose_scale_prefers_the_largest_standard_scale_that_fits() -> None:
    assert d.choose_scale(100, 50, 260, 140) == 2
    assert d.choose_scale(1000, 50, 260, 140) == 0.2
    assert d.choose_scale(10000, 10000, 260, 140) == d.STANDARD_SCALES[-1]


def test_fit_scale_says_when_nothing_fits() -> None:
    assert d.fit_scale(100, 50, 260, 140) == 2
    assert d.fit_scale(10000, 10000, 260, 140) is None


def test_split_table_rows_never_drops_or_reorders_a_row() -> None:
    rows = [{"id": "R{}".format(i), "lines": [""] * (1 + i % 3)} for i in range(30)]
    for first in (0, 5, 24, 200):
        pages = d.split_table_rows(rows, first, 20)
        assert [r for page in pages for r in page] == rows, first
        for index, page in enumerate(pages):
            capacity = first if index == 0 else 20
            continued = index + 1 < len(pages) and bool(page)
            used = sum(len(r["lines"]) for r in page) + (1 if continued else 0)
            assert used <= capacity or len(page) == 1, (first, index)
    assert d.split_table_rows(rows, 0, 20)[0] == []          # all of it moves to sheet 2
    assert len(d.split_table_rows(rows, 200, 20)) == 1        # all of it fits sheet 1


def _tray(holes: int = 30):
    """A 130 x 210 x 40 tray (a sample tray base) with many distinct holes."""
    w, h, z = 130.0, 210.0, 40.0
    features = []
    loops = [_rect(0, 0, w, h)]
    for i in range(holes):
        loop = _circle(10 + (i % 5) * 26, 12 + (i // 5) * 32, 2.0 + 0.3 * i)
        loops.append(loop)
        features.append(d.Feature(d.classify_loop(loop, True), "top", 0.0, z, True))
    top = d.ViewData("top", loops, [], [], (0, 0, w, h), [0.0, z])
    top.features.extend(features)
    views = {"top": top,
             "front": d.ViewData("front", [_rect(0, 0, w, z)], [], [], (0, 0, w, z), []),
             "right": d.ViewData("right", [_rect(0, 0, h, z)], [], [], (0, 0, h, z), [])}
    return views, d.group_features(features)


def test_a_217_mm_tray_is_drawn_at_1_to_2_with_every_feature_on_some_sheet() -> None:
    """2026-10-08: a sample tray base came out at 1:10 in the top-left corner, with 12 of 33
    feature groups "not fitting this sheet".  A full-width table band left the views a
    sliver of height; the table now moves to sheet 2 when that buys a larger scale."""
    views, groups = _tray()
    assert len(groups) == 30
    sheets, scale, layout = d.compose_sheets(views, groups, title="tray",
                                             source_label="tray.stl", volume=1.0,
                                             openscad_version="test", part=None)
    assert scale == 0.5
    assert len(sheets) == layout["sheets"] >= 2
    assert layout["table_overflow"] == []
    ids = [g.group_id for g in groups]
    assert sorted(i for page in layout["table_sheets"] for i in page) == sorted(ids)
    printed = "".join(sheets)
    for gid in ids:
        assert re.search(r">{}  ".format(gid), printed), gid    # a table row, on some sheet
    assert dict(layout["title_fields"])["SHEET"] == "1 / {}".format(len(sheets))
    if layout["table_sheets"][0]:
        assert "TABLE CONTINUES ON SHEET 2" in sheets[0]
    else:
        assert "FEATURE TABLE ON SHEET 2" in sheets[0]
    assert "SHEET" in sheets[1] and "{} / {}".format(2, len(sheets)) in sheets[1]
    import xml.dom.minidom

    for text in sheets:
        xml.dom.minidom.parseString(text)


def _plate_with_holes(w: float, h: float, z: float, holes: int):
    features = []
    loops = [_rect(0, 0, w, h)]
    for i in range(holes):
        loop = _circle(6 + (i % 4) * (w - 12) / 3.0, 6 + (i // 4) * 14, 2.0 + 0.4 * i)
        loops.append(loop)
        features.append(d.Feature(d.classify_loop(loop, True), "top", 0.0, z, True))
    top = d.ViewData("top", loops, [], [], (0, 0, w, h), [0.0, z])
    top.features.extend(features)
    views = {"top": top,
             "front": d.ViewData("front", [_rect(0, 0, w, z)], [], [], (0, 0, w, z), []),
             "right": d.ViewData("right", [_rect(0, 0, h, z)], [], [], (0, 0, h, z), [])}
    return views, d.group_features(features)


_ISO_2768_M = {"manufacturing": {"general_tolerance": {"standard": "ISO 2768-m"}}}


def test_a_short_table_stays_on_one_sheet_at_1_to_1_rather_than_2_to_1_on_two() -> None:
    """The LiDAR warm-up plate (65 x 65 x 6, 5 rows) fitted at 1:1 on one sheet; moving its
    table to sheet 2 to reach 2:1 is a second sheet nobody needed."""
    views, groups = _plate_with_holes(65.0, 65.0, 6.0, 5)
    sheets, scale, layout = d.compose_sheets(views, groups, title="lidar",
                                             source_label="lidar.stl", volume=1.0,
                                             openscad_version="test", part=None)
    assert len(sheets) == 1 and scale == 1
    assert layout["arrangement"]["table_on_sheet_1"]
    two = [c for c in layout["scale_candidates"] if c["sheets"] == 2 and c["fits"]]
    assert two and max(c["scale"] for c in two) == 2     # the bigger scale was on offer


def test_a_short_table_goes_beside_the_views_before_it_goes_to_sheet_2() -> None:
    """The sample tray lid (130 x 210 x 3.5, 7 rows, ISO 2768-m declared) went to 1:2 with its
    table on sheet 2; this synthetic variant keeps it on sheet 1 under the views."""
    views, groups = _plate_with_holes(130.0, 210.0, 3.5, 7)
    tol = d.resolve_tolerance_block(ir=_ISO_2768_M, source_manifest=_ISO_2768_M)
    assert tol["specified"]
    sheets, scale, layout = d.compose_sheets(views, groups, title="lid", source_label="lid.stl",
                                             volume=1.0, openscad_version="test", part=None,
                                             tolerance=tol)
    assert scale == 0.5 and len(sheets) == 1
    assert layout["arrangement"]["table_position"] == "under the views"
    assert layout["table_sheets"] == [[g.group_id for g in groups]]
    assert dict(layout["title_fields"])["SHEET"] == "1 / 1"


def test_choose_arrangement_prefers_fewer_sheets_unless_they_are_below_1_to_1() -> None:
    def option(scale: float, sheets: int, fits: bool = True) -> dict:
        return {"scale": scale, "fits": fits, "pages": [[{}]] * sheets}

    assert d.choose_arrangement([option(1, 1), option(2, 2)])["scale"] == 1
    assert d.choose_arrangement([option(0.2, 1), option(0.5, 2)])["scale"] == 0.5
    assert d.choose_arrangement([option(0.5, 1), option(0.5, 2)])["pages"].__len__() == 1
    # the first of a sheet count is kept on a tie, and a strictly larger scale replaces it
    first, later = option(0.5, 1), option(0.5, 1)
    assert d.choose_arrangement([first, later]) is first
    assert d.choose_arrangement([option(0.2, 1), option(0.5, 1)])["scale"] == 0.5
    assert d.choose_arrangement([option(0.1, 1, fits=False), option(0.2, 2)])["scale"] == 0.2


def test_a_small_part_keeps_the_original_single_sheet_layout() -> None:
    sheets, scale, layout = d.compose_sheets(_plate_views(), [], title="plate",
                                             source_label="plate.stl", volume=1.0,
                                             openscad_version="test", part=None)
    assert len(sheets) == 1 and scale == 2
    assert layout["arrangement"] == {"stepped": False, "tolerance_block": "right",
                                     "table_on_sheet_1": False,
                                     "table_position": "under the views", "sheets": 1}
    assert dict(layout["title_fields"])["SHEET"] == "1 / 1"


def test_standard_scales_are_the_iso_5455_series() -> None:
    """1.25, 1.5, 2.5, 3, 4, 0.75, 0.4, 0.3 and 0.15 are not scales anybody draws to."""
    assert set(d.STANDARD_SCALES) == {10, 5, 2, 1, 0.5, 0.2, 0.1}
    assert list(d.STANDARD_SCALES) == sorted(d.STANDARD_SCALES, reverse=True)
    for scale, text in ((1, "1:1"), (2, "2:1"), (0.5, "1:2"), (0.1, "1:10")):
        assert d._scale_text(scale) == text


def test_compose_sheet_writes_valid_svg_with_callouts_and_table() -> None:
    plate = _rect(0, 0, 60, 40)
    thru = _circle(15, 20, 6)
    top = d.ViewData("top", [plate, thru], [], [], (0, 0, 60, 40), [0.0, 3.0, 9.0])
    f = d.Feature(d.classify_loop(thru, True), "top", 0.0, 9.0, True)
    top.features.append(f)
    front = d.ViewData("front", [_rect(0, 0, 60, 9)], [], [], (0, 0, 60, 9), [])
    right = d.ViewData("right", [_rect(0, 0, 40, 9)], [], [], (0, 0, 40, 9), [])
    groups = d.group_features([f])
    svg, scale, layout = d.compose_sheet({"top": top, "front": front, "right": right}, groups,
                                         title="plate", source_label="plate.stl", volume=7000.0,
                                         openscad_version="test", part=None)
    assert svg.startswith("<?xml") and svg.rstrip().endswith("</svg>")
    assert scale in d.STANDARD_SCALES
    assert "H1" in svg and "THRU" in svg and "\u23006" in svg
    assert "TOP" in svg and "FRONT" in svg and "RIGHT" in svg
    # W6: one frame.  The model-coordinate datum note is gone; an origin indicator is drawn.
    assert "datum (" not in svg and 'class="origin"' in svg
    assert layout["origin_mm"]["top"] == [0.0, 0.0]
    import xml.dom.minidom

    xml.dom.minidom.parseString(svg)   # well-formed


# --------------------------------------------------------------------------------------
# end to end (OpenSCAD)
# --------------------------------------------------------------------------------------

_PART = """
$fn = 48;
difference() {
  union() { cube([60, 40, 3]); translate([45, 30, 3]) cylinder(d=8, h=6);
            translate([45, 10, 3]) cylinder(d=8, h=6); }
  translate([15, 20, -1]) cylinder(d=6, h=10);
  translate([45, 30, 4]) cylinder(d=2.7, h=10);
  translate([45, 10, 4]) cylinder(d=2.7, h=10);
  translate([25, 8, -1]) cube([10, 8, 10]);
}
"""


def test_find_chrome_honours_the_override_and_never_invents_a_binary() -> None:
    import os

    from phimech.synth import raster

    saved = os.environ.get("PHIMECH_CHROME")
    try:
        os.environ["PHIMECH_CHROME"] = "/nonexistent/chrome-for-phimech-test"
        assert raster.find_chrome() is None      # an override that is not a file is not a tool
        assert d._find_chrome() is None
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "chrome"
            fake.write_text("", encoding="utf-8")
            os.environ["PHIMECH_CHROME"] = str(fake)
            assert d._find_chrome() == str(fake)
    finally:
        if saved is None:
            os.environ.pop("PHIMECH_CHROME", None)
        else:
            os.environ["PHIMECH_CHROME"] = saved


def test_manifest_facts_reads_a_design_irs_declared_material() -> None:
    """`manufacturing.material` in a PhiMech design IR is the author's declaration, the same
    as `component.material` on a handoff manifest; without either the sheet says UNSPECIFIED."""
    facts = d.manifest_facts({"project": "p", "rev": "B", "manufacturing": {"material": "PETG"}})
    assert facts["material"] == "PETG" and facts["revision"] == "B"
    assert d.manifest_facts({"project": "p", "manufacturing": {"process": "fdm"}})["material"] is None
    handoff = d.manifest_facts({"component": {"material": "6061-T6"}, "manufacturing": {"material": "PETG"}})
    assert handoff["material"] == "6061-T6"    # the handoff's own record wins over a stray block


def test_to_pdf_returns_none_without_a_converter_and_leaves_no_wrapper_behind() -> None:
    import os

    saved = (os.environ.get("PHIMECH_CHROME"), d.shutil.which)
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "sheet.svg"
        svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        os.environ["PHIMECH_CHROME"] = "/nonexistent/chrome"
        d.shutil.which = lambda name: None  # type: ignore[assignment]
        try:
            assert d.to_pdf(svg, Path(tmp) / "sheet.pdf") is None
            assert d.rasterize(svg, Path(tmp) / "sheet.png") is None
        finally:
            d.shutil.which = saved[1]  # type: ignore[assignment]
            if saved[0] is None:
                os.environ.pop("PHIMECH_CHROME", None)
            else:
                os.environ["PHIMECH_CHROME"] = saved[0]
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["sheet.svg"]


def test_pdf_wrapper_declares_the_sheet_as_the_page() -> None:
    html = d._HTML_WRAPPER.format(w="420mm", h="297mm", page="@page{size:420mm 297mm;margin:0}",
                                  src="file:///x.svg")
    assert "@page{size:420mm 297mm;margin:0}" in html and "width:420mm" in html


def test_drawing_result_carries_pdf_fields_in_its_dict() -> None:
    fields = {f.name for f in d.dataclasses_fields(d.DrawingResult)} if hasattr(d, "dataclasses_fields") \
        else set(d.DrawingResult.__dataclass_fields__)
    assert {"pdf", "pdf_tool"} <= fields


def test_render_drawing_end_to_end() -> None:
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        scad = Path(tmp) / "part.scad"
        scad.write_text(_PART, encoding="utf-8")
        result = d.render_drawing(scad, Path(tmp) / "part.svg", png=False)
        assert Path(result.path).is_file() and result.png is None
        assert result.bbox_mm["max"] == [60.0, 40.0, 9.0]
        assert result.levels_z == [0.0, 3.0, 4.0, 9.0]
        groups = {(g["kind"], g["hole"], round(g["size"][0], 2)): g for g in result.groups}
        assert groups[("circle", True, 6.0)]["thru"]
        assert groups[("rect", True, 10.0)]["thru"]
        bores = groups[("circle", True, 2.7)]
        assert bores["qty"] == 2 and not bores["thru"] and bores["depth"] == 5.0
        bosses = groups[("circle", False, 8.0)]
        assert bosses["qty"] == 2 and bosses["depth_from"] == 3.0 and bosses["depth_to"] == 9.0
        assert sorted(bosses["positions"]) == [[45.0, 10.0], [45.0, 30.0]]


#: The part that would have caught the measurement bug. Every z level of ``_PART`` above
#: happens to be a real face, which is why a slice grid built from vertex coordinates looked
#: right there for so long. Here the cone meeting the cylinder puts a vertex ring at z=5 where
#: no face exists, and the thru hole passes through a 4 mm plate under a 12 mm part - so a
#: level list built from vertices prints a height that is not one, and a depth taken from the
#: level span draws a hole through 8 mm of air.
_STEPPED_PART = """
$fn = 32;
difference() {
  union() {
    cube([40, 24, 4]);
    translate([6, 12, 4]) cylinder(d1=9, d2=7, h=1);
    translate([6, 12, 4]) cylinder(d=7, h=8);
    translate([34, 12, 4]) cylinder(d1=9, d2=7, h=1);
    translate([34, 12, 4]) cylinder(d=7, h=8);
  }
  translate([6, 12, 2]) cylinder(d=3, h=20);
  translate([34, 12, 2]) cylinder(d=3, h=20);
  translate([20, 12, -1]) cylinder(d=6, h=20);
}
"""


def test_render_drawing_levels_depths_and_diameters_are_measured_not_sliced() -> None:
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        scad = Path(tmp) / "stepped.scad"
        scad.write_text(_STEPPED_PART, encoding="utf-8")
        result = d.render_drawing(scad, Path(tmp) / "stepped.svg", png=False)
        record = result.to_dict()

        # W2: every printed level is a face. z=5 is a vertex ring where the cone meets the
        # cylinder and no face exists, so it is not a level however many vertices sit there.
        assert result.levels_z == [0.0, 2.0, 4.0, 12.0]
        areas = [entry["area_mm2"] for entry in record["level_faces"]]
        assert min(areas) >= d.MIN_LEVEL_FACE_AREA, areas

        groups = {(g["kind"], g["hole"], round(g["size"][0], 2)): g for g in record["groups"]}

        # W4: the thru hole spans the plate it passes through, not the height of the part.
        thru = groups[("circle", True, 6.0)]
        assert thru["thru"] and (thru["depth_from"], thru["depth_to"]) == (0.0, 4.0)
        assert thru["segments"] == [[0.0, 4.0]]

        # W3: the boss is its wall (7.0), not a section through its 9 mm conical base, and it
        # starts where the wall starts, not at the level below.
        boss = groups[("circle", False, 7.0)]
        assert boss["qty"] == 2 and boss["measured"]
        assert abs(boss["size"][0] - 7.0) <= 0.02
        assert (boss["depth_from"], boss["depth_to"]) == (5.0, 12.0)
        skirt = boss["skirts"][0]
        assert abs(skirt["outward"] - 1.0) <= 0.02 and abs(skirt["run"] - 1.0) <= 0.02

        bore = groups[("circle", True, 3.0)]
        assert bore["qty"] == 2 and not bore["thru"]
        assert (bore["depth_from"], bore["depth_to"]) == (2.0, 12.0) and bore["depth"] == 10.0

        # nothing hidden is drawn where the part has no material
        svg = Path(result.path).read_text(encoding="utf-8")
        assert "⌀7" in svg and "⌀6" in svg
        assert "⌀9" not in svg                # the skirt is never a diameter


def test_mcp_impl_rejects_relative_and_non_svg_paths() -> None:
    from phimech.mcp.tools_drawing import drawing_sheet_impl

    out = drawing_sheet_impl("relative.stl", "/tmp/x.svg")
    assert out["ok"] is False and "absolute" in out["error"]
    out = drawing_sheet_impl(str(Path(__file__).resolve()), "/tmp/x.png")
    assert out["ok"] is False and ".svg" in out["error"]


# --------------------------------------------------------------------------------------
# W5-W10: what gets dimensioned, how it is labelled, and what the title block says
# --------------------------------------------------------------------------------------

_LINE_RE = re.compile(r"<line\b([^>]*?)/>")
_POLY_RE = re.compile(r"<polygon\b([^>]*?)/>")
_ATTR_RE = re.compile(r'([A-Za-z_][-A-Za-z0-9_]*)\s*=\s*"([^"]*)"')
_LEADER_RE = re.compile(r'<g class="leader" data-id="([^"]+)">(.*?)</g>', re.S)


def _svg_lines(svg: str, cls: str) -> list:
    found = []
    for match in _LINE_RE.finditer(svg):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        if attrs.get("class") == cls:
            found.append(((float(attrs["x1"]), float(attrs["y1"])),
                          (float(attrs["x2"]), float(attrs["y2"]))))
    return found


def _svg_polys(svg: str, cls: str) -> list:
    found = []
    for match in _POLY_RE.finditer(svg):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        if attrs.get("class") == cls:
            found.append([tuple(float(v) for v in p.split(","))
                          for p in attrs["points"].split()])
    return found


def _paper(record: dict, view: str, point) -> tuple:
    layout = record["layout"]
    ox, oy = layout["origin_paper"][view]
    mx, my = layout["origin_mm"][view]
    scale = layout["paper_scale"]
    return (ox + scale * (point[0] - mx), oy - scale * (point[1] - my))



def coaxial_duplicates(record: dict) -> list:
    """Printed groups that describe one hole twice: the defect the fold exists to prevent.

    Two rules, both from the counterbored bracket: no printed group may sit at the positions
    of another printed group's *declared counterbore* with that diameter and z range, and no
    two printed circular holes may share a position set at all.  A bore inside a boss shares
    positions too, which is why the second rule is holes-only."""
    printed = list(record["groups"])

    def same(a: dict, b: dict) -> bool:
        return sorted(tuple(p) for p in a["positions"]) == \
            sorted(tuple(p) for p in b["positions"])

    found = []
    for group in printed:
        cbore = group.get("counterbore") or {}
        diameter = cbore.get("diameter")
        if diameter is None:
            continue
        for other in printed:
            if other is group or not same(other, group):
                continue
            if (abs(other["size"][0] - diameter) <= 0.3
                    and abs(other["depth_from"] - cbore["from"]) <= 0.2
                    and abs(other["depth_to"] - cbore["to"]) <= 0.2):
                found.append(("counterbore printed twice", group["id"], other["id"]))
    for i, a in enumerate(printed):
        for b in printed[i + 1:]:
            if a["hole"] and b["hole"] and a["kind"] == "circle" and b["kind"] == "circle" \
                    and same(a, b):
                found.append(("coaxial holes at one position set", a["id"], b["id"]))
    return found


def audit_sheet(record: dict, svg: str) -> dict:
    """The presentation gates, measured off the finished sheet rather than trusted.

    ``drawing_checks`` owns coverage, collision, clipping and traceability; these are the
    ones it does not yet have - where a leader goes, what it crosses, whether the table and
    the view agree to the character, and whether a printed tolerance has a provenance."""
    from phimech.check import drawing_checks as dc

    texts = dc.parse_texts(svg)
    outline = _svg_polys(svg, "outline")
    geometry = outline + _svg_polys(svg, "section")
    witness = _svg_lines(svg, "ext") + _svg_lines(svg, "dim")
    leaders = [(m.group(1), _svg_lines(m.group(2), "leader"))
               for m in _LEADER_RE.finditer(svg)]
    out = {"witness_lines": len(witness), "leaders": len(leaders)}

    inside = 0
    for (x0, y0), (x1, y1) in witness:
        if sum(d.polyline_loop_crossings([(x0, y0), (x1, y1)], lp) for lp in outline):
            inside += 1
            continue
        for t in (0.25, 0.5, 0.75):
            probe = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if sum(1 for lp in outline if d.point_in_loop(probe, lp)) % 2 == 1:
                inside += 1
                break
    out["witness_lines_over_the_part"] = inside

    segments = [seg for _, segs in leaders for seg in segs]
    out["leader_leader_crossings"] = sum(
        1 for i, a in enumerate(segments) for b in segments[i + 1:]
        if d.segments_cross(a[0], a[1], b[0], b[1]))
    out["leader_witness_crossings"] = sum(
        1 for a in segments for b in witness if d.segments_cross(a[0], a[1], b[0], b[1]))

    centres = {}
    for group in record["groups"]:
        centres[group["id"]] = [_paper(record, group["view"], p) for p in group["positions"]]
    for mark in record.get("markings", []):
        x0, y0, x1, y1 = mark["bbox"]
        centres[mark["id"]] = [_paper(record, mark["view"],
                                      ((x0 + x1) / 2.0, (y0 + y1) / 2.0))]
    radius = {g["id"]: record["layout"]["paper_scale"] * max(g["size"]) / 2.0
              for g in record["groups"]}
    for mark in record.get("markings", []):
        x0, y0, x1, y1 = mark["bbox"]
        radius[mark["id"]] = record["layout"]["paper_scale"] * max(x1 - x0, y1 - y0) / 2.0
    missed, unpermitted = [], []
    for ident, segs in leaders:
        start = segs[0][0]
        near = min(centres[ident], key=lambda c: math.hypot(c[0] - start[0], c[1] - start[1]))
        if math.hypot(near[0] - start[0], near[1] - start[1]) > radius[ident] + 0.05:
            missed.append(ident)
        for loop in geometry:
            allowed = 1 if d.point_in_loop(near, loop) else 0
            hits = sum(d.polyline_loop_crossings(list(seg), loop) for seg in segs)
            if hits > allowed:
                unpermitted.append((ident, hits, allowed))
    out["leaders_off_their_feature"] = missed
    out["leader_geometry_crossings_unpermitted"] = unpermitted

    out["centre_marks"] = svg.count('<g class="cmark">')
    out["circular_members"] = sum(g["qty"] for g in record["groups"]
                                  if g["kind"] == "circle")

    worst = 1.0e9
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            worst = min(worst, max(max(a.bbox[0] - b.bbox[2], b.bbox[0] - a.bbox[2]),
                                   max(a.bbox[1] - b.bbox[3], b.bbox[1] - a.bbox[3])))
    out["min_text_separation_mm"] = round(worst, 3)

    out["off_grammar"] = [t.content for t in texts
                          if re.match(r"^[HBM]\d+  ", t.content)
                          and not d.callout_matches_grammar(t.content)]
    out["arrows"] = sum(svg.count(symbol) for symbol in d.FORBIDDEN_SYMBOLS)
    out["scale_off_series"] = record["scale"] not in d.STANDARD_SCALES

    dimensions = set()
    cells = set()
    for text in texts:
        if not text.is_dimension:
            continue
        if re.match(r"^-?\d+(\.\d+)?$", text.content):
            dimensions.add(text.content)
        cells.update(re.findall(r"\(([^,]+), ([^)]+)\)", text.content))
    mismatch = []
    for group in record["groups"]:
        mx, my = record["layout"]["origin_mm"][group["view"]]
        for cx, cy in group["positions"]:
            pair = (d._dimnum(round(cx, 3) - mx), d._dimnum(round(cy, 3) - my))
            if pair not in cells or pair[0] not in dimensions or pair[1] not in dimensions:
                mismatch.append((group["id"], pair))
    out["frame_mismatches"] = mismatch

    tolerance = record.get("tolerance") or {}
    allowed_values = set()
    for key in ("linear", "angular", "broken_edges"):
        allowed_values |= {d._dimnum(float(row[2])) for row in (tolerance.get(key) or ())}
    orphans = []
    for text in texts:
        for token in re.findall("±\\s*(\\d+(?:\\.\\d+)?)", text.content):
            if not tolerance.get("specified") or token not in allowed_values:
                orphans.append(text.content)
    out["tolerance_values_without_provenance"] = orphans
    out["banner"] = d.BANNER_TEXT in svg
    fields = record["layout"]["title_fields"]
    out["empty_title_fields"] = [k for k, v in fields if not str(v).strip()]
    out["missing_title_fields"] = [k for k in record["layout"]["required_title_fields"]
                                   if k not in {f[0] for f in fields}]
    out["marking_rows"] = sum(1 for t in texts
                              if re.match(r"^M\d+  ", t.content) and t.mono)
    out["coaxial_duplicates"] = coaxial_duplicates(record)
    return out


def test_text_metrics_match_the_gate_that_reads_the_sheet_back() -> None:
    """Laying out with a different width estimate than the collision gate uses is a bug."""
    from phimech.check import drawing_checks as dc

    for sample in ("H1  4x ⌀2.5 ⤓4", "POSITION FROM ORIGIN  X, Y (mm)", "61.3",
                   "(5.7, 82.7)  (61.3, 82.7)", "INSPECTION DRAWING — NOT FOR MANUFACTURE"):
        for mono in (False, True):
            assert abs(d.text_width_mm(sample, 2.6, mono)
                       - dc.text_width_mm(sample, 2.6, mono)) < 1e-9


def _group(kind: str, hole: bool, size, depth_from: float, depth_to: float, thru: bool,
           count: int = 1, corner_r: float = 0.0) -> "d.Group":
    members = []
    for i in range(count):
        loop = _circle(10 + 12 * i, 10, size[0]) if kind == "circle" else \
            _rect(10 + 12 * i, 10, 10 + 12 * i + size[0], 10 + size[1])
        shape = d.classify_loop(loop, is_hole=hole)
        members.append(d.Feature(shape, "top", depth_from, depth_to, thru))
    return d.Group("H1" if hole else "B1", kind, hole, thru, tuple(size), corner_r,
                   depth_to - depth_from, members, depth_from, depth_to)


def test_every_callout_matches_the_documented_grammar_and_uses_iso_symbols() -> None:
    thru = _group("circle", True, (5.2, 5.2), 0.0, 8.0, True, count=3)
    blind = _group("circle", True, (2.5, 2.5), 2.0, 6.0, False, count=4)
    boss = _group("circle", False, (7.0, 7.0), 3.0, 6.0, False, count=4)
    pad = _group("rect", False, (8.0, 40.0), 8.0, 45.0, False)
    for group, expected in ((thru, "H1  3X ⌀5.2 THRU"),
                            (blind, "H1  4X ⌀2.5 ⤓4"),
                            (boss, "B1  4X BOSS ⌀7, 3 HIGH"),
                            (pad, "B1  PAD 8 x 40, 37 HIGH")):
        text = d._group_callout(group)
        assert text == expected, text
        assert d.callout_matches_grammar(text)
        assert not any(symbol in text for symbol in d.FORBIDDEN_SYMBOLS)
    # W9: an arrow is not a drafting symbol, and prose is not a callout
    assert not d.callout_matches_grammar("B1  4X BOSS ⌀7 ↑3")
    assert not d.callout_matches_grammar("H1  4X 2.5 dia 4 deep")


def test_counterbore_is_printed_only_when_its_numbers_were_measured() -> None:
    """``⌀5.2 THRU ⌴⌀9.5 ⤓4`` is expressible - but only from the result.

    A number the machine-readable result does not carry traces back to nothing, and a sheet
    that prints one is indistinguishable from a sheet that typed it."""
    group = _group("circle", True, (5.2, 5.2), 0.0, 8.0, True)
    group.members[0].counterbore = {"diameter": 9.5, "depth": 4.0}
    rich = d._group_callout(group, [5.2, 9.5, 4.0, 8.0])
    assert rich == "H1  ⌀5.2 THRU ⌴⌀9.5 ⤓4"
    assert d.callout_matches_grammar(rich)
    thin = d._group_callout(group, [5.2, 8.0])          # the pool has no 9.5 and no 4.0
    assert thin == "H1  ⌀5.2 THRU"


def _glyph(cx: float, cy: float, w: float, h: float, kind: str) -> "d.Feature":
    loop = _rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) if kind == "rect" else \
        [(cx - w / 2, cy - h / 2), (cx, cy - h / 2 + 0.3), (cx + w / 2, cy - h / 2),
         (cx + w / 2, cy + h / 2), (cx, cy + h / 2 - 0.3), (cx - w / 2, cy + h / 2)]
    return d.Feature(d.classify_loop(loop, is_hole=True), "top", 1.09, 1.6, False)


def test_cluster_markings_folds_shallow_mixed_loops_into_one_row() -> None:
    """W7: six 0.5 mm glyph pockets are a marking, not six openings with six positions."""
    glyphs = [_glyph(20.1, 34, 4.9, 4.8, "profile"), _glyph(24.0, 34, 1.0, 4.8, "rect"),
              _glyph(27.5, 34, 4.1, 4.8, "profile"), _glyph(30.9, 34, 1.0, 4.8, "rect"),
              _glyph(33.9, 34, 3.7, 4.8, "profile"), _glyph(38.5, 34, 4.5, 4.8, "profile")]
    hole = d.Feature(d.classify_loop(_circle(3.5, 3.5, 2.5), True), "top", 1.6, 11.6, False)
    kept, markings = d.cluster_markings(glyphs + [hole], "top")
    assert [f.shape.size[0] for f in kept] == [2.5]
    assert len(markings) == 1 and markings[0].loops == 6
    row = markings[0].row()
    assert row == ("M1  SHALLOW RELIEF REGION, 6 LOOPS, 0.51 DEEP, GEOMETRY PER MODEL "
                   "— NOT A HOLE").replace("M1", markings[0].ident or "")
    assert d.MARKING_GRAMMAR.match(d.replace(markings[0], ident="M1").row())


def test_cluster_markings_leaves_an_identical_shallow_pattern_alone() -> None:
    """Five identical 0.5 mm pockets are a feature pattern somebody has to make."""
    pockets = [_glyph(10 + 6 * i, 34, 3.0, 3.0, "rect") for i in range(5)]
    kept, markings = d.cluster_markings(pockets, "top")
    assert markings == [] and len(kept) == 5


def test_marking_text_comes_from_the_source_or_is_not_printed() -> None:
    mark = d.Marking("M1", "top", 6, 0.51, 1.09, 1.6, (17.0, 31.0, 41.0, 37.0))
    declared = d.apply_marking_text([mark], {"label": "MINI4K", "size": 5.0, "depth": 0.5,
                                             "emboss": False})
    assert declared[0].row() == "M1  MARKING 'MINI4K' 5 TEXT DEBOSSED 0.5"
    assert d.MARKING_GRAMMAR.match(declared[0].row())
    # no metadata, the wrong depth, or two regions and one label: the generic row stands
    assert d.apply_marking_text([mark], None)[0].row() == mark.row()
    assert d.apply_marking_text([mark], {"label": "X", "size": 5.0, "depth": 3.0})[0].row() \
        == mark.row()
    pair = [mark, d.replace(mark, ident="M2")]
    assert [m.row() for m in d.apply_marking_text(pair, {"label": "X", "size": 5.0,
                                                         "depth": 0.5})] == \
        [m.row() for m in pair]


def test_ordinates_locate_every_member_not_the_first_and_one_twin() -> None:
    """W5: the bracket's hole at (45, 20) had no x dimension anywhere on the sheet."""
    holes = _group("circle", True, (5.2, 5.2), 0.0, 8.0, True, count=3)
    for member, centre in zip(holes.members, ((20.0, 10.0), (45.0, 20.0), (20.0, 30.0))):
        member.shape = d.replace(member.shape, center=centre)
    view = d.ViewData("top", [_rect(0, 0, 60, 40)], [], list(holes.members), (0, 0, 60, 40),
                      [0.0, 8.0])
    frame = d._ViewFrame(view, 1.0, 0.0, 0.0)
    xs, ys = d.ordinate_values(frame, [holes])
    assert xs == [20.0, 45.0] and ys == [10.0, 20.0, 30.0]


def test_resolve_tolerance_block_never_invents_one() -> None:
    """PhiMech cannot choose a tolerance; without a declaration the answer is UNSPECIFIED."""
    block = d.resolve_tolerance_block(ir=None, source_manifest=None)
    assert block["specified"] is False and block["provenance"] == "none"
    assert block["standard"] is None and not block["linear"]
    lines = [text for text, _ in d.tolerance_lines(block)]
    assert lines[0] == "GENERAL TOLERANCES: UNSPECIFIED"
    assert "SURFACE FINISH: UNSPECIFIED" in lines


def test_tolerance_class_is_printed_and_expanded() -> None:
    """A class nobody can read is a tolerance nobody applies, so the ranges are printed too."""
    block = {"specified": True, "provenance": "ir", "standard": "ISO 2768-m",
             "linear": ((0.5, 3.0, 0.1), (3.0, 6.0, 0.1), (6.0, 30.0, 0.2)),
             "angular": ((0.0, 10.0, 1.0),), "finish": "Ra 3.2"}
    lines = [text for text, _ in d.tolerance_lines(block)]
    assert lines[0] == "GENERAL TOLERANCES  ISO 2768-m"
    assert any("over 0.5 up to 3" in line and "±0.1" in line for line in lines)
    assert any("±1°" in line for line in lines)
    assert "SURFACE FINISH: Ra 3.2" in lines
    assert any(line.startswith("Source: ir") for line in lines)


def test_banner_stands_until_material_finish_and_tolerance_are_all_declared() -> None:
    views = _plate_views()
    declared = {"specified": True, "provenance": "manifest", "standard": "ISO 2768-m",
                "linear": ((0.5, 3.0, 0.1),), "angular": (), "finish": "Ra 3.2"}
    manifest = {"component": {"material": "PETG", "approved_by": "R. Mehta"}}
    svg, _, layout = d.compose_sheet(views, [], title="plate", source_label="plate.stl",
                                     volume=1.0, openscad_version="test", part="PLATE",
                                     manifest=manifest, tolerance=declared,
                                     source_sha256="a" * 64)
    assert layout["banner"] is None and d.BANNER_TEXT not in svg
    fields = dict(layout["title_fields"])
    assert fields["MATERIAL"] == "PETG" and fields["SURFACE FINISH"] == "Ra 3.2"
    assert all(str(v).strip() for v in fields.values())
    assert set(d.REQUIRED_TITLE_FIELDS) <= set(fields)
    # take any one of the four declarations away and the banner comes back
    for drop in ("material", "tolerance", "finish", "approval"):
        thin_manifest = {"component": dict(manifest["component"])}
        thin = dict(declared)
        if drop == "material":
            thin_manifest["component"].pop("material")
        elif drop == "approval":
            thin_manifest["component"].pop("approved_by")
        elif drop == "finish":
            thin["finish"] = None
        else:
            thin = {"specified": False, "provenance": "none", "standard": None,
                    "linear": (), "angular": (), "finish": "Ra 3.2"}
        _, _, partial = d.compose_sheet(views, [], title="plate", source_label="plate.stl",
                                        volume=1.0, openscad_version="test", part="PLATE",
                                        manifest=thin_manifest, tolerance=thin)
        assert partial["banner"] == d.BANNER_TEXT, drop

    svg, _, layout = d.compose_sheet(views, [], title="plate", source_label="plate.stl",
                                     volume=1.0, openscad_version="test", part="PLATE")
    assert layout["banner"] == d.BANNER_TEXT and d.BANNER_TEXT in svg
    fields = dict(layout["title_fields"])
    assert fields["MATERIAL"] == fields["SURFACE FINISH"] == d.UNSPECIFIED
    assert all(str(v).strip() for v in fields.values())


def _plate_views() -> dict:
    return {"top": d.ViewData("top", [_rect(0, 0, 60, 40)], [], [], (0, 0, 60, 40), [0.0, 9.0]),
            "front": d.ViewData("front", [_rect(0, 0, 60, 9)], [], [], (0, 0, 60, 9), []),
            "right": d.ViewData("right", [_rect(0, 0, 40, 9)], [], [], (0, 0, 40, 9), [])}


def test_approved_by_is_empty_until_a_human_declares_one() -> None:
    """An "APPROVED BY" carrying a name attests that a person signed the drawing off."""
    svg, _, layout = d.compose_sheet(_plate_views(), [], title="plate",
                                     source_label="plate.stl", volume=1.0,
                                     openscad_version="test", part="PLATE")
    fields = dict(layout["title_fields"])
    assert fields["APPROVED BY"] == d.UNSIGNED and d.UNSIGNED.strip()
    assert fields["DRAWN BY"].startswith("PHIMECH")
    assert layout["approval"] == {"approved_by": None, "provenance": None,
                                  "printed": d.UNSIGNED}
    assert d.BANNER_TEXT in svg

    signed = {"component": {"material": "PETG", "approved_by": "R. Mehta"}}
    svg, _, layout = d.compose_sheet(_plate_views(), [], title="plate",
                                     source_label="plate.stl", volume=1.0,
                                     openscad_version="test", part="PLATE", manifest=signed,
                                     tolerance={"specified": True, "provenance": "manifest",
                                                "standard": "ISO 2768-m",
                                                "linear": ((0.5, 3.0, 0.1),), "angular": (),
                                                "finish": "Ra 3.2"})
    fields = dict(layout["title_fields"])
    assert fields["APPROVED BY"] == "R. Mehta"
    assert layout["approval"]["provenance"] == "component.approved_by"
    assert d.BANNER_TEXT not in svg          # signed, specified, and only then unbannered


def test_an_approver_never_comes_from_the_machine() -> None:
    """No default, no env var, no git identity, no OS user: those would forge a signature."""
    assert d.manifest_facts(None)["approved_by"] is None
    assert d.manifest_facts({"component": {"material": "PETG"}})["approved_by"] is None
    source = Path(d.__file__).read_text(encoding="utf-8")
    for forbidden in ("getuser", "getpass", "os.environ", "os.getenv", "LOGNAME", "USERNAME",
                      "pwd.getpw", "git config", "gitconfig"):
        assert forbidden not in source, forbidden


def test_an_approver_alone_does_not_lift_the_banner() -> None:
    signed = {"component": {"approved_by": "R. Mehta"}}
    svg, _, layout = d.compose_sheet(_plate_views(), [], title="plate",
                                     source_label="plate.stl", volume=1.0,
                                     openscad_version="test", part="PLATE", manifest=signed)
    assert dict(layout["title_fields"])["APPROVED BY"] == "R. Mehta"
    assert layout["banner"] == d.BANNER_TEXT and d.BANNER_TEXT in svg


def test_attribution_footer_is_always_printed_and_names_the_source() -> None:
    line = d.attribution_footer("PHIMECH 0.1.0", "2026-09-24", "tray.stl", "ab" * 32)
    assert line.startswith("GENERATED BY PHIMECH 0.1.0 ON 2026-09-24")
    assert "tray.stl" in line and "sha256 abababababababab" in line
    assert d.attribution_footer("PHIMECH 0.1.0", "2026-09-24", "tray.stl", None).endswith(
        d.UNSPECIFIED)
    svg, _, _ = d.compose_sheet(_plate_views(), [], title="plate", source_label="/tmp/plate.stl",
                                volume=1.0, openscad_version="test", part="PLATE",
                                source_sha256="cd" * 32)
    assert "GENERATED BY PHIMECH" in svg and "sha256 cdcdcdcdcdcdcdcd" in svg


def _resolver() -> None:
    """Skip cleanly when the tolerance resolver is not installed; it is a separate module."""
    import importlib

    try:
        importlib.import_module("phimech.ir.tolerance")
    except ImportError:
        _skip("phimech.ir.tolerance is not installed")


def test_nothing_declared_and_a_rejected_declaration_are_different_sheets() -> None:
    """A typo'd tolerance must not be indistinguishable from no tolerance at all.

    Both are safe - nothing is invented either way - but only one of them is honest: an author
    who declared something and had it discarded will otherwise assume it is on the drawing."""
    _resolver()
    silent = d.resolve_tolerance_block()
    assert silent["specified"] is False and silent["rejected"] is None
    assert [t for t, _ in d.tolerance_lines(silent)][0] == "GENERAL TOLERANCES: UNSPECIFIED"
    assert d.tolerance_notes(silent) == []

    broken = d.resolve_tolerance_block(
        ir={"manufacturing": {"general_tolerance": {"linear": "nope"}}})
    assert broken["specified"] is False
    assert broken["rejected"] and "linear" in broken["rejected"]
    lines = [t for t, _ in d.tolerance_lines(broken)]
    assert lines[0] == d.REJECTED_PREFIX
    assert any(line.startswith("Reason:") for line in lines)
    assert any("DECLARED AND REJECTED" in note for note in d.tolerance_notes(broken))


def test_an_unbounded_band_reads_as_drafting_and_never_as_inf() -> None:
    _resolver()
    blanket = d.resolve_tolerance_block(
        ir={"manufacturing": {"general_tolerance": {"plus_minus": 0.2}}})
    assert blanket["specified"] and blanket["linear"][0][1] is None
    rendered = [t for t, _ in d.tolerance_lines(blanket)]
    assert any(line.strip().startswith("over 0") and "±0.2" in line
               for line in rendered)
    assert not any("inf" in line.lower() for line in rendered)
    import json

    json.dumps(blanket)      # None, not Infinity: the sidecar has to be valid JSON


def test_an_unrecognised_standard_is_carried_verbatim_with_no_table() -> None:
    """Falling back to a class we happen to know would put somebody else's numbers on it."""
    _resolver()
    block = d.resolve_tolerance_block(
        ir={"manufacturing": {"general_tolerance": "ASME Y14.5"}})
    assert block["standard"] == "ASME Y14.5"
    assert block["linear"] == () and block["angular"] == () and block["broken_edges"] == ()
    lines = [t for t, _ in d.tolerance_lines(block)]
    assert lines[0] == "GENERAL TOLERANCES  ASME Y14.5"
    assert any("not expanded" in line or "does not hold the table" in line for line in lines)
    assert not any("±" in line for line in lines)


def test_a_chamfer_is_toleranced_by_the_broken_edge_table_not_the_linear_one() -> None:
    """A 1 mm chamfer under ISO 2768-m is +-0.2 (table 2), not the +-0.1 table 1 would give."""
    _resolver()
    block = d.resolve_tolerance_block(ir={"manufacturing": {"general_tolerance": "ISO 2768-m"}})
    assert d.edge_break_tolerance(block, 1.0) == 0.2
    assert d.edge_break_tolerance(block, 4.0) == 0.5
    assert abs(d.edge_break_tolerance(block, 1.0) - block["linear"][0][2]) > 1e-9
    lines = [t for t, _ in d.tolerance_lines(block, edges=[1.0])]
    assert any("BROKEN EDGES" in line for line in lines)
    assert any("1 mm chamfer/radius here: ±0.2" in line for line in lines)
    assert d.edge_break_tolerance({"broken_edges": ()}, 1.0) is None


def test_a_conflict_between_two_declarations_is_printed_not_resolved_away() -> None:
    _resolver()
    block = d.resolve_tolerance_block(
        ir={"manufacturing": {"general_tolerance": "ISO 2768-m"}},
        source_manifest={"component": {"general_tolerance": "ISO 2768-c"}})
    assert block["conflicts"]
    assert any(note.startswith("GENERAL TOLERANCE CONFLICT")
               for note in d.tolerance_notes(block))


def test_a_declared_sheet_prints_the_class_its_table_and_no_banner() -> None:
    _resolver()
    _require_openscad()
    manifest = {"rev": "B", "title": "plate",
                "manufacturing": {"general_tolerance": "ISO 2768-m", "finish": "Ra 3.2"},
                "component": {"material": "PETG", "approved_by": "R. Mehta"}}
    with tempfile.TemporaryDirectory() as tmp:
        scad = Path(tmp) / "part.scad"
        scad.write_text(_PART, encoding="utf-8")
        result = d.render_drawing(scad, Path(tmp) / "part.svg", png=False, manifest=manifest)
        record = result.to_dict()
        svg = Path(result.path).read_text(encoding="utf-8")
        assert record["tolerance"]["provenance"] == "ir"
        assert "GENERAL TOLERANCES  ISO 2768-m" in svg
        assert "over 6 up to 30" in svg and "BROKEN EDGES" in svg
        assert d.BANNER_TEXT not in svg
        fields = dict(record["layout"]["title_fields"])
        assert fields["MATERIAL"] == "PETG" and fields["APPROVED BY"] == "R. Mehta"
        assert record["approval_provenance"] == "component.approved_by"

        found = audit_sheet(record, svg)
        assert found["tolerance_values_without_provenance"] == []
        assert found["min_text_separation_mm"] >= d.MIN_TEXT_GAP_MM
        assert found["empty_title_fields"] == []

        from phimech.check import drawing_checks as dc

        checks = dc.run_checks(result, svg_path=result.path)
        assert checks["summary"]["status"] == "pass", checks["summary"]


def _hole_group(dia: float, z0: float, z1: float, thru: bool, centres) -> "d.Group":
    members = []
    for cx, cy in centres:
        shape = d.classify_loop(_circle(cx, cy, dia), is_hole=True)
        members.append(d.Feature(shape, "top", z0, z1, thru))
    return d.Group("", "circle", True, thru, (dia, dia), 0.0, z1 - z0, members, z0, z1)


def test_a_counterbore_is_folded_into_its_hole_and_never_listed_as_a_hole() -> None:
    """Two rows at the same three positions tell a machinist to find six holes."""
    positions = [(20.0, 10.0), (45.0, 20.0), (20.0, 30.0)]
    bore = _hole_group(5.2, 0.0, 4.0, True, positions)
    for member in bore.members:
        member.counterbore = {"size": [9.5, 9.5], "depth": 4.0, "from": 4.0, "to": 8.0,
                              "diameter": 9.5}
    cbore = _hole_group(9.5, 4.0, 8.0, False, positions)
    groups = [cbore, bore]
    folded = d.fold_modifiers(groups)
    assert [(m is cbore, p is bore) for m, p in folded] == [(True, True)]
    # the hole is one feature: its extent is the bore plus the counterbore, and the callout
    # states the counterbore as a modifier of the bore
    assert (bore.depth_from, bore.depth_to, bore.depth) == (0.0, 8.0, 8.0)
    bore.group_id = "H1"
    assert d._group_callout(bore, [5.2, 9.5, 4.0]) == "H1  3X ⌀5.2 THRU ⌴⌀9.5 ⤓4"
    assert d._depth_text(bore) == "THRU (Z 0 TO 8)"
    assert d.feature_noun(bore) == "HOLE, COUNTERBORED"


def test_a_counterbore_on_a_blind_hole_gives_the_whole_hole_its_depth() -> None:
    """ASME reads the small hole's depth from the face it enters, not from the c'bore floor."""
    bore = _hole_group(5.2, 2.0, 4.0, False, [(10.0, 10.0)])
    bore.members[0].counterbore = {"size": [9.5, 9.5], "depth": 4.0, "from": 4.0, "to": 8.0,
                                   "diameter": 9.5}
    cbore = _hole_group(9.5, 4.0, 8.0, False, [(10.0, 10.0)])
    assert [m is cbore for m, _ in d.fold_modifiers([cbore, bore])] == [True]
    assert (bore.depth_from, bore.depth_to, bore.depth) == (2.0, 8.0, 6.0)
    bore.group_id = "H1"
    assert d._group_callout(bore, [5.2, 9.5, 4.0, 6.0]) == \
        "H1  ⌀5.2 ⤓6 ⌴⌀9.5 ⤓4"


def test_a_countersink_cone_folds_but_a_hole_with_its_own_wall_never_does() -> None:
    """The cone has no wall, so its "diameter" is wherever the cut landed; a hole has one."""
    bore = _hole_group(4.0, 0.0, 7.0, True, [(20.0, 20.0)])
    for member in bore.members:
        member.measured = True
        member.skirts = (d.Skirt(7.0, 2.0, 2.0),)
    cone = _hole_group(6.0, 0.0, 10.0, False, [(20.0, 20.0)])   # measured stays False
    assert [(m is cone, p is bore) for m, p in d.fold_modifiers([cone, bore])] == [(True, True)]
    assert (bore.depth_from, bore.depth_to) == (0.0, 9.0)
    assert d.feature_noun(bore) == "HOLE, COUNTERSUNK"

    # a real coaxial hole - one with a wall of its own - is a feature and stays
    bore2 = _hole_group(4.0, 0.0, 7.0, True, [(20.0, 20.0)])
    for member in bore2.members:
        member.measured = True
        member.skirts = (d.Skirt(7.0, 2.0, 2.0),)
    real = _hole_group(6.0, 0.0, 10.0, False, [(20.0, 20.0)])
    for member in real.members:
        member.measured = True
    assert d.fold_modifiers([real, bore2]) == []


def test_a_hole_inside_a_boss_is_two_features_and_both_are_printed() -> None:
    """The fold must not swallow a bore coaxial with the boss it is drilled into."""
    bore = _hole_group(2.5, 2.0, 6.0, False, [(3.5, 3.5)])
    boss = d.Group("", "circle", False, False, (7.0, 7.0), 0.0, 3.0,
                   [d.Feature(d.classify_loop(_circle(3.5, 3.5, 7.0), is_hole=False),
                              "top", 3.0, 6.0, False)], 3.0, 6.0)
    assert d.fold_modifiers([boss, bore]) == []


def test_counterbored_hole_prints_one_row_one_callout_one_position_set() -> None:
    _require_openscad()
    source = Path("/tmp/bracket-fixed.step")
    if not source.is_file():
        _skip("no counterbored bracket fixture at {}".format(source))
    with tempfile.TemporaryDirectory() as tmp:
        result = d.render_drawing(source, Path(tmp) / "cb.svg", png=False)
        record = result.to_dict()
        svg = Path(result.path).read_text(encoding="utf-8")
        ids = [g["id"] for g in record["groups"]]
        assert len(record["groups"]) == 3 and len(ids) == len(set(ids))
        assert not any("role" in g or "of" in g for g in record["groups"])
        assert coaxial_duplicates(record) == []
        assert len(record["layout"]["callouts"]["placed"]) == 3
        assert "H1  3X ⌀5.2 THRU ⌴⌀9.5 ⤓4" in svg
        assert "THRU (Z 0 TO 8)" in svg and "HOLE, COUNTERBORED" in svg
        # the counterbore appears only as a modifier of H1, never as a hole of its own
        assert svg.count("\u23009.5") == svg.count("\u2334\u23009.5")
        from phimech.check import drawing_checks as dc

        checks = dc.run_checks(result, svg_path=result.path)
        assert checks["summary"]["status"] == "pass", checks["summary"]


def test_render_drawing_sheet_is_legible_and_self_consistent() -> None:
    """The presentation gates, on a real sheet: leaders, frame, symbology, title block."""
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        scad = Path(tmp) / "stepped.scad"
        scad.write_text(_STEPPED_PART, encoding="utf-8")
        result = d.render_drawing(scad, Path(tmp) / "stepped.svg", png=False)
        record = result.to_dict()
        svg = Path(result.path).read_text(encoding="utf-8")
        found = audit_sheet(record, svg)

        assert found["witness_lines_over_the_part"] == 0
        assert found["leader_leader_crossings"] == 0
        assert found["leader_witness_crossings"] == 0
        assert found["leaders_off_their_feature"] == []
        assert found["leader_geometry_crossings_unpermitted"] == []
        assert found["centre_marks"] == found["circular_members"]
        assert found["min_text_separation_mm"] >= d.MIN_TEXT_GAP_MM
        assert found["off_grammar"] == [] and found["arrows"] == 0
        assert not found["scale_off_series"]
        assert found["frame_mismatches"] == []
        assert found["tolerance_values_without_provenance"] == []
        assert found["banner"] and found["empty_title_fields"] == []
        assert found["missing_title_fields"] == []
        assert dict(record["layout"]["title_fields"])["APPROVED BY"] == d.UNSIGNED
        assert "GENERATED BY PHIMECH" in svg
        assert found["coaxial_duplicates"] == []
        assert record["layout"]["callouts"]["in_table_only"] == []

        from phimech.check import drawing_checks as dc

        checks = dc.run_checks(result, svg_path=result.path)
        statuses = {c["name"]: c["status"] for c in checks["checks"]}
        assert statuses["feature_coverage"] == "pass", checks["checks"]
        assert checks["summary"]["status"] == "pass", checks["summary"]

# --------------------------------------------------------------------------------------
# dense views: ordinate staggering and margin leaders
# --------------------------------------------------------------------------------------


def test_spread_positions_keeps_order_gap_and_bounds() -> None:
    values = [10.0, 10.3, 10.6, 11.0, 40.0]
    slots = d.spread_positions(values, 2.35, 0.0, 50.0)
    assert slots is not None and len(slots) == len(values)
    assert all(b - a >= 2.35 - 1e-9 for a, b in zip(slots, slots[1:]))
    assert all(0.0 <= v <= 50.0 for v in slots)
    assert slots[-1] == 40.0                     # a value with room keeps its own position
    assert d.spread_positions(values, 20.0, 0.0, 50.0) is None
    clamped = d.spread_positions([0.0, 0.1], 3.0, 1.0, 10.0)
    assert clamped is not None and clamped[0] >= 1.0


def test_plan_ordinates_staggers_close_values_instead_of_striking_them() -> None:
    """178 and 177.4 at 1:2 are 0.3 mm apart: one label each, never on the other's line."""
    values = [88.7, 89.0, 95.0, 120.0]
    plan, refused = d.plan_ordinates(values, 50.0, 160.0)
    assert refused == []
    slots = [slot for _, slot, _ in plan]
    assert all(b - a >= d._ORD_PITCH / 2.0 - 1e-6 for a, b in zip(slots, slots[1:]))
    for column in (0, 1):
        same = [slot for _, slot, c in plan if c == column]
        assert all(b - a >= d._ORD_PITCH - 1e-6 for a, b in zip(same, same[1:]))
    # sparse values are not moved at all: no jog where none is needed
    sparse, _ = d.plan_ordinates([60.0, 90.0, 130.0], 50.0, 160.0)
    assert [(v, s) for v, s, _ in sparse] == [(60.0, 60.0), (90.0, 90.0), (130.0, 130.0)]
    # more values than the edge can carry: the most crowded go to the table, none is lost
    many = [50.0 + 0.5 * i for i in range(200)]
    kept, refused = d.plan_ordinates(many, 50.0, 100.0)
    assert len(kept) + len(refused) == 200 and refused


def _dense_view_sheet():
    """A 66 x 110 mm view packed with 30 internal features, on a bare A3 sheet."""
    import types

    svg = d._Svg(420.0, 297.0)
    ox, oy, w, h = 50.0, 165.0, 66.0, 110.0
    fr = types.SimpleNamespace(ox=ox, oy=oy, top=oy - h, right=ox + w,
                               paper_box=(ox, oy - h, ox + w, oy))
    outline = [(ox, oy), (ox + w, oy), (ox + w, oy - h), (ox, oy - h)]
    svg.poly(outline, width=0.5, cls="outline")
    callouts, loops = [], []
    for i in range(30):
        cx = ox + 8.0 + (i % 5) * 12.5
        cy = oy - 8.0 - (i // 5) * 18.0
        loop = [(cx - 3, cy - 2), (cx + 3, cy - 2), (cx + 3, cy + 2), (cx - 3, cy + 2)]
        svg.poly(loop, width=0.2)
        ident = "B{}".format(i + 1)
        loops.append((ident, loop))
        callouts.append(d._Callout(ident, "{}  PAD {} x 4, {} HIGH".format(ident, 6 + i, i + 1),
                                   ident, [d._Target((cx, cy), 3.6, loop, d._bbox_of(loop))],
                                   (2, 24.0)))
    return svg, fr, callouts, [outline], loops


def test_dense_view_leaders_go_to_the_margin_without_overlaps() -> None:
    from phimech.check import drawing_checks as dc

    svg, fr, callouts, drawn, loops = _dense_view_sheet()
    placed, dropped, over = d._place_callouts(svg, fr, callouts, drawn, loops)
    assert over == []
    assert sorted(placed + dropped) == sorted(c.ident for c in callouts)   # none vanishes
    assert len(placed) >= 25
    text = svg.render()
    texts = dc.parse_texts(text)
    lines = dc.parse_lines(text)
    outlines = dc.parse_outlines(text)
    assert outlines
    over_view = [t.content for t in texts
                 if any(dc.box_hits_loop(dc._shrink(t.bbox, dc.TEXT_OVERLAP_MM), lp)
                        for lp in outlines)]
    assert over_view == []
    struck = [(t.content, kind) for t in texts for kind, a, b in lines
              if dc.segment_hits_box(a, b, dc._shrink(t.bbox, dc.TEXT_OVERLAP_MM))]
    assert struck == []
    assert not any(dc._overlap(a.bbox, b.bbox, dc.TEXT_OVERLAP_MM)
                   for i, a in enumerate(texts) for b in texts[i + 1:])
    leaders = [((float(m["x1"]), float(m["y1"])), (float(m["x2"]), float(m["y2"])))
               for m in (dict(_ATTR_RE.findall(g.group(1))) for g in _LINE_RE.finditer(text))
               if m.get("class") == "leader"]
    assert not any(d.segments_cross(a[0], a[1], b[0], b[1])
                   for i, a in enumerate(leaders) for b in leaders[i + 1:])


#: A 130 x 210 x 40 tray with synthetic mounting features: internal pads (ledges)
#: of a dozen sizes, bosses, wall slots.  At 1:2 the old placer printed 21 callouts over the
#: view and let 15 lines strike labels.
_DENSE_TRAY = """
$fn = 32;
difference() {
  union() {
    difference() { cube([130, 210, 40]); translate([2, 2, 2]) cube([126, 206, 40]); }
    for (i = [0:5]) translate([4 + i * 3, 10 + i * 30, 2]) cube([20 + i * 2.5, 8 + i, 3 + i]);
    for (i = [0:5]) translate([80 - i * 2, 14 + i * 31, 2])
      cube([12 + i * 3, 5 + i * 0.5, 4 + i * 1.5]);
    for (p = [[8, 8], [122, 8], [8, 202], [122, 202]])
      translate([p[0], p[1], 2]) cylinder(d = 9, h = 30);
    for (p = [[40, 100], [90, 100]]) translate([p[0], p[1], 2]) cylinder(d = 6, h = 8);
    for (p = [[30, 150], [100, 150], [30, 60], [100, 60]])
      translate([p[0], p[1], 2]) cylinder(d = 7.5, h = 12);
  }
  for (p = [[8, 8], [122, 8], [8, 202], [122, 202]])
    translate([p[0], p[1], 6]) cylinder(d = 3.3, h = 40);
  for (p = [[40, 100], [90, 100]]) translate([p[0], p[1], 4]) cylinder(d = 2.5, h = 20);
  for (i = [0:7]) translate([-1, 60 + i * 4, 15]) cube([4, 1.5, 12]);
  for (p = [[20, 30], [110, 30], [20, 180], [110, 180]])
    translate([p[0], p[1], -1]) cube([4.5, 12, 5]);
  translate([50, 190, -1]) cube([30, 8, 5]);
}
"""


def test_dense_tray_sheet_has_zero_label_overlaps() -> None:
    _require_openscad()
    from phimech.check import drawing_checks as dc

    with tempfile.TemporaryDirectory() as tmp:
        scad = Path(tmp) / "tray.scad"
        scad.write_text(_DENSE_TRAY, encoding="utf-8")
        result = d.render_drawing(scad, Path(tmp) / "tray.svg", png=False, pdf=False)
        record = result.to_dict()
        svg = Path(result.path).read_text(encoding="utf-8")
        checks = {c["name"]: c for c in dc.run_checks(result, svg_path=result.path)["checks"]}
        for name in ("text_no_collision", "label_view_overlap", "label_line_overlap",
                     "feature_coverage"):
            assert checks[name]["status"] == "pass", (name, checks[name]["evidence"])
            assert checks[name]["value"] == 0
        callouts = record["layout"]["callouts"]
        assert callouts["printed_over_the_view"] == []
        ids = {g["id"] for g in record["groups"]}
        assert set(callouts["placed"]) | set(callouts["in_table_only"]) == ids
        notes = [t.content for t in dc.parse_texts(svg) if t.content.startswith("Leader(s)")]
        for ident in callouts["in_table_only"]:      # never silently dropped: named in a note
            assert any(re.search(r"\b{}\b".format(ident), note) for note in notes), notes
        found = audit_sheet(record, svg)
        assert found["leader_leader_crossings"] == 0
        assert found["leader_witness_crossings"] == 0
        assert found["witness_lines_over_the_part"] == 0
        assert found["min_text_separation_mm"] >= d.MIN_TEXT_GAP_MM


if __name__ == "__main__":  # direct runner for the system python
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except _Skipped as exc:
                print("skip ", name, exc)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, exc)
    raise SystemExit(1 if failures else 0)
