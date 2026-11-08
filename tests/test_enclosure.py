"""Tests for `phimech.synth.enclosure` and `phimech.synth.features`.

Everything here is pure AST: an `Envelope` is built by hand, `Placed` directly, and the
`Placement` is a stub, so the suite runs in milliseconds with no CAD binary and no other synth
slice present. The one test that needs a kernel (`test_openscad_render_is_solid`) skips itself.

Runnable two ways so the CI gate works with or without pytest::

    /usr/bin/python3 tests/test_enclosure.py
    python3 -m pytest tests/test_enclosure.py
"""

from __future__ import annotations

import os as _os

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from phimech.components.envelope import Access, Envelope, Hole, Placed  # noqa: E402
from phimech.csg.ast import NODE_TYPES, Node, walk  # noqa: E402
from phimech.synth import enclosure as E  # noqa: E402
from phimech.synth import features as F  # noqa: E402

WALL = 2.0
CLEARANCE = 1.5
COMP_CLEARANCE = 1.0

# --------------------------------------------------------------------------------------
# fixtures, hand-built
# --------------------------------------------------------------------------------------

#: A 50x30x10 board on 4 mm standoffs: 4 mounting holes 3 mm in from each corner, a USB
#: socket on its front face, and the board itself 2 mm up from the envelope bottom (there are
#: bottom-side parts underneath). Numbers chosen so every expectation below is exact.
PCB_SIZE = (50.0, 30.0, 10.0)
PCB_HOLES = ((3.0, 3.0), (47.0, 3.0), (47.0, 27.0), (3.0, 27.0))
MOUNT_PLANE_Z = 2.0
STANDOFF_H = 4.0


def make_pcb(ref="PCB1", thermal_w=0.0, access=True):
    acc = (Access("usb", "front", (25.0, 4.0), (9.0, 4.0), "rect", 0.5),) if access else ()
    return Envelope(
        ref=ref,
        size=PCB_SIZE,
        holes=tuple(Hole(at, 3.2, "H%d" % (i + 1)) for i, at in enumerate(PCB_HOLES)),
        access=acc,
        mount_plane_z=MOUNT_PLANE_Z,
        mount_thickness=1.6,
        name="test board",
        source="kicad_pcb",
        thermal_w=thermal_w,
    )


class StubPlacement(object):
    """The slice of `phimech.synth.place.Placement` that `synthesize()` actually touches.

    Written by hand (rather than imported) so this suite does not depend on place.py landing;
    if the real class is importable the last test checks the two agree.
    """

    def __init__(self, placed):
        self.items = list(placed)
        self.by_ref = dict((p.ref, p) for p in self.items)

    def bbox(self):
        mins, maxs = zip(*(p.world_bbox() for p in self.items))
        return (
            tuple(min(m[i] for m in mins) for i in range(3)),
            tuple(max(m[i] for m in maxs) for i in range(3)),
        )


def comp(mount_style="standoff", **kw):
    c = {
        "ref": "PCB1",
        "clearance": COMP_CLEARANCE,
        "mount": {"style": mount_style, "standoff_h": STANDOFF_H, "fastener": "m3", "hole_d": 3.2},
    }
    c.update(kw)
    return c


def build(spec=None, mfg=None, comps=None, placed=None, title="unit-test"):
    """Synthesise with the default board and only the spec keys a test cares about."""
    p = placed if placed is not None else Placed(make_pcb(), (0.0, 0.0, 0.0))
    items = p if isinstance(p, list) else [p]
    s = {"style": "tray_lid", "wall": WALL, "clearance": CLEARANCE}
    s.update(spec or {})
    return E.synthesize(
        StubPlacement(items),
        s,
        mfg or {},
        components_by_ref=comps if comps is not None else {"PCB1": comp()},
        title=title,
    )


def assert_valid_ast(node, label="node"):
    """Every node reachable from `node` is a known AST kind, and children are Nodes."""
    count = 0
    for n in walk(node):
        assert isinstance(n, Node), "%s: %r is not a Node" % (label, n)
        assert n.kind in NODE_TYPES, "%s: unknown kind %r" % (label, n.kind)
        for c in getattr(n, "children", ()) or ():
            assert isinstance(c, Node), "%s: child of %s is %r" % (label, n.kind, c)
        count += 1
    assert count > 1, "%s: degenerate AST (%d nodes)" % (label, count)
    return count


def group_names(node):
    return set(n.name for n in walk(node) if n.kind == "group" and n.name)


# --------------------------------------------------------------------------------------
# features.py — frames
# --------------------------------------------------------------------------------------


def test_face_frame_round_trips_every_face():
    """A face-local (u, v) must map to a point *on* that face, for all six faces."""
    lo, hi = (0.0, 0.0, 0.0), (50.0, 30.0, 10.0)
    for face in F.FACES:
        origin, u, v, n = F.face_frame(face, lo, hi)
        fw, fh = F.face_extent(face, lo, hi)
        axis = [i for i in range(3) if abs(n[i]) > 0.5][0]
        expected = hi[axis] if n[axis] > 0 else lo[axis]
        for (cu, cv) in ((0.0, 0.0), (fw, 0.0), (fw, fh), (fw / 2.0, fh / 2.0)):
            pt = tuple(origin[i] + u[i] * cu + v[i] * cv for i in range(3))
            assert abs(pt[axis] - expected) < 1e-9, "%s: %r off the face plane" % (face, pt)
            for i in range(3):
                assert lo[i] - 1e-9 <= pt[i] <= hi[i] + 1e-9, "%s: %r outside the box" % (face, pt)
        # the frame corners must span the whole face, and the normal must point outward
        assert F.face_of_normal(n) == face
        assert abs(fw * fh - _face_area(face, lo, hi)) < 1e-9


def _face_area(face, lo, hi):
    _u, _v, n = F.face_axes(face)
    size = [hi[i] - lo[i] for i in range(3)]
    axis = [i for i in range(3) if abs(n[i]) > 0.5][0]
    dims = [size[i] for i in range(3) if i != axis]
    return dims[0] * dims[1]


def test_face_frame_origins_match_the_documented_convention():
    """The table in the module docstring is the contract other slices code against."""
    lo, hi = (1.0, 2.0, 3.0), (11.0, 22.0, 33.0)
    expected = {
        "top": (1.0, 2.0, 33.0),
        "bottom": (1.0, 2.0, 3.0),
        "front": (1.0, 2.0, 3.0),
        "back": (11.0, 22.0, 3.0),
        "left": (1.0, 22.0, 3.0),
        "right": (11.0, 2.0, 3.0),
    }
    for face, origin in expected.items():
        assert F.face_frame(face, lo, hi)[0] == origin, face


def test_component_face_to_world_follows_rotation():
    """A component's `front` USB becomes a `right` cutout when the part is turned 90 deg,
    and its 9x4 opening keeps its shape in the new face's own axes."""
    env = make_pcb()
    access = env.access[0]

    face0, c0, s0 = F.component_face_to_world(Placed(env, (0.0, 0.0, 0.0)), access)
    assert face0 == "front"
    assert c0 == (25.0, 0.0, 4.0)
    assert s0 == (9.0, 4.0)

    # rotate 90 deg about Z: local +X -> world +Y, local -Y (the front normal) -> world +X
    face90, c90, s90 = F.component_face_to_world(Placed(env, (0.0, 0.0, 0.0), (0.0, 0.0, 90.0)), access)
    assert face90 == "right"
    assert c90 == (0.0, 25.0, 4.0)
    assert s90 == (9.0, 4.0), "a Z rotation must not swap width for height on a side wall"

    # 180 deg puts it on the back face, mirrored in u
    face180, _c180, _s180 = F.component_face_to_world(
        Placed(env, (0.0, 0.0, 0.0), (0.0, 0.0, 180.0)), access
    )
    assert face180 == "back"


def test_hole_to_world_lands_on_the_mount_plane():
    env = make_pcb()
    placed = Placed(env, (10.0, 20.0, 5.0))
    pts = [F.hole_to_world(placed, h) for h in env.holes]
    assert (13.0, 23.0, 7.0) in pts
    assert (57.0, 47.0, 7.0) in pts
    assert F.mount_plane_z(placed) == 7.0


def test_fastener_spec_normalises_the_ir_enum():
    for name, size, heat in (
        ("m2", "m2", False),
        ("M3", "m3", False),
        ("heat-set-m3", "m3", True),
        ("self-tap", "m3", False),
        ("nonsense", "m3", False),
    ):
        got_size, row, got_heat = F.fastener_spec(name)
        assert (got_size, got_heat) == (size, heat), name
        assert row["clearance_d"] > row["pilot_d"] > 0.0, name


# --------------------------------------------------------------------------------------
# enclosure.py — the tray_lid default
# --------------------------------------------------------------------------------------


def test_tray_lid_has_base_and_lid():
    r = build()
    assert set(r.parts) == {"base", "lid"}
    assert r.part_names()[0] == "base", "the floor part must come first"
    for name, node in r.parts.items():
        assert_valid_ast(node, name)
        assert name in group_names(node), "%s must keep its Group name for split output" % name


def test_outer_bbox_is_cavity_plus_two_walls_on_x_and_y():
    """The wall arithmetic is the load-bearing claim of the whole module."""
    r = build()
    (ox0, oy0, oz0), (ox1, oy1, oz1) = r.outer_bbox
    (cx0, cy0, cz0), (cx1, cy1, cz1) = r.inner_cavity
    assert abs((cx0 - ox0) - WALL) < 1e-9 and abs((ox1 - cx1) - WALL) < 1e-9
    assert abs((cy0 - oy0) - WALL) < 1e-9 and abs((oy1 - cy1) - WALL) < 1e-9
    assert abs(r.outer_size[0] - (r.cavity_size[0] + 2 * WALL)) < 1e-9
    assert abs(r.outer_size[1] - (r.cavity_size[1] + 2 * WALL)) < 1e-9
    # floor below, lid above
    assert abs((cz0 - oz0) - r.floor) < 1e-9
    assert abs((oz1 - cz1) - WALL) < 1e-9
    # datum: Z = 0 is the outer bottom, which is what a slicer expects
    assert oz0 == 0.0
    assert r.world_offset[2] > 0.0, "the shift is what moved the floor to Z=0"


def test_cavity_reserves_room_under_the_board_for_standoffs():
    """A board on 4 mm standoffs needs 4 mm of air under it that no bbox describes."""
    r = build()
    cav_floor = r.inner_cavity[0][2]
    board_bottom = MOUNT_PLANE_Z + r.world_offset[2]
    assert abs((board_bottom - cav_floor) - STANDOFF_H) < 1e-9

    # with no standoffs the cavity floor rises to the keep-out
    flat = build(comps={"PCB1": comp("adhesive")})
    assert flat.cavity_size[2] < r.cavity_size[2]


def test_four_standoffs_at_the_expected_world_xy():
    r = build()
    assert len(r.standoff_positions) == 4
    got = sorted((round(s["at"][0], 6), round(s["at"][1], 6)) for s in r.standoff_positions)
    assert got == sorted(PCB_HOLES), got
    for s in r.standoff_positions:
        assert abs(s["h"] - STANDOFF_H) < 1e-9, "a standoff spans floor -> mount plane"
        assert s["z0"] == r.inner_cavity[0][2], "standoffs start on the cavity floor"
        assert s["hole_d"] == F.FASTENERS["m3"]["pilot_d"], "self-tap is the default bore"
        assert s["od"] > s["hole_d"], "a bore wider than its boss is not a boss"
        assert s["ref"] == "PCB1" and s["hole_ref"] is not None


def test_heat_set_and_through_bores_differ_from_self_tap():
    pilot = build().standoff_positions[0]["hole_d"]
    heat = build(comps={"PCB1": comp("standoff", mount=dict(comp()["mount"], fastener="heat-set-m3"))})
    through = build(comps={"PCB1": comp("standoff", mount=dict(comp()["mount"], through=True))})
    assert heat.standoff_positions[0]["hole_d"] == F.HEAT_SET["m3"]["hole_d"] > pilot
    assert through.standoff_positions[0]["hole_d"] == F.FASTENERS["m3"]["clearance_d"] > pilot
    assert heat.standoff_positions[0]["style"] == "heat-set"
    assert through.standoff_positions[0]["style"] == "through"


def test_front_access_becomes_a_front_cutout_on_the_base():
    r = build()
    fronts = [c for c in r.cutouts if c["face"] == "front" and c["kind"] == "access"]
    assert len(fronts) == 1
    cut = fronts[0]
    assert cut["ref"] == "PCB1" and cut["name"] == "usb"
    assert cut["parts"] == ["base"], "a side-wall feature is cut from the tray, not the lid"
    assert cut["size"] == (9.0, 4.0)
    # the centre sits on the component's front plane and inside the shell
    assert abs(cut["center"][1] - 0.0) < 1e-9
    assert r.outer_bbox[0][1] < cut["center"][1] < r.outer_bbox[1][1]
    assert cut["depth"] > WALL, "the cutter must reach past the wall to break out"


def test_through_lid_access_is_cut_from_the_lid():
    env = make_pcb()
    top = Access("display", "top", (25.0, 15.0), (20.0, 12.0), "rect", 0.4, 0.0, True)
    env = Envelope(
        ref=env.ref, size=env.size, holes=env.holes, access=(env.access[0], top),
        mount_plane_z=env.mount_plane_z, mount_thickness=env.mount_thickness, source=env.source,
    )
    r = build(placed=Placed(env, (0.0, 0.0, 0.0)))
    lid_cuts = [c for c in r.cutouts if "lid" in c["parts"]]
    assert [c["name"] for c in lid_cuts] == ["display"]
    assert lid_cuts[0]["face"] == "top"


def test_lid_gets_four_screw_posts_and_matching_countersinks():
    r = build()
    assert len(r.screw_positions) == 4
    assert r.fastening == "screws"
    assert group_names(r.parts["base"]).issuperset({"screw_post"})
    assert group_names(r.parts["lid"]).issuperset({"lip", "countersink"})
    (cx0, cy0, _z0), (cx1, cy1, _z1) = r.inner_cavity
    for s in r.screw_positions:
        x, y = s["at"]
        assert cx0 < x < cx1 and cy0 < y < cy1, "a post must sit inside the cavity"
        assert s["h"] > 0 and s["od"] > s["hole_d"] > 0
    assert len(set(s["at"] for s in r.screw_positions)) == 4, "posts must not be stacked"


def test_screw_posts_do_not_collide_with_the_components():
    """The cavity is widened when a corner post would otherwise land on a board."""
    r = build()
    board = Placed(make_pcb(), (0.0, 0.0, 0.0)).world_bbox()
    for s in r.screw_positions:
        x, y = s["at"]
        rad = s["od"] / 2.0
        clear = (
            x + rad <= board[0][0] or x - rad >= board[1][0]
            or y + rad <= board[0][1] or y - rad >= board[1][1]
        )
        assert clear, "post at %r (od %.1f) overlaps the board footprint" % (s["at"], s["od"])
    assert any("widened" in n for n in r.notes)


def test_screw_count_is_honoured():
    for n in (2, 4, 6):
        r = build(spec={"lid": {"fastening": "screws", "screw_count": n}})
        assert len(r.screw_positions) == n, n


def test_every_fastening_mode_builds():
    seen = {}
    for mode in ("screws", "snap", "friction", "magnets", "hinge", "none"):
        r = build(spec={"lid": {"fastening": mode}})
        assert set(r.parts) == {"base", "lid"}
        for name, node in r.parts.items():
            assert_valid_ast(node, "%s/%s" % (mode, name))
        assert "lip" in group_names(r.parts["lid"]), "every mode keeps the alignment lip"
        seen[mode] = group_names(r.parts["base"]) | group_names(r.parts["lid"])
    assert "snap_hook" in seen["snap"] and "snap_recess" in seen["snap"]
    assert "magnet_pocket" in seen["magnets"]
    assert "screw_post" in seen["screws"]
    assert "screw_post" not in seen["friction"]


def test_every_mount_style_builds():
    expected = {
        "standoff": "standoff",
        "pocket": "pocket_ledge",
        "clip": "snap_hook",
        "slot": "rail",
    }
    for style, marker in expected.items():
        r = build(comps={"PCB1": comp(style)})
        names = group_names(r.parts["base"])
        assert marker in names, "%s should print a %s" % (style, marker)
        assert_valid_ast(r.parts["base"], style)
    for style in ("adhesive", "none"):
        r = build(comps={"PCB1": comp(style)})
        assert_valid_ast(r.parts["base"], style)
        assert any(style in n for n in r.notes)


def test_every_style_produces_parts():
    expected = {
        "tray_lid": {"base", "lid"},
        "shell": {"shell"},
        "clamshell": {"lower", "upper"},
        "sleeve": {"sleeve"},
        "open_frame": {"base"},
    }
    for style, names in expected.items():
        r = build(spec={"style": style})
        assert set(r.parts) == names, style
        assert r.style == style
        for name, node in r.parts.items():
            assert_valid_ast(node, "%s/%s" % (style, name))
        assert r.outer_bbox[0][2] == 0.0, "%s: Z=0 must be the print bed" % style
        assert r.summary()["parts"] == list(r.parts), style


def test_auto_vents_appear_only_when_something_is_hot():
    cold = build()
    assert not [c for c in cold.cutouts if c["kind"] == "vent"]

    hot = build(placed=Placed(make_pcb(thermal_w=2.5), (0.0, 0.0, 0.0)),
                comps={"PCB1": comp(thermal_w=2.5)})
    vents = [c for c in hot.cutouts if c["kind"] == "vent"]
    assert sorted(c["face"] for c in vents) == ["left", "right"], "default faces are left/right"
    assert all(c.get("auto") for c in vents)
    assert group_names(hot.parts["base"]).issuperset({"vents_left", "vents_right"})
    assert any("auto vents" in n for n in hot.notes)

    off = build(placed=Placed(make_pcb(thermal_w=2.5), (0.0, 0.0, 0.0)),
                comps={"PCB1": comp(thermal_w=2.5)}, spec={"vents": {"auto": False}})
    assert not [c for c in off.cutouts if c["kind"] == "vent"]

    picked = build(placed=Placed(make_pcb(thermal_w=2.5), (0.0, 0.0, 0.0)),
                   comps={"PCB1": comp(thermal_w=2.5)},
                   spec={"vents": {"auto": True, "faces": ["front"]}})
    assert [c["face"] for c in picked.cutouts if c["kind"] == "vent"] == ["front"]


def test_label_text_group_is_present_on_the_lid():
    r = build(spec={"text": {"label": "SENSOR-1", "size": 6.0, "depth": 0.6}})
    names = group_names(r.parts["lid"])
    assert "text_deboss" in names
    assert any(n.startswith("text:") and "SENSOR-1" in n for n in names), names
    assert any("SENSOR-1" in n for n in r.notes)

    # the design title is the default label, and emboss flips which operation is used
    titled = build(title="my-box")
    assert any("my-box" in n for n in group_names(titled.parts["lid"]) if n.startswith("text:"))
    up = build(spec={"text": {"label": "UP", "emboss": True}})
    assert "text_emboss" in group_names(up.parts["lid"])

    # a long label is scaled to fit rather than overflowing the lid
    long_r = build(spec={"text": {"label": "A" * 60, "size": 10.0}})
    assert any("scaled down" in n for n in long_r.notes)


def test_enclosure_level_features_are_applied():
    r = build(spec={
        "features": [
            {"kind": "foot", "count": 4},
            {"kind": "cable_gland", "face": "back", "params": {"thread": "pg7"}},
            {"kind": "cutout", "face": "left", "size": [8.0, 6.0]},
            {"kind": "wall_mount_tab", "face": "back"},
            {"kind": "din_clip"},
            {"kind": "text", "face": "top", "text": "REV A", "params": {"emboss": False}},
        ],
    })
    names = group_names(r.parts["base"]) | group_names(r.parts["lid"])
    assert {"foot_recess", "wall_mount_tab", "din_clip"}.issubset(names), names
    kinds = set(c["kind"] for c in r.cutouts)
    assert {"cable_gland", "cutout"}.issubset(kinds), kinds
    gland = [c for c in r.cutouts if c["kind"] == "cable_gland"][0]
    assert gland["size"] == (12.5, 12.5) and gland["shape"] == "round", "PG7 default"
    assert_valid_ast(r.parts["base"], "features/base")


# --------------------------------------------------------------------------------------
# through-features pierce the material they are declared on (floor, lid, wall, plate)
# --------------------------------------------------------------------------------------

#: The 2026-10-08 report: an open_frame with a 2.8 mm floor and an M4 clearance hole made with
#: a `cable_gland`. The cutter was sized from `wall`, centred on the surface, and reached only
#: (wall + 2) / 2 deep - the hole came out blind.
M4_CLEAR = 4.4


def _cylinder_cutters(node, d):
    return [n for n in walk(node) if n.kind == "cylinder" and abs(2.0 * n.r - d) < 1e-6]


def _assert_round_cutter_pierces(r, part, d, wall_t):
    """A `span="both"` cutter is centred on the outer surface, so it reaches h/2 inward."""
    cutters = _cylinder_cutters(r.parts[part], d)
    assert cutters, "no d%.2f cutter in %s" % (d, part)
    reach = min(c.h for c in cutters) / 2.0
    assert reach >= wall_t + 0.5, "cutter reaches %.3f into %.3f of material" % (reach, wall_t)


def test_cable_gland_pierces_a_thick_open_frame_floor():
    r = build(spec={"style": "open_frame", "floor": 2.8, "features": [
        {"kind": "cable_gland", "face": "bottom", "at": [10.0, 10.0], "params": {"d": M4_CLEAR}},
    ]})
    rec = [c for c in r.cutouts if c["kind"] == "cable_gland"][0]
    assert rec["through"] is True and rec["wall_t"] == 2.8, rec
    assert abs(rec["surface_center"][2] - 0.0) < 1e-9, "bottom surface is the print bed"
    _assert_round_cutter_pierces(r, "base", M4_CLEAR, 2.8)


def test_through_features_follow_the_thickness_of_their_own_face():
    """Floor, wall and lid are independent numbers; each face's cutter is sized from its own."""
    r = build(spec={"wall": 3.5, "floor": 4.0, "features": [
        {"kind": "cable_gland", "face": "bottom", "at": [10.0, 10.0], "params": {"d": 5.0}},
        {"kind": "cable_gland", "face": "left", "params": {"d": 6.0}},
        {"kind": "cutout", "face": "top", "at": [10.0, 10.0], "size": [7.0, 7.0],
         "shape": "round"},
    ]})
    by_face = dict((c["face"], c) for c in r.cutouts if c["kind"] in ("cutout", "cable_gland"))
    assert by_face["bottom"]["wall_t"] == 4.0
    assert by_face["left"]["wall_t"] == 3.5
    assert by_face["top"]["wall_t"] == 3.5 and by_face["top"]["parts"] == ["lid"]
    _assert_round_cutter_pierces(r, "base", 5.0, 4.0)
    _assert_round_cutter_pierces(r, "base", 6.0, 3.5)
    _assert_round_cutter_pierces(r, "lid", 7.0, 3.5)


def test_open_frame_top_feature_lands_on_the_plate_not_the_lid_plane():
    """An open_frame has no lid; its top face is the plate's top, `floor` above the bed."""
    r = build(spec={"style": "open_frame", "floor": 2.8, "features": [
        {"kind": "cutout", "face": "top", "at": [10.0, 10.0], "size": [M4_CLEAR, M4_CLEAR],
         "shape": "round"},
    ]})
    rec = [c for c in r.cutouts if c["kind"] == "cutout"][0]
    assert abs(rec["surface_center"][2] - 2.8) < 1e-9, rec
    assert rec["wall_t"] == 2.8
    _assert_round_cutter_pierces(r, "base", M4_CLEAR, 2.8)


def test_access_cutouts_record_what_the_through_gate_needs():
    r = build(spec={"wall": 3.0})
    usb = [c for c in r.cutouts if c["kind"] == "access"][0]
    assert usb["through"] is True and usb["wall_t"] == 3.0
    ox0, oy0, _oz0 = r.outer_bbox[0]
    assert abs(usb["surface_center"][1] - oy0) < 1e-9, "front surface is the outer -Y face"


def test_through_features_are_open_on_both_kernels():
    """Render an open_frame plate with a gland and a round `cutout` through a 2.8 mm floor on
    each kernel that is installed, and let the `cutout_through` gate measure the mesh."""
    from pathlib import Path

    from phimech.check.gates import _gate_cutout_through

    r = build(spec={"style": "open_frame", "floor": 2.8, "features": [
        {"kind": "cable_gland", "face": "bottom", "at": [10.0, 10.0], "params": {"d": M4_CLEAR}},
        {"kind": "cutout", "face": "bottom", "at": [30.0, 10.0], "size": [M4_CLEAR, M4_CLEAR],
         "shape": "round"},
    ]})
    backends = []
    try:
        from phimech.backends.openscad import OpenSCADBackend

        if OpenSCADBackend().available() and _rotate_emitter_is_sane():
            backends.append(OpenSCADBackend())
    except ImportError:
        pass
    try:
        from phimech.backends.freecad import FreeCADBackend

        if FreeCADBackend().available():
            backends.append(FreeCADBackend())
    except ImportError:
        pass
    if not backends:
        _skip("neither openscad nor freecadcmd is installed")
        return
    tmp = tempfile.mkdtemp(prefix="phimech-thru-")
    try:
        for backend in backends:
            out = Path(tmp) / ("base-%s.stl" % backend.name)
            backend.render(r.parts["base"], out, "stl")

            class _M(object):
                extra = {"stl": str(out)}

            gate = _gate_cutout_through(r, {"base": _M()})
            assert gate.passed is True, "%s: %s" % (backend.name, gate.message)
            # the gland, the round cutout and the board's USB opening
            assert "over 3 opening(s)" in gate.message, gate.message
            print("    %s: %s" % (backend.name, gate.message))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_open_frame_wall_mount_tabs_are_ears_attached_to_the_plate():
    """2026-10-08: on an open_frame the tabs stood at mid-height of the outer box, in the air
    above the plate, and printed as separate bodies. They are now in-plane ears that overlap
    the plate edge."""
    r = build(spec={"style": "open_frame", "floor": 2.4,
                    "features": [{"kind": "wall_mount_tab", "face": "back"}]})
    assert any("mounting ears" in n for n in r.notes), r.notes
    cubes = [n for n in walk(r.parts["base"]) if n.kind == "cube"]
    ears = [c for c in cubes if abs(c.size[2] - 2.4) < 1e-9 and abs(c.size[0] - 16.0) < 1e-9]
    assert len(ears) == 2, [c.size for c in cubes]
    tabs = [n for n in walk(r.parts["base"]) if n.kind == "group" and n.name == "wall_mount_tab"]
    assert len(tabs) == 2
    try:
        from phimech.backends.openscad import OpenSCADBackend
    except ImportError:
        return
    backend = OpenSCADBackend()
    if not backend.available():
        _skip("openscad binary does not respond")
        return
    from pathlib import Path

    from phimech.check.mesh import count_bodies, read_stl

    tmp = tempfile.mkdtemp(prefix="phimech-ear-")
    try:
        out = Path(tmp) / "base.stl"
        backend.render(r.parts["base"], out, "stl")
        tris = read_stl(str(out))
        assert count_bodies(tris) == 1, "the ears print as separate bodies"
        _vol, mn, mx = _stl_volume_and_bbox(str(out))
        assert mx[2] <= r.outer_size[2] + 1e-6 and mx[1] > r.outer_bbox[1][1] + 10.0, (mn, mx)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_open_frame_label_sits_on_the_plate_top_and_clears_a_fan_hole():
    """The label used to be debossed from `lid_t` above the bed - inside a thicker plate, a
    buried void - and centred over whatever hole was there, which then cut it away."""
    r = build(spec={"style": "open_frame", "floor": 2.4, "text": {"label": "FAN", "size": 6.0},
                    "features": [{"kind": "cutout", "face": "bottom", "size": [14.0, 14.0],
                                  "shape": "round"}]})
    assert len(r.labels) == 1 and not r.labels[0].get("skipped"), r.labels
    lab = r.labels[0]["rect"]
    hole = [c for c in r.cutouts if c["kind"] == "cutout"][0]
    hx, hy = hole["center"][0], hole["center"][1]
    gap_y = max(lab[1] - (hy + 7.0), (hy - 7.0) - lab[3])
    gap_x = max(lab[0] - (hx + 7.0), (hx - 7.0) - lab[2])
    assert max(gap_x, gap_y) >= 1.0, (lab, hole["center"])
    deboss = [n for n in walk(r.parts["base"]) if n.kind == "group" and n.name == "text_deboss"]
    assert deboss, "label missing"


def test_label_with_no_clear_spot_is_skipped_and_recorded():
    r = build(spec={"style": "open_frame", "floor": 2.4, "text": {"label": "FAN", "size": 6.0},
                    "features": [{"kind": "cutout", "face": "bottom", "size": [70.0, 50.0]}]})
    assert r.labels and r.labels[0].get("skipped") is True, r.labels
    assert any("skipped" in n for n in r.notes)


def test_assembly_and_exploded_are_valid_and_exploded_lifts_the_lid():
    r = build()
    asm, exp = r.assembly(), r.exploded(gap=30.0)
    assert_valid_ast(asm, "assembly")
    assert_valid_ast(exp, "exploded")
    assert "cmp_PCB1" in group_names(asm), "components must ride along in the assembly"
    assert {"base", "lid"}.issubset(group_names(asm))
    lid_z_asm = _min_z_of_group(asm, "lid")
    lid_z_exp = _min_z_of_group(exp, "lid")
    assert lid_z_exp is not None and lid_z_asm is not None
    assert lid_z_exp > lid_z_asm, "the exploded lid must sit above the assembled lid"
    assert abs((lid_z_exp - lid_z_asm) - 30.0) < 1e-6
    base_z_asm = _min_z_of_group(asm, "base")
    assert abs(_min_z_of_group(exp, "base") - base_z_asm) < 1e-9, "the tray does not move"


def _min_z_of_group(root, name):
    """Accumulated Z translation down to the named group — enough to compare part heights."""
    def rec(node, z):
        if node.kind == "translate":
            z = z + node.v[2]
        if node.kind == "group" and node.name == name:
            return z
        for c in getattr(node, "children", ()) or ():
            got = rec(c, z)
            if got is not None:
                return got
        return None
    return rec(root, 0.0)


def test_summary_reports_the_numbers_a_gate_needs():
    r = build(spec={"features": [{"kind": "foot"}]})
    s = r.summary()
    assert s["style"] == "tray_lid"
    assert s["parts"] == ["base", "lid"]
    assert s["wall_mm"] == WALL and s["floor_mm"] == r.floor
    assert s["standoffs"] == 4 and s["screws"] == 4
    assert s["cutouts"] == len(r.cutouts) >= 1
    assert s["components"] == ["PCB1"]
    assert s["outer_size_mm"] == [round(v, 3) for v in r.outer_size]
    # a hollow box weighs less than a solid one but more than nothing
    ow, od, oh = r.outer_size
    assert 0.0 < s["estimated_material_mm3"] < ow * od * oh
    assert isinstance(s["notes"], list)
    import json

    json.dumps(r.to_dict())  # the report has to survive serialisation


def test_manufacturing_min_wall_raises_a_thin_wall():
    r = build(spec={"wall": 0.8}, mfg={"min_wall": 1.6})
    assert r.wall == 1.6 and r.floor == 1.6
    assert any("min_wall" in n for n in r.notes)
    assert abs(r.outer_size[0] - (r.cavity_size[0] + 2 * 1.6)) < 1e-9
    # a wall thicker than the minimum is left alone
    assert build(spec={"wall": 3.0}, mfg={"min_wall": 1.6}).wall == 3.0


def test_oversize_part_is_flagged_against_the_printer_bed():
    r = build(mfg={"printer_bed": [20.0, 20.0, 20.0]})
    assert any(n.startswith("WARNING") and "printer_bed" in n for n in r.notes)
    assert not [n for n in build(mfg={"printer_bed": [300.0, 300.0, 300.0]}).notes
                if n.startswith("WARNING")]


def test_two_components_share_one_cavity():
    pcb = Placed(make_pcb(), (0.0, 0.0, 0.0))
    batt = Placed(
        Envelope(ref="BAT1", size=(70.0, 20.0, 20.0), mount_plane_z=0.0, name="18650"),
        (0.0, 34.0, 0.0),
    )
    r = build(placed=[pcb, batt], comps={"PCB1": comp(), "BAT1": comp("pocket")})
    assert set(r.component_bodies) == {"PCB1", "BAT1"}
    assert r.cavity_size[0] >= 70.0 and r.cavity_size[1] >= 54.0
    assert r.cavity_size[2] >= 20.0, "the taller part sets the cavity height"
    assert len(r.standoff_positions) == 4, "only the board declares holes"
    assert_valid_ast(r.assembly(), "two-component assembly")


def test_empty_placement_is_an_error():
    try:
        E.synthesize(StubPlacement([]), {}, {}, components_by_ref={})
    except ValueError as exc:
        assert "zero components" in str(exc)
    else:
        raise AssertionError("an enclosure around nothing must not be synthesised")


def test_unknown_style_falls_back_with_a_note():
    r = build(spec={"style": "geodesic-dome"})
    assert r.style == "tray_lid"
    assert any("geodesic-dome" in n for n in r.notes)


def test_real_placement_class_is_compatible_if_present():
    """If place.py has landed, the real Placement must work where the stub does."""
    try:
        from phimech.synth.place import Placement  # type: ignore
    except ImportError:
        return  # slice not present yet; the stub already proved the contract
    placed = [Placed(make_pcb(), (0.0, 0.0, 0.0))]
    try:
        real = Placement(placed)  # type: ignore[call-arg]
    except TypeError:
        return  # different constructor; place.py's own tests own that shape
    r = E.synthesize(real, {"style": "tray_lid"}, {}, components_by_ref={"PCB1": comp()})
    assert set(r.parts) == {"base", "lid"}


# --------------------------------------------------------------------------------------
# backend check — skipped unless openscad is installed
# --------------------------------------------------------------------------------------


def _rotate_emitter_is_sane():
    """True if the .scad emitter produces a rotation OpenSCAD actually applies.

    OpenSCAD's `rotate(v=[x,y,z])` means `rotate(a=undef, v=[x,y,z])`, which is the *identity*
    — the angle goes in `a`. An emitter that writes `rotate(v=...)` therefore drops every
    rotation, and since this module orients every cutter and wall feature by rotating about the
    origin, the render would be geometric nonsense. That is a bug in the backend slice, not
    here, so this test verifies the emitter first and reports precisely which layer is at fault
    rather than blaming the enclosure.
    """
    from phimech.backends.scad_emit import emit
    from phimech.csg.ast import Cube, Rotate

    src = emit(Rotate((0.0, 90.0, 0.0), [Cube((10.0, 1.0, 1.0))]))
    return "rotate(v=" not in src


def test_openscad_render_is_solid():
    """The base must render to a real solid: volume > 0 and less than its bounding box."""
    if not (shutil.which("openscad") or os.environ.get("PHIMECH_OPENSCAD")):
        _skip("openscad not installed")
        return
    try:
        from phimech.backends.openscad import OpenSCADBackend
    except ImportError:
        _skip("phimech.backends.openscad not present yet")
        return
    backend = OpenSCADBackend()
    if not backend.available():
        _skip("openscad binary does not respond")
        return
    if not _rotate_emitter_is_sane():
        _skip("scad_emit writes `rotate(v=..)`, which OpenSCAD ignores — fix the backend slice")
        return

    r = build()
    ow, od, oh = r.outer_size
    tmp = tempfile.mkdtemp(prefix="phimech-encl-")
    try:
        from pathlib import Path

        out = Path(tmp) / "base.stl"
        res = backend.render(r.parts["base"], out)
        assert out.exists() and res.bytes > 0
        vol, mn, mx = _stl_volume_and_bbox(str(out))
        assert vol > 0.0, "the base rendered empty"
        assert vol < ow * od * oh, "a tray cannot weigh as much as a solid block: %.1f" % vol
        for i, span in enumerate((ow, od)):
            assert abs((mx[i] - mn[i]) - span) < 0.2, "axis %d: %r vs %r" % (i, mx[i] - mn[i], span)
        assert mn[2] > -0.05, "Z=0 must be the print bed, got %r" % (mn[2],)
        assert mx[2] <= oh + 0.05, "the tray is taller than the shell: %r > %r" % (mx[2], oh)
        assert mx[2] < oh - 0.5, "an open tray must stop short of the lid's outer face"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stl_volume_and_bbox(path):
    """Signed-tetrahedron volume + bbox straight from the STL, so no numpy-stl is needed."""
    import struct

    tris = []
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:2048]:
        pts = []
        for line in data.decode("utf-8", "ignore").splitlines():
            if "vertex" in line:
                pts.append(tuple(float(v) for v in line.split()[1:4]))
        tris = [tuple(pts[i:i + 3]) for i in range(0, len(pts) - 2, 3)]
    else:
        count = struct.unpack("<I", data[80:84])[0]
        for i in range(count):
            off = 84 + i * 50 + 12
            tris.append(tuple(struct.unpack("<3f", data[off + j * 12:off + j * 12 + 12]) for j in range(3)))
    vol = 0.0
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for a, b, c in tris:
        vol += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
        for p in (a, b, c):
            for i in range(3):
                mn[i] = min(mn[i], p[i])
                mx[i] = max(mx[i], p[i])
    return abs(vol), tuple(mn), tuple(mx)


def _skip(reason):
    # Only hand a skip to pytest when pytest is actually
    # driving, because Skipped derives from BaseException and escapes a direct runner.
    if _os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            import pytest

            pytest.skip(reason)
        except ImportError:
            pass
    print("    SKIP: %s" % reason)


# --------------------------------------------------------------------------------------
# plain-python runner, so `python3 tests/test_enclosure.py` works without pytest
# --------------------------------------------------------------------------------------


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  ok   %s" % name)
        except Exception as exc:  # noqa: BLE001 - a runner reports, it does not handle
            import traceback

            failed.append(name)
            print("  FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
            traceback.print_exc()
    print("\n%d/%d passed%s" % (len(tests) - len(failed), len(tests),
                                "" if not failed else "; failed: " + ", ".join(failed)))
    if F.USING_FALLBACK_OPS:
        print("note: phimech.csg.ops was not importable; features.py built from bare AST")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
