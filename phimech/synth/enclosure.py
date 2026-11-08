"""Enclosure synthesis — placed components in, printable parts out.

This is the module that earns PhiMech its name: given a `Placement` (components already
positioned relative to each other by `synth/place.py`) and the `enclosure` block of the design
IR, it derives every dimension of the casing and returns it as CSG.

The one rule that shapes the whole file: **coordinates are never model output.** The agent says
"tray_lid, 2 mm wall, 1.5 mm clearance, M3 standoffs, screws" and everything else — cavity size,
wall positions, standoff heights, cutout centres, vent pitch — is computed here from the
component envelopes. There is no `position: [x, y, z]` for a wall anywhere in the IR.

Pipeline inside `synthesize()`
----------------------------
1. **Cavity.** Union of the component keep-outs (each world bbox grown by its `clearance`),
   expanded by `enclosure.clearance` on ±X/±Y and +Z. The cavity floor is pulled down to the
   lowest *mount plane minus standoff height*, because a board on 4 mm standoffs needs 4 mm of
   air under it that no component bbox describes.
2. **Shell.** `outer = cavity ± wall`, with `floor` under it and the lid's `wall` over it.
   The solid is `rounded_box(outer) - rounded_box(cavity)`; the cavity cutter is extended
   upward past the rim so the difference never leaves a co-planar skin (`EPS` over-cut).
3. **Mounts.** Per component `mount.style`: `standoff` → a boss per envelope hole from the
   floor to the mount plane; `pocket` → four corner ledges; `clip` → two snap hooks; `slot` →
   two rails; `adhesive`/`none` → nothing.
4. **Lid fastening.** `screws` → corner posts in the base + countersinks in the lid; `snap` →
   hooks in the base + recesses in the lid lip; `magnets` → matching disc pockets in both;
   `friction` → a tighter lip; `hinge`/`none` → lip only.
5. **Cutouts.** Every component `Access` is mapped through `features.component_face_to_world()`
   onto an enclosure face and cut out of whichever part owns that face.
6. **Auto vents.** If `vents.auto` and anything declares `thermal_w`, slot vents on the named
   faces.
7. **Text.** `enclosure.text.label` (or the design title) embossed or debossed on the lid.
8. **Zero the datum.** Every part and every component body is finally shifted so that Z = 0 is
   the outer bottom of the floor — the plane a printer prints on and a slicer expects.

Styles
------
======================================================================================
`tray_lid`   (default) open-top tray + a separate lid plate with an alignment lip.
`shell`      one sealed body, no lid. Not printable as-is; slice it or use it for CNC.
`clamshell`  two half-height halves split at mid Z, mating lip on the lower half.
`sleeve`     a tube open at both ±X ends; the assembly slides in.
`open_frame` floor plate + mounts only, no walls. For chassis-mounted electronics.
======================================================================================

stdlib only; must import under python 3.9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phimech.components.envelope import Hole, Placed
from phimech.csg.ast import Cube, Difference, Group, Node, Translate, Union
from phimech.synth import features as F
from phimech.synth.features import EPS, ops

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Box = Tuple[Vec3, Vec3]

STYLES = ("shell", "tray_lid", "clamshell", "sleeve", "open_frame")

#: Which part owns the floor and which owns the top, per style. `None` = the style has no
#: separate top part, so a `top`/`through_lid` feature is cut out of the floor part instead.
STYLE_PARTS: Dict[str, Tuple[str, Optional[str]]] = {
    "tray_lid": ("base", "lid"),
    "shell": ("shell", None),
    "clamshell": ("lower", "upper"),
    "sleeve": ("sleeve", None),
    "open_frame": ("base", None),
}

#: Faces that belong to the lid/top part rather than the walls.
_TOP_FACES = ("top",)

_MAGNET_D = 6.2
_MAGNET_H = 2.2
_FOOT_D = 8.0
_FOOT_H = 1.0
_GLAND_D = {"pg7": 12.5, "pg9": 15.2, "pg11": 18.6, "m12": 12.0, "m16": 16.0}


# --------------------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------------------


@dataclass
class EnclosureParts:
    """Everything `synthesize()` produced, in the printer datum frame (Z = 0 = floor bottom).

    Mutable on purpose: `pipeline.py` attaches measurements to it after rendering, and the
    gates read `standoff_positions` / `cutouts` as data rather than re-deriving geometry.
    """

    parts: Dict[str, Node] = field(default_factory=dict)
    component_bodies: Dict[str, Node] = field(default_factory=dict)
    inner_cavity: Box = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    outer_bbox: Box = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    standoff_positions: List[Dict[str, Any]] = field(default_factory=list)
    cutouts: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    world_offset: Vec3 = (0.0, 0.0, 0.0)
    style: str = "tray_lid"
    wall: float = 2.0
    floor: float = 2.0
    screw_positions: List[Dict[str, Any]] = field(default_factory=list)
    fastening: str = "screws"
    # Glyph height actually cut, AFTER any auto-scaling to fit the face. The legibility gate
    # must judge what was built, not what was asked for: a label inherited from the title with
    # no `enclosure.text` block was invisible to the gate, so a run that silently shrank text to
    # 2.2 mm still reported "no text".
    text_sizes: List[float] = field(default_factory=list)
    # Where each label was cut, as `{"label", "part", "face", "rect": (x0, y0, x1, y1), "size"}`:
    # the footprint `text_legible` checks against the through-cutouts on the same part, so a
    # label a later hole cut away cannot be reported as legible.
    labels: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------------------------

    def part_names(self) -> List[str]:
        """Printable part names, in assembly order (floor part first)."""
        return list(self.parts.keys())

    def assembly(self) -> Node:
        """All parts plus the component bodies, in place.

        Each part already carries its own `Group` name from synthesis, so nothing is re-wrapped
        here: an extra naming group *outside* the part's shift transform would make the name the
        outermost node and hide the translation from anything that reads names positionally
        (the exploded view, per-part file splitting). Component bodies get a `cmp_<ref>` group
        because their provider geometry is unnamed.
        """
        children: List[Node] = list(self.parts.values())
        for ref, body in self.component_bodies.items():
            children.append(Group([body], name="cmp_" + ref))
        return Group([Union(children)], name="assembly")

    def exploded(self, gap: float = 30.0) -> Node:
        """Assembly with the lid lifted by `gap` and the components by `gap / 2` — the view a
        human uses to check that cutouts line up with the things behind them.

        The lift is applied *outside* each part, so a part's own geometry and group name are
        untouched and the offset is readable straight off the tree.
        """
        floor_part, _top = STYLE_PARTS.get(self.style, ("base", "lid"))
        children: List[Node] = []
        for name, node in self.parts.items():
            dz = 0.0 if name == floor_part else float(gap)
            children.append(Translate((0.0, 0.0, dz), [node]) if dz else node)
        for ref, body in self.component_bodies.items():
            children.append(
                Translate((0.0, 0.0, gap / 2.0), [Group([body], name="cmp_" + ref)])
            )
        return Group([Union(children)], name="exploded")

    # ---------------------------------------------------------------------------------

    @property
    def outer_size(self) -> Vec3:
        return _size_of(self.outer_bbox)

    @property
    def cavity_size(self) -> Vec3:
        return _size_of(self.inner_cavity)

    def estimated_volume(self) -> float:
        """Material volume, bbox approximation: outer box minus cavity box. Cheap and always
        available — the real number needs a backend and lands in `measurements` later."""
        ow, od, oh = self.outer_size
        cw, cd, ch = self.cavity_size
        return max(0.0, ow * od * oh - cw * cd * ch)

    def summary(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "parts": self.part_names(),
            "outer_size_mm": [round(v, 3) for v in self.outer_size],
            "cavity_size_mm": [round(v, 3) for v in self.cavity_size],
            "outer_bbox": {"min": list(self.outer_bbox[0]), "max": list(self.outer_bbox[1])},
            "cavity_bbox": {"min": list(self.inner_cavity[0]), "max": list(self.inner_cavity[1])},
            "wall_mm": self.wall,
            "floor_mm": self.floor,
            "fastening": self.fastening,
            "standoffs": len(self.standoff_positions),
            "cutouts": len(self.cutouts),
            "screws": len(self.screw_positions),
            "components": sorted(self.component_bodies),
            "world_offset": list(self.world_offset),
            "estimated_material_mm3": round(self.estimated_volume(), 1),
            "text_sizes": list(self.text_sizes),
            "labels": [dict(lb) for lb in self.labels],
            "notes": list(self.notes),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = self.summary()
        d["standoff_positions"] = [dict(s) for s in self.standoff_positions]
        d["screw_positions"] = [dict(s) for s in self.screw_positions]
        d["cutout_details"] = [dict(c) for c in self.cutouts]
        return d


# --------------------------------------------------------------------------------------
# part builder
# --------------------------------------------------------------------------------------


class _Part(object):
    """Accumulator for one printable part: solids unioned, then cutters differenced.

    Kept as an explicit two-phase build because ordering is the classic CSG footgun — a bore
    added before its boss would be filled in again by the boss.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.solids: List[Node] = []
        self.cutters: List[Node] = []

    def add(self, node: Optional[Node]) -> None:
        if node is not None:
            self.solids.append(node)

    def cut(self, node: Optional[Node]) -> None:
        if node is not None:
            self.cutters.append(node)

    def build(self) -> Node:
        if not self.solids:
            return Group([], name=self.name)
        solid = self.solids[0] if len(self.solids) == 1 else Union(list(self.solids))
        if self.cutters:
            solid = Difference([solid] + list(self.cutters))
        return Group([solid], name=self.name)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _size_of(box: Box) -> Vec3:
    (x0, y0, z0), (x1, y1, z1) = box
    return (x1 - x0, y1 - y0, z1 - z0)


def _num(d: Dict[str, Any], key: str, default: float) -> float:
    v = d.get(key)
    return default if v is None else float(v)


def _grow(box: Box, dx: float, dy: float, dz_lo: float, dz_hi: float) -> Box:
    (x0, y0, z0), (x1, y1, z1) = box
    return ((x0 - dx, y0 - dy, z0 - dz_lo), (x1 + dx, y1 + dy, z1 + dz_hi))


def _union_box(a: Optional[Box], b: Box) -> Box:
    if a is None:
        return b
    return (
        tuple(min(a[0][i], b[0][i]) for i in range(3)),  # type: ignore[return-value]
        tuple(max(a[1][i], b[1][i]) for i in range(3)),  # type: ignore[return-value]
    )


def _mount_of(comp: Dict[str, Any]) -> Dict[str, Any]:
    m = comp.get("mount") or {}
    return {
        "style": m.get("style", "pocket"),
        "holes": m.get("holes") or [],
        "hole_d": _num(m, "hole_d", 3.2),
        "fastener": m.get("fastener", "m3"),
        "standoff_h": _num(m, "standoff_h", 4.0),
        "face": m.get("face", "bottom"),
        "through": bool(m.get("through", False)),
    }


def _holes_of(placed: Placed, mount: Dict[str, Any]) -> List[Hole]:
    """Envelope holes win (a PCB extractor found the real ones); the IR `mount.holes` list is
    the fallback for sources that cannot introspect themselves."""
    if placed.envelope.holes:
        return list(placed.envelope.holes)
    return [Hole((float(p[0]), float(p[1])), mount["hole_d"]) for p in mount["holes"] if len(p) >= 2]


def _iter_placed(placement: Any) -> List[Placed]:
    items = getattr(placement, "items", None)
    if items is None:
        raise TypeError("placement has no `items`; expected a phimech.synth.place.Placement")
    return list(items() if callable(items) else items)


def _body_of(placed: Placed) -> Node:
    """Renderable geometry for a component: its provider body if it has one, else its bbox."""
    env = placed.envelope
    node = env.body if env.body is not None else ops.box(env.size[0], env.size[1], env.size[2])
    return ops.place(node, placed.position, placed.rotation)


def _shift(node: Node, offset: Vec3) -> Node:
    if all(abs(v) < 1e-12 for v in offset):
        return node
    return Translate(offset, [node])


def _shift_box(box: Box, offset: Vec3) -> Box:
    return (
        tuple(box[0][i] + offset[i] for i in range(3)),  # type: ignore[return-value]
        tuple(box[1][i] + offset[i] for i in range(3)),  # type: ignore[return-value]
    )


def _face_point(face: str, box: Box, at: Optional[Sequence[float]]) -> Vec3:
    """A `(u, v)` point on a face of `box`, defaulting to the face centre when `at` is None."""
    if at is None:
        return F.face_center(face, box[0], box[1])
    origin, u, v, _n = F.face_frame(face, box[0], box[1])
    return (
        origin[0] + u[0] * at[0] + v[0] * at[1],
        origin[1] + u[1] * at[0] + v[1] * at[1],
        origin[2] + u[2] * at[0] + v[2] * at[1],
    )


def _outward_depth(face: str, center: Sequence[float], outer: Box) -> float:
    """Distance from `center` to the outer surface of `face`, i.e. how deep a cutter aimed at
    that face has to run to break out. Never negative; `+1` of slack for rounding."""
    _u, _v, n = F.face_axes(face)
    axis = 0 if abs(n[0]) > 0.5 else (1 if abs(n[1]) > 0.5 else 2)
    surface = outer[1][axis] if n[axis] > 0 else outer[0][axis]
    return abs(surface - float(center[axis])) + 1.0


#: How far a through-cutter overshoots the wall it pierces, on BOTH sides, mm. Matches the
#: `+1` slack in `_outward_depth`: generous next to any wall a printer can make, small enough
#: that it cannot reach the opposite wall of any cavity worth enclosing.
_THROUGH_OVERSHOOT = 1.0


def _face_thickness(face: str, style: str, wall: float, floor: float, lid_t: float) -> float:
    """Thickness of the material a through-feature on `face` has to pierce.

    The floor and the lid are not the walls: `enclosure.floor` is set independently of
    `enclosure.wall`, and an open_frame has *only* a floor - its plate is `floor` thick
    whichever face of it a feature is declared on (top or bottom).
    """
    if face == "bottom":
        return floor
    if face in _TOP_FACES:
        return floor if style == "open_frame" else lid_t
    return wall


def _material_box(style: str, outer: Box, floor: float) -> Box:
    """The box whose faces are the part's real outer surfaces.

    For every walled style that is `outer`. An open_frame is a plate `floor` thick at the
    bottom of `outer`; its top face is the plate's top, not the lid plane `outer` reserves for
    the components above it - a top-face cutter placed on `outer` would cut only air.
    """
    if style != "open_frame":
        return outer
    return (outer[0], (outer[1][0], outer[1][1], outer[0][2] + floor))


def _surface_point(face: str, center: Sequence[float], surface_box: Box) -> Vec3:
    """`center` projected along the face normal onto the outer surface of `face`."""
    _u, _v, n = F.face_axes(face)
    axis = 0 if abs(n[0]) > 0.5 else (1 if abs(n[1]) > 0.5 else 2)
    out = [float(center[0]), float(center[1]), float(center[2])]
    out[axis] = surface_box[1][axis] if n[axis] > 0 else surface_box[0][axis]
    return (out[0], out[1], out[2])


# --------------------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------------------


def synthesize(
    placement: Any,
    enclosure_spec: Optional[Dict[str, Any]],
    manufacturing: Optional[Dict[str, Any]] = None,
    *,
    components_by_ref: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "",
) -> EnclosureParts:
    """Synthesise an enclosure around `placement`.

    `enclosure_spec` is the IR `enclosure` block (schema defaults may already be applied; every
    field is re-defaulted here so the function is usable from a bare dict or `{}`).
    `manufacturing` supplies `min_wall` / `printer_bed`, which can only *raise* a wall, never
    lower one. `components_by_ref` carries the IR component dicts — `mount`, `clearance` and
    `thermal_w` live there, not on the envelope.
    """
    spec = dict(enclosure_spec or {})
    mfg = dict(manufacturing or {})
    comps = dict(components_by_ref or {})
    notes: List[str] = []

    style = spec.get("style", "tray_lid")
    if style not in STYLES:
        notes.append("unknown style %r; falling back to tray_lid" % (style,))
        style = "tray_lid"

    wall = _num(spec, "wall", 2.0)
    min_wall = _num(mfg, "min_wall", 0.0)
    if min_wall > wall:
        notes.append("wall raised %.2f -> %.2f to satisfy manufacturing.min_wall" % (wall, min_wall))
        wall = min_wall
    floor = _num(spec, "floor", wall) if spec.get("floor") is not None else wall
    if min_wall > floor:
        floor = min_wall
    clearance = _num(spec, "clearance", 1.5)
    corner_r = _num(spec, "corner_radius", 3.0)
    edge_r = _num(spec, "edge_radius", 1.0)

    lid_spec = dict(spec.get("lid") or {})
    fastening = lid_spec.get("fastening", "screws")
    lip_h = _num(lid_spec, "lip", 1.5)
    screw = lid_spec.get("screw", "m3")
    screw_count = int(lid_spec.get("screw_count", 4) or 4)
    countersunk = bool(lid_spec.get("countersunk", True))

    if F.USING_FALLBACK_OPS:
        notes.append("phimech.csg.ops unavailable; using features.py fallback (chamfers, not fillets)")

    placed_items = _iter_placed(placement)
    if not placed_items:
        raise ValueError("cannot synthesise an enclosure around zero components")

    # -- 1. cavity ---------------------------------------------------------------------
    keepout: Optional[Box] = None
    floor_z_candidates: List[float] = []
    for p in placed_items:
        comp = comps.get(p.ref, {})
        c = _num(comp, "clearance", 1.0)
        if "clearance" in (p.envelope.meta or {}):
            c = 0.0  # place.py already grew this envelope; growing again would double-count
        keepout = _union_box(keepout, _grow(p.world_bbox(), c, c, c, c))
        mount = _mount_of(comp)
        if mount["style"] in ("standoff", "pocket"):
            floor_z_candidates.append(F.mount_plane_z(p) - mount["standoff_h"])
    assert keepout is not None

    cav_min, cav_max = _grow(keepout, clearance, clearance, 0.0, clearance)
    cav_floor_z = min([cav_min[2]] + floor_z_candidates)
    cav_min = (cav_min[0], cav_min[1], cav_floor_z)
    cavity: Box = (cav_min, cav_max)

    # Corner screw posts need floor area that no component keep-out describes. Widening the
    # cavity here — before any wall exists — is what keeps step 4 from dropping a pillar
    # through a PCB. Reserving the space is the only honest option: the alternative is a post
    # that collides, and a collision is not something a later gate can fix.
    if fastening == "screws" and STYLE_PARTS[style][1] is not None:
        cavity = _reserve_screw_corners(cavity, keepout, clearance, screw, corner_r, wall, notes)
    cav_min, cav_max = cavity

    # -- 2. outer shell ----------------------------------------------------------------
    lid_t = wall
    outer: Box = (
        (cav_min[0] - wall, cav_min[1] - wall, cav_min[2] - floor),
        (cav_max[0] + wall, cav_max[1] + wall, cav_max[2] + lid_t),
    )
    outer_w, outer_d, outer_h = _size_of(outer)
    r_in = max(0.0, corner_r - wall)

    floor_name, top_name = STYLE_PARTS[style]
    parts: Dict[str, _Part] = {floor_name: _Part(floor_name)}
    if top_name:
        parts[top_name] = _Part(top_name)

    split_z = outer[0][2] + outer_h / 2.0  # clamshell parting plane
    _build_shell(style, parts, floor_name, top_name, outer, cavity, wall, floor, lid_t,
                 corner_r, edge_r, r_in, lip_h, split_z, notes)

    result = EnclosureParts(
        inner_cavity=cavity,
        outer_bbox=outer,
        style=style,
        wall=wall,
        floor=floor,
        fastening=fastening,
        notes=notes,
    )

    # -- 3. component mounts -----------------------------------------------------------
    floor_part = parts[floor_name]
    if style != "sleeve":
        _add_mounts(placed_items, comps, floor_part, cavity, wall, result)
    else:
        notes.append("sleeve style: component mounts are the user's problem (no floor to boss)")

    # -- 4. lid fastening --------------------------------------------------------------
    if top_name is not None:
        _add_fastening(
            fastening, parts[floor_name], parts[top_name], cavity, outer, wall, lid_t,
            lip_h, screw, screw_count, countersunk, corner_r, style, split_z, result
        )
    elif fastening not in ("none",):
        notes.append("style %r has no separate lid; `lid.fastening=%s` ignored" % (style, fastening))

    # -- 5. access cutouts + explicit features -----------------------------------------
    _add_access_cutouts(placed_items, parts, floor_name, top_name, outer, result,
                        style=style, wall=wall, floor=floor, lid_t=lid_t)
    _add_spec_features(spec, parts, floor_name, top_name, outer, cavity, wall, floor, result,
                       style=style, lid_t=lid_t)

    # -- 6. auto vents -----------------------------------------------------------------
    _add_auto_vents(spec, placed_items, comps, parts, floor_name, outer, cavity, wall, result,
                    floor=floor)

    # -- 7. text -----------------------------------------------------------------------
    _add_text(spec, parts, floor_name, top_name, outer, title, wall, lid_t, result,
              style=style, floor=floor)

    # -- 8. zero the datum -------------------------------------------------------------
    offset: Vec3 = (0.0, 0.0, -outer[0][2])
    result.world_offset = offset
    result.parts = {name: _shift(p.build(), offset) for name, p in parts.items()}
    result.component_bodies = {p.ref: _shift(_body_of(p), offset) for p in placed_items}
    result.inner_cavity = _shift_box(cavity, offset)
    result.outer_bbox = _shift_box(outer, offset)
    for rec in result.standoff_positions:
        rec["z0"] = rec["z0"] + offset[2]
    for rec in result.screw_positions:
        rec["z0"] = rec["z0"] + offset[2]
    for cut in result.cutouts:
        cut["center"] = tuple(cut["center"][i] + offset[i] for i in range(3))
        if cut.get("surface_center") is not None:
            cut["surface_center"] = tuple(cut["surface_center"][i] + offset[i] for i in range(3))

    _check_bed(mfg, result)
    return result


# --------------------------------------------------------------------------------------
# 2. shell
# --------------------------------------------------------------------------------------


def _rbox_at(box: Box, r: float, edge_r: float = 0.0) -> Node:
    w, d, h = _size_of(box)
    return Translate(box[0], [ops.rounded_box(w, d, h, r, edge_r=edge_r)])


def _build_shell(
    style: str,
    parts: Dict[str, _Part],
    floor_name: str,
    top_name: Optional[str],
    outer: Box,
    cavity: Box,
    wall: float,
    floor: float,
    lid_t: float,
    corner_r: float,
    edge_r: float,
    r_in: float,
    lip_h: float,
    split_z: float,
    notes: List[str],
) -> None:
    """Build the bare walls/floor/lid of the chosen style. Everything else is bolted on later."""
    (ox0, oy0, oz0), (ox1, oy1, oz1) = outer
    (cx0, cy0, cz0), (cx1, cy1, cz1) = cavity

    if style == "open_frame":
        plate: Box = ((ox0, oy0, oz0), (ox1, oy1, oz0 + floor))
        parts[floor_name].add(_rbox_at(plate, corner_r, edge_r))
        notes.append("open_frame: no walls, so no cutouts or vents are generated for side faces")
        return

    if style == "tray_lid":
        tray: Box = ((ox0, oy0, oz0), (ox1, oy1, cz1))
        # over-cut upward by 2*wall: the cavity must break the rim, never graze it
        cut: Box = ((cx0, cy0, cz0), (cx1, cy1, cz1 + 2 * wall))
        parts[floor_name].add(Difference([_rbox_at(tray, corner_r, edge_r), _rbox_at(cut, r_in)]))
        lid_box: Box = ((ox0, oy0, cz1), (ox1, oy1, cz1 + lid_t))
        assert top_name is not None
        parts[top_name].add(_rbox_at(lid_box, corner_r, edge_r))
        parts[top_name].add(F.lip_ring(cavity, wall, lip_h, 0.15, corner_radius=corner_r))
        return

    body = Difference([_rbox_at(outer, corner_r, edge_r), _rbox_at(cavity, r_in)])

    if style == "shell":
        parts[floor_name].add(body)
        notes.append("shell: one sealed body — slice it before printing; it has no opening")
        return

    if style == "sleeve":
        span = ((ox0 - wall - EPS, cy0, cz0), (ox1 + wall + EPS, cy1, cz1))
        through = Difference([_rbox_at(outer, corner_r, edge_r), _rbox_at(span, r_in)])
        parts[floor_name].add(through)
        notes.append("sleeve: open at both X ends; the assembly slides in along X")
        return

    # clamshell: cut the sealed body at mid Z and give the lower half a mating lip
    assert top_name is not None
    pad_w, pad_d = (ox1 - ox0) + 2 * EPS, (oy1 - oy0) + 2 * EPS
    hi_pad = Cube((pad_w, pad_d, (oz1 - split_z) + 2 * EPS), False)
    lo_pad = Cube((pad_w, pad_d, (split_z - oz0) + EPS), False)
    lower = Difference([body, Translate((ox0 - EPS, oy0 - EPS, split_z), [hi_pad])])
    upper = Difference([body, Translate((ox0 - EPS, oy0 - EPS, oz0 - EPS), [lo_pad])])
    male = F.lip_ring(
        ((cx0, cy0, cz0), (cx1, cy1, split_z + lip_h)), wall, lip_h, 0.15, corner_radius=corner_r
    )
    female = F.lip_ring(
        ((cx0, cy0, cz0), (cx1, cy1, split_z + lip_h + EPS)),
        wall, lip_h + EPS, -0.15, corner_radius=corner_r,
        thickness=max(0.8, wall * 0.5) + 0.3,
    )
    parts[floor_name].add(Union([lower, male]))
    parts[top_name].add(upper)
    parts[top_name].cut(female)
    notes.append("clamshell: parting plane at z=%.2f (mid height); lip on the lower half" % split_z)



# --------------------------------------------------------------------------------------
# 3. component mounts
# --------------------------------------------------------------------------------------


#: How far a snap-hook root is buried in the floor so the union fuses (mm). Half the floor at
#: most, so a thin floor is never pierced.
_CLIP_ROOT_SINK = 0.6


def _add_mounts(
    placed_items: List[Placed],
    comps: Dict[str, Dict[str, Any]],
    floor_part: _Part,
    cavity: Box,
    wall: float,
    result: EnclosureParts,
) -> None:
    cav_floor = cavity[0][2]
    for p in placed_items:
        comp = comps.get(p.ref, {})
        mount = _mount_of(comp)
        style = mount["style"]
        if style == "standoff":
            holes = _holes_of(p, mount)
            if not holes:
                result.notes.append(
                    "%s: mount.style=standoff but no holes known; falling back to pocket ledges" % p.ref
                )
                style = "pocket"
            else:
                plane = F.mount_plane_z(p)
                h = plane - cav_floor
                if h <= 0.2:
                    result.notes.append("%s: mount plane sits on the floor; standoffs skipped" % p.ref)
                    continue
                bore_style = "heat-set" if mount["fastener"].startswith("heat-set") else (
                    "through" if mount["through"] else "self-tap"
                )
                for hole in holes:
                    xy = F.hole_to_world(p, hole)
                    node, rec = F.boss_for_hole(
                        xy, cav_floor, h, mount["fastener"], bore_style, wall=wall, hole_d=hole.d
                    )
                    floor_part.add(node)
                    rec["ref"] = p.ref
                    rec["hole_ref"] = hole.ref
                    result.standoff_positions.append(rec)
                continue

        mn, mx = p.world_bbox()
        fp: Box = ((mn[0], mn[1], cav_floor), (mx[0], mx[1], cav_floor))
        if style == "pocket":
            h = max(0.8, F.mount_plane_z(p) - cav_floor)
            for ledge in F.pocket_ledges(fp, cav_floor, h):
                floor_part.add(ledge)
        elif style == "clip":
            long_x = (mx[0] - mn[0]) >= (mx[1] - mn[1])
            h = max(2.0, F.mount_plane_z(p) - cav_floor)
            cx, cy = (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0
            if long_x:
                spots = [((cx, mn[1] - 0.3), "back"), ((cx, mx[1] + 0.3), "front")]
            else:
                spots = [((mn[0] - 0.3, cy), "right"), ((mx[0] + 0.3, cy), "left")]
            # Sink the hook root INTO the floor rather than standing it exactly on the floor plane.
            # Rooted at cav_floor a hook meets the plate across a zero-thickness coplanar face,
            # which is the one union geometry the the project architecture notes call out as unreliable. This is
            # defensive, not a fix for a reproduced failure: the 3-solid manifold failure that
            # prompted it turned out to be a stale library envelope, and the case still passes with
            # the sink at 0. Kept because a real overlap makes the fusion unambiguous for every
            # backend at zero volume cost - the extra length is buried inside the plate.
            sink = min(_CLIP_ROOT_SINK, max(0.2, cav_floor * 0.5))
            for (hx, hy), face in spots:
                floor_part.add(
                    F.snap_hook((hx, hy, cav_floor - sink), face, height=h + 2.0 + sink)
                )
        elif style == "slot":
            plane = F.mount_plane_z(p)
            thick = max(0.8, p.envelope.mount_thickness or 1.6)
            h = max(3.0, plane - cav_floor + thick + 1.5)
            for rail in F.rail_pair(fp, cav_floor, h, plane - cav_floor, thick):
                floor_part.add(rail)
        elif style in ("adhesive", "none"):
            result.notes.append("%s: mount.style=%s — nothing printed to hold it" % (p.ref, style))


# --------------------------------------------------------------------------------------
# 4. lid fastening
# --------------------------------------------------------------------------------------


def _screw_inset(screw: str, corner_r: float, wall: float) -> float:
    """How far a post centre sits in from the cavity wall: enough for its own radius plus a
    sliver of clearance, and never inside the shell's corner fillet."""
    _size, row, _heat = F.fastener_spec(screw)
    return max(row["od"] / 2.0 + 0.2, corner_r - wall)


def _reserve_screw_corners(
    cavity: Box,
    keepout: Box,
    clearance: float,
    screw: str,
    corner_r: float,
    wall: float,
    notes: List[str],
) -> Box:
    """Widen `cavity` along one axis until the four corner posts clear every keep-out.

    A post clears if it is outside the keep-out rectangle on *either* axis, so growing one
    axis is enough and growing the axis with the smaller perpendicular extent adds the least
    material. Returns the cavity unchanged when the posts already fit.
    """
    _size, row, _heat = F.fastener_spec(screw)
    need = _screw_inset(screw, corner_r, wall) + row["od"] / 2.0
    grow = need - clearance
    if grow <= 0.0:
        return cavity
    (x0, y0, z0), (x1, y1, z1) = cavity
    if (y1 - y0) <= (x1 - x0):  # grow X: its perpendicular extent (depth) is the smaller one
        notes.append(
            "cavity widened %.2f mm on each X side so the %s corner posts clear the components"
            % (grow, screw)
        )
        return ((x0 - grow, y0, z0), (x1 + grow, y1, z1))
    notes.append(
        "cavity widened %.2f mm on each Y side so the %s corner posts clear the components"
        % (grow, screw)
    )
    return ((x0, y0 - grow, z0), (x1, y1 + grow, z1))


def _corner_posts(cavity: Box, inset: float, count: int) -> List[Vec2]:
    """`count` post centres inside the cavity: the four corners first (that is where a lid
    needs the clamping force), then midpoints of the long edges for higher counts."""
    (x0, y0, _z0), (x1, y1, _z1) = cavity
    ax0, ay0, ax1, ay1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    if count <= 2:
        cy = (ay0 + ay1) / 2.0
        return [(ax0, cy), (ax1, cy)]
    corners = [(ax0, ay0), (ax1, ay0), (ax1, ay1), (ax0, ay1)]
    if count == 3:
        return corners[:2] + [((ax0 + ax1) / 2.0, ay1)]
    out = list(corners)
    extra = count - 4
    if extra > 0:
        along_x = (ax1 - ax0) >= (ay1 - ay0)
        per_side = (extra + 1) // 2
        for i in range(per_side):
            t = (i + 1.0) / (per_side + 1.0)
            if along_x:
                out.append((ax0 + t * (ax1 - ax0), ay0))
                if len(out) < count:
                    out.append((ax0 + t * (ax1 - ax0), ay1))
            else:
                out.append((ax0, ay0 + t * (ay1 - ay0)))
                if len(out) < count:
                    out.append((ax1, ay0 + t * (ay1 - ay0)))
    return out[:count]


def _add_fastening(
    fastening: str,
    base: _Part,
    lid: _Part,
    cavity: Box,
    outer: Box,
    wall: float,
    lid_t: float,
    lip_h: float,
    screw: str,
    screw_count: int,
    countersunk: bool,
    corner_r: float,
    style: str,
    split_z: float,
    result: EnclosureParts,
) -> None:
    (cx0, cy0, cz0), (cx1, cy1, cz1) = cavity
    top_z = cz1 if style == "tray_lid" else split_z
    post_h = top_z - cz0

    if fastening == "screws":
        _size, row, _heat = F.fastener_spec(screw)
        inset = _screw_inset(screw, corner_r, wall)
        for xy in _corner_posts(cavity, inset, max(2, screw_count)):
            node, rec = F.lid_screw_post(xy, cz0, post_h, screw, wall=wall)
            base.add(node)
            rec["screw"] = screw
            result.screw_positions.append(rec)
            if countersunk:
                lid.cut(F.countersink_cutter(xy, top_z, lid_t, screw))
            else:
                lid.cut(F.cutout_node("top", (xy[0], xy[1], top_z + lid_t / 2.0),
                                      (row["clearance_d"], row["clearance_d"]), "round",
                                      lid_t + 2.0, 0.0, span="both"))
        result.notes.append(
            "%d %s lid screws; posts run the full cavity height — check them against components"
            % (len(result.screw_positions), screw)
        )
        return

    if fastening == "snap":
        long_x = (cx1 - cx0) >= (cy1 - cy0)
        q = 0.25  # hooks at the quarter points: far enough from the corners to actually flex
        if long_x:
            xs = [cx0 + (cx1 - cx0) * q, cx0 + (cx1 - cx0) * (1 - q)]
            spots = [((x, cy0), "back") for x in xs] + [((x, cy1), "front") for x in xs]
        else:
            ys = [cy0 + (cy1 - cy0) * q, cy0 + (cy1 - cy0) * (1 - q)]
            spots = [((cx0, y), "right") for y in ys] + [((cx1, y), "left") for y in ys]
        hook_h = min(6.0, max(3.0, post_h * 0.5))
        for (hx, hy), face in spots:
            base.add(F.snap_hook((hx, hy, top_z - hook_h), face, height=hook_h))
            lid.cut(F.snap_recess((hx, hy, top_z - hook_h + hook_h - 2.4), face))
        result.notes.append("4 snap hooks on the base rim with matching recesses in the lid lip")
        return

    if fastening == "magnets":
        inset = _MAGNET_D / 2.0 + max(1.6, wall * 0.8)
        for xy in _corner_posts(cavity, inset, 4):
            od = _MAGNET_D + 2 * max(1.6, wall * 0.8)
            base.add(Group([Translate((xy[0], xy[1], cz0), [ops.cyl(od, post_h)])], name="magnet_post"))
            base.cut(F.magnet_pocket(xy, top_z - _MAGNET_H))
            lid.cut(F.magnet_pocket(xy, top_z))
            result.screw_positions.append(
                {"at": xy, "z0": top_z - _MAGNET_H, "h": _MAGNET_H, "od": od,
                 "hole_d": _MAGNET_D, "screw": "magnet-d6x2"}
            )
        result.notes.append("4 d%.1f x %.1f magnet pockets in both parts — glue the magnets in"
                            % (_MAGNET_D, _MAGNET_H))
        return

    if fastening == "friction":
        result.notes.append("friction fit: lip clearance tightened to 0.05 mm — expect to sand it")
        return
    if fastening == "hinge":
        result.notes.append("hinge fastening is not synthesised yet; lid is loose (lip only)")
        return
    result.notes.append("lid.fastening=none: the lid is held by the alignment lip alone")


# --------------------------------------------------------------------------------------
# 5. cutouts
# --------------------------------------------------------------------------------------


def _target_part(face: str, through_lid: bool, floor_name: str, top_name: Optional[str]) -> List[str]:
    """Which parts a cutter on `face` has to be differenced from."""
    if face in _TOP_FACES or through_lid:
        return [top_name] if top_name else [floor_name]
    if face == "bottom":
        return [floor_name]
    if top_name and top_name in ("upper",):  # clamshell: a side feature can straddle the split
        return [floor_name, top_name]
    return [floor_name]


def _add_access_cutouts(
    placed_items: List[Placed],
    parts: Dict[str, _Part],
    floor_name: str,
    top_name: Optional[str],
    outer: Box,
    result: EnclosureParts,
    *,
    style: str = "tray_lid",
    wall: float = 2.0,
    floor: float = 2.0,
    lid_t: float = 2.0,
) -> None:
    # `outer`, not `_material_box()`: a component's access opening faces away from the
    # component, so on an open_frame its top-facing opening looks up into free air above the
    # lid plane - the plate is behind it and must not be what the opening is judged against.
    for p in placed_items:
        for a in p.envelope.access:
            face, center, size = F.component_face_to_world(p, a)
            # From the feature out past the outer surface: always longer than the wall.
            depth = _outward_depth(face, center, outer)
            node = F.cutout_node(face, center, size, a.shape, depth, a.tolerance)
            targets = _target_part(face, a.through_lid, floor_name, top_name)
            for name in targets:
                if name in parts:
                    parts[name].cut(node)
            result.cutouts.append(
                {
                    "ref": p.ref,
                    "name": a.name or (p.ref + ":" + a.face),
                    "kind": "access",
                    "face": face,
                    "center": center,
                    "size": (round(size[0], 3), round(size[1], 3)),
                    "shape": a.shape,
                    "tolerance": a.tolerance,
                    "depth": round(depth, 3),
                    "parts": [n for n in targets if n in parts],
                    # what the `cutout_through` gate needs to prove the opening is open
                    "through": True,
                    "wall_t": round(_face_thickness(face, style, wall, floor, lid_t), 3),
                    "surface_center": _surface_point(face, center, outer),
                }
            )


def _add_spec_features(
    spec: Dict[str, Any],
    parts: Dict[str, _Part],
    floor_name: str,
    top_name: Optional[str],
    outer: Box,
    cavity: Box,
    wall: float,
    floor: float,
    result: EnclosureParts,
    *,
    style: str = "tray_lid",
    lid_t: Optional[float] = None,
) -> None:
    """Enclosure-level features from `enclosure.features[]` — the ones not tied to a component."""
    lid_t = wall if lid_t is None else lid_t
    surface_box = _material_box(style, outer, floor)
    for feat in spec.get("features") or []:
        kind = feat.get("kind")
        face = feat.get("face") or ("bottom" if kind in ("din_clip", "foot") else "top")
        params = dict(feat.get("params") or {})
        at = feat.get("at")
        size = feat.get("size")
        depth = _num(feat, "depth", wall) if feat.get("depth") is not None else wall
        targets = [n for n in _target_part(face, False, floor_name, top_name) if n in parts]

        if kind == "vent":
            fw, fh = F.face_extent(face, surface_box[0], surface_box[1])
            region = None
            if at is not None or size is not None:
                cu, cv = (at[0], at[1]) if at else (fw / 2.0, fh / 2.0)
                sw, sh = (size[0], size[1]) if size else (fw, fh)
                region = ((cu, cv), (sw, sh))
            vent_depth = _face_thickness(face, style, wall, floor, lid_t) + _THROUGH_OVERSHOOT
            node = F.vent_pattern(
                face, surface_box, _num(feat, "slot_w", 1.5), _num(feat, "pitch", 4.0),
                vent_depth + _THROUGH_OVERSHOOT, max(2.0, _num(spec, "corner_radius", 3.0)),
                region=region,
            )
            if node is None:
                result.notes.append("vent feature on the %s face does not fit; skipped" % face)
                continue
            for n in targets:
                parts[n].cut(_inward(face, node, vent_depth))
            result.cutouts.append({"kind": "vent", "face": face,
                                   "center": _face_point(face, surface_box, at),
                                   "size": (0.0, 0.0), "shape": "slot", "parts": targets})
        elif kind in ("cutout", "cable_gland"):
            if kind == "cable_gland":
                d = _num(params, "d", _GLAND_D.get(str(params.get("thread", "pg7")).lower(), 12.5))
                sz: Vec2 = (d, d)
                shape = "round"
            else:
                sz = (float(size[0]), float(size[1])) if size else (10.0, 10.0)
                shape = feat.get("shape", "rect")
            # `center` is on the outer surface of the material the feature is declared on, so a
            # cutter centred there (`span="both"`) reaches half its length inward. It must be
            # sized from THAT face's thickness - the floor and an open_frame plate are not
            # `wall` thick - plus overshoot, or a thick floor is left with a blind hole.
            center = _face_point(face, surface_box, at)
            thick = _face_thickness(face, style, wall, floor, lid_t)
            node = F.cutout_node(face, center, sz, shape, 2.0 * (thick + _THROUGH_OVERSHOOT),
                                 _num(feat, "tolerance", 0.0), span="both")
            for n in targets:
                parts[n].cut(node)
            result.cutouts.append({"kind": kind, "face": face, "center": center,
                                   "size": (sz[0], sz[1]), "shape": shape, "parts": targets,
                                   "tolerance": _num(feat, "tolerance", 0.0),
                                   "through": True, "wall_t": round(thick, 3),
                                   "surface_center": center})
        elif kind == "boss":
            xy = _face_point(face, outer, at)
            h = _num(params, "h", 6.0)
            node, rec = F.lid_screw_post((xy[0], xy[1]), cavity[0][2], h,
                                         params.get("fastener", "m3"), wall=wall)
            parts[floor_name].add(node)
            rec["screw"] = params.get("fastener", "m3")
            rec["kind"] = "feature_boss"
            result.standoff_positions.append(rec)
        elif kind == "text":
            emboss = bool(params.get("emboss", False))
            node = F.text_on_face(face, outer, feat.get("text") or "", _num(feat, "size", 6.0),
                                  depth, emboss)
            if node is None:
                result.notes.append("text feature on the %s face has no text; skipped" % face)
                continue
            for n in targets:
                parts[n].add(node) if emboss else parts[n].cut(node)
        elif kind == "wall_mount_tab":
            tab_face = feat.get("face") or "back"
            fw, fh = F.face_extent(tab_face, outer[0], outer[1])
            if style == "open_frame":
                # No wall to stand a tab on: the outer box's mid-height is air above the plate,
                # where a tab floats as a separate body. Extend the plate with ears instead.
                for frac in (0.25, 0.75):
                    pt = _face_point(tab_face, surface_box, (fw * frac, 0.0))
                    parts[floor_name].add(F.keyhole_ear(pt, tab_face, thickness=floor))
                result.notes.append("2 keyhole mounting ears on the %s edge of the plate"
                                    % tab_face)
                continue
            for frac in (0.25, 0.75):
                pt = _face_point(tab_face, outer, (fw * frac, fh * 0.5))
                parts[floor_name].add(F.keyhole_tab(pt, tab_face))
            result.notes.append("2 keyhole wall-mount tabs on the %s face" % tab_face)
        elif kind == "din_clip":
            cx = (outer[0][0] + outer[1][0]) / 2.0
            cy = (outer[0][1] + outer[1][1]) / 2.0
            width = min(_num(params, "width", 40.0), _size_of(outer)[0] - 4.0)
            parts[floor_name].add(F.din_clip((cx, cy, outer[0][2]), width=max(10.0, width)))
            result.notes.append("DIN rail clip pad on the bottom (35 mm rail)")
        elif kind == "foot":
            count = int(feat.get("count") or 4)
            d = _num(params, "d", _FOOT_D)
            inset = d / 2.0 + max(1.5, wall)
            for xy in _corner_posts(outer, inset, max(3, count)):
                parts[floor_name].cut(F.foot_recess(xy, outer[0][2], d, _num(params, "h", _FOOT_H)))
            result.notes.append("%d rubber-foot recesses (d%.1f x %.1f) in the outer floor"
                                % (max(3, count), d, _num(params, "h", _FOOT_H)))
        else:
            result.notes.append("feature kind %r not implemented; skipped" % (kind,))


def _inward(face: str, node: Node, depth: float) -> Node:
    """`vent_pattern` builds its slots along the outward normal from a point on the outer
    surface; shift them back by `depth` so they cut *into* the wall instead of into air."""
    _u, _v, n = F.face_axes(face)
    return Translate((-n[0] * depth, -n[1] * depth, -n[2] * depth), [node])


# --------------------------------------------------------------------------------------
# 6. auto vents
# --------------------------------------------------------------------------------------


def _add_auto_vents(
    spec: Dict[str, Any],
    placed_items: List[Placed],
    comps: Dict[str, Dict[str, Any]],
    parts: Dict[str, _Part],
    floor_name: str,
    outer: Box,
    cavity: Box,
    wall: float,
    result: EnclosureParts,
    *,
    floor: Optional[float] = None,
) -> None:
    vents = dict(spec.get("vents") or {})
    if not vents.get("auto", True):
        return
    total_w = 0.0
    for p in placed_items:
        comp = comps.get(p.ref, {})
        total_w += _num(comp, "thermal_w", 0.0) or p.envelope.thermal_w
    if total_w <= 0.0:
        return
    faces = list(vents.get("faces") or ("left", "right"))
    slot_w = _num(vents, "slot_w", 1.5)
    pitch = _num(vents, "pitch", 4.0)
    margin = max(3.0, _num(spec, "corner_radius", 3.0) + 1.0, wall + 1.0)
    # vents live on the walls, which belong to the floor part in every style that has walls
    wall_box: Box = ((outer[0][0], outer[0][1], outer[0][2]), (outer[1][0], outer[1][1], cavity[1][2]))
    made = 0
    for face in faces:
        if face in ("top", "bottom") and face not in ("bottom",):
            result.notes.append("auto vents on the %s face are skipped (that is the lid)" % face)
            continue
        # a bottom vent pierces the floor, which is `floor` thick, not `wall`
        face_depth = (floor if face == "bottom" and floor is not None else wall) + 1.0
        node = F.vent_pattern(face, wall_box, slot_w, pitch, face_depth, margin)
        if node is None:
            result.notes.append("auto vents: the %s face is too small for %.1f mm slots" % (face, slot_w))
            continue
        parts[floor_name].cut(_inward(face, node, face_depth))
        result.cutouts.append(
            {
                "kind": "vent",
                "face": face,
                "center": F.face_center(face, wall_box[0], wall_box[1]),
                "size": tuple(round(v, 3) for v in F.face_extent(face, wall_box[0], wall_box[1])),
                "shape": "slot",
                "parts": [floor_name],
                "auto": True,
            }
        )
        made += 1
    if made:
        result.notes.append(
            "auto vents: %.1f W dissipated, slot vents (%.1f mm / %.1f pitch) on %s"
            % (total_w, slot_w, pitch, ", ".join(faces))
        )


# --------------------------------------------------------------------------------------
# 7. text
# --------------------------------------------------------------------------------------


def _add_text(
    spec: Dict[str, Any],
    parts: Dict[str, _Part],
    floor_name: str,
    top_name: Optional[str],
    outer: Box,
    title: str,
    wall: float,
    lid_t: float,
    result: EnclosureParts,
    *,
    style: str = "tray_lid",
    floor: Optional[float] = None,
) -> None:
    tspec = dict(spec.get("text") or {})
    label = tspec.get("label") or title
    if not label:
        return
    size = _num(tspec, "size", 6.0)
    depth = _num(tspec, "depth", 0.6)
    emboss = bool(tspec.get("emboss", False))
    target = top_name if top_name else floor_name
    lid_w = _size_of(outer)[0]
    # 0.8 * size is the fallback glyph advance; keep the whole string inside 80 % of the target
    est_w = 0.8 * size * max(1, len(label))
    limit = 0.8 * lid_w
    if est_w > limit:
        size = max(2.0, size * limit / est_w)
        result.notes.append(
            "label text scaled down to %.1f mm to fit the %s" % (size, target)
        )
    # the plate of an open_frame is `floor` thick; every other label sits on a lid/top wall
    face_t = floor if (style == "open_frame" and floor is not None) else lid_t
    if not emboss and depth >= face_t - 0.4:
        depth = max(0.2, face_t - 0.6)
        result.notes.append(
            "label depth clipped to %.2f mm to keep the %s solid" % (depth, target)
        )
    # The label must sit on the TARGET part's own top surface. `outer` is the whole enclosure
    # envelope, whose top is the lid plane; on a lidless style (open_frame/plate) that plane is
    # tens of mm above the part, so the cutter missed the geometry entirely while the report
    # still claimed a label had been debossed. Retarget the box to the part being labelled.
    face_box = outer
    if style == "open_frame" and floor is not None:
        # The plate's top, `floor` above the bed. Falling through to the `lid_t` slab below
        # put the label 0.6 mm *inside* any plate thicker than `wall`: a buried void, no label.
        face_box = _material_box(style, outer, floor)
    elif top_name is None:
        part_box = getattr(parts.get(target), "bbox", None)
        if callable(part_box):
            try:
                part_box = part_box()
            except Exception:
                part_box = None
        if part_box is not None:
            face_box = part_box
        else:
            (x0, y0, z0), (x1, y1, _z1) = outer
            face_box = ((x0, y0, z0), (x1, y1, z0 + max(lid_t, 0.1)))
    spot = _label_spot(face_box, label, size, target, style, result.cutouts)
    if spot is None:
        result.notes.append(
            "label %r skipped: every spot on the %s top face is taken by a through-cutout"
            % (label, target)
        )
        # recorded so `text_legible` reports the asked-for label as absent, not as 6 mm
        result.labels.append({"label": label, "part": target, "face": "top",
                              "size": round(size, 3), "rect": None, "skipped": True})
        return
    center_v, rect = spot
    fw, fh = F.face_extent("top", face_box[0], face_box[1])
    if abs(center_v - fh / 2.0) > 1e-9:
        result.notes.append("label moved off-centre (v=%.1f of %.1f mm) to clear a cutout"
                            % (center_v, fh))
    node = F.text_on_face("top", face_box, label, size, depth, emboss,
                          center=_face_point("top", face_box, (fw / 2.0, center_v)))
    if node is None:
        return
    if emboss:
        parts[target].add(node)
    else:
        parts[target].cut(node)
    result.text_sizes.append(round(size, 3))
    result.labels.append({"label": label, "part": target, "face": "top", "size": round(size, 3),
                          "rect": tuple(round(v, 3) for v in rect)})
    result.notes.append("label %r %s on the %s top face" % (label, "embossed" if emboss else "debossed", target))


#: Text footprint, as multiples of the glyph size: 0.8 per character is the fallback advance
#: the fit check above uses; 1.6 covers ascender + descender with `valign="center"` (measured
#: 1.59 x size on "thru-repro" in Liberation Sans Bold).
_LABEL_ADVANCE = 0.8
_LABEL_HEIGHT = 1.6
#: Clear air kept between a label and a cutout edge, mm.
_LABEL_CLEARANCE = 1.0


def label_rect(
    center_xy: Sequence[float], label: str, size: float
) -> Tuple[float, float, float, float]:
    """Conservative XY footprint `(x0, y0, x1, y1)` of a centred top-face label."""
    w = _LABEL_ADVANCE * size * max(1, len(label))
    h = _LABEL_HEIGHT * size
    cx, cy = float(center_xy[0]), float(center_xy[1])
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def through_rects_under_top(cutouts: Sequence[Dict[str, Any]], part: str, style: str
                            ) -> List[Tuple[float, float, float, float]]:
    """XY rects of the through-cutouts that open a part's top face: those declared on its top
    and, on an open_frame (one plate), those declared on its bottom too."""
    faces = ("top", "bottom") if style == "open_frame" else ("top",)
    rects = []
    for c in cutouts:
        if c.get("face") not in faces or part not in (c.get("parts") or []):
            continue
        if c.get("kind") == "vent" and not any(c.get("size") or ()):
            continue  # a vent pattern records no footprint
        sx, sy = c["size"][0] / 2.0, c["size"][1] / 2.0
        grow = float(c.get("tolerance") or 0.0)
        cx, cy = c["center"][0], c["center"][1]
        rects.append((cx - sx - grow, cy - sy - grow, cx + sx + grow, cy + sy + grow))
    return rects


def _label_spot(face_box: Box, label: str, size: float, part: str, style: str,
                cutouts: Sequence[Dict[str, Any]]) -> Optional[Tuple[float, Tuple[float, ...]]]:
    """A `v` on the top face where the label clears every through-cutout, preferring the centre,
    and its footprint; None when there is none. Cutouts are cut before text, so a label placed
    over a hole is simply cut away while the report still claims it."""
    rects = through_rects_under_top(cutouts, part, style)
    fw, fh = F.face_extent("top", face_box[0], face_box[1])
    half_h = _LABEL_HEIGHT * size / 2.0
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8, 0.0, 1.0):  # 0 and 1 clamp to the edge margin
        v = min(max(fh * frac, half_h + _LABEL_CLEARANCE), fh - half_h - _LABEL_CLEARANCE)
        pt = _face_point("top", face_box, (fw / 2.0, v))
        rect = label_rect(pt, label, size)
        c = _LABEL_CLEARANCE
        if all(rect[2] + c <= r[0] or r[2] + c <= rect[0] or rect[3] + c <= r[1]
               or r[3] + c <= rect[1] for r in rects):
            return v, rect
    return None


# --------------------------------------------------------------------------------------
# 8. gates that need no backend
# --------------------------------------------------------------------------------------


def _check_bed(mfg: Dict[str, Any], result: EnclosureParts) -> None:
    bed = mfg.get("printer_bed")
    if not bed or len(bed) < 3:
        return
    ow, od, oh = result.outer_size
    bx, by, bz = float(bed[0]), float(bed[1]), float(bed[2])
    fits = (ow <= bx and od <= by) or (od <= bx and ow <= by)
    if not fits or oh > bz:
        result.notes.append(
            "WARNING: %.1f x %.1f x %.1f mm exceeds printer_bed %.0f x %.0f x %.0f — split it"
            % (ow, od, oh, bx, by, bz)
        )


__all__ = ["EnclosureParts", "STYLES", "STYLE_PARTS", "synthesize"]
