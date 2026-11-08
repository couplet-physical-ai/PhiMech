"""Face frames and feature geometry — the vocabulary `enclosure.py` builds walls out of.

Two responsibilities live here, and nothing else:

1. **Frames.** A component declares an `Access` feature as "on my `front` face, at (u, v)".
   The enclosure has to cut a hole in *its* wall at the corresponding world point. Every
   mapping between a face name, a 2D coordinate on that face, and assembly space is done by
   `face_frame()` / `component_face_to_world()` so the convention exists exactly once.
2. **Feature solids.** Cutters (cutouts, vents, countersinks, magnet pockets, foot recesses)
   and additive features (standoffs, screw posts, snap hooks, lips, tabs) as CSG `Node`s,
   already positioned in assembly space. `enclosure.py` unions the additive ones into a part
   and differences the cutters out of it; it never computes trigonometry itself.

Why the frames are spelled out instead of using a matrix library: `phimech/synth` is
stdlib-only (python 3.9, FreeCAD's bundled python, the MCP venv) and every rotation in the
placement model is a multiple of 90°, so exact integer arithmetic beats a general transform.

Face convention (matches `components/envelope.py`)
-------------------------------------------------
A face frame is `(origin, u, v, n)`: `origin` is the 3D point that a face-local `(0, 0)` maps
to, `u`/`v` are unit vectors spanning the face, `n` is the **outward** normal. So
`world = origin + u * at[0] + v * at[1]`. `u`, `v` are chosen so that `(0, 0)` is always the
face's *lower-left corner when looking at the face from outside* — i.e. what a human drawing
a panel layout expects::

              +Z                          looking at the FRONT face (-Y) from outside:
               |    back (+Y)
               |   /                          v (+Z)
        +------|--/-------+                   ^
        |      | /        |                   |   (u,v) = (0,0) is at the
   left |      |/         | right             |   bottom-left corner
   (-X) |      +----------|--- +X             +--------> u (+X)
        |     /           |
        |    /  front     |               top face (+Z), seen from above:
        +---/---(-Y)------+                   v (+Y)
           /                                  ^
        bottom (-Z)                           |
                                              +--------> u (+X)

    face     origin                       u     v     n
    -------  ---------------------------   ----  ----  ----
    top      (min.x, min.y, max.z)         +X    +Y    +Z
    bottom   (min.x, min.y, min.z)         +X    +Y    -Z
    front    (min.x, min.y, min.z)         +X    +Z    -Y
    back     (max.x, max.y, min.z)         -X    +Z    +Y
    left     (min.x, max.y, min.z)         -Y    +Z    -X
    right    (max.x, min.y, min.z)         +Y    +Z    +X

`bottom` is the one frame whose `(u, v, n)` is left-handed (there is no rotation that maps the
axes onto it). Every cross-section this module orients onto a face is symmetric about both
face axes, so orienting cutters uses `FACE_ROT`, which matches `u` and `n` exactly and may
flip `v` — see `FACE_ROT` for the derivation.

stdlib only; must import under python 3.9.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phimech.components.envelope import Access, Hole, Placed
from phimech.csg.ast import (
    Cube,
    Cylinder,
    Difference,
    Group,
    Hull,
    LinearExtrude,
    Node,
    Polygon,
    Rotate,
    Translate,
    Union,
)

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

FACES = ("top", "bottom", "left", "right", "front", "back")

#: Over-cut used everywhere a difference would otherwise leave a co-planar, paper-thin skin.
EPS = 0.01

# --------------------------------------------------------------------------------------
# fastener tables
# --------------------------------------------------------------------------------------

#: Boss/standoff geometry per fastener size. `pilot_d` is the self-tapping hole (the screw
#: cuts its own thread), `clearance_d` is a through hole, `head_d`/`head_depth` size a
#: countersink for a DIN 7991-ish cap. `od` is the recommended boss outside diameter.
FASTENERS: Dict[str, Dict[str, float]] = {
    "m2": {"od": 5.0, "pilot_d": 1.7, "clearance_d": 2.4, "head_d": 4.0, "head_depth": 1.2},
    "m2.5": {"od": 6.0, "pilot_d": 2.2, "clearance_d": 2.9, "head_d": 5.0, "head_depth": 1.5},
    "m3": {"od": 7.0, "pilot_d": 2.5, "clearance_d": 3.4, "head_d": 6.0, "head_depth": 1.8},
    "m4": {"od": 9.0, "pilot_d": 3.3, "clearance_d": 4.5, "head_d": 8.0, "head_depth": 2.3},
}

#: Heat-set (brass) insert pockets — a straight bore, sized so the insert melts in flush.
HEAT_SET: Dict[str, Dict[str, float]] = {
    "m2": {"hole_d": 3.2, "depth": 4.0, "od": 5.6},
    "m2.5": {"hole_d": 3.6, "depth": 4.5, "od": 6.4},
    "m3": {"hole_d": 4.0, "depth": 5.0, "od": 7.0},
    "m4": {"hole_d": 5.6, "depth": 6.0, "od": 9.6},
}


def fastener_spec(fastener: str) -> Tuple[str, Dict[str, float], bool]:
    """Normalise an IR fastener enum to `(size_key, table_row, is_heat_set)`.

    The IR allows `m2 | m2.5 | m3 | m4 | heat-set-m3 | self-tap`; downstream code only wants
    to know "which size" and "is it an insert pocket".
    """
    f = (fastener or "m3").strip().lower()
    heat = f.startswith("heat-set")
    if heat:
        size = f[len("heat-set") :].lstrip("-") or "m3"
    elif f == "self-tap":
        size = "m3"
    else:
        size = f
    if size not in FASTENERS:
        size = "m3"
    return size, FASTENERS[size], heat


# --------------------------------------------------------------------------------------
# ops: prefer phimech.csg.ops, fall back to a private minimal implementation
# --------------------------------------------------------------------------------------

try:  # `phimech.csg.ops` is the shared helper layer; synth must survive without it too
    from phimech.csg import ops as _csg_ops  # type: ignore
except ImportError:  # pragma: no cover - only while csg/ops.py is absent
    _csg_ops = None

#: True when `phimech.csg.ops` could not be imported and `_Ops` is building from bare AST
#: primitives instead. Surfaced in `EnclosureParts.summary()` notes so a report never
#: silently claims ops-quality rounding.
USING_FALLBACK_OPS = _csg_ops is None


class _Ops(object):
    """Geometry helpers in the frame `features.py`/`enclosure.py` want.

    `phimech.csg.ops` is delegated to wherever its contract matches; this class exists
    because three of its conventions do not match what an enclosure needs, and adapting once
    here is safer than adapting at twenty call sites:

    * **Origin.** `ops.slot()` and `ops.vent_slots()` place geometry with a corner (or a
      margin offset) at the origin; every cutter here is oriented by rotating about the
      origin, so it must be *centred* there or the rotation moves it off the face.
    * **Axis.** `ops.vent_slots()` lays slots out horizontally along +X. Vents on a side wall
      have to be vertical (long axis along the face's `v`) so the wall keeps its hoop
      strength and prints without bridging.
    * **Failure mode.** `ops.vent_slots()` raises when the face is too small and
      `ops.heat_set_boss()` raises when the boss is too short. Synth must degrade to "no
      vents on this face" and carry a note, never abort a build.

    It also avoids `ops.cut_through()` / `ops.tube()` for bores: those wrap the cutter in a
    `Minkowski` with a sphere, which is correct but multiplies CGAL time per boss. Enclosures
    have dozens of bosses, so bores here over-cut by translating and lengthening instead.
    """

    EPS = EPS
    HEAT_SET = HEAT_SET

    delegates = _csg_ops

    # primitives ------------------------------------------------------------------------

    @staticmethod
    def box(w: float, d: float, h: float, *, center: bool = False) -> Node:
        """Corner at the origin (or centred), growing +X/+Y/+Z."""
        return Cube((float(w), float(d), float(h)), bool(center))

    @staticmethod
    def cyl(d: float, h: float, *, center: bool = False, segments: Optional[int] = None) -> Node:
        """Axis on +Z, base at z = 0 (or centred on the origin)."""
        return Cylinder(float(h), float(d) / 2.0, None, bool(center), segments)

    @classmethod
    def bore(cls, node: Node, d: float, z0: float, h: float) -> Node:
        """Drill a +Z hole of diameter `d` through `node`, over-cut by EPS at both ends."""
        if d <= 0.0:
            return node
        cutter = Translate((0.0, 0.0, z0 - EPS), [cls.cyl(d, h + 2 * EPS)])
        return Difference([node, cutter])

    @classmethod
    def tube(cls, od: float, id: float, h: float) -> Node:  # noqa: A002 - mirrors ops' name
        return cls.bore(cls.cyl(od, h), id, 0.0, h)

    # 2D / rounded ------------------------------------------------------------------------

    @classmethod
    def rounded_rect(cls, w: float, d: float, r: float) -> Node:
        if cls.delegates is not None:
            return cls.delegates.rounded_rect(w, d, r)
        return Polygon(tuple(_rounded_rect_points(w, d, r)))

    @classmethod
    def rounded_box(cls, w: float, d: float, h: float, r: float, *, edge_r: float = 0.0) -> Node:
        if min(float(w), float(d), float(h)) <= 0.0:
            return cls.box(max(w, EPS), max(d, EPS), max(h, EPS))
        if cls.delegates is not None:
            return cls.delegates.rounded_box(w, d, h, r, edge_r=edge_r)
        r = max(0.0, min(float(r), w / 2.0, d / 2.0))
        if r <= 0.0:
            return cls.box(w, d, h)
        return LinearExtrude(float(h), [cls.rounded_rect(w, d, r)])

    @classmethod
    def slot(cls, length: float, width: float, h: float) -> Node:
        """Stadium prism **centred on the origin in XY**, long axis along X, base at z = 0."""
        length, width = float(length), float(width)
        if length < width:
            length, width = width, length
        node = (
            cls.delegates.slot(length, width, h)
            if cls.delegates is not None
            else LinearExtrude(float(h), [cls.rounded_rect(length, width, width / 2.0)])
        )
        return Translate((-length / 2.0, -width / 2.0, 0.0), [node])

    # fastener features -------------------------------------------------------------------

    @classmethod
    def countersunk_hole(cls, d: float, h: float, head_d: float, head_depth: float) -> Node:
        """Cutter: shaft from z = 0 to z = h with the cone opening upward at the top.

        Built here rather than delegated because a cutter needs the EPS over-cut past both
        faces of the plate and a disc above the cone — otherwise the screw head bottoms out
        on a co-planar remnant of the lid's outer skin.
        """
        h, head_depth = float(h), max(0.0, float(head_depth))
        shaft = Translate((0.0, 0.0, -EPS), [cls.cyl(d, h + 2 * EPS)])
        if head_depth <= 0.0:
            return shaft
        cone = Translate(
            (0.0, 0.0, h - head_depth),
            [Cylinder(head_depth, float(d) / 2.0, float(head_d) / 2.0, False, None)],
        )
        cap = Translate((0.0, 0.0, h - EPS), [cls.cyl(head_d, head_depth + EPS)])
        return Union([shaft, cone, cap])

    @classmethod
    def screw_boss(cls, od: float, id: float, h: float, *, fillet_r: float = 0.0) -> Node:  # noqa: A002
        """Hollow boss with a conical skirt at its root.

        Not delegated: `ops.screw_boss()` hulls a wide base disc with the full-height shaft,
        and because a hull is convex that taper runs the *whole* height — a 14 mm post comes
        out as a 14 mm cone, which eats the cavity and prints a needle-tipped boss. The skirt
        here is bounded to `fillet_r` tall, which is what the argument means.
        """
        h = max(float(h), EPS)
        od = float(od)
        fillet = max(0.0, min(float(fillet_r), h / 3.0))
        shaft = cls.cyl(od, h)
        if fillet <= 0.0:
            return cls.bore(shaft, id, 0.0, h)
        skirt = Cylinder(fillet, od / 2.0 + fillet, od / 2.0, False, None)
        return cls.bore(Union([shaft, skirt]), id, 0.0, h)

    @classmethod
    def standoff(cls, h: float, od: float, hole_d: float, *, base_d: Optional[float] = None) -> Node:
        """Hollow standoff with an optional conical flange at the floor.

        `ops.standoff()` stacks a straight `base_d` disc, which leaves a 90° re-entrant step
        that concentrates stress exactly where a standoff snaps off the floor. A cone spreads
        the load and prints without a bridge.
        """
        h = max(float(h), EPS)
        od = float(od)
        outer: Node = cls.cyl(od, h)
        if base_d is not None and float(base_d) > od:
            flange_h = min(1.0, h / 2.0)
            outer = Union([outer, Cylinder(flange_h, float(base_d) / 2.0, od / 2.0, False, None)])
        return cls.bore(outer, hole_d, 0.0, h)

    @classmethod
    def heat_set_boss(cls, insert: str, h: float) -> Node:
        row = HEAT_SET.get(insert, HEAT_SET["m3"])
        return cls.standoff(h, row["od"], row["hole_d"])

    # patterns ---------------------------------------------------------------------------

    @classmethod
    def vent_slots(
        cls,
        face_w: float,
        face_h: float,
        slot_w: float,
        pitch: float,
        depth: float,
        *,
        margin: float = 3.0,
    ) -> Optional[Node]:
        """Vertical stadium slots filling a `face_w x face_h` region **centred on the origin**,
        cutting along +Z from `-EPS` to `depth + EPS`. Returns None when nothing fits — a face
        too small to vent is a note, not an error."""
        usable_w = float(face_w) - 2.0 * float(margin)
        usable_h = float(face_h) - 2.0 * float(margin)
        slot_w, pitch = float(slot_w), max(float(pitch), float(slot_w) + 0.6)
        if usable_w <= slot_w or usable_h <= slot_w or slot_w <= 0.0:
            return None
        count = int((usable_w - slot_w) // pitch) + 1
        if count < 1:
            return None
        span = (count - 1) * pitch
        length = float(depth) + 2 * EPS
        cutters: List[Node] = []
        for i in range(count):
            x = -span / 2.0 + i * pitch
            vertical = Rotate((0.0, 0.0, 90.0), [cls.slot(usable_h, slot_w, length)])
            cutters.append(Translate((x, 0.0, -EPS), [vertical]))
        return Union(cutters)

    @classmethod
    def text_emboss(
        cls,
        text: str,
        size: float,
        depth: float,
        *,
        font: str = "Liberation Sans:style=Bold",
        halign: str = "center",
    ) -> Node:
        """Text as the AST's `text:<json>` group convention, which backends expand.

        `ops.text_emboss()` returns an *empty* group, so on its own it contributes no solid
        and a debossed label would silently do nothing. Placeholder glyph bars are added as
        children here so the label is real geometry in every backend, and any backend that
        understands the `text:` payload can discard them and typeset properly.
        """
        payload = (
            cls.delegates.text_emboss(text, size, depth, font=font, halign=halign)
            if cls.delegates is not None
            else Group([], name="text:" + str(text))
        )
        return Group(_glyph_bars(text, size, depth, halign), name=payload.name)

    # combinators -------------------------------------------------------------------------

    @staticmethod
    def union_all(nodes: Sequence[Node]) -> Node:
        kept = [n for n in nodes if n is not None]
        if not kept:
            raise ValueError("union_all requires at least one node")
        return kept[0] if len(kept) == 1 else Union(list(kept))

    @staticmethod
    def diff_all(base: Node, cutters: Sequence[Node]) -> Node:
        kept = [c for c in cutters if c is not None]
        return base if not kept else Difference([base] + kept)

    @staticmethod
    def place(node: Node, at: Sequence[float], rot: Sequence[float] = (0.0, 0.0, 0.0)) -> Node:
        inner = node if all(abs(float(a)) < 1e-12 for a in rot) else Rotate(_v3(rot), [node])
        return Translate(_v3(at), [inner]) if any(abs(float(a)) > 1e-12 for a in at) else inner

    @classmethod
    def offset_polygon(cls, points: Sequence[Vec2], delta: float) -> List[Vec2]:
        if cls.delegates is not None:
            return [(float(p[0]), float(p[1])) for p in cls.delegates.offset_polygon(points, delta)]
        return _offset_polygon(points, delta)


#: The geometry namespace synth builds with. Delegates to `phimech.csg.ops` where the
#: conventions agree; see `_Ops` for the three places they do not.
ops = _Ops()


def _glyph_bars(text: str, size: float, depth: float, halign: str = "center") -> List[Node]:
    """One bar per glyph, extruded +Z by `depth`, centred on the origin in XY.

    Indicative geometry only — it makes a label a real solid (so a deboss actually removes
    material and volume gates see the change) without the AST needing a font engine.
    """
    chars = list(text or "")
    if not chars:
        return []
    size, depth = float(size), float(depth)
    cw, gap = 0.62 * size, 0.18 * size
    adv = cw + gap
    total = adv * len(chars) - gap
    x0 = {"left": 0.0, "right": -total, "center": -total / 2.0}.get(halign, -total / 2.0)
    bars: List[Node] = []
    for i, c in enumerate(chars):
        if c.isspace():
            continue
        bars.append(Translate((x0 + i * adv, -size / 2.0, 0.0), [Cube((cw, size, depth), False)]))
    return bars


# --------------------------------------------------------------------------------------
# small vector / polygon helpers
# --------------------------------------------------------------------------------------


def _v3(v: Sequence[float]) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Sequence[float], k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def rotate_vec90(v: Sequence[float], rot: Sequence[float]) -> Vec3:
    """Rotate a vector by Euler angles restricted to multiples of 90° (X, then Y, then Z).

    Same maths as `Placed.world_bbox()`; exact, so face normals stay exactly axis-aligned
    and `face_of_normal()` never has to guess.
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    for axis, deg in enumerate(rot):
        if abs(deg / 90.0 - round(deg / 90.0)) > 1e-9:
            raise ValueError("rotation %r is not a multiple of 90 degrees" % (tuple(rot),))
        for _ in range(int(round(deg / 90.0)) % 4):
            if axis == 0:
                y, z = -z, y
            elif axis == 1:
                x, z = z, -x
            else:
                x, y = -y, x
    return (float(x), float(y), float(z))


def _rounded_rect_points(w: float, d: float, r: float, segments: int = 8) -> List[Vec2]:
    """CCW outline of a `w × d` rectangle with corner radius `r`, corner at the origin."""
    w, d = float(w), float(d)
    r = max(0.0, min(float(r), w / 2.0, d / 2.0))
    if r <= 1e-9:
        return [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]
    pts: List[Vec2] = []
    for cx, cy, a0 in ((w - r, d - r, 0.0), (r, d - r, 90.0), (r, r, 180.0), (w - r, r, 270.0)):
        for i in range(segments + 1):
            a = math.radians(a0 + 90.0 * i / segments)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _offset_polygon(points: Sequence[Vec2], delta: float) -> List[Vec2]:
    """Naive vertex-bisector offset — adequate for the convex keep-out outlines synth uses."""
    n = len(points)
    if n < 3:
        return list(points)
    out: List[Vec2] = []
    for i in range(n):
        px, py = points[(i - 1) % n]
        cx, cy = points[i]
        nx, ny = points[(i + 1) % n]
        e1 = _norm2((cx - px, cy - py))
        e2 = _norm2((nx - cx, ny - cy))
        n1 = (e1[1], -e1[0])
        n2 = (e2[1], -e2[0])
        bx, by = n1[0] + n2[0], n1[1] + n2[1]
        length = math.hypot(bx, by)
        if length < 1e-9:
            out.append((cx, cy))
            continue
        scale = delta * math.sqrt(2.0 / max(1e-9, 1.0 + _dot2(n1, n2)))
        out.append((cx + bx / length * scale, cy + by / length * scale))
    return out


def _norm2(v: Vec2) -> Vec2:
    length = math.hypot(v[0], v[1])
    return (v[0] / length, v[1] / length) if length > 1e-12 else (0.0, 0.0)


def _dot2(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------

#: Unit `(u, v, n)` of every face frame. See the module docstring for the picture.
FACE_AXES: Dict[str, Tuple[Vec3, Vec3, Vec3]] = {
    "top": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "bottom": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "front": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "back": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "left": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
    "right": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
}

#: Euler rotation (degrees, X→Y→Z) that maps local +X onto the face's `u` and local +Z onto
#: the face's outward normal. `v` is matched too except on `bottom`, where the frame is
#: left-handed and `v` comes out flipped — harmless because every cross-section oriented with
#: this table is symmetric about both face axes.
FACE_ROT: Dict[str, Vec3] = {
    "top": (0.0, 0.0, 0.0),
    "bottom": (180.0, 0.0, 0.0),
    "front": (90.0, 0.0, 0.0),
    "back": (90.0, 0.0, 180.0),
    "left": (90.0, 0.0, 270.0),
    "right": (90.0, 0.0, 90.0),
}

#: Euler rotation for features that must stand *up* (world +Z) while facing outward: local +Y
#: becomes the face's outward normal and local +Z stays world +Z. Used by snap hooks, keyhole
#: tabs and other wall-mounted parts that a printer sees as vertical.
WALL_ROT: Dict[str, Vec3] = {
    "front": (0.0, 0.0, 180.0),
    "back": (0.0, 0.0, 0.0),
    "left": (0.0, 0.0, 90.0),
    "right": (0.0, 0.0, 270.0),
    "top": (90.0, 0.0, 0.0),
    "bottom": (270.0, 0.0, 0.0),
}


def face_axes(face: str) -> Tuple[Vec3, Vec3, Vec3]:
    """`(u, v, n)` unit vectors of a face frame."""
    try:
        return FACE_AXES[face]
    except KeyError:
        raise ValueError("unknown face %r; expected one of %s" % (face, ", ".join(FACES))) from None


def face_frame(face: str, bbox_min: Sequence[float], bbox_max: Sequence[float]) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    """Return `(origin, u, v, normal)` for `face` of the axis-aligned box `bbox_min..bbox_max`.

    A face-local `(u, v)` maps to `origin + u * u_vec + v * v_vec`. See the module docstring
    for the ASCII picture and the full table.
    """
    u, v, n = face_axes(face)
    x0, y0, z0 = float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])
    x1, y1, z1 = float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])
    origins: Dict[str, Vec3] = {
        "top": (x0, y0, z1),
        "bottom": (x0, y0, z0),
        "front": (x0, y0, z0),
        "back": (x1, y1, z0),
        "left": (x0, y1, z0),
        "right": (x1, y0, z0),
    }
    return origins[face], u, v, n


def face_extent(face: str, bbox_min: Sequence[float], bbox_max: Sequence[float]) -> Vec2:
    """Size of a face in its own `(u, v)` frame — what a vent or text pattern has to fit in."""
    u, v, _ = face_axes(face)
    size = tuple(float(bbox_max[i]) - float(bbox_min[i]) for i in range(3))
    return (
        abs(u[0]) * size[0] + abs(u[1]) * size[1] + abs(u[2]) * size[2],
        abs(v[0]) * size[0] + abs(v[1]) * size[1] + abs(v[2]) * size[2],
    )


def face_center(face: str, bbox_min: Sequence[float], bbox_max: Sequence[float]) -> Vec3:
    """Centre point of a face of the box, in assembly space."""
    origin, u, v, _ = face_frame(face, bbox_min, bbox_max)
    w, h = face_extent(face, bbox_min, bbox_max)
    return _add(origin, _add(_mul(u, w / 2.0), _mul(v, h / 2.0)))


def face_of_normal(n: Sequence[float]) -> str:
    """Face name whose outward normal is (closest to) `n`. Input is expected axis-aligned."""
    best, best_dot = "top", -2.0
    for face, (_u, _v, nn) in FACE_AXES.items():
        d = _dot(n, nn)
        if d > best_dot:
            best, best_dot = face, d
    return best


def component_face_to_world(placed: Placed, access: Access) -> Tuple[str, Vec3, Vec2]:
    """Map a component-local `Access` onto the assembly.

    Returns `(world_face, center, size)`: which enclosure face the feature points at, the
    feature centre in assembly space, and its size expressed in *that* face's `(u, v)` frame
    (so a 90°-rotated component's 12×6 USB slot becomes 6×12 when it lands on a side wall).
    """
    env = placed.envelope
    local_origin, lu, lv, ln = face_frame(access.face, (0.0, 0.0, 0.0), env.size)
    local_pt = _add(local_origin, _add(_mul(lu, access.at[0]), _mul(lv, access.at[1])))
    center = _add(rotate_vec90(local_pt, placed.rotation), placed.position)
    world_n = rotate_vec90(ln, placed.rotation)
    world_face = face_of_normal(world_n)
    wu = rotate_vec90(lu, placed.rotation)
    wv = rotate_vec90(lv, placed.rotation)
    fu, fv, _fn = face_axes(world_face)
    su = abs(access.size[0] * _dot(wu, fu)) + abs(access.size[1] * _dot(wv, fu))
    sv = abs(access.size[0] * _dot(wu, fv)) + abs(access.size[1] * _dot(wv, fv))
    return world_face, center, (su, sv)


def hole_to_world(placed: Placed, hole: Hole) -> Vec3:
    """Assembly-space centre of a component mounting hole, on the component mounting face."""
    local = (hole.at[0], hole.at[1], placed.envelope.mount_plane_z)
    return _add(rotate_vec90(local, placed.rotation), placed.position)


def mount_plane_z(placed: Placed) -> float:
    """World Z of the component's rigid mounting surface (a PCB's bottom copper)."""
    mn, _mx = placed.world_bbox()
    return mn[2] + placed.envelope.mount_plane_z


# --------------------------------------------------------------------------------------
# cutters
# --------------------------------------------------------------------------------------


def _oriented(face: str, center: Sequence[float], local: Node) -> Node:
    """Place a cutter built along +Z (cross-section centred in XY) onto `face` at `center`."""
    return Translate(_v3(center), [Rotate(FACE_ROT[face], [local])])


def cutout_node(
    face: str,
    center: Sequence[float],
    size: Sequence[float],
    shape: str = "rect",
    depth: float = 4.0,
    tolerance: float = 0.5,
    *,
    span: str = "outward",
) -> Node:
    """A cutter for one opening on `face`, centred on `center` (a point on the feature).

    The prism runs along the face's **outward normal** from `-EPS` (just inside the feature)
    to `depth + EPS`, so it always breaks through the wall no matter how the wall thickness
    rounds; the cross-section is `size` grown by `tolerance` on every side. `span="both"`
    centres the prism on `center` instead — used when `center` is already on the outer surface
    and the cutter must reach inward as well.
    """
    su = max(0.1, float(size[0]) + 2 * float(tolerance))
    sv = max(0.1, float(size[1]) + 2 * float(tolerance))
    length = float(depth) + 2 * EPS
    z0 = -length / 2.0 if span == "both" else -EPS
    shape = (shape or "rect").lower()
    if shape == "round":
        local = Translate((0.0, 0.0, z0), [ops.cyl(max(su, sv), length)])
    elif shape == "slot":
        stadium = ops.slot(max(su, sv), min(su, sv), length)
        if sv > su:  # a tall slot: rotate the stadium so its long axis runs along v
            stadium = Rotate((0.0, 0.0, 90.0), [stadium])
        local = Translate((0.0, 0.0, z0), [stadium])
    else:
        local = Translate((-su / 2.0, -sv / 2.0, z0), [ops.box(su, sv, length)])
    return Group([_oriented(face, center, local)], name="cutout")


def vent_pattern(
    face: str,
    bbox: Tuple[Sequence[float], Sequence[float]],
    slot_w: float = 1.5,
    pitch: float = 4.0,
    depth: float = 4.0,
    margin: float = 3.0,
    *,
    region: Optional[Tuple[Vec2, Vec2]] = None,
) -> Optional[Node]:
    """Slot vents over a face of `bbox`, or None if the face is too small to take any.

    `region` optionally narrows the pattern to a `(center_uv, size_uv)` window on that face,
    which is how a hot component gets vented without perforating the whole wall.
    """
    origin, u, v, _n = face_frame(face, bbox[0], bbox[1])
    fw, fh = face_extent(face, bbox[0], bbox[1])
    if region is None:
        center = face_center(face, bbox[0], bbox[1])
        w, h = fw, fh
    else:
        (cu, cv), (w, h) = region
        center = _add(origin, _add(_mul(u, cu), _mul(v, cv)))
        w, h = min(w, fw), min(h, fh)
    slots = ops.vent_slots(w, h, slot_w, pitch, depth, margin=margin)
    if slots is None:
        return None
    return Group([_oriented(face, center, slots)], name="vents_" + face)


def text_on_face(
    face: str,
    bbox: Tuple[Sequence[float], Sequence[float]],
    text: str,
    size: float = 6.0,
    depth: float = 0.6,
    emboss: bool = False,
    *,
    center: Optional[Sequence[float]] = None,
) -> Optional[Node]:
    """Label centred on a face, or None for an empty string.

    Embossed labels sit *on* the face (the caller unions them in), debossed ones start `depth`
    below it (the caller differences them out) — the geometry differs only in how far along the
    face normal it starts.
    """
    if not (text or "").strip():
        return None
    if center is None:
        center = face_center(face, bbox[0], bbox[1])
    glyphs = ops.text_emboss(text, size, depth + (0.0 if emboss else EPS), halign="center")
    z0 = -EPS if emboss else -(depth + EPS)
    local = Translate((0.0, 0.0, z0), [glyphs])
    return Group([_oriented(face, center, local)], name=("text_emboss" if emboss else "text_deboss"))


def countersink_cutter(xy: Sequence[float], z0: float, thickness: float, screw: str = "m3") -> Node:
    """Through hole + countersink in a plate whose bottom is at `z0` and top at `z0+thickness`.
    The cone opens upward, so the screw head finishes flush with the lid's outer face."""
    _size, row, _heat = fastener_spec(screw)
    node = ops.countersunk_hole(row["clearance_d"], thickness, row["head_d"], row["head_depth"])
    return Group([Translate((float(xy[0]), float(xy[1]), z0), [node])], name="countersink")


def magnet_pocket(xy: Sequence[float], z0: float, d: float = 6.2, h: float = 2.2) -> Node:
    """Blind pocket for a disc magnet, opening upward from `z0`."""
    return Group(
        [Translate((float(xy[0]), float(xy[1]), z0 - EPS), [ops.cyl(d, h + EPS)])],
        name="magnet_pocket",
    )


def foot_recess(xy: Sequence[float], z0: float, d: float = 8.0, h: float = 1.0) -> Node:
    """Recess for a self-adhesive rubber foot in the outer floor (`z0` is the floor bottom)."""
    return Group(
        [Translate((float(xy[0]), float(xy[1]), z0 - EPS), [ops.cyl(d, h + EPS)])],
        name="foot_recess",
    )


# --------------------------------------------------------------------------------------
# additive features
# --------------------------------------------------------------------------------------


def boss_for_hole(
    hole_world_xy: Sequence[float],
    z0: float,
    h: float,
    fastener: str = "m3",
    style: str = "self-tap",
    *,
    wall: float = 2.0,
    hole_d: Optional[float] = None,
) -> Tuple[Node, Dict[str, Any]]:
    """A standoff under a component mounting hole, from `z0` up by `h`.

    `style` picks the bore: `self-tap` (screw cuts its own thread), `through` (clearance hole,
    nut or screw from below) or `heat-set` (insert pocket). Returns the node plus the record
    that ends up in `EnclosureParts.standoff_positions`, because gates want the numbers.

    For `through`, the bore is the *larger* of the component's own hole and the screw's
    clearance diameter: a boss narrower than the screw is unusable no matter what the PCB
    drilled, and a boss narrower than the PCB hole would not let the screw line up.
    """
    size, row, heat = fastener_spec(fastener)
    if style == "heat-set" or heat:
        insert = HEAT_SET[size]
        bore, od = insert["hole_d"], max(insert["od"], insert["hole_d"] + 2 * max(1.6, wall * 0.8))
        kind = "heat-set"
    elif style == "through":
        bore = max(float(hole_d or 0.0), row["clearance_d"])
        od = max(row["od"], bore + 2 * max(1.6, wall * 0.8))
        kind = "through"
    else:
        bore = row["pilot_d"]
        od = max(row["od"], bore + 2 * max(1.6, wall * 0.8))
        kind = "self-tap"
    h = max(h, 0.4)
    node = ops.standoff(h, od, bore, base_d=od + 1.6)
    x, y = float(hole_world_xy[0]), float(hole_world_xy[1])
    record: Dict[str, Any] = {
        "at": (x, y),
        "z0": float(z0),
        "h": float(h),
        "od": float(od),
        "hole_d": float(bore),
        "fastener": size,
        "style": kind,
    }
    return Group([Translate((x, y, float(z0)), [node])], name="standoff"), record


def lid_screw_post(
    xy: Sequence[float],
    z0: float,
    h: float,
    screw: str = "m3",
    *,
    wall: float = 2.0,
    heat_set: bool = False,
) -> Tuple[Node, Dict[str, Any]]:
    """A pillar in the base that the lid screw bites into, from `z0` up by `h`."""
    size, row, _heat = fastener_spec(screw)
    if heat_set:
        bore, od = HEAT_SET[size]["hole_d"], HEAT_SET[size]["od"]
    else:
        bore, od = row["pilot_d"], row["od"]
    od = max(od, bore + 2 * max(1.6, wall * 0.8))
    node = ops.screw_boss(od, bore, max(h, 0.4), fillet_r=min(1.5, wall))
    x, y = float(xy[0]), float(xy[1])
    record = {"at": (x, y), "z0": float(z0), "h": float(max(h, 0.4)), "od": float(od), "hole_d": float(bore)}
    return Group([Translate((x, y, float(z0)), [node])], name="screw_post"), record


def snap_hook(
    at: Sequence[float],
    face: str,
    *,
    width: float = 8.0,
    thickness: float = 1.6,
    height: float = 6.0,
    hook: float = 0.9,
) -> Node:
    """Cantilever snap tab standing up from `at` (its root, on the inner wall of `face`).

    Local frame: X along the wall, +Y outward (the barb direction), +Z up. The barb is a hull
    between the tab face and a thin lip so FDM prints it without a support-needing overhang.
    """
    tab = Translate((-width / 2.0, 0.0, 0.0), [ops.box(width, thickness, height)])
    barb_low = Translate((-width / 2.0, 0.0, height - 2.2), [ops.box(width, thickness + hook, EPS)])
    barb_tip = Translate((-width / 2.0, 0.0, height - 0.6), [ops.box(width, thickness, EPS)])
    local = Union([tab, Hull([barb_low, barb_tip])])
    return Group([Translate(_v3(at), [Rotate(WALL_ROT[face], [local])])], name="snap_hook")


def snap_recess(
    at: Sequence[float],
    face: str,
    *,
    width: float = 8.0,
    depth: float = 1.2,
    height: float = 2.4,
    tolerance: float = 0.2,
) -> Node:
    """The pocket in the lid lip that a `snap_hook` barb latches into."""
    w = width + 2 * tolerance
    local = Translate((-w / 2.0, -EPS, 0.0), [ops.box(w, depth + EPS, height + 2 * tolerance)])
    return Group([Translate(_v3(at), [Rotate(WALL_ROT[face], [local])])], name="snap_recess")


def lip_ring(
    inner_bbox: Tuple[Sequence[float], Sequence[float]],
    wall: float,
    lip_h: float,
    tolerance: float = 0.15,
    *,
    corner_radius: float = 0.0,
    thickness: Optional[float] = None,
) -> Node:
    """Alignment lip that drops from the top of `inner_bbox` down into the cavity by `lip_h`.

    Sized `tolerance` smaller than the cavity all round so the two parts actually close; the
    ring wall is half the enclosure wall by default, which keeps the lip printable without
    stealing usable cavity.
    """
    (x0, y0, _), (x1, y1, z1) = _v3(inner_bbox[0]), _v3(inner_bbox[1])
    t = float(thickness) if thickness else max(0.8, wall * 0.5)
    w = (x1 - x0) - 2 * tolerance
    d = (y1 - y0) - 2 * tolerance
    r_out = max(0.0, corner_radius - wall - tolerance)
    outer = Translate((x0 + tolerance, y0 + tolerance, z1 - lip_h), [ops.rounded_box(w, d, lip_h, r_out)])
    bore = Translate(
        (x0 + tolerance + t, y0 + tolerance + t, z1 - lip_h - EPS),
        [ops.rounded_box(w - 2 * t, d - 2 * t, lip_h + 2 * EPS, max(0.0, r_out - t))],
    )
    return Group([Difference([outer, bore])], name="lip")


def pocket_ledges(
    footprint_bbox: Tuple[Sequence[float], Sequence[float]],
    z0: float,
    h: float,
    *,
    ledge: float = 3.0,
    tolerance: float = 0.3,
) -> List[Node]:
    """Four corner ledges that a `pocket`-mounted component drops onto, `tolerance` clear of
    its keep-out. Cheaper to print than a full shelf and it lets debris fall through."""
    (x0, y0, _), (x1, y1, _) = _v3(footprint_bbox[0]), _v3(footprint_bbox[1])
    x0, y0, x1, y1 = x0 + tolerance, y0 + tolerance, x1 - tolerance, y1 - tolerance
    lx = min(ledge, max(0.5, (x1 - x0) / 3.0))
    ly = min(ledge, max(0.5, (y1 - y0) / 3.0))
    out: List[Node] = []
    for cx, cy in ((x0, y0), (x1 - lx, y0), (x0, y1 - ly), (x1 - lx, y1 - ly)):
        out.append(Group([Translate((cx, cy, float(z0)), [ops.box(lx, ly, h)])], name="pocket_ledge"))
    return out


def rail_pair(
    footprint_bbox: Tuple[Sequence[float], Sequence[float]],
    z0: float,
    h: float,
    groove_z: float,
    groove: float = 1.8,
    *,
    thickness: float = 1.6,
    tolerance: float = 0.3,
    engage: float = 1.2,
) -> List[Node]:
    """A card guide: two side rails, each with a horizontal groove the board slides into.

    The board is captured, not just fenced, so `engage` mm of each long edge has to sit *inside*
    the rail — the rails therefore overlap the footprint rather than standing clear of it, and
    the groove is `groove + 2 * tolerance` tall so a 1.6 mm PCB slides without binding. The
    board enters from an X end, which is why this pairs with the `sleeve` style.
    """
    (x0, y0, _), (x1, y1, _) = _v3(footprint_bbox[0]), _v3(footprint_bbox[1])
    length, z0 = x1 - x0, float(z0)
    gz0 = float(groove_z) - tolerance
    gh = float(groove) + 2 * tolerance
    out: List[Node] = []
    for inner_y, sign in ((y1 - engage, +1.0), (y0 + engage - thickness, -1.0)):
        rail = Translate((x0, inner_y, z0), [ops.box(length, thickness, h)])
        cut_y = inner_y - EPS if sign > 0 else inner_y + thickness - engage - EPS
        slot_cut = Translate(
            (x0 - EPS, cut_y, gz0), [ops.box(length + 2 * EPS, engage + EPS, gh)]
        )
        out.append(Group([Difference([rail, slot_cut])], name="rail"))
    return out


def keyhole_tab(
    at: Sequence[float],
    face: str,
    *,
    width: float = 16.0,
    height: float = 14.0,
    thickness: float = 3.0,
    screw_d: float = 4.2,
    head_d: float = 8.0,
) -> Node:
    """Protruding wall-mount tab with a keyhole slot, standing proud of `face` at `at`."""
    plate = Translate((-width / 2.0, 0.0, 0.0), [ops.rounded_box(width, thickness, height, 0.0)])
    big = Translate((0.0, -EPS, height * 0.30), [Rotate((270.0, 0.0, 0.0), [ops.cyl(head_d, thickness + 2 * EPS)])])
    slot = Translate(
        (0.0, -EPS, height * 0.62),
        [Rotate((270.0, 0.0, 0.0), [ops.slot(height * 0.5, screw_d, thickness + 2 * EPS)])],
    )
    local = Difference([plate, big, Rotate((0.0, 0.0, 0.0), [slot])])
    return Group([Translate(_v3(at), [Rotate(WALL_ROT[face], [local])])], name="wall_mount_tab")


def keyhole_ear(
    at: Sequence[float],
    face: str,
    *,
    width: float = 16.0,
    reach: float = 14.0,
    thickness: float = 3.0,
    embed: float = 1.0,
    screw_d: float = 4.2,
    head_d: float = 8.0,
) -> Node:
    """A keyhole mounting ear in the plane of a plate, for a part with no walls to hang a
    `keyhole_tab` on (an open_frame).

    `at` is a point on the plate's outer edge `face` at the plate's bottom. The ear is
    `thickness` thick (the plate's), runs `reach` outward and `embed` back into the plate so the
    union shares real volume with it rather than one face - a tab that only touches a rounded
    plate edge prints as a separate body. The keyhole is bored along Z, entirely in the part of
    the ear that stands clear of the plate.
    """
    ear = Translate((-width / 2.0, -embed, 0.0), [ops.box(width, reach + embed, thickness)])
    head_y = reach - head_d / 2.0 - 1.0
    head = Translate((0.0, head_y, -EPS), [ops.cyl(head_d, thickness + 2 * EPS)])
    # `ops.slot` occupies 0..len x 0..width; turned 90 deg it runs along +Y over -width..0 in X
    slot_y0 = 2.0  # clear of the plate edge
    slot_len = max(screw_d, head_y - slot_y0)
    slot = Translate(
        (screw_d / 2.0, slot_y0, -EPS),
        [Rotate((0.0, 0.0, 90.0), [ops.slot(slot_len, screw_d, thickness + 2 * EPS)])],
    )
    local = Difference([ear, head, slot])
    return Group([Translate(_v3(at), [Rotate(WALL_ROT[face], [local])])], name="wall_mount_tab")


def din_clip(at: Sequence[float], *, width: float = 40.0, depth: float = 12.0, height: float = 8.0) -> Node:
    """A slotted block on the outer floor that a 35 mm DIN rail clip screws onto."""
    block = Translate((-width / 2.0, -depth / 2.0, -height), [ops.box(width, depth, height)])
    slot_h = min(3.0, height * 0.5)
    groove = Translate(
        (-width / 2.0 - EPS, -depth / 2.0 + 2.0, -slot_h),
        [ops.box(width + 2 * EPS, depth - 4.0, slot_h + EPS)],
    )
    return Group([Difference([block, groove])], name="din_clip")


__all__ = [
    "EPS",
    "FACES",
    "FACE_AXES",
    "FACE_ROT",
    "WALL_ROT",
    "FASTENERS",
    "HEAT_SET",
    "USING_FALLBACK_OPS",
    "ops",
    "fastener_spec",
    "rotate_vec90",
    "face_axes",
    "face_frame",
    "face_extent",
    "face_center",
    "face_of_normal",
    "component_face_to_world",
    "hole_to_world",
    "mount_plane_z",
    "cutout_node",
    "vent_pattern",
    "text_on_face",
    "countersink_cutter",
    "magnet_pocket",
    "foot_recess",
    "boss_for_hole",
    "lid_screw_post",
    "snap_hook",
    "snap_recess",
    "lip_ring",
    "pocket_ledges",
    "rail_pair",
    "keyhole_ear",
    "keyhole_tab",
    "din_clip",
]
