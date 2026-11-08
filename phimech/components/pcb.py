"""KiCad/PhiCad board -> PhiMech `Envelope`: the bridge that makes enclosure synthesis real.

This is the most important envelope provider: for a board-driven design the board is the thing the
enclosure exists to hold, and every dimension the enclosure needs — outline, thickness, mounting
holes, the tallest part above and below the board, where the connectors are and which way they
face — is already in the `.kicad_pcb` that PhiCad emitted. Reading it is strictly better than
asking the agent to retype numbers, which is rule 4 (coordinates are never model output).

Frames
------
KiCad works in **millimetres with +Y pointing DOWN** (screen coordinates) and an arbitrary sheet
origin — the fixture board sits around x=100..158, y=75..143 because that is where it landed on
an A4 sheet. PhiMech's component-local frame (see `components/envelope.py`) has **+Y up**, its
origin at the **min-X/min-Y corner of the board outline**, and Z=0 at the mounting face. So every
point read from the file goes through exactly one conversion::

    px = kx - min_kx
    py = max_ky - ky          # the Y flip
    pz = mount_plane_z ...    # board bottom sits at mount_plane_z = below-board height

Consequences worth stating out loud, because they are the classic source of mirrored enclosures:

* A footprint near the **top** of the board as drawn in KiCad (small `ky`) ends up with a
  **large** `py` in PhiMech's frame. `tests/test_pcb.py::test_y_flip` pins this.
* A footprint rotation angle needs **no sign change**: KiCad's rotation is visually CCW in a
  Y-down frame, and flipping Y turns it into a mathematically CCW rotation in a Y-up frame. The
  two flips cancel. (`_fp_to_board` does the file-frame maths, `_flip` the global conversion.)
* Board bottom is at `Z = mount_plane_z`, board top at `mount_plane_z + thickness`. Bottom-side
  parts occupy `[0, mount_plane_z]`, top-side parts start at `mount_plane_z + thickness`.

What this module does *not* do: it never opens FreeCAD, never resolves `${KICAD10_3DMODEL_DIR}`,
and never guesses at copper. Heights come from `components/heights.py`; the model path is recorded
so a later pass can measure the handful of parts where a millimetre matters.

stdlib only; must import under python 3.9.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from phimech.components import heights as H
from phimech.components import sexpr as S
from phimech.components.envelope import Access, Envelope, Hole
from phimech.csg import ast

Vec2 = Tuple[float, float]
BBox2 = Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)

#: Endpoints closer than this are the same vertex when chaining Edge.Cuts segments (mm).
CHAIN_TOL = 0.01

#: Arcs are sampled into this many points (endpoints included) — enough for a 3D-printed
#: enclosure lip to follow a filleted board corner without exploding the polygon count.
ARC_SAMPLES = 8

#: A full circle on Edge.Cuts (round board) is sampled at this resolution.
CIRCLE_SAMPLES = 48

#: An NPTH pad at least this wide is treated as a mounting hole even without a MountingHole
#: footprint — below it is a via-in-pad / thermal / tooling hole, not something to boss up to.
MIN_NPTH_MOUNTING_D = 2.0

#: Slack added around a connector's pad row to size its enclosure opening (mm, each side).
OPENING_MARGIN = 1.0

#: Fallback board outline inflation when no closed Edge.Cuts loop exists (mm).
BBOX_FALLBACK_MARGIN = 1.0

BOARD_COLOR = (0.05, 0.45, 0.18, 1.0)
PART_COLOR = (0.18, 0.18, 0.20, 1.0)

EDGE_LAYER = "Edge.Cuts"

_GRAPHIC_HEADS = ("gr_line", "gr_rect", "gr_arc", "gr_poly", "gr_circle")
_FP_GRAPHIC_HEADS = ("fp_line", "fp_rect", "fp_arc", "fp_poly", "fp_circle")


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def _rot_in_file_frame(pt: Vec2, deg: float) -> Vec2:
    """Rotate a footprint-local point into board (file) coordinates.

    pcbnew's convention in its Y-down frame: `x' = x*cos + y*sin`, `y' = -x*sin + y*cos`, which
    reads as a visually counter-clockwise rotation for a positive angle.
    """
    if not deg:
        return (pt[0], pt[1])
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (pt[0] * c + pt[1] * s, -pt[0] * s + pt[1] * c)


def _fp_to_board(pt: Vec2, origin: Vec2, deg: float) -> Vec2:
    """Footprint-local point -> board (KiCad, Y-down) coordinates."""
    r = _rot_in_file_frame(pt, deg)
    return (origin[0] + r[0], origin[1] + r[1])


def _arc_points(start: Vec2, mid: Vec2, end: Vec2, n: int = ARC_SAMPLES) -> List[Vec2]:
    """Sample a KiCad 3-point arc into `n` points (start and end included).

    Falls back to the straight chord when the three points are collinear (a degenerate arc that
    KiCad tolerates and we must not divide by zero on).
    """
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [start, end]
    ux = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / d
    uy = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)
    # Walk start -> mid -> end the short way round each hop so the sampled arc passes through mid.
    def _delta(f: float, t: float) -> float:
        dd = (t - f) % (2 * math.pi)
        return dd - 2 * math.pi if dd > math.pi else dd

    # Sample each half (start->mid, mid->end) separately so the mid point is always one of the
    # samples. That matters: the mid is usually the arc's extreme, and a chord that cuts inside
    # it would under-report the board outline — an enclosure must never be smaller than the board.
    per_half = max(2, int(n) // 2 + 1)
    out: List[Vec2] = []
    for f, t in ((a1, a1 + _delta(a1, a2)), (a2, a2 + _delta(a2, a3))):
        for i in range(per_half):
            a = f + (t - f) * (i / float(per_half - 1))
            p = (ux + r * math.cos(a), uy + r * math.sin(a))
            if not out or not _same(out[-1], p, 1e-9):
                out.append(p)
    return out


def _same(a: Vec2, b: Vec2, tol: float = CHAIN_TOL) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _dedupe(points: Sequence[Vec2], tol: float = CHAIN_TOL) -> List[Vec2]:
    out: List[Vec2] = []
    for p in points:
        if not out or not _same(out[-1], p, tol):
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and _same(out[0], out[-1], tol):
        out.pop()
    return out


def _chain(pieces: Sequence[Sequence[Vec2]], tol: float = CHAIN_TOL) -> Optional[List[Vec2]]:
    """Chain polylines into one closed loop, or None if they do not close.

    Greedy: walk forward from the first piece, consuming whichever remaining piece starts or ends
    at the current tail, then do the same backwards from the head. Board outlines are simple
    closed curves, so greedy is exact and O(n^2) on a handful of segments.
    """
    remaining = [list(p) for p in pieces if len(p) >= 2]
    if not remaining:
        return None
    loop = remaining.pop(0)
    progressed = True
    while remaining and progressed:
        progressed = False
        tail, headp = loop[-1], loop[0]
        for i, piece in enumerate(remaining):
            if _same(piece[0], tail, tol):
                loop.extend(piece[1:])
            elif _same(piece[-1], tail, tol):
                loop.extend(list(reversed(piece))[1:])
            elif _same(piece[-1], headp, tol):
                loop = piece[:-1] + loop
            elif _same(piece[0], headp, tol):
                loop = list(reversed(piece))[:-1] + loop
            else:
                continue
            remaining.pop(i)
            progressed = True
            break
    if remaining:
        return None
    if len(loop) < 4 or not _same(loop[0], loop[-1], tol):
        return None
    return _dedupe(loop, tol)


def _signed_area(pts: Sequence[Vec2]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def _ccw(pts: Sequence[Vec2]) -> List[Vec2]:
    """Envelope footprints are CCW by contract; enforce it here once."""
    out = [(float(p[0]), float(p[1])) for p in pts]
    return out if _signed_area(out) >= 0 else list(reversed(out))


def _bbox(pts: Iterable[Vec2]) -> Optional[BBox2]:
    xs: List[float] = []
    ys: List[float] = []
    for p in pts:
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _is_axis_aligned_rect(pts: Sequence[Vec2], tol: float = CHAIN_TOL) -> bool:
    if len(pts) != 4:
        return False
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        if abs(x1 - x2) > tol and abs(y1 - y2) > tol:
            return False
    return True


# --------------------------------------------------------------------------------------
# reading pieces out of the tree
# --------------------------------------------------------------------------------------


def _pts_of(node: S.SExpr) -> List[Vec2]:
    """Points of a `(pts (xy x y) ...)` child form."""
    pts_form = S.find(node, "pts")
    if pts_form is None:
        return []
    out: List[Vec2] = []
    for xy in S.find_all(pts_form, "xy"):
        if len(xy) >= 3:
            out.append((float(xy[1]), float(xy[2])))
    return out


def _graphic_pieces(
    node: S.SExpr,
    heads: Sequence[str],
    layer: str = EDGE_LAYER,
    origin: Vec2 = (0.0, 0.0),
    rot: float = 0.0,
) -> List[List[Vec2]]:
    """Polylines contributed by graphic items on `layer`, in board (KiCad) coordinates.

    Used for both the top-level `gr_*` items and a footprint's `fp_*` items — mechanical parts
    from PhiCad routinely carry board cutouts inside a footprint, and ignoring those yields an
    outline that is silently too big.
    """
    pieces: List[List[Vec2]] = []
    for head in heads:
        for g in S.find_all(node, head):
            if S.layer_of(g) != layer:
                continue
            kind = head.split("_", 1)[1]
            if kind in ("line",):
                a = S.get_vec(g, "start")
                b = S.get_vec(g, "end")
                if a and b:
                    pa = _fp_to_board((a[0], a[1]), origin, rot)
                    pb = _fp_to_board((b[0], b[1]), origin, rot)
                    pieces.append([pa, pb])
            elif kind == "rect":
                a = S.get_vec(g, "start")
                b = S.get_vec(g, "end")
                if a and b:
                    corners = [(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])]
                    board = [_fp_to_board(c, origin, rot) for c in corners]
                    pieces.append(board + [board[0]])
            elif kind == "arc":
                a = S.get_vec(g, "start")
                m = S.get_vec(g, "mid")
                b = S.get_vec(g, "end")
                if a and m and b:
                    sampled = _arc_points((a[0], a[1]), (m[0], m[1]), (b[0], b[1]))
                    pieces.append([_fp_to_board(p, origin, rot) for p in sampled])
            elif kind == "poly":
                pts = _pts_of(g)
                if len(pts) >= 3:
                    board = [_fp_to_board(p, origin, rot) for p in pts]
                    pieces.append(board + [board[0]])
            elif kind == "circle":
                c = S.get_vec(g, "center")
                e = S.get_vec(g, "end")
                if c and e:
                    r = math.hypot(e[0] - c[0], e[1] - c[1])
                    ring = [
                        (
                            c[0] + r * math.cos(2 * math.pi * i / CIRCLE_SAMPLES),
                            c[1] + r * math.sin(2 * math.pi * i / CIRCLE_SAMPLES),
                        )
                        for i in range(CIRCLE_SAMPLES)
                    ]
                    board = [_fp_to_board(p, origin, rot) for p in ring]
                    pieces.append(board + [board[0]])
    return pieces


def _footprint_placement(fp: S.SExpr) -> Tuple[Vec2, float]:
    """`((x, y), rot_deg)` of a footprint in board coordinates.

    KiCad 10 writes `(transform (translate x y) (rotate deg))`; KiCad <= 9 wrote `(at x y deg)`.
    Both appear in the wild (PhiCad emits the former), so both are read.
    """
    tf = S.find(fp, "transform")
    if tf is not None:
        t = S.get_vec(tf, "translate", [0.0, 0.0]) or [0.0, 0.0]
        r = S.get_attr(tf, "rotate", 0.0)
        rot = float(r) if isinstance(r, (int, float)) else 0.0
        return ((float(t[0]), float(t[1])), rot)
    at = S.get_vec(fp, "at", [0.0, 0.0]) or [0.0, 0.0]
    rot = float(at[2]) if len(at) > 2 else 0.0
    return ((float(at[0]), float(at[1])), rot)


def _property(fp: S.SExpr, name: str) -> Optional[str]:
    for p in S.find_all(fp, "property"):
        if len(p) >= 3 and p[1] == name:
            v = p[2]
            return v if isinstance(v, str) else str(v)
    return None


def _footprint_side(fp: S.SExpr) -> str:
    """`'F'` (top) or `'B'` (bottom) — which copper side the part is soldered to."""
    lay = S.get_attr(fp, "layer", "F.Cu")
    return "B" if isinstance(lay, str) and lay.upper().startswith("B") else "F"


def _drill_of(pad: S.SExpr) -> Optional[float]:
    """Round-drill diameter of a pad, mm. Oval drills report their larger dimension."""
    d = S.find(pad, "drill")
    if d is None:
        return None
    vals = [v for v in d[1:] if isinstance(v, float)]
    if not vals:
        return None
    return float(max(vals))


def _pad_info(
    fp: S.SExpr, origin: Vec2, rot: float
) -> Tuple[List[Dict[str, Any]], Optional[BBox2]]:
    """Pads in board coordinates plus the bbox of the pad row (the part's electrical extent)."""
    pads: List[Dict[str, Any]] = []
    corners: List[Vec2] = []
    for pad in S.find_all(fp, "pad"):
        number = pad[1] if len(pad) > 1 else ""
        ptype = pad[2] if len(pad) > 2 else ""
        shape = pad[3] if len(pad) > 3 else ""
        at = S.get_vec(pad, "at", [0.0, 0.0]) or [0.0, 0.0]
        size = S.get_vec(pad, "size", [0.0, 0.0]) or [0.0, 0.0]
        local = (float(at[0]), float(at[1]))
        board = _fp_to_board(local, origin, rot)
        w = float(size[0]) if len(size) > 0 else 0.0
        h = float(size[1]) if len(size) > 1 else w
        # Pad rotation is ignored for the extent: half the diagonal of the pad in either
        # orientation is bounded by half its larger side, and a millimetre of pad is noise next
        # to a wall thickness. What matters is that we never *under*-estimate, hence max().
        half = max(w, h) / 2.0
        corners.extend([(board[0] - half, board[1] - half), (board[0] + half, board[1] + half)])
        pads.append(
            {
                "number": str(number),
                "type": str(ptype),
                "shape": str(shape),
                "at_board": [board[0], board[1]],
                "size": [w, h],
                "drill": _drill_of(pad),
            }
        )
    return pads, _bbox(corners)


def _courtyard_bbox(fp: S.SExpr, origin: Vec2, rot: float) -> Optional[BBox2]:
    """Bbox of the courtyard (`*.CrtYd`) graphics — KiCad's nearest thing to a body outline."""
    pts: List[Vec2] = []
    for layer in ("F.CrtYd", "B.CrtYd"):
        for piece in _graphic_pieces(fp, _FP_GRAPHIC_HEADS, layer=layer, origin=origin, rot=rot):
            pts.extend(piece)
    return _bbox(pts)


# --------------------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------------------


def inspect(path: Union[str, Path]) -> Dict[str, Any]:
    """Everything PhiMech can learn from a `.kicad_pcb`, in the PhiMech frame.

    Shape (stable; treat as the wire format between the PCB provider and everything else)::

        {
          "path": str, "thickness": float, "generator": str, "title": str,
          "outline": {"kind": "rect"|"poly"|"bbox_fallback",
                      "points": [[x, y], ...],           # CCW, PhiMech frame
                      "bbox": {"min": [x, y], "max": [x, y], "size": [w, d]},
                      "kicad_bbox": {...}},              # same box in file coordinates
          "footprints": [{"ref", "value", "footprint", "layer", "at", "rot",
                          "height_above", "height_below", "is_connector", "is_hole",
                          "hole_d", "pads", "model", "bbox", "connector"} ...],
          "holes": [{"ref", "at", "d", "source"} ...],
          "connectors": [ <the connector subset of footprints, with "face"> ],
          "stats": {...}, "warnings": [str, ...]
        }

    Every `at`/`points`/`bbox` is in the PhiMech frame: origin at the board outline's min-X/min-Y
    corner, +Y up. `rot` is degrees CCW in that frame (numerically identical to KiCad's).
    """
    t0 = time.time()
    p = Path(path)
    board = S.parse_file(p)
    if S.head(board) != "kicad_pcb":
        raise ValueError("%s is not a .kicad_pcb (top-level form is %r)" % (p, S.head(board)))

    warnings: List[str] = []
    general = S.find(board, "general")
    thickness_raw = S.get_attr(general, "thickness", 1.6) if general is not None else 1.6
    thickness = float(thickness_raw) if isinstance(thickness_raw, (int, float)) else 1.6

    footprints = S.find_all(board, "footprint")

    # --- outline in file coordinates -------------------------------------------------
    pieces = _graphic_pieces(board, _GRAPHIC_HEADS, layer=EDGE_LAYER)
    for fp in footprints:
        origin, rot = _footprint_placement(fp)
        pieces.extend(
            _graphic_pieces(fp, _FP_GRAPHIC_HEADS, layer=EDGE_LAYER, origin=origin, rot=rot)
        )

    loop_k = _chain(pieces) if pieces else None
    if loop_k is None:
        # Fall back to the extent of everything solderable + a margin. Warn loudly: an enclosure
        # built on a guessed outline is a prototype, not a part.
        pts: List[Vec2] = []
        for fp in footprints:
            origin, rot = _footprint_placement(fp)
            _pads, pb = _pad_info(fp, origin, rot)
            if pb:
                pts.extend([(pb[0], pb[1]), (pb[2], pb[3])])
        for piece in pieces:
            pts.extend(piece)
        bb = _bbox(pts)
        if bb is None:
            raise ValueError(
                "%s has no Edge.Cuts and no pads — nothing to build an outline from" % p
            )
        m = BBOX_FALLBACK_MARGIN
        loop_k = [
            (bb[0] - m, bb[1] - m),
            (bb[2] + m, bb[1] - m),
            (bb[2] + m, bb[3] + m),
            (bb[0] - m, bb[3] + m),
        ]
        outline_kind = "bbox_fallback"
        warnings.append(
            "no closed Edge.Cuts loop found (%d segment(s)); outline is the bbox of copper + %g mm"
            % (len(pieces), m)
        )
    else:
        outline_kind = "rect" if _is_axis_aligned_rect(loop_k) else "poly"

    kb = _bbox(loop_k)
    assert kb is not None  # a chained loop always has points
    min_kx, min_ky, max_kx, max_ky = kb

    def flip(pt: Sequence[float]) -> Vec2:
        """The one and only KiCad -> PhiMech XY conversion (see module docstring)."""
        return (float(pt[0]) - min_kx, max_ky - float(pt[1]))

    outline_pts = _ccw([flip(p) for p in loop_k])
    ob = _bbox(outline_pts)
    assert ob is not None
    width = ob[2] - ob[0]
    depth = ob[3] - ob[1]

    # --- footprints ------------------------------------------------------------------
    fp_out: List[Dict[str, Any]] = []
    holes: List[Dict[str, Any]] = []
    for fp in footprints:
        name = str(fp[1]) if len(fp) > 1 else ""
        origin, rot = _footprint_placement(fp)
        side = _footprint_side(fp)
        ref = _property(fp, "Reference") or ""
        value = _property(fp, "Value") or ""
        pads, pad_bb = _pad_info(fp, origin, rot)
        crt_bb = _courtyard_bbox(fp, origin, rot)
        above, below = H.height_for(name, side)
        is_hole = H.is_mounting_hole(name)
        hole_d: Optional[float] = None
        if is_hole:
            hole_d = H.hole_diameter_from_name(name)
            if hole_d is None:
                drills = [pd["drill"] for pd in pads if pd["drill"]]
                hole_d = float(max(drills)) if drills else None
            if hole_d is None:
                warnings.append("%s (%s): mounting hole with no diameter; ignored" % (ref, name))
                is_hole = False
        at = flip(origin)
        conn_info = H.connector_face_info(name)

        def flip_bb(bb: Optional[BBox2]) -> Optional[Dict[str, List[float]]]:
            if bb is None:
                return None
            a = flip((bb[0], bb[1]))
            b = flip((bb[2], bb[3]))
            return {
                "min": [min(a[0], b[0]), min(a[1], b[1])],
                "max": [max(a[0], b[0]), max(a[1], b[1])],
            }

        pad_bbox_p = flip_bb(pad_bb)
        crt_bbox_p = flip_bb(crt_bb)
        entry: Dict[str, Any] = {
            "ref": ref,
            "value": value,
            "footprint": name,
            "layer": side,
            "at": [at[0], at[1]],
            "rot": rot,
            "height_above": above,
            "height_below": below,
            "is_connector": conn_info is not None,
            "is_hole": is_hole,
            "hole_d": hole_d,
            "pads": len(pads),
            "model": _model_path(fp),
            "height_pattern": H.matched_pattern(name),
            "bbox": crt_bbox_p or pad_bbox_p,
            "pad_bbox": pad_bbox_p,
            "at_kicad": [origin[0], origin[1]],
        }
        if conn_info is not None:
            entry["connector"] = dict(conn_info)
        fp_out.append(entry)

        if is_hole and hole_d:
            holes.append(
                {"ref": ref or name, "at": list(at), "d": float(hole_d), "source": "footprint"}
            )
        else:
            # A bare NPTH pad is a mounting hole too — plenty of real boards use a pad instead of
            # a MountingHole footprint, and a boss that misses it snaps a standoff.
            for pd in pads:
                if pd["type"] == "np_thru_hole" and (pd["drill"] or 0) >= MIN_NPTH_MOUNTING_D:
                    pa = flip(pd["at_board"])
                    holes.append(
                        {
                            "ref": ref or name,
                            "at": [pa[0], pa[1]],
                            "d": float(pd["drill"]),
                            "source": "npth_pad",
                        }
                    )

    holes = _dedupe_holes(holes)
    connectors = [f for f in fp_out if f["is_connector"]]

    tall_above = max([f["height_above"] for f in fp_out] + [0.0])
    tall_below = max([f["height_below"] for f in fp_out] + [0.0])

    return {
        "path": str(p),
        "thickness": thickness,
        "generator": _str_attr(board, "generator"),
        "title": _title(board),
        "outline": {
            "kind": outline_kind,
            "points": [[x, y] for x, y in outline_pts],
            "bbox": {
                "min": [ob[0], ob[1]],
                "max": [ob[2], ob[3]],
                "size": [width, depth],
            },
            "kicad_bbox": {"min": [min_kx, min_ky], "max": [max_kx, max_ky]},
            "segments": len(pieces),
        },
        "footprints": fp_out,
        "holes": holes,
        "connectors": connectors,
        "stats": {
            "footprints": len(fp_out),
            "holes": len(holes),
            "connectors": len(connectors),
            "board_size": [width, depth],
            "tallest_above": tall_above,
            "tallest_below": tall_below,
            "stack_height": tall_below + thickness + tall_above,
            "parse_ms": round((time.time() - t0) * 1000.0, 2),
        },
        "warnings": warnings,
    }


def _dedupe_holes(holes: List[Dict[str, Any]], tol: float = 0.05) -> List[Dict[str, Any]]:
    """Drop holes that repeat the same centre (a MountingHole footprint *and* its NPTH pad)."""
    out: List[Dict[str, Any]] = []
    for h in holes:
        if any(_same((h["at"][0], h["at"][1]), (o["at"][0], o["at"][1]), tol) for o in out):
            continue
        out.append(h)
    return out


def _model_path(fp: S.SExpr) -> Optional[str]:
    m = S.find(fp, "model")
    if m is None or len(m) < 2 or not isinstance(m[1], str):
        return None
    return m[1]


def _str_attr(board: S.SExpr, name: str) -> Optional[str]:
    v = S.get_attr(board, name)
    return v if isinstance(v, str) else None


def _title(board: S.SExpr) -> Optional[str]:
    tb = S.find(board, "title_block")
    if tb is None:
        return None
    v = S.get_attr(tb, "title")
    return v if isinstance(v, str) else None


# --------------------------------------------------------------------------------------
# envelope
# --------------------------------------------------------------------------------------


def envelope_from_pcb(
    path: Union[str, Path],
    *,
    ref: str = "PCB1",
    thickness: Optional[float] = None,
    component_height_top: Optional[float] = None,
    component_height_bottom: Optional[float] = None,
    expose: Sequence[str] = (),
) -> Envelope:
    """Build the `Envelope` that synth places and wraps walls around.

    * `size` = (outline width, outline depth, below + thickness + above); `mount_plane_z` = below,
      so the board's own slab occupies `[below, below + thickness]` and a standoff must be at
      least `below` tall for the board to seat.
    * `footprint` = the extracted outline (CCW) — a chamfered or filleted board yields a real
      polygon, which is the point of doing this instead of using a bbox.
    * `holes` = mounting holes, which `synth/features.py` turns into bosses.
    * `access` = one entry per ref in `expose`. Nothing is exposed unless asked for: an enclosure
      full of speculative holes is worse than one the agent has to iterate on.
    * `body` = board slab (green, holes drilled) + one grey placeholder box per part. Renders and
      exploded views only; placement reasons about the envelope.
    """
    info = inspect(path)
    th = float(thickness) if thickness is not None else float(info["thickness"])
    fps: List[Dict[str, Any]] = info["footprints"]

    above = (
        float(component_height_top)
        if component_height_top is not None
        else max([f["height_above"] for f in fps] + [0.0])
    )
    below = (
        float(component_height_bottom)
        if component_height_bottom is not None
        else max([f["height_below"] for f in fps] + [0.0])
    )

    outline = tuple((float(x), float(y)) for x, y in info["outline"]["points"])
    w, d = info["outline"]["bbox"]["size"]
    mount_plane_z = below

    holes = tuple(
        Hole(at=(float(h["at"][0]), float(h["at"][1])), d=float(h["d"]), ref=h["ref"], plated=False)
        for h in info["holes"]
    )

    by_ref = {f["ref"]: f for f in fps if f["ref"]}
    access: List[Access] = []
    unknown: List[str] = []
    for r in expose:
        f = by_ref.get(r)
        if f is None:
            unknown.append(r)
            continue
        a = _access_for(f, info, mount_plane_z=mount_plane_z, thickness=th)
        if a is not None:
            access.append(a)
    warnings = list(info["warnings"])
    if unknown:
        warnings.append("expose: no such footprint ref(s): %s" % ", ".join(sorted(unknown)))

    body = _body(info, thickness=th, mount_plane_z=mount_plane_z)

    meta = dict(info)
    meta["warnings"] = warnings
    meta["exposed"] = [a.name for a in access]
    meta["heights"] = {"above": above, "below": below, "thickness": th}

    return Envelope(
        ref=ref,
        size=(float(w), float(d), float(below + th + above)),
        footprint=outline,
        holes=holes,
        access=tuple(access),
        mount_plane_z=float(mount_plane_z),
        mount_thickness=float(th),
        body=body,
        name=info.get("title") or Path(path).stem,
        source="kicad_pcb",
        meta=meta,
    )


def _access_for(
    f: Dict[str, Any], info: Dict[str, Any], *, mount_plane_z: float, thickness: float
) -> Optional[Access]:
    """One `Access` for an exposed footprint, on the face its plug actually arrives from.

    Vertical parts (pin headers, JST, terminal blocks) mate along +Z, so the opening belongs on
    the component's `top` face at the part's XY. Right-angle parts (USB, barrel, RJ45, anything
    `_Horizontal`) mate sideways, so the opening belongs on whichever board edge they sit closest
    to, positioned along that edge and centred on the connector body's height above the board.
    """
    name = str(f["footprint"])
    conn = f.get("connector") or H.connector_face_info(name)
    pad_bb = f.get("pad_bbox") or f.get("bbox")
    at = (float(f["at"][0]), float(f["at"][1]))
    body_h = float(f["height_above"]) or H.body_height(name)
    ref = str(f["ref"]) or name

    if pad_bb:
        pw = float(pad_bb["max"][0]) - float(pad_bb["min"][0])
        pd_ = float(pad_bb["max"][1]) - float(pad_bb["min"][1])
        cx = (float(pad_bb["max"][0]) + float(pad_bb["min"][0])) / 2.0
        cy = (float(pad_bb["max"][1]) + float(pad_bb["min"][1])) / 2.0
    else:
        pw = pd_ = 2.0
        cx, cy = at

    face = str((conn or {}).get("face") or "top")
    if face == "top":
        return Access(
            name=ref,
            face="top",
            at=(cx, cy),
            size=(pw + 2 * OPENING_MARGIN, pd_ + 2 * OPENING_MARGIN),
            shape="rect",
            protrusion=0.0,
        )

    # edge-mating: pick the nearest outline-bbox edge.
    bb = info["outline"]["bbox"]
    x0, y0 = float(bb["min"][0]), float(bb["min"][1])
    x1, y1 = float(bb["max"][0]), float(bb["max"][1])
    dists = {
        "left": at[0] - x0,
        "right": x1 - at[0],
        "front": at[1] - y0,
        "back": y1 - at[1],
    }
    edge = min(dists, key=lambda k: dists[k])

    opening = (conn or {}).get("opening")
    z_centre = mount_plane_z + thickness + body_h / 2.0
    if edge in ("left", "right"):
        along = cy
        width = float(opening[0]) if opening else pd_ + 2 * OPENING_MARGIN
    else:
        along = cx
        width = float(opening[0]) if opening else pw + 2 * OPENING_MARGIN
    height = float(opening[1]) if opening else body_h
    return Access(
        name=ref,
        face=edge,
        at=(along, z_centre),
        size=(width, height),
        shape="rect",
        protrusion=float((conn or {}).get("protrusion") or 0.0),
    )


def _body(info: Dict[str, Any], *, thickness: float, mount_plane_z: float) -> ast.Node:
    """Placeholder assembly geometry: green board slab with its holes drilled, grey part boxes.

    Deliberately crude — it exists so a render/exploded view shows something recognisable and so
    the manifold gate has real geometry to chew on. Nothing in synth measures it.
    """
    outline = [(float(x), float(y)) for x, y in info["outline"]["points"]]
    bb = info["outline"]["bbox"]
    w, d = float(bb["size"][0]), float(bb["size"][1])
    kind = info["outline"]["kind"]

    if kind == "rect" or len(outline) < 3:
        slab: ast.Node = ast.Cube(size=(w, d, thickness))
    else:
        slab = ast.LinearExtrude(height=thickness, children=[ast.Polygon(points=tuple(outline))])

    drills: List[ast.Node] = []
    for h in info["holes"]:
        r = float(h["d"]) / 2.0
        drills.append(
            ast.Cylinder(h=thickness + 2.0, r=r, segments=32).translate(
                (float(h["at"][0]), float(h["at"][1]), -1.0)
            )
        )
    board_geom: ast.Node = ast.Difference([slab] + drills) if drills else slab
    board = ast.Color(BOARD_COLOR, [board_geom.translate((0.0, 0.0, mount_plane_z))], name="board")

    parts: List[ast.Node] = []
    top_z = mount_plane_z + thickness
    for f in info["footprints"]:
        if f["is_hole"]:
            continue
        box = f.get("bbox") or f.get("pad_bbox")
        if box:
            bx0, by0 = float(box["min"][0]), float(box["min"][1])
            bx1, by1 = float(box["max"][0]), float(box["max"][1])
        else:
            bx0, by0 = float(f["at"][0]) - 1.0, float(f["at"][1]) - 1.0
            bx1, by1 = bx0 + 2.0, by0 + 2.0
        bw = max(bx1 - bx0, 0.2)
        bd = max(by1 - by0, 0.2)
        ha = float(f["height_above"])
        hb = float(f["height_below"])
        if ha > 0.0:
            parts.append(ast.Cube(size=(bw, bd, ha)).translate((bx0, by0, top_z)))
        if hb > 0.0:
            parts.append(ast.Cube(size=(bw, bd, hb)).translate((bx0, by0, mount_plane_z - hb)))
    children: List[ast.Node] = [board]
    if parts:
        children.append(ast.Color(PART_COLOR, [ast.Union(parts)], name="components"))
    return ast.Group(children, name="pcb")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _fmt_text(info: Dict[str, Any]) -> str:
    o = info["outline"]
    st = info["stats"]
    lines: List[str] = []
    lines.append("board   %s" % info["path"])
    if info.get("title"):
        lines.append("title   %s  (generator: %s)" % (info["title"], info.get("generator")))
    lines.append(
        "outline %s, %d point(s), %.2f x %.2f mm, thickness %.2f mm"
        % (
            o["kind"],
            len(o["points"]),
            o["bbox"]["size"][0],
            o["bbox"]["size"][1],
            info["thickness"],
        )
    )
    lines.append(
        "stack   %.2f mm below + %.2f board + %.2f above = %.2f mm"
        % (st["tallest_below"], info["thickness"], st["tallest_above"], st["stack_height"])
    )
    lines.append(
        "counts  %d footprint(s), %d mounting hole(s), %d connector(s)"
        % (st["footprints"], st["holes"], st["connectors"])
    )
    lines.append("")
    lines.append(
        "%-6s %-10s %-46s %-3s %9s %6s %6s %5s"
        % ("REF", "VALUE", "FOOTPRINT", "SID", "AT (x,y)", "ABOVE", "BELOW", "TAG")
    )
    for f in sorted(info["footprints"], key=lambda f: (f["ref"] or "~")):
        tag = "hole" if f["is_hole"] else ("conn" if f["is_connector"] else "")
        lines.append(
            "%-6s %-10s %-46s %-3s %4.1f,%4.1f %6.2f %6.2f %5s"
            % (
                f["ref"],
                (f["value"] or "")[:10],
                f["footprint"].rsplit(":", 1)[-1][:46],
                f["layer"],
                f["at"][0],
                f["at"][1],
                f["height_above"],
                f["height_below"],
                tag,
            )
        )
    if info["holes"]:
        lines.append("")
        lines.append("mounting holes (PhiMech frame):")
        for h in info["holes"]:
            lines.append(
                "  %-4s  at (%7.3f, %7.3f)  d %.2f  [%s]"
                % (h["ref"], h["at"][0], h["at"][1], h["d"], h["source"])
            )
    if info["connectors"]:
        lines.append("")
        lines.append("connectors:")
        for c in info["connectors"]:
            lines.append(
                "  %-4s  %-40s  mates %-4s  body %.2f mm"
                % (
                    c["ref"],
                    c["footprint"].rsplit(":", 1)[-1][:40],
                    (c.get("connector") or {}).get("face", "?"),
                    c["height_above"],
                )
            )
    lines.append("")
    lines.append("outline points (PhiMech frame, CCW):")
    lines.append("  " + " ".join("(%.2f,%.2f)" % (x, y) for x, y in o["points"]))
    for wmsg in info["warnings"]:
        lines.append("WARNING: %s" % wmsg)
    lines.append("parsed in %.1f ms" % st["parse_ms"])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python3 -m phimech.components.pcb inspect board.kicad_pcb [--json]`."""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="phimech.components.pcb", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    ins = sub.add_parser("inspect", help="report what the PCB extractor sees")
    ins.add_argument("board")
    ins.add_argument("--json", action="store_true", help="machine-readable output")
    env = sub.add_parser("envelope", help="report the Envelope built from the board")
    env.add_argument("board")
    env.add_argument("--ref", default="PCB1")
    env.add_argument("--expose", default="", help="comma-separated footprint refs to expose")
    env.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "inspect":
        info = inspect(args.board)
        print(json.dumps(info, indent=2, sort_keys=True) if args.json else _fmt_text(info))
        return 1 if info["warnings"] else 0
    if args.cmd == "envelope":
        expose = tuple(r.strip() for r in args.expose.split(",") if r.strip())
        e = envelope_from_pcb(args.board, ref=args.ref, expose=expose)
        if args.json:
            print(json.dumps(e.to_dict(), indent=2, sort_keys=True))
        else:
            print("ref            %s (%s)" % (e.ref, e.name))
            print("size           %.2f x %.2f x %.2f mm" % e.size)
            print(
                "mount_plane_z  %.2f  (board %.2f mm thick)"
                % (e.mount_plane_z, e.mount_thickness)
            )
            print("footprint      %d point(s)" % len(e.outline))
            for h in e.holes:
                print("hole           %-4s (%7.3f, %7.3f) d %.2f" % (h.ref, h.at[0], h.at[1], h.d))
            for a in e.access:
                print(
                    "access         %-4s face %-6s at (%7.3f, %7.3f) size %.2f x %.2f"
                    % (a.name, a.face, a.at[0], a.at[1], a.size[0], a.size[1])
                )
        return 0
    ap.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARC_SAMPLES",
    "CHAIN_TOL",
    "MIN_NPTH_MOUNTING_D",
    "OPENING_MARGIN",
    "inspect",
    "envelope_from_pcb",
    "main",
]
