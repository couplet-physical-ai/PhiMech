"""Ergonomic, backend-neutral CSG constructors.

The synth layer should describe intent with these small helpers rather than hand-building
AST dataclasses.  Every helper returns an immutable :class:`~phimech.csg.ast.Node`; units
are millimetres and degrees, and no CAD kernel is imported here.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from phimech.csg.ast import (
    Circle,
    Cube,
    Cylinder,
    Difference,
    Group,
    Hull,
    LinearExtrude,
    Minkowski,
    Node,
    Rotate,
    RoundedCube,
    RoundedSquare,
    Sphere,
    Square,
    Translate,
    Union,
)

EPS = 0.01

HEAT_SET = {
    "m2": (3.2, 3.0),
    "m2.5": (3.5, 3.4),
    "m3": (4.0, 4.0),
    "m4": (5.6, 5.0),
}

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def _vec3(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("expected a three-component vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _nonzero(vector: Sequence[float]) -> bool:
    return any(float(component) != 0.0 for component in vector)


def _translate(node: Node, at: Sequence[float]) -> Node:
    """Apply an optional translation without adding noisy identity nodes."""
    position = _vec3(at)
    return Translate(position, [node]) if _nonzero(position) else node


def _place_raw(node: Node, at: Sequence[float], rot: Sequence[float]) -> Node:
    rotation = _vec3(rot)
    position = _vec3(at)
    if _nonzero(rotation):
        node = Rotate(rotation, [node])
    if _nonzero(position):
        node = Translate(position, [node])
    return node


def box(w: float, d: float, h: float, *, center: bool = False, at: Vec3 = (0, 0, 0)) -> Node:
    """Make an axis-aligned box, optionally translating its local origin."""
    node = Cube((float(w), float(d), float(h)), bool(center))
    return _translate(node, at)


def cyl(
    d: float,
    h: float,
    *,
    at: Vec3 = (0, 0, 0),
    axis: str = "z",
    segments: Optional[int] = None,
) -> Node:
    """Make a cylinder of diameter ``d`` along one of the Cartesian axes."""
    node: Node = Cylinder(float(h), float(d) / 2.0, segments=segments)
    axis_name = axis.lower()
    if axis_name == "x":
        node = Rotate((0.0, 90.0, 0.0), [node])
    elif axis_name == "y":
        node = Rotate((-90.0, 0.0, 0.0), [node])
    elif axis_name != "z":
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return _translate(node, at)


def tube(d_outer: float, d_inner: float, h: float, **kw: object) -> Node:
    """Make a hollow cylinder, using the same placement for both walls."""
    outer = cyl(d_outer, h, **kw)  # type: ignore[arg-type]
    inner = cyl(d_inner, h, **kw)  # type: ignore[arg-type]
    return cut_through(outer, inner)


def rounded_box(w: float, d: float, h: float, r: float, *, edge_r: float = 0.0) -> Node:
    """Make a box with rounded XY corners and optionally rounded outer edges.

    Returns a `RoundedCube`, which each backend renders in the way it does best: OpenSCAD
    desugars it back to the historical convex hull (`rounded_box_as_hull`, byte-identical
    output), while FreeCAD fillets the edges natively.

    That split is the point. A hull has no curved faces, so the old always-hull form exported a
    60x40x25 rounded box as 130 planar faces and **zero** cylindrical surfaces, and a real
    enclosure as 9,174 faces in a 9.7 MB STEP file that no one can edit. The fillet form is 10
    faces. The earlier docstring called the hull "deliberately backend-neutral", which was true
    and was exactly the problem: it levelled both kernels down to the weaker one's output.
    """
    width, depth, height = float(w), float(d), float(h)
    corner = min(max(float(r), 0.0), width / 2.0, depth / 2.0)
    edge = min(max(float(edge_r), 0.0), width / 2.0, depth / 2.0, height / 2.0)
    if corner <= 0.0 and edge <= 0.0:
        return box(width, depth, height)
    return RoundedCube(size=(width, depth, height), r=corner, edge_r=edge)


def rounded_box_as_hull(node: RoundedCube) -> Node:
    """Desugar a `RoundedCube` to the convex hull the OpenSCAD lane has always emitted.

    Kept verbatim so the OpenSCAD output of every existing design is unchanged by the switch to
    a `rounded_cube` node. A regression here would show up as a changed golden render volume,
    which is precisely what the golden tests are for.
    """
    width, depth, height = (float(v) for v in node.size)
    corner, edge = node.r, node.edge_r
    corner = min(max(float(corner), 0.0), width / 2.0, depth / 2.0)
    if edge <= 0.0:
        if corner <= 0.0:
            return box(width, depth, height)
        centres = (
            (corner, corner),
            (width - corner, corner),
            (width - corner, depth - corner),
            (corner, depth - corner),
        )
        return Hull([cyl(2.0 * corner, height, at=(x, y, 0.0)) for x, y in centres])

    nodes: List[Node] = []
    # Spheres preserve the exact outer bounds while rounding every corner.
    for x in (edge, width - edge):
        for y in (edge, depth - edge):
            for z in (edge, height - edge):
                nodes.append(_translate(Sphere(edge), (x, y, z)))

    # Cylinders along all twelve edges make the hull's rounding stable for long boxes.
    if width > 2.0 * edge:
        for y in (edge, depth - edge):
            for z in (edge, height - edge):
                nodes.append(cyl(2.0 * edge, width - 2.0 * edge, at=(edge, y, z), axis="x"))
    if depth > 2.0 * edge:
        for x in (edge, width - edge):
            for z in (edge, height - edge):
                nodes.append(cyl(2.0 * edge, depth - 2.0 * edge, at=(x, edge, z), axis="y"))
    if corner > 0.0 and height > 2.0 * edge:
        for x in (corner, width - corner):
            for y in (corner, depth - corner):
                nodes.append(cyl(2.0 * corner, height - 2.0 * edge, at=(x, y, edge)))
    return Hull(nodes)


def rounded_rect(w: float, d: float, r: float) -> Node:
    """Make a 2D rounded rectangle in the XY plane.

    Returns a `RoundedSquare`. OpenSCAD desugars it to the historical hull of circles; FreeCAD
    fillets a rectangular face. The hull form was worse than merely faceted here: FreeCAD cannot
    build a planar hull at all, so one `rounded_rect` anywhere in a part forced the **whole part**
    through mesh-to-solid conversion -- `rpi-zero-case-base` came out as 13,758 faces and 13 MB.
    """
    width, depth = float(w), float(d)
    radius = min(max(float(r), 0.0), width / 2.0, depth / 2.0)
    if radius <= 0.0:
        return Square((width, depth))
    return RoundedSquare(size=(width, depth), r=radius)


def rounded_rect_as_hull(node: RoundedSquare) -> Node:
    """Desugar a `RoundedSquare` to the hull of circles the OpenSCAD lane has always emitted."""
    width, depth = (float(v) for v in node.size)
    radius = min(max(float(node.r), 0.0), width / 2.0, depth / 2.0)
    if radius <= 0.0:
        return Square((width, depth))
    centres = (
        (radius, radius),
        (width - radius, radius),
        (width - radius, depth - radius),
        (radius, depth - radius),
    )
    return Hull([_translate(Circle(radius), (x, y, 0.0)) for x, y in centres])


def slot(length: float, width: float, h: float) -> Node:
    """Make a horizontal, rounded-ended slot extruded along +Z."""
    return LinearExtrude(float(h), [rounded_rect(float(length), float(width), float(width) / 2.0)])


def countersunk_hole(d: float, h: float, head_d: float, head_depth: float) -> Node:
    """Make a through-hole cutter with a conical countersink at its top face."""
    height = float(h)
    depth = min(max(float(head_depth), 0.0), max(height, 0.0))
    shaft = cyl(float(d), height)
    if depth <= 0.0:
        return shaft
    cone = Cylinder(
        depth,
        float(d) / 2.0,
        r2=float(head_d) / 2.0,
    )
    cone = _translate(cone, (0.0, 0.0, height - depth))
    return union_all((shaft, cone))


def screw_boss(od: float, id: float, h: float, *, fillet_r: float = 0) -> Node:
    """Make a hollow boss, with an optional convex base transition."""
    height = float(h)
    fillet = min(max(float(fillet_r), 0.0), max(height / 2.0, 0.0))
    if fillet <= 0.0:
        return tube(float(od), float(id), height)
    base = cyl(float(od) + 2.0 * fillet, fillet)
    upper = cyl(float(od), height - fillet, at=(0.0, 0.0, fillet))
    outer = Hull([base, upper])
    return cut_through(outer, cyl(float(id), height, at=(0.0, 0.0, 0.0)))


def standoff(h: float, od: float, hole_d: float, *, base_d: Optional[float] = None) -> Node:
    """Make a hollow standoff, optionally with a one-mm mounting flange."""
    height = float(h)
    outer: Node = cyl(float(od), height)
    if base_d is not None and float(base_d) > float(od):
        flange_h = min(1.0, max(height, 0.0))
        outer = union_all((outer, cyl(float(base_d), flange_h)))
    return cut_through(outer, cyl(float(hole_d), height))


def heat_set_boss(insert: str = "m3", h: float = 6) -> Node:
    """Make a boss for a standard heat-set insert.

    The four table values are the manufacturer's recommended hole diameter and minimum
    embedment.  A two-mm radial wall is used because the helper has no separate wall
    argument; callers needing another wall can use :func:`screw_boss` directly.
    """
    key = insert.lower()
    try:
        hole_d, min_depth = HEAT_SET[key]
    except KeyError:
        choices = ", ".join(HEAT_SET)
        raise ValueError(
            "unknown heat-set insert {!r}; choose {}".format(insert, choices)
        ) from None
    height = float(h)
    if height < min_depth:
        raise ValueError("{} heat-set boss needs at least {} mm depth".format(key, min_depth))
    return screw_boss(hole_d + 4.0, hole_d, height)


def vent_slots(
    face_w: float,
    face_h: float,
    slot_w: float,
    pitch: float,
    depth: float,
    *,
    margin: float = 3,
) -> Node:
    """Make horizontal slot cutters laid out inside an XY face rectangle."""
    width, height, sw, step, gap = (
        float(face_w),
        float(face_h),
        float(slot_w),
        float(pitch),
        float(margin),
    )
    usable_w = width - 2.0 * gap
    usable_h = height - 2.0 * gap
    if usable_w <= 0.0 or usable_h < sw or sw <= 0.0 or step <= 0.0:
        raise ValueError("face is too small for the requested vent slots")
    count = int(math.floor((usable_h - sw) / step)) + 1
    cutters = [
        _translate(slot(usable_w, sw, float(depth)), (gap, gap + i * step, 0.0))
        for i in range(count)
    ]
    return union_all(cutters)


def cut_through(target: Node, cutter: Node) -> Node:
    """Subtract an isotropically over-cut cutter from ``target``.

    Co-planar cutter and target faces can leave a paper-thin skin in either CAD kernel.
    A Minkowski sum with a tiny sphere extends the cutter by ``EPS`` in place, unlike a
    scale wrapper which would also move a cutter that has already been translated.
    """
    overcut = Minkowski([cutter, Sphere(EPS)])
    return Difference([target, overcut])


def union_all(nodes: Iterable[Node]) -> Node:
    """Build a union from a non-empty iterable of nodes."""
    children = list(nodes)
    if not children:
        raise ValueError("union_all requires at least one node")
    return Union(children)


def diff_all(base: Node, cutters: Iterable[Node]) -> Node:
    """Subtract every cutter from ``base``; no cutters leaves the base unchanged."""
    children = [base] + list(cutters)
    return base if len(children) == 1 else Difference(children)


def place(node: Node, at: Vec3, rot: Vec3 = (0, 0, 0)) -> Node:
    """Apply Euler rotation followed by translation without mutating ``node``."""
    return _place_raw(node, at, rot)


# --------------------------------------------------------------------------------------
# boolean volume sanity
#
# A B-rep boolean can fail without raising: OCC's `cut` on a self-intersecting or tangent tool
# may return the base unchanged, or the base with the tool *fused on*. Nothing downstream
# notices -- the result is a valid, manifold, wrong solid. Volume is the cheap invariant that
# catches it: a subtraction can only remove material and a union can only add it. The checks
# below are pure arithmetic so that this module stays kernel-free and usable without CAD dependencies;
# the caller supplies volumes, or a `measure` callback that obtains them from a backend.
#
# Pattern (measure before and after a boolean, refuse the impossible) adapted from agentcad's
# `safe_cut`; the implementation here is PhiMech's own.
# --------------------------------------------------------------------------------------

#: Relative slack before a boolean is declared impossible. The two volumes compared are
#: separate tessellations of the same curved faces, and 0.5 % is well above that noise while a
#: genuinely failed cut is off by tens of percent.
BOOLEAN_VOLUME_TOLERANCE = 0.005

VolumeOf = Callable[[Node], float]


class ImpossibleBoolean(ValueError):
    """A boolean whose measured result contradicts what the operation can do.

    Carries the numbers so an agent can act on them: `kind`, `result_volume`,
    `reference_volume` (the base of a cut, the largest child of a union) and `tolerance`.
    """

    def __init__(
        self, kind: str, result_volume: float, reference_volume: float, tolerance: float
    ) -> None:
        self.kind = kind
        self.result_volume = float(result_volume)
        self.reference_volume = float(reference_volume)
        self.tolerance = float(tolerance)
        if kind == "difference":
            what = "a difference grew: result {:.6g} mm3 > base {:.6g} mm3"
        elif kind == "union":
            what = "a union shrank: result {:.6g} mm3 < largest child {:.6g} mm3"
        else:
            what = "an intersection grew: result {:.6g} mm3 > smallest child {:.6g} mm3"
        super().__init__(
            "impossible {}: {} (tolerance {:.3g} %)".format(
                kind, what.format(self.result_volume, self.reference_volume), tolerance * 100.0
            )
        )


def check_boolean_volume(
    kind: str,
    result_volume: float,
    *,
    base_volume: Optional[float] = None,
    child_volumes: Sequence[float] = (),
    tolerance: float = BOOLEAN_VOLUME_TOLERANCE,
) -> Dict[str, float]:
    """Refuse a boolean whose volumes are impossible; return the numbers when they are not.

    * ``difference``: the result may not exceed ``base_volume`` by more than ``tolerance``.
    * ``union``: the result may not fall short of the largest of ``child_volumes``.
    * ``intersection``: the result may not exceed the smallest of ``child_volumes``.

    Returns ``{"kind", "result_volume", "reference_volume", "ratio", "limit"}`` where
    ``ratio`` is on a "> 1 is bad" axis for every kind and ``limit`` is ``1 + tolerance``, so a
    gate can report one number against one threshold.
    """
    result = float(result_volume)
    if kind == "difference":
        if base_volume is None:
            raise ValueError("check_boolean_volume('difference') needs base_volume")
        reference = float(base_volume)
        ratio = _ratio(result, reference)
    elif kind == "union":
        if not child_volumes:
            raise ValueError("check_boolean_volume('union') needs child_volumes")
        reference = max(float(v) for v in child_volumes)
        ratio = _ratio(reference, result)
    elif kind == "intersection":
        if not child_volumes:
            raise ValueError("check_boolean_volume('intersection') needs child_volumes")
        reference = min(float(v) for v in child_volumes)
        ratio = _ratio(result, reference)
    else:
        raise ValueError("unknown boolean kind {!r}".format(kind))
    limit = 1.0 + float(tolerance)
    if ratio > limit:
        raise ImpossibleBoolean(kind, result, reference, tolerance)
    return {
        "kind": kind,
        "result_volume": result,
        "reference_volume": reference,
        "ratio": ratio,
        "limit": limit,
    }


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator with 0 / 0 read as "nothing changed" and x / 0 as impossible."""
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def safe_cut(
    target: Node,
    cutter: Node,
    *,
    measure: VolumeOf,
    tolerance: float = BOOLEAN_VOLUME_TOLERANCE,
) -> Node:
    """`cut_through`, then refuse the result if it has more volume than the target had.

    ``measure`` returns the volume in mm3 of a node -- typically
    ``lambda node: backend.measure(node).volume``. It is a callback so this module never
    imports a kernel. Raises `ImpossibleBoolean` with both volumes.
    """
    node = cut_through(target, cutter)
    check_boolean_volume(
        "difference", measure(node), base_volume=measure(target), tolerance=tolerance
    )
    return node


def safe_union(
    nodes: Iterable[Node],
    *,
    measure: VolumeOf,
    tolerance: float = BOOLEAN_VOLUME_TOLERANCE,
) -> Node:
    """`union_all`, then refuse the result if it is smaller than its largest child."""
    children = list(nodes)
    node = union_all(children)
    check_boolean_volume(
        "union",
        measure(node),
        child_volumes=[measure(child) for child in children],
        tolerance=tolerance,
    )
    return node


def text_emboss(
    text: str,
    size: float,
    depth: float,
    *,
    font: str = "Liberation Sans:style=Bold",
    halign: str = "center",
) -> Node:
    """Encode text intent in the AST's agreed text-group convention.

    There is intentionally no backend-specific Text primitive in the AST.  Backends render
    a ``Group`` whose name starts with ``text:`` and contains this JSON payload.
    """
    payload = {
        "text": str(text),
        "size": float(size),
        "depth": float(depth),
        "font": str(font),
        "halign": str(halign),
        "valign": "center",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return Group([], name="text:" + encoded)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Return signed shoelace area; positive values indicate counter-clockwise winding."""
    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        float(points[i][0]) * float(points[(i + 1) % len(points)][1])
        - float(points[(i + 1) % len(points)][0]) * float(points[i][1])
        for i in range(len(points))
    )


def polygon_bbox(points: Sequence[Sequence[float]]) -> Tuple[Vec2, Vec2]:
    """Return ``((min_x, min_y), (max_x, max_y))`` for a non-empty polygon."""
    if not points:
        raise ValueError("polygon_bbox requires at least one point")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def is_ccw(points: Sequence[Sequence[float]]) -> bool:
    """Return whether a polygon has positive signed area."""
    return polygon_area(points) > 0.0


def ensure_ccw(points: Sequence[Sequence[float]]) -> List[Vec2]:
    """Return a list with the requested polygon winding, reversing a clockwise input."""
    result = [(float(point[0]), float(point[1])) for point in points]
    return result if is_ccw(result) else list(reversed(result))


def _line_intersection(
    p: Vec2,
    direction_a: Vec2,
    q: Vec2,
    direction_b: Vec2,
) -> Vec2:
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(denominator) < 1e-12:
        raise ValueError("polygon has parallel adjacent edges; miter join is undefined")
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    t = (q_minus_p[0] * direction_b[1] - q_minus_p[1] * direction_b[0]) / denominator
    return (p[0] + t * direction_a[0], p[1] + t * direction_a[1])


def offset_polygon(points: Sequence[Sequence[float]], delta: float) -> List[Vec2]:
    """Offset a CCW simple polygon by ``delta`` using straight miter joins.

    Positive ``delta`` grows a CCW polygon outward.  The implementation uses intersections
    of adjacent edge-offset lines, which keeps corners sharp and does not need a geometry
    library.
    """
    if len(points) < 3:
        raise ValueError("offset_polygon requires at least three points")
    polygon = [(float(point[0]), float(point[1])) for point in points]
    if not is_ccw(polygon):
        raise ValueError("offset_polygon expects counter-clockwise points")
    lines = []
    amount = float(delta)
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            raise ValueError("offset_polygon does not accept duplicate adjacent points")
        normal = (dy / length, -dx / length)
        lines.append(
            (
                (start[0] + amount * normal[0], start[1] + amount * normal[1]),
                (dx, dy),
            )
        )
    result: List[Vec2] = []
    for index in range(len(polygon)):
        previous = lines[(index - 1) % len(lines)]
        current = lines[index]
        result.append(_line_intersection(previous[0], previous[1], current[0], current[1]))
    return result


__all__ = [
    "EPS",
    "HEAT_SET",
    "box",
    "cyl",
    "tube",
    "rounded_box",
    "rounded_rect",
    "slot",
    "countersunk_hole",
    "screw_boss",
    "standoff",
    "heat_set_boss",
    "vent_slots",
    "cut_through",
    "union_all",
    "diff_all",
    "place",
    "BOOLEAN_VOLUME_TOLERANCE",
    "ImpossibleBoolean",
    "check_boolean_volume",
    "safe_cut",
    "safe_union",
    "text_emboss",
    "offset_polygon",
    "polygon_area",
    "polygon_bbox",
    "is_ccw",
    "ensure_ccw",
]
