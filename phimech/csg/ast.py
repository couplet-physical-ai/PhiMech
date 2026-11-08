"""Backend-agnostic CSG AST — the single geometry contract of PhiMech.

Everything in PhiMech that produces geometry produces one of these nodes. Everything that
renders geometry (the OpenSCAD backend, the FreeCAD backend) consumes one of these nodes.
Nothing else crosses that boundary: no backend ever sees an IR dict, and no IR code ever
emits a `.scad` string or touches `Part`.

Design rules
------------
* Plain dataclasses, frozen, stdlib only. This module must import in python 3.9 (the macOS
  system python), inside FreeCAD's bundled python, and inside the MCP venv.
* Units are millimetres, angles are degrees. Always.
* Nodes are immutable values; transforms wrap children rather than mutating them.
* Every node round-trips through `to_dict()` / `from_dict()` so the AST can be logged,
  diffed, cached, and shipped over MCP as JSON.

Primitive set is deliberately the OpenSCAD set: it is the smallest set FreeCAD can also
produce natively (Part.makeBox / makeCylinder / makeSphere / makePolygon+extrude / revolve),
so a node is renderable by both backends without approximation.

The one deliberate exception is `ScadModule`: a call into a vendored OpenSCAD library (BOSL2).
It is OpenSCAD-only by nature and the FreeCAD backend refuses it by name, which is the honest
way to get threads, sweeps and rounding without teaching the core AST a hundred new primitives.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union as _Union

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

#: What a `ScadModule` argument may hold: JSON scalars and (nested) arrays of them. No objects —
#: OpenSCAD has no dictionary literal, and BOSL2 structs are lists of pairs anyway.
JSONValue = _Union[None, bool, int, float, str, List["JSONValue"]]


def _vec3(v: Sequence[float]) -> Vec3:
    if len(v) != 3:
        raise ValueError(f"expected 3 components, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _vec2(v: Sequence[float]) -> Vec2:
    if len(v) != 2:
        raise ValueError(f"expected 2 components, got {v!r}")
    return (float(v[0]), float(v[1]))


# --------------------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """Base of every CSG node. `kind` is the discriminator used in the JSON form."""

    kind: str = field(init=False, default="node")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        for f in fields(self):
            if f.name == "kind":
                continue
            v = getattr(self, f.name)
            out[f.name] = _encode(v)
        return out

    def to_json(self, indent: Optional[int] = None) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # ergonomic combinators -------------------------------------------------------------

    def translate(self, v: Sequence[float]) -> "Translate":
        return Translate(_vec3(v), [self])

    def rotate(self, v: Sequence[float]) -> "Rotate":
        return Rotate(_vec3(v), [self])

    def scale(self, v: Sequence[float]) -> "Scale":
        return Scale(_vec3(v), [self])

    def mirror(self, normal: Sequence[float]) -> "Mirror":
        return Mirror(_vec3(normal), [self])

    def color(self, rgba: Sequence[float], name: Optional[str] = None) -> "Color":
        return Color(tuple(float(c) for c in rgba), [self], name)  # type: ignore[arg-type]

    def named(self, name: str) -> "Group":
        return Group([self], name=name)

    def __add__(self, other: "Node") -> "Union":
        return Union([self, other])

    def __sub__(self, other: "Node") -> "Difference":
        return Difference([self, other])

    def __and__(self, other: "Node") -> "Intersection":
        return Intersection([self, other])


def _encode(v: Any) -> Any:
    if isinstance(v, Node):
        return v.to_dict()
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {k: _encode(x) for k, x in v.items()}
    return v


# --------------------------------------------------------------------------------------
# primitives (3D)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cube(Node):
    """Axis-aligned box. `center=False` puts one corner at the origin (OpenSCAD semantics)."""

    size: Vec3 = (1.0, 1.0, 1.0)
    center: bool = False
    kind: str = field(init=False, default="cube")


@dataclass(frozen=True)
class Cylinder(Node):
    """Cylinder/cone along +Z. `r2` (top radius) defaults to `r` for a straight cylinder."""

    h: float = 1.0
    r: float = 1.0
    r2: Optional[float] = None
    center: bool = False
    segments: Optional[int] = None  # $fn; None = backend default
    kind: str = field(init=False, default="cylinder")


@dataclass(frozen=True)
class RoundedCube(Node):
    """A box with rounded edges, occupying 0..size on each axis.

    This exists because rounding an edge is a *fillet*, and expressing it as a convex hull of
    cylinders throws the curvature away. Measured on a 60x40x25 box, the hull form exports as
    130 planar faces and **zero** cylindrical surfaces, while the fillet form is 10 faces --
    six planes and four cylinders, which is the correct answer. See `phimech/brep/README.md`.

    `r` rounds the four vertical (Z-parallel) edges. `edge_r` additionally rounds the eight
    horizontal ones; when `edge_r` is set and `r` is not, all twelve are rounded by `edge_r`.
    Both are clamped to what the box can accept, so an over-large radius degrades instead of
    failing the render.

    Backends differ *by design* and that difference is the whole point: the OpenSCAD lane
    desugars this to the historical hull, so its output is unchanged, while the FreeCAD lane
    fillets natively and produces a true B-rep.
    """

    size: Vec3 = (1.0, 1.0, 1.0)
    r: float = 0.0
    edge_r: float = 0.0
    segments: Optional[int] = None
    kind: str = field(init=False, default="rounded_cube")

    def radii(self) -> Tuple[float, float]:
        """Clamped (vertical, horizontal) fillet radii. One definition, used by both lanes."""
        width, depth, height = (float(v) for v in self.size)
        horizontal = min(max(float(self.edge_r), 0.0), width / 2.0, depth / 2.0, height / 2.0)
        vertical = min(max(float(self.r), 0.0), width / 2.0, depth / 2.0)
        if vertical <= 0.0:
            vertical = horizontal
        return vertical, horizontal


@dataclass(frozen=True)
class Sphere(Node):
    r: float = 1.0
    segments: Optional[int] = None
    kind: str = field(init=False, default="sphere")


@dataclass(frozen=True)
class Polyhedron(Node):
    """Explicit mesh: `points` are Vec3, `faces` index into them (outward-facing CCW)."""

    points: Tuple[Vec3, ...] = ()
    faces: Tuple[Tuple[int, ...], ...] = ()
    kind: str = field(init=False, default="polyhedron")


@dataclass(frozen=True)
class Import(Node):
    """Reference an external STEP/STL/3MF file — how vendor component models enter a scene."""

    path: str = ""
    kind: str = field(init=False, default="import")


# --------------------------------------------------------------------------------------
# primitives (2D) + extrusions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Square(Node):
    size: Vec2 = (1.0, 1.0)
    center: bool = False
    kind: str = field(init=False, default="square")


@dataclass(frozen=True)
class RoundedSquare(Node):
    """A 2D rectangle with rounded corners, occupying 0..size in XY.

    The 2D sibling of `RoundedCube`, and it fixes a *worse* problem. Expressed as
    `Hull([circles])` the FreeCAD backend cannot build it at all -- a hull of coplanar points has
    no volume, and qhull reports a flat initial simplex -- so any part containing one fell back
    to mesh-to-solid conversion for the **whole part**. That is how `rpi-zero-case-base` reached
    13,758 faces in a 13 MB STEP file.

    OpenSCAD desugars this back to the historical hull of circles; FreeCAD fillets a rectangular
    face, so a `linear_extrude` of it becomes a prism with real cylindrical sides.
    """

    size: Vec2 = (1.0, 1.0)
    r: float = 0.0
    segments: Optional[int] = None
    kind: str = field(init=False, default="rounded_square")

    def radius(self) -> float:
        """The corner radius, clamped to what the rectangle can accept."""
        width, depth = (float(v) for v in self.size)
        return min(max(float(self.r), 0.0), width / 2.0, depth / 2.0)


@dataclass(frozen=True)
class Circle(Node):
    r: float = 1.0
    segments: Optional[int] = None
    kind: str = field(init=False, default="circle")


@dataclass(frozen=True)
class Polygon(Node):
    """2D polygon in the XY plane. `paths` optional; when given, paths[0] is the outer loop
    and subsequent paths are holes (OpenSCAD semantics)."""

    points: Tuple[Vec2, ...] = ()
    paths: Optional[Tuple[Tuple[int, ...], ...]] = None
    kind: str = field(init=False, default="polygon")


@dataclass(frozen=True)
class LinearExtrude(Node):
    """Extrude 2D children along +Z by `height`. `twist` in degrees, `scale` at the top."""

    height: float = 1.0
    children: List[Node] = field(default_factory=list)
    center: bool = False
    twist: float = 0.0
    # Qualified deliberately: this module also defines a CSG `Union` node, and because
    # annotations are strings here a bare `Union[...]` would resolve to that node class
    # rather than to the typing construct if anything ever asked for these type hints.
    scale: typing.Union[float, Vec2] = 1.0
    kind: str = field(init=False, default="linear_extrude")


@dataclass(frozen=True)
class RotateExtrude(Node):
    """Revolve 2D children (which must lie at x >= 0) around the Z axis by `angle` degrees."""

    children: List[Node] = field(default_factory=list)
    angle: float = 360.0
    segments: Optional[int] = None
    kind: str = field(init=False, default="rotate_extrude")


# --------------------------------------------------------------------------------------
# booleans
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Union(Node):
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="union")


@dataclass(frozen=True)
class Difference(Node):
    """children[0] minus all the rest."""

    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="difference")


@dataclass(frozen=True)
class Intersection(Node):
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="intersection")


@dataclass(frozen=True)
class Hull(Node):
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="hull")


@dataclass(frozen=True)
class Minkowski(Node):
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="minkowski")


# --------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Translate(Node):
    v: Vec3 = (0.0, 0.0, 0.0)
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="translate")


@dataclass(frozen=True)
class Rotate(Node):
    """Euler rotation, degrees, applied X then Y then Z (OpenSCAD `rotate([x,y,z])`)."""

    v: Vec3 = (0.0, 0.0, 0.0)
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="rotate")


@dataclass(frozen=True)
class Scale(Node):
    v: Vec3 = (1.0, 1.0, 1.0)
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="scale")


@dataclass(frozen=True)
class Mirror(Node):
    normal: Vec3 = (1.0, 0.0, 0.0)
    children: List[Node] = field(default_factory=list)
    kind: str = field(init=False, default="mirror")


@dataclass(frozen=True)
class Color(Node):
    """Display-only. Backends may ignore it; it never changes geometry."""

    rgba: Tuple[float, ...] = (0.7, 0.7, 0.7, 1.0)
    children: List[Node] = field(default_factory=list)
    name: Optional[str] = None
    kind: str = field(init=False, default="color")


@dataclass(frozen=True)
class Group(Node):
    """Semantic grouping: `name` survives into FreeCAD object labels and .scad comments.
    Used to keep "lid", "base", "boss_J1" identifiable in the output."""

    children: List[Node] = field(default_factory=list)
    name: Optional[str] = None
    kind: str = field(init=False, default="group")


# --------------------------------------------------------------------------------------
# library calls (OpenSCAD-only)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScadModule(Node):
    """A call to a module from a vendored OpenSCAD library — `threaded_rod(d=3, l=12, pitch=0.5)`.

    Why this node exists: hand-rolling an ISO thread, a swept handle or a continuous-curvature
    rounding out of cubes and cylinders is where agents fail, and BOSL2 (BSD-2-Clause, vendored
    under `phimech/vendor/BOSL2/`) already does each of them correctly. Rather than mirror its
    vocabulary primitive by primitive, the AST carries the *call*: which module, which named
    arguments, which library file to include.

    Contract
    --------
    * `module` is a bare identifier; `args` are named arguments whose values are JSON scalars or
      arrays (formatted as OpenSCAD literals by the emitter, so `[[0, 0], [10, 0]]` is a path and
      `"hex"` a string). Positional arguments are deliberately absent — named ones are diffable
      and cannot silently shift meaning when the library reorders a signature.
    * `include` is the library file the module needs, relative to `phimech/vendor/`
      (`"BOSL2/std.scad"`, `"BOSL2/screws.scad"`, `"BOSL2/threading.scad"`). Validation refuses
      anything outside that directory; the emitter also includes `BOSL2/std.scad` first because
      the library requires it.
    * `children`, if any, become the module's child block (`module(...) { ... }`) — BOSL2 uses
      that for attachments; the sweeps take their profile as an *argument* (a point list).
    * **OpenSCAD-only.** The FreeCAD backend lists `scad_module` under `UNSUPPORTED` and refuses
      a tree containing one before launching anything, so STEP export is not available for this
      node; the pipeline routes such parts to OpenSCAD (STL/3MF/OBJ/PNG). Units stay mm/degrees.
    * Dimensionality is not inferred from library source: `is_2d()` treats a `ScadModule` as 3D,
      so a BOSL2 2D shape cannot be a `LinearExtrude` child — use BOSL2's own `linear_sweep`
      (profile as points) or the core `RoundedSquare` instead.
    """

    module: str = ""
    args: Mapping[str, JSONValue] = field(default_factory=dict)
    include: str = "BOSL2/std.scad"
    children: Tuple[Node, ...] = ()
    kind: str = field(init=False, default="scad_module")

    def __post_init__(self) -> None:
        # Frozen, so normalise through object.__setattr__: a plain dict keeps to_dict() and JSON
        # equality stable, and a tuple keeps children immutable whatever iterable was passed.
        object.__setattr__(self, "args", dict(self.args or {}))
        object.__setattr__(self, "children", tuple(self.children or ()))


# --------------------------------------------------------------------------------------
# registry / decode
# --------------------------------------------------------------------------------------

NODE_TYPES: Dict[str, Type[Node]] = {
    cls.kind: cls  # type: ignore[misc]
    for cls in (
        Cube,
        RoundedCube,
        Cylinder,
        Sphere,
        Polyhedron,
        Import,
        Square,
        RoundedSquare,
        Circle,
        Polygon,
        LinearExtrude,
        RotateExtrude,
        Union,
        Difference,
        Intersection,
        Hull,
        Minkowski,
        Translate,
        Rotate,
        Scale,
        Mirror,
        Color,
        Group,
        ScadModule,
    )
}

PRIMITIVES_3D = {"cube", "rounded_cube", "cylinder", "sphere", "polyhedron", "import"}
PRIMITIVES_2D = {"square", "rounded_square", "circle", "polygon"}
EXTRUSIONS = {"linear_extrude", "rotate_extrude"}
BOOLEANS = {"union", "difference", "intersection", "hull", "minkowski"}
TRANSFORMS = {"translate", "rotate", "scale", "mirror", "color", "group"}
#: Calls into vendored OpenSCAD libraries. Renderable by the OpenSCAD backend only.
LIBRARY = {"scad_module"}


def from_dict(d: Dict[str, Any]) -> Node:
    """Inverse of `Node.to_dict()`. Raises ValueError on unknown kinds or bad shapes."""
    if not isinstance(d, dict) or "kind" not in d:
        raise ValueError(f"not a CSG node: {d!r}")
    kind = d["kind"]
    cls = NODE_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"unknown CSG node kind {kind!r}")
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name == "kind" or f.name not in d:
            continue
        v = d[f.name]
        if f.name == "children":
            v = [from_dict(c) for c in v]
        elif f.name == "args":
            if not isinstance(v, dict):
                raise ValueError("scad_module args must be a JSON object, got {!r}".format(v))
            v = dict(v)
        elif f.name in ("size", "v", "normal") and isinstance(v, (list, tuple)):
            v = _vec3(v) if len(v) == 3 else _vec2(v)
        elif f.name == "points":
            v = tuple(tuple(float(x) for x in p) for p in v)
        elif f.name in ("faces", "paths") and v is not None:
            v = tuple(tuple(int(i) for i in p) for p in v)
        elif f.name == "rgba":
            v = tuple(float(x) for x in v)
        elif f.name == "scale" and isinstance(v, (list, tuple)):
            v = _vec2(v)
        kwargs[f.name] = v
    return cls(**kwargs)  # type: ignore[arg-type]


def from_json(s: str) -> Node:
    import json

    return from_dict(json.loads(s))


def walk(node: Node):
    """Pre-order traversal."""
    yield node
    for c in getattr(node, "children", ()) or ():
        yield from walk(c)


def is_2d(node: Node) -> bool:
    """True if the node evaluates to 2D geometry (needed to validate extrusion children)."""
    k = node.kind
    if k in PRIMITIVES_2D:
        return True
    if k in PRIMITIVES_3D or k in EXTRUSIONS or k in LIBRARY:
        # A library call's dimensionality is not knowable without evaluating its source, and
        # its children are attachments rather than a profile, so it is never inferred as 2D.
        return False
    ch = getattr(node, "children", []) or []
    return bool(ch) and all(is_2d(c) for c in ch)


__all__ = [
    "Node",
    "Vec2",
    "Vec3",
    "Cube",
    "RoundedCube",
    "RoundedSquare",
    "Cylinder",
    "Sphere",
    "Polyhedron",
    "Import",
    "Square",
    "Circle",
    "Polygon",
    "LinearExtrude",
    "RotateExtrude",
    "Union",
    "Difference",
    "Intersection",
    "Hull",
    "Minkowski",
    "Translate",
    "Rotate",
    "Scale",
    "Mirror",
    "Color",
    "Group",
    "ScadModule",
    "JSONValue",
    "NODE_TYPES",
    "PRIMITIVES_2D",
    "PRIMITIVES_3D",
    "EXTRUSIONS",
    "BOOLEANS",
    "TRANSFORMS",
    "LIBRARY",
    "from_dict",
    "from_json",
    "walk",
    "is_2d",
]
