"""The script that runs *inside* FreeCAD's python — the only PhiMech code that imports Part.

Why a separate file instead of `-c '...'`
-----------------------------------------
`freecadcmd` executes a python file passed as its first positional argument. Feeding it code
on stdin or via `-c` makes it wait on stdin forever for anything non-trivial (so always pass a script file). So `phimech/backends/freecad.py` writes a job JSON, then runs
`freecadcmd <this file> <job.json>` as a child process.

Why this file must not import phimech
------------------------------------
It runs under FreeCAD's bundled interpreter (3.11.14 conda-forge in 1.1.3), whose `sys.path`
is not ours and whose numpy/protobuf ABI is not ours. Importing `phimech.csg.ast` would tie
the two interpreters together for no gain — the CSG AST already has a JSON form, so this
file just walks the dict by its `kind` discriminator. That duplication deliberately isolates the FreeCAD subprocess. Keep it stdlib + FreeCAD-only and 3.9-compatible
(no `match`, no `X | Y` annotations) so it stays readable next to the rest of the codebase.

Protocol
--------
Input: a JSON job file named by `$PHIMECH_JOB` (`sys.argv[1]` also works but see below)::

    {"node": <csg node dict>,          # required
     "outputs": {"step": "/abs.step", "stl": "/abs.stl"},   # zero or more, one pass
     "measure": true,                  # compute bbox/volume/area/solids
     "deflection": 0.05,               # mesh linear deflection, mm
     "angular": 0.2}                   # mesh angular deflection, rad

Output: exactly one marker line on stdout, because `freecadcmd` prints its banner and OCC
prints progress bars — the parent parses only the marker::

    PHIMECH_RESULT {"measurement": ..., "outputs": ..., "warnings": [...], "seconds": 1.2}
    PHIMECH_ERROR  {"error": "...", "type": "...", "traceback": "..."}

Exit code is 0 on success and 1 on failure. Note `freecadcmd` swallows uncaught exceptions
and *still* exits 0, so every failure path here must call `sys.exit(1)` explicitly.

Two `freecadcmd` quirks this file is shaped around (both probed on 1.1.3, both silent)
--------------------------------------------------------------------------------------
1. It executes the script with `__name__` set to the *file stem* (`freecad_worker`), not
   `__main__`. A conventional `if __name__ == "__main__":` guard therefore never fires and the
   process exits 0 having done nothing. Hence the unguarded `_bootstrap()` call at the bottom.
2. Every positional argument *after* the script is treated as a document to open. A job file
   ending in `.json` gets handed to FreeCAD's FEM YAML/JSON mesh importer, which dies with
   `invalid literal for int() with base 10: 'kind'`. So the parent passes the path in
   `$PHIMECH_JOB`; argv is only read as a fallback for hand-running the worker.

Semantics notes (differences from OpenSCAD that callers must know)
-----------------------------------------------------------------
* `segments` ($fn) is ignored: FreeCAD builds exact B-rep curves. A cylinder's volume here is
  the analytic one, while OpenSCAD returns the volume of an n-gon prism. Golden comparisons
  across backends must therefore use tolerances, not equality.
* `minkowski` is not implemented — OCC's `makeOffsetShape` cannot do a general Minkowski sum
  and its sphere-offset special case returns wrong volumes on a plain cube in 1.1.3. The
  parent advertises this in `FreeCADBackend.UNSUPPORTED` and routes those nodes to OpenSCAD.
* `hull` *is* implemented, via the scipy that ships inside FreeCAD 1.1.3 (scipy 1.16.3).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import FreeCAD  # type: ignore[import-not-found]
import Part  # type: ignore[import-not-found]

Vector = FreeCAD.Vector

RESULT_MARKER = "PHIMECH_RESULT"
ERROR_MARKER = "PHIMECH_ERROR"

# Mesh tessellation defaults; overridable per job so the pipeline can trade size for fidelity.
DEFAULT_DEFLECTION = 0.05
DEFAULT_ANGULAR = 0.2

MESH_FORMATS = ("stl", "obj", "off", "3mf", "amf")
MESH_SUFFIXES = (".stl", ".obj", ".off", ".3mf", ".amf", ".ply")
BREP_SUFFIXES = (".step", ".stp", ".iges", ".igs", ".brep", ".brp")

# Fonts probed in order for `text:` groups. macOS ships Arial in Supplemental; Helvetica.ttc
# is the fallback that exists on every install.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

WARNINGS = []  # type: list


class WorkerError(RuntimeError):
    """A modelling failure we can explain, as opposed to an OCC crash."""


def warn(message):
    # type: (str) -> None
    if message not in WARNINGS:
        WARNINGS.append(message)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _is_null(shape):
    # type: (object) -> bool
    return shape is None or (hasattr(shape, "isNull") and shape.isNull())


def _null_shape():
    # type: () -> object
    return Part.Shape()


def _vec(v):
    # type: (object) -> object
    return Vector(float(v[0]), float(v[1]), float(v[2] if len(v) > 2 else 0.0))


def _children(node, kind):
    # type: (dict, str) -> list
    """Build every child, dropping nulls — every op must tolerate empty groups."""
    shapes = []
    for child in node.get("children") or []:
        shape = build(child)
        if not _is_null(shape):
            shapes.append(shape)
    return shapes


def _fuse_all(shapes):
    # type: (list) -> object
    """Fuse a list into one shape. `removeSplitter` deletes the seam faces a fuse leaves
    behind, which matters because those seams survive into STEP and confuse downstream
    fillet/face selection."""
    if not shapes:
        return _null_shape()
    if len(shapes) == 1:
        return shapes[0]
    fused = shapes[0].multiFuse(shapes[1:])
    try:
        cleaned = fused.removeSplitter()
        if not _is_null(cleaned) and cleaned.isValid():
            return cleaned
    except Exception as exc:  # OCC is allowed to decline; the un-cleaned fuse is still valid
        warn("removeSplitter failed after union ({}); keeping seam faces".format(exc))
    return fused


def _as_face(shape, what):
    # type: (object, str) -> object
    """Coerce a 2D result (Face/Shell/Compound of faces) into something extrudable."""
    if _is_null(shape):
        raise WorkerError("{}: no 2D geometry to work with".format(what))
    if shape.ShapeType in ("Face", "Shell", "Compound", "CompSolid"):
        return shape
    if shape.ShapeType == "Wire":
        return Part.Face(shape)
    raise WorkerError("{}: expected 2D geometry, got a {}".format(what, shape.ShapeType))


def _face_from_points(points):
    # type: (list) -> object
    """Planar face from an ordered point loop, closing it if the caller did not.

    Falls back to a triangle fan when OCC refuses the loop, which happens for the slightly
    non-planar quads that meshy `polyhedron` input is full of.
    """
    pts = [_vec(p) for p in points]
    if len(pts) < 3:
        raise WorkerError("face needs >= 3 points, got {}".format(len(pts)))
    loop = list(pts)
    if (loop[0] - loop[-1]).Length > 1e-9:
        loop.append(loop[0])
    try:
        return Part.Face(Part.makePolygon(loop))
    except Exception:
        tris = []
        for i in range(1, len(pts) - 1):
            fan = [pts[0], pts[i], pts[i + 1], pts[0]]
            try:
                tris.append(Part.Face(Part.makePolygon(fan)))
            except Exception:
                continue  # degenerate sliver; the shell sews without it
        if not tris:
            raise WorkerError("could not build a face from {} points".format(len(pts)))
        warn("non-planar face triangulated into {} facets".format(len(tris)))
        return Part.Shell(tris) if len(tris) > 1 else tris[0]


def _solid_from_mesh(topology, tolerance=0.05):
    # type: (object, float) -> object
    """Mesh topology -> solid. Used by `import` of STL/OBJ/3MF and as the polyhedron fallback."""
    shape = Part.Shape()
    shape.makeShapeFromMesh(topology, tolerance)
    try:
        solid = Part.Solid(shape)
        if not _is_null(solid):
            return solid
    except Exception as exc:
        warn("mesh is not a closed solid ({}); keeping it as a shell".format(exc))
    return shape


def _find_font():
    # type: () -> object
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


# --------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------


def _build_cube(node):
    # type: (dict) -> object
    size = node.get("size", (1.0, 1.0, 1.0))
    x, y, z = (float(size[0]), float(size[1]), float(size[2]))
    if min(x, y, z) <= 0:
        raise WorkerError("cube size must be positive, got [{}, {}, {}]".format(x, y, z))
    shape = Part.makeBox(x, y, z)
    if node.get("center"):
        shape.translate(Vector(-x / 2.0, -y / 2.0, -z / 2.0))
    return shape


def _is_vertical(edge):
    # type: (object) -> bool
    """True for a straight edge parallel to Z.

    Checked from the endpoints rather than by curve type, because a fillet's own edges are not
    lines and this must stay correct when called on an already-filleted shape.
    """
    points = edge.Vertexes
    if len(points) != 2:
        return False
    delta = points[1].Point - points[0].Point
    if delta.Length < 1e-9:
        return False
    return abs(abs(delta.normalize().z) - 1.0) < 1e-6


def _build_rounded_cube(node):
    # type: (dict) -> object
    """A box with genuinely filleted edges -- the reason the `rounded_cube` node exists.

    OCC's `makeFillet` gives real cylindrical and toroidal faces, so a 60x40x25 rounded box is
    10 faces (6 planes + 4 cylinders) instead of the 130 planar facets the hull form produced.
    Verified identical to build123d's answer on the same box, to the last decimal: 59806.8583.

    Radii are clamped by the caller (`ast.RoundedCube.radii()`), and clamped again here because
    a worker must not trust its input. If a fillet still fails, the box is returned unfilleted
    rather than failing the whole render: a square-cornered part is wrong in a way a human sees
    immediately, while a failed render loses every other part in the build.
    """
    size = node.get("size", (1.0, 1.0, 1.0))
    x, y, z = (float(size[0]), float(size[1]), float(size[2]))
    if min(x, y, z) <= 0:
        raise WorkerError("rounded_cube size must be positive, got [{}, {}, {}]".format(x, y, z))

    horizontal = min(max(float(node.get("edge_r", 0.0) or 0.0), 0.0), x / 2.0, y / 2.0, z / 2.0)
    vertical = min(max(float(node.get("r", 0.0) or 0.0), 0.0), x / 2.0, y / 2.0)
    if vertical <= 0.0:
        vertical = horizontal

    shape = Part.makeBox(x, y, z)
    if vertical <= 0.0 and horizontal <= 0.0:
        return shape

    # A radius equal to half a side leaves zero flat face between two fillets, which OCC will
    # attempt and often fail on. Backing off by a hair keeps it solvable and is invisible.
    limit = 1e-4
    vertical = min(vertical, x / 2.0 - limit, y / 2.0 - limit) if vertical > 0 else 0.0
    horizontal = (min(horizontal, x / 2.0 - limit, y / 2.0 - limit, z / 2.0 - limit)
                  if horizontal > 0 else 0.0)

    try:
        if horizontal > 0.0 and abs(horizontal - vertical) < 1e-9:
            # Same radius everywhere: one pass over all twelve edges is more robust than two.
            return shape.makeFillet(vertical, shape.Edges)
        result = shape
        if vertical > 0.0:
            edges = [e for e in result.Edges if _is_vertical(e)]
            if edges:
                result = result.makeFillet(vertical, edges)
        if horizontal > 0.0:
            edges = [e for e in result.Edges if not _is_vertical(e) and e.Length > limit]
            if edges:
                result = result.makeFillet(horizontal, edges)
        return result
    except Exception as exc:  # noqa: BLE001 - see the docstring: degrade, do not lose the build
        warn("rounded_cube fillet failed ({}: {}); returning square corners".format(
            type(exc).__name__, exc))
        return Part.makeBox(x, y, z)


def _build_cylinder(node):
    # type: (dict) -> object
    h = float(node.get("h", 1.0))
    r1 = float(node.get("r", 1.0))
    r2 = node.get("r2")
    r2 = r1 if r2 is None else float(r2)
    if h <= 0:
        raise WorkerError("cylinder h must be positive, got {}".format(h))
    if min(r1, r2) < 0 or max(r1, r2) <= 0:
        raise WorkerError("cylinder needs a positive radius, got r={} r2={}".format(r1, r2))
    shape = Part.makeCylinder(r1, h) if abs(r1 - r2) < 1e-12 else Part.makeCone(r1, r2, h)
    if node.get("center"):
        shape.translate(Vector(0.0, 0.0, -h / 2.0))
    return shape


def _build_sphere(node):
    # type: (dict) -> object
    r = float(node.get("r", 1.0))
    if r <= 0:
        raise WorkerError("sphere r must be positive, got {}".format(r))
    return Part.makeSphere(r)


def _build_polyhedron(node):
    # type: (dict) -> object
    points = node.get("points") or ()
    faces = node.get("faces") or ()
    if not points or not faces:
        raise WorkerError("polyhedron needs both points and faces")
    built = []
    for face in faces:
        try:
            built.append(_face_from_points([points[i] for i in face]))
        except IndexError:
            raise WorkerError("polyhedron face {} indexes past {} points".format(
                list(face), len(points)))
        except WorkerError:
            continue
    if not built:
        raise WorkerError("polyhedron produced no usable faces")
    flat = []
    for shape in built:
        flat.extend(shape.Faces)
    try:
        solid = Part.Solid(Part.Shell(flat))
        if solid.isValid():
            return solid
    except Exception as exc:
        warn("polyhedron shell did not close ({}); rebuilding through a mesh".format(exc))
    verts = [_vec(p) for p in points]
    tris = []
    for face in faces:
        for i in range(1, len(face) - 1):
            tris.append((int(face[0]), int(face[i]), int(face[i + 1])))
    return _solid_from_mesh((verts, tris))


def _build_import(node):
    # type: (dict) -> object
    path = node.get("path") or ""
    if not path:
        raise WorkerError("import needs a path")
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        raise WorkerError("import: no such file {}".format(path))
    suffix = os.path.splitext(path)[1].lower()
    if suffix in BREP_SUFFIXES:
        shape = Part.Shape()
        shape.read(path)  # handles STEP/IGES/BREP without needing a document
        if _is_null(shape):
            raise WorkerError("import: {} contained no geometry".format(path))
        return shape
    if suffix in MESH_SUFFIXES:
        import Mesh  # type: ignore[import-not-found]  # lazy: only meshy imports pay for it

        mesh = Mesh.Mesh(path)
        if mesh.CountFacets == 0:
            raise WorkerError("import: {} contained no facets".format(path))
        return _solid_from_mesh(mesh.Topology)
    raise WorkerError("import: unsupported suffix {!r} ({})".format(suffix, path))


# --------------------------------------------------------------------------------------
# 2D primitives + extrusions
# --------------------------------------------------------------------------------------


def _build_square(node):
    # type: (dict) -> object
    size = node.get("size", (1.0, 1.0))
    x, y = float(size[0]), float(size[1])
    if min(x, y) <= 0:
        raise WorkerError("square size must be positive, got [{}, {}]".format(x, y))
    x0, y0 = (-x / 2.0, -y / 2.0) if node.get("center") else (0.0, 0.0)
    return _face_from_points([(x0, y0, 0), (x0 + x, y0, 0), (x0 + x, y0 + y, 0), (x0, y0 + y, 0)])


def _build_rounded_square(node):
    # type: (dict) -> object
    """A rectangular face with filleted corners.

    Built by filleting a thin box and taking its bottom face, rather than by assembling arcs and
    lines by hand. That reuses OCC's fillet solver, which already handles the tangency and the
    degenerate cases, and it cannot produce a face whose wire fails to close -- the failure mode
    of hand-built profiles. The box is 1 mm tall and discarded; only the face survives.

    Why it matters more than `rounded_cube`: the old `Hull([circles])` form was not merely
    faceted here, it was **unbuildable** by this backend, because a hull of coplanar points has
    no volume. One `rounded_rect` anywhere therefore pushed the entire part onto the
    mesh-to-solid fallback -- 13,758 faces for `rpi-zero-case-base`.
    """
    size = node.get("size", (1.0, 1.0))
    x, y = float(size[0]), float(size[1])
    if min(x, y) <= 0:
        raise WorkerError("rounded_square size must be positive, got [{}, {}]".format(x, y))
    limit = 1e-4
    r = min(max(float(node.get("r", 0.0) or 0.0), 0.0), x / 2.0 - limit, y / 2.0 - limit)
    x0, y0 = (-x / 2.0, -y / 2.0) if node.get("center") else (0.0, 0.0)
    if r <= 0.0:
        return _face_from_points(
            [(x0, y0, 0), (x0 + x, y0, 0), (x0 + x, y0 + y, 0), (x0, y0 + y, 0)]
        )
    try:
        block = Part.makeBox(x, y, 1.0)
        block.translate(Vector(x0, y0, 0.0))
        filleted = block.makeFillet(r, [e for e in block.Edges if _is_vertical(e)])
        # The bottom face is the one whose centre sits at z = 0.
        bottom = sorted(filleted.Faces, key=lambda f: f.CenterOfMass.z)[0]
        return Part.Face(bottom.OuterWire)
    except Exception as exc:  # noqa: BLE001 - a square profile beats losing the part
        warn("rounded_square fillet failed ({}: {}); using square corners".format(
            type(exc).__name__, exc))
        return _face_from_points(
            [(x0, y0, 0), (x0 + x, y0, 0), (x0 + x, y0 + y, 0), (x0, y0 + y, 0)]
        )


def _build_circle(node):
    # type: (dict) -> object
    r = float(node.get("r", 1.0))
    if r <= 0:
        raise WorkerError("circle r must be positive, got {}".format(r))
    return Part.Face(Part.Wire(Part.makeCircle(r)))


def _build_polygon(node):
    # type: (dict) -> object
    points = node.get("points") or ()
    if len(points) < 3:
        raise WorkerError("polygon needs >= 3 points, got {}".format(len(points)))
    paths = node.get("paths")
    if not paths:
        return _face_from_points(points)
    outer = _face_from_points([points[i] for i in paths[0]])
    for hole in paths[1:]:
        outer = outer.cut(_face_from_points([points[i] for i in hole]))
    return outer


def _build_linear_extrude(node):
    # type: (dict) -> object
    height = float(node.get("height", 1.0))
    if height <= 0:
        raise WorkerError("linear_extrude height must be positive, got {}".format(height))
    twist = float(node.get("twist", 0.0) or 0.0)
    if abs(twist) > 1e-9:
        raise WorkerError(
            "linear_extrude twist={} is not supported by the freecad backend "
            "(OCC has no twisted-prism primitive); render this node with openscad".format(twist))
    profile = _as_face(_fuse_all(_children(node, "linear_extrude")), "linear_extrude")
    scale = node.get("scale", 1.0)
    sx, sy = (float(scale), float(scale)) if isinstance(scale, (int, float)) else (
        float(scale[0]), float(scale[1]))
    if abs(sx - 1.0) < 1e-12 and abs(sy - 1.0) < 1e-12:
        solid = profile.extrude(Vector(0.0, 0.0, height))
    else:
        if min(sx, sy) <= 0:
            raise WorkerError("linear_extrude scale must be positive, got [{}, {}]".format(sx, sy))
        wires = profile.Wires
        if len(wires) != 1:
            raise WorkerError(
                "linear_extrude scale is only supported for a single-loop profile "
                "(this one has {} loops); render it with openscad".format(len(wires)))
        top = wires[0].copy()
        matrix = FreeCAD.Matrix()
        matrix.scale(sx, sy, 1.0)
        top = top.transformGeometry(matrix)
        top.translate(Vector(0.0, 0.0, height))
        solid = Part.makeLoft([wires[0], top.Wires[0]], True)
    if node.get("center"):
        solid.translate(Vector(0.0, 0.0, -height / 2.0))
    return solid


def _build_rotate_extrude(node):
    # type: (dict) -> object
    """OpenSCAD `rotate_extrude`: the XY profile is swung around the *Y* axis of its own
    plane, so profile-x becomes radius and profile-y becomes world z. Rotating the face 90
    degrees about X maps XY->XZ, after which a plain revolve about Z reproduces OpenSCAD.
    Checked against the analytic torus 2*pi^2*R*r^2 in tests/test_backend_freecad.py.
    """
    angle = float(node.get("angle", 360.0))
    if abs(angle) < 1e-9:
        raise WorkerError("rotate_extrude angle must be non-zero")
    profile = _as_face(_fuse_all(_children(node, "rotate_extrude")), "rotate_extrude")
    if profile.BoundBox.XMin < -1e-6:
        raise WorkerError(
            "rotate_extrude profile must lie at x >= 0, but its XMin is {:.4f}".format(
                profile.BoundBox.XMin))
    profile.rotate(Vector(0, 0, 0), Vector(1, 0, 0), 90)
    return profile.revolve(Vector(0, 0, 0), Vector(0, 0, 1), angle)


# --------------------------------------------------------------------------------------
# booleans
# --------------------------------------------------------------------------------------


def _build_union(node):
    # type: (dict) -> object
    return _fuse_all(_children(node, "union"))


def _build_difference(node):
    # type: (dict) -> object
    shapes = _children(node, "difference")
    if not shapes:
        return _null_shape()
    return shapes[0].cut(shapes[1:]) if len(shapes) > 1 else shapes[0]


def _build_intersection(node):
    # type: (dict) -> object
    shapes = _children(node, "intersection")
    if not shapes:
        return _null_shape()
    return shapes[0].common(shapes[1:]) if len(shapes) > 1 else shapes[0]


def _build_hull(node):
    # type: (dict) -> object
    """Convex hull via the scipy bundled with FreeCAD 1.1.3.

    OCC exposes no hull in 1.1.3 (`Part.makeConvexHull` and `Shape.makeConvexHull` are both
    absent — probed), so we tessellate the children, hull the point cloud, and sew the
    resulting triangles into a solid. Curved children become faceted, exactly as they do in
    OpenSCAD, so this is a fair match for the OpenSCAD backend rather than an exact B-rep.
    """
    shapes = _children(node, "hull")
    if not shapes:
        return _null_shape()
    try:
        from scipy.spatial import ConvexHull  # type: ignore[import-not-found]
    except ImportError:
        raise WorkerError(
            "hull needs scipy inside FreeCAD's python and it is missing here; "
            "render this node with the openscad backend")
    points = []
    for shape in shapes:
        verts = shape.tessellate(0.1)[0] if shape.Faces else [v.Point for v in shape.Vertexes]
        points.extend([(p.x, p.y, p.z) for p in verts])
    if len(points) < 4:
        raise WorkerError("hull needs >= 4 points, got {}".format(len(points)))
    # qhull refuses a point cloud that is flat (all points on one plane): it reports
    # "QH6154 ... Initial simplex is flat" and the raw dump tells the caller nothing about
    # which node failed or what to do. A hull of coplanar points has no volume, so it can
    # never become a solid here — detect it up front and say so in one actionable line.
    _extent = [
        max(p[axis] for p in points) - min(p[axis] for p in points) for axis in range(3)
    ]
    if min(_extent) <= 1e-9:
        raise WorkerError(
            "hull children are coplanar (extent {:.4g} x {:.4g} x {:.4g} mm): a flat hull has "
            "no volume, so it cannot be a solid. Give the hull some thickness, or render this "
            "node with the openscad backend, which hulls in 2D.".format(*_extent)
        )
    hull = ConvexHull(points)
    faces = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        a, b, c = [hull.points[i] for i in simplex]
        # qhull does not promise a winding; flip the triangle when its normal opposes the
        # outward plane equation, otherwise the shell has mixed orientation and Solid() fails.
        normal = (
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        )
        if sum(n * e for n, e in zip(normal, equation[:3])) < 0:
            b, c = c, b
        try:
            faces.append(Part.Face(Part.makePolygon([_vec(a), _vec(b), _vec(c), _vec(a)])))
        except Exception:
            continue  # degenerate coplanar sliver from qhull; the shell closes without it
    if not faces:
        raise WorkerError("hull produced no usable faces")
    solid = Part.Solid(Part.Shell(faces))
    try:
        cleaned = solid.removeSplitter()
        if not _is_null(cleaned) and cleaned.isValid() and cleaned.Solids:
            return cleaned
    except Exception:
        pass
    return solid


def _build_minkowski(node):
    # type: (dict) -> object
    raise WorkerError(
        "minkowski is not implemented in the freecad backend: OCC's makeOffsetShape cannot "
        "express a general Minkowski sum, and its sphere special case returns wrong volumes "
        "in 1.1.3. Render this node with the openscad backend "
        "(FreeCADBackend.UNSUPPORTED lists it, so `for_format` routes around it).")


def _build_scad_module(node):
    # type: (dict) -> object
    # Only reachable when the parent's can_render() check is bypassed (hand-run worker): say
    # *why* rather than "unknown kind", because the kind is known and deliberately refused.
    raise WorkerError(
        "scad_module {}() from <{}> is a call into an OpenSCAD library (BOSL2); FreeCAD has no "
        "OpenSCAD interpreter, so there is nothing to build a B-rep from. Render it with the "
        "openscad backend (FreeCADBackend.UNSUPPORTED lists it).".format(
            node.get("module", "?"), node.get("include", "?")))


# --------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------


def _build_translate(node):
    # type: (dict) -> object
    shape = _fuse_all(_children(node, "translate"))
    if _is_null(shape):
        return shape
    shape.translate(_vec(node.get("v", (0, 0, 0))))
    return shape


def _build_rotate(node):
    # type: (dict) -> object
    """OpenSCAD `rotate([x, y, z])` is X, then Y, then Z about the origin, in degrees."""
    shape = _fuse_all(_children(node, "rotate"))
    if _is_null(shape):
        return shape
    v = node.get("v", (0, 0, 0))
    origin = Vector(0, 0, 0)
    for angle, axis in (
        (float(v[0]), Vector(1, 0, 0)),
        (float(v[1]), Vector(0, 1, 0)),
        (float(v[2]), Vector(0, 0, 1)),
    ):
        if abs(angle) > 1e-12:
            shape.rotate(origin, axis, angle)
    return shape


def _build_scale(node):
    # type: (dict) -> object
    shape = _fuse_all(_children(node, "scale"))
    if _is_null(shape):
        return shape
    sx, sy, sz = [float(c) for c in node.get("v", (1, 1, 1))]
    if min(sx, sy, sz) <= 0:
        raise WorkerError("scale factors must be positive, got [{}, {}, {}]".format(sx, sy, sz))
    if abs(sx - sy) < 1e-12 and abs(sy - sz) < 1e-12:
        if abs(sx - 1.0) > 1e-12:
            shape.scale(sx)  # keeps exact B-rep geometry for the uniform case
        return shape
    matrix = FreeCAD.Matrix()
    matrix.scale(sx, sy, sz)
    return shape.transformGeometry(matrix)  # non-uniform: circles become ellipses (BSplines)


def _build_mirror(node):
    # type: (dict) -> object
    shape = _fuse_all(_children(node, "mirror"))
    if _is_null(shape):
        return shape
    normal = _vec(node.get("normal", (1, 0, 0)))
    if normal.Length < 1e-12:
        raise WorkerError("mirror normal must be non-zero")
    return shape.mirror(Vector(0, 0, 0), normal)


def _build_color(node):
    # type: (dict) -> object
    return _fuse_all(_children(node, "color"))  # display-only, never changes geometry


def _build_group(node):
    # type: (dict) -> object
    name = node.get("name") or ""
    children = _children(node, "group")
    if children:
        return _fuse_all(children)
    if name.startswith("text:"):
        return _build_text(name)
    return _null_shape()


def _build_text(spec):
    # type: (str) -> object
    """An empty `Group` named `text:...` becomes extruded solid text.

    Two accepted spellings, both documented here because there is no `Text` node in the AST
    and this is the escape hatch synth uses for lid labels:
      * `text:LABEL`                      -> size 6 mm, depth 1 mm (schema defaults)
      * `text:<size>:<depth>:LABEL`       -> explicit, e.g. `text:8:0.6:SAMPLE`
    The label may contain colons; only the first three fields are parsed.
    """
    body = spec[len("text:"):]
    size, depth = 6.0, 1.0
    parts = body.split(":", 2)
    if len(parts) == 3:
        try:
            size, depth, body = float(parts[0]), float(parts[1]), parts[2]
        except ValueError:
            body = spec[len("text:"):]  # not a numeric prefix; the whole thing is the label
    if not body:
        warn("empty text: group skipped")
        return _null_shape()
    font = _find_font()
    if font is None:
        warn("text {!r} skipped: no usable font found (tried {})".format(
            body, ", ".join(FONT_CANDIDATES)))
        return _null_shape()
    try:
        char_wires = Part.makeWireString(body, font, size)
    except Exception as exc:
        warn("text {!r} skipped: makeWireString failed ({})".format(body, exc))
        return _null_shape()
    faces = []
    for char in char_wires:
        wires = char if isinstance(char, (list, tuple)) else [char]
        closed = [w for w in wires if w.isClosed()]
        if not closed:
            continue
        outer = max(closed, key=lambda w: Part.Face(w).Area)
        face = Part.Face(outer)
        for wire in closed:
            if wire is not outer:
                face = face.cut(Part.Face(wire))  # counters of 'o', 'e', 'A'...
        faces.append(face)
    if not faces:
        warn("text {!r} produced no closed outlines".format(body))
        return _null_shape()
    profile = _fuse_all(faces)
    return profile.extrude(Vector(0.0, 0.0, depth))


BUILDERS = {
    "cube": _build_cube,
    "rounded_cube": _build_rounded_cube,
    "cylinder": _build_cylinder,
    "sphere": _build_sphere,
    "polyhedron": _build_polyhedron,
    "import": _build_import,
    "square": _build_square,
    "rounded_square": _build_rounded_square,
    "circle": _build_circle,
    "polygon": _build_polygon,
    "linear_extrude": _build_linear_extrude,
    "rotate_extrude": _build_rotate_extrude,
    "union": _build_union,
    "difference": _build_difference,
    "intersection": _build_intersection,
    "hull": _build_hull,
    "minkowski": _build_minkowski,
    "scad_module": _build_scad_module,
    "translate": _build_translate,
    "rotate": _build_rotate,
    "scale": _build_scale,
    "mirror": _build_mirror,
    "color": _build_color,
    "group": _build_group,
}


def build(node):
    # type: (dict) -> object
    """Interpret one CSG node dict into a `Part` shape. Mirrors `phimech.csg.ast.from_dict`
    without importing it (see the module docstring for why)."""
    if not isinstance(node, dict) or "kind" not in node:
        raise WorkerError("not a CSG node: {!r}".format(node))
    kind = node["kind"]
    builder = BUILDERS.get(kind)
    if builder is None:
        raise WorkerError("unknown CSG node kind {!r}; known kinds: {}".format(
            kind, ", ".join(sorted(BUILDERS))))
    return builder(node)


# --------------------------------------------------------------------------------------
# measurement + export
# --------------------------------------------------------------------------------------


def _bound_box(shape):
    # type: (object) -> object
    """Tight bbox, clamped to the conservative one.

    `Shape.BoundBox` is OCC's *conservative* box, derived from the surfaces' control polygons:
    a torus of outer radius 12 reports 12.9887. Gates compare bbox against print-bed size, so
    an 8%-oversized box would reject parts that actually fit. `optimalBoundingBox()` is exact
    and costs ~5 ms, so it is worth calling every time.

    The clamp handles the other direction: `optimalBoundingBox()` reuses the shape's *cached
    triangulation* if one exists, and a mesh export (`MeshPart.meshFromShape`) leaves one
    behind whose facets sit up to `deflection` outside the true surface. Measuring a meshed
    solid then reported ZMax 12.00497 for an exactly-12 box. `run()` measures before exporting
    for that reason, and since the true bbox is always a subset of the conservative one,
    clamping into it removes any residual mesh overshoot.
    """
    loose = shape.BoundBox
    try:
        optimal = shape.optimalBoundingBox()
    except Exception as exc:
        warn("optimalBoundingBox failed ({}); using the conservative BoundBox".format(exc))
        return loose
    if optimal is None:
        return loose
    return FreeCAD.BoundBox(
        max(optimal.XMin, loose.XMin), max(optimal.YMin, loose.YMin),
        max(optimal.ZMin, loose.ZMin),
        min(optimal.XMax, loose.XMax), min(optimal.YMax, loose.YMax),
        min(optimal.ZMax, loose.ZMax),
    )


def measure(shape):
    # type: (object) -> dict
    if _is_null(shape):
        return {"bbox": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
                "volume": 0.0, "area": 0.0, "solids": 0, "empty": True}
    bb = _bound_box(shape)
    volume = 0.0
    try:
        volume = float(shape.Volume)
    except Exception:
        volume = 0.0  # 2D results (Face/Shell) have no volume; area still carries meaning
    if not math.isfinite(volume):
        volume = 0.0
    return {
        "bbox": {"min": [bb.XMin, bb.YMin, bb.ZMin], "max": [bb.XMax, bb.YMax, bb.ZMax]},
        "volume": volume,
        "area": float(shape.Area),
        "solids": len(shape.Solids),
        "valid": bool(shape.isValid()),
        "shape_type": shape.ShapeType,
    }


def _mesh_of(shape, deflection, angular):
    # type: (object, float, float) -> object
    import MeshPart  # type: ignore[import-not-found]

    return MeshPart.meshFromShape(Shape=shape, LinearDeflection=deflection,
                                  AngularDeflection=angular, Relative=False)


def _export_fcstd(shape, node, path):
    # type: (object, dict, str) -> None
    """One `Part::Feature` per top-level Group child, labelled with its group name, so the
    lid/base/boss structure the synth pipeline built stays selectable in the FreeCAD GUI."""
    doc = FreeCAD.newDocument("phimech")
    try:
        pieces = []
        if node.get("kind") == "group":
            for index, child in enumerate(node.get("children") or []):
                label = child.get("name") if child.get("kind") == "group" else None
                built = build(child)
                if not _is_null(built):
                    pieces.append((label or "part{}".format(index + 1), built))
        if not pieces:
            pieces = [(node.get("name") or "solid", shape)]
        for label, piece in pieces:
            obj = doc.addObject("Part::Feature", "part")
            obj.Label = label
            obj.Shape = piece
        doc.recompute()
        doc.saveAs(path)
    finally:
        FreeCAD.closeDocument(doc.Name)


def export(shape, node, fmt, path, deflection, angular):
    # type: (object, dict, str, str, float, float) -> None
    fmt = fmt.lower().lstrip(".")
    if _is_null(shape):
        raise WorkerError("nothing to export: the node evaluated to empty geometry")
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    if fmt in ("step", "stp"):
        shape.exportStep(path)
    elif fmt == "brep":
        shape.exportBrep(path)
    elif fmt in MESH_FORMATS:
        _mesh_of(shape, deflection, angular).write(path)
    elif fmt == "fcstd":
        _export_fcstd(shape, node, path)
    elif fmt in ("png", "svg", "dxf"):
        raise WorkerError(
            "the freecad backend cannot write .{} headlessly (that needs the GUI's view "
            "provider); use the openscad backend for png".format(fmt))
    else:
        raise WorkerError("the freecad backend cannot write .{}".format(fmt))
    if not os.path.exists(path):
        raise WorkerError("export of .{} reported success but wrote no file".format(fmt))


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def run(job):
    # type: (dict) -> dict
    started = time.time()
    node = job.get("node")
    if node is None:
        raise WorkerError("job has no 'node'")
    deflection = float(job.get("deflection") or DEFAULT_DEFLECTION)
    angular = float(job.get("angular") or DEFAULT_ANGULAR)

    build_started = time.time()
    shape = build(node)
    build_seconds = time.time() - build_started

    # Measure BEFORE exporting: a mesh export caches a triangulation on the shape, and
    # `optimalBoundingBox()` would then measure those facets instead of the exact surface.
    measurement = measure(shape) if job.get("measure", True) else None

    outputs = {}
    for fmt, path in sorted((job.get("outputs") or {}).items()):
        export_started = time.time()
        export(shape, node, fmt, path, deflection, angular)
        outputs[fmt] = {
            "path": path,
            "bytes": os.path.getsize(path),
            "seconds": round(time.time() - export_started, 4),
        }

    result = {
        "outputs": outputs,
        "warnings": list(WARNINGS),
        "build_seconds": round(build_seconds, 4),
        "seconds": round(time.time() - started, 4),
        "freecad": ".".join(FreeCAD.Version()[:3]),
    }
    if measurement is not None:
        result["measurement"] = measurement
    return result


def _job_path(argv):
    # type: (list) -> object
    """`$PHIMECH_JOB` wins: a positional job path would also be opened as a document (quirk 2)."""
    env = os.environ.get("PHIMECH_JOB")
    if env:
        return env
    for arg in argv[1:]:
        if arg.lower().endswith(".json") and os.path.exists(arg):
            return arg
    return None


def main(argv):
    # type: (list) -> int
    job_path = _job_path(argv)
    try:
        if not job_path:
            raise WorkerError("no job file: pass it as argv[1] or set $PHIMECH_JOB")
        with open(job_path) as handle:
            job = json.load(handle)
        result = run(job)
    except Exception as exc:
        payload = {
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "warnings": list(WARNINGS),
        }
        sys.stdout.write("{} {}\n".format(ERROR_MARKER, json.dumps(payload)))
        sys.stdout.flush()
        return 1
    sys.stdout.write("{} {}\n".format(RESULT_MARKER, json.dumps(result)))
    sys.stdout.flush()
    return 0


def _bootstrap():
    # type: () -> None
    """Run the job if one was given. Deliberately *not* behind an `if __name__ == "__main__"`
    guard: `freecadcmd` imports this file under the name `freecad_worker`, so such a guard
    would never fire (quirk 1 in the module docstring). Gating on the presence of a job file
    instead keeps the module importable for introspection without side effects.
    """
    if _job_path(sys.argv) is None:
        return
    sys.exit(main(sys.argv))


_bootstrap()
