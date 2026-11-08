"""Structural validation and summaries for the backend-neutral CSG AST.

Validation is intentionally about the tree and its contracts, not about exact CAD-kernel
manifoldness.  The render backends and mesh gates own geometric truth; this module catches
bad dimensions, malformed references, and nodes that cannot be interpreted consistently.
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional

from phimech.csg.ast import (
    BOOLEANS,
    EXTRUSIONS,
    LIBRARY,
    PRIMITIVES_2D,
    PRIMITIVES_3D,
    TRANSFORMS,
    Group,
    Import,
    Node,
    Polygon,
    Polyhedron,
    ScadModule,
    is_2d,
)

_ALLOWED_IMPORT_SUFFIXES = {".stl", ".step", ".stp", ".3mf", ".obj", ".off", ".amf", ".dxf", ".svg"}

#: Depth at which a nested `scad_module` argument stops being a plausible point list or struct
#: and starts being a mistake. BOSL2's deepest common shape is `profiles` for `skin()`: a list
#: of profiles, each a list of points, each a list of numbers — three levels.
_MAX_ARG_DEPTH = 4


def _finite_errors(value: Any, path: str, errors: List[str]) -> None:
    """Collect non-finite numeric leaves without depending on a third-party validator."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            errors.append("{}: value must be finite".format(path))
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_errors(child, "{}.{}".format(path, key), errors)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_errors(child, "{}[{}]".format(path, index), errors)


def _positive(value: Any, label: str, errors: List[str]) -> None:
    try:
        if float(value) <= 0.0:
            errors.append("{} must be > 0".format(label))
    except (TypeError, ValueError, OverflowError):
        errors.append("{} must be a positive number".format(label))


def _nonnegative(value: Any, label: str, errors: List[str]) -> None:
    try:
        if float(value) < 0.0:
            errors.append("{} must be >= 0".format(label))
    except (TypeError, ValueError, OverflowError):
        errors.append("{} must be a non-negative number".format(label))


def _vector_length(value: Any, expected: int, label: str, errors: List[str]) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        errors.append("{} must have {} components".format(label, expected))


def _children(node: Node, path: str, errors: List[str]) -> List[Node]:
    children = getattr(node, "children", None)
    if children is None:
        return []
    if not isinstance(children, (list, tuple)):
        errors.append("{} children must be a list".format(path))
        return []
    result: List[Node] = []
    for index, child in enumerate(children):
        if not isinstance(child, Node):
            errors.append("{} children[{}] is not a CSG node".format(path, index))
        else:
            result.append(child)
    return result


def validate(node: Node) -> List[str]:
    """Return human-readable structural errors and ``warn:`` diagnostics for ``node``.

    A one-child ``Difference`` is retained as a warning rather than an error because it is
    useful while a synth pipeline is assembling optional cutters. An empty ``Group`` is an
    intentional semantic placeholder; a ``text:`` group carries backend-rendered text payload.
    """
    errors: List[str] = []

    if not isinstance(node, Node):
        return ["root is not a CSG node"]

    def visit(current: Node, path: str) -> None:
        # to_dict is the canonical recursive shape, so checking it catches NaN/inf in
        # vectors, dimensions, and any future numeric AST field as well.
        try:
            _finite_errors(current.to_dict(), path, errors)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append("{} could not be encoded: {}".format(path, exc))

        kind = current.kind
        children = _children(current, path, errors)

        if kind in BOOLEANS or kind in TRANSFORMS or kind in EXTRUSIONS:
            if not children:
                if isinstance(current, Group):
                    pass
                else:
                    errors.append("{} ({}) must have at least one child".format(path, kind))
            if kind == "difference" and children and len(children) < 2:
                errors.append(
                    "warn: {} difference has fewer than two children; no real cut".format(path)
                )

        if kind in EXTRUSIONS:
            for index, child in enumerate(children):
                if not is_2d(child):
                    errors.append("{} children[{}] of {} must be 2D".format(path, index, kind))

        if kind == "cube":
            _vector_length(current.size, 3, "{}.size".format(path), errors)
            for index, value in enumerate(current.size):
                _positive(value, "{}.size[{}]".format(path, index), errors)
        elif kind == "cylinder":
            _positive(current.h, "{}.h".format(path), errors)
            if current.r2 is None:
                _positive(current.r, "{}.r".format(path), errors)
            else:
                _nonnegative(current.r, "{}.r".format(path), errors)
                _nonnegative(current.r2, "{}.r2".format(path), errors)
                try:
                    if max(float(current.r), float(current.r2)) <= 0.0:
                        errors.append("{} needs at least one positive radius".format(path))
                except (TypeError, ValueError, OverflowError):
                    pass
        elif kind == "sphere":
            _positive(current.r, "{}.r".format(path), errors)
        elif kind == "square":
            _vector_length(current.size, 2, "{}.size".format(path), errors)
            for index, value in enumerate(current.size):
                _positive(value, "{}.size[{}]".format(path, index), errors)
        elif kind == "circle":
            _positive(current.r, "{}.r".format(path), errors)
        elif kind == "linear_extrude":
            _positive(current.height, "{}.height".format(path), errors)
        elif kind == "rotate_extrude":
            _positive(current.angle, "{}.angle".format(path), errors)
        elif kind == "scale":
            _vector_length(current.v, 3, "{}.v".format(path), errors)
            for index, value in enumerate(current.v):
                try:
                    if float(value) == 0.0:
                        errors.append("{} scale factor[{}] must be nonzero".format(path, index))
                except (TypeError, ValueError, OverflowError):
                    errors.append("{} scale factor[{}] must be numeric".format(path, index))
        elif kind == "import":
            _validate_import(current, path, errors)
        elif kind == "polygon":
            _validate_polygon(current, path, errors)
        elif kind == "polyhedron":
            _validate_polyhedron(current, path, errors)
        elif kind == "scad_module":
            _validate_scad_module(current, path, errors)
        elif kind not in PRIMITIVES_2D | PRIMITIVES_3D | EXTRUSIONS | BOOLEANS | TRANSFORMS | LIBRARY:
            errors.append("{} has unknown node kind {!r}".format(path, kind))

        for index, child in enumerate(children):
            visit(child, "{}.children[{}]".format(path, index))

    visit(node, "root")
    return errors


def _validate_import(node: Import, path: str, errors: List[str]) -> None:
    if not isinstance(node.path, str):
        errors.append("{} import path must be a string".format(path))
        return
    path_obj = Path(node.path)
    if not path_obj.is_absolute():
        errors.append("{} import path must be absolute".format(path))
    if not path_obj.exists():
        errors.append("{} import path does not exist: {}".format(path, node.path))
    if path_obj.suffix.lower() not in _ALLOWED_IMPORT_SUFFIXES:
        errors.append(
            "{} import suffix {!r} is not allowed (expected {})".format(
                path, path_obj.suffix, ", ".join(sorted(_ALLOWED_IMPORT_SUFFIXES))
            )
        )


def _polygon_loop_area(points: Any, indices: List[int]) -> Optional[float]:
    """Return signed shoelace area, or ``None`` when point values are not numeric."""
    try:
        area_twice = 0.0
        for index, point_index in enumerate(indices):
            current = points[point_index]
            following = points[indices[(index + 1) % len(indices)]]
            area_twice += float(current[0]) * float(following[1])
            area_twice -= float(following[0]) * float(current[1])
        return area_twice / 2.0
    except (IndexError, TypeError, ValueError, OverflowError):
        return None


def _validate_polygon(node: Polygon, path: str, errors: List[str]) -> None:
    if len(node.points) < 3:
        errors.append("{} polygon needs at least three points".format(path))
    points_are_pairs = True
    for index, point in enumerate(node.points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            points_are_pairs = False
        _vector_length(point, 2, "{}.points[{}]".format(path, index), errors)

    if node.paths is None:
        loops = [("outline", list(range(len(node.points))))]
    else:
        loops = [
            ("paths[{}]".format(path_index), list(loop))
            for path_index, loop in enumerate(node.paths)
        ]
        if not loops:
            errors.append("{} paths must contain at least one loop".format(path))

    point_count = len(node.points)
    for label, loop in loops:
        if len(loop) < 3:
            errors.append("{} {} needs at least three indices".format(path, label))
            continue
        indices_are_valid = True
        for point_index, value in enumerate(loop):
            if not isinstance(value, int) or value < 0 or value >= point_count:
                errors.append(
                    "{} {}[{}] references invalid point {}".format(
                        path, label, point_index, value
                    )
                )
                indices_are_valid = False
        if not indices_are_valid or not points_are_pairs:
            continue
        area = _polygon_loop_area(node.points, loop)
        if area == 0.0:
            errors.append("{} polygon {} encloses zero area".format(path, label))


def _arg_value_errors(value: Any, label: str, depth: int, errors: List[str]) -> None:
    """A `scad_module` argument must have an OpenSCAD literal: scalar, string, or nested list."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, numbers.Real):
        return  # finiteness is already checked over to_dict() by the visitor
    if isinstance(value, (list, tuple)):
        if depth >= _MAX_ARG_DEPTH:
            errors.append(
                "{} is nested more than {} levels deep; a point list is two, skin() profiles are "
                "three".format(label, _MAX_ARG_DEPTH)
            )
            return
        for index, item in enumerate(value):
            _arg_value_errors(item, "{}[{}]".format(label, index), depth + 1, errors)
        return
    errors.append(
        "{} must be a JSON scalar or array (OpenSCAD has no literal for {}); BOSL2 structs "
        "are lists of [name, value] pairs".format(label, type(value).__name__)
    )


def _validate_scad_module(node: ScadModule, path: str, errors: List[str]) -> None:
    """The three ways a library call could smuggle text into the .scad program, closed here.

    The module name and each argument name are checked against identifier patterns so neither
    can carry a `;` or a call; the include is resolved by `phimech.vendor.bosl2` so it cannot
    leave `phimech/vendor/`; values are restricted to what the emitter formats as literals.
    """
    from phimech.vendor import bosl2

    if not isinstance(node.module, str) or not bosl2.MODULE_NAME.match(node.module):
        errors.append(
            "{}.module {!r} must be an OpenSCAD identifier (letters, digits, '_'; not starting "
            "with a digit)".format(path, node.module)
        )
    try:
        bosl2.resolve_include(node.include)
    except bosl2.IncludeError as exc:
        errors.append("{}.include: {}".format(path, exc))
    if not isinstance(node.args, dict):
        errors.append("{}.args must be a mapping of argument name to value".format(path))
        return
    for name, value in node.args.items():
        if not isinstance(name, str) or not bosl2.ARG_NAME.match(name):
            errors.append(
                "{}.args has invalid argument name {!r} (identifier, optionally starting with "
                "'$' as in $fn)".format(path, name)
            )
        _arg_value_errors(value, "{}.args[{!r}]".format(path, name), 0, errors)


def _validate_polyhedron(node: Polyhedron, path: str, errors: List[str]) -> None:
    for index, point in enumerate(node.points):
        _vector_length(point, 3, "{}.points[{}]".format(path, index), errors)
    point_count = len(node.points)
    for face_index, face in enumerate(node.faces):
        if len(face) < 3:
            errors.append("{} face[{}] needs at least three indices".format(path, face_index))
        for index, value in enumerate(face):
            if not isinstance(value, int) or value < 0 or value >= point_count:
                errors.append(
                    "{} face[{}][{}] references invalid point {}".format(
                        path, face_index, index, value
                    )
                )


def stats(node: Node) -> Dict[str, Any]:
    """Return node counts, maximum tree depth, and import paths in traversal order."""
    if not isinstance(node, Node):
        raise ValueError("stats expects a CSG node")
    counts: Dict[str, int] = {}
    imports: List[str] = []
    max_depth = 0

    def visit(current: Node, depth: int) -> None:
        nonlocal max_depth
        counts[current.kind] = counts.get(current.kind, 0) + 1
        max_depth = max(max_depth, depth)
        if isinstance(current, Import):
            imports.append(current.path)
        for child in getattr(current, "children", ()) or ():
            if isinstance(child, Node):
                visit(child, depth + 1)

    visit(node, 1)
    return {"counts": counts, "depth": max_depth, "imports": imports}


def assert_valid(node: Node) -> None:
    """Raise ``ValueError`` for errors while allowing explicit ``warn:`` diagnostics."""
    errors = [message for message in validate(node) if not message.startswith("warn:")]
    if errors:
        raise ValueError("; ".join(errors))


__all__ = ["validate", "stats", "assert_valid"]
