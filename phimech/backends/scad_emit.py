"""Deterministic OpenSCAD source generation for the PhiMech CSG AST.

The emitter is deliberately independent of the OpenSCAD executable.  Keeping all source
formatting here gives renders, cache keys, and checked-in golden files the same stable text.

Library calls (`ScadModule`) add exactly two things to the program: `include <…>` lines directly
under the header, deduplicated and with `BOSL2/std.scad` first, and `module(name=literal, …)`
statements whose arguments are formatted by `_literal()`.  The include paths are resolved by the
OpenSCAD backend through `OPENSCADPATH`, never by absolute path, so the emitted text is the same
on every machine.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from phimech.csg.ast import (
    Circle,
    Cube,
    Cylinder,
    Group,
    Import,
    LinearExtrude,
    Node,
    Polygon,
    Polyhedron,
    RotateExtrude,
    ScadModule,
    Sphere,
    Square,
    walk,
)
from phimech.backends.base import BackendError
from phimech.vendor import bosl2


_HEADER_VERSION = "0.1.0"


def _phimech_version() -> str:
    """Return the package version without making source checkouts require installation."""
    try:
        return metadata.version("phimech")
    except metadata.PackageNotFoundError:
        return _HEADER_VERSION


def _number(value: Any) -> str:
    """Format a number in a compact, stable form accepted by OpenSCAD."""
    number = float(value)
    if number == 0.0:
        return "0"
    text = repr(number)
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e", 1)
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return mantissa + "e" + exponent
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _vector(values: Sequence[Any]) -> str:
    return "[" + ", ".join(_number(value) for value in values) + "]"


def _matrix(values: Iterable[Sequence[Any]]) -> str:
    return "[" + ", ".join(_vector(value) for value in values) + "]"


def _string(value: Any) -> str:
    """Quote a string using JSON escaping, which is also valid OpenSCAD string syntax."""
    return json.dumps(str(value), ensure_ascii=False)


def _comment(value: Any) -> str:
    """Keep safe labels readable and quote controls that could terminate a line comment.

    Group names are metadata, not source.  JSON quoting makes every line separator and C0/C1
    control visible instead of letting it create another OpenSCAD statement.  Printable labels
    retain their historical form so ordinary source and golden files do not churn.
    """
    text = str(value)
    controls = any(
        ord(character) < 0x20
        or 0x7F <= ord(character) <= 0x9F
        or character in ("\u2028", "\u2029")
        for character in text
    )
    if not controls and not text.endswith("\\"):
        return text
    return json.dumps(text, ensure_ascii=True).replace("\x7f", "\\u007f")


def _fn(value: Optional[int], default: int) -> int:
    return default if value is None else int(value)


def _literal(value: Any) -> str:
    """Format a JSON value as the OpenSCAD literal a library module expects.

    `bool` is tested before numbers because it is an `int` subclass and would otherwise print as
    `1`; `None` becomes `undef`, which BOSL2 reads as "argument not given".  Lists nest, so a
    point list, a `[["type", "circle"], ["r", 2]]` struct and a `skin()` profile stack all work.
    """
    if value is None:
        return "undef"
    if isinstance(value, bool):
        return _bool(value)
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    raise BackendError(
        "scad_module argument has no OpenSCAD literal: {!r} ({})".format(value, type(value).__name__)
    )


def _scad_module_prefix(node: ScadModule) -> str:
    """`name(a=1, b="x")` for a library call, re-checking the identifiers the validator checks.

    Emission does not run validation (it is the pure half), so the same two regexes guard it
    independently: a module name or argument name that is not an identifier is the only way a
    node could inject source here, and it is refused rather than escaped.
    """
    if not bosl2.MODULE_NAME.match(node.module or ""):
        raise BackendError("scad_module name is not an identifier: {!r}".format(node.module))
    parts: List[str] = []
    for name, value in node.args.items():
        if not isinstance(name, str) or not bosl2.ARG_NAME.match(name):
            raise BackendError("scad_module argument name is not an identifier: {!r}".format(name))
        parts.append(name + "=" + _literal(value))
    return node.module + "(" + ", ".join(parts) + ")"


def includes_for(node: Node) -> List[str]:
    """The `include <…>` paths a tree needs, deduplicated, in the order they will be emitted.

    `BOSL2/std.scad` comes first whenever any BOSL2 file is used (the library requires it);
    after that, first use wins.  Paths are checked as text only — the validator has already
    required them to exist inside `phimech/vendor`, and this half must stay filesystem-free.
    """
    ordered: List[str] = []
    for sub in walk(node):
        if not isinstance(sub, ScadModule):
            continue
        for include in bosl2.required_includes(sub.include):
            if not bosl2.INCLUDE_TEXT.match(include) or any(
                part in ("", ".", "..") for part in include.split("/")
            ):
                raise BackendError("scad_module include is not a vendored path: {!r}".format(include))
            if include not in ordered:
                ordered.append(include)
    return ordered


def _block(
    prefix: str, children: Sequence[Node], level: int, fn_default: int
) -> List[str]:
    """Render a module block while preserving child order and indentation."""
    indent = "  " * level
    lines = [indent + prefix + " {"]
    for child in children:
        lines.extend(_emit_node(child, level + 1, fn_default))
    lines.append(indent + "}")
    return lines


def _text_group(node: Group, level: int) -> Optional[List[str]]:
    """Render the compact text convention, or return None for an ordinary group."""
    if not node.name or not node.name.startswith("text:"):
        return None
    try:
        spec = json.loads(node.name[len("text:") :])
    except (TypeError, ValueError) as exc:
        raise BackendError("invalid text group JSON") from exc
    if not isinstance(spec, dict):
        raise BackendError("text group JSON must be an object")
    required = ("text", "size", "depth", "font", "halign", "valign")
    missing = [key for key in required if key not in spec]
    if missing:
        raise BackendError("text group is missing keys: " + ", ".join(missing))
    indent = "  " * level
    args = (
        "height="
        + _number(spec["depth"])
        + ") text("
        + _string(spec["text"])
        + ", size="
        + _number(spec["size"])
        + ", font="
        + _string(spec["font"])
        + ", halign="
        + _string(spec["halign"])
        + ", valign="
        + _string(spec["valign"])
        + ");"
    )
    return [indent + "linear_extrude(" + args]


def _emit_node(node: Node, level: int, fn_default: int) -> List[str]:
    """Render one AST node without the file header."""
    if not isinstance(node, Node):
        raise BackendError("cannot emit a non-CSG node: {!r}".format(node))
    indent = "  " * level
    kind = node.kind

    if kind == "cube":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Cube)
        return [
            indent
            + "cube(size="
            + _vector(value.size)
            + ", center="
            + _bool(value.center)
            + ");"
        ]
    if kind == "rounded_cube":
        # OpenSCAD has no fillet, so this desugars to the convex hull this lane has always
        # emitted -- byte-identical output, verified by the golden renders. The FreeCAD lane
        # fillets the same node natively instead; that divergence is the point of the node.
        from phimech.csg.ops import rounded_box_as_hull

        return _emit_node(rounded_box_as_hull(node), level, fn_default)  # type: ignore[arg-type]
    if kind == "rounded_square":
        # Same reasoning as `rounded_cube`: OpenSCAD has no 2D fillet, so emit the hull of
        # circles it always emitted, byte-identical. FreeCAD fillets a face instead.
        from phimech.csg.ops import rounded_rect_as_hull

        return _emit_node(rounded_rect_as_hull(node), level, fn_default)  # type: ignore[arg-type]
    if kind == "cylinder":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Cylinder)
        r2 = value.r if value.r2 is None else value.r2
        return [
            indent
            + "cylinder(h="
            + _number(value.h)
            + ", r1="
            + _number(value.r)
            + ", r2="
            + _number(r2)
            + ", center="
            + _bool(value.center)
            + ", $fn="
            + str(_fn(value.segments, fn_default))
            + ");"
        ]
    if kind == "sphere":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Sphere)
        return [
            indent
            + "sphere(r="
            + _number(value.r)
            + ", $fn="
            + str(_fn(value.segments, fn_default))
            + ");"
        ]
    if kind == "polyhedron":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Polyhedron)
        return [
            indent
            + "polyhedron(points="
            + _matrix(value.points)
            + ", faces="
            + _matrix(value.faces)
            + ");"
        ]
    if kind == "import":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Import)
        source = Path(value.path).expanduser().resolve()
        if source.suffix.lower() in (".step", ".stp"):
            raise BackendError("STEP import requires the freecad backend")
        return [indent + "import(" + _string(str(source)) + ");"]
    if kind == "square":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Square)
        return [
            indent
            + "square(size="
            + _vector(value.size)
            + ", center="
            + _bool(value.center)
            + ");"
        ]
    if kind == "circle":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Circle)
        return [
            indent
            + "circle(r="
            + _number(value.r)
            + ", $fn="
            + str(_fn(value.segments, fn_default))
            + ");"
        ]
    if kind == "polygon":
        value = node  # type: ignore[assignment]
        assert isinstance(value, Polygon)
        line = indent + "polygon(points=" + _matrix(value.points)
        if value.paths is not None:
            line += ", paths=" + _matrix(value.paths)
        return [line + ");"]
    if kind == "linear_extrude":
        value = node  # type: ignore[assignment]
        assert isinstance(value, LinearExtrude)
        scale = (
            _number(value.scale)
            if isinstance(value.scale, (int, float))
            else _vector(value.scale)
        )
        prefix = (
            "linear_extrude(height="
            + _number(value.height)
            + ", center="
            + _bool(value.center)
            + ", twist="
            + _number(value.twist)
            + ", scale="
            + scale
            + ")"
        )
        return _block(prefix, value.children, level, fn_default)
    if kind == "rotate_extrude":
        value = node  # type: ignore[assignment]
        assert isinstance(value, RotateExtrude)
        prefix = (
            "rotate_extrude(angle="
            + _number(value.angle)
            + ", $fn="
            + str(_fn(value.segments, fn_default))
            + ")"
        )
        return _block(prefix, value.children, level, fn_default)
    if kind in ("union", "difference", "intersection", "hull", "minkowski"):
        value = node  # type: ignore[assignment]
        name = kind
        return _block(name + "()", value.children, level, fn_default)  # type: ignore[attr-defined]
    if kind in ("translate", "rotate", "scale"):
        value = node  # type: ignore[assignment]
        # `rotate` MUST be emitted positionally. In OpenSCAD `rotate(v=[x,y,z])` names the
        # rotation *axis* (and needs a separate `a=` angle), so a named `v=` silently applies
        # NO rotation at all: `rotate(v=[0,90,0]) cylinder(h=20)` measures 20 mm on Z, while
        # `rotate([0,90,0])` correctly measures 20 mm on X. Emitting `v=` collapsed every
        # rounded box into a wedge. translate/scale accept `v=` fine, but positional is
        # uniform and equally valid, so all three use it.
        prefix = kind + "(" + _vector(value.v) + ")"
        return _block(prefix, value.children, level, fn_default)  # type: ignore[attr-defined]
    if kind == "mirror":
        value = node  # type: ignore[assignment]
        return _block(
            "mirror(v=" + _vector(value.normal) + ")", value.children, level, fn_default
        )  # type: ignore[attr-defined]
    if kind == "color":
        value = node  # type: ignore[assignment]
        return _block(
            "color(" + _vector(value.rgba) + ")", value.children, level, fn_default
        )  # type: ignore[attr-defined]
    if kind == "group":
        value = node  # type: ignore[assignment]
        text_lines = _text_group(value, level)  # type: ignore[arg-type]
        if text_lines is not None:
            return text_lines
        name = value.name if value.name is not None else "group"
        return [indent + "// " + _comment(name)] + _block(
            "union()", value.children, level, fn_default
        )
    if kind == "scad_module":
        value = node  # type: ignore[assignment]
        assert isinstance(value, ScadModule)
        prefix = _scad_module_prefix(value)
        if value.children:
            return _block(prefix, value.children, level, fn_default)
        return [indent + prefix + ";"]
    raise BackendError("unsupported CSG node kind {!r}".format(kind))


def emit(node: Node, *, fn_default: int = 64) -> str:
    """Return stable, readable OpenSCAD source for every PhiMech AST node.

    Trees without a library call emit exactly what they always did; the golden renders depend
    on that.  Trees with one gain `include <…>` lines between the header and the body.
    """
    if fn_default <= 0:
        raise ValueError("fn_default must be positive")
    body = _emit_node(node, 0, int(fn_default))
    header = "// generated by phimech " + _phimech_version() + " — do not edit"
    includes = ["include <" + include + ">" for include in includes_for(node)]
    return header + "\n" + "\n".join(includes + body) + "\n"


def emit_file(node: Node, path: Path) -> Path:
    """Write emitted source to `path` and return its path for pipeline convenience."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(emit(node), encoding="utf-8")
    return destination


__all__ = ["emit", "emit_file", "includes_for"]
