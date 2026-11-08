"""Tolerant aliases for the design IR: forgive high-confidence misspellings, refuse ambiguity.

Why this exists
---------------
Models writing the IR misspell keys and units in predictable ways: ``diameter`` for ``d``,
``height`` for ``h``, ``minWall`` for ``min_wall``, ``"25 mm"`` for ``25``.  agentcad measured 35
failures in one benchmark run from exactly this class of error and fixed it by *accepting* the
spelling and normalising it - tolerant parsing beats more prompt text.  This module is PhiMech's
version of that fix, with two rules that keep it honest:

1. **Only high-confidence rewrites are applied**, and every one is recorded as
   ``{path, from, to}`` so a caller can see what was changed.  A design that was already valid
   goes through untouched (``normalisations == []``) and is byte-identical afterwards.
2. **Ambiguity is refused, never guessed.**  ``"w": 40`` alone does not say which axis; a scalar
   where ``[x, y, z]`` is required could be a cube or a typo; ``"2 ft"`` is a unit this IR does
   not speak.  Each refusal carries a paste-ready corrected fragment with the unknowns marked
   (``{"size": [40, "<y?>", "<z?>"]}``) and the schema pointer it was judged against.

The canonical key list is derived from ``design-ir.schema.json`` at import time, so the alias
table cannot drift from the schema: a row whose canonical key is not in the schema fails the
import, and the derived rows (camelCase and ``_mm``-suffix spellings) follow the schema
automatically.

Everything here is stdlib and imports under python 3.9, like the rest of ``phimech/ir``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)


_SCHEMA_PATH = Path(__file__).with_name("design-ir.schema.json")
with _SCHEMA_PATH.open("r", encoding="utf-8") as _schema_file:
    SCHEMA: Dict[str, Any] = json.load(_schema_file)


# --------------------------------------------------------------------------------------
# schema walking: which keys exist where
# --------------------------------------------------------------------------------------


def _deref(schema: Mapping[str, Any], pointer: str) -> Tuple[Dict[str, Any], str]:
    """Follow ``$ref`` chains, returning the resolved schema and the pointer it lives at.

    The pointer is what a refusal reports as ``schema``: ``#/$defs/source/oneOf/0`` tells a reader
    exactly which rule the input was judged against, which a prose message cannot.
    """
    current: Dict[str, Any] = dict(schema)
    seen: Set[str] = set()
    while "$ref" in current:
        ref = str(current["$ref"])
        if ref in seen or not ref.startswith("#/"):
            raise ValueError("unsupported schema reference: %s" % ref)
        seen.add(ref)
        node: Any = SCHEMA
        for part in ref[2:].split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
        if not isinstance(node, dict):
            raise ValueError("schema reference does not point to an object: %s" % ref)
        current, pointer = dict(node), ref
    return current, pointer


def _object_schemas() -> List[Tuple[str, Dict[str, Any]]]:
    """Every object schema that declares properties, with its resolved pointer.

    ``oneOf`` branches are included individually because the six source providers declare
    different keys: ``d`` exists for a cylinder and nowhere else, which is what lets
    ``diameter`` be a safe alias there and an unknown key everywhere else.
    """
    found: List[Tuple[str, Dict[str, Any]]] = []
    visited: Set[str] = set()

    def visit(schema: Mapping[str, Any], pointer: str) -> None:
        resolved, pointer = _deref(schema, pointer)
        if pointer in visited:
            return
        visited.add(pointer)
        properties = resolved.get("properties")
        if isinstance(properties, dict):
            found.append((pointer, resolved))
            for name, sub_schema in properties.items():
                if isinstance(sub_schema, dict):
                    visit(sub_schema, pointer + "/properties/" + name)
        additional = resolved.get("additionalProperties")
        if isinstance(additional, dict):
            visit(additional, pointer + "/additionalProperties")
        items = resolved.get("items")
        if isinstance(items, dict):
            visit(items, pointer + "/items")
        for index, branch in enumerate(resolved.get("oneOf", ())):
            if isinstance(branch, dict):
                visit(branch, "%s/oneOf/%d" % (pointer, index))

    visit(SCHEMA, "#")
    return found


_OBJECT_SCHEMAS: List[Tuple[str, Dict[str, Any]]] = _object_schemas()

#: Every property name the schema declares anywhere.  The alias table is checked against this
#: set at import time, which is what stops it drifting from the schema.
CANONICAL_KEYS: FrozenSet[str] = frozenset(
    key for _, schema in _OBJECT_SCHEMAS for key in schema["properties"]
)


def _squash(text: str) -> str:
    """Case- and separator-insensitive form: ``minWall``, ``Min-Wall``, ``MIN WALL`` -> ``minwall``.

    Only letters and digits survive, so this matches spellings that differ in nothing but case
    and separators - it can never turn one word into another.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _property_type(schema: Mapping[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """``(json type, vector length)`` of a property schema; the length is None unless fixed."""
    resolved, _ = _deref(schema, "")
    kind = resolved.get("type")
    if kind == "array":
        items = resolved.get("items")
        if isinstance(items, dict):
            item_type, _ = _property_type(items)
            if item_type == "number" and resolved.get("minItems") == resolved.get("maxItems"):
                return "array", int(resolved["minItems"])
        return "array", None
    if isinstance(kind, str):
        return kind, None
    if "enum" in resolved or "const" in resolved:
        return "string", None
    return None, None


# --------------------------------------------------------------------------------------
# the alias table
# --------------------------------------------------------------------------------------


class Alias(NamedTuple):
    """One accepted spelling.

    ``factor`` scales the value on the way through; it is 2 for ``radius -> d`` and 1 for every
    plain rename.  ``rule`` groups rows in the README table and in each normalisation record.
    """

    alias: str
    canonical: str
    rule: str
    note: str = ""
    factor: float = 1.0


#: Hand-written rows: spellings a model is likely to write, each mapping to exactly one schema
#: key *in the objects where that key exists*.  ``spacing`` maps to ``gap`` in a placement
#: constraint and to ``pitch`` in a vent or feature; the two never share an object, and an
#: import-time check enforces that no object can receive two different targets for one alias.
SPELLING_ALIASES: Tuple[Alias, ...] = (
    Alias("diameter", "d", "spelling", "cylinder source"),
    Alias("dia", "d", "spelling", "cylinder source"),
    Alias("diam", "d", "spelling", "cylinder source"),
    Alias("radius", "d", "scaled", "cylinder source; the value is doubled", 2.0),
    Alias("r", "d", "scaled", "cylinder source; the value is doubled", 2.0),
    Alias("height", "h", "spelling", "cylinder source"),
    Alias("length", "h", "spelling", "cylinder source: the extent along `axis`"),
    Alias("dimensions", "size", "spelling", "box source, access, features"),
    Alias("dims", "size", "spelling", "box source, access, features"),
    Alias("position", "at", "spelling", "access and features only; a constraint keeps `position`"),
    Alias("pos", "at", "spelling", "access and features"),
    Alias("center", "at", "spelling", "access and features: `at` is the feature centre"),
    Alias("centre", "at", "spelling", "access and features"),
    Alias("thickness", "wall", "spelling", "enclosure only; a `kicad_pcb` source keeps `thickness`"),
    Alias("wall_thickness", "wall", "spelling", "enclosure"),
    Alias("walls", "wall", "spelling", "enclosure"),
    Alias("distance", "gap", "spelling", "placement constraint"),
    Alias("spacing", "gap", "spelling", "placement constraint"),
    Alias("spacing", "pitch", "spelling", "vents and features"),
    Alias("anchor", "to", "spelling", "placement constraint"),
    Alias("relative_to", "to", "spelling", "placement constraint"),
    Alias("target", "to", "spelling", "placement constraint"),
    Alias("designator", "ref", "spelling", "component or constraint"),
    Alias("refdes", "ref", "spelling", "component or constraint"),
    Alias("type", "kind", "spelling", "source and enclosure features"),
    Alias("type", "relation", "spelling", "placement constraint"),
    Alias("file", "path", "spelling", "kicad_pcb and import sources"),
    Alias("filename", "path", "spelling", "kicad_pcb and import sources"),
    Alias("key", "part", "spelling", "library source"),
    Alias("exposed", "expose", "spelling", "kicad_pcb source"),
    Alias("rotation", "orientation", "spelling", "component"),
    Alias("params", "parameters", "spelling", "design root only; a library source keeps `params`"),
    Alias("constraints", "placement", "spelling", "design root"),
    Alias("placements", "placement", "spelling", "design root"),
    Alias("parts", "components", "spelling", "design root"),
    Alias("case", "enclosure", "spelling", "design root"),
    Alias("housing", "enclosure", "spelling", "design root"),
    Alias("revision", "rev", "spelling", "design root"),
    Alias("format", "formats", "spelling", "outputs; a bare string is wrapped in a list"),
    Alias("text", "label", "spelling", "enclosure text block; a feature keeps `text`"),
    Alias("side", "face", "spelling", "mount, access, features"),
    Alias("qty", "count", "spelling", "enclosure features"),
    Alias("quantity", "count", "spelling", "enclosure features"),
    Alias("hole_diameter", "hole_d", "spelling", "mount"),
    Alias("standoff_height", "standoff_h", "spelling", "mount"),
    Alias("slot_width", "slot_w", "spelling", "vents"),
    Alias("screw_size", "screw", "spelling", "lid"),
    Alias("bed", "printer_bed", "spelling", "manufacturing"),
    Alias("bed_size", "printer_bed", "spelling", "manufacturing"),
    Alias("nozzle_diameter", "nozzle", "spelling", "manufacturing"),
    Alias("max_overhang", "max_overhang_deg", "spelling", "manufacturing"),
    Alias("filament", "material", "spelling", "manufacturing"),
    Alias("surface_finish", "finish", "spelling", "manufacturing"),
    Alias("tol", "tolerance", "spelling", "access"),
    Alias("embossed", "emboss", "spelling", "enclosure text"),
    Alias("countersink", "countersunk", "spelling", "lid"),
    Alias("split", "split_parts", "spelling", "outputs"),
    Alias("exploded", "exploded_view", "spelling", "outputs"),
    Alias("cover", "lid", "spelling", "enclosure"),
    Alias("minimum", "min", "spelling", "parameter"),
    Alias("maximum", "max", "spelling", "parameter"),
    Alias("default", "value", "spelling", "parameter"),
    Alias("units", "unit", "spelling", "parameter; the design root keeps `units`"),
)

#: Numeric properties whose value is not a length, or whose unit depends on where the key sits, so
#: a ``_mm`` spelling of them is *not* an alias (``scale_mm`` would be a unit error, not a typo).
#: Parameter bounds are unit-dependent, and a tolerance band's ``plus_minus`` is millimetres in a
#: linear table and degrees in an angular one - accepting ``plus_minus_mm`` on an angular band
#: would quietly bless a wrong unit on a number a machinist holds.
_UNITLESS_KEYS: FrozenSet[str] = frozenset(
    {"scale", "thermal_w", "count", "screw_count", "max_overhang_deg", "value", "min", "max",
     "step", "plus_minus"}
)

#: Numeric properties measured in degrees.  Everything else numeric is millimetres.
ANGLE_KEYS: FrozenSet[str] = frozenset({"orientation", "max_overhang_deg"})


def _derived_aliases() -> Tuple[Alias, ...]:
    """Rows the schema implies: camelCase spellings and ``_mm``/``_deg`` suffixed spellings.

    Generated rather than typed so that a key added to the schema is covered the day it lands.
    """
    rows: List[Alias] = []
    numeric: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
    for _, schema in _OBJECT_SCHEMAS:
        for key, sub_schema in schema["properties"].items():
            if isinstance(sub_schema, dict):
                numeric.setdefault(key, _property_type(sub_schema))
    for key in sorted(CANONICAL_KEYS):
        if "_" in key:
            rows.append(Alias(_camel(key), key, "case", "camelCase spelling"))
        kind, _ = numeric.get(key, (None, None))
        if kind in ("number", "integer", "array") and key not in _UNITLESS_KEYS:
            if numeric[key][0] == "array" and numeric[key][1] is None:
                continue  # arrays of non-numbers (formats, expose, faces) carry no unit
            suffix = "_deg" if key in ANGLE_KEYS else "_mm"
            if not key.endswith(suffix):
                rows.append(Alias(key + suffix, key, "suffix", "unit spelled in the key"))
    return tuple(rows)


DERIVED_ALIASES: Tuple[Alias, ...] = _derived_aliases()

#: The complete table: hand rows first, then the schema-derived rows.
ALIASES: Tuple[Alias, ...] = SPELLING_ALIASES + DERIVED_ALIASES


def _check_table() -> None:
    """Import-time guard: the table must agree with the schema and never be ambiguous."""
    for row in ALIASES:
        if row.canonical not in CANONICAL_KEYS:
            raise RuntimeError(
                "alias %r -> %r: %r is not a key in design-ir.schema.json"
                % (row.alias, row.canonical, row.canonical)
            )
        if row.alias == row.canonical:
            raise RuntimeError("alias %r maps to itself" % row.alias)
    for pointer, schema in _OBJECT_SCHEMAS:
        declared = set(schema["properties"])
        targets: Dict[str, Set[str]] = {}
        for row in ALIASES:
            if row.canonical in declared and row.alias not in declared:
                targets.setdefault(row.alias, set()).add(row.canonical)
        for alias, canonicals in targets.items():
            if len(canonicals) > 1:
                raise RuntimeError(
                    "alias %r is ambiguous at %s: %s" % (alias, pointer, sorted(canonicals))
                )
        squashed = Counter(_squash(key) for key in declared)
        clash = [key for key, count in squashed.items() if count > 1]
        if clash:
            raise RuntimeError("keys at %s collide after squashing: %s" % (pointer, clash))
    missing_angles = ANGLE_KEYS - CANONICAL_KEYS
    if missing_angles:
        raise RuntimeError("ANGLE_KEYS not in the schema: %s" % sorted(missing_angles))


_check_table()


# --------------------------------------------------------------------------------------
# units: "25 mm", "2.5cm", "1in", "90deg", "90°"
# --------------------------------------------------------------------------------------

#: Accepted suffixes and their factor into the IR's unit.  Everything not listed here is refused:
#: a metre, a mil or a radian in a millimetre IR says the author is not thinking in this IR's
#: units, and that conversion has to be theirs, made visibly, not ours, made silently.
UNIT_FACTORS: Dict[str, Dict[str, str]] = {
    "mm": {
        "mm": "1", "millimeter": "1", "millimeters": "1", "millimetre": "1", "millimetres": "1",
        "cm": "10", "centimeter": "10", "centimeters": "10", "centimetre": "10",
        "centimetres": "10",
        "in": "25.4", "inch": "25.4", "inches": "25.4", '"': "25.4",
    },
    "deg": {"deg": "1", "degree": "1", "degrees": "1", "°": "1"},
    "g": {"g": "1", "gram": "1", "grams": "1"},
    "count": {},
}

#: A JSON-ish number.  Written ``\d+(?:\.\d*)?`` rather than ``\d+\.?\d*`` deliberately: the latter
#: puts two digit quantifiers side by side, so a string of digits that fails the match later (a bad
#: unit, say) can be split between them in many ways and the engine backtracks through all of them.
#: Requiring the '.' to enter the fractional part leaves exactly one way to parse any input.
_NUMBER = r"\d+(?:\.\d*)?|\.\d+"
_QUANTITY = re.compile(
    r"^\s*([-+]?(?:" + _NUMBER + r")(?:[eE][-+]?\d+)?)\s*([^\d\s.+\-]*)\s*$"
)
_DIMENSIONS = re.compile(
    r"^\s*([-+]?(?:" + _NUMBER + r"))\s*(?:[x×*]\s*([-+]?(?:" + _NUMBER + r"))\s*)+"
    r"([^\d\s.+\-]*)\s*$",
    re.IGNORECASE,
)


def _to_number(literal: str, factor: str) -> Any:
    """Exact decimal scaling, then the nearest float; an integral result is an ``int``.

    Decimal arithmetic means ``"1.5in"`` becomes the nearest double to 38.1 rather than to
    1.5 * 25.4 evaluated in binary, and ``"3cm"`` comes back as ``30``, not ``30.0``.
    """
    from decimal import Decimal

    value = Decimal(literal) * Decimal(factor)
    if value == value.to_integral_value() and abs(value) < 10 ** 15:
        return int(value)
    return float(value)


def parse_quantity(text: str, family: str = "mm") -> Tuple[Optional[Any], str, Optional[str]]:
    """Read a number with an optional unit suffix into the IR's unit for ``family``.

    Returns ``(value, unit, problem)``.  ``value`` is the converted number when the text is
    acceptable; otherwise ``problem`` is ``"unknown"`` (a suffix this IR does not accept) or
    ``"family"`` (a length where an angle is required, or the reverse).  A string that is not a
    quantity at all returns ``(None, "", None)`` and is left for the schema validator to name.
    """
    match = _QUANTITY.match(text)
    if match is None:
        return None, "", None
    literal, unit = match.group(1), match.group(2)
    table = UNIT_FACTORS.get(family, {})
    key = unit.lower()
    if key == "":
        return _to_number(literal, "1"), "", None
    if key in table:
        return _to_number(literal, table[key]), unit, None
    for other, other_table in UNIT_FACTORS.items():
        if other != family and key in other_table:
            return None, unit, "family"
    return None, unit, "unknown"


def _accepted_units(family: str) -> str:
    units = [unit for unit in UNIT_FACTORS.get(family, {}) if len(unit) <= 3]
    return ", ".join(units) if units else "a bare number"


# --------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------


class Refusal(NamedTuple):
    """An input the loader will not guess at, with the fragment that would fix it.

    ``suggestion`` is keyed by the property name so it can be pasted into the object that
    contains ``path``; unknown values are marked ``"<x?>"`` / ``"<mm?>"`` rather than invented.
    ``schema`` is the JSON pointer into ``design-ir.schema.json`` the input was judged against.
    """

    path: str
    rule: str
    message: str
    suggestion: Any
    schema: str

    def render(self) -> str:
        return "%s: %s — suggestion: %s" % (
            self.path or "design", self.message, json.dumps(self.suggestion, ensure_ascii=False)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path or "design",
            "rule": self.rule,
            "message": self.message,
            "suggestion": self.suggestion,
            "schema": self.schema,
        }


#: Every refusal rule, for the README and for tests that assert the catalogue is documented.
REFUSAL_RULES: Dict[str, str] = {
    "vector-scalar": "one number where `[x, y, z]` or `[x, y]` is required: a cube, or a typo?",
    "vector-length": "a vector with too few or too many values",
    "vector-object": "a vector written as an object with an axis name the IR does not know",
    "vector-string": "a `WxDxH` string with the wrong count of values or an unaccepted unit",
    "axis-incomplete": "`w`, `width`, `height`, ... as bare keys without every axis present",
    "axis-ambiguous": "`length`, `l`, `t` or `thickness` on a box: no axis is implied",
    "alias-conflict": "an alias and its canonical key both present with different values",
    "unit-unknown": "a suffix that is not mm, cm, in (lengths) or deg, ° (angles)",
    "unit-family": "a length where an angle is required, or the reverse",
    "units-not-mm": "`units` set to anything but millimetres",
    "enclosure-radius": "`radius` on an enclosure: `corner_radius` or `edge_radius`?",
    "constraint-xyz": "`x`, `y`, `z` on a placement constraint: `position` or `offset`?",
}

_PLACEHOLDER = ("<x?>", "<y?>", "<z?>")

#: Axis names accepted when a fixed-length vector is written as an object.
_VEC3_AXES: Dict[str, int] = {
    "x": 0, "w": 0, "width": 0, "size_x": 0, "sx": 0, "dx": 0, "rx": 0,
    "y": 1, "d": 1, "depth": 1, "size_y": 1, "sy": 1, "dy": 1, "ry": 1,
    "z": 2, "h": 2, "height": 2, "size_z": 2, "sz": 2, "dz": 2, "rz": 2,
}
_VEC2_AXES: Dict[str, int] = {
    "x": 0, "u": 0, "w": 0, "width": 0,
    "y": 1, "v": 1, "h": 1, "height": 1,
}
#: Names that *look* like an axis but do not say which one on a box.
_AMBIGUOUS_AXES: FrozenSet[str] = frozenset({"l", "length", "len", "t", "thickness"})

#: Objects that may be written as a bare string, keyed by the single property that string means.
#: An entry is only safe where no *other* property of that object could take a string, so the
#: rewrite is a shape fix rather than a guess; an import-time check enforces that.
_STRING_SHORTHAND: Tuple[str, ...] = ("label", "standard")


def _check_string_shorthand() -> None:
    """Guard: a bare string must have exactly one property it could be, in every object."""
    for pointer, schema in _OBJECT_SCHEMAS:
        properties: Dict[str, Any] = schema["properties"]
        present = [key for key in _STRING_SHORTHAND if key in properties]
        if not present:
            continue
        if len(present) > 1:
            raise RuntimeError("two string shorthands at %s: %s" % (pointer, present))
        others = [
            key for key, sub in properties.items()
            if key != present[0] and isinstance(sub, dict) and _property_type(sub)[0] == "string"
        ]
        if others:
            raise RuntimeError(
                "%r is not the only string property at %s: %s" % (present[0], pointer, others))


_check_string_shorthand()

#: Bare sibling keys that together spell a vector: ``{"kind": "box", "w": 40, "d": 20, "h": 8}``.
#: Applied only when the canonical vector is declared on the object and absent from it.
_SIBLING_VECTORS: Tuple[Tuple[str, int, Dict[str, int]], ...] = (
    ("size", 3, {
        "x": 0, "w": 0, "width": 0, "size_x": 0, "sx": 0,
        "y": 1, "d": 1, "depth": 1, "size_y": 1, "sy": 1,
        "z": 2, "h": 2, "height": 2, "size_z": 2, "sz": 2,
    }),
    ("size", 2, {"w": 0, "width": 0, "h": 1, "height": 1}),
    ("at", 2, {"x": 0, "u": 0, "y": 1, "v": 1}),
)


def _join(path: str, key: str) -> str:
    return key if not path else path + "." + key


def _index(path: str, index: int) -> str:
    return "%s[%d]" % (path, index) if path else "[%d]" % index


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_private(key: Any) -> bool:
    return isinstance(key, str) and key.startswith("_")


def _describe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# the walker
# --------------------------------------------------------------------------------------


class _Walker:
    """Walk a design against the schema, rewriting in place and recording every decision.

    ``path`` is the human-facing location (``components[0].source``); ``pattern`` is the same
    location with indices and parameter names elided (``components[].source``), which is what
    makes unknown keys countable across many designs.
    """

    def __init__(self) -> None:
        self.normalisations: List[Dict[str, Any]] = []
        self.refusals: List[Refusal] = []
        self.unknown: Counter = Counter()

    # -- bookkeeping -------------------------------------------------------------------------

    def record(self, path: str, source: Any, target: Any, rule: str, kind: str = "key",
               **extra: Any) -> None:
        """``kind`` is "key" when ``from``/``to`` are key names and "value" when they are values."""
        entry: Dict[str, Any] = {"path": path or "design", "from": source, "to": target,
                                 "rule": rule, "kind": kind}
        entry.update(extra)
        self.normalisations.append(entry)

    def refuse(self, path: str, rule: str, message: str, suggestion: Any, pointer: str) -> None:
        if rule not in REFUSAL_RULES:
            raise RuntimeError("undocumented refusal rule %r" % rule)
        self.refusals.append(Refusal(path, rule, message, suggestion, pointer))

    # -- dispatch ----------------------------------------------------------------------------

    def walk(self, node: Any, schema: Mapping[str, Any], path: str, pattern: str, pointer: str,
             key: Optional[str], family: str) -> Any:
        resolved, pointer = _deref(schema, pointer)
        if "oneOf" in resolved:
            return self._one_of(node, resolved, path, pattern, pointer)
        properties = resolved.get("properties")
        if isinstance(properties, dict):
            if isinstance(node, dict):
                return self._object(node, resolved, path, pattern, pointer)
            if isinstance(node, str) and not node.startswith("$"):
                for only in _STRING_SHORTHAND:
                    if only in properties:
                        # `"text": "Sample"`, `"general_tolerance": "ISO 2768-m"`: in each of these
                        # objects a bare string is the one property it could possibly be.
                        self.record(path, node, {only: node}, "shape", kind="value")
                        return {only: node}
            return node
        additional = resolved.get("additionalProperties")
        if isinstance(additional, dict) and isinstance(node, dict):
            for name in list(node):
                if _is_private(name):
                    continue
                node[name] = self.walk(node[name], additional, _join(path, name),
                                       _join(pattern, "*"), pointer + "/additionalProperties",
                                       name, family)
            return node
        if resolved.get("type") == "array":
            return self._array(node, resolved, path, pattern, pointer, key, family)
        return self._scalar(node, resolved, path, pointer, key, family)

    # -- objects -----------------------------------------------------------------------------

    def _one_of(self, node: Any, schema: Mapping[str, Any], path: str, pattern: str,
                pointer: str) -> Any:
        kinds: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for index, branch in enumerate(schema.get("oneOf", ())):
            if not isinstance(branch, dict):
                continue
            kind_schema = branch.get("properties", {}).get("kind", {})
            if isinstance(kind_schema, dict) and "const" in kind_schema:
                kinds[str(kind_schema["const"])] = (index, branch)
        if not isinstance(node, dict) or not kinds:
            return node
        if "kind" not in node:
            # The discriminator has to be found before a branch can be chosen, so its own aliases
            # (`type`, `Kind`) are handled here rather than by the per-object rename pass.
            for name in list(node):
                if isinstance(name, str) and not _is_private(name) \
                        and _squash(name) in ("kind", "type"):
                    self._rename(node, name, "kind")
                    self.record(path, name, "kind", "spelling")
                    break
        value = node.get("kind")
        if isinstance(value, str) and value not in kinds:
            for member in kinds:
                if _squash(member) == _squash(value):
                    node["kind"] = member
                    self.record(_join(path, "kind"), value, member, "case", kind="value")
                    break
        selected = kinds.get(node.get("kind")) if isinstance(node.get("kind"), str) else None
        if selected is None:
            return node  # the validator names the accepted kinds
        index, branch = selected
        return self._object(node, branch, path, pattern, "%s/oneOf/%d" % (pointer, index))

    @staticmethod
    def _rename(node: Dict[str, Any], old: str, new: str, value: Any = None,
                replace: bool = False) -> None:
        """Rename a key in place, keeping its position so the document order is preserved."""
        items = list(node.items())
        node.clear()
        for name, item in items:
            if name == old:
                node[new] = value if replace else item
            else:
                node[name] = item

    def _match_key(self, key: str, declared: Set[str],
                   properties: Mapping[str, Any]) -> Optional[Tuple[str, str, float]]:
        """Which canonical key, if any, a stray key is a high-confidence spelling of."""
        squashed = _squash(key)
        for row in ALIASES:
            if _squash(row.alias) == squashed and row.canonical in declared:
                return row.canonical, row.rule, row.factor
        for canonical in declared:
            if _squash(canonical) == squashed:
                return canonical, "case", 1.0
        for suffix, angle in (("_mm", False), ("_deg", True)):
            if squashed.endswith(_squash(suffix)):
                stem = squashed[: -len(_squash(suffix))]
                for canonical in declared:
                    if _squash(canonical) != stem or canonical in _UNITLESS_KEYS:
                        continue
                    kind, _ = _property_type(properties[canonical])
                    if kind not in ("number", "integer", "array"):
                        continue
                    if (canonical in ANGLE_KEYS) == angle:
                        return canonical, "suffix", 1.0
        return None

    def _scaled(self, value: Any, factor: float, family: str) -> Optional[Any]:
        if _is_number(value):
            return _to_number(repr(value), repr(factor))
        if isinstance(value, str):
            number, _, problem = parse_quantity(value, family)
            if number is not None and problem is None:
                return _to_number(repr(number), repr(factor))
        return None

    def _object(self, node: Dict[str, Any], schema: Mapping[str, Any], path: str, pattern: str,
                pointer: str) -> Dict[str, Any]:
        properties: Dict[str, Any] = schema["properties"]
        declared = set(properties)

        # Keys that were understood - renamed, folded into a vector, or refused - so the unknown-key
        # report only counts spellings the table genuinely does not know.
        explained: Set[str] = set()

        # 1. renames: aliases, case/separator spellings, unit-suffixed keys
        family = "mm"
        produced: Dict[str, Any] = {}
        for key in list(node):
            if not isinstance(key, str) or key in declared or _is_private(key):
                continue
            match = self._match_key(key, declared, properties)
            if match is None:
                continue
            explained.add(key)
            canonical, rule, factor = match
            value = node[key]
            if factor != 1.0:
                scaled = self._scaled(value, factor, family)
                if scaled is None:
                    continue  # not a number we can scale; left for the validator
                new_value = scaled
            else:
                new_value = value
            existing = node[canonical] if canonical in node else produced.get(canonical)
            if canonical in node or canonical in produced:
                if existing == new_value:
                    del node[key]
                    self.record(path, key, canonical, rule, dropped=True)
                    continue
                self.refuse(
                    _join(path, canonical), "alias-conflict",
                    "%r and %r are both given and disagree (%s vs %s); keep one" % (
                        key, canonical, _describe(new_value), _describe(existing)),
                    {canonical: "<%s or %s?>" % (_describe(existing), _describe(new_value))},
                    pointer + "/properties/" + canonical,
                )
                continue
            extra: Dict[str, Any] = {}
            if factor != 1.0:
                extra["value"] = {"from": value, "to": new_value}
            self._rename(node, key, canonical, new_value, replace=True)
            produced[canonical] = new_value
            self.record(path, key, canonical, rule, **extra)
        if pointer == "#/$defs/parameter":
            # A parameter's numbers are in its declared unit, so "25 mm" is only right for a
            # millimetre parameter; read the unit after renames so `units` -> `unit` counts.
            unit = node.get("unit", "mm")
            family = unit if unit in UNIT_FACTORS else "mm"

        # 2. sibling keys that spell a vector, e.g. w/d/h on a box source
        for canonical, dims, axes in _SIBLING_VECTORS:
            if canonical not in declared or canonical in node:
                continue
            kind, length = _property_type(properties[canonical])
            if kind != "array" or length != dims:
                continue
            present = [
                key for key in node
                if isinstance(key, str) and not _is_private(key) and key not in declared
                and (key.lower() in axes or (dims == 3 and key.lower() in _AMBIGUOUS_AXES))
            ]
            if not present:
                continue
            explained.update(present)
            slots: List[Optional[str]] = [None] * dims
            ambiguous: List[str] = []
            duplicated = False
            for key in present:
                slot = axes.get(key.lower())
                if slot is None:
                    ambiguous.append(key)
                elif slots[slot] is not None:
                    duplicated = True
                else:
                    slots[slot] = key
            suggestion = {
                canonical: [node[name] if name else _PLACEHOLDER[i] for i, name in enumerate(slots)]
            }
            where = pointer + "/properties/" + canonical
            if ambiguous:
                self.refuse(
                    path, "axis-ambiguous",
                    "%s does not say which axis of %s it is (%s); the IR wants [x, y, z]" % (
                        ", ".join(repr(k) for k in ambiguous), canonical,
                        ", ".join("%s=%s" % (k, _describe(node[k])) for k in present)),
                    suggestion, where,
                )
                continue
            if duplicated or not all(slots):
                self.refuse(
                    path, "axis-incomplete",
                    "%s alone does not give every axis of %s; fill the marked values" % (
                        ", ".join(repr(k) for k in present), canonical),
                    suggestion, where,
                )
                continue
            names = [name for name in slots if name]
            values = [node[name] for name in names]
            items = list(node.items())
            node.clear()
            placed = False
            for name, item in items:
                if name in names:
                    if not placed:
                        node[canonical] = values
                        placed = True
                    continue
                node[name] = item
            self.record(path, ",".join(names), canonical, "vector", value=values)

        # 3. context-specific ambiguities the schema alone cannot express
        if {"corner_radius", "edge_radius"} <= declared:
            for key in list(node):
                if isinstance(key, str) and _squash(key) in ("radius", "r"):
                    explained.add(key)
                    shown = _describe(node[key])
                    self.refuse(
                        _join(path, key), "enclosure-radius",
                        "%r on an enclosure could be the outside corner radius or the edge "
                        "fillet; name one" % key,
                        {"corner_radius": "<%s if the outside corners?>" % shown,
                         "edge_radius": "<%s if the edge fillet?>" % shown},
                        pointer,
                    )
        if {"position", "offset"} <= declared and "at" not in declared:
            axes_present = [k for k in node if isinstance(k, str) and k.lower() in ("x", "y", "z")]
            if axes_present:
                explained.update(axes_present)
                vector = list(_PLACEHOLDER)
                for key in axes_present:
                    vector["xyz".index(key.lower())] = node[key]
                self.refuse(
                    path, "constraint-xyz",
                    "%s on a placement constraint could be an absolute `position` (with "
                    "relation `at`) or an `offset` from the anchor; name one" % (
                        ", ".join(repr(k) for k in axes_present)),
                    {"position": vector, "offset": vector},
                    pointer,
                )

        # 4. whatever is left and undeclared is unknown: counted, never guessed
        for key in node:
            if isinstance(key, str) and key not in declared and not _is_private(key) \
                    and key not in explained:
                self.unknown[_join(pattern, key)] += 1

        # 5. recurse into declared properties
        for key in list(node):
            if key in declared and isinstance(properties[key], dict):
                child_family = family
                if key in ANGLE_KEYS:
                    child_family = "deg"
                elif pointer != "#/$defs/parameter":
                    child_family = "mm"
                node[key] = self.walk(node[key], properties[key], _join(path, key),
                                      _join(pattern, key), pointer + "/properties/" + key,
                                      key, child_family)
        return node

    # -- arrays ------------------------------------------------------------------------------

    def _array(self, node: Any, schema: Mapping[str, Any], path: str, pattern: str,
               pointer: str, key: Optional[str], family: str) -> Any:
        items = schema.get("items")
        item_schema: Optional[Dict[str, Any]] = None
        item_pointer = pointer + "/items"
        if isinstance(items, dict):
            item_schema, item_pointer = _deref(items, item_pointer)
        _, dims = _property_type(schema)
        name = key or "value"
        if dims is not None:
            return self._vector(node, dims, path, pointer, name, family)
        if item_schema is None:
            return node
        item_kind, _ = _property_type(item_schema)
        if isinstance(node, str) and item_kind == "string" and not node.startswith("$"):
            parts = [part.strip() for part in node.split(",")] if "," in node else [node]
            self.record(path, node, parts, "wrap", kind="value")
            node = parts
        if isinstance(node, list):
            for index, item in enumerate(node):
                node[index] = self.walk(item, item_schema, _index(path, index),
                                        pattern + "[]", item_pointer, key, family)
        return node

    def _vector(self, node: Any, dims: int, path: str, pointer: str, name: str,
                family: str) -> Any:
        placeholders = list(_PLACEHOLDER[:dims])
        if isinstance(node, list):
            if len(node) != dims:
                shown = [node[i] if i < len(node) else placeholders[i] for i in range(dims)]
                extra = ", ".join(_describe(v) for v in node[dims:])
                self.refuse(
                    path, "vector-length",
                    "%d value(s) given where %s needs exactly %d%s" % (
                        len(node), name, dims,
                        ("; extra: " + extra) if extra else ""),
                    {name: shown}, pointer,
                )
            return self._vector_items(node, path, pointer, name, family)
        if isinstance(node, dict):
            axes = _VEC3_AXES if dims == 3 else _VEC2_AXES
            slots: List[Any] = list(placeholders)
            unknown: List[str] = []
            for key, value in node.items():
                slot = axes.get(str(key).lower())
                if slot is None or slot >= dims:
                    unknown.append(str(key))
                else:
                    slots[slot] = value
            if unknown:
                self.refuse(
                    path, "vector-object",
                    "%s names no axis of %s; accepted: %s" % (
                        ", ".join(repr(k) for k in unknown), name,
                        ", ".join(sorted(k for k, i in axes.items() if i < dims))),
                    {name: slots}, pointer,
                )
                return node
            if any(isinstance(v, str) and v in placeholders for v in slots):
                self.refuse(
                    path, "axis-incomplete",
                    "the object for %s does not give every axis; fill the marked values" % name,
                    {name: slots}, pointer,
                )
                return node
            self.record(path, dict(node), slots, "vector", kind="value")
            return self._vector_items(slots, path, pointer, name, family)
        if isinstance(node, str):
            if node.startswith("$"):
                return node
            match = _DIMENSIONS.match(node)
            if match is None:
                return node  # not a dimension string; the validator names the type
            numbers = re.findall(r"[-+]?\d+\.?\d*", node[: match.start(3)] if match.group(3) else node)
            unit = match.group(3).lower()
            table = UNIT_FACTORS.get(family, {})
            if unit and unit not in table:
                self.refuse(
                    path, "vector-string",
                    "%r ends in %r, which is not an accepted unit (%s)" % (
                        node, match.group(3), _accepted_units(family)),
                    {name: ["<%s?>" % family] * dims}, pointer,
                )
                return node
            if len(numbers) != dims:
                self.refuse(
                    path, "vector-string",
                    "%r has %d values where %s needs %d" % (node, len(numbers), name, dims),
                    {name: [_to_number(numbers[i], "1") if i < len(numbers) else placeholders[i]
                            for i in range(dims)]},
                    pointer,
                )
                return node
            values = [_to_number(n, table.get(unit, "1")) for n in numbers]
            self.record(path, node, values, "vector", kind="value")
            return values
        if _is_number(node):
            suggestion = [node] + placeholders[1:]
            self.refuse(
                path, "vector-scalar",
                "%s is one number where %s needs %d values%s" % (
                    _describe(node), name, dims,
                    "; if a cube was meant, repeat it" if dims == 3 else ""),
                {name: suggestion}, pointer,
            )
        return node

    def _vector_items(self, values: List[Any], path: str, pointer: str, name: str,
                      family: str) -> List[Any]:
        for index, item in enumerate(values):
            if not isinstance(item, str) or item.startswith("$"):
                continue
            number, unit, problem = parse_quantity(item, family)
            if problem is not None:
                shown = [v if i != index else "<%s?>" % family for i, v in enumerate(values)]
                self.refuse(
                    _index(path, index), "unit-unknown" if problem == "unknown" else "unit-family",
                    self._unit_message(item, unit, problem, name, family),
                    {name: shown}, pointer,
                )
                continue
            if number is None:
                continue
            self.record(_index(path, index), item, number, "unit" if unit else "string-number",
                        kind="value")
            values[index] = number
        return values

    # -- scalars -----------------------------------------------------------------------------

    @staticmethod
    def _unit_message(text: str, unit: str, problem: str, name: str, family: str) -> str:
        if problem == "family":
            wanted = "an angle in degrees" if family == "deg" else "a length in millimetres"
            return "%s is %s, but %r is written in %r" % (name, wanted, text, unit)
        return "%r uses %r, which is not a unit this IR accepts for %s (%s)" % (
            text, unit, name, _accepted_units(family))

    def _scalar(self, node: Any, schema: Mapping[str, Any], path: str, pointer: str,
                key: Optional[str], family: str) -> Any:
        kind = schema.get("type")
        name = key or "value"
        if isinstance(node, str) and node.startswith("$"):
            return node  # a parameter reference; resolved later, never a unit string
        if kind in ("number", "integer") and isinstance(node, str):
            number, unit, problem = parse_quantity(node, family)
            if problem is not None:
                self.refuse(
                    path, "unit-unknown" if problem == "unknown" else "unit-family",
                    self._unit_message(node, unit, problem, name, family),
                    {name: "<%s?>" % family}, pointer,
                )
                return node
            if number is None:
                return node
            if kind == "integer" and not isinstance(number, int):
                return node
            self.record(path, node, number, "unit" if unit else "string-number", kind="value")
            return number
        if kind == "integer" and isinstance(node, float) and node.is_integer():
            self.record(path, node, int(node), "integer", kind="value")
            return int(node)
        if kind == "boolean" and isinstance(node, str) and node.strip().lower() in ("true", "false"):
            value = node.strip().lower() == "true"
            self.record(path, node, value, "boolean", kind="value")
            return value
        if isinstance(node, str) and ("enum" in schema or "const" in schema):
            allowed = schema["enum"] if "enum" in schema else [schema["const"]]
            if node in allowed:
                return node
            if key == "units":
                if _squash(node) in ("mm", "millimeter", "millimeters", "millimetre",
                                     "millimetres"):
                    self.record(path, node, "mm", "spelling", kind="value")
                    return "mm"
                self.refuse(
                    path, "units-not-mm",
                    "PhiMech is millimetres throughout; %r is not converted for you - convert "
                    "the numbers and declare \"mm\"" % node,
                    {"units": "mm"}, pointer,
                )
                return node
            for member in allowed:
                if isinstance(member, str) and _squash(member) == _squash(node):
                    self.record(path, node, member, "case", kind="value")
                    return member
        return node


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------


class Normalised(NamedTuple):
    """What ``normalise()`` did: the document, every rewrite, every refusal, every unknown key."""

    design: Dict[str, Any]
    normalisations: List[Dict[str, Any]]
    refusals: List[Refusal]
    unknown_keys: Dict[str, int]


#: Unknown keys seen by every ``normalise()`` call in this process, with their frequency.  The
#: bench logs this after a run: the keys models actually write are how the alias table grows.
UNKNOWN_KEYS: Counter = Counter()


def normalise(design: Dict[str, Any]) -> Normalised:
    """Apply every high-confidence alias to ``design`` in place and report what happened.

    Runs before defaults, parameter substitution and schema validation, so everything downstream
    still sees only canonical keys and plain numbers.  Refusals are returned, not raised: they
    are diagnostics the caller reports beside the schema's, each with its suggestion.
    """
    if not isinstance(design, dict):
        return Normalised(design, [], [], {})
    walker = _Walker()
    walker.walk(design, SCHEMA, "", "", "#", None, "mm")
    UNKNOWN_KEYS.update(walker.unknown)
    return Normalised(design, walker.normalisations, walker.refusals, dict(walker.unknown))


def analyse(design: Mapping[str, Any]) -> Normalised:
    """``normalise()`` on a deep copy: reports what *would* change without changing anything.

    ``validate()`` uses this to attach suggestions to a design it was handed directly, and the
    global unknown-key counter is deliberately left alone so analysis never inflates it.
    """
    if not isinstance(design, dict):
        return Normalised(dict(design) if isinstance(design, Mapping) else {}, [], [], {})
    duplicate = copy.deepcopy(design)
    walker = _Walker()
    walker.walk(duplicate, SCHEMA, "", "", "#", None, "mm")
    return Normalised(duplicate, walker.normalisations, walker.refusals, dict(walker.unknown))


def unknown_key_report(docs: Optional[Iterable[Mapping[str, Any]]] = None) -> Dict[str, int]:
    """``{key: count}`` of keys the schema does not declare and no alias explains.

    Keys are pattern-qualified (``components[].source.diameter``) because the same word means
    different things in different objects.  With ``docs`` the report covers exactly those
    documents; without, it is everything ``normalise()`` has seen in this process.
    """
    if docs is None:
        return dict(UNKNOWN_KEYS)
    total: Counter = Counter()
    for doc in docs:
        total.update(analyse(doc).unknown_keys)
    return dict(total)


# --------------------------------------------------------------------------------------
# README rendering
# --------------------------------------------------------------------------------------

README_BEGIN = "<!-- aliases:begin (generated by `python3 -m phimech.ir.aliases readme`) -->"
README_END = "<!-- aliases:end -->"

_RULE_TITLES = (
    ("spelling", "Spelling aliases"),
    ("scaled", "Aliases that scale the value"),
    ("case", "camelCase spellings (every schema key with an underscore; derived)"),
    ("suffix", "Unit spelled in the key (every length or angle property; derived)"),
)


def render_table() -> str:
    """The alias table and the refusal catalogue as Markdown, for the IR README.

    Rendered from the live table so the README cannot list an alias that no longer exists; a
    test asserts the README block equals this text.
    """
    lines: List[str] = []
    for rule, title in _RULE_TITLES:
        rows = [row for row in ALIASES if row.rule == rule]
        if not rows:
            continue
        lines.append("**%s** (%d)" % (title, len(rows)))
        lines.append("")
        if rule in ("case", "suffix"):
            lines.append("| Write | Read as |")
            lines.append("| --- | --- |")
            for row in rows:
                lines.append("| `%s` | `%s` |" % (row.alias, row.canonical))
        else:
            lines.append("| Write | Read as | Where |")
            lines.append("| --- | --- | --- |")
            for row in rows:
                lines.append("| `%s` | `%s` | %s |" % (row.alias, row.canonical, row.note))
        lines.append("")
    lines.append("**Refused, with a paste-ready fragment** (%d rules)" % len(REFUSAL_RULES))
    lines.append("")
    lines.append("| Rule | What it catches |")
    lines.append("| --- | --- |")
    for rule, text in REFUSAL_RULES.items():
        lines.append("| `%s` | %s |" % (rule, text))
    return "\n".join(lines) + "\n"


def readme_block() -> str:
    return README_BEGIN + "\n" + render_table() + README_END


def update_readme(path: Optional[Path] = None) -> bool:
    """Rewrite the generated block in the IR README; returns True when the file changed."""
    readme = path or Path(__file__).with_name("README.md")
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_BEGIN), text.find(README_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("no alias block markers in %s" % readme)
    updated = text[:start] + readme_block() + text[end + len(README_END):]
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8")
    return True


# --------------------------------------------------------------------------------------
# CLI hook: `phimech ir normalise <design.json> [--write]`
# --------------------------------------------------------------------------------------


def _read_design(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("YAML input requires PyYAML; save the design as JSON") from exc
            design = yaml.safe_load(handle)
        else:
            design = json.load(handle)
    if not isinstance(design, dict):
        raise ValueError("design file must contain a JSON/YAML object")
    return design


def _write_design(path: Path, design: Dict[str, Any]) -> None:
    """Atomic rewrite, author's key order kept: a normalised file should read like the original."""
    text = json.dumps(design, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _handle_normalise(args: argparse.Namespace) -> int:
    """Print every normalisation and refusal; exit 1 while any refusal remains.

    ``--write`` rewrites the file with the normalisations applied even when refusals remain: the
    refused parts are untouched, and the exit code still says the design is not done.
    """
    path = Path(args.file).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        design = _read_design(path)
    except (OSError, ValueError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    outcome = normalise(design)
    written = False
    if getattr(args, "write", False) and outcome.normalisations:
        if path.suffix.lower() in (".yaml", ".yml"):
            print("error: --write only rewrites JSON designs", file=sys.stderr)
            return 2
        _write_design(path, outcome.design)
        written = True
    report = {
        "file": str(path),
        "normalisations": outcome.normalisations,
        "refusals": [refusal.to_dict() for refusal in outcome.refusals],
        "unknown_keys": outcome.unknown_keys,
        "written": written,
        "ok": not outcome.refusals,
    }
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if report["ok"] else 1
    for entry in outcome.normalisations:
        detail = ""
        if "value" in entry:
            detail = "  (%s)" % _describe(entry["value"])
        print("%-10s %s: %s -> %s%s" % (
            entry["rule"], entry["path"], _describe(entry["from"]), _describe(entry["to"]), detail))
    for refusal in outcome.refusals:
        print("REFUSED    %s" % refusal.render())
        print("           schema: %s" % refusal.schema)
    for key, count in sorted(outcome.unknown_keys.items()):
        print("unknown    %s (x%d)" % (key, count))
    print("%d normalisation(s), %d refusal(s), %d unknown key(s)%s" % (
        len(outcome.normalisations), len(outcome.refusals), len(outcome.unknown_keys),
        "; wrote %s" % path if written else ""))
    return 0 if report["ok"] else 1


def register_cli(subparsers: Any) -> None:
    """Add ``normalise`` (and the US spelling) to the ``ir`` command group.

    The coordinator calls this from ``phimech/cli.py`` with the ``ir`` subparsers object; the
    handler follows the CLI's ``handler(args) -> int`` convention.
    """
    parser = subparsers.add_parser(
        "normalise", aliases=["normalize"],
        help="apply the tolerant aliases to a design and print every change and refusal",
    )
    parser.add_argument("file")
    parser.add_argument(
        "--write", action="store_true",
        help="rewrite the JSON file with the normalisations applied (refusals stay as they are)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.set_defaults(handler=_handle_normalise)


def doctor_section() -> Dict[str, Any]:
    """Health of the alias layer: the table agrees with the schema and the examples need nothing.

    ``ok`` is False when any bundled example would be rewritten, because the examples are the
    canonical spelling and a normalisation there means either the example or the table is wrong.
    """
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    touched: Dict[str, int] = {}
    for example in sorted(examples_dir.glob("*.json")):
        try:
            outcome = analyse(json.loads(example.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            touched[example.name] = -1
            continue
        if outcome.normalisations or outcome.refusals:
            touched[example.name] = len(outcome.normalisations) + len(outcome.refusals)
    return {
        "ok": not touched,
        "aliases": len(ALIASES),
        "refusal_rules": len(REFUSAL_RULES),
        "canonical_keys": len(CANONICAL_KEYS),
        "schema": str(_SCHEMA_PATH),
        "examples_needing_normalisation": touched,
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m phimech.ir.aliases", description="PhiMech design IR aliases")
    commands = parser.add_subparsers(dest="command")
    table = commands.add_parser("table", help="print the alias table as Markdown")
    table.set_defaults(handler=lambda args: (sys.stdout.write(render_table()), 0)[1])
    readme = commands.add_parser("readme", help="rewrite the generated block in ir/README.md")
    readme.set_defaults(handler=lambda args: (print(
        "updated" if update_readme() else "already current"), 0)[1])
    register_cli(commands)
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help(sys.stderr)
        return 2
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ALIASES",
    "ANGLE_KEYS",
    "Alias",
    "CANONICAL_KEYS",
    "DERIVED_ALIASES",
    "Normalised",
    "REFUSAL_RULES",
    "Refusal",
    "SPELLING_ALIASES",
    "UNIT_FACTORS",
    "UNKNOWN_KEYS",
    "analyse",
    "doctor_section",
    "normalise",
    "parse_quantity",
    "register_cli",
    "render_table",
    "unknown_key_report",
    "update_readme",
]
