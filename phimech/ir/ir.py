"""Load, default, and validate the PhiMech design intermediate representation.

The IR is deliberately ordinary JSON.  Keeping loading and validation here (rather than
making callers depend on jsonschema) means the fast synth check works with the system
Python that ships on macOS as well as in the optional MCP environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from phimech.ir import aliases as _aliases


_SCHEMA_PATH = Path(__file__).with_name("design-ir.schema.json")
with _SCHEMA_PATH.open("r", encoding="utf-8") as _schema_file:
    SCHEMA: Dict[str, Any] = json.load(_schema_file)



def _schema_for_ref(ref: str) -> Dict[str, Any]:
    """Resolve the local references used by the design schema."""
    if not ref.startswith("#/"):
        raise ValueError("only local schema references are supported: %s" % ref)
    value: Any = SCHEMA
    for part in ref[2:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    if not isinstance(value, dict):
        raise ValueError("schema reference does not point to an object: %s" % ref)
    return value


def _resolved_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Follow a schema reference, which is enough for this small local schema."""
    current: Dict[str, Any] = dict(schema)
    seen: Set[str] = set()
    while "$ref" in current:
        ref = current["$ref"]
        if ref in seen:
            raise ValueError("cyclic schema reference: %s" % ref)
        seen.add(ref)
        current = _schema_for_ref(ref)
    return current


def _source_branch(schema: Mapping[str, Any], value: Any) -> Optional[Dict[str, Any]]:
    """Choose a discriminator branch without trying every possible source schema."""
    for branch in schema.get("oneOf", ()):
        if not isinstance(branch, dict):
            continue
        properties = branch.get("properties", {})
        kind_schema = properties.get("kind", {}) if isinstance(properties, dict) else {}
        if isinstance(kind_schema, dict) and "const" in kind_schema:
            if isinstance(value, dict) and value.get("kind") == kind_schema["const"]:
                return branch
    return None


def _fill_defaults(value: Any, schema: Mapping[str, Any]) -> None:
    """Recursively apply JSON-schema defaults to objects that are present.

    JSON Schema describes defaults but does not apply them.  This small walker applies the
    same convention locally and follows the schema's discriminator so a source gets only
    the defaults belonging to its selected provider.
    """
    resolved = _resolved_schema(schema)
    one_of = resolved.get("oneOf")
    if one_of:
        branch = _source_branch(resolved, value)
        if branch is not None:
            _fill_defaults(value, branch)
        return

    if isinstance(value, dict):
        properties = resolved.get("properties", {})
        if not isinstance(properties, dict):
            return
        for name, property_schema in properties.items():
            if (
                name not in value
                and isinstance(property_schema, dict)
                and "default" in property_schema
            ):
                value[name] = copy.deepcopy(property_schema["default"])
            if name in value and isinstance(property_schema, dict):
                _fill_defaults(value[name], property_schema)
    elif isinstance(value, list) and isinstance(resolved.get("items"), dict):
        for item in value:
            _fill_defaults(item, resolved["items"])


def apply_defaults(design: Dict[str, Any]) -> Dict[str, Any]:
    """Fill declared defaults and the two ergonomic IR defaults.

    ``outputs`` and ``manufacturing`` are optional objects in the schema, but creating them
    when omitted makes their nested defaults useful to the pipeline.  An omitted enclosure
    is intentionally left omitted: that is the meaningful bare-assembly form of the IR.
    """
    if not isinstance(design, dict):
        return design

    # These objects have no parent-level JSON-schema default, so seed them before walking
    # their properties.  This gives every design stable output and manufacturing policy.
    if "outputs" not in design:
        design["outputs"] = {}
    if "manufacturing" not in design:
        design["manufacturing"] = {}

    _fill_defaults(design, SCHEMA)

    if "title" not in design and "project" in design:
        design["title"] = design["project"]
    enclosure = design.get("enclosure")
    if isinstance(enclosure, dict) and "floor" not in enclosure:
        if "wall" in enclosure:
            enclosure["floor"] = copy.deepcopy(enclosure["wall"])
    return design


def _absolute_design_path(path: Union[str, Path]) -> Path:
    """Return a stable absolute path without requiring the input to exist."""
    return Path(path).expanduser().resolve()


def _resolve_source_paths(design: Dict[str, Any], base_dir: Path) -> None:
    """Make external component paths independent of the process working directory."""
    components = design.get("components")
    if not isinstance(components, list):
        return
    for component in components:
        if not isinstance(component, dict):
            continue
        source = component.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            continue
        source_path = Path(source["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        source["path"] = str(source_path.resolve())


#: Marker for a parameter reference. A bare ``"$name"`` string, nothing else - no arithmetic, no
#: interpolation. Expressions would make the IR a language, and the numbers in a report have to be
#: traceable to something a human can read off the design.
PARAM_PREFIX = "$"


def parameter_errors(design: Mapping[str, Any]) -> List[str]:
    """Diagnose the ``parameters`` block itself: bad bounds, or a value outside them.

    Bounds are checked here rather than at write time alone, because a design can also be authored
    or hand-edited straight into a file, and an out-of-bounds value must not survive that route.
    """
    errors: List[str] = []
    parameters = design.get("parameters")
    if parameters is None:
        return errors
    if not isinstance(parameters, dict):
        return ["parameters: expected type object, got %s" % _type_name(parameters)]
    for name, spec in sorted(parameters.items()):
        where = "parameters.%s" % name
        if not isinstance(spec, dict):
            errors.append("%s: expected type object, got %s" % (where, _type_name(spec)))
            continue
        value = spec.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append("%s.value: expected a number, got %s" % (where, _type_name(value)))
            continue
        low, high = spec.get("min"), spec.get("max")
        numeric = lambda item: isinstance(item, (int, float)) and not isinstance(item, bool)
        if numeric(low) and numeric(high) and low > high:
            errors.append("%s: min %s is greater than max %s" % (where, low, high))
        if numeric(low) and value < low:
            errors.append("%s.value %s is below min %s" % (where, value, low))
        if numeric(high) and value > high:
            errors.append("%s.value %s is above max %s" % (where, value, high))
    return errors


def resolve_parameters(design: Dict[str, Any]) -> List[str]:
    """Substitute every ``"$name"`` reference in place with its parameter value.

    Runs *before* validation, so the schema and every downstream stage keep seeing plain numbers and
    need no knowledge of parameters at all. Returns diagnostics rather than raising, because an
    unknown reference is a design error the caller reports alongside the others.

    The ``parameters`` block itself is never walked - a parameter cannot reference a parameter.
    That is a deliberate limit: chained references need cycle detection and an evaluation order,
    and neither earns its keep while one level covers sliders and hand edits.
    """
    errors: List[str] = []
    parameters = design.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    values: Dict[str, Any] = {}
    for name, spec in parameters.items():
        if isinstance(spec, dict) and isinstance(spec.get("value"), (int, float)):
            if not isinstance(spec["value"], bool):
                values[name] = spec["value"]

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            return {key: walk(item, _path_key(path, key)) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item, _path_index(path, index)) for index, item in enumerate(node)]
        if isinstance(node, str) and node.startswith(PARAM_PREFIX) and len(node) > 1:
            name = node[1:]
            if name in values:
                return values[name]
            errors.append(
                "%s: unknown parameter reference %r (declared: %s)"
                % (path or "design", node, ", ".join(sorted(values)) or "none")
            )
            return node
        return node

    for key, item in list(design.items()):
        if key == "parameters" or (isinstance(key, str) and key.startswith("_")):
            continue
        design[key] = walk(item, key)
    return errors


def set_parameter(
    path: Union[str, Path], name: str, value: float, *, allow_new: bool = False
) -> Dict[str, Any]:
    """Write one parameter value back to a design file, refusing an out-of-bounds edit.

    This is the supported manual-edit path, and the reason it edits the *file* rather than a loaded
    design: ``load()`` returns a document whose references have already been substituted, so the
    parameter is gone by then. Editing the source keeps one number in one place, however many fields
    reference it.

    Atomic (temp file plus rename), so a refused or failed edit cannot leave a half-written design.
    """
    design_path = _absolute_design_path(path)
    raw = json.loads(design_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("design file must contain a JSON object")
    parameters = raw.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    if name not in parameters:
        if not allow_new:
            raise KeyError(
                "no parameter %r in %s (declared: %s)"
                % (name, design_path.name, ", ".join(sorted(parameters)) or "none")
            )
        # Created empty, not pre-filled: the value is written below like any other edit, so
        # `previous` correctly reports None for a parameter that did not exist yet.
        parameters[name] = {}
    spec = parameters[name]
    if not isinstance(spec, dict):
        raise ValueError("parameters.%s must be an object" % name)
    numeric = lambda item: isinstance(item, (int, float)) and not isinstance(item, bool)
    low, high = spec.get("min"), spec.get("max")
    if numeric(low) and value < low:
        raise ValueError(
            "%s=%s is below its declared min %s; widen the bound in the design if that is intended"
            % (name, value, low)
        )
    if numeric(high) and value > high:
        raise ValueError(
            "%s=%s is above its declared max %s; widen the bound in the design if that is intended"
            % (name, value, high)
        )
    previous = spec.get("value")
    spec["value"] = value
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = design_path.with_name(design_path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(design_path))
    return {"name": name, "previous": previous, "value": value, "path": str(design_path)}


def load(path: Union[str, Path], normalise: bool = True) -> Dict[str, Any]:
    """Load a JSON design (or optional PyYAML design) and attach its source metadata.

    With ``normalise`` (the default) the tolerant aliases in :mod:`phimech.ir.aliases` are applied
    first - ``diameter`` becomes ``d``, ``"25 mm"`` becomes ``25`` - and every rewrite is recorded
    under ``design["_normalisations"]`` as ``{path, from, to, rule}``.  The list is empty for a
    design that was already canonical, and it is a private key, so ``dumps()`` and validation
    never see it.  Ambiguous input is *not* rewritten; ``validate()`` reports it with a suggestion.
    """
    input_path = _absolute_design_path(path)
    suffix = input_path.suffix.lower()
    try:
        with input_path.open("r", encoding="utf-8") as design_file:
            if suffix in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise RuntimeError(
                        "YAML input requires PyYAML; install the optional YAML dependency "
                        "or save the design as JSON"
                    ) from exc
                design = yaml.safe_load(design_file)
            else:
                design = json.load(design_file)
    except OSError:
        raise

    if not isinstance(design, dict):
        raise ValueError("design file must contain a JSON/YAML object")
    # Aliases go first: defaults, parameter substitution and path resolution all key on canonical
    # names, so `thickness` has to be `wall` before `floor` can default to it.
    design["_normalisations"] = _aliases.normalise(design).normalisations if normalise else []
    apply_defaults(design)
    # Substitute parameter references before anything else inspects the document, so defaults,
    # path resolution, the schema and every synth stage all see plain numbers.
    parameter_diagnostics = resolve_parameters(design)
    _resolve_source_paths(design, input_path.parent)
    design["_meta"] = {"path": str(input_path), "base_dir": str(input_path.parent)}
    if parameter_diagnostics:
        # Carried rather than raised: validate() reports it with every other diagnostic, which is
        # what an agent iterating on a design needs to see.
        design["_parameter_errors"] = parameter_diagnostics
    return design


def _strip_private(value: Any) -> Any:
    """Remove loader metadata before schema validation or canonical serialization."""
    if isinstance(value, dict):
        return {
            key: _strip_private(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_schema_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _path_key(path: str, key: str) -> str:
    return key if not path else path + "." + key


def _path_index(path: str, index: int) -> str:
    return "%s[%d]" % (path, index) if path else "[%d]" % index


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str, errors: List[str]) -> None:
    """Validate the JSON-schema features used by design-ir.schema.json."""
    resolved = _resolved_schema(schema)

    one_of = resolved.get("oneOf")
    if one_of:
        branch = _source_branch(resolved, value)
        if branch is not None:
            _validate_schema(value, branch, path, errors)
            return

        # This fallback keeps the helper useful if another oneOf is added without a kind
        # discriminator.  A source with an unknown kind gets one actionable message rather
        # than the internals of all six failed alternatives.
        branch_errors: List[List[str]] = []
        for candidate in one_of:
            candidate_errors: List[str] = []
            if isinstance(candidate, dict):
                _validate_schema(value, candidate, path, candidate_errors)
            else:
                candidate_errors.append(path + ": invalid schema branch")
            branch_errors.append(candidate_errors)
        valid_count = sum(1 for candidate_errors in branch_errors if not candidate_errors)
        if valid_count != 1:
            expected = []
            for candidate in one_of:
                if isinstance(candidate, dict):
                    kind = candidate.get("properties", {}).get("kind", {}).get("const")
                    if kind is not None:
                        expected.append(str(kind))
            if expected:
                errors.append(
                    "%s: source kind must be one of %s"
                    % (path or "value", ", ".join(expected))
                )
            else:
                errors.append("%s: must match exactly one schema alternative" % (path or "value"))
        return

    expected_type = resolved.get("type")
    if expected_type is not None:
        type_matches = (
            any(_is_schema_type(value, item) for item in expected_type)
            if isinstance(expected_type, list)
            else _is_schema_type(value, expected_type)
        )
        if not type_matches:
            expected_text = (
                ", ".join(expected_type)
                if isinstance(expected_type, list)
                else str(expected_type)
            )
            errors.append(
                "%s: expected type %s, got %s"
                % (path or "value", expected_text, _type_name(value))
            )
            return

    if "const" in resolved and value != resolved["const"]:
        errors.append(
            "%s: must be %r (got %r)" % (path or "value", resolved["const"], value)
        )

    if "enum" in resolved and value not in resolved["enum"]:
        errors.append(
            "%s: must be one of %s (got %r)"
            % (path or "value", ", ".join(repr(item) for item in resolved["enum"]), value)
        )

    if isinstance(value, str) and "pattern" in resolved:
        if re.search(resolved["pattern"], value) is None:
            errors.append(
                "%s: does not match pattern %r" % (path or "value", resolved["pattern"])
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in resolved and value < resolved["minimum"]:
            errors.append(
                "%s: must be >= %s (got %s)"
                % (path or "value", resolved["minimum"], value)
            )
        if "maximum" in resolved and value > resolved["maximum"]:
            errors.append(
                "%s: must be <= %s (got %s)"
                % (path or "value", resolved["maximum"], value)
            )

    if isinstance(value, list):
        if "minItems" in resolved and len(value) < resolved["minItems"]:
            errors.append(
                "%s: requires at least %d item(s)"
                % (path or "value", resolved["minItems"])
            )
        if "maxItems" in resolved and len(value) > resolved["maxItems"]:
            errors.append(
                "%s: allows at most %d item(s)" % (path or "value", resolved["maxItems"])
            )
        item_schema = resolved.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, _path_index(path, index), errors)

    if isinstance(value, dict):
        properties = resolved.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for required in resolved.get("required", ()):
            if required not in value:
                errors.append("%s: required property %r is missing" % (path or "value", required))
        if resolved.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(
                        "%s: additional property %r is not allowed"
                        % (path or "value", key)
                    )
        for key, property_schema in properties.items():
            if key in value and isinstance(property_schema, dict):
                _validate_schema(value[key], property_schema, _path_key(path, key), errors)


def _jsonschema_errors(design: Dict[str, Any]) -> List[str]:
    """Run the optional full Draft 2020-12 validator when installed."""
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-not-found]
    except ImportError:
        return []

    validator = Draft202012Validator(SCHEMA)
    messages: List[str] = []
    for error in validator.iter_errors(design):
        path = ""
        for part in error.absolute_path:
            path = _path_index(path, part) if isinstance(part, int) else _path_key(path, str(part))
        messages.append("%s: %s" % (path or "value", error.message))
    return messages


def _library_declares_holes(source: Mapping[str, Any]) -> bool:
    """True when ``source`` is a library part whose entry already declares mounting holes.

    synth (`enclosure._holes_of`) takes envelope holes before ``mount.holes``, so a library
    part such as ``raspberry-pi-4b`` already gets standoffs on its real hole pattern; demanding
    the agent repeat the pattern in the IR only invited a second, possibly wrong, copy. An
    unknown part or bad params is reported by the envelope stage, so it is "no holes" here.
    """
    if source.get("kind") != "library" or not isinstance(source.get("part"), str):
        return False
    try:
        from phimech.components import library

        params = source.get("params")
        envelope = library.make(source["part"], "U1", **(params if isinstance(params, dict)
                                                           else {}))
    except Exception:
        return False
    return bool(getattr(envelope, "holes", ()))


def _semantic_errors(design: Dict[str, Any]) -> List[str]:
    """Check relationships and filesystem facts JSON Schema cannot express."""
    errors: List[str] = []
    components = design.get("components")
    refs: Set[str] = set()
    if isinstance(components, list):
        for index, component in enumerate(components):
            if not isinstance(component, dict):
                continue
            ref = component.get("ref")
            if not isinstance(ref, str):
                continue
            if ref in refs:
                errors.append("components[%d].ref: duplicate component ref %r" % (index, ref))
            refs.add(ref)

            source = component.get("source")
            if isinstance(source, dict):
                kind = source.get("kind")
                if kind in ("kicad_pcb", "import") and isinstance(source.get("path"), str):
                    source_path = Path(source["path"]).expanduser()
                    if not source_path.is_absolute():
                        meta = design.get("_meta")
                        base_dir = meta.get("base_dir") if isinstance(meta, dict) else None
                        source_path = (
                            (Path(base_dir) / source_path)
                            if isinstance(base_dir, str)
                            else source_path
                        )
                    if not source_path.exists():
                        errors.append(
                            "components[%d].source.path: file does not exist: %s"
                            % (index, source["path"])
                        )

                mount = component.get("mount")
                if isinstance(mount, dict) and mount.get("style") == "standoff":
                    if (kind != "kicad_pcb" and not mount.get("holes")
                            and not _library_declares_holes(source)):
                        errors.append(
                            "components[%d].mount.holes: required for standoff mounting "
                            "of a source that declares no holes (only kicad_pcb boards and "
                            "library parts with mounting holes supply their own)" % index
                        )

    placement = design.get("placement")
    graph: Dict[str, List[str]] = {}
    if isinstance(placement, list):
        for index, constraint in enumerate(placement):
            if not isinstance(constraint, dict):
                continue
            ref = constraint.get("ref")
            target = constraint.get("to")
            if isinstance(ref, str) and ref not in refs:
                errors.append(
                    "placement[%d].ref: unknown component ref %r" % (index, ref)
                )
            if isinstance(target, str) and target not in refs:
                errors.append(
                    "placement[%d].to: unknown component ref %r" % (index, target)
                )
            relation = constraint.get("relation")
            if relation == "at" and "position" not in constraint:
                errors.append(
                    "placement[%d].position: required for an 'at' relation" % index
                )
            if isinstance(ref, str) and isinstance(target, str) and ref in refs and target in refs:
                graph.setdefault(ref, []).append(target)

    # DFS is small and gives a useful cycle path instead of merely saying that placement
    # failed.  Edges point from a constrained component to its anchor.
    state: Dict[str, int] = {}
    stack: List[str] = []
    reported_cycles: Set[Tuple[str, ...]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for target in graph.get(node, ()):
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1 and target in stack:
                cycle = tuple(stack[stack.index(target) :] + [target])
                normalized = min(cycle[i:-1] + cycle[:i] for i in range(len(cycle) - 1))
                if normalized not in reported_cycles:
                    reported_cycles.add(normalized)
                    errors.append("placement cycle: %s" % " -> ".join(cycle))
        stack.pop()
        state[node] = 2

    for ref in refs:
        if state.get(ref, 0) == 0:
            visit(ref)

    enclosure = design.get("enclosure")
    manufacturing = design.get("manufacturing")
    if isinstance(enclosure, dict) and isinstance(manufacturing, dict):
        wall = enclosure.get("wall")
        min_wall = manufacturing.get("min_wall")
        if (
            isinstance(wall, (int, float))
            and not isinstance(wall, bool)
            and isinstance(min_wall, (int, float))
            and not isinstance(min_wall, bool)
            and wall < min_wall
        ):
            errors.append(
                "warn: enclosure.wall %.3g is below manufacturing.min_wall %.3g"
                % (wall, min_wall)
            )
        lid = enclosure.get("lid")
        if isinstance(lid, dict):
            screw_count = lid.get("screw_count")
            if (
                isinstance(screw_count, (int, float))
                and not isinstance(screw_count, bool)
                and screw_count < 2
            ):
                errors.append("enclosure.lid.screw_count: must be >= 2")

    return errors


def _dedupe(messages: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for message in messages:
        if message not in seen:
            result.append(message)
            seen.add(message)
    return result


def _plain(message: str) -> Dict[str, Any]:
    """Wrap a string diagnostic in the structured form; the path is the text before the colon."""
    head, separator, _ = message.partition(": ")
    path = head if separator and " " not in head.strip() else "design"
    return {"path": path, "message": message, "suggestion": None, "schema": None, "rule": None}


def _alias_diagnostics(clean_design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Refusals, and the rewrites the loader *would* make, each with its paste-ready fragment.

    A design that came through ``load()`` has no pending rewrites, so only refusals appear.  A
    design validated directly (an MCP ``write_ir`` candidate, a hand-built dict) may still carry
    an alias; naming the canonical spelling beside the schema's "additional property" error is
    what turns that error into a one-line fix.
    """
    outcome = _aliases.analyse(clean_design)
    entries: List[Dict[str, Any]] = []
    for refusal in outcome.refusals:
        entry = refusal.to_dict()
        # Every diagnostic string starts with its path, so the structured form does too.
        entry["message"] = "%s: %s" % (entry["path"], entry["message"])
        entries.append(entry)
    for change in outcome.normalisations:
        path, source, target = change["path"], change["from"], change["to"]
        if change.get("kind", "key") == "key":
            value = change.get("value")
            if isinstance(value, dict) and "to" in value:
                value = value["to"]  # a scaled alias such as radius -> d
            elif value is None:
                value = _value_at(clean_design, path, str(source))
            suggestion: Any = {target: value}
            message = "%s: %r is written %r in the IR" % (path, source, target)
        else:
            key = path.rsplit(".", 1)[-1] or "value"
            suggestion = {key: target}
            message = "%s: %s is read as %s" % (
                path, json.dumps(source, ensure_ascii=False), json.dumps(target, ensure_ascii=False))
        entries.append({
            "path": path,
            "message": message + " (the loader normalises this; `phimech ir normalise` rewrites it)",
            "suggestion": suggestion,
            "schema": None,
            "rule": change["rule"],
        })
    return entries


def _value_at(design: Mapping[str, Any], path: str, key: str) -> Any:
    """The value under ``key`` in the object at ``path`` (``components[0].source``), if any."""
    node: Any = design
    if path and path != "design":
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
            if part.startswith("["):
                index = int(part[1:-1])
                node = node[index] if isinstance(node, list) and index < len(node) else None
            else:
                node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                return "<value?>"
    if isinstance(node, dict) and key in node:
        return node[key]
    return "<value?>"


def diagnostics(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structured diagnostics: ``{path, message, suggestion, schema, rule}`` per finding.

    ``suggestion`` is a paste-ready fragment for the object containing ``path`` (unknown values
    marked ``"<x?>"``) and is None for findings that have no mechanical fix.  ``validate()`` is
    this list rendered to strings; callers that want to show a fix use this one.
    """
    if not isinstance(design, dict):
        return [_plain("value: expected type object, got %s" % _type_name(design))]
    clean_design = _strip_private(design)
    entries: List[Dict[str, Any]] = []
    # Reference failures are found during load (the only place the references still exist), so they
    # are carried on the design and replayed here.
    carried = design.get("_parameter_errors")
    if isinstance(carried, list):
        entries.extend(_plain(str(item)) for item in carried)
    entries.extend(_plain(item) for item in parameter_errors(clean_design))
    schema_errors: List[str] = []
    _validate_schema(clean_design, SCHEMA, "", schema_errors)
    entries.extend(_plain(item) for item in schema_errors)
    entries.extend(_alias_diagnostics(clean_design))
    entries.extend(_plain(item) for item in _semantic_errors(design))
    entries.extend(_plain(item) for item in _jsonschema_errors(clean_design))
    unique: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for entry in entries:
        rendered = _render_diagnostic(entry)
        if rendered not in seen:
            seen.add(rendered)
            unique.append(entry)
    return unique


def _render_diagnostic(entry: Mapping[str, Any]) -> str:
    message = str(entry["message"])
    if entry.get("suggestion") is None:
        return message
    return "%s — suggestion: %s" % (
        message, json.dumps(entry["suggestion"], ensure_ascii=False))


def validate(design: Dict[str, Any]) -> List[str]:
    """Return schema and semantic diagnostics; an empty list means the design is valid.

    A diagnostic with a mechanical fix ends in ``— suggestion: {...}``, a fragment to paste into
    the object it names; :func:`diagnostics` returns the same findings structured.
    """
    return [_render_diagnostic(entry) for entry in diagnostics(design)]


def load_and_validate(
    path: Union[str, Path], normalise: bool = True
) -> Tuple[Dict[str, Any], List[str]]:
    """Load once so path resolution and validation inspect the same design object."""
    design = load(path, normalise=normalise)
    return design, validate(design)


def dumps(design: Dict[str, Any]) -> str:
    """Serialize a deterministic, metadata-free JSON representation of a design."""
    return json.dumps(_strip_private(design), ensure_ascii=False, sort_keys=True, indent=2)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a PhiMech design IR")
    subparsers = parser.add_subparsers(dest="command")
    check_parser = subparsers.add_parser("check", help="validate one JSON or YAML design")
    check_parser.add_argument("file")
    args = parser.parse_args(argv)
    if args.command != "check":
        parser.print_help(sys.stderr)
        return 2

    try:
        _, errors = load_and_validate(args.file)
    except Exception as exc:
        print("error: %s" % exc)
        return 1
    if errors:
        for error in errors:
            print(error)
        return 1
    print("valid: %s" % args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "SCHEMA",
    "apply_defaults",
    "diagnostics",
    "dumps",
    "load",
    "load_and_validate",
    "validate",
]
