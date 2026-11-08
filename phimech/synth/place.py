"""Resolve relative component placement into deterministic assembly coordinates.

Placement deliberately keeps two boxes for each component.  ``Placed`` items are the real
component envelopes, while private keep-out ``Placed`` items use ``Envelope.grown``.  The
latter are the boxes used for all relative constraints and overlap checks.  Keeping the body
origin separate is important: a clearance grows on *both* sides, so a component at the
assembly origin has a keep-out that extends into negative coordinates.

The first unconstrained component's body origin remains at ``(0, 0, 0)``.  We do not shift
all results so that the keep-out union starts at zero; the enclosure synthesizer owns that
later offset.  If several components have no constraints, they are put on a +X shelf with a
2.0 mm air gap.  ``auto_arrange`` uses the same convention and can wrap shelves along +Y.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phimech.components.envelope import Envelope, Placed, _rot90


Vec3 = Tuple[float, float, float]
BBox = Tuple[Vec3, Vec3]
DEFAULT_SHELF_GAP = 2.0
_OVERLAP_TOLERANCE = 1e-6
_RELATIONS = {
    "at",
    "above",
    "below",
    "beside",
    "behind",
    "in_front",
    "stacked_on",
    "centered_in",
}
_ALIGNMENTS = {"min", "center", "max"}


def _vec3(value: Sequence[float], name: str) -> Vec3:
    """Coerce a schema vector while producing a useful error for hand-written IR."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("{} must be a 3-item vector, got {!r}".format(name, value))
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        raise ValueError("{} must contain numbers, got {!r}".format(name, value)) from None


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vec3, factor: float) -> Vec3:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def _union(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return (
        tuple(min(box[0][axis] for box in boxes) for axis in range(3)),
        tuple(max(box[1][axis] for box in boxes) for axis in range(3)),
    )  # type: ignore[return-value]


def _rotated_bounds(envelope: Envelope, rotation: Vec3) -> BBox:
    """Get exact rotated bounds through the shared ``Placed`` implementation.

    ``Placed`` validates the right-angle restriction and applies the same Euler convention
    used by the renderers, avoiding a second subtly different size calculation here.
    """
    return Placed(envelope, (0.0, 0.0, 0.0), rotation).world_bbox()


def _clearance_rotation(clearance: float, rotation: Vec3) -> Vec3:
    """Return the rotated local offset from a grown-box origin to the body origin."""
    return _rot90((clearance, clearance, clearance), rotation)


def _aligned_min(anchor_min: float, anchor_max: float, target_size: float, align: str) -> float:
    if align == "min":
        return anchor_min
    if align == "max":
        return anchor_max - target_size
    return (anchor_min + anchor_max - target_size) / 2.0


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return "{:.3f}".format(value).rstrip("0").rstrip(".")


def _format_vec(value: Vec3) -> str:
    return "({})".format(", ".join(_format_number(v) for v in value))


@dataclass
class Placement:
    """Resolved component placement plus diagnostics.

    ``items`` and ``by_ref`` contain body envelopes, which is what assembly rendering needs.
    ``bbox`` and ``errors`` use the grown keep-out boxes.  The private maps are intentionally
    optional constructor fields so callers can still construct a four-field ``Placement`` in
    small tests or downstream utilities.
    """

    items: List[Placed]
    by_ref: Dict[str, Placed]
    errors: List[str]
    order: List[str]
    _keepouts: Dict[str, Placed] = field(default_factory=dict, repr=False, compare=False)
    _reasons: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def keepouts(self) -> List[Placed]:
        """Return keep-out placements in resolution order for enclosure/gate consumers."""
        return [self._keepouts[ref] for ref in self.order if ref in self._keepouts]

    @property
    def keepout_by_ref(self) -> Dict[str, Placed]:
        """Expose the grown boxes without changing the body-oriented public item contract."""
        return self._keepouts

    @property
    def body_items(self) -> List[Placed]:
        """Alias for ``items`` that makes the body/keep-out distinction explicit."""
        return self.items

    def bbox(self) -> BBox:
        """Return the union of all grown keep-out boxes."""
        boxes = [placed.world_bbox() for placed in self.keepouts]
        if not boxes:
            boxes = [placed.world_bbox() for placed in self.items]
        return _union(boxes)

    def bbox_bodies(self) -> BBox:
        """Return the union of the un-grown component body boxes."""
        return _union([placed.world_bbox() for placed in self.items])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bodies, keep-out bounds, diagnostics, and both assembly extents."""
        item_dicts: List[Dict[str, Any]] = []
        for item in self.items:
            data = item.to_dict()
            keepout = self._keepouts.get(item.ref)
            if keepout is not None:
                keep_min, keep_max = keepout.world_bbox()
                data["keepout_bbox"] = {"min": list(keep_min), "max": list(keep_max)}
            item_dicts.append(data)
        body_min, body_max = self.bbox_bodies()
        keep_min, keep_max = self.bbox()
        return {
            "items": item_dicts,
            "order": list(self.order),
            "errors": list(self.errors),
            "bbox": {"min": list(keep_min), "max": list(keep_max)},
            "bbox_bodies": {"min": list(body_min), "max": list(body_max)},
        }

    def explain(self) -> str:
        """Return one concise, human-readable resolution line per component."""
        lines: List[str] = []
        for ref in self.order:
            item = self.by_ref.get(ref)
            if item is None:
                continue
            reason = self._reasons.get(ref, "unconstrained")
            body_min, body_max = item.world_bbox()
            lines.append(
                "{} {} → pos {} bbox {}–{}".format(
                    ref,
                    reason,
                    _format_vec(item.position),
                    _format_vec(body_min),
                    _format_vec(body_max),
                )
            )
        return "\n".join(lines)


def _constraint_map(
    design: Dict[str, Any], component_refs: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Validate and index the one constraint that can place each component."""
    refs = set(component_refs)
    constraints = design.get("placement", []) or []
    if not isinstance(constraints, list):
        raise ValueError("design['placement'] must be a list")
    out: Dict[str, Dict[str, Any]] = {}
    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise ValueError("each placement constraint must be an object")
        ref = constraint.get("ref")
        relation = constraint.get("relation")
        if ref not in refs:
            raise ValueError("placement constraint refers to unknown component {!r}".format(ref))
        if ref in out:
            raise ValueError("component {!r} has more than one placement constraint".format(ref))
        if relation not in _RELATIONS:
            raise ValueError("unsupported placement relation {!r}".format(relation))
        if relation != "at":
            anchor = constraint.get("to")
            if anchor not in refs:
                raise ValueError(
                    "{} constraint for {} needs known anchor 'to'".format(relation, ref)
                )
            if anchor == ref:
                raise ValueError("component {!r} cannot be placed relative to itself".format(ref))
        align = constraint.get("align", "center")
        if align not in _ALIGNMENTS:
            raise ValueError("placement align must be min, center, or max; got {!r}".format(align))
        try:
            float(constraint.get("gap", 2.0))
        except (TypeError, ValueError):
            raise ValueError("placement gap must be a number") from None
        # Validate vectors now, before any partial resolution can obscure the bad entry.
        _vec3(constraint.get("offset", (0.0, 0.0, 0.0)), "placement offset")
        if relation == "at" and "position" in constraint:
            _vec3(constraint["position"], "placement position")
        out[ref] = constraint
    return out


def _resolution_order(
    component_refs: Sequence[str], constraints: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Stable topological sort, preferring unconstrained roots as the requested shelf pass."""
    index = {ref: number for number, ref in enumerate(component_refs)}
    dependencies: Dict[str, Optional[str]] = {ref: None for ref in component_refs}
    children: Dict[str, List[str]] = {ref: [] for ref in component_refs}
    indegree: Dict[str, int] = {ref: 0 for ref in component_refs}
    for ref, constraint in constraints.items():
        if constraint.get("relation") == "at":
            continue
        anchor = constraint["to"]
        dependencies[ref] = anchor
        children[anchor].append(ref)
        indegree[ref] += 1

    ready = [ref for ref in component_refs if indegree[ref] == 0]
    resolved: List[str] = []
    while ready:
        ready.sort(key=lambda ref: (0 if ref not in constraints else 1, index[ref]))
        ref = ready.pop(0)
        resolved.append(ref)
        for child in children[ref]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(resolved) != len(component_refs):
        cycle = [ref for ref in component_refs if indegree[ref] > 0]
        raise ValueError(
            "placement constraints contain a cycle involving {}".format(", ".join(cycle))
        )
    return resolved


def _raw_position_for_keepout_min(
    grown: Envelope, clearance: float, rotation: Vec3, target_min: Vec3
) -> Vec3:
    """Convert a desired grown-box min corner into the body's local-origin position."""
    rotated_min, _ = _rotated_bounds(grown, rotation)
    keepout_origin = _sub(target_min, rotated_min)
    return _add(keepout_origin, _clearance_rotation(clearance, rotation))


def _keepout_position_for_raw(
    clearance: float, rotation: Vec3, raw_position: Vec3
) -> Vec3:
    """Place the grown origin so its inner body starts at ``raw_position``."""
    return _sub(raw_position, _clearance_rotation(clearance, rotation))


def _relation_target_min(
    relation: str,
    anchor_box: BBox,
    target_size: Vec3,
    gap: float,
    align: str,
) -> Vec3:
    """Compute the grown target box min for one relative relation."""
    if relation == "stacked_on":
        relation = "above"
    anchor_min, anchor_max = anchor_box
    target = [0.0, 0.0, 0.0]
    if relation == "beside":
        target[0] = anchor_max[0] + gap
        target[1] = _aligned_min(anchor_min[1], anchor_max[1], target_size[1], align)
        target[2] = _aligned_min(anchor_min[2], anchor_max[2], target_size[2], align)
    elif relation == "behind":
        target[0] = _aligned_min(anchor_min[0], anchor_max[0], target_size[0], align)
        target[1] = anchor_max[1] + gap
        target[2] = _aligned_min(anchor_min[2], anchor_max[2], target_size[2], align)
    elif relation == "in_front":
        target[0] = _aligned_min(anchor_min[0], anchor_max[0], target_size[0], align)
        target[1] = anchor_min[1] - gap - target_size[1]
        target[2] = _aligned_min(anchor_min[2], anchor_max[2], target_size[2], align)
    elif relation == "above":
        target[0] = _aligned_min(anchor_min[0], anchor_max[0], target_size[0], align)
        target[1] = _aligned_min(anchor_min[1], anchor_max[1], target_size[1], align)
        target[2] = anchor_max[2] + gap
    elif relation == "below":
        target[0] = _aligned_min(anchor_min[0], anchor_max[0], target_size[0], align)
        target[1] = _aligned_min(anchor_min[1], anchor_max[1], target_size[1], align)
        target[2] = anchor_min[2] - gap - target_size[2]
    elif relation == "centered_in":
        target[0] = (anchor_min[0] + anchor_max[0] - target_size[0]) / 2.0
        target[1] = (anchor_min[1] + anchor_max[1] - target_size[1]) / 2.0
        target[2] = (anchor_min[2] + anchor_max[2] - target_size[2]) / 2.0
    else:
        raise ValueError("relation {!r} does not have an anchor target".format(relation))
    return (target[0], target[1], target[2])


def _overlap_errors(order: Sequence[str], keepouts: Dict[str, Placed]) -> List[str]:
    errors: List[str] = []
    for index, left_ref in enumerate(order):
        left_min, left_max = keepouts[left_ref].world_bbox()
        for right_ref in order[index + 1 :]:
            right_min, right_max = keepouts[right_ref].world_bbox()
            intersection = tuple(
                min(left_max[axis], right_max[axis]) - max(left_min[axis], right_min[axis])
                for axis in range(3)
            )
            if all(depth > _OVERLAP_TOLERANCE for depth in intersection):
                errors.append(
                    "overlap: {} and {} (grown AABB intersection {})".format(
                        left_ref, right_ref, _format_vec(intersection)  # type: ignore[arg-type]
                    )
                )
    return errors


def place(design: Dict[str, Any], envelopes: Dict[str, Envelope]) -> Placement:
    """Resolve design constraints while measuring all separations on grown boxes.

    The returned body positions are the actual component-origin positions.  A component with
    clearance ``c`` therefore has a private keep-out origin offset by the rotated vector
    ``(c, c, c)``; this makes a beside gap include both components' clearances rather than
    accidentally treating the expanded-box origin as the component origin.
    """
    components = design.get("components", [])
    if not isinstance(components, list):
        raise ValueError("design['components'] must be a list")

    component_refs: List[str] = []
    component_by_ref: Dict[str, Dict[str, Any]] = {}
    raw_by_ref: Dict[str, Envelope] = {}
    grown_by_ref: Dict[str, Envelope] = {}
    rotations: Dict[str, Vec3] = {}
    clearances: Dict[str, float] = {}

    for component in components:
        if not isinstance(component, dict):
            raise ValueError("each component must be an object")
        ref = component.get("ref")
        if not isinstance(ref, str) or not ref:
            raise ValueError("each component needs a non-empty ref")
        if ref in component_by_ref:
            raise ValueError("duplicate component ref {!r}".format(ref))
        if ref not in envelopes:
            raise ValueError("no envelope supplied for component {!r}".format(ref))
        try:
            clearance = float(component.get("clearance", 1.0))
        except (TypeError, ValueError):
            raise ValueError("{} clearance must be a number".format(ref)) from None
        if clearance < 0.0:
            raise ValueError("{} clearance must be non-negative".format(ref))
        rotation = _vec3(
            component.get("orientation", (0.0, 0.0, 0.0)), "{} orientation".format(ref)
        )
        for angle in rotation:
            if abs(angle / 90.0 - round(angle / 90.0)) > 1e-9:
                raise ValueError(
                    "{} orientation {} is not a multiple of 90 degrees".format(ref, rotation)
                )
        component_refs.append(ref)
        component_by_ref[ref] = component
        raw_by_ref[ref] = envelopes[ref]
        grown_by_ref[ref] = envelopes[ref].grown(clearance)
        rotations[ref] = rotation
        clearances[ref] = clearance

    constraints = _constraint_map(design, component_refs)
    order = _resolution_order(component_refs, constraints)
    bodies: Dict[str, Placed] = {}
    keepouts: Dict[str, Placed] = {}
    reasons: Dict[str, str] = {}
    shelf_cursor: Optional[float] = None

    for ref in order:
        component = component_by_ref[ref]
        raw = raw_by_ref[ref]
        grown = grown_by_ref[ref]
        rotation = rotations[ref]
        clearance = clearances[ref]
        rotated_min, rotated_max = _rotated_bounds(grown, rotation)
        target_size = (
            rotated_max[0] - rotated_min[0],
            rotated_max[1] - rotated_min[1],
            rotated_max[2] - rotated_min[2],
        )
        constraint = constraints.get(ref)

        if constraint is None:
            if shelf_cursor is None:
                raw_position = (0.0, 0.0, 0.0)
            else:
                # Keep all shelf bodies on the same local Y/Z origin while using the
                # grown X min for the air gap.  This is exact for arbitrary right-angle
                # rotations and guarantees that shelf entries cannot overlap in X.
                rotated_clearance = _clearance_rotation(clearance, rotation)
                raw_position = (
                    shelf_cursor - rotated_min[0] + rotated_clearance[0],
                    0.0,
                    0.0,
                )
            reasons[ref] = "unconstrained"
        elif constraint["relation"] == "at":
            requested = _vec3(constraint.get("position", (0.0, 0.0, 0.0)), "placement position")
            offset = _vec3(constraint.get("offset", (0.0, 0.0, 0.0)), "placement offset")
            raw_position = _add(requested, offset)
            reasons[ref] = "at"
        else:
            relation = constraint["relation"]
            anchor_ref = constraint["to"]
            anchor = keepouts[anchor_ref]
            gap = float(constraint.get("gap", 2.0))
            align = constraint.get("align", "center")
            if relation == "stacked_on":
                gap = 0.0
                align = "center"
            target_min = _relation_target_min(
                relation,
                anchor.world_bbox(),
                target_size,
                gap,
                align,
            )
            offset = _vec3(constraint.get("offset", (0.0, 0.0, 0.0)), "placement offset")
            raw_position = _raw_position_for_keepout_min(
                grown,
                clearance,
                rotation,
                _add(target_min, offset),
            )
            reasons[ref] = "{} {}".format(relation, anchor_ref)

        body = Placed(raw, raw_position, rotation)
        keepout_position = _keepout_position_for_raw(clearance, rotation, raw_position)
        keepout = Placed(grown, keepout_position, rotation)
        bodies[ref] = body
        keepouts[ref] = keepout

        if constraint is None:
            shelf_cursor = keepout.world_bbox()[1][0] + DEFAULT_SHELF_GAP

    items = [bodies[ref] for ref in order]
    errors = _overlap_errors(order, keepouts)
    return Placement(items, bodies, errors, order, keepouts, reasons)


def auto_arrange(
    envelopes: Dict[str, Envelope],
    *,
    gap: float = DEFAULT_SHELF_GAP,
    max_width: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Create a simple +X shelf with optional +Y row wrapping.

    The generated constraints intentionally use ``at`` only for the first item and row starts;
    all other items are relative, so the result remains useful if the placement resolver later
    receives envelopes with different clearances.  ``max_width`` is measured in the supplied
    envelope widths and a component wider than the limit is still emitted as a one-item row.
    """
    try:
        gap = float(gap)
    except (TypeError, ValueError):
        raise ValueError("auto_arrange gap must be a number") from None
    if max_width is not None:
        try:
            max_width = float(max_width)
        except (TypeError, ValueError):
            raise ValueError("auto_arrange max_width must be a number") from None
        if max_width <= 0.0:
            raise ValueError("auto_arrange max_width must be positive")

    refs = list(envelopes.keys())
    if not refs:
        return []
    constraints: List[Dict[str, Any]] = []
    row_first: Optional[str] = None
    previous: Optional[str] = None
    row_width = 0.0
    row_first_depth = 0.0
    row_depth = 0.0
    for ref in refs:
        width, depth, _ = envelopes[ref].size
        starts_new_row = (
            previous is not None
            and max_width is not None
            and row_width + gap + width > max_width + _OVERLAP_TOLERANCE
        )
        if previous is None:
            constraint: Dict[str, Any] = {
                "ref": ref,
                "relation": "at",
                "position": [0.0, 0.0, 0.0],
            }
            row_first = ref
            row_width = width
            row_first_depth = depth
            row_depth = depth
        elif starts_new_row:
            assert row_first is not None
            # ``behind`` can align X to the first item but only knows that anchor's
            # depth.  Offset Y by the rest of the previous row so rows clear its
            # deepest item, while preserving a relative constraint in the output.
            constraint = {
                "ref": ref,
                "relation": "behind",
                "to": row_first,
                "gap": gap,
                "align": "min",
                "offset": [0.0, row_depth - row_first_depth, 0.0],
            }
            row_first = ref
            row_width = width
            row_first_depth = depth
            row_depth = depth
        else:
            constraint = {
                "ref": ref,
                "relation": "beside",
                "to": previous,
                "gap": gap,
                "align": "min",
            }
            row_width += gap + width
            row_depth = max(row_depth, depth)
        constraints.append(constraint)
        previous = ref
    return constraints


__all__ = ["Placement", "place", "auto_arrange", "DEFAULT_SHELF_GAP"]
