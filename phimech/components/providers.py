"""Dispatch design-IR component sources to envelope providers.

This module is the narrow boundary between the design IR and component implementations.  It
keeps optional PCB parsers, CAD backends, and the import cache lazy so a simple box does not
require either CAD executable to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phimech.csg.ast import Cube, Cylinder, Node, Rotate, Translate, from_dict
from phimech.components.envelope import Access, Envelope, FACES, Hole

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def _path(value: Any, base_dir: Optional[Any]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser() / path
    return path.resolve()


def _optional_number(value: Any, label: str) -> Optional[float]:
    """An explicit IR override as a float, or ``None`` so the provider measures it itself."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a number, got {!r}".format(label, value)) from None


def _cylinder_size(d: float, h: float, axis: str) -> Vec3:
    if axis == "x":
        return (h, d, d)
    if axis == "y":
        return (d, h, d)
    if axis == "z":
        return (d, d, h)
    raise ValueError("cylinder axis must be one of 'x', 'y', or 'z'")


def _cylinder_body(d: float, h: float, axis: str) -> Node:
    radius = d / 2.0
    primitive = Cylinder(h=h, r=radius, segments=48)
    if axis == "x":
        return Translate((0.0, radius, radius), [Rotate((0.0, 90.0, 0.0), [primitive])])
    if axis == "y":
        return Translate((radius, 0.0, radius), [Rotate((-90.0, 0.0, 0.0), [primitive])])
    if axis == "z":
        return Translate((radius, radius, 0.0), [primitive])
    raise ValueError("cylinder axis must be one of 'x', 'y', or 'z'")


def _source_envelope(
    component: Dict[str, Any],
    *,
    base_dir: Optional[Any],
) -> Envelope:
    source = component["source"]
    kind = source["kind"]
    ref = str(component["ref"])

    if kind == "box":
        size = tuple(float(v) for v in source["size"])
        return Envelope(ref=ref, size=size, body=Cube(size), source="box")  # type: ignore[arg-type]

    if kind == "cylinder":
        d = float(source["d"])
        h = float(source["h"])
        axis = str(source.get("axis", "z"))
        size = _cylinder_size(d, h, axis)
        return Envelope(ref=ref, size=size, body=_cylinder_body(d, h, axis), source="cylinder")

    if kind == "kicad_pcb":
        # PCB extraction is optional and belongs to another provider; importing it here would
        # make even library-only designs fail when the parser's dependencies are absent.
        from phimech.components.pcb import envelope_from_pcb

        # Every dimension is `None` unless the IR *explicitly* overrides it, so the extractor
        # supplies the referenced board's own numbers.  This used to pass hardcoded fallbacks
        # (`thickness=1.6`, `component_height_bottom=0.0`) whose effect was to *replace* real
        # extracted values: through-hole lead protrusion in a board fixture
        # became 0.0, so `mount_plane_z` was 0 instead of 3.0 and a 1 mm standoff looked legal
        # for a board that needs at least 3 mm to seat.  A default that overrides measured
        # geometry is not a default.
        return envelope_from_pcb(
            _path(source["path"], base_dir),
            ref=ref,
            thickness=_optional_number(source.get("thickness"), "source.thickness"),
            component_height_top=_optional_number(
                source.get("component_height_top"), "source.component_height_top"
            ),
            component_height_bottom=_optional_number(
                source.get("component_height_bottom"), "source.component_height_bottom"
            ),
            expose=tuple(source.get("expose", ())),
        )

    if kind == "import":
        from phimech.components.imports import envelope_from_import

        return envelope_from_import(
            source["path"],
            ref,
            scale=float(source.get("scale", 1.0)),
            base_dir=base_dir,
        )

    if kind == "library":
        from phimech.components.library import make

        params = source.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("library source params must be an object")
        return make(str(source["part"]), ref=ref, **params)

    if kind == "csg":
        body = from_dict(source["csg"])
        try:
            from phimech.backends.base import for_format

            backend = for_format("stl")
            measurement = backend.measure(body)
        except Exception as exc:
            message = (
                "cannot measure csg component {!r}; an available CAD backend is required: {}"
            ).format(ref, exc)
            raise RuntimeError(message) from exc
        size = measurement.bbox.size
        if any(float(v) <= 0.0 for v in size):
            raise RuntimeError(
                "backend returned a non-positive CSG bounding box for {!r}".format(ref)
            )
        return Envelope(ref=ref, size=size, body=body, source="csg")

    raise ValueError("unknown component source kind {!r}".format(kind))


def _face_centre(face: str, size: Vec3) -> Vec2:
    w, d, h = size
    if face in ("top", "bottom"):
        return (w / 2.0, d / 2.0)
    if face in ("front", "back"):
        return (w / 2.0, h / 2.0)
    return (d / 2.0, h / 2.0)


def _access_from_dict(entry: Dict[str, Any], size: Vec3, index: int) -> Access:
    face = str(entry["face"])
    if face not in FACES:
        raise ValueError("access face {!r} is not one of {}".format(face, ", ".join(FACES)))
    raw_size = entry.get("size")
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raise ValueError(
            "access size must contain two numbers"
        )
    access_size = (float(raw_size[0]), float(raw_size[1]))
    raw_at = entry.get("at")
    if raw_at is None:
        at = _face_centre(face, size)
    elif isinstance(raw_at, (list, tuple)) and len(raw_at) == 2:
        at = (float(raw_at[0]), float(raw_at[1]))
    else:
        raise ValueError("access at must contain two numbers when provided")
    shape = str(entry.get("shape", "rect"))
    if shape not in ("rect", "round", "slot"):
        raise ValueError("access shape {!r} is not rect, round, or slot".format(shape))
    return Access(
        name=str(entry.get("name", "ACCESS{}".format(index))),
        face=face,
        at=at,
        size=access_size,
        shape=shape,
        tolerance=float(entry.get("tolerance", 0.5)),
        protrusion=float(entry.get("protrusion", 0.0)),
        through_lid=bool(entry.get("through_lid", False)),
    )


def _apply_component_fields(component: Dict[str, Any], envelope: Envelope) -> Envelope:
    extra_access: List[Access] = []
    raw_access = component.get("access", ())
    if not isinstance(raw_access, (list, tuple)):
        raise ValueError("component access must be an array")
    for index, entry in enumerate(raw_access, 1):
        if not isinstance(entry, dict):
            raise ValueError("component access entries must be objects")
        extra_access.append(_access_from_dict(entry, envelope.size, index))

    holes = envelope.holes
    mount = component.get("mount") or {}
    if not isinstance(mount, dict):
        raise ValueError("component mount must be an object")
    explicit_holes = mount.get("holes", ())
    if not holes and explicit_holes:
        d = float(mount.get("hole_d", 3.2))
        converted: List[Hole] = []
        for raw in explicit_holes:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError("mount.holes entries must contain two numbers")
            converted.append(Hole((float(raw[0]), float(raw[1])), d))
        holes = tuple(converted)

    name = component["name"] if "name" in component else envelope.name
    thermal_w = float(component["thermal_w"]) if "thermal_w" in component else envelope.thermal_w
    return Envelope(
        ref=envelope.ref,
        size=envelope.size,
        footprint=envelope.footprint,
        holes=holes,
        access=envelope.access + tuple(extra_access),
        mount_plane_z=envelope.mount_plane_z,
        mount_thickness=envelope.mount_thickness,
        body=envelope.body,
        name=str(name),
        source=envelope.source,
        thermal_w=thermal_w,
        meta=dict(envelope.meta),
    )


def envelope_for(component: Dict[str, Any], *, base_dir: Optional[Any] = None) -> Envelope:
    """Build one envelope from a component IR object and apply its common fields."""
    if not isinstance(component, dict) or "source" not in component:
        raise ValueError("component must be an object with a source")
    envelope = _source_envelope(component, base_dir=base_dir)
    return _apply_component_fields(component, envelope)


def envelopes_for(design: Dict[str, Any], *, base_dir: Optional[Any] = None) -> Dict[str, Envelope]:
    """Build all component envelopes, keyed by their reference designators."""
    components = design.get("components", ())
    if not isinstance(components, (list, tuple)):
        raise ValueError("design components must be an array")
    return {
        str(component["ref"]): envelope_for(component, base_dir=base_dir)
        for component in components
    }


__all__ = ["envelope_for", "envelopes_for"]
