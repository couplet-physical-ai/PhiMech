"""Envelope — what synth needs to know about a component, whatever its geometry source.

Every provider in `phimech/components/` (PCB extractor, parametric library, STEP import, raw
CSG) returns one of these. `synth/place.py` positions envelopes; `synth/enclosure.py` grows
walls around their union; `synth/features.py` turns `access` entries into cutouts and `holes`
into bosses. The provider's detailed geometry (`body`) is only used for the assembly render
and the exploded view — placement and enclosure logic reason about the envelope alone, which
is what keeps them deterministic and fast.

Coordinate convention (component-local frame)
--------------------------------------------
* Origin at the **minimum X/Y corner of the footprint, at Z = 0 on the mounting face** — i.e.
  the component sits on the XY plane and extends into +Z. A PCB's mounting face is its
  bottom copper side; `size[2]` therefore includes bottom-side components + board + top-side
  components, and `mount_plane_z` says where the board itself starts.
* `footprint` is the 2D outline in that frame (CCW). For a rectangular part it is 4 points.
* `holes` are XY centres in that frame, on the mounting face.
* `access` features are on a named face; `at` is 2D in that face's own frame (see
  `face_frame()` in synth/features.py for the mapping).

stdlib only; must import under python 3.9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from phimech.csg.ast import Node

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

FACES = ("top", "bottom", "left", "right", "front", "back")


@dataclass(frozen=True)
class Hole:
    """A mounting hole on the component's mounting face."""

    at: Vec2
    d: float
    ref: Optional[str] = None  # e.g. footprint ref "H1"
    plated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"at": list(self.at), "d": self.d, "ref": self.ref, "plated": self.plated}


@dataclass(frozen=True)
class Access:
    """A feature that must reach the outside: a port, button, display window, LED, antenna."""

    name: str
    face: str  # one of FACES, in the component-local frame
    at: Vec2  # centre on that face, mm
    size: Vec2  # width, height on that face
    shape: str = "rect"  # rect | round | slot
    tolerance: float = 0.5
    protrusion: float = 0.0  # how far the feature already sticks out past the envelope face
    through_lid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "face": self.face,
            "at": list(self.at),
            "size": list(self.size),
            "shape": self.shape,
            "tolerance": self.tolerance,
            "protrusion": self.protrusion,
            "through_lid": self.through_lid,
        }


@dataclass(frozen=True)
class Envelope:
    ref: str
    size: Vec3  # overall bounding size in the local frame (mm)
    footprint: Tuple[Vec2, ...] = ()  # 2D outline, CCW; empty → rectangle from size
    holes: Tuple[Hole, ...] = ()
    access: Tuple[Access, ...] = ()
    mount_plane_z: float = 0.0  # where the rigid mounting surface sits (PCB: bottom of board)
    mount_thickness: float = 0.0  # thickness of that surface (PCB: board thickness)
    body: Optional[Node] = None  # detailed/placeholder geometry for renders; may be None
    name: str = ""
    source: str = ""  # provider tag: "kicad_pcb", "library:18650", "import", "box", ...
    thermal_w: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    # convenience -----------------------------------------------------------------------

    @property
    def outline(self) -> Tuple[Vec2, ...]:
        if self.footprint:
            return self.footprint
        w, d, _ = self.size
        return ((0.0, 0.0), (w, 0.0), (w, d), (0.0, d))

    @property
    def top_z(self) -> float:
        return self.size[2]

    def grown(self, clearance: float) -> "Envelope":
        """Envelope with `clearance` added on every side (keep-out). Footprint is offset by
        the simple rectangular expansion — polygon offsetting lives in synth."""
        w, d, h = self.size
        c = clearance
        return Envelope(
            ref=self.ref,
            size=(w + 2 * c, d + 2 * c, h + 2 * c),
            footprint=(),  # rectangular keep-out; polygon offsetting is synth's job
            holes=tuple(Hole((hh.at[0] + c, hh.at[1] + c), hh.d, hh.ref, hh.plated) for hh in self.holes),
            access=tuple(
                Access(a.name, a.face, (a.at[0] + c, a.at[1] + c), a.size, a.shape, a.tolerance, a.protrusion, a.through_lid)
                for a in self.access
            ),
            mount_plane_z=self.mount_plane_z + c,
            mount_thickness=self.mount_thickness,
            body=self.body,
            name=self.name,
            source=self.source,
            thermal_w=self.thermal_w,
            meta=dict(self.meta, clearance=c),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "name": self.name,
            "source": self.source,
            "size": list(self.size),
            "footprint": [list(p) for p in self.outline],
            "holes": [h.to_dict() for h in self.holes],
            "access": [a.to_dict() for a in self.access],
            "mount_plane_z": self.mount_plane_z,
            "mount_thickness": self.mount_thickness,
            "thermal_w": self.thermal_w,
            "has_body": self.body is not None,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Placed:
    """An envelope after `synth/place.py`: the transform from local frame to assembly frame."""

    envelope: Envelope
    position: Vec3  # translation of the local origin in assembly space
    rotation: Vec3 = (0.0, 0.0, 0.0)  # Euler degrees, applied before translation

    @property
    def ref(self) -> str:
        return self.envelope.ref

    def world_bbox(self) -> Tuple[Vec3, Vec3]:
        """Axis-aligned bbox in assembly space. Rotations are restricted to multiples of 90°
        by place.py, so the bbox is exact — assert that here rather than approximate."""
        rx, ry, rz = self.rotation
        for r in (rx, ry, rz):
            if abs(r / 90.0 - round(r / 90.0)) > 1e-9:
                raise ValueError(f"{self.ref}: rotation {self.rotation} is not a multiple of 90°")
        w, d, h = self.envelope.size
        corners = [(x, y, z) for x in (0.0, w) for y in (0.0, d) for z in (0.0, h)]
        pts = [_rot90(c, self.rotation) for c in corners]
        mn = tuple(min(p[i] for p in pts) + self.position[i] for i in range(3))
        mx = tuple(max(p[i] for p in pts) + self.position[i] for i in range(3))
        return mn, mx  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        mn, mx = self.world_bbox()
        return {
            "ref": self.ref,
            "position": list(self.position),
            "rotation": list(self.rotation),
            "world_bbox": {"min": list(mn), "max": list(mx)},
            "envelope": self.envelope.to_dict(),
        }


def _rot90(p: Vec3, rot: Vec3) -> Vec3:
    """Rotate a point by Euler angles that are multiples of 90° (X, then Y, then Z)."""
    x, y, z = p
    for axis, deg in enumerate(rot):
        k = int(round(deg / 90.0)) % 4
        for _ in range(k):
            if axis == 0:
                y, z = -z, y
            elif axis == 1:
                x, z = z, -x
            else:
                x, y = -y, x
    return (float(x), float(y), float(z))


__all__ = ["FACES", "Hole", "Access", "Envelope", "Placed", "Vec2", "Vec3"]
