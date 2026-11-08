"""Small, dependency-free STL metrics used by the design gates.

The check layer intentionally has its own reader instead of importing a CAD backend.  This
keeps gates useful after a render, when only an STL path may be available, and lets the
stdlib-only subset run on the system Python.  Coordinates are millimetres.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Triangle = Tuple[Vec3, Vec3, Vec3]
BBox = Tuple[Vec3, Vec3]

_MAX_ESTIMATE_TRIANGLES = 20000
_EPSILON = 1.0e-12


def _triangle(value: Sequence[Sequence[float]]) -> Triangle:
    if len(value) != 3:
        raise ValueError("triangle must contain three vertices")
    points: List[Vec3] = []
    for point in value:
        if len(point) != 3:
            raise ValueError("STL vertex must contain three coordinates")
        points.append((float(point[0]), float(point[1]), float(point[2])))
    return (points[0], points[1], points[2])


def _ascii_triangles(data: bytes) -> Optional[List[Triangle]]:
    """Return ASCII facets, or None when the byte stream is not text STL."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    vertices: List[Vec3] = []
    found = False
    for line in text.splitlines():
        fields = line.strip().split()
        if not fields or fields[0].lower() != "vertex":
            continue
        found = True
        if len(fields) != 4:
            raise ValueError("invalid ASCII STL vertex record")
        try:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError as exc:
            raise ValueError("invalid ASCII STL vertex number") from exc
    if not found:
        return None
    if len(vertices) % 3:
        raise ValueError("ASCII STL has an incomplete triangle")
    triangles = [_triangle(vertices[i : i + 3]) for i in range(0, len(vertices), 3)]
    if not triangles:
        raise ValueError("STL contains no triangles")
    return triangles


def _binary_triangles(data: bytes) -> List[Triangle]:
    if len(data) < 84:
        raise ValueError("STL is too short to contain a binary header")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + count * 50
    if expected > len(data):
        raise ValueError("binary STL is truncated")
    if count == 0:
        raise ValueError("STL contains no triangles")

    triangles: List[Triangle] = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles.append(
            (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
        )
        offset += 50
    return triangles


def read_stl(path: str) -> List[Triangle]:
    """Read either ASCII or binary STL without importing a CAD or mesh package."""
    source = Path(path)
    data = source.read_bytes()
    ascii_triangles = _ascii_triangles(data)
    if ascii_triangles is not None:
        return ascii_triangles
    return _binary_triangles(data)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normal(triangle: Triangle) -> Vec3:
    return _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))


def bbox(tris: Iterable[Triangle]) -> BBox:
    """Return ``(minimum, maximum)`` for all triangle vertices."""
    points = [point for triangle in tris for point in triangle]
    if not points:
        raise ValueError("cannot calculate a bbox for an empty mesh")
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )  # type: ignore[return-value]


def volume(tris: Iterable[Triangle]) -> float:
    """Return solid volume from the signed divergence-theorem tetrahedra.

    The sum is signed while it is accumulated; its absolute value is returned because a
    consistently inward-wound STL is still a valid solid for volume and gate purposes.
    """
    signed = 0.0
    found = False
    for v0, v1, v2 in tris:
        found = True
        signed += _dot(v0, _cross(v1, v2)) / 6.0
    if not found:
        raise ValueError("cannot calculate volume for an empty mesh")
    return abs(signed)


def area(tris: Iterable[Triangle]) -> float:
    """Return the sum of triangle areas in square millimetres."""
    total = 0.0
    for triangle in tris:
        normal = _normal(triangle)
        total += 0.5 * math.sqrt(_dot(normal, normal))
    return total


def _vertex_key(point: Vec3, tol: float) -> Tuple[int, int, int]:
    if tol <= 0.0:
        raise ValueError("vertex quantisation tolerance must be positive")
    return tuple(int(round(coordinate / tol)) for coordinate in point)  # type: ignore[return-value]


def _edge_counts(
    tris: Sequence[Triangle], tol: float
) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int]:
    counts: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    for triangle in tris:
        keys = [_vertex_key(point, tol) for point in triangle]
        for first, second in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
            edge = (first, second) if first <= second else (second, first)
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def is_watertight(tris: Iterable[Triangle], tol: float = 1.0e-6) -> Tuple[bool, int]:
    """Check that every quantised undirected edge is used exactly twice."""
    triangles = list(tris)
    if not triangles:
        return False, 0
    counts = _edge_counts(triangles, tol)
    open_edges = sum(1 for uses in counts.values() if uses != 2)
    return open_edges == 0, open_edges


def count_shells(tris: Iterable[Triangle], tol: float = 1.0e-6) -> int:
    """Count connected triangle components using quantised shared vertices."""
    return len(_shell_roots(list(tris), tol)[1])


def count_bodies(tris: Iterable[Triangle], tol: float = 1.0e-6, touch: float = 0.01) -> int:
    """Count separate solid *bodies*: shells that enclose material, not voids.

    A hollow part is one body with an outer shell and inner shells; a debossed glyph sealed
    inside a plate is a void. Both are wound opposite to the outside of the mesh, so their
    signed volume has the opposite sign to the whole mesh's - only same-sign shells are bodies.
    A tab floating beside a plate is a second same-sign shell: two bodies, however watertight.

    Shells whose bounding boxes touch (within `touch` mm) are counted as one body: a boss
    standing on a plate across a coplanar face can come out of a kernel as two shells that
    share no vertex, yet a slicer fuses them into one print. Only a real gap separates bodies.
    """
    triangles = list(tris)
    roots, members = _shell_roots(triangles, tol)
    signed: Dict[int, float] = {}
    boxes: Dict[int, List[List[float]]] = {}
    for index, triangle in enumerate(triangles):
        v0, v1, v2 = triangle
        root = roots[index]
        signed[root] = signed.get(root, 0.0) + _dot(v0, _cross(v1, v2)) / 6.0
        box = boxes.setdefault(root, [[math.inf] * 3, [-math.inf] * 3])
        for point in triangle:
            for axis in range(3):
                box[0][axis] = min(box[0][axis], point[axis])
                box[1][axis] = max(box[1][axis], point[axis])
    total = sum(signed.values())
    sign = 1.0 if total >= 0.0 else -1.0
    solid = [root for root in members if signed.get(root, 0.0) * sign > _EPSILON]
    parent = {root: root for root in solid}

    def find(root: int) -> int:
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    for i, a in enumerate(solid):
        for b in solid[i + 1:]:
            ba, bb = boxes[a], boxes[b]
            if all(ba[0][k] <= bb[1][k] + touch and bb[0][k] <= ba[1][k] + touch for k in range(3)):
                parent[find(a)] = find(b)
    return len({find(root) for root in solid})


def _shell_roots(triangles: List[Triangle], tol: float) -> Tuple[List[int], set]:
    """Per-triangle connected-component root, and the set of roots."""
    if not triangles:
        return [], set()
    parent = list(range(len(triangles)))
    rank = [0] * len(triangles)
    owner: Dict[Tuple[int, int, int], int] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for index, triangle in enumerate(triangles):
        for point in triangle:
            key = _vertex_key(point, tol)
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                union(index, previous)
    roots = [find(index) for index in range(len(triangles))]
    return roots, set(roots)


def _ray_triangle(origin: Vec3, direction: Vec3, triangle: Triangle) -> Optional[float]:
    """Return a positive ray hit distance using the Moller-Trumbore test."""
    edge1 = _sub(triangle[1], triangle[0])
    edge2 = _sub(triangle[2], triangle[0])
    h = _cross(direction, edge2)
    determinant = _dot(edge1, h)
    if abs(determinant) <= _EPSILON:
        return None
    inverse = 1.0 / determinant
    s = _sub(origin, triangle[0])
    u = inverse * _dot(s, h)
    if u < -_EPSILON or u > 1.0 + _EPSILON:
        return None
    q = _cross(s, edge1)
    v = inverse * _dot(direction, q)
    if v < -_EPSILON or u + v > 1.0 + _EPSILON:
        return None
    distance = inverse * _dot(edge2, q)
    return distance if distance > _EPSILON else None


def first_hit(
    tris: Iterable[Triangle], origin: Vec3, direction: Vec3, max_distance: float
) -> Optional[float]:
    """Distance along `direction` (a unit vector) from `origin` to the nearest triangle within
    `max_distance`, or None when the segment meets no surface.

    Faces parallel to the ray - the bore of a hole drilled along it - are never hits, so a ray
    down the axis of an open hole reports None and one down a blind hole reports its floor.
    """
    lo = [min(origin[i], origin[i] + direction[i] * max_distance) for i in range(3)]
    hi = [max(origin[i], origin[i] + direction[i] * max_distance) for i in range(3)]
    best: Optional[float] = None
    for triangle in tris:
        if any(
            max(p[i] for p in triangle) < lo[i] - 1.0e-9
            or min(p[i] for p in triangle) > hi[i] + 1.0e-9
            for i in range(3)
        ):
            continue
        distance = _ray_triangle(origin, direction, triangle)
        if distance is not None and distance <= max_distance and (best is None or distance < best):
            best = distance
    return best


def _decimate(tris: Sequence[Triangle], maximum: int) -> List[Triangle]:
    if len(tris) <= maximum:
        return list(tris)
    stride = int(math.ceil(float(len(tris)) / maximum))
    return list(tris[::stride])[:maximum]


def min_wall_estimate(tris: Iterable[Triangle], samples: int = 200) -> Optional[float]:
    """Estimate the thinnest wall by inward centroid rays.

    This is deliberately a cheap heuristic, not a thickness solver.  At most 20,000
    triangles are retained by deterministic decimation, and at most ``samples`` triangle
    centroids cast rays.  A hit must have an opposite-facing normal; concave geometry and
    grazing rays can therefore produce no result or a conservative-looking result.
    """
    mesh = _decimate(list(tris), _MAX_ESTIMATE_TRIANGLES)
    if not mesh or samples <= 0:
        return None
    selected_count = min(int(samples), len(mesh))
    if selected_count <= 0:
        return None
    step = float(len(mesh)) / float(selected_count)
    selected = [mesh[min(int(index * step), len(mesh) - 1)] for index in range(selected_count)]
    best: Optional[float] = None

    for triangle in selected:
        normal = _normal(triangle)
        normal_length = math.sqrt(_dot(normal, normal))
        if normal_length <= _EPSILON:
            continue
        outward = (
            normal[0] / normal_length,
            normal[1] / normal_length,
            normal[2] / normal_length,
        )
        direction = (-outward[0], -outward[1], -outward[2])
        origin = (
            (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0 + direction[0] * 1.0e-7,
            (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0 + direction[1] * 1.0e-7,
            (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3.0 + direction[2] * 1.0e-7,
        )
        nearest: Optional[float] = None
        for candidate in mesh:
            candidate_normal = _normal(candidate)
            if _dot(normal, candidate_normal) >= -_EPSILON:
                continue
            distance = _ray_triangle(origin, direction, candidate)
            if distance is not None and (nearest is None or distance < nearest):
                nearest = distance
        if nearest is not None and (best is None or nearest < best):
            best = nearest
    return best


def overhang_fraction(tris: Iterable[Triangle], max_deg: float = 45.0) -> float:
    """Return overhang area / total area for downward faces beyond ``max_deg``.

    The angle is measured from the downward vertical: a horizontal bottom face is 0 degrees
    and is therefore not an overhang at the usual 45-degree limit.  Equality at the limit is
    included with a tiny tolerance so a nominally 45-degree rotated surface is not lost to
    floating-point roundoff.
    """
    triangles = list(tris)
    total = area(triangles)
    if total <= _EPSILON:
        return 0.0
    try:
        limit = float(max_deg)
    except (TypeError, ValueError):
        raise ValueError("max_deg must be a number") from None
    overhang = 0.0
    for triangle in triangles:
        normal = _normal(triangle)
        normal_length = math.sqrt(_dot(normal, normal))
        if normal_length <= _EPSILON or normal[2] >= 0.0:
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, -normal[2] / normal_length))))
        if angle >= limit - 1.0e-9:
            overhang += 0.5 * normal_length
    return overhang / total


__all__ = [
    "BBox",
    "Triangle",
    "Vec3",
    "area",
    "bbox",
    "count_bodies",
    "count_shells",
    "first_hit",
    "is_watertight",
    "min_wall_estimate",
    "overhang_fraction",
    "read_stl",
    "volume",
]
