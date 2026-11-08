"""Dimensioned inspection drawings (SVG) of any PhiMech geometry, generated, never hand-drawn.

Why this exists
---------------
A preview sheet answers "does it look right"; it cannot answer "is the hole 30.48 from the edge".
The person holding the printed part with calipers needs an *engineering drawing*: orthographic
views, overall sizes, every hole called out with its diameter, depth and position, and a table
they can tick off. Until now the only such drawing in the ecosystem was a one-off matplotlib
script for one part. This module makes the drawing a by-product of the geometry itself, so any
CSG AST, ``.scad``, STL or synth build gets one from a single call, and the numbers on the sheet
are *measured from the mesh*, not re-typed from a parameter file.

How it works
------------
1. The source is normalised to an STL (same path as ``preview``) and read with the stdlib STL
   reader. Two things are then measured from the mesh itself, once, and everything else is
   derived from them:

   * **levels** are the planes where the mesh has real *face* area normal to a view's depth
     axis (:func:`face_levels`). A tessellated cone puts a ring of vertices at every facet row;
     those rows belong to the tessellator, not to the part, and are never levels. A machinist
     can only measure to a face.
   * **wall bands** are the connected patches of mesh *parallel* to that axis
     (:func:`wall_bands`) - the wall of a hole, bore, boss, pocket or step. A band's coordinate
     span along the axis *is* the feature's extent, and a round band's fitted radius *is* the
     feature's radius.
2. For each of the three third-angle views (top, front, right) OpenSCAD exports **one** 2D SVG
   containing the silhouette (``projection(cut=false)``) and a *stepped section* midway between
   every pair of adjacent levels, plus one clearly marked *transition* section inside any sloped
   band (``projection(cut=true)``), each translated by a large known offset so a single export
   carries them all. Sectioning is what makes bosses, blind bores and steps appear as lines,
   which a plain silhouette never shows.
3. The SVG paths are parsed back to closed loops (y un-negated), un-offset, and classified:
   circle (vertex radii within 3 % of their mean), rectangle / rounded slot (area against its
   bbox), or a generic profile. A loop that recurs in adjacent slices is one feature; a hole
   that is also in the silhouette is *thru*; a solid loop that is not the silhouette is a boss
   or step.
4. Each feature is then **re-measured against its own wall band**: its size, centre, depth range
   and (for a hole through separated slabs) its individual material segments come from the wall,
   never from which cuts happened to see it. A sloped skirt where the wall ends - a base fillet,
   a chamfer, a countersink - is reported *as a skirt*, so that a section through a boss's base
   fillet can no longer be mistaken for the boss diameter. Coaxial walls of a different radius
   are reported as a counterbore.
5. Sections smaller than an extrusion width, or tiny and shallow, are modelling slivers (the
   crevice where a fillet meets a corner) and go to a note, not the hole table.
6. Features are grouped by kind, size and depth, given IDs (H1.., B1..), and the sheet is laid
   out third-angle with overall dimensions, one positional dimension pair per group (from the
   part's lower-left datum), leader callouts, a level chain on the front view, a hole table in
   model coordinates, and a title block. Scale is chosen from the standard series to fit A3.

Design constraints, and why
---------------------------
* **stdlib only, OpenSCAD only, mm only**: identical to ``preview.py`` and for the same reasons
  (3.9 system python, one renderer, one unit). SVG is written as text, so the sheet needs no
  graphics library; a PNG copy is made only if a rasteriser happens to be installed
  (``rsvg-convert``, macOS ``qlmanage``, a headless Chrome) and is reported as ``png: null``
  otherwise, never faked.
* **Advisory, never a gate.** The drawing helps a human check a part; no verdict is derived
  from it. The one machine-readable output is the feature list, which callers may compare.
* **Silhouette + stepped sections, not hidden-line removal.** Thin lines show every feature
  edge at every level regardless of visibility. That is a deliberate trade: a true HLR needs a
  B-rep kernel, while this works on any mesh and is enough to *inspect* a printed part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from phimech.backends.base import BackendError

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

#: View name -> (OpenSCAD rotation that maps the view onto the XY plane, the model axis that
#: becomes Z' after that rotation and its sign, paper-X model axis, paper-Y model axis).
#: front looks from -y (paper X = x, paper Y = z); right looks from +x (paper X = y, paper Y = z).
VIEWS: Dict[str, Dict[str, Any]] = {
    "top": {"rotate": "", "depth_axis": 2, "depth_sign": 1.0, "ax": 0, "ay": 1},
    "front": {"rotate": "rotate([-90,0,0])", "depth_axis": 1, "depth_sign": -1.0, "ax": 0, "ay": 2},
    "right": {"rotate": "rotate([-90,0,0]) rotate([0,0,-90])", "depth_axis": 0, "depth_sign": 1.0,
              "ax": 1, "ay": 2},
}
VIEW_ORDER: Tuple[str, ...] = ("top", "front", "right")

#: Face planes closer than this fraction of the extent are one level (mesh noise, an EPS
#: over-cut), and at most this many sections are cut per view.
LEVEL_MERGE_FRACTION = 0.004
MAX_SLICES = 14
COORD_DECIMALS = 3

#: A plane earns the name "level" only when at least this much real face area lies in it. One
#: square millimetre is below anything a caliper can reach and above every rounding artefact,
#: and it is what keeps a tessellation ring on a chamfer cone off the sheet.
MIN_LEVEL_FACE_AREA = 1.0
#: At most this many levels; when a part has more, the *largest* faces are kept, because those
#: are the ones a machinist puts a caliper on.
MAX_LEVELS = MAX_SLICES + 1
#: A band between two levels holding at least this many distinct coordinates is a sloped run
#: (fillet, chamfer, draft) and earns one extra, explicitly marked, transition section.
MIN_TRANSITION_COORDS = 3

#: |n[axis]| / |n| below this and a facet is parallel to the axis (a wall); a facet whose
#: coordinate span along the axis is below the coordinate resolution is flat (a face).
FLAT_NORMAL_TOL = 1.0e-3
#: A wall patch smaller than this is mesh debris, not a feature wall.
MIN_BAND_AREA = 0.5
#: A feature and its wall band are the same thing when their axes agree to this (or to this
#: fraction of the smaller of the two) and their sizes to this (or this fraction of the band).
BAND_CENTER_MM = 0.3
BAND_CENTER_FRACTION = 0.10
BAND_SIZE_MM = 0.3
BAND_SIZE_FRACTION = 0.15
#: A skirt reaching further from a wall than this many of the wall's own radii belongs to some
#: other feature; a real fillet or chamfer is a surface of revolution, so its reach must agree
#: to this fraction across this many angular sectors or it is not reported at all.
SKIRT_REACH = 2.0
SKIRT_SECTORS = 8
SKIRT_ROUND_TOLERANCE = 0.10

#: Circle test: relative spread of vertex radii about their mean; OpenSCAD polygons are exact.
CIRCLE_TOLERANCE = 0.03
CIRCLE_AREA_TOLERANCE = 0.06   # a 48-gon has 99.4 % of the circle's area; a rounded rect ~55 %
MIN_CIRCLE_VERTICES = 8
#: A section that changes size slightly between adjacent slices (fillet, chamfer, draft) is the
#: same feature: centres within this many mm (or 5 % of size) and sizes within this (or 8 %).
TAPER_CENTER_MM = 0.15
TAPER_SIZE_MM = 0.6
#: A section this small *and* this shallow is a modelling sliver, not a feature; so is any
#: section narrower than one FDM extrusion (0.4 mm), which is the width of the crevice a fillet
#: leaves where it meets a corner and is below anything that can be made or measured. Slivers
#: go to a note on the sheet, never to the hole table.
SLIVER_AREA_MM2 = 3.0
SLIVER_DEPTH_MM = 1.0
SLIVER_WIDTH_MM = 0.4
#: Loops that match to this many mm across slices are the same feature.
MATCH_TOLERANCE = 0.05
#: Sizes are grouped to this resolution for the table (0.05 mm is below any FDM tolerance).
SIZE_QUANTUM = 0.05

#: The ISO 5455 scale series, largest first.  The old list carried 1.25, 1.5, 2.5, 3, 4,
#: 0.75, 0.4, 0.3 and 0.15, none of which are standard scales: a drawing at "1.25:1" asks
#: every reader to do arithmetic the standard exists to remove.  Only the 1 / 2 / 5 / 10
#: decades are used, which is what a rule, a scale ruler and a machinist all expect.
STANDARD_SCALES: Tuple[float, ...] = (10, 5, 2, 1, 0.5, 0.2, 0.1)
SHEET_MM: Tuple[float, float] = (420.0, 297.0)  # A3 landscape
#: Lines the feature table may take on sheet 1 before the rest continues on the next sheet.
#: The table is also where a position goes that the view could not carry, so it is generous;
#: past it a table would push the views down to a scale nobody can read.  Nothing is dropped:
#: every row past the cap is drawn on a continuation sheet (see ``compose_sheets``).
MAX_TABLE_ROWS = 24


# --------------------------------------------------------------------------------------
# geometry: levels, projections, loops
# --------------------------------------------------------------------------------------

Point = Tuple[float, float]
Loop = List[Point]
Vertex = Tuple[float, float, float]
Triangle = Tuple[Vertex, Vertex, Vertex]

#: depth axis -> the two model axes that view puts on paper, in paper (x, y) order. Derived
#: from ``VIEWS`` so the mesh measurement and the projections can never disagree about which
#: model coordinate is which on the sheet.
_PAPER_AXES: Dict[int, Tuple[int, int]] = {
    int(spec["depth_axis"]): (int(spec["ax"]), int(spec["ay"])) for spec in VIEWS.values()
}


def _read_triangles(stl_path: Path) -> List[Triangle]:
    from phimech.check.mesh import read_stl  # stdlib STL reader shared with the gates

    return [(tri[0], tri[1], tri[2]) for tri in read_stl(str(stl_path))]


def _read_vertices(stl_path: Path) -> List[Vertex]:
    return [v for tri in _read_triangles(stl_path) for v in tri]


def _cross(tri: Triangle) -> Vertex:
    """Twice the triangle's area vector; its direction is the outward normal (STL winding)."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def _norm(v: Vertex) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def cluster_levels(values: Sequence[float], extent: float, max_slices: int = MAX_SLICES) -> List[float]:
    """Distinct coordinate values, merged when closer than ``LEVEL_MERGE_FRACTION`` of the
    extent and thinned to ``max_slices + 1`` evenly chosen values when there are many.

    This is **not** how the sheet's levels are chosen - :func:`face_levels` is, because a
    coordinate is only a level when a real face lies in it. This helper survives as what it
    always was underneath: a way to see how many distinct coordinates a band of the mesh holds,
    which is how :func:`section_cuts` recognises a sloped (tessellated) run from a prismatic
    one. The extremes are always kept."""
    if not values:
        return []
    ordered = sorted(set(round(v, COORD_DECIMALS) for v in values))
    if len(ordered) == 1:
        return ordered
    gap = max(extent * LEVEL_MERGE_FRACTION, 10 ** -COORD_DECIMALS)
    merged: List[float] = [ordered[0]]
    for v in ordered[1:]:
        if v - merged[-1] > gap:
            merged.append(v)
        # else: absorbed into the previous level (keep the lower value)
    if len(merged) <= max_slices + 1:
        return merged
    picks = [merged[round(i * (len(merged) - 1) / max_slices)] for i in range(max_slices + 1)]
    return sorted(set(picks))


def slice_positions(levels: Sequence[float]) -> List[Tuple[float, float, float]]:
    """(cut position, lower level, upper level) at the midpoint of every adjacent level pair."""
    return [((a + b) / 2.0, a, b) for a, b in zip(levels, levels[1:])]


@dataclass(frozen=True)
class Level:
    """A plane normal to a view's depth axis in which the mesh has real face area.

    ``face_area`` is the mm2 of mesh lying in the plane - the evidence that this is a surface
    and not a row of tessellation vertices. ``extreme`` marks a level kept because it bounds the
    part rather than because of its area: the first and last section must sit inside the part
    even for a sphere, whose extremes are points."""

    value: float
    face_area: float
    extreme: bool = False


@dataclass(frozen=True)
class Cut:
    """One section plane: where to cut, which levels bound it, and what it is sampling.

    ``transition`` marks a cut placed inside a *sloped* run (a fillet, chamfer or draft) rather
    than between two faces. A transition section shows what is there, but its cross-section is
    a function of where the cut landed, so nothing it alone reports may become a dimension."""

    position: float
    lower: float
    upper: float
    transition: bool = False


def face_levels(
    triangles: Sequence[Triangle], axis: int, *, min_area: float = MIN_LEVEL_FACE_AREA,
    max_levels: int = MAX_LEVELS,
) -> List[Level]:
    """The planes normal to ``axis`` that carry real face area, plus the two extremes.

    Why not the distinct vertex coordinates (which is what this used to do): a tessellated cone
    or fillet puts a ring of vertices at every facet row, so a 45 degree chamfer 1 mm tall
    reported 44.33 and 44.67 as if they were heights of the part - change the tessellation and
    the "dimensions" change. A coordinate earns the name level only when at least ``min_area``
    mm2 of mesh actually lies in that plane, which is the only place a caliper can rest."""
    if not triangles:
        return []
    coords = [v[axis] for tri in triangles for v in tri]
    low, high = min(coords), max(coords)
    gap = max((high - low) * LEVEL_MERGE_FRACTION, 10 ** -COORD_DECIMALS)
    flat = 10 ** -COORD_DECIMALS
    planes: Dict[float, float] = {}
    for tri in triangles:
        values = [v[axis] for v in tri]
        if max(values) - min(values) > flat:
            continue                                  # sloped: it lies in no single plane
        area = abs(_cross(tri)[axis]) / 2.0           # for a flat facet this is its own area
        if area <= 0.0:
            continue
        key = round(values[0], COORD_DECIMALS)
        planes[key] = planes.get(key, 0.0) + area
    merged: List[List[float]] = []                    # [area-weighted value, summed area]
    for value in sorted(planes):
        area = planes[value]
        if merged and value - merged[-1][0] <= gap:
            total = merged[-1][1] + area
            merged[-1] = [(merged[-1][0] * merged[-1][1] + value * area) / total, total]
        else:
            merged.append([value, area])
    chosen: Dict[float, Level] = {
        round(value, COORD_DECIMALS): Level(round(value, COORD_DECIMALS), area)
        for value, area in merged if area >= min_area
    }
    for bound in (low, high):
        near = [v for v in chosen if abs(v - bound) <= gap]
        if near:
            found = chosen[near[0]]
            chosen[near[0]] = Level(found.value, found.face_area, True)
            continue
        value = round(bound, COORD_DECIMALS)
        area = max((a for v, a in merged if abs(v - bound) <= gap), default=0.0)
        chosen[value] = Level(value, area, True)
    levels = sorted(chosen.values(), key=lambda lv: lv.value)
    if len(levels) > max_levels:
        extremes = [lv for lv in levels if lv.extreme]
        interior = sorted((lv for lv in levels if not lv.extreme),
                          key=lambda lv: -lv.face_area)[:max(max_levels - len(extremes), 0)]
        levels = sorted(extremes + interior, key=lambda lv: lv.value)
    return levels


def section_cuts(
    levels: Sequence[Level], coords: Sequence[float], *, max_cuts: int = MAX_SLICES
) -> List[Cut]:
    """One section midway between every pair of adjacent levels, plus one inside each sloped run.

    Between two faces the mesh may be a cone, a fillet or a draft whose cross-section changes
    continuously. Sampling it once, and saying so, is all a drawing can honestly do; promoting
    its tessellation rings to levels prints the tessellator's settings as if they were the
    part's dimensions."""
    values = [lv.value for lv in levels]
    cuts = [Cut(position, lower, upper)
            for position, lower, upper in slice_positions(values)]
    if not cuts:
        return cuts
    extent = values[-1] - values[0]
    gap = max(extent * LEVEL_MERGE_FRACTION, 10 ** -COORD_DECIMALS)
    ordered = sorted(set(round(c, COORD_DECIMALS) for c in coords))
    extra: List[Tuple[float, Cut]] = []
    for cut in cuts:
        inner = [v for v in ordered if cut.lower + gap < v < cut.upper - gap]
        if len(inner) < MIN_TRANSITION_COORDS:
            continue                                  # prismatic between these two faces
        middle = (inner[0] + inner[-1]) / 2.0
        if abs(middle - cut.position) <= gap:
            continue                                  # the ordinary section already samples it
        extra.append((inner[-1] - inner[0],
                      Cut(middle, cut.lower, cut.upper, transition=True)))
    room = max(max_cuts - len(cuts), 0)
    for _, cut in sorted(extra, key=lambda item: -item[0])[:room]:
        cuts.append(cut)
    return sorted(cuts, key=lambda c: c.position)


@dataclass(frozen=True)
class Skirt:
    """Sloped mesh meeting a wall at one end: a base fillet, a chamfer or a countersink.

    Why it is reported apart from the size: a section through a boss's base fillet is wider than
    the boss, and calling that width the diameter is exactly how a 7.00 mm boss came to be
    printed as 7.29. The slope is real and worth stating - as a slope."""

    at: float          # the wall coordinate the slope meets
    outward: float     # how far past the wall it reaches, radially, mm
    run: float         # how far along the axis it reaches, mm

    @property
    def angle_deg(self) -> float:
        """Half-angle from the axis, degrees: 45 for a 1x1 chamfer, 45 for a quarter-round."""
        return math.degrees(math.atan2(self.outward, self.run)) if self.run else 90.0

    def to_dict(self) -> Dict[str, Any]:
        return {"at": round(self.at, 3), "outward": round(self.outward, 3),
                "run": round(self.run, 3), "angle_deg": round(self.angle_deg, 1)}


@dataclass(frozen=True)
class WallBand:
    """A connected patch of mesh parallel to one axis: the wall of exactly one feature.

    A feature's depth and diameter are properties of its own wall, never of the slice grid that
    happened to see it. ``lo``/``hi`` are the wall's coordinate span along ``axis`` in model
    millimetres - which for a through hole is the span of material it passes through. ``is_hole``
    comes from the facet normals: a wall whose normals point at its own axis has its material
    outside it, which is what a hole is."""

    axis: int
    center: Point
    size: Tuple[float, float]
    radius: float               # fitted mean radius; 0.0 when the wall is not round
    circular: bool
    lo: float
    hi: float
    is_hole: bool
    area: float                 # facet area of the wall, mm2
    skirts: Tuple[Skirt, ...] = ()

    @property
    def depth(self) -> float:
        return self.hi - self.lo


def _union_find(count: int) -> Tuple[Any, Any]:
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


def _vertex_key(v: Vertex) -> Tuple[int, int, int]:
    q = 10 ** COORD_DECIMALS
    return (int(round(v[0] * q)), int(round(v[1] * q)), int(round(v[2] * q)))


def wall_bands(
    triangles: Sequence[Triangle], axis: int, *, min_area: float = MIN_BAND_AREA
) -> List[WallBand]:
    """Every connected wall parallel to ``axis``, measured: centre, size, span and parity.

    Grouping the axis-parallel facets by shared vertices separates one feature's wall from the
    next without any notion of "feature" at all, so the answer cannot be biased by how the part
    was sliced. Each wall is then re-read for the sloped skirt (fillet, chamfer, countersink)
    that meets it at either end, so that skirt can be reported as a skirt instead of being
    mistaken for the feature's size."""
    walls: List[Triangle] = []
    normals: List[Vertex] = []
    sloped: List[Triangle] = []
    flat_span = 10 ** -COORD_DECIMALS
    for tri in triangles:
        normal = _cross(tri)
        length = _norm(normal)
        if length <= 0.0:
            continue
        if abs(normal[axis]) / length <= FLAT_NORMAL_TOL:
            walls.append(tri)
            normals.append(normal)
        elif max(v[axis] for v in tri) - min(v[axis] for v in tri) > flat_span:
            sloped.append(tri)
    find, union = _union_find(len(walls))
    first_at: Dict[Tuple[int, int, int], int] = {}
    for index, tri in enumerate(walls):
        for v in tri:
            key = _vertex_key(v)
            owner = first_at.setdefault(key, index)
            if owner != index:
                union(owner, index)
    patches: Dict[int, List[int]] = {}
    for index in range(len(walls)):
        patches.setdefault(find(index), []).append(index)

    ax, ay = _PAPER_AXES[axis]
    slope_index = _index_by_vertex(sloped)
    bands: List[WallBand] = []
    for members in patches.values():
        points = {(round(v[ax], COORD_DECIMALS), round(v[ay], COORD_DECIMALS))
                  for index in members for v in walls[index]}
        if len(points) < 3:
            continue
        area = sum(_norm(normals[index]) / 2.0 for index in members)
        if area < min_area:
            continue
        coords = [v[axis] for index in members for v in walls[index]]
        lo, hi = min(coords), max(coords)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        size = (max(xs) - min(xs), max(ys) - min(ys))
        circular = _is_round(points, center, size)
        radius = 0.0
        if circular:
            # A closed regular polygon's vertices average to its exact centre, which beats the
            # bbox midpoint by half a facet; the fitted radius is then the real one.
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
            radii = [math.hypot(x - center[0], y - center[1]) for x, y in points]
            radius = sum(radii) / len(radii)
            size = (2.0 * radius, 2.0 * radius)
        inward = 0.0
        for index in members:
            tri, normal = walls[index], normals[index]
            dx = sum(v[ax] for v in tri) / 3.0 - center[0]
            dy = sum(v[ay] for v in tri) / 3.0 - center[1]
            inward += dx * normal[ax] + dy * normal[ay]
        skirts = _wall_skirts(members, walls, axis, center, radius, lo, hi, sloped, slope_index)
        bands.append(WallBand(axis, center, size, radius, circular, lo, hi, inward < 0.0,
                              area, skirts))
    bands.sort(key=lambda b: (b.lo, b.center[0], b.center[1]))
    return bands


class Mesh:
    """The measured mesh: triangles, plus the levels and wall bands derived from them, once.

    All three views interrogate the same mesh, and the band build is a union-find over every
    facet; doing it per view would triple the cost for answers that cannot differ. Levels and
    bands are therefore memoised per axis, and the triangles are read exactly once."""

    def __init__(self, triangles: Sequence[Triangle]) -> None:
        self.triangles: List[Triangle] = list(triangles)
        self.vertices: List[Vertex] = [v for tri in self.triangles for v in tri]
        self._levels: Dict[Tuple[int, int], List[Level]] = {}
        self._bands: Dict[int, List[WallBand]] = {}

    @classmethod
    def read(cls, stl_path: Path) -> "Mesh":
        return cls(_read_triangles(stl_path))

    def levels(self, axis: int, max_levels: int = MAX_LEVELS) -> List[Level]:
        key = (axis, max_levels)
        if key not in self._levels:
            self._levels[key] = face_levels(self.triangles, axis, max_levels=max_levels)
        return self._levels[key]

    def bands(self, axis: int) -> List[WallBand]:
        if axis not in self._bands:
            self._bands[axis] = wall_bands(self.triangles, axis)
        return self._bands[axis]


def _is_round(points: Sequence[Point], center: Point, size: Tuple[float, float]) -> bool:
    """Is this wall a cylinder, or a rounded rectangle whose only vertices are its corner arcs?

    Equal radii alone cannot tell them apart - every vertex of a rounded rectangle sits on a
    corner arc, so they *are* all nearly equidistant from the centre, which is how a 63 x 73 mm
    tray outline came to be measured as a 94.6 mm circle. A cylinder differs in two ways a
    rounded rectangle cannot fake: its bbox is square, and its vertices go all the way round
    instead of clustering into four corners."""
    if len(points) < MIN_CIRCLE_VERTICES or max(size) <= 0.0:
        return False
    if abs(size[0] - size[1]) > CIRCLE_TOLERANCE * max(size):
        return False
    radii = [math.hypot(x - center[0], y - center[1]) for x, y in points]
    mean = sum(radii) / len(radii)
    if mean <= 0.0 or (max(radii) - min(radii)) / mean > CIRCLE_TOLERANCE:
        return False
    sectors = {int((math.atan2(y - center[1], x - center[0]) + math.pi)
                   / (2.0 * math.pi) * SKIRT_SECTORS) % SKIRT_SECTORS for x, y in points}
    return len(sectors) >= SKIRT_SECTORS


def _index_by_vertex(triangles: Sequence[Triangle]) -> Dict[Tuple[int, int, int], List[int]]:
    index: Dict[Tuple[int, int, int], List[int]] = {}
    for i, tri in enumerate(triangles):
        for v in tri:
            index.setdefault(_vertex_key(v), []).append(i)
    return index


def _wall_skirts(
    members: Sequence[int], walls: Sequence[Triangle], axis: int, center: Point, radius: float,
    lo: float, hi: float, sloped: Sequence[Triangle],
    slope_index: Dict[Tuple[int, int, int], List[int]],
) -> Tuple[Skirt, ...]:
    """The sloped mesh continuous with this wall at each end, measured radially and axially.

    Round walls only, because that is where mistaking a skirt for the size does damage (a boss,
    a bore, a countersink). The search floods through *sloped* facets from the ones that share a
    vertex with the wall's end ring and stops of its own accord at the next flat face - so the
    fillet is exactly the mesh between the wall and the face it runs out onto - bounded by
    ``SKIRT_REACH`` times the wall's own radius, because a skirt that reaches further from a
    wall than the wall's own radius belongs to something else.

    The flood is then *checked*, not trusted: a fillet or chamfer is a surface of revolution, so
    the reach that repeats all the way round the axis is the skirt, and a sector that reaches
    further has run into a neighbour's skirt and is that neighbour's, not this wall's. Reporting
    needs a majority of the sectors to agree on one reach; otherwise nothing is reported, rather
    than a fabricated radius."""
    if radius <= 0.0:
        return ()
    ax, ay = _PAPER_AXES[axis]
    tol = 10 ** -COORD_DECIMALS
    bound = SKIRT_REACH * radius
    found: List[Skirt] = []
    for end in (lo, hi):
        ring = {_vertex_key(v) for index in members for v in walls[index]
                if abs(v[axis] - end) <= tol}
        seen = {i for key in ring for i in slope_index.get(key, ())}
        queue = list(seen)
        while queue:
            tri = sloped[queue.pop()]
            for v in tri:
                for neighbour in slope_index.get(_vertex_key(v), ()):
                    if neighbour in seen:
                        continue
                    other = sloped[neighbour]
                    if any(math.hypot(p[ax] - center[0], p[ay] - center[1]) > bound
                           or abs(p[axis] - end) > bound for p in other):
                        continue
                    seen.add(neighbour)
                    queue.append(neighbour)
        if not seen:
            continue
        points = [(math.hypot(v[ax] - center[0], v[ay] - center[1]),
                   math.atan2(v[ay] - center[1], v[ax] - center[0]), v[axis])
                  for i in seen for v in sloped[i]]
        sectors: Dict[int, float] = {}
        for reach, angle, _ in points:
            key = int((angle + math.pi) / (2.0 * math.pi) * SKIRT_SECTORS) % SKIRT_SECTORS
            sectors[key] = max(sectors.get(key, 0.0), reach)
        if len(sectors) < SKIRT_SECTORS:
            continue                      # not all the way round: not this wall's skirt
        ordered = sorted(sectors.values())
        median = ordered[len(ordered) // 2]
        agree = [r for r in ordered if abs(r - median) <= SKIRT_ROUND_TOLERANCE * median]
        if len(agree) * 2 < SKIRT_SECTORS:
            continue                      # merged into a neighbour: no honest reach to report
        reach = sum(agree) / len(agree)
        span = [p[2] for p in points if p[0] <= reach + tol]
        run = (max(span) - end) if end == hi else (end - min(span))
        outward = reach - radius
        if outward > tol and run > tol:
            found.append(Skirt(end, outward, run))
    return tuple(found)


def _fmt(v: float) -> str:
    return "{:.4f}".format(v).rstrip("0").rstrip(".") or "0"


def projection_scad(
    stl_path: Path, view: str, cuts: Sequence[float], offset: float
) -> str:
    """OpenSCAD source exporting the silhouette at x-offset 0 and slice *k* at ``k * offset``.

    ``cuts`` are model coordinates along the view's depth axis. After the view rotation the depth
    axis is Z' = depth_sign * axis, so a cut at model value c needs translate(z = -depth_sign*c)
    for ``projection(cut=true)``, which cuts at Z' = 0."""
    spec = VIEWS[view]
    stl = str(stl_path).replace("\\", "/").replace('"', '\\"')
    body = '{} import("{}", convexity=10);'.format(spec["rotate"], stl)
    lines = ["// PhiMech drawing: {} view silhouette + {} stepped sections".format(view, len(cuts)),
             "projection(cut=false) {}".format(body)]
    for k, c in enumerate(cuts, start=1):
        dz = -spec["depth_sign"] * c
        lines.append(
            "translate([{},0]) projection(cut=true) translate([0,0,{}]) {}".format(
                _fmt(k * offset), _fmt(dz), body
            )
        )
    return "\n".join(lines) + "\n"


_PATH_RE = re.compile(r'<path[^>]*\sd="([^"]*)"', re.S)
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_openscad_svg(text: str) -> List[Loop]:
    """Closed loops from an OpenSCAD 2D SVG export, in model orientation (SVG y negated back).

    OpenSCAD writes one ``<path>`` whose ``d`` is a sequence of ``M x,y L x,y ... z`` subpaths,
    each a simple polygon; holes are just further subpaths. Only M/L/z are emitted, so the parser
    accepts exactly those and refuses anything else rather than guess."""
    loops: List[Loop] = []
    for d in _PATH_RE.findall(text):
        current: Loop = []
        tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d)
        i = 0
        cmd = ""
        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha() and tok not in ("M", "L", "z", "Z"):
                raise ValueError("unsupported SVG path command {!r} from OpenSCAD".format(tok))
            if tok in ("M", "L"):
                cmd = tok
                i += 1
                if tok == "M" and current:
                    loops.append(current)
                    current = []
                continue
            if tok in ("z", "Z"):
                if current:
                    loops.append(current)
                    current = []
                cmd = ""
                i += 1
                continue
            if cmd not in ("M", "L") or i + 1 >= len(tokens):
                raise ValueError("unsupported SVG path syntax from OpenSCAD near {!r}".format(tok))
            x, y = float(tok), float(tokens[i + 1])
            current.append((x, -y))
            i += 2
        if current:
            loops.append(current)
    cleaned: List[Loop] = []
    for loop in loops:
        if len(loop) > 1 and _close(loop[0], loop[-1], 1e-9):
            loop = loop[:-1]
        if len(loop) >= 3:
            cleaned.append(loop)
    return cleaned


def _close(a: Point, b: Point, tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def unpack_offsets(loops: Sequence[Loop], offset: float, count: int) -> List[List[Loop]]:
    """Split loops exported with ``projection_scad`` back into ``count + 1`` groups (silhouette
    first) by which offset bin their centroid x falls in, and shift each back to model x."""
    groups: List[List[Loop]] = [[] for _ in range(count + 1)]
    for loop in loops:
        cx = sum(p[0] for p in loop) / len(loop)
        k = int(math.floor((cx + offset / 2.0) / offset))
        if 0 <= k <= count:
            groups[k].append([(x - k * offset, y) for x, y in loop])
    return groups


# --------------------------------------------------------------------------------------
# geometry: loop analysis
# --------------------------------------------------------------------------------------


def signed_area(loop: Sequence[Point]) -> float:
    a = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def loop_bbox(loop: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in loop]
    ys = [p[1] for p in loop]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_loop(pt: Point, loop: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


@dataclass
class Shape:
    """One closed loop, classified. Sizes mm; ``center`` is the loop centroid (bbox centre for
    non-circles). ``is_hole`` is nesting parity within its own slice."""

    kind: str                    # "circle" | "rect" | "slot" | "profile"
    center: Point
    size: Tuple[float, float]    # (diameter, diameter) for a circle; (w, h) otherwise
    corner_r: float
    area: float
    bbox: Tuple[float, float, float, float]
    is_hole: bool
    loop: Loop = field(repr=False)

    @property
    def label(self) -> str:
        return _shape_label(self.kind, self.size, self.corner_r)

    def matches(self, other: "Shape", tol: float = MATCH_TOLERANCE) -> bool:
        return (
            self.kind == other.kind
            and _close(self.center, other.center, tol)
            and abs(self.size[0] - other.size[0]) <= tol
            and abs(self.size[1] - other.size[1]) <= tol
        )

    def similar(self, other: "Shape") -> bool:
        """Same feature seen through a fillet, chamfer or draft: near-coincident centre, size
        within a few percent, hole/solid alike; a circle may degrade to a coarse profile."""
        if self.is_hole != other.is_hole:
            return False
        round_kinds = {"circle", "profile"}
        square_kinds = {"rect", "slot", "profile"}
        if not ((self.kind in round_kinds and other.kind in round_kinds)
                or (self.kind in square_kinds and other.kind in square_kinds)):
            return False
        big = max(self.size + other.size)
        ctol = max(TAPER_CENTER_MM, 0.05 * big)
        stol = max(TAPER_SIZE_MM, 0.08 * big)
        return (_close(self.center, other.center, ctol)
                and abs(self.size[0] - other.size[0]) <= stol
                and abs(self.size[1] - other.size[1]) <= stol)


#: Printed dimensions are rounded to two decimals; a third is used when two would move the
#: value further than this.  Every position on the sheet is matched back to the measurement it
#: came from - ``drawing_checks.POSITION_TOLERANCE_MM`` is 0.005 mm and ``TRACE_TOLERANCE_MM``
#: 0.01 mm - so a value that has drifted further than this is no longer traceable to anything.
_DIM_ROUND_MM = 0.004


def _num(v: float) -> str:
    return "{:.2f}".format(v).rstrip("0").rstrip(".") or "0"


def _dimnum(value: float) -> str:
    """A printed dimension: two decimals, or three when two would lose the measurement.

    Two decimals is the drafting default and is what nearly every value needs.  A value whose
    2 dp form has drifted by more than ``_DIM_ROUND_MM`` is printed to three rather than
    printed as a number that traces back to no measurement at all."""
    if abs(value) < 5e-4:
        return "0"
    text = "{:.2f}".format(value).rstrip("0").rstrip(".")
    if text not in ("", "-0", "0") and abs(float(text) - value) <= _DIM_ROUND_MM:
        return text
    return "{:.3f}".format(value).rstrip("0").rstrip(".") or "0"


def _shape_label(kind: str, size: Tuple[float, float], corner_r: float) -> str:
    """How a section of this kind and size is written, for a single shape or for a whole group.

    One function, so a group's row and its members' callouts can never print different text for
    the same thing.  ``_dimnum`` and not a coarser rounding, because this text is printed as a
    dimension and a dimension has to trace to the measurement behind it: quantising a corner
    radius to 0.1 mm printed "R2" for a wall the result recorded as 2.03."""
    if kind == "circle":
        return "⌀{}".format(_dimnum(size[0]))
    w, h = size
    if kind == "slot":
        return "{} x {} R{}".format(_dimnum(w), _dimnum(h), _dimnum(corner_r))
    return "{} x {}".format(_dimnum(w), _dimnum(h))


def classify_loop(loop: Loop, is_hole: bool) -> Shape:
    """Circle, axis-aligned rectangle, rounded slot, or generic profile."""
    x0, y0, x1, y1 = loop_bbox(loop)
    w, h = x1 - x0, y1 - y0
    area = abs(signed_area(loop))
    n = len(loop)
    cx = sum(p[0] for p in loop) / n
    cy = sum(p[1] for p in loop) / n
    if n >= MIN_CIRCLE_VERTICES:
        radii = [math.hypot(x - cx, y - cy) for x, y in loop]
        mean = sum(radii) / n
        circle_area = math.pi * mean * mean
        if (mean > 0 and (max(radii) - min(radii)) / mean <= CIRCLE_TOLERANCE
                and abs(area - circle_area) <= CIRCLE_AREA_TOLERANCE * circle_area):
            # Vertices lie ON the circle, so the true diameter is 2*mean, not the bbox. The
            # area check is what tells a circle from a rounded rectangle whose only vertices
            # are on its corner arcs (all at nearly the same radius from the centre).
            d = 2.0 * mean
            return Shape("circle", (cx, cy), (d, d), 0.0, area, (x0, y0, x1, y1), is_hole, loop)
    bbox_area = w * h
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if bbox_area > 0:
        fill = area / bbox_area
        if fill >= 0.995 and n <= 8:
            return Shape("rect", centre, (w, h), 0.0, area, (x0, y0, x1, y1), is_hole, loop)
        if fill >= 0.80 and n > 8:
            # A rounded rectangle loses (4 - pi) r^2 of its bbox; invert for r.
            r = math.sqrt(max(bbox_area - area, 0.0) / (4.0 - math.pi))
            if r <= min(w, h) / 2.0 + MATCH_TOLERANCE:
                return Shape("slot", centre, (w, h), r, area, (x0, y0, x1, y1), is_hole, loop)
    return Shape("profile", centre, (w, h), 0.0, area, (x0, y0, x1, y1), is_hole, loop)


def classify_slice(loops: Sequence[Loop]) -> List[Shape]:
    """Classify every loop of one 2D slice, with hole/solid decided by nesting parity."""
    shapes: List[Shape] = []
    for i, loop in enumerate(loops):
        depth = 0
        probe = loop[0]
        for j, other in enumerate(loops):
            if i != j and point_in_loop(probe, other):
                depth += 1
        shapes.append(classify_loop(loop, is_hole=(depth % 2 == 1)))
    return shapes


# --------------------------------------------------------------------------------------
# views: run OpenSCAD, aggregate slices into features
# --------------------------------------------------------------------------------------


@dataclass
class Feature:
    """A classified shape that persists across one or more adjacent slices of a view.

    ``depth_from`` / ``depth_to`` are model coordinates along the view's depth axis. Once
    :func:`measure_features` has run they are the span of the feature's **own wall**, and
    ``segments`` lists the material the wall actually passes through - which for a hole through
    two separated slabs is two ranges, not one. Until then they are only the levels the feature
    was seen between, and ``measured`` says so. ``thru`` means it is also in the silhouette (a
    hole all the way through). ``levels`` records the slice cuts it appeared at, for the record."""

    shape: Shape
    view: str
    depth_from: float
    depth_to: float
    thru: bool
    levels: List[float] = field(default_factory=list)
    group_id: str = ""
    #: (shape, depth span, transition) per slice it was seen in.
    sections: List[Tuple[Shape, float, bool]] = field(default_factory=list)
    #: True once the feature's own wall was found and used for its size and extent.
    measured: bool = False
    #: (from, to) per continuous run of material the wall bounds, low to high.
    segments: List[Tuple[float, float]] = field(default_factory=list)
    #: The sloped mesh at either end of the wall: a base fillet, a chamfer, a countersink.
    skirts: Tuple[Skirt, ...] = ()
    #: A coaxial wall of a larger radius opening one end of a hole.
    counterbore: Optional[Dict[str, Any]] = None

    def settle(self) -> None:
        """Adopt the section that best represents the feature: a real one over a transition
        (whose cross-section depends on where the cut landed), the widest span among equals."""
        if not self.sections:
            return
        self.shape = max(self.sections, key=lambda item: (not item[2], item[1]))[0]

    @property
    def depth(self) -> float:
        return self.depth_to - self.depth_from

    @property
    def transition_only(self) -> bool:
        """Seen only where the mesh slopes, so nothing about its size is a dimension."""
        return bool(self.sections) and all(item[2] for item in self.sections)

    def to_dict(self) -> Dict[str, Any]:
        s = self.shape
        record: Dict[str, Any] = {
            "id": self.group_id,
            "view": self.view,
            "kind": s.kind,
            "hole": s.is_hole,
            "center": [round(s.center[0], 3), round(s.center[1], 3)],
            "size": [round(s.size[0], 3), round(s.size[1], 3)],
            "corner_r": round(s.corner_r, 3),
            "depth_from": round(self.depth_from, 3),
            "depth_to": round(self.depth_to, 3),
            "thru": self.thru,
            "measured": self.measured,
            "segments": [[round(a, 3), round(b, 3)] for a, b in self.segments],
        }
        if self.skirts:
            record["skirts"] = [s.to_dict() for s in self.skirts]
        if self.counterbore:
            record["counterbore"] = self.counterbore
        return record


@dataclass
class ViewData:
    """Everything one orthographic view contributes to the sheet."""

    name: str
    silhouette: List[Loop]
    slices: List[Tuple[float, float, float, List[Loop]]]  # (cut, lower, upper, loops)
    features: List[Feature]
    bbox: Tuple[float, float, float, float]
    levels: List[float]
    seconds: float = 0.0
    #: The cuts behind ``slices``, in the same order, saying which sampled a sloped run.
    cuts: List[Cut] = field(default_factory=list)
    #: Sections dismissed as modelling slivers, as ``{"kind", "size", "area", "width", "at"}``.
    slivers: List[Dict[str, Any]] = field(default_factory=list)
    #: Shallow relief clusters classified as markings, not as holes (see :func:`cluster_markings`).
    markings: List["Marking"] = field(default_factory=list)

    @property
    def size(self) -> Tuple[float, float]:
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]


def _openscad_binary() -> str:
    from phimech.synth import preview

    return preview._openscad_binary()


def _run_openscad(command: Sequence[str], timeout: float, what: str) -> str:
    from phimech.synth import preview

    return preview._run_openscad(command, timeout, what)


def export_view(
    stl_path: Path, view: str, cuts: Sequence[float], workdir: Path, extent: float,
    timeout: float
) -> Tuple[List[Loop], List[List[Loop]], float]:
    """Run OpenSCAD once for a view; return (silhouette loops, per-slice loops, seconds)."""
    started = time.monotonic()
    offset = max(10.0 * extent, 100.0)
    scad = workdir / "{}.scad".format(view)
    svg = workdir / "{}.svg".format(view)
    scad.write_text(projection_scad(stl_path, view, cuts, offset), encoding="utf-8")
    command = [_openscad_binary(), "-o", str(svg)]
    from phimech.backends.openscad import OpenSCADBackend

    probe = getattr(OpenSCADBackend(), "_supports_manifold", None)
    if callable(probe) and probe():
        command.extend(["--backend", "manifold"])
    command.append(str(scad))
    log = _run_openscad(command, timeout, "the {} view projections".format(view))
    if not svg.is_file() or svg.stat().st_size == 0:
        raise BackendError("OpenSCAD wrote no SVG for the {} view".format(view), stderr=log)
    groups = unpack_offsets(parse_openscad_svg(svg.read_text(encoding="utf-8")), offset, len(cuts))
    return groups[0], groups[1:], time.monotonic() - started


def _same_loop_set(shape: Shape, shapes: Sequence[Shape]) -> bool:
    return any(shape.matches(other) for other in shapes)


def _is_outline(shape: Shape, outline: Sequence[Shape]) -> bool:
    """The part's own outer wall at some z: equal to, or a filleted version of, the silhouette."""
    return any(shape.matches(o) or shape.similar(o) for o in outline)


def aggregate_features(
    view: str,
    silhouette: List[Loop],
    slices: Sequence[Tuple[float, float, float, List[Loop]]],
    cuts: Optional[Sequence[Cut]] = None,
) -> List[Feature]:
    """Merge the classified loops of consecutive slices into features with a depth range.

    Outer silhouette loops are the part outline and are not features. A hole present in the
    silhouette is *thru*; every other loop is a feature bounded by the levels it was seen
    between - a provisional extent that :func:`measure_features` then replaces with the
    feature's own wall. Solid loops whose outline equals the silhouette's are skipped."""
    sil_shapes = classify_slice(silhouette)
    outline = [s for s in sil_shapes if not s.is_hole]
    transitions = [c.transition for c in cuts] if cuts is not None else [False] * len(slices)
    features: List[Feature] = []

    def find(shape: Shape) -> Optional[Feature]:
        for f in features:
            if f.shape.matches(shape):
                return f
        for f in features:
            if f.shape.similar(shape):
                return f
        return None

    # Thru holes come from the silhouette; their extent is measured from their wall later.
    for s in sil_shapes:
        if s.is_hole:
            features.append(Feature(s, view, math.inf, -math.inf, True))

    for (cut, lower, upper, loops), transition in zip(slices, transitions):
        for s in classify_slice(loops):
            if not s.is_hole and _is_outline(s, outline):
                continue
            existing = find(s)
            if existing is None:
                existing = Feature(s, view, lower, upper, False)
                features.append(existing)
            existing.depth_from = min(existing.depth_from, lower)
            existing.depth_to = max(existing.depth_to, upper)
            existing.levels.append(cut)
            existing.sections.append((s, upper - lower, transition))
    for f in features:
        f.settle()
    return features


def _finish_depths(features: List[Feature], levels: Sequence[float]) -> None:
    """Fall back to the level span for anything whose own wall could not be found.

    A feature only reaches here unmeasured when its wall is fused to the part outline - a step
    standing on a plate shares the plate's outer face - so the levels that bound it are still
    real faces, which is the whole point of measuring levels from faces. It is recorded as
    ``measured: false`` either way."""
    if not levels:
        return
    lo, hi = min(levels), max(levels)
    for f in features:
        if f.measured:
            continue
        if f.thru or f.depth_from == math.inf:
            f.depth_from, f.depth_to = lo, hi
        f.segments = [(f.depth_from, f.depth_to)]


def _band_candidates(bands: Sequence[WallBand], shape: Shape) -> List[WallBand]:
    """Walls coaxial with ``shape`` and of the same parity: where its own wall must be."""
    found: List[WallBand] = []
    for band in bands:
        if band.is_hole != shape.is_hole:
            continue
        reach = max(BAND_CENTER_MM,
                    BAND_CENTER_FRACTION * min(max(shape.size), max(band.size)))
        if math.hypot(band.center[0] - shape.center[0],
                      band.center[1] - shape.center[1]) <= reach:
            found.append(band)
    return found


def measure_features(features: Sequence[Feature], bands: Sequence[WallBand]) -> None:
    """Re-measure every feature against its own wall: size, centre, extent and segments.

    This is the whole point of the module's claim that the numbers are measured. The sections
    are what *found* a feature and what classifies it (circle / slot / profile); they are a poor
    way to *size* it, because a cut through a boss's base fillet is wider than the boss and a
    cut is only ever somewhere between two levels. The wall is neither: it is the surface a
    caliper touches, it starts and stops where the material does, and for a hole through two
    separated slabs it says so in ``segments`` instead of drawing a prism through the gap."""
    for f in features:
        sections = [s for s, _, _ in f.sections] or [f.shape]
        candidates = _band_candidates(bands, f.shape)
        best: Optional[WallBand] = None
        best_delta = math.inf
        for band in candidates:
            delta = min(max(abs(band.size[0] - s.size[0]), abs(band.size[1] - s.size[1]))
                        for s in sections)
            if delta < best_delta:
                best, best_delta = band, delta
        if best is None:
            continue
        limit = max(BAND_SIZE_MM, BAND_SIZE_FRACTION * max(best.size))
        if best_delta > limit:
            continue
        stack = [b for b in candidates
                 if abs(b.size[0] - best.size[0]) <= limit
                 and abs(b.size[1] - best.size[1]) <= limit]
        f.segments = sorted((b.lo, b.hi) for b in stack)
        f.depth_from = min(lo for lo, _ in f.segments)
        f.depth_to = max(hi for _, hi in f.segments)
        f.shape = replace(f.shape, center=best.center, size=best.size)
        f.skirts = best.skirts
        f.measured = True
        f.counterbore = _counterbore(f, best, candidates, limit)


def _counterbore(
    f: Feature, band: WallBand, candidates: Sequence[WallBand], limit: float
) -> Optional[Dict[str, Any]]:
    """A coaxial wall of a larger radius opening one end of a hole, measured, not assumed.

    A counterbore is exactly that: a second, wider, cylindrical wall on the same axis that meets
    the bore's mouth. Reporting it needs nothing but the two walls; a countersink instead leaves
    a *sloped* skirt at the mouth, which is already carried by ``skirts``."""
    if not f.shape.is_hole:
        return None
    gap = max(BAND_SIZE_MM, 10 ** -COORD_DECIMALS)
    wider = [b for b in candidates
             if max(b.size) > max(band.size) + limit
             and (abs(b.lo - band.hi) <= gap or abs(b.hi - band.lo) <= gap)]
    if not wider:
        return None
    step = max(wider, key=lambda b: b.area)
    record: Dict[str, Any] = {
        "size": [round(step.size[0], 3), round(step.size[1], 3)],
        "depth": round(step.depth, 3),
        "from": round(step.lo, 3),
        "to": round(step.hi, 3),
    }
    if step.circular:
        record["diameter"] = round(2.0 * step.radius, 3)
    return record


def _section_width(shape: Shape) -> float:
    """Twice area over perimeter: the width of a thin strip, the radius of a circle.

    It is the one number that separates a real small feature from a sliver of void: a 1.6 mm
    hole reads 0.8, the 0.2 mm crevice a fillet leaves against a corner reads 0.2."""
    loop = shape.loop
    n = len(loop)
    perimeter = sum(math.hypot(loop[(i + 1) % n][0] - loop[i][0],
                               loop[(i + 1) % n][1] - loop[i][1]) for i in range(n))
    return 2.0 * shape.area / perimeter if perimeter > 0.0 else 0.0


def split_slivers(features: Sequence[Feature]) -> Tuple[List[Feature], List[Dict[str, Any]]]:
    """(features worth a table row, slivers worth only a note).

    A sliver is a section of the model, not a feature of the part: the crevice where a boss's
    fillet runs into a filleted corner, a fillet run-out, a tessellation lune. Two independent
    criteria catch them, because they fail in two different ways - too small *and* too shallow
    to be anything, or narrower than the 0.4 mm an FDM nozzle lays down and a drill could make.
    Neither is a size threshold on real features: a 1.6 mm hole 0.5 mm deep survives both."""
    kept: List[Feature] = []
    slivers: List[Dict[str, Any]] = []
    for f in features:
        width = _section_width(f.shape)
        tiny = f.shape.area < SLIVER_AREA_MM2 and f.depth < SLIVER_DEPTH_MM
        thin = 0.0 < width < SLIVER_WIDTH_MM
        if not (tiny or thin):
            kept.append(f)
            continue
        slivers.append({
            "view": f.view,
            "kind": ("hole " if f.shape.is_hole else "solid ") + f.shape.kind,
            "at": [round(f.shape.center[0], 3), round(f.shape.center[1], 3)],
            "size": [round(f.shape.size[0], 3), round(f.shape.size[1], 3)],
            "area_mm2": round(f.shape.area, 4),
            "width_mm": round(width, 4),
            "depth_mm": round(f.depth, 3),
            "why": "narrower than {} mm".format(_num(SLIVER_WIDTH_MM)) if thin else
                   "under {} mm2 and under {} mm deep".format(_num(SLIVER_AREA_MM2),
                                                              _num(SLIVER_DEPTH_MM)),
        })
    return kept, slivers



@dataclass(frozen=True)
class Marking:
    """A cluster of shallow, mixed, co-planar loops: a moulded-in or engraved marking.

    Why a marking is not a hole-table row: the six loops of a debossed "MINI4K" are genuine
    0.5 mm pockets, so the sliver filter is right to leave them alone - but they are letters.
    Printing six profiled-opening rows with six positions spends the sheet's whole dimension
    budget on glyphs while the posts and hooks that hold the payload get nothing, and it tells
    a machinist to inspect a serif.  This is therefore a *classification*, made from what was
    measured - depth, z band, span, variety of shape - and the marking's **content is never
    guessed**: with no text block from the source the row says exactly that."""

    ident: str
    view: str
    loops: int
    depth: float
    z_from: float
    z_to: float
    bbox: Tuple[float, float, float, float]
    #: All four come from the source's own text block or none of them do.
    label: Optional[str] = None
    text_size: Optional[float] = None
    embossed: Optional[bool] = None
    source_depth: Optional[float] = None

    @property
    def declared(self) -> bool:
        return bool(self.label) and self.text_size is not None and self.source_depth is not None

    def row(self) -> str:
        """The one line this marking contributes to the table, in one of exactly two forms."""
        if self.declared:
            return "{}  MARKING '{}' {} TEXT {} {}".format(
                self.ident, self.label, _dimnum(float(self.text_size or 0.0)),
                "EMBOSSED" if self.embossed else "DEBOSSED",
                _dimnum(float(self.source_depth or 0.0)))
        return ("{}  SHALLOW RELIEF REGION, {} LOOPS, {} DEEP, GEOMETRY PER MODEL "
                "\u2014 NOT A HOLE".format(self.ident, self.loops, _dimnum(self.depth)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.ident,
            "view": self.view,
            "loops": self.loops,
            "depth": round(self.depth, 3),
            "z_from": round(self.z_from, 3),
            "z_to": round(self.z_to, 3),
            "bbox": [round(v, 3) for v in self.bbox],
            "declared": self.declared,
            "label": self.label,
            "text_size": self.text_size,
            "embossed": self.embossed,
            "source_depth": self.source_depth,
            "row": self.row(),
        }


def _cluster_span(boxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float]:
    return (max(b[2] for b in boxes) - min(b[0] for b in boxes),
            max(b[3] for b in boxes) - min(b[1] for b in boxes))


def cluster_markings(features: Sequence[Feature],
                     view: str = "") -> Tuple[List[Feature], List[Marking]]:
    """(features that are still features, markings).

    Three measured conditions have to hold together, and each one is there to stop a real
    feature being swallowed: the loops are no deeper than ``MARKING_MAX_DEPTH`` (a marking is
    relief, not a pocket), at least ``MARKING_MIN_LOOPS`` of them share one z band inside a
    ``MARKING_MAX_SPAN`` box (letters of one word), and they are of at least
    ``MARKING_MIN_KINDS`` different shapes.  That last one is the important one: a row of five
    identical shallow pockets is a feature pattern somebody has to make and inspect, and it
    stays in the table with its dimensions."""
    shallow: List[Feature] = []
    rest: List[Feature] = []
    for f in features:
        if f.shape.is_hole and not f.thru and 0.0 < f.depth <= MARKING_MAX_DEPTH:
            shallow.append(f)
        else:
            rest.append(f)
    bands: Dict[Tuple[int, int], List[Feature]] = {}
    for f in shallow:
        key = (int(round(f.depth_from / MARKING_Z_TOLERANCE)),
               int(round(f.depth_to / MARKING_Z_TOLERANCE)))
        bands.setdefault(key, []).append(f)
    markings: List[Marking] = []
    for key in sorted(bands):
        members = sorted(bands[key], key=lambda f: (f.shape.center[0], f.shape.center[1]))
        cluster: List[Feature] = []
        for f in members + [None]:                            # type: ignore[list-item]
            if f is not None:
                trial = cluster + [f]
                span = _cluster_span([g.shape.bbox for g in trial])
                if max(span) <= MARKING_MAX_SPAN:
                    cluster = trial
                    continue
            if cluster:
                kinds = {(g.shape.kind, _q(g.shape.size[0]), _q(g.shape.size[1]))
                         for g in cluster}
                if len(cluster) >= MARKING_MIN_LOOPS and len(kinds) >= MARKING_MIN_KINDS:
                    boxes = [g.shape.bbox for g in cluster]
                    markings.append(Marking(
                        "", view or cluster[0].view, len(cluster),
                        max(g.depth for g in cluster),
                        min(g.depth_from for g in cluster), max(g.depth_to for g in cluster),
                        (min(b[0] for b in boxes), min(b[1] for b in boxes),
                         max(b[2] for b in boxes), max(b[3] for b in boxes))))
                else:
                    rest.extend(cluster)
            cluster = [f] if f is not None else []
    return rest, markings


def apply_marking_text(markings: Sequence[Marking],
                       text_block: Optional[Mapping[str, Any]]) -> List[Marking]:
    """Attach the source's own text block to the one marking it can only be.

    Only when there is exactly one marking region and one declared text block, and the
    declared depth agrees with the measured one: two regions and one label is an assignment
    this module would be inventing, and a label whose depth does not match the geometry
    describes some other feature.  Everything else keeps the generic row."""
    markings = list(markings)
    if len(markings) != 1 or not isinstance(text_block, Mapping):
        return markings
    label = text_block.get("label")
    size = text_block.get("size")
    depth = text_block.get("depth")
    if not label or size is None or depth is None:
        return markings
    try:
        size_mm, depth_mm = float(size), float(depth)
    except (TypeError, ValueError):
        return markings
    if abs(depth_mm - markings[0].depth) > 0.3:
        return markings
    return [replace(markings[0], label=str(label), text_size=size_mm,
                    embossed=bool(text_block.get("emboss")), source_depth=depth_mm)]


def extract_view(
    stl_path: Path, view: str, mesh: "Mesh", workdir: Path, timeout: float,
    max_slices: int = MAX_SLICES, sections: bool = True
) -> ViewData:
    """One view's silhouette, and (``sections=True``) its stepped sections and features.

    Sections are cut for the top view only: a part is built up along z, so its faces along z are
    few and meaningful (plate top, boss top), whereas along x or y a tray floor is one face and
    everything else is a wall. Side views get their feature edges projected from the top view's
    prisms instead (:func:`feature_extent_boxes`). Every view still measures its own wall bands,
    because a hole seen through the silhouette of a side view is measured the same way."""
    spec = VIEWS[view]
    axis = int(spec["depth_axis"])
    vertices = mesh.vertices
    extent_all = max(max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3))
    levels = mesh.levels(axis, max_slices + 1)
    cuts = section_cuts(levels, [v[axis] for v in vertices], max_cuts=max_slices) \
        if sections else []
    silhouette, per_slice, seconds = export_view(
        stl_path, view, [c.position for c in cuts], workdir, extent_all, timeout
    )
    slices = [(c.position, c.lower, c.upper, loops) for c, loops in zip(cuts, per_slice)]
    features = aggregate_features(view, silhouette, slices, cuts)
    measure_features(features, mesh.bands(axis))
    values = [lv.value for lv in levels]
    _finish_depths(features, values)
    features, slivers = split_slivers(features)
    features, markings = cluster_markings(features, view)
    if silhouette:
        xs = [p[0] for loop in silhouette for p in loop]
        ys = [p[1] for loop in silhouette for p in loop]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        ax, ay = int(spec["ax"]), int(spec["ay"])
        bbox = (min(v[ax] for v in vertices), min(v[ay] for v in vertices),
                max(v[ax] for v in vertices), max(v[ay] for v in vertices))
    return ViewData(view, silhouette, slices, features, bbox, values, seconds, list(cuts),
                    slivers, markings)


# --------------------------------------------------------------------------------------
# grouping: the hole table
# --------------------------------------------------------------------------------------


@dataclass
class Group:
    """Identical features (same kind, size, depth, thru-ness) sharing one table row and ID.

    Every number here is the group's **representative** - the mean over its members - and it is
    the only value printed for the group, anywhere. Grouping quantises to 0.05 mm, so a 7.27 and
    a 7.31 fall in one bucket; printing ``members[0]``'s own size then made the row read 7.27 or
    7.31 depending on which happened to be found first, and the sheet disagreed with itself
    between the callout and the table."""

    group_id: str
    kind: str
    is_hole: bool
    thru: bool
    size: Tuple[float, float]
    corner_r: float
    depth: float
    members: List[Feature]
    depth_from: float = 0.0
    depth_to: float = 0.0

    @property
    def label(self) -> str:
        return _shape_label(self.kind, self.size, self.corner_r)

    @property
    def view(self) -> str:
        return self.members[0].view

    @property
    def measured(self) -> bool:
        return all(m.measured for m in self.members)

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "id": self.group_id,
            "qty": len(self.members),
            "kind": self.kind,
            "hole": self.is_hole,
            "thru": self.thru,
            "size": [round(self.size[0], 3), round(self.size[1], 3)],
            "corner_r": round(self.corner_r, 3),
            "depth": round(self.depth, 3),
            "view": self.view,
            "positions": [
                [round(m.shape.center[0], 3), round(m.shape.center[1], 3)] for m in self.members
            ],
            "depth_from": round(self.depth_from, 3),
            "depth_to": round(self.depth_to, 3),
            "measured": self.measured,
            "segments": [[round(a, 3), round(b, 3)] for a, b in self.members[0].segments],
        }
        first = self.members[0]
        if first.skirts:
            # the skirt's own record says how far it reaches and at what angle; the *hole*
            # is what turns that into a diameter and a span, so it is stated here, where the
            # bore radius is known.  Both are measured numbers, and a reader of the result -
            # or of the traceability gate - needs the diameter, not just the reach.
            record["skirts"] = []
            for skirt in first.skirts:
                entry = skirt.to_dict()
                high = abs(skirt.at - self.depth_to) <= abs(skirt.at - self.depth_from)
                span = (skirt.at, skirt.at + skirt.run) if high else \
                    (skirt.at - skirt.run, skirt.at)
                entry["diameter"] = round(self.size[0] + 2.0 * skirt.outward, 3)
                entry["reach"] = round(skirt.outward, 3)
                entry["from"] = round(min(span), 3)
                entry["to"] = round(max(span), 3)
                record["skirts"].append(entry)
        if first.counterbore:
            record["counterbore"] = first.counterbore
        return record


def _q(v: float, quantum: float = SIZE_QUANTUM) -> float:
    return round(round(v / quantum) * quantum, 6)


#: A modifier's diameter and z range must match the parent's own record of it to this, in mm
#: or as this fraction of the diameter.  Tight on purpose: the fold removes a row from the
#: sheet, and removing the wrong one hides a hole.
MODIFIER_SIZE_MM = 0.3
MODIFIER_SIZE_FRACTION = 0.05
MODIFIER_Z_MM = 0.2


def group_features(features: Sequence[Feature]) -> List[Group]:
    """Group the *top view's* features for the table (the other views only draw lines).

    Holes are numbered H1.. by size descending; solids (bosses, steps, windows seen as solid
    islands) B1.. likewise. Sizes are quantised to ``SIZE_QUANTUM`` so tessellation noise does
    not split a group.  Counterbores and countersinks are then folded into the hole they
    belong to (:func:`fold_modifiers`) *before* the numbers are handed out, so the holes are
    H1..Hn with no gap where a folded one used to be and no id used twice."""
    buckets: Dict[Tuple[Any, ...], List[Feature]] = {}
    for f in features:
        s = f.shape
        key = (s.is_hole, s.kind, _q(s.size[0]), _q(s.size[1]), _q(s.corner_r),
               f.thru, _q(f.depth_from, 0.1), _q(f.depth_to, 0.1))
        buckets.setdefault(key, []).append(f)
    holes = [(k, v) for k, v in buckets.items() if k[0]]
    solids = [(k, v) for k, v in buckets.items() if not k[0]]

    def order(item: Tuple[Tuple[Any, ...], List[Feature]]) -> Tuple[float, float, float]:
        k = item[0]
        return (-(k[2] * k[3]), k[6], k[7])

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    ordered: List[Tuple[str, Group]] = []
    for prefix, items in (("H", sorted(holes, key=order)), ("B", sorted(solids, key=order))):
        for k, members in items:
            members.sort(key=lambda m: (m.shape.center[1], m.shape.center[0]))
            # the representative is the mean of what was measured, not whichever member the
            # bucket happened to see first
            size = (mean([m.shape.size[0] for m in members]),
                    mean([m.shape.size[1] for m in members]))
            depth_from = mean([m.depth_from for m in members])
            depth_to = mean([m.depth_to for m in members])
            ordered.append((prefix, Group("", k[1], k[0], k[5], size,
                                          mean([m.shape.corner_r for m in members]),
                                          round(depth_to - depth_from, 6), members,
                                          depth_from, depth_to)))
    folded = fold_modifiers([group for _, group in ordered])
    consumed = {id(modifier) for modifier, _ in folded}
    ordered = [entry for entry in ordered if id(entry[1]) not in consumed]
    counts = {"H": 0, "B": 0}
    for prefix, group in ordered:
        counts[prefix] += 1
        group.group_id = "{}{}".format(prefix, counts[prefix])
        for member in group.members:
            member.group_id = group.group_id
    return [group for _, group in ordered]


def _same_positions(inner: "Group", outer: "Group") -> bool:
    """Is every member of ``inner`` coaxial with a member of ``outer``?"""
    if not inner.members or len(inner.members) > len(outer.members):
        return False
    centres = [m.shape.center for m in outer.members]
    for member in inner.members:
        cx, cy = member.shape.center
        if not any(abs(cx - ox) <= MATCH_TOLERANCE and abs(cy - oy) <= MATCH_TOLERANCE
                   for ox, oy in centres):
            return False
    return True


def _modifier_signatures(group: "Group") -> List[Tuple[str, float, float, float]]:
    """(role, diameter, z from, z to) for every modifier this hole's own walls recorded.

    A counterbore is a second, wider, *cylindrical* wall the measurement already attached to
    the bore; a countersink is the sloped skirt at its mouth.  Both are measured facts about
    one hole, and both are what a coaxial "extra" group at the same positions actually is."""
    found: List[Tuple[str, float, float, float]] = []
    if not group.is_hole or not group.members:
        return found
    first = group.members[0]
    record = first.counterbore
    if isinstance(record, dict) and record.get("from") is not None:
        size = record.get("diameter")
        if size is None:
            size = (record.get("size") or [0.0])[0]
        found.append(("counterbore", float(size), float(record["from"]), float(record["to"])))
    for skirt in first.skirts:
        outer = group.size[0] + 2.0 * skirt.outward
        high = abs(skirt.at - group.depth_to) <= abs(skirt.at - group.depth_from)
        span = (skirt.at, skirt.at + skirt.run) if high else (skirt.at - skirt.run, skirt.at)
        found.append(("countersink", outer, min(span), max(span)))
    return found


def fold_modifiers(groups: Sequence["Group"]) -> List[Tuple["Group", "Group"]]:
    """Fold every counterbore / countersink group into the hole it is part of.

    Returns the ``(modifier, parent)`` pairs; the caller drops the modifiers.  They leave the
    result entirely - the parent's ``counterbore`` and ``skirts`` records already carry every
    number they measured, and ``drawing_checks.measured_values`` pools those - so no consumer
    has to know that a second record ever existed, and no id is ever used twice.

    Why this is not grouping's business until now: the measurement finds a counterbore twice -
    once as the coaxial wall it attaches to the bore, and once as a section in its own right -
    and both are true.  What is not true is that there are two holes.  A sheet that prints
    a "3X \u23009.5 \u29134" row beside a "3X \u23005.2 THRU \u2334\u23009.5 \u29134" row at
    the same three positions tells a machinist to find six holes, and it is *confidently*
    wrong - worse than the old behaviour of losing the counterbore altogether.

    The fold is deliberately narrow.  A group is only folded when the parent's own wall
    measurement already named it - the counterbore record, or the skirt at the mouth - and its
    diameter and z range agree with that record.  A countersink cone has no wall of its own
    (``measured`` is False) and its apparent diameter is a function of where the cut landed,
    so for that one the test is that its diameter lies between the bore and the skirt's reach
    and its span overlaps the skirt's.  Nothing with a wall of its own is ever folded away."""
    folded: List[Tuple[Group, Group]] = []
    consumed: set = set()
    for parent in groups:
        if id(parent) in consumed:
            continue
        for role, diameter, low, high in _modifier_signatures(parent):
            for other in groups:
                if other is parent or id(other) in consumed or not other.is_hole:
                    continue
                if other.kind != "circle" or not _same_positions(other, parent):
                    continue
                limit = max(MODIFIER_SIZE_MM, MODIFIER_SIZE_FRACTION * diameter)
                if role == "counterbore":
                    fits = (abs(other.size[0] - diameter) <= limit
                            and abs(other.depth_from - low) <= MODIFIER_Z_MM
                            and abs(other.depth_to - high) <= MODIFIER_Z_MM)
                else:
                    # the cone has no wall, so its size is whatever the cut saw: it only has
                    # to sit between the bore and the skirt's reach, and overlap its span
                    fits = (not other.measured
                            and parent.size[0] - limit <= other.size[0] <= diameter + limit
                            and other.depth_from <= high + MODIFIER_Z_MM
                            and other.depth_to >= low - MODIFIER_Z_MM)
                if not fits:
                    continue
                consumed.add(id(other))
                folded.append((other, parent))
                # the hole is one feature: its extent runs from the far end of the bore to the
                # face the modifier opens onto, which is what a depth is measured from
                parent.depth_from = min(parent.depth_from, low)
                parent.depth_to = max(parent.depth_to, high)
                parent.depth = round(parent.depth_to - parent.depth_from, 6)
                break
    return folded


def feature_extents(f: Feature, target: str) -> Tuple[float, float, float, float]:
    """The rectangle (x0, y0, x1, y1) a prismatic feature of one view occupies in another view.

    A feature detected in view V is a prism: its cross-section along V's paper axes, extruded
    between ``depth_from`` and ``depth_to`` along V's depth axis. In any other orthographic view
    each paper axis is one of those three model axes, so the feature is an axis-aligned box."""
    return _extent_box(f, target, f.depth_from, f.depth_to)


def _extent_box(
    f: Feature, target: str, low: float, high: float
) -> Tuple[float, float, float, float]:
    src = VIEWS[f.view]
    dst = VIEWS[target]
    s = f.shape
    spans = {
        src["ax"]: (s.center[0] - s.size[0] / 2.0, s.center[0] + s.size[0] / 2.0),
        src["ay"]: (s.center[1] - s.size[1] / 2.0, s.center[1] + s.size[1] / 2.0),
        src["depth_axis"]: (low, high),
    }
    x0, x1 = spans[dst["ax"]]
    y0, y1 = spans[dst["ay"]]
    return x0, y0, x1, y1


def box_in_silhouette(
    box: Tuple[float, float, float, float], silhouette: Sequence[Loop], samples: int = 3
) -> bool:
    """Does this hidden-edge rectangle lie wholly within the view's own outline?

    A wall band is prismatic by construction, so a box measured from one always does. A box
    whose extent had to fall back to the level span need not: the bracket's 8 x 40 leg is
    prismatic to z 44 and chamfered above it, so a prism drawn to the level at 45 puts two
    corners outside the part. Drawing those tells a machinist there is material where there is
    none, so the box is dropped instead. Parity against every loop is what "inside" means here,
    because the silhouette's holes are loops too."""
    x0, y0, x1, y1 = box
    inset = 10 ** -COORD_DECIMALS
    xs = [x0 + inset + (x1 - x0 - 2 * inset) * i / (samples - 1.0) for i in range(samples)]
    ys = [y0 + inset + (y1 - y0 - 2 * inset) * i / (samples - 1.0) for i in range(samples)]
    for x in xs:
        for y in ys:
            hits = sum(1 for loop in silhouette if point_in_loop((x, y), loop))
            if hits % 2 == 0:
                return False
    return True


def feature_extent_boxes(f: Feature, target: str) -> List[Tuple[float, float, float, float]]:
    """One rectangle per run of material the feature passes through, in ``target``'s view.

    A hole bored through two slabs with a gap between them is *not* one prism: drawing it as
    one puts a hidden edge across empty space, outside the part's own silhouette, which is
    exactly what a machinist reads as "there is material here". Each measured wall segment gets
    its own box; a feature whose wall could not be found keeps its single level span."""
    spans = f.segments or [(f.depth_from, f.depth_to)]
    return [_extent_box(f, target, low, high) for low, high in spans]


# --------------------------------------------------------------------------------------
# sheet: drafting symbols, text metrics, layout and SVG
# --------------------------------------------------------------------------------------

_FONT = "Helvetica, Arial, sans-serif"
_MONO = "Menlo, Consolas, monospace"
_INK = "#111"
_DIM = "#1d4ed8"      # dimensions in blue so they read apart from geometry
_HIDDEN = "#666"
_CENTRE = "#7c3aed"   # centre marks and centre lines: neither geometry nor dimension
_WARN = "#b91c1c"
_TABLE_BG = "#f3f4f6"
_TEXT_MM = 2.6
_DIM_STEP = 7.0       # paper mm between stacked dimension lines
_MAX_LEVEL_CHAIN = 8
_SHORT_DIM_MM = 7.0   # paper span below which a dimension's text goes beyond the arrows

#: Drafting symbols, ISO 129-1 / ASME Y14.5.  ``↓``/``↑`` are *arrows*: they are not
#: drafting symbols, they do not say "depth", and nothing here may emit them.
SYM_DIAMETER = "⌀"      # diameter
SYM_DEPTH = "⤓"         # depth / deep
SYM_COUNTERBORE = "⌴"   # counterbore / spotface
SYM_COUNTERSINK = "⌵"   # countersink
SYM_RADIUS = "R"
SYM_DEGREE = "°"
#: Symbols a callout must never contain, checked by :func:`callout_matches_grammar` and by
#: ``tests/test_drawing.py`` over every sheet this module can produce.
FORBIDDEN_SYMBOLS: Tuple[str, ...] = ("↓", "↑")

#: Note text size and line pitch.  The pitch is set so consecutive note lines clear each other
#: by more than ``MIN_TEXT_GAP_MM`` as well: a note block is the one place where "lines of a
#: paragraph" could excuse a tighter spacing, and excusing it once is how a sheet ends up with
#: two different definitions of legible.
_NOTE_SIZE = 2.1
_NOTE_STEP = 3.9

#: Minimum clear space between two annotation labels, paper mm.  ISO 3098 sets the minimum
#: spacing between two lines of text at 0.5 x the character height; at the 2.6 mm text this
#: sheet uses that is 1.3 mm, so 1.5 mm is one notch clearer than lines of a paragraph - which
#: is the point: two labels must read as two labels.  It is also 5 x the 0.3 mm slack the
#: collision gate allows, so "no collision" is not the same claim as "legible".
MIN_TEXT_GAP_MM = 1.5

#: Shallow relief clustering (moulded-in text, logos, knurl relief).  Loops no deeper than
#: ``MARKING_MAX_DEPTH``, of at least ``MARKING_MIN_KINDS`` different shapes, at least
#: ``MARKING_MIN_LOOPS`` of them inside a ``MARKING_MAX_SPAN`` box in one z band, are a marking
#: and not a hole pattern.  Identical repeated pockets are deliberately *not* caught: a row of
#: five identical 0.5 mm pockets is a feature pattern somebody has to make.
MARKING_MAX_DEPTH = 1.0
MARKING_MIN_LOOPS = 3
MARKING_MIN_KINDS = 2
MARKING_MAX_SPAN = 40.0
MARKING_Z_TOLERANCE = 0.05

#: Ordinate dimensioning geometry, paper mm: how far the first row sits off the view, how far
#: apart the stagger rows are, and how many rows a view may use before the remaining positions
#: are carried by the coordinate table instead.
_ORD_LEAD = 5.0
_ORD_ROW = 9.0
_ORD_MAX_ROWS = 3
#: Paper mm reserved off the top and the left of the top view for the ordinate stack.
_ORD_RESERVE = 40.0
#: Radius of the drawn ordinate origin indicator, paper mm (ISO 129-1 uses a small open circle).
_ORIGIN_R = 1.3

#: Leader placement: the directions and stand-off distances a callout may try, paper mm.  A
#: leader is *placed*, not stacked in a column, so it lands beside the feature it names.
_LEADER_DIRS = 16
_LEADER_DISTANCES: Tuple[float, ...] = (9.0, 13.0, 18.0, 24.0, 31.0, 39.0, 50.0, 64.0,
                                       80.0, 100.0, 125.0)
_LEADER_SHOULDER = 2.5

#: Glyph advance as a fraction of the font size.  This mirrors ``drawing_checks._ADVANCE``
#: exactly and deliberately: the gate reconstructs every label's paper box from the SVG with
#: that table, so laying the sheet out with a *different* estimate would place labels this
#: module believes are clear and the gate reports as colliding.  Boxes are inflated by
#: ``_TEXT_PAD`` before any freedom test, which absorbs the difference between either estimate
#: and a real font.
_ADVANCE_DEFAULT = 0.55
_ADVANCE: Dict[str, float] = {}
for _ch in " .,:;'`|!":
    _ADVANCE[_ch] = 0.28
for _ch in "ijl":
    _ADVANCE[_ch] = 0.24
for _ch in "ft()[]{}/\\-":
    _ADVANCE[_ch] = 0.33
for _ch in "0123456789":
    _ADVANCE[_ch] = 0.556
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _ADVANCE[_ch] = 0.70
for _ch in "MW":
    _ADVANCE[_ch] = 0.90
for _ch in "mw":
    _ADVANCE[_ch] = 0.85
_MONO_ADVANCE = 0.60
_TEXT_PAD = 0.4


def text_width_mm(content: str, size: float, mono: bool = False) -> float:
    """Estimated advance width of ``content`` in paper millimetres."""
    if mono:
        return len(content) * _MONO_ADVANCE * size
    return sum(_ADVANCE.get(ch, _ADVANCE_DEFAULT) for ch in content) * size


def text_box(content: str, x: float, y: float, size: float, anchor: str, baseline: str,
             mono: bool, rotate: float) -> Tuple[float, float, float, float]:
    """The paper box a label occupies, reconstructed the way the gate reconstructs it."""
    width = text_width_mm(content, size, mono)
    if anchor == "middle":
        x0 = x - width / 2.0
    elif anchor == "end":
        x0 = x - width
    else:
        x0 = x
    if baseline in ("middle", "central"):
        y0, y1 = y - size / 2.0, y + size / 2.0
    else:
        y0, y1 = y - 0.80 * size, y + 0.20 * size
    box = (x0, y0, x0 + width, y1)
    if not rotate:
        return box
    radians = math.radians(rotate)
    cos, sin = math.cos(radians), math.sin(radians)
    xs: List[float] = []
    ys: List[float] = []
    for px, py in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])):
        dx, dy = px - x, py - y
        xs.append(x + dx * cos - dy * sin)
        ys.append(y + dx * sin + dy * cos)
    return (min(xs), min(ys), max(xs), max(ys))


def _boxes_clear(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float],
                 gap: float) -> bool:
    """True when ``a`` and ``b`` are at least ``gap`` apart in x or in y."""
    return (a[2] + gap <= b[0] or b[2] + gap <= a[0]
            or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _box_inside(inner: Tuple[float, float, float, float],
                outer: Tuple[float, float, float, float]) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_cross(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> bool:
    """Do segments a-b and c-d properly cross?  Touching at an endpoint does not count.

    Leaders start *on* the boundary of the feature they name and end *on* their own label, so
    an endpoint touch is the normal case and only a genuine crossing is a defect."""
    d1 = _orient(c[0], c[1], d[0], d[1], a[0], a[1])
    d2 = _orient(c[0], c[1], d[0], d[1], b[0], b[1])
    d3 = _orient(a[0], a[1], b[0], b[1], c[0], c[1])
    d4 = _orient(a[0], a[1], b[0], b[1], d[0], d[1])
    if (d1 > eps and d2 < -eps or d1 < -eps and d2 > eps) and \
            (d3 > eps and d4 < -eps or d3 < -eps and d4 > eps):
        return True
    return False


def polyline_loop_crossings(points: Sequence[Point], loop: Sequence[Point]) -> int:
    """How many times a polyline properly crosses a closed loop."""
    hits = 0
    n = len(loop)
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        for j in range(n):
            if segments_cross(a, b, loop[j], loop[(j + 1) % n]):
                hits += 1
    return hits


def segment_hits_box(p0: Point, p1: Point, box: Tuple[float, float, float, float]) -> bool:
    """Does segment p0-p1 pass through the open ``box``?  Liang-Barsky clipping.

    Mirrors ``drawing_checks.segment_hits_box``, which is how the gate finds a label struck
    through by a line; laid out with the same test, the sheet cannot fail it."""
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    t0, t1 = 0.0, 1.0
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    for p, q in ((-dx, p0[0] - box[0]), (dx, box[2] - p0[0]),
                 (-dy, p0[1] - box[1]), (dy, box[3] - p0[1])):
        if p == 0.0:
            if q <= 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 >= t1:
            return False
    return True


#: Segment classes that are annotation, not geometry: no label may sit on one.
_ANNOTATION_KINDS = ("dim", "ext", "leader")


def fit_scale(view_w: float, view_h: float, avail_w: float, avail_h: float) -> Optional[float]:
    """Largest standard scale whose scaled views fit the paper area, or None if none does."""
    for s in STANDARD_SCALES:
        if view_w * s <= avail_w and view_h * s <= avail_h:
            return s
    return None


def choose_scale(view_w: float, view_h: float, avail_w: float, avail_h: float) -> float:
    """Largest standard scale whose scaled views fit the available paper area."""
    found = fit_scale(view_w, view_h, avail_w, avail_h)
    return STANDARD_SCALES[-1] if found is None else found


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


class _Svg:
    """A tiny SVG string builder in paper millimetres, y down.

    It also *remembers* what it drew: every label's paper box and every annotation segment.
    Layout is a placement problem - a label has to go where nothing else is, a leader has to
    reach its feature without crossing another leader - and neither question can be answered
    by a builder that only concatenates strings."""

    def __init__(self, width: float, height: float) -> None:
        self.w, self.h = width, height
        self.parts: List[str] = []
        #: every label drawn, as (box, content); the placer tests candidates against these.
        self.boxes: List[Tuple[Tuple[float, float, float, float], str]] = []
        #: every dimension / extension / leader segment drawn, as (kind, p0, p1).
        self.segments: List[Tuple[str, Point, Point]] = []
        #: rectangles no annotation may enter (the bottom band, the sheet margin).
        self.reserved: List[Tuple[float, float, float, float]] = []

    # -- primitives --------------------------------------------------------------------

    def line(self, x0: float, y0: float, x1: float, y1: float, *, width: float = 0.35,
             color: str = _INK, dash: Optional[str] = None, arrows: bool = False,
             cls: Optional[str] = None) -> None:
        extra = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        if arrows:
            extra += ' marker-start="url(#arw)" marker-end="url(#arw)"'
        if cls:
            extra += ' class="{}"'.format(cls)
            self.segments.append((cls, (x0, y0), (x1, y1)))
        self.parts.append(
            '<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}" stroke="{}" '
            'stroke-width="{:.2f}" stroke-linecap="round"{}/>'.format(
                x0, y0, x1, y1, color, width, extra))

    def poly(self, pts: Sequence[Point], *, width: float = 0.35, color: str = _INK,
             dash: Optional[str] = None, fill: str = "none", cls: Optional[str] = None) -> None:
        d = " ".join("{:.3f},{:.3f}".format(x, y) for x, y in pts)
        extra = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        if cls:
            extra += ' class="{}"'.format(cls)
        self.parts.append(
            '<polygon points="{}" fill="{}" stroke="{}" stroke-width="{:.2f}" '
            'stroke-linejoin="round"{}/>'.format(d, fill, color, width, extra))

    def rect(self, x: float, y: float, w: float, h: float, *, width: float = 0.35,
             color: str = _INK, fill: str = "none", dash: Optional[str] = None) -> None:
        extra = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        self.parts.append(
            '<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="{}" stroke="{}" '
            'stroke-width="{:.2f}"{}/>'.format(x, y, w, h, fill, color, width, extra))

    def text(self, x: float, y: float, s: str, *, size: float = _TEXT_MM, anchor: str = "start",
             color: str = _INK, bold: bool = False, mono: bool = False, rotate: float = 0.0,
             baseline: str = "auto") -> Tuple[float, float, float, float]:
        weight = ' font-weight="bold"' if bold else ""
        tr = ' transform="rotate({:.1f} {:.3f} {:.3f})"'.format(rotate, x, y) if rotate else ""
        self.parts.append(
            '<text x="{:.3f}" y="{:.3f}" font-family="{}" font-size="{:.2f}" text-anchor="{}" '
            'fill="{}" dominant-baseline="{}"{}{}>{}</text>'.format(
                x, y, _MONO if mono else _FONT, size, anchor, color, baseline, weight, tr,
                _esc(s)))
        box = text_box(s, x, y, size, anchor, baseline, mono, rotate)
        self.boxes.append((box, s))
        return box

    # -- placement ---------------------------------------------------------------------

    def reserve(self, box: Tuple[float, float, float, float]) -> None:
        self.reserved.append(box)

    def label_fits(self, box: Tuple[float, float, float, float],
                   gap: float = MIN_TEXT_GAP_MM) -> bool:
        """Is this label box clear of every label already drawn, and on the sheet?"""
        padded = (box[0] - _TEXT_PAD, box[1] - _TEXT_PAD, box[2] + _TEXT_PAD, box[3] + _TEXT_PAD)
        if not _box_inside(padded, (0.0, 0.0, self.w, self.h)):
            return False
        for keep_out in self.reserved:
            if not _boxes_clear(padded, keep_out, 0.0):
                return False
        for other, _ in self.boxes:
            if not _boxes_clear(padded, other, gap):
                return False
        # a label an annotation line runs through is a struck-through number
        for kind, p0, p1 in self.segments:
            if kind in _ANNOTATION_KINDS and segment_hits_box(p0, p1, padded):
                return False
        return True

    def path_clear(self, points: Sequence[Point], pad: float = 0.3) -> bool:
        """Does a polyline about to be drawn stay out of every label already on the sheet,
        and cross no dimension, extension or leader line?"""
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            for box, _ in self.boxes:
                grown = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
                if segment_hits_box(a, b, grown):
                    return False
            for kind, p0, p1 in self.segments:
                if kind in _ANNOTATION_KINDS and segments_cross(a, b, p0, p1):
                    return False
        return True

    def mark(self) -> Tuple[int, int, int, int]:
        """A restore point: layout alternatives are drawn, scored, and rolled back."""
        return (len(self.parts), len(self.boxes), len(self.segments), len(self.reserved))

    def restore(self, point: Tuple[int, int, int, int]) -> None:
        del self.parts[point[0]:]
        del self.boxes[point[1]:]
        del self.segments[point[2]:]
        del self.reserved[point[3]:]

    def place_text(self, x: float, y: float, s: str, **kw: Any) -> Optional[
            Tuple[float, float, float, float]]:
        """Draw ``s`` only if its box is clear; returns the box, or None if it was refused."""
        box = text_box(s, x, y, kw.get("size", _TEXT_MM), kw.get("anchor", "start"),
                       kw.get("baseline", "auto"), kw.get("mono", False), kw.get("rotate", 0.0))
        if not self.label_fits(box):
            return None
        return self.text(x, y, s, **kw)

    def render(self) -> str:
        head = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
            'viewBox="0 0 {w} {h}">\n'
            '<defs><marker id="arw" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="3" '
            'markerHeight="3" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="{dim}"/></marker></defs>\n'
            '<rect width="{w}" height="{h}" fill="white"/>\n'
        ).format(w=self.w, h=self.h, dim=_DIM)
        return head + "\n".join(self.parts) + "\n</svg>\n"


class _ViewFrame:
    """Maps a view's model coordinates to paper.

    The view's **origin** - the lower-left corner of its silhouette bbox - sits at paper
    ``(ox, oy)``, and it is the one frame the whole sheet works in: the drawn origin
    indicator, every ordinate value and every coordinate in the table are distances from it.
    The model coordinates stay in the JSON result, where a number cannot be mistaken for a
    dimension a machinist is meant to measure."""

    def __init__(self, view: ViewData, scale: float, ox: float, oy: float) -> None:
        self.view, self.s, self.ox, self.oy = view, scale, ox, oy
        self.x0, self.y0, self.x1, self.y1 = view.bbox
        #: the origin in model mm, rounded exactly as the JSON result rounds it, so a sheet
        #: value and a table value are the same subtraction and cannot disagree.
        self.ox_model = round(self.x0, 3)
        self.oy_model = round(self.y0, 3)

    def px(self, x: float) -> float:
        return self.ox + self.s * (x - self.x0)

    def py(self, y: float) -> float:
        return self.oy - self.s * (y - self.y0)

    def dx(self, x: float) -> float:
        """Model x as a distance from the drawn origin, to the result's own rounding."""
        return round(x, 3) - self.ox_model

    def dy(self, y: float) -> float:
        return round(y, 3) - self.oy_model

    @property
    def top(self) -> float:
        return self.py(self.y1)

    @property
    def right(self) -> float:
        return self.px(self.x1)

    @property
    def paper_box(self) -> Tuple[float, float, float, float]:
        return (self.ox, self.top, self.right, self.oy)


def _dim_h(svg: _Svg, fr: _ViewFrame, xa: float, xb: float, y_line: float, ya: float,
           yb: float, text: Optional[str] = None) -> None:
    """Horizontal dimension between model x ``xa`` and ``xb`` at paper y ``y_line``.

    Extension lines rise from model y ``ya`` / ``yb`` - which callers pass as an *outer* edge
    of the view, never as a feature's own coordinate, so that no extension line is ever drawn
    across the part."""
    pa, pb = fr.px(xa), fr.px(xb)
    for x, y in ((pa, ya), (pb, yb)):
        y_from = fr.py(y)
        gap = -1.0 if y_line < y_from else 1.0
        svg.line(x, y_from + gap, x, y_line + gap, width=0.18, color=_DIM, cls="ext")
    svg.line(pa, y_line, pb, y_line, width=0.25, color=_DIM, arrows=abs(pb - pa) > 4, cls="dim")
    label = text if text is not None else _dimnum(abs(xb - xa))
    if abs(pb - pa) < _SHORT_DIM_MM:   # too short for text between the arrows: put it beyond
        svg.text(max(pa, pb) + 1.2, y_line - 0.8, label, anchor="start", color=_DIM)
    else:
        svg.text((pa + pb) / 2.0, y_line - 0.8, label, anchor="middle", color=_DIM)


def _dim_v(svg: _Svg, fr: _ViewFrame, ya: float, yb: float, x_line: float, xa: float,
           xb: float, text: Optional[str] = None, short_below: bool = False) -> None:
    """Vertical dimension; ``short_below`` puts a short one's text under its lower end.

    The level chain needs that: its dimensions all start at the base, and the text of a short
    one printed above its upper end sat exactly where the extension lines of the taller ones
    run out to their own dimension lines - a struck-through number.  Under the base nothing
    runs."""
    pa, pb = fr.py(ya), fr.py(yb)
    for y, x in ((pa, xa), (pb, xb)):
        x_from = fr.px(x)
        gap = -1.0 if x_line < x_from else 1.0
        svg.line(x_from + gap, y, x_line + gap, y, width=0.18, color=_DIM, cls="ext")
    svg.line(x_line, pa, x_line, pb, width=0.25, color=_DIM, arrows=abs(pb - pa) > 4, cls="dim")
    label = text if text is not None else _dimnum(abs(yb - ya))
    if abs(pb - pa) < _SHORT_DIM_MM and short_below:
        svg.text(x_line - 0.8, max(pa, pb) + 1.2, label, anchor="end", color=_DIM, rotate=-90)
    elif abs(pb - pa) < _SHORT_DIM_MM:
        svg.text(x_line - 0.8, min(pa, pb) - 1.2, label, anchor="start", color=_DIM, rotate=-90)
    else:
        svg.text(x_line - 0.8, (pa + pb) / 2.0, label, anchor="middle", color=_DIM, rotate=-90)


def _draw_centre_mark(svg: _Svg, cx: float, cy: float, radius: float) -> None:
    """ISO 128 centre mark and centre lines for one circular feature, in paper mm.

    A circular feature is located by its centre, and a drawing has to *show* that centre: the
    ordinate value outside the view means nothing if a reader cannot see which point on the
    part it refers to.  The short cross is the centre mark; the dash-dot pair running past the
    feature are its centre lines."""
    arm = max(radius * 0.35, 0.8)
    reach = radius + max(1.6, radius * 0.25)
    svg.parts.append('<g class="cmark">')
    svg.line(cx - arm, cy, cx + arm, cy, width=0.18, color=_CENTRE)
    svg.line(cx, cy - arm, cx, cy + arm, width=0.18, color=_CENTRE)
    svg.parts.append("</g>")
    svg.parts.append('<g class="cline">')
    for x0, y0, x1, y1 in ((cx - reach, cy, cx - arm * 1.6, cy), (cx + arm * 1.6, cy,
                                                                  cx + reach, cy),
                           (cx, cy - reach, cx, cy - arm * 1.6), (cx, cy + arm * 1.6, cx,
                                                                  cy + reach)):
        svg.line(x0, y0, x1, y1, width=0.15, color=_CENTRE, dash="2.4,0.7,0.5,0.7")
    svg.parts.append("</g>")


def _circular_centres(fr: _ViewFrame, groups: Sequence["Group"]) -> List[Tuple[float, float,
                                                                               float]]:
    """(paper x, paper y, paper radius) of every circular feature *member* drawn in this view.

    One per member, not one per distinct centre: a bore inside a boss is two circular features
    that happen to be coaxial, and both are entitled to a centre mark."""
    found: List[Tuple[float, float, float]] = []
    for g in groups:
        if g.kind != "circle":
            continue
        for member in g.members:
            if member.view != fr.view.name:
                continue
            cx, cy = member.shape.center
            found.append((fr.px(cx), fr.py(cy), fr.s * max(member.shape.size) / 2.0))
    return found


def _draw_geometry(svg: _Svg, fr: _ViewFrame, others: Sequence[ViewData],
                   centres: Sequence[Tuple[float, float, float]] = ()) -> List[List[Point]]:
    """Draw one view; return every section and outline loop it drew, in paper millimetres.

    The loops are returned because a leader has to be routed around *what is on the paper*,
    not around the subset of it the feature table happens to name: a boss's base-fillet ring
    is a line a leader can cross just as visibly as the boss."""
    v = fr.view
    drawn: List[List[Point]] = []
    # thin section edges first (under), silhouette last (over)
    seen: List[Shape] = []
    for _, _, _, loops in v.slices:
        for s in classify_slice(loops):
            if _same_loop_set(s, seen):
                continue
            seen.append(s)
            paper = [(fr.px(x), fr.py(y)) for x, y in s.loop]
            drawn.append(paper)
            svg.poly(paper, width=0.2, color=_HIDDEN, cls="section")
    for other in others:
        for f in other.features:
            for box in feature_extent_boxes(f, v.name):
                if not box_in_silhouette(box, v.silhouette):
                    continue      # no material there: a hidden edge across air is a lie
                x0, y0, x1, y1 = box
                pts = [(fr.px(x0), fr.py(y0)), (fr.px(x1), fr.py(y0)),
                       (fr.px(x1), fr.py(y1)), (fr.px(x0), fr.py(y1))]
                dash = "1.2,0.8" if f.shape.is_hole else None
                svg.poly(pts, width=0.2, color=_HIDDEN, dash=dash, cls="hidden")
    for loop in v.silhouette:
        paper = [(fr.px(x), fr.py(y)) for x, y in loop]
        drawn.append(paper)
        svg.poly(paper, width=0.5, cls="outline")
    for cx, cy, radius in centres:
        _draw_centre_mark(svg, cx, cy, radius)
    return drawn


def _draw_origin(svg: _Svg, fr: _ViewFrame) -> None:
    """The one drawn origin every value on this sheet is measured from.

    ISO 129-1 marks the origin of an ordinate system with a small open circle at a physical
    corner of the part.  Drawing it is what makes the frame checkable by eye: the reader can
    see where zero is instead of inferring it from a note."""
    svg.parts.append('<g class="origin">')
    svg.parts.append(
        '<circle cx="{:.3f}" cy="{:.3f}" r="{:.2f}" fill="none" stroke="{}" '
        'stroke-width="0.3"/>'.format(fr.ox, fr.oy, _ORIGIN_R, _DIM))
    svg.line(fr.ox - _ORIGIN_R * 2.2, fr.oy, fr.ox + _ORIGIN_R * 2.2, fr.oy, width=0.18,
             color=_DIM)
    svg.line(fr.ox, fr.oy - _ORIGIN_R * 2.2, fr.ox, fr.oy + _ORIGIN_R * 2.2, width=0.18,
             color=_DIM)
    svg.parts.append("</g>")


def ordinate_values(fr: _ViewFrame, groups: Sequence["Group"]) -> Tuple[List[float],
                                                                        List[float]]:
    """Distinct x and y ordinates this view must carry, as distances from its origin.

    Every *member* of every group contributes, which is the whole change: dimensioning the
    first member of a group and one twin leaves the third hole of a row with no position
    anywhere on the sheet."""
    xs: List[float] = []
    ys: List[float] = []
    for g in groups:
        for member in g.members:
            if member.view != fr.view.name:
                continue
            cx, cy = member.shape.center
            for value, seen in ((fr.dx(cx), xs), (fr.dy(cy), ys)):
                if not any(abs(value - v) < MATCH_TOLERANCE for v in seen):
                    seen.append(value)
    return sorted(xs), sorted(ys)


#: Ordinate label size, paper mm, and the spacing it implies.  Two labels in one column need
#: their padded boxes ``MIN_TEXT_GAP_MM`` apart; an extension line passing a label in a nearer
#: column needs to clear its padded box.  Half the label pitch satisfies the second, which is
#: why two columns staggered by half a pitch double what one column carries - and why a third
#: column adds nothing (its lines would have to pass between labels a quarter-pitch apart).
_ORD_TEXT = 2.4
_ORD_PITCH = _ORD_TEXT + 2.0 * _TEXT_PAD + MIN_TEXT_GAP_MM
#: Paper mm along the extension line spent on a jog, and where it starts off the view edge.
_ORD_JOG = 3.0
_ORD_JOG_FROM = 2.0


def spread_positions(values: Sequence[float], gap: float, lo: float,
                     hi: float) -> Optional[List[float]]:
    """Positions for sorted ``values``, each at least ``gap`` from the next, inside [lo, hi].

    The least-squares displacement solution: subtracting ``i * gap`` turns "at least gap
    apart" into "non-decreasing", which pool-adjacent-violators solves exactly, and the
    bounds then clamp the pooled blocks.  Deterministic, order preserving - so extension lines
    jogged from each value to its position can never cross - and a value with room keeps its
    own position (no jog).  None when the values need more than [lo, hi]."""
    n = len(values)
    if n == 0:
        return []
    if (n - 1) * gap > hi - lo + 1e-9:
        return None
    targets = [v - i * gap for i, v in enumerate(values)]
    blocks: List[List[float]] = []          # [mean, weight]
    for t in targets:
        blocks.append([t, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            mean, weight = blocks.pop()
            blocks[-1][0] = (blocks[-1][0] * blocks[-1][1] + mean * weight) / \
                (blocks[-1][1] + weight)
            blocks[-1][1] += weight
    upper = hi - (n - 1) * gap
    out: List[float] = []
    for mean, weight in blocks:
        u = min(max(mean, lo), upper)
        out.extend([u] * int(weight))
    return [u + i * gap for i, u in enumerate(out)]


def _thin_to_fit(values: Sequence[float], capacity: int) -> Tuple[List[float], List[float]]:
    """(kept, refused): drop the most crowded values until ``capacity`` remain.

    Crowding is the gap to the nearer neighbour; the later of the tightest pair goes first, so
    the result is deterministic.  A refused value is never lost - the coordinate table, in the
    same frame, carries every position."""
    kept = list(values)
    refused: List[float] = []
    while len(kept) > max(capacity, 0):
        best_i, best_gap = len(kept) - 1, math.inf
        for i in range(1, len(kept)):
            if kept[i] - kept[i - 1] < best_gap - 1e-9:
                best_gap, best_i = kept[i] - kept[i - 1], i
        refused.append(kept.pop(best_i))
    return kept, sorted(refused)


def plan_ordinates(values: Sequence[float], lo: float, hi: float,
                   pitch: float = _ORD_PITCH) -> Tuple[List[Tuple[float, float, int]],
                                                       List[float]]:
    """Where each ordinate label goes along one edge: ([(value_paper, slot, column)], refused).

    Values are paper positions along the edge.  One column at the full pitch is preferred when
    it is enough; otherwise two columns staggered by half a pitch, alternating - every second
    label in the far column, its extension line passing exactly between two near labels.  When
    even that cannot hold them all the most crowded values are refused to the table.  The
    displacement from a value to its slot is drawn as a jog in the extension line."""
    ordered = sorted(values)
    best: Optional[Tuple[Tuple[int, float, int], List[Tuple[float, float, int]],
                         List[float]]] = None
    for columns in (1, 2):
        gap = pitch / columns
        capacity = int(math.floor((hi - lo) / gap + 1e-9)) + 1
        kept, refused = _thin_to_fit(ordered, capacity)
        slots = spread_positions(kept, gap, lo, hi)
        if slots is None:
            continue
        # a label takes the near column whenever the previous near label is a full pitch
        # away; the far one otherwise.  Slots are >= pitch/2 apart, so two far labels are
        # never closer than a pitch and a far label's line clears every near label.
        plan = []
        last_near = -math.inf
        for v, slot in zip(kept, slots):
            column = 0 if columns == 1 or slot - last_near >= pitch - 1e-6 else 1
            if column == 0:
                last_near = slot
            plan.append((v, slot, column))
        moved = sum(abs(v - slot) for v, slot, _ in plan)
        # fewer refused first, then less jogging; one column wins an exact tie
        score = (len(refused), round(moved, 3), columns)
        if best is None or score < best[0]:
            best = (score, plan, refused)
    if best is None:
        return [], ordered
    return best[1], best[2]


def _draw_ordinates(svg: _Svg, fr: _ViewFrame, xs: Sequence[float],
                    ys: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Ordinate dimensions off the top and the left of the view; returns what would not fit.

    Baseline/ordinate dimensioning replaces the old "first member of a group plus one twin":
    every distinct coordinate gets one value, measured from the drawn origin, so a reader
    locates any feature by reading one number per axis with no chain to follow.  Extension
    lines start at the view's own bounding edge and run *outward* only - an extension line
    drawn across the part reads as an edge that is not there.

    Values closer together than a label is tall are not stacked in whatever row happens to be
    free (that is how one value's extension line came to strike through its neighbour's
    number): :func:`plan_ordinates` spreads them along the edge, order preserved, and the
    extension line jogs from the value to its label.  A label is still only drawn where it is
    clear of every other label and line; anything else is refused and carried by the
    coordinate table, which is in the same frame."""
    limit = _ORD_LEAD + _ORD_MAX_ROWS * _ORD_ROW + 4.0      # where the overall dimension runs
    half = _ORD_TEXT / 2.0 + _TEXT_PAD + 0.2
    refused_x: List[float] = []
    refused_y: List[float] = []
    for axis in ("x", "y"):
        values = xs if axis == "x" else ys
        if axis == "x":
            paper = {fr.px(fr.ox_model + v): v for v in values}
            lo, hi = fr.ox + half, fr.right - half
        else:
            paper = {fr.py(fr.oy_model + v): v for v in values}
            lo, hi = fr.top + half, fr.oy - half
        plan, refused = plan_ordinates(list(paper), lo, hi)
        refused_values = [paper[r] for r in refused]
        jogs = any(abs(pos - slot) > 1e-3 for pos, slot, _ in plan)
        lead = _ORD_LEAD + (_ORD_JOG if jogs else 0.0)
        texts = {pos: _dimnum(paper[pos]) for pos, _, _ in plan}
        near = max((text_width_mm(texts[pos], _ORD_TEXT) for pos, _, col in plan if col == 0),
                   default=0.0)
        tips = (lead, lead + 1.0 + near + MIN_TEXT_GAP_MM + 1.0)
        for pos, slot, col in plan:
            tip = tips[col]
            text = texts[pos]
            if tip + 1.0 + text_width_mm(text, _ORD_TEXT) > limit - 1.5:
                refused_values.append(paper[pos])
                continue
            if axis == "x":
                path = [(pos, fr.top - 1.0)]
                if abs(pos - slot) > 1e-3:
                    path += [(pos, fr.top - _ORD_JOG_FROM),
                             (slot, fr.top - _ORD_JOG_FROM - _ORD_JOG)]
                path.append((slot, fr.top - tip))
                label = (slot, fr.top - tip - 1.0)
                kw: Dict[str, Any] = {"anchor": "start", "rotate": -90}
            else:
                path = [(fr.ox - 1.0, pos)]
                if abs(pos - slot) > 1e-3:
                    path += [(fr.ox - _ORD_JOG_FROM, pos),
                             (fr.ox - _ORD_JOG_FROM - _ORD_JOG, slot)]
                path.append((fr.ox - tip, slot))
                label = (fr.ox - tip - 1.0, slot)
                kw = {"anchor": "end", "rotate": 0.0}
            box = text_box(text, label[0], label[1], _ORD_TEXT, kw["anchor"], "middle", False,
                           kw["rotate"])
            if not svg.label_fits(box) or not svg.path_clear(path):
                refused_values.append(paper[pos])
                continue
            for a, b in zip(path, path[1:]):
                svg.line(a[0], a[1], b[0], b[1], width=0.18, color=_DIM, cls="ext")
            svg.text(label[0], label[1], text, size=_ORD_TEXT, baseline="middle", color=_DIM,
                     **kw)
        (refused_x if axis == "x" else refused_y).extend(sorted(refused_values))
    return refused_x, refused_y


# --------------------------------------------------------------------------------------
# leaders: placed beside the feature they name, never stacked in one column
# --------------------------------------------------------------------------------------


@dataclass
class _Target:
    """One member a leader may point at: its paper centre, radius and boundary loop."""

    centre: Point
    radius: float
    loop: List[Point]
    bbox: Tuple[float, float, float, float]


@dataclass
class _Callout:
    """One leader to place: what it says, and every member it could legitimately point at.

    Any member of a group is a correct target - the quantity prefix says how many there are -
    so a leader that cannot be placed beside the first hole is tried against the others before
    the callout is given up and left to the table."""

    ident: str
    text: str
    short: str
    targets: List[_Target]
    #: Who keeps a leader when the margin runs out: holes and markings first, then bosses,
    #: then pads - largest first to go, because a big pad is usually an internal ledge whose
    #: table row says everything the leader would.
    rank: Tuple[int, float] = (0, 0.0)


def _ray_loop_point(centre: Point, direction: Point, loop: Sequence[Point],
                    fallback: float) -> Point:
    """Where a ray leaving ``centre`` meets ``loop``: the point a leader must terminate on.

    A leader that stops in the white space beside a hole names nothing; it has to land on the
    feature's own boundary.  When the centre is outside its own loop (a C-shaped profile) the
    nearest vertex in that direction is used instead of a fabricated point on a circle."""
    cx, cy = centre
    ux, uy = direction
    best: Optional[float] = None
    n = len(loop)
    for i in range(n):
        ax, ay = loop[i]
        bx, by = loop[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        denominator = ux * ey - uy * ex
        if abs(denominator) < 1e-12:
            continue
        t = ((ax - cx) * ey - (ay - cy) * ex) / denominator
        s = ((ax - cx) * uy - (ay - cy) * ux) / denominator
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9 and (best is None or t < best):
            best = t
    if best is None:
        return (cx + ux * fallback, cy + uy * fallback)
    return (cx + ux * best, cy + uy * best)


def _bbox_of(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_apart(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _loop_crossings(polyline: Sequence[Point], loop: Sequence[Point],
                    box: Tuple[float, float, float, float]) -> int:
    """:func:`polyline_loop_crossings` with the bounding-box rejections a search needs."""
    line_box = _bbox_of(polyline)
    if _bbox_apart(line_box, box):
        return 0
    hits = 0
    n = len(loop)
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        seg = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        for j in range(n):
            c, d = loop[j], loop[(j + 1) % n]
            if _bbox_apart(seg, (min(c[0], d[0]), min(c[1], d[1]),
                                 max(c[0], d[0]), max(c[1], d[1]))):
                continue
            if segments_cross(a, b, c, d):
                hits += 1
    return hits


def _leader_is_clear(polyline: Sequence[Point], svg: _Svg, placed: Sequence[Sequence[Point]],
                     obstacles: Sequence[Tuple[List[Point],
                                               Tuple[float, float, float, float], int]]) -> bool:
    """Every crossing rule a leader must satisfy, checked as segment intersections.

    A leader may leave each region that *encloses* its own feature exactly once - the part's
    outline, and any pocket the feature sits inside - because the label is outside them and
    there is no other way out.  That is the "final approach" allowance, and it is granted per
    enclosing loop rather than as a blanket exemption.  Everything else is forbidden: another
    leader, a dimension line, an extension line, or any feature the leader does not name."""
    for other in placed:
        for i in range(len(polyline) - 1):
            for j in range(len(other) - 1):
                if segments_cross(polyline[i], polyline[i + 1], other[j], other[j + 1]):
                    return False
    for kind, p0, p1 in svg.segments:
        if kind not in ("dim", "ext", "leader"):
            continue
        for i in range(len(polyline) - 1):
            if segments_cross(polyline[i], polyline[i + 1], p0, p1):
                return False
    for loop, box, allowed in obstacles:
        if _loop_crossings(polyline, loop, box) > allowed:
            return False
    return True


_Obstacle = Tuple[List[Point], Tuple[float, float, float, float], int]


def _obstacles_for(target: "_Target",
                   loops: Sequence[Tuple[List[Point],
                                         Tuple[float, float, float, float]]]) -> List[_Obstacle]:
    """Every loop a leader to ``target`` must respect, with how often it may cross each.

    Once for a loop that encloses the feature's centre - the part outline, the pocket it sits
    in, its own base fillet - because the label is outside them and the leader has to come
    out.  Never for anything else.  Computed once per target because the placement search
    tries hundreds of candidate positions against the same loops."""
    return [(loop, box, 1 if point_in_loop(target.centre, loop) else 0) for loop, box in loops]


def _draw_leader(svg: _Svg, ident: str, polyline: Sequence[Point], label: str, tx: float,
                 anchor: str) -> None:
    svg.parts.append('<g class="leader" data-id="{}">'.format(_esc(ident)))
    for a, b in zip(polyline, polyline[1:]):
        svg.line(a[0], a[1], b[0], b[1], width=0.2, color=_INK, cls="leader")
    svg.parts.append("</g>")
    svg.text(tx, polyline[-1][1], label, anchor=anchor, baseline="middle")


def _path_avoids(polyline: Sequence[Point],
                 boxes: Sequence[Tuple[float, float, float, float]]) -> bool:
    return not any(segment_hits_box(a, b, box)
                   for a, b in zip(polyline, polyline[1:]) for box in boxes)


def _place_local(svg: _Svg, fr: _ViewFrame, callouts: Sequence[_Callout],
                 all_loops: Sequence[Tuple[List[Point], Tuple[float, float, float, float]]]
                 ) -> Tuple[List[str], List[_Callout]]:
    """Each leader beside its feature: (placed ids, callouts that found no place).

    Any member of the group, sixteen directions out of it, increasing stand-off, and the first
    position whose label is clear of every label and annotation line by ``MIN_TEXT_GAP_MM``,
    off the view, and whose leader crosses no other leader, dimension line, label or feature
    it does not name.  The long callout is tried first and the bare id second, because a
    legible id pointing at the right hole plus a table row beats an unplaceable sentence."""
    placed_lines: List[List[Point]] = [
        [p0, p1] for kind, p0, p1 in svg.segments if kind == "leader"]
    placed: List[str] = []
    missing: List[_Callout] = []
    view_cx = (fr.ox + fr.right) / 2.0
    view_cy = (fr.top + fr.oy) / 2.0
    view_box = _inflate(fr.paper_box, 0.5)
    for callout in sorted(callouts, key=lambda c: -max(t.radius for t in c.targets)):
        done = False
        for label in (callout.text, callout.short):
            for distance in _LEADER_DISTANCES:
                for target in callout.targets:
                    cx, cy = target.centre
                    obstacles = _obstacles_for(target, all_loops)
                    away = math.atan2(cy - view_cy, cx - view_cx) \
                        if (cx, cy) != (view_cx, view_cy) else 0.0
                    order = sorted(
                        range(_LEADER_DIRS),
                        key=lambda k: abs(((2.0 * math.pi * k / _LEADER_DIRS - away + math.pi)
                                           % (2.0 * math.pi)) - math.pi))
                    for k in order:
                        angle = 2.0 * math.pi * k / _LEADER_DIRS
                        ux, uy = math.cos(angle), math.sin(angle)
                        landing = (cx + ux * (target.radius + distance),
                                   cy + uy * (target.radius + distance))
                        side = 1.0 if ux >= 0.0 else -1.0
                        shoulder = (landing[0] + side * _LEADER_SHOULDER, landing[1])
                        anchor = "start" if side > 0 else "end"
                        tx = shoulder[0] + side * 0.8
                        box = text_box(label, tx, shoulder[1], _TEXT_MM, anchor, "middle",
                                       False, 0.0)
                        if not _boxes_clear(box, view_box, 0.0) or not svg.label_fits(box):
                            continue
                        start = _ray_loop_point((cx, cy), (ux, uy), target.loop,
                                                max(target.radius, 0.4))
                        polyline = [start, landing, shoulder]
                        if not _leader_is_clear(polyline, svg, placed_lines, obstacles) \
                                or not svg.path_clear(polyline) \
                                or not _path_avoids(polyline, svg.reserved):
                            continue
                        _draw_leader(svg, callout.ident, polyline, label, tx, anchor)
                        placed_lines.append(polyline)
                        placed.append(callout.ident)
                        done = True
                        break
                    if done:
                        break
                if done:
                    break
            if done:
                break
        if not done:
            missing.append(callout)
    return placed, missing


#: Margin columns: how far outside the view the label column starts, and the row pitch -
#: one label plus its pad and ``MIN_TEXT_GAP_MM``, so stacked labels can never collide.
_MARGIN_LEAD = 10.0
_MARGIN_PITCH = _TEXT_MM + 2.0 * _TEXT_PAD + MIN_TEXT_GAP_MM + 0.05
#: At most this many members of a group are tried as the leader's target in the margin, the
#: ones nearest the column: any member is a correct target and the far ones only add length.
_MARGIN_TARGETS = 4


@dataclass
class _Slot:
    """One feasible (callout, row) pairing in the margin column."""

    cost: float
    row: int
    polyline: List[Point]
    label: str
    tx: float
    anchor: str


def _polylines_cross(a: Sequence[Point], b: Sequence[Point]) -> bool:
    return any(segments_cross(a[i], a[i + 1], b[j], b[j + 1])
               for i in range(len(a) - 1) for j in range(len(b) - 1))


def _margin_options(svg: _Svg, fr: _ViewFrame, callout: _Callout, rows: Sequence[float],
                    sides: Sequence[Tuple[float, float]],
                    view_box: Tuple[float, float, float, float]) -> Dict[Tuple[int, int],
                                                                         _Slot]:
    """Every margin row ``callout`` can use, with its cheapest leader there.

    A row is usable when the label (long form, else the bare id) fits there clear of every
    label, line and reserved area and off the view, and a straight leader from one of the
    group's members reaches it without touching a label or crossing a dimension, extension
    or existing leader line.  Crossing the part's own edges is allowed here - a dense part
    has no other way out - which is what distinguishes the margin from the local search."""
    options: Dict[Tuple[int, int], _Slot] = {}
    for side_index, (column_x, side) in enumerate(sides):
        anchor = "start" if side > 0 else "end"
        targets = sorted(callout.targets, key=lambda t: -side * t.centre[0])[:_MARGIN_TARGETS]
        for row, y in enumerate(rows):
            shoulder = (column_x - side * 0.8, y)
            landing = (shoulder[0] - side * _LEADER_SHOULDER, y)
            for label, penalty in ((callout.text, 0.0), (callout.short, 40.0)):
                box = text_box(label, column_x, y, _TEXT_MM, anchor, "middle", False, 0.0)
                if not _boxes_clear(box, view_box, 0.0) or not svg.label_fits(box):
                    continue
                best: Optional[_Slot] = None
                for target in targets:
                    cx, cy = target.centre
                    dx, dy = landing[0] - cx, landing[1] - cy
                    length = math.hypot(dx, dy)
                    if length < 1e-6 or side * dx <= 0.0:
                        continue
                    start = _ray_loop_point((cx, cy), (dx / length, dy / length), target.loop,
                                            max(target.radius, 0.4))
                    polyline = [start, landing, shoulder]
                    # a near-horizontal leader is the easiest to follow; length breaks ties
                    cost = abs(y - cy) + 0.05 * length + penalty
                    if best is not None and cost >= best.cost:
                        continue
                    if not svg.path_clear(polyline) or not _path_avoids(polyline,
                                                                        svg.reserved):
                        continue
                    best = _Slot(cost, row, polyline, label, column_x, anchor)
                if best is not None:
                    options[(side_index, row)] = best
                    break
    return options


def _place_margin(svg: _Svg, fr: _ViewFrame, callouts: Sequence[_Callout]
                  ) -> Tuple[List[str], List[_Callout]]:
    """Leaders to labels stacked in the margin beside the view: (placed ids, left over).

    The deterministic fallback for a view too dense for the local search.  Labels go in a
    column ``_MARGIN_LEAD`` outside the view's right edge, one per row at ``_MARGIN_PITCH``
    over the whole height of the sheet, so no two can touch.  Each callout is offered every
    row it can legally use (:func:`_margin_options`); callouts are then served in priority
    order (holes, bosses, pads - big pads last) and each takes its cheapest free row, which
    keeps a leader near-horizontal.  Crossings between the new leaders are then removed by
    swapping the rows of crossing pairs while that reduces the total, and anything still
    crossing moves to a free row where it crosses nothing, or goes to the table.  A callout
    that gets no row is listed in the table only and named in the sheet note - never dropped
    silently."""
    view_box = _inflate(fr.paper_box, 0.5)
    sides = [(fr.right + _MARGIN_LEAD, 1.0)]
    first = int(math.ceil((6.0 - fr.top) / _MARGIN_PITCH))
    last = int(math.floor((svg.h - 6.0 - fr.top) / _MARGIN_PITCH))
    rows = [fr.top + k * _MARGIN_PITCH for k in range(first, last + 1)]
    options = {c.ident: _margin_options(svg, fr, c, rows, sides, view_box) for c in callouts}
    by_id = {c.ident: c for c in callouts}

    taken: Dict[Tuple[int, int], str] = {}
    chosen: Dict[str, _Slot] = {}
    chosen_key: Dict[str, Tuple[int, int]] = {}
    for callout in sorted(callouts, key=lambda c: (c.rank, c.ident)):
        free = [(slot.cost, key) for key, slot in options[callout.ident].items()
                if key not in taken]
        if not free:
            continue
        _, key = min(free)
        taken[key] = callout.ident
        chosen[callout.ident] = options[callout.ident][key]
        chosen_key[callout.ident] = key

    def crossings(ident: str, slot: _Slot) -> int:
        return sum(1 for other, o in chosen.items()
                   if other != ident and _polylines_cross(slot.polyline, o.polyline))

    # untangle: swap the rows of a crossing pair whenever that lowers their crossings
    for _ in range(4 * len(chosen) + 4):
        improved = False
        idents = sorted(chosen)
        for i, a in enumerate(idents):
            for b in idents[i + 1:]:
                if not _polylines_cross(chosen[a].polyline, chosen[b].polyline):
                    continue
                ka, kb = chosen_key[a], chosen_key[b]
                sa, sb = options[a].get(kb), options[b].get(ka)
                if sa is None or sb is None:
                    continue
                before = crossings(a, chosen[a]) + crossings(b, chosen[b])
                old_a, old_b = chosen[a], chosen[b]
                chosen[a], chosen[b] = sa, sb
                after = crossings(a, sa) + crossings(b, sb)
                if after < before:
                    chosen_key[a], chosen_key[b] = kb, ka
                    taken[kb], taken[ka] = a, b
                    improved = True
                else:
                    chosen[a], chosen[b] = old_a, old_b
        if not improved:
            break

    # whatever still crosses moves to a free row where it crosses nothing, or to the table
    while True:
        tangled = sorted((crossings(ident, slot), by_id[ident].rank, ident)
                         for ident, slot in chosen.items() if crossings(ident, slot))
        if not tangled:
            break
        _, _, worst = max(tangled)
        del taken[chosen_key[worst]]
        del chosen[worst]
        del chosen_key[worst]
        free = sorted((slot.cost, key) for key, slot in options[worst].items()
                      if key not in taken and crossings(worst, slot) == 0)
        if free:
            key = free[0][1]
            taken[key] = worst
            chosen[worst] = options[worst][key]
            chosen_key[worst] = key

    placed: List[str] = []
    for ident in sorted(chosen, key=lambda i: chosen_key[i]):
        slot = chosen[ident]
        _draw_leader(svg, ident, slot.polyline, slot.label, slot.tx, slot.anchor)
        placed.append(ident)
    return placed, [c for c in callouts if c.ident not in chosen]


def _place_callouts(svg: _Svg, fr: _ViewFrame, callouts: Sequence[_Callout],
                    drawn: Sequence[List[Point]],
                    loops: Sequence[Tuple[str, List[Point]]]) -> Tuple[List[str], List[str],
                                                                       List[str]]:
    """Place every leader; (placed, left to the table, printed over the view - always empty).

    Three deterministic plans, best kept: every leader beside its feature (the local search,
    which reads best and is what a sparse part gets, unchanged); the local search with the
    leftovers led out to a margin column; and every leader in the margin column.  The plan
    that places the most leaders wins; on a tie the local search, then the all-margin column
    (regular beats mixed).  No plan prints a label over the view: a note across the geometry
    hides the very edges it describes, so a callout with nowhere to go is listed in the table
    only and the caller names it in a sheet note."""
    all_loops = [(loop, _bbox_of(loop)) for _, loop in loops]
    all_loops.extend((loop, _bbox_of(loop)) for loop in drawn)
    start = svg.mark()
    local_placed, local_missing = _place_local(svg, fr, callouts, all_loops)
    if not local_missing:
        return local_placed, [], []
    svg.restore(start)
    margin_placed, _ = _place_margin(svg, fr, callouts)
    svg.restore(start)
    _place_local(svg, fr, callouts, all_loops)
    mixed_placed, _ = _place_margin(svg, fr, local_missing)
    mixed_count = len(local_placed) + len(mixed_placed)
    scores = [(len(local_placed), 2, "local"), (len(margin_placed), 1, "margin"),
              (mixed_count, 0, "mixed")]
    best = max(scores)[2]
    svg.restore(start)
    if best == "local":
        placed, missing = _place_local(svg, fr, callouts, all_loops)
    elif best == "margin":
        placed, missing = _place_margin(svg, fr, callouts)
    else:
        placed, missing = _place_local(svg, fr, callouts, all_loops)
        more, missing = _place_margin(svg, fr, missing)
        placed = placed + more
    order = {c.ident: i for i, c in enumerate(callouts)}
    return placed, sorted((c.ident for c in missing), key=order.get), []


def _view_callouts(fr: _ViewFrame, groups: Sequence["Group"], markings: Sequence["Marking"],
                   pool: Optional[Sequence[float]] = None) -> Tuple[
                       List[_Callout], List[Tuple[str, List[Point]]]]:
    """The leaders this view owns, and every feature loop on it in paper coordinates."""
    callouts: List[_Callout] = []
    loops: List[Tuple[str, List[Point]]] = []
    for g in groups:
        targets: List[_Target] = []
        for member in g.members:
            if member.view != fr.view.name:
                continue
            loop = [(fr.px(x), fr.py(y)) for x, y in member.shape.loop]
            loops.append((g.group_id, loop))
            targets.append(_Target((fr.px(member.shape.center[0]),
                                    fr.py(member.shape.center[1])),
                                   fr.s * max(member.shape.size) / 2.0, loop, _bbox_of(loop)))
        if targets:
            area = g.size[0] * g.size[1]
            rank = (0, 0.0) if g.is_hole else ((1, area) if g.kind == "circle" else (2, area))
            callouts.append(_Callout(g.group_id, _group_callout(g, pool), g.group_id, targets,
                                     rank))
    for mark in markings:
        if mark.view != fr.view.name:
            continue
        x0, y0, x1, y1 = mark.bbox
        loop = [(fr.px(x0), fr.py(y0)), (fr.px(x1), fr.py(y0)),
                (fr.px(x1), fr.py(y1)), (fr.px(x0), fr.py(y1))]
        loops.append((mark.ident, loop))
        callouts.append(_Callout(mark.ident, mark.row(), mark.ident,
                                 [_Target((fr.px((x0 + x1) / 2.0), fr.py((y0 + y1) / 2.0)),
                                          fr.s * max(x1 - x0, y1 - y0) / 2.0, loop,
                                          _bbox_of(loop))]))
    return callouts, loops


# --------------------------------------------------------------------------------------
# callouts: one documented grammar, and nothing printed that was not measured
# --------------------------------------------------------------------------------------

_G_NUM = r"[0-9]+(?:\.[0-9]+)?"
_G_SIZE = r"(?:{dia}{n}|{n} x {n}(?: R{n})?)".format(dia=SYM_DIAMETER, n=_G_NUM)

#: The *only* shape a feature callout may take.  ``tests/test_drawing.py`` asserts every
#: callout of every sheet this module can produce matches it, which is what stops a callout
#: from drifting back into prose ("BOSS 7 up 3") one convenience at a time.
CALLOUT_GRAMMAR = re.compile(
    "^(?P<id>[HB][0-9]+)  "
    "(?P<qty>[0-9]+X )?"
    "(?P<feature>(?:BOSS |PAD )?)"
    "(?P<size>{size})"
    "(?P<extent> THRU| {dep}{n}|, {n} HIGH)"
    "(?P<counterbore>(?: {cb}{dia}{n} {dep}{n})?)"
    "(?P<countersink>(?: {cs}{dia}{n} x {n}{deg})?)$".format(
        size=_G_SIZE, n=_G_NUM, dep=SYM_DEPTH, cb=SYM_COUNTERBORE, cs=SYM_COUNTERSINK,
        dia=SYM_DIAMETER, deg=SYM_DEGREE))

#: The two forms a marking row may take.  There is no third: without metadata from the source
#: the generic form is printed, and the content of a marking is never guessed from its shape.
MARKING_GRAMMAR = re.compile(
    "^(?P<id>M[0-9]+)  (?:"
    "MARKING '(?P<label>[^']*)' (?P<textsize>{n}) TEXT (?P<relief>DEBOSSED|EMBOSSED) "
    "(?P<depth>{n})"
    "|SHALLOW RELIEF REGION, (?P<loops>[0-9]+) LOOPS, (?P<deep>{n}) DEEP, "
    "GEOMETRY PER MODEL — NOT A HOLE)$".format(n=_G_NUM))

#: What a measured feature is called in drafting language.  Every entry is a statement about
#: geometry that was measured - round or not, material removed or added - and nothing else:
#: "tapped", "reamed" or "clearance" would be intent this module cannot see.
FEATURE_NOUN: Dict[Tuple[bool, str], str] = {
    (True, "circle"): "HOLE",
    (True, "slot"): "OPENING, RADIUSED CORNERS",
    (True, "rect"): "OPENING, SQUARE CORNERS",
    (True, "profile"): "OPENING, PROFILED",
    (False, "circle"): "BOSS",
    (False, "slot"): "PAD, RADIUSED CORNERS",
    (False, "rect"): "PAD, SQUARE CORNERS",
    (False, "profile"): "PAD, PROFILED",
}


def feature_noun(g: "Group") -> str:
    """What the table calls this feature, including the modifiers folded into it.

    A counterbore folded into its bore stops being a row of its own, so the surviving row has
    to say it is there: "HOLE, COUNTERBORED" is the fact, and the sizes are on the callout
    (and always in ``groups[].counterbore`` / ``groups[].skirts`` in the JSON, which is where
    a countersink's cone angle lives when the sheet cannot state it traceably)."""
    noun = FEATURE_NOUN.get((g.is_hole, g.kind), g.kind.upper())
    words = {"counterbore": "COUNTERBORED", "countersink": "COUNTERSUNK"}
    roles = sorted({role for role, _, _, _ in _modifier_signatures(g)})
    if roles:
        noun += ", " + " AND ".join(words[role] for role in roles)
    return noun


def callout_matches_grammar(text: str) -> bool:
    """Is ``text`` a legal callout or marking row, and free of non-drafting arrows?"""
    if any(symbol in text for symbol in FORBIDDEN_SYMBOLS):
        return False
    return bool(CALLOUT_GRAMMAR.match(text) or MARKING_GRAMMAR.match(text))


def measured_pool(groups: Sequence["Group"], views: Dict[str, ViewData],
                  levels: Sequence[float], bbox: Optional[Sequence[float]] = None) -> List[float]:
    """Every length this result carries - and therefore every length the sheet may print.

    The sheet's own invariant is that no number on it was typed: a value that the machine
    readable result does not contain cannot be traced back to a measurement by anyone, and is
    indistinguishable from an invention.  ``drawing_checks.measured_values`` then re-derives
    the same pool from the JSON and checks the sheet against it independently; this is the
    module holding itself to the rule, not a copy of the gate."""
    values: List[float] = [0.0]
    if bbox:
        values.extend(float(v) for v in bbox)
    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for name, view in views.items():
        box = tuple(round(v, 3) for v in view.bbox)
        boxes[name] = box                                      # type: ignore[assignment]
        values.extend(box)
        values.append(box[2] - box[0])
        values.append(box[3] - box[1])
    ordered = [round(z, 3) for z in levels]
    if ordered:
        base = min(ordered)
        values.extend(ordered)
        values.extend(level - base for level in ordered)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                values.append(abs(b - a))
    for g in groups:
        values.extend([round(g.size[0], 3), round(g.size[1], 3), round(g.corner_r, 3),
                       round(g.depth, 3), round(g.depth_from, 3), round(g.depth_to, 3)])
        # a counterbore or countersink is measured off the parent's own coaxial wall, so its
        # diameter and depth are measurements of this hole and belong in the pool.  Without
        # them the only way to keep the modifier printable would be to leave it standing as a
        # second group at the same positions - a drawing that says six holes for three.
        for _, diameter, low, high in _modifier_signatures(g):
            values.extend([round(diameter, 3), round(low, 3), round(high, 3),
                           round(high - low, 3)])
        box = boxes.get(g.view, (0.0, 0.0, 0.0, 0.0))
        points = [(round(m.shape.center[0], 3), round(m.shape.center[1], 3)) for m in g.members]
        for px, py in points:
            values.extend([px, py, px - box[0], py - box[1], box[2] - px, box[3] - py])
        for i, first in enumerate(points):
            for second in points[i + 1:]:
                values.append(abs(second[0] - first[0]))
                values.append(abs(second[1] - first[1]))
    return values


def _in_pool(value: float, pool: Optional[Sequence[float]]) -> bool:
    if pool is None:
        return True
    return any(abs(value - m) <= 0.01 for m in pool)


def _extra_callout(g: "Group", pool: Optional[Sequence[float]]) -> str:
    """The counterbore / countersink segment of a callout, when the measurement supports it.

    Both are measured facts - a counterbore is a coaxial wall of a larger radius, a countersink
    is the sloped skirt at a hole's mouth - but a number the result does not carry cannot be
    traced, so a segment whose values are not in ``pool`` is left off rather than printed."""
    if not g.is_hole or not g.members:
        return ""
    record = g.members[0].counterbore
    if isinstance(record, dict):
        diameter = record.get("diameter")
        depth = record.get("depth")
        if diameter is not None and depth is not None \
                and _in_pool(float(diameter), pool) and _in_pool(float(depth), pool):
            return " {}{}{} {}{}".format(SYM_COUNTERBORE, SYM_DIAMETER, _dimnum(float(diameter)),
                                         SYM_DEPTH, _dimnum(float(depth)))
    for skirt in g.members[0].skirts:
        diameter = g.size[0] + 2.0 * skirt.outward
        included = 2.0 * skirt.angle_deg
        if g.kind == "circle" and _in_pool(diameter, pool) and _in_pool(included, pool):
            return " {}{}{} x {}{}".format(SYM_COUNTERSINK, SYM_DIAMETER, _dimnum(diameter),
                                           _dimnum(included), SYM_DEGREE)
    return ""


def _group_callout(g: "Group", pool: Optional[Sequence[float]] = None) -> str:
    """The leader note for one group, in ISO 129-1 / ASME Y14.5 symbols.

    ``↓``/``↑`` are arrows and were never a depth symbol; ``⤓`` is.  A raised
    feature has no depth at all, so it is called a boss or a pad and its height is stated as a
    height - "↑19" was not a callout, it was a number with an arrow in front of it."""
    qty = "{}X ".format(len(g.members)) if len(g.members) > 1 else ""
    feature = "" if g.is_hole else ("BOSS " if g.kind == "circle" else "PAD ")
    if g.thru:
        extent = " THRU"
    elif g.is_hole:
        extent = " {}{}".format(SYM_DEPTH, _dimnum(g.depth))
    else:
        extent = ", {} HIGH".format(_dimnum(g.depth))
    return "{}  {}{}{}{}{}".format(g.group_id, qty, feature, g.label, extent,
                                   _extra_callout(g, pool))


def _level_chain_width(levels: Sequence[float]) -> float:
    """Paper width the level stack needs, so the layout can reserve it."""
    return _DIM_STEP * _level_chain_count(levels) + 6


def _level_chain_count(levels: Sequence[float]) -> int:
    inner = max(len(levels) - 2, 0)          # levels strictly between base and top
    return 1 + (inner if inner <= _MAX_LEVEL_CHAIN else 0)


def _draw_level_chain(svg: _Svg, fr: _ViewFrame, levels: Sequence[float], x_line: float) -> None:
    """Heights of the top view's z-levels beside the right view (front/right share z).

    Stacked from the base rather than chained segment-to-segment: a 1.5 mm step would not fit
    its own text, while every stacked line is 7 mm from its neighbour."""
    inner = list(levels[1:-1]) if _level_chain_count(levels) > 1 else []
    for i, level in enumerate(inner):
        _dim_v(svg, fr, fr.y0, level, x_line + _DIM_STEP * i, fr.x1, fr.x1,
               short_below=True)
    _dim_v(svg, fr, fr.y0, fr.y1, x_line + _DIM_STEP * len(inner), fr.x1, fr.x1)


def _depth_text(g: "Group") -> str:
    """The extent column of the table, in drafting language: an axis, a range, a depth.

    "THRU, Z 0 TO 4" was self-contradictory - the bore goes through the 8 mm plate and 0 TO 4
    was only the part of it below the counterbore - so a through feature states its range as
    the span it passes through, in brackets, and never as a second bounded depth."""
    axis = "XYZ"[VIEWS[g.view]["depth_axis"]]
    if g.thru:
        return "THRU ({} {} TO {})".format(axis, _dimnum(g.depth_from), _dimnum(g.depth_to))
    return "{} {} TO {} ({})".format(axis, _dimnum(g.depth_from), _dimnum(g.depth_to),
                                     _dimnum(g.depth))


def _position_cells(g: "Group", frames: Dict[str, _ViewFrame]) -> List[str]:
    """Every member's position, as a distance from the drawn origin of its own view.

    One frame for the whole sheet (W6): the table used to print model coordinates while the
    view printed distances from a bbox corner, so the same hole was ``(3.5, 3.5)`` in one place
    and ``8 / 13.7`` in the other.  These are formatted by ``_dimnum`` exactly as the ordinate
    values are, so a table entry and its dimension are the same characters."""
    frame = frames.get(g.view)
    cells: List[str] = []
    for member in g.members:
        cx, cy = member.shape.center
        if frame is None:
            cells.append("({}, {})".format(_dimnum(cx), _dimnum(cy)))
        else:
            cells.append("({}, {})".format(_dimnum(frame.dx(cx)), _dimnum(frame.dy(cy))))
    return cells


def _wrap_cells(cells: Sequence[str], width: float, size: float) -> List[str]:
    """Pack coordinate cells into as few lines as fit ``width`` paper mm."""
    lines: List[str] = []
    current = ""
    for cell in cells:
        candidate = cell if not current else current + "  " + cell
        if current and text_width_mm(candidate, size, True) > width:
            lines.append(current)
            current = cell
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


_TABLE_SIZE = 2.3
_TABLE_ROW = 4.6
_TABLE_PAD = 1.5
_TABLE_HEADERS: Tuple[str, ...] = ("CALLOUT", "FEATURE", "EXTENT ALONG DEPTH AXIS (mm)",
                                   "POSITION FROM ORIGIN  X, Y (mm)")


def table_plan(groups: Sequence["Group"], markings: Sequence["Marking"],
               frames: Dict[str, _ViewFrame], width: float,
               pool: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Column widths and wrapped rows for the feature table, measured from its own content.

    The table is not decoration: it is where a position goes when the view cannot carry its
    ordinate, and it is where the callout text lives for a group whose leader could not be
    placed.  So it is sized from what it has to say, not from fixed offsets that silently
    clip."""
    body: List[Dict[str, Any]] = []
    for g in groups:
        body.append({
            "id": g.group_id,
            "cells": [_group_callout(g, pool), feature_noun(g), _depth_text(g)],
            "positions": _position_cells(g, frames),
        })
    fixed = [max([text_width_mm(_TABLE_HEADERS[i], 2.4)]
                 + [text_width_mm(row["cells"][i], _TABLE_SIZE, True) for row in body]
                 or [0.0]) + 2 * _TABLE_PAD
             for i in range(3)]
    rest = width - sum(fixed)
    if rest < 34.0:                       # give the coordinate column room by taking it back
        rest = 34.0
        fixed = [w * (width - rest) / max(sum(fixed), 1e-9) for w in fixed]
    offsets = [0.0, fixed[0], fixed[0] + fixed[1], fixed[0] + fixed[1] + fixed[2]]
    rows: List[Dict[str, Any]] = []
    for row in body:
        lines = _wrap_cells(row["positions"], rest - 2 * _TABLE_PAD, _TABLE_SIZE)
        rows.append({"kind": "group", "id": row["id"], "cells": row["cells"], "lines": lines})
    for mark in markings:
        rows.append({"kind": "marking", "id": mark.ident, "cells": [mark.row()], "lines": [""]})
    # Every row is kept.  Which of them sheet 1 can carry is a layout decision taken with the
    # scale (``split_table_rows``); the rest continue on the next sheet, never off the drawing.
    lines = sum(_row_lines(row) for row in rows)
    return {"offsets": offsets, "widths": fixed + [rest], "rows": rows,
            "height": _TABLE_ROW * (1 + lines), "lines": lines, "overflow": [],
            "groups": len(body), "markings": len(markings)}


#: Paper mm a bottom band keeps clear above the footer.
_BAND_PAD = 6.0
#: Paper mm the front row needs under its baseline: the view label at 5.5, the width
#: dimension at ``_DIM_STEP + 4`` with its arrowheads, and a clear gap to the band below.
_BELOW_FRONT = 15.0


def _table_sheet_note(pages: Sequence[Sequence[Mapping[str, Any]]],
                      plan: Mapping[str, Any]) -> Optional[str]:
    """The sheet-1 note that says where the feature table went, when it is not all here."""
    if len(pages) < 2 or pages[0]:
        return None
    where = "SHEET 2" if len(pages) == 2 else "SHEETS 2-{}".format(len(pages))
    return "FEATURE TABLE ON {} OF {} ({} feature group(s){}).".format(
        where, len(pages), plan.get("groups", 0),
        ", {} marking(s)".format(plan["markings"]) if plan.get("markings") else "")


def _arrangement_facts(option: Mapping[str, Any]) -> Dict[str, Any]:
    pages = option["pages"]
    return {"stepped": option["stepped"],
            "tolerance_block": "left" if option["tol_left"] else "right",
            "table_on_sheet_1": bool(pages[0]),
            "table_position": ("sheet 2" if not pages[0] and len(pages) > 1 else
                               "beside the views" if option.get("side") else
                               "under the views"),
            "sheets": len(pages)}


#: A sheet at this scale or larger is read without help, so a drawing that fits on fewer
#: sheets at it is not split up to be drawn larger still.
LEGIBLE_SCALE = 1.0


def choose_arrangement(options: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The arrangement to draw, from candidates in order of preference (``options[0]`` is the
    original layout).

    Among arrangements on the same number of sheets the first is kept unless a later one
    fits at a strictly larger scale.  Across sheet counts, fewer sheets win unless they are
    below ``LEGIBLE_SCALE`` *and* more sheets buy a larger scale: one 1:1 sheet beats a 2:1
    sheet with its table on sheet 2, but a 1:5 sheet does not beat a 1:2 one."""
    best: Dict[int, Mapping[str, Any]] = {}
    for option in options:
        count = len(option["pages"])
        held = best.get(count)
        if held is None or (option["fits"] and (not held["fits"]
                                                or option["scale"] > held["scale"])):
            best[count] = option
    chosen: Optional[Mapping[str, Any]] = None
    for count in sorted(best):
        option = best[count]
        if chosen is None:
            chosen = option
        elif option["fits"] and (not chosen["fits"] or (
                chosen["scale"] < LEGIBLE_SCALE and option["scale"] > chosen["scale"])):
            chosen = option
    assert chosen is not None
    return chosen


def _row_lines(row: Mapping[str, Any]) -> int:
    return max(len(row["lines"]), 1)


def split_table_rows(rows: Sequence[Dict[str, Any]], first: int,
                     following: int) -> List[List[Dict[str, Any]]]:
    """Cut the table rows into sheets, in order: ``first`` lines on sheet 1, ``following`` on
    every continuation sheet.  A sheet that continues spends one line saying so.

    A row is never split and never skipped: a row taller than a whole sheet still gets a sheet
    of its own, so ``sum(pages, []) == rows`` always holds - no feature leaves the drawing."""
    pages: List[List[Dict[str, Any]]] = [[]]
    used = 0
    capacity = first
    for index, row in enumerate(rows):
        need = _row_lines(row)
        owed = 1 if index + 1 < len(rows) else 0
        if used + need + owed > capacity and (pages[-1] or capacity < following):
            pages.append([])
            used, capacity = 0, following
        pages[-1].append(row)
        used += need
    return pages


def table_page(plan: Mapping[str, Any], rows: Sequence[Dict[str, Any]],
               continued_on: Optional[int] = None,
               continued_from: Optional[int] = None) -> Dict[str, Any]:
    """The part of a planned table that one sheet draws."""
    lines = sum(_row_lines(row) for row in rows)
    page = dict(plan)
    page.update(rows=list(rows), continued_on=continued_on, continued_from=continued_from,
                height=_TABLE_ROW * (1 + lines + (1 if continued_on else 0)), lines=lines)
    return page


def _draw_table(svg: _Svg, x: float, y: float, width: float, plan: Dict[str, Any]) -> float:
    """Draw the planned feature table; returns the paper y just below it."""
    offsets = plan["offsets"]
    svg.rect(x, y, width, _TABLE_ROW, fill=_TABLE_BG, width=0.25)
    for dx, name in zip(offsets, _TABLE_HEADERS):
        svg.text(x + dx + _TABLE_PAD, y + _TABLE_ROW - 1.3, name, size=2.4, bold=True)
    yy = y + _TABLE_ROW
    last_group_y = yy
    for row in plan["rows"]:
        if row["kind"] == "marking":
            svg.text(x + _TABLE_PAD, yy + _TABLE_ROW - 1.3, row["cells"][0], size=_TABLE_SIZE,
                     mono=True)
            svg.line(x, yy + _TABLE_ROW, x + width, yy + _TABLE_ROW, width=0.15, color="#999")
            yy += _TABLE_ROW
            continue
        for dx, cell in zip(offsets, row["cells"]):
            svg.text(x + dx + _TABLE_PAD, yy + _TABLE_ROW - 1.3, cell, size=_TABLE_SIZE,
                     mono=True)
        for index, line in enumerate(row["lines"]):
            svg.text(x + offsets[3] + _TABLE_PAD, yy + _TABLE_ROW - 1.3 + index * _TABLE_ROW,
                     line, size=_TABLE_SIZE, mono=True, color=_DIM)
        yy += _TABLE_ROW * max(len(row["lines"]), 1)
        svg.line(x, yy, x + width, yy, width=0.15, color="#999")
        last_group_y = yy
    if plan.get("continued_on"):
        svg.text(x + _TABLE_PAD, yy + _TABLE_ROW - 1.3,
                 "TABLE CONTINUES ON SHEET {}".format(plan["continued_on"]),
                 size=_TABLE_SIZE, mono=True, bold=True)
        yy += _TABLE_ROW
    svg.rect(x, y, width, yy - y, width=0.3)
    for dx in offsets[1:]:
        svg.line(x + dx, y, x + dx, last_group_y, width=0.15, color="#999")
    return yy


# --------------------------------------------------------------------------------------
# general tolerances: carried and displayed, never originated
# --------------------------------------------------------------------------------------

#: What a tolerance block looks like when nothing declared one.  PhiMech cannot choose a
#: tolerance: +/-0.1 against +/-0.3 is a decision about fit and function that the geometry does
#: not contain, and a machinist holds whatever the drawing says.  So the only honest output in
#: the absence of a declaration is the word UNSPECIFIED, and the sheet keeps its banner.
UNSPECIFIED = "UNSPECIFIED"
#: Literally what the sheet prints when a *declaration was made and could not be used*.  That
#: is a different fact from "nothing was declared", and collapsing the two is the quiet
#: failure: an author who typos a tolerance would get a sheet reading UNSPECIFIED and no
#: indication their intent had been discarded, then assume the tolerance is on the drawing.
REJECTED_PREFIX = "GENERAL TOLERANCES: UNSPECIFIED \u2014 DECLARATION REJECTED"
_TOLERANCE_UNSPECIFIED: Dict[str, Any] = {
    "specified": False, "provenance": "none", "standard": None, "standard_class": None,
    "linear": (), "angular": (), "broken_edges": (), "finish": None,
    "applies_from_mm": None, "notes": (), "conflicts": (), "rejected": None,
}


def _get(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, Mapping):
        return block.get(name, default)
    return getattr(block, name, default)


def _bands(raw: Any) -> Tuple[Tuple[float, Optional[float], float], ...]:
    """Resolver bands as ``(low, high-or-None, plus_minus)``.

    ``None`` rather than ``float("inf")`` for an open-ended band: infinity is not valid JSON,
    and a reader of the sidecar must not be able to mistake it for a number."""
    rows: List[Tuple[float, Optional[float], float]] = []
    for row in raw or ():
        values = list(row)
        if len(values) != 3:
            continue
        low, high, plus_minus = float(values[0]), values[1], float(values[2])
        top = None if high is None or not math.isfinite(float(high)) else float(high)
        rows.append((low, top, plus_minus))
    return tuple(rows)


def resolve_tolerance_block(ir: Any = None, source_manifest: Any = None) -> Dict[str, Any]:
    """The tolerance block for this sheet, from ``phimech.ir.tolerance`` when it is installed.

    The resolver is the single source of truth and lives with the IR, because a declaration
    belongs to the design, not to the drawing.  This module only *presents* it.  Three
    outcomes, and they are deliberately distinguishable on the sheet and in the result:

    * nothing declared -> ``specified: false``, ``rejected: null``, the sheet says UNSPECIFIED;
    * a declaration the resolver refuses (``ValueError``: a malformed band, a class plus an
      overriding table, an empty object) -> ``specified: false`` **and** ``rejected: <reason>``,
      which the sheet prints.  Swallowing it would be safe but dishonest;
    * the resolver missing or faulting -> ``rejected`` says so, again on the sheet.

    In no case is a class chosen here, and there is no default anywhere on this path."""
    try:
        from phimech.ir.tolerance import resolve_tolerance
    except ImportError as exc:
        record = dict(_TOLERANCE_UNSPECIFIED)
        if ir is None and source_manifest is None:
            return record
        record["rejected"] = "phimech.ir.tolerance is not installed: {}".format(exc)
        return record
    try:
        block = resolve_tolerance(ir=ir, source_manifest=source_manifest)
    except ValueError as exc:
        record = dict(_TOLERANCE_UNSPECIFIED)
        record["rejected"] = str(exc)
        return record
    except Exception as exc:                # noqa: BLE001 - a fault is reported, not hidden
        record = dict(_TOLERANCE_UNSPECIFIED)
        record["rejected"] = "{}: {}".format(type(exc).__name__, exc)
        return record
    specified = bool(_get(block, "specified", False))
    provenance = str(_get(block, "provenance", "none") or "none")
    finish = _get(block, "finish")
    if not specified or provenance == "none":
        record = dict(_TOLERANCE_UNSPECIFIED)
        record["finish"] = str(finish) if finish else None
        record["notes"] = tuple(str(n) for n in (_get(block, "notes") or ()))
        record["conflicts"] = tuple(str(c) for c in (_get(block, "conflicts") or ()))
        return record
    return {
        "specified": True,
        "provenance": provenance,
        "standard": str(_get(block, "standard")) if _get(block, "standard") else None,
        "standard_class": (str(_get(block, "standard_class"))
                           if _get(block, "standard_class") else None),
        "linear": _bands(_get(block, "linear")),
        "angular": _bands(_get(block, "angular")),
        "broken_edges": _bands(_get(block, "broken_edges")),
        "finish": str(finish) if finish else None,
        "applies_from_mm": _get(block, "applies_from_mm"),
        "notes": tuple(str(n) for n in (_get(block, "notes") or ())),
        "conflicts": tuple(str(c) for c in (_get(block, "conflicts") or ())),
        "rejected": None,
    }


def edge_break_tolerance(tolerance: Mapping[str, Any], size_mm: float) -> Optional[float]:
    """The general +- for a chamfer or external radius of this size.

    Delegated to the resolver's own ``broken_edge_for`` rather than re-implemented, because the
    answer is not the linear one: a 1 mm chamfer under ISO 2768-m is +-0.2 from table 2, where
    the linear table would wrongly give it +-0.1.  ``None`` means no band covers that size, and
    that is an answer - the edge needs its own tolerance - not a reason to reach for a number.
    """
    bands = tolerance.get("broken_edges") or ()
    if not bands:
        return None
    try:
        from phimech.ir.tolerance import UNBOUNDED, ToleranceBlock
    except ImportError:
        return None
    rows = tuple((float(low), UNBOUNDED if high is None else float(high), float(pm))
                 for low, high, pm in bands)
    block = ToleranceBlock(specified=True, provenance=str(tolerance.get("provenance") or "none"),
                           broken_edges=rows,
                           applies_from_mm=tolerance.get("applies_from_mm"))
    return block.broken_edge_for(size_mm)


def _band_line(low: float, high: Optional[float], plus_minus: float, unit: str) -> str:
    """One row of a printed tolerance table; an open-ended band says so instead of "inf"."""
    where = ("over {}".format(_dimnum(low)) if high is None
             else "over {} up to {}".format(_dimnum(low), _dimnum(high)))
    return "   {:<22} \u00b1{}{}".format(where, _dimnum(plus_minus), unit)


def tolerance_lines(tolerance: Mapping[str, Any],
                    edges: Sequence[float] = ()) -> List[Tuple[str, bool]]:
    """(text, is_heading) for the GENERAL TOLERANCES region.

    A declared class is printed *and expanded*: "ISO 2768-m" alone means nothing to a reader
    without the standard on the bench, and a tolerance nobody can read is a tolerance nobody
    applies.  A standard this build cannot expand is printed by name with no table - inventing
    the table of a class we do not know would be worse than printing none."""
    rejected = tolerance.get("rejected")
    if not tolerance.get("specified"):
        lines = [(REJECTED_PREFIX if rejected else
                  "GENERAL TOLERANCES: {}".format(UNSPECIFIED), True)]
        if rejected:
            lines.append(("Reason: {}".format(rejected), False))
            lines.append(("A tolerance was declared and could not be used. None is assumed; "
                          "fix the declaration and redraw.", False))
        lines.append(("SURFACE FINISH: {}".format(tolerance.get("finish") or UNSPECIFIED),
                      False))
        if not rejected:
            lines.append(("No general tolerance was declared by the source. None is assumed.",
                          False))
        return lines
    standard = tolerance.get("standard")
    lines = [("GENERAL TOLERANCES  {}".format(standard or "PER TABLE BELOW"), True)]
    if tolerance.get("linear"):
        lines.append(("LINEAR (mm)", False))
        for low, high, plus_minus in tolerance["linear"]:
            lines.append((_band_line(low, high, plus_minus, ""), False))
    if tolerance.get("broken_edges"):
        lines.append(("BROKEN EDGES \u2014 chamfer height / external radius (mm)", False))
        for low, high, plus_minus in tolerance["broken_edges"]:
            lines.append((_band_line(low, high, plus_minus, ""), False))
        for size in sorted({round(value, 1) for value in edges if value > 0.0}):
            found = edge_break_tolerance(tolerance, size)
            if found is not None:
                lines.append(("   {} mm chamfer/radius here: \u00b1{}".format(
                    _dimnum(size), _dimnum(found)), False))
    if tolerance.get("angular"):
        lines.append(("ANGULAR (shorter side, mm)", False))
        for low, high, plus_minus in tolerance["angular"]:
            lines.append((_band_line(low, high, plus_minus, SYM_DEGREE), False))
    if standard and not (tolerance.get("linear") or tolerance.get("angular")
                         or tolerance.get("broken_edges")):
        lines.append(("This build does not hold the table for {}; the class is carried "
                      "verbatim and is not expanded here.".format(standard), False))
    lines.append(("SURFACE FINISH: {}".format(tolerance.get("finish") or UNSPECIFIED), False))
    lines.append(("Source: {}".format(tolerance.get("provenance")), False))
    return lines


def tolerance_notes(tolerance: Mapping[str, Any]) -> List[str]:
    """The resolver's own prose, for the notes block: how it expanded, and what disagreed.

    A conflict between two declaring documents is exactly the failure mode a silent pick
    hides, so it is printed rather than resolved out of sight."""
    notes = ["General tolerance: {}".format(note) for note in tolerance.get("notes") or ()]
    notes.extend("GENERAL TOLERANCE CONFLICT: {}".format(conflict)
                 for conflict in tolerance.get("conflicts") or ())
    if tolerance.get("rejected"):
        notes.append("A general tolerance was DECLARED AND REJECTED: {}. None is assumed and "
                     "none was substituted.".format(tolerance["rejected"]))
    return notes


#: The tolerance region gets columns rather than height once it passes this, because an
#: expanded ISO 2768 table is ~23 lines and stacking them all would push the views down to a
#: scale at which nobody can read the part the tolerance applies to.
_TOL_MAX_H = 40.0
_TOL_MAX_COLUMNS = 3
#: The compact (under-the-views) tolerance block: at most this many columns, none narrower.
_TOL_COMPACT_COLUMNS = 5
_TOL_MIN_COLUMN = 44.0
_TOL_LINE = 4.0


def wrap_tolerance_lines(lines: Sequence[Tuple[str, bool]],
                         width: float) -> List[Tuple[str, bool]]:
    """Wrap the region's prose to the block it has to fit; a clipped reason is no reason."""
    wrapped: List[Tuple[str, bool]] = []
    for text, heading in lines:
        size = 2.5 if heading else 2.2
        for index, line in enumerate(_wrap_text(text, width, size, mono=not heading)):
            wrapped.append((line if index == 0 else "   " + line, heading and index == 0))
    return wrapped


def tolerance_block_layout(lines: Sequence[Tuple[str, bool]], width: float,
                           compact: bool = False) -> Tuple[List[Tuple[str, bool]], int, float]:
    """(wrapped lines, column count, height) for the GENERAL TOLERANCES region.

    ``compact`` is for the block laid out under the views, where every millimetre of height
    is scale: it takes the column count (up to ``_TOL_COMPACT_COLUMNS``) that makes it least
    tall, provided no column gets narrower than ``_TOL_MIN_COLUMN``."""
    if compact:
        best: Optional[Tuple[List[Tuple[str, bool]], int, float]] = None
        for count in range(1, _TOL_COMPACT_COLUMNS + 1):
            if count > 1 and width / count < _TOL_MIN_COLUMN:
                break
            lines_c = wrap_tolerance_lines(lines, width / count - 4.0)
            height = 3.0 + _TOL_LINE * int(math.ceil(len(lines_c) / float(count))) + 1.5
            if best is None or height < best[2] - 1e-9:
                best = (lines_c, count, height)
        if best is not None:
            return best
    wrapped: List[Tuple[str, bool]] = []
    columns = 1
    for columns in range(1, _TOL_MAX_COLUMNS + 1):
        wrapped = wrap_tolerance_lines(lines, width / columns - 4.0)
        rows = int(math.ceil(len(wrapped) / float(columns)))
        if 3.0 + _TOL_LINE * rows + 1.5 <= _TOL_MAX_H:
            break
    rows = int(math.ceil(len(wrapped) / float(columns)))
    return wrapped, columns, 3.0 + _TOL_LINE * rows + 1.5


def tolerance_block_height(lines: Sequence[Tuple[str, bool]], columns: int = 1) -> float:
    rows = int(math.ceil(len(lines) / float(max(columns, 1))))
    return 3.0 + _TOL_LINE * rows + 1.5


def _draw_tolerance_block(svg: _Svg, x: float, y: float, w: float,
                          lines: Sequence[Tuple[str, bool]], columns: int = 1) -> None:
    height = tolerance_block_height(lines, columns)
    svg.rect(x, y, w, height, width=0.4)
    rows = int(math.ceil(len(lines) / float(max(columns, 1))))
    column_w = w / max(columns, 1)
    for index, (text, heading) in enumerate(lines):
        column, row = index // rows, index % rows
        svg.text(x + 2.0 + column * column_w, y + 3.0 + row * _TOL_LINE + 2.8, text,
                 size=2.5 if heading else 2.2, bold=heading, mono=not heading)


# --------------------------------------------------------------------------------------
# ISO 7200 title block
# --------------------------------------------------------------------------------------

_TB_ROW = 4.8
_TB_TITLE_H = 8.0
_TB_BANNER_H = 6.0
_TB_PROJECTION_H = 9.0
BANNER_TEXT = "INSPECTION DRAWING — NOT FOR MANUFACTURE"
#: An approval field with nobody in it.  Printed as a visible empty marker rather than left
#: blank, so a missing signature reads as missing and not as an oversight - and it is *never*
#: filled in by this module.  An "APPROVED BY" carrying a name is an attestation that a person
#: signed the drawing off; deriving one from a config default, an environment variable, a git
#: identity or the OS user would be forging that person's approval.  The only way a name
#: reaches this field is an explicit declaration for this drawing, on the source manifest,
#: by the same route as material and tolerance, with its provenance recorded in the result.
UNSIGNED = "———"
#: Height of the attribution footer strip, paper mm.
_FOOTER_H = 5.0
#: The title block fields this sheet must carry.  "0 empty required fields" is checkable only
#: against a list, so here it is; every one is filled from a measurement, from the generator's
#: own facts, or with the literal ``UNSPECIFIED``.
REQUIRED_TITLE_FIELDS: Tuple[str, ...] = (
    "MATERIAL", "SURFACE FINISH", "REVISION", "DATE", "SCALE", "UNITS", "SHEET", "SHEET SIZE",
    "DRAWN BY", "APPROVED BY", "GENERATOR", "RENDERER", "SOURCE FILE", "SOURCE SHA-256",
)


def _fit(text: str, room: float, size: float, mono: bool) -> str:
    """Trim free text to the cell it has to live in; numbers are never passed through here."""
    if text_width_mm(text, size, mono) <= room:
        return text
    while text and text_width_mm(text + "…", size, mono) > room:
        text = text[:-1]
    return text + "…"


def title_block_height(fields: Sequence[Tuple[str, str]], banner: bool) -> float:
    rows = (len(fields) + 1) // 2
    return ((_TB_BANNER_H if banner else 0.0) + _TB_TITLE_H + _TB_ROW * rows
            + _TB_PROJECTION_H)


def _draw_projection_symbol(svg: _Svg, x: float, y: float, height: float) -> None:
    """The ISO 128 third-angle symbol: the frustum's end view left, its profile right.

    A projection symbol is not decoration - a reader who assumes first angle reads every side
    view on the wrong side of the part - so it is drawn, and the words are printed next to it
    for anyone who does not read the symbol."""
    r_out = height / 2.0
    r_in = r_out / 2.0
    cx, cy = x + r_out, y + r_out
    svg.parts.append('<g class="projection">')
    for radius in (r_out, r_in):
        svg.parts.append(
            '<circle cx="{:.3f}" cy="{:.3f}" r="{:.2f}" fill="none" stroke="{}" '
            'stroke-width="0.25"/>'.format(cx, cy, radius, _INK))
    left = x + 2.0 * r_out + 1.5
    right = left + height * 1.2
    svg.poly([(left, cy - r_in), (right, cy - r_out), (right, cy + r_out), (left, cy + r_in)],
             width=0.25)
    svg.parts.append("</g>")


def _draw_title_block(svg: _Svg, x: float, y: float, w: float, *, title: str, drawing_no: str,
                      fields: Sequence[Tuple[str, str]], banner: Optional[str]) -> float:
    """ISO 7200 data fields, in a block, with nothing left blank; returns its height."""
    height = title_block_height(fields, banner is not None)
    svg.rect(x, y, w, height, width=0.5)
    yy = y
    if banner is not None:
        svg.rect(x, yy, w, _TB_BANNER_H, fill="#fee2e2", color=_WARN, width=0.3)
        svg.text(x + w / 2.0, yy + _TB_BANNER_H - 1.7, banner, size=2.8, anchor="middle",
                 bold=True, color=_WARN)
        yy += _TB_BANNER_H
    svg.text(x + 2.0, yy + _TB_TITLE_H - 2.2, _fit(title, w * 0.62, 3.6, False), size=3.6,
             bold=True)
    svg.text(x + w - 2.0, yy + _TB_TITLE_H - 2.4, _fit(drawing_no, w * 0.33, 2.6, True),
             size=2.6, anchor="end", mono=True)
    yy += _TB_TITLE_H
    svg.line(x, yy, x + w, yy, width=0.3)
    grid_top = yy
    half = w / 2.0
    for index, (label, value) in enumerate(fields):
        column, row = index % 2, index // 2
        cx = x + column * half
        cy = grid_top + row * _TB_ROW
        svg.text(cx + 1.6, cy + _TB_ROW - 1.4, label, size=1.9, color="#555")
        svg.text(cx + 25.0, cy + _TB_ROW - 1.4,
                 _fit(value, half - 27.0, 2.3, True), size=2.3, mono=True)
    rows = (len(fields) + 1) // 2
    strip = grid_top + rows * _TB_ROW
    svg.line(x, strip, x + w, strip, width=0.3)
    svg.line(x + half, grid_top, x + half, strip, width=0.15, color="#999")
    _draw_projection_symbol(svg, x + 2.0, strip + 1.6, 5.6)
    svg.text(x + 24.0, strip + 5.6, "THIRD ANGLE PROJECTION", size=2.2, baseline="middle")
    svg.text(x + w - 2.0, strip + 5.6, "ALL DIMENSIONS IN MILLIMETRES", size=2.2, anchor="end",
             baseline="middle")
    return height


def attribution_footer(signature: str, date: str, source_name: str,
                       source_sha256: Optional[str]) -> str:
    """The provenance line printed on every sheet, in every case.

    Unlike material or an approver this is not a claim about the part or about a person: it is
    a fact about where the sheet came from, and it is what lets somebody holding a print tie it
    back to the exact geometry - the file, and the first bytes of its digest."""
    digest = (source_sha256 or "")[:16] or UNSPECIFIED
    return "GENERATED BY {} ON {}  —  SOURCE {}  sha256 {}".format(
        signature, date, source_name, digest)


def _draw_footer(svg: _Svg, x: float, y: float, width: float, text: str) -> None:
    svg.line(x, y - 1.6, x + width, y - 1.6, width=0.2, color="#999")
    svg.text(x, y + 1.8, _fit(text, width, 2.2, False), size=2.2, color="#555")


def manifest_facts(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Material, revision, part name and marking text - verbatim from the source or ``None``.

    Two shapes are understood because two exist: the SolidWorks handoff ``source.json``
    (``component.material``) and a PhiMech design IR (``rev``, ``enclosure.text``).  Anything
    else yields nothing, which prints as ``UNSPECIFIED``: a field this function guessed would
    be indistinguishable on the sheet from one a human declared."""
    facts: Dict[str, Any] = {"material": None, "revision": None, "part_name": None,
                             "title": None, "finish": None, "text": None,
                             "approved_by": None, "approval_provenance": None}
    if not isinstance(manifest, Mapping):
        return facts
    component = manifest.get("component")
    if isinstance(component, Mapping):
        facts["material"] = component.get("material") or None
        facts["revision"] = component.get("revision") or None
        facts["part_name"] = component.get("name") or None
        facts["finish"] = component.get("finish") or None
    for where, holder in (("component.approved_by", component),
                          ("approval.approved_by", manifest.get("approval")),
                          ("approved_by", manifest)):
        if not isinstance(holder, Mapping):
            continue
        name = holder.get("approved_by")
        if name:
            facts["approved_by"] = str(name)
            facts["approval_provenance"] = where
            break
    manufacturing = manifest.get("manufacturing")
    if isinstance(manufacturing, Mapping):
        # a PhiMech design IR: `manufacturing.material` is declared by the author, exactly as
        # `component.material` is on a handoff manifest (finish reaches the sheet through
        # `resolve_tolerance_block`, which reads the same block)
        facts["material"] = facts["material"] or (manufacturing.get("material") or None)
    facts["revision"] = facts["revision"] or (manifest.get("rev") or None)
    facts["title"] = manifest.get("title") or None
    facts["part_name"] = facts["part_name"] or (manifest.get("project") or None)
    enclosure = manifest.get("enclosure")
    if isinstance(enclosure, Mapping) and isinstance(enclosure.get("text"), Mapping):
        facts["text"] = dict(enclosure["text"])
    return facts


def _scale_text(scale: float) -> str:
    """ISO 5455 notation: an enlargement is ``n:1``, a reduction ``1:n``."""
    if scale >= 1.0:
        return "{}:1".format(_num(scale))
    return "1:{}".format(_num(1.0 / scale))


def sheet_notes(scale: float, views: Dict[str, ViewData], markings: Sequence["Marking"],
                table_only: Sequence[str],
                tolerance: Optional[Mapping[str, Any]] = None,
                table_note: Optional[str] = None) -> List[str]:
    """The drafting notes.  Not a changelog, not a disclaimer about the pipeline."""
    notes = [
        "ALL DIMENSIONS IN MILLIMETRES. THIRD ANGLE PROJECTION. SCALE {} ON A3.".format(
            _scale_text(scale)),
        "Positions are ordinate dimensions from the drawn origin at the lower-left corner of "
        "each view; the table is in the same frame.",
        "Thick = outline. Thin = section edges at measured levels. Dashed = hidden feature "
        "edges. Chain = centre lines. Blue = dimensions.",
        "Dimensions are measured from a tessellated mesh: coordinates resolved to {:g} mm, "
        "feature sizes grouped within {:g} mm, values printed to 0.01 mm. This is a "
        "MEASUREMENT UNCERTAINTY, NOT a manufacturing tolerance.".format(
            10 ** -COORD_DECIMALS, SIZE_QUANTUM),
    ]
    slivers = [s for v in views.values() for s in v.slivers]
    if slivers:
        notes.append(
            "{} section(s) suppressed as modelling slivers (under {} mm2 and {} mm deep, or "
            "narrower than {} mm): they are not features and carry no dimension.".format(
                len(slivers), _num(SLIVER_AREA_MM2), _num(SLIVER_DEPTH_MM),
                _num(SLIVER_WIDTH_MM)))
    if markings:
        notes.append(
            "{} shallow relief region(s) are listed as markings, not holes: geometry per the "
            "model, no dimension is quoted for them.".format(len(markings)))
    if table_only:
        notes.append(
            "Leader(s) {} would not fit legibly on the view; their callout and position are "
            "in the table.".format(", ".join(table_only)))
    if table_note:
        notes.append(table_note)
    notes.extend(tolerance_notes(tolerance or {}))
    return notes


@dataclass
class DrawingResult:
    """What ``render_drawing`` produced. Lengths mm, volume mm3, timings s."""

    path: str
    png: Optional[str]
    png_tool: Optional[str]
    scale: float
    sheet_mm: List[float]
    bbox_mm: Dict[str, List[float]]
    volume_mm3: float
    levels_z: List[float]
    groups: List[Dict[str, Any]]
    views: Dict[str, Dict[str, Any]]
    render_s: float
    source: str = ""
    part: Optional[str] = None
    openscad_version: Optional[str] = None
    bytes: int = 0
    #: Face area, mm2, of each entry in ``levels_z`` - the evidence it is a surface.
    level_areas: List[float] = field(default_factory=list)
    #: Sections dismissed as modelling slivers; the count is also printed in the sheet notes.
    slivers: List[Dict[str, Any]] = field(default_factory=list)
    #: Shallow relief clusters reported as markings rather than as holes.
    markings: List[Dict[str, Any]] = field(default_factory=list)
    #: The general-tolerance block exactly as the resolver returned it, provenance included.
    #: ``specified: false`` is the honest answer, and is what the sheet prints as UNSPECIFIED.
    tolerance: Dict[str, Any] = field(default_factory=dict)
    #: What the sheet managed to print: origins, ordinates, leaders, title-block fields.
    layout: Dict[str, Any] = field(default_factory=dict)
    #: sha256 of the file the geometry came from, so a sheet can be tied to its source.
    source_sha256: Optional[str] = None
    #: Material and revision read verbatim from a source manifest; never defaulted, never
    #: guessed - ``None`` here is what the sheet prints as UNSPECIFIED.
    material: Optional[str] = None
    revision: Optional[str] = None
    #: The approver, only ever as declared for this drawing, and the manifest key it came
    #: from.  ``None`` prints as the empty ``UNSIGNED`` marker and keeps the banner.
    approved_by: Optional[str] = None
    approval_provenance: Optional[str] = None
    dxf: Optional[str] = None
    dxf_entities: Optional[Dict[str, int]] = None
    pdf: Optional[str] = None
    pdf_tool: Optional[str] = None
    #: Every sheet of the drawing, in order: ``{"sheet", "svg", "png"}``.  Sheet 1 is ``path``;
    #: a feature table too long for it continues on sheet 2, 3, ...  The PDF has every page.
    sheets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sheets": self.sheets,
            "png": self.png,
            "png_tool": self.png_tool,
            "pdf": self.pdf,
            "pdf_tool": self.pdf_tool,
            "dxf": self.dxf,
            "dxf_entities": self.dxf_entities,
            "scale": self.scale,
            "sheet_mm": self.sheet_mm,
            "bbox_mm": self.bbox_mm,
            "volume_mm3": round(self.volume_mm3, 3),
            "levels_z": [round(z, 3) for z in self.levels_z],
            # the evidence for every printed level, in the shape `levels_are_faces` reads, so
            # the gate can check the sheet without re-reading the mesh
            "level_faces": [{"z": round(z, 3), "area_mm2": round(a, 3)}
                            for z, a in zip(self.levels_z, self.level_areas)],
            "slivers": self.slivers,
            "markings": self.markings,
            "tolerance": self.tolerance,
            "layout": self.layout,
            "groups": self.groups,
            "views": self.views,
            "render_s": round(self.render_s, 3),
            "source": self.source,
            "source_sha256": self.source_sha256,
            "part": self.part,
            "material": self.material,
            "revision": self.revision,
            "approved_by": self.approved_by,
            "approval_provenance": self.approval_provenance,
            "backend": "openscad",
            "openscad_version": self.openscad_version,
            "bytes": self.bytes,
            "units": "mm; groups[].positions and depths are MODEL coordinates. The sheet works "
                     "in one frame per view: the drawn origin at that view's silhouette bbox "
                     "lower-left corner, and every ordinate and every table coordinate is a "
                     "distance from it (layout.origin_mm gives that corner in model mm)",
        }


def _wrap_text(text: str, width: float, size: float, mono: bool = False) -> List[str]:
    """Word-wrap a note to the paper width it has; a clipped note is a note nobody read.

    ``mono`` must match how the line will actually be drawn - a monospaced line is wider than
    the proportional estimate, so wrapping one with the wrong metric overflows its column."""
    lines: List[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else current + " " + word
        if current and text_width_mm(candidate, size, mono) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _inflate(box: Tuple[float, float, float, float],
             amount: float) -> Tuple[float, float, float, float]:
    return (box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount)


def compose_sheets(
    views: Dict[str, ViewData], groups: Sequence["Group"], *, title: str, source_label: str,
    volume: float, openscad_version: Optional[str], part: Optional[str],
    markings: Sequence["Marking"] = (), manifest: Optional[Mapping[str, Any]] = None,
    tolerance: Optional[Mapping[str, Any]] = None, source_sha256: Optional[str] = None,
    generator: Optional[str] = None,
) -> Tuple[List[str], float, Dict[str, Any]]:
    """Lay the three views out third-angle on A3; return (svg text per sheet, scale, layout).

    The scale is the largest ISO 5455 scale at which the views fit the paper the tables and the
    title block leave free.  Several arrangements of those are tried - the original full-width
    bottom band; bands that step down beside the title column; the tolerance block moved under
    the views; the feature table moved to sheet 2 - and the original is kept unless another
    gives a strictly larger scale.  A feature table longer than sheet 1 can carry continues on
    sheet 2, 3, ...: no row is ever dropped.

    The layout dict is the sheet's own account of what it managed to print - which leaders it
    placed, which positions went to the table, how many centre marks it drew, what every title
    block field was filled from.  A drawing that quietly drops a callout looks exactly like one
    that had nothing to say, so it says."""
    top, front, right = views["top"], views["front"], views["right"]
    W, D = top.size
    H = front.size[1]
    sheet_w, sheet_h = SHEET_MM
    margin = 10.0
    gap = 14.0
    title_w = 160.0

    facts = manifest_facts(manifest)
    tol = dict(tolerance) if tolerance is not None else resolve_tolerance_block(
        ir=manifest, source_manifest=manifest)
    material = str(facts["material"]) if facts["material"] else UNSPECIFIED
    finish = str(tol.get("finish") or facts["finish"] or "") or UNSPECIFIED
    approved_by = str(facts["approved_by"]) if facts["approved_by"] else UNSIGNED
    # An unapproved drawing must never be able to look approved, so the banner waits on a
    # signature as well as on material, finish and a declared general tolerance.
    complete = (material != UNSPECIFIED and finish != UNSPECIFIED
                and bool(tol.get("specified")) and approved_by != UNSIGNED)
    banner = None if complete else BANNER_TEXT
    # a chamfer is toleranced by the broken-edge table, not the linear one, so the sizes the
    # mesh actually measured are handed to the tolerance region to be looked up there
    edges = sorted({round(max(skirt.outward, skirt.run), 1)
                    for g in groups for m in g.members for skirt in m.skirts})
    tol_lines, tol_columns, tol_h = tolerance_block_layout(tolerance_lines(tol, edges),
                                                           title_w)
    title_h = title_block_height([("", "")] * len(REQUIRED_TITLE_FIELDS), banner is not None)

    # the table's coordinates are distances from each view's origin, which no scale changes,
    # so it can be planned - and the bands sized from it - before the scale is known.
    provisional = {name: _ViewFrame(view, 1.0, 0.0, 0.0) for name, view in views.items()}
    table_w = sheet_w - 2 * margin - title_w - 6.0
    pool = measured_pool(groups, views, top.levels)
    plan = table_plan(groups, markings, provisional, table_w, pool)
    footer_top = sheet_h - margin - _FOOTER_H

    left_margin = margin + _ORD_RESERVE
    top_margin = margin + _ORD_RESERVE
    chain_w = _level_chain_width(top.levels)
    avail_w = sheet_w - left_margin - margin - chain_w - 46.0
    tol_x = sheet_w - margin - title_w
    tol_band = tol_h + 2.0
    # under the views the block may be as wide as the table, and lays itself out shorter
    tol_lines_l, tol_columns_l, tol_h_l = tolerance_block_layout(
        tolerance_lines(tol, edges), table_w, compact=True)
    tol_band_l = tol_h_l + 2.0
    # a continuation sheet carries a heading, then the table down to the title block's band
    cont_top = margin + 8.0
    cont_lines = max(int((footer_top - _BAND_PAD - cont_top) / _TABLE_ROW) - 1, 1)

    top_need = margin / 2.0 + 1.5 + _TEXT_MM + _ORD_LEAD + _ORD_MAX_ROWS * _ORD_ROW + 4.0

    def notes_height(note: Optional[str]) -> float:
        lines = sum(len(_wrap_text(text, table_w - 2.0, _NOTE_SIZE))
                    for text in sheet_notes(1.0, views, markings, ["H0"], tol, note))
        return 3.5 + _NOTE_STEP * lines

    side_x0 = sheet_w - margin - table_w
    side_y0 = margin + 1.0

    def candidate(stepped: bool, tol_left: bool, first_lines: int,
                  side: bool = False) -> Dict[str, Any]:
        """Where the bands go for one arrangement, and the largest scale its views fit.

        ``stepped`` False is the original sheet: one band across the full width, as tall as
        the taller of its two halves.  ``stepped`` True lets each half keep its own height,
        so a front row that ends left of the title column may come down beside it.  ``side``
        puts the feature table in the top-right corner, beside the top view and above the
        front row, instead of in the band under the views - the paper a short table needs is
        often already empty there."""
        pages = split_table_rows(plan["rows"], first_lines, cont_lines)
        on_first = pages[0]
        # a part with no features keeps its (empty) table, as it always has
        draw_table = bool(on_first) or not plan["rows"]
        table_h = table_page(plan, on_first, 2 if len(pages) > 1 else None)["height"] \
            if draw_table else 0.0
        note = _table_sheet_note(pages, plan)
        band_table_h = 0.0 if side else table_h
        left_h = (band_table_h + notes_height(note) + (tol_band_l if tol_left else 0.0)
                  + _BAND_PAD)
        right_h = title_h + (0.0 if tol_left else tol_band) + _BAND_PAD
        if not stepped:
            left_h = right_h = max(left_h, right_h)
        left_top, right_top = footer_top - left_h, footer_top - right_h
        found: Optional[float] = None
        limit = min(left_top, right_top)
        for s in STANDARD_SCALES:
            if (W + D) * s > avail_w:
                continue
            row_right = left_margin + (W + D) * s + gap + 6.0 + _DIM_STEP + chain_w
            bottom = left_top if row_right <= tol_x - 4.0 else min(left_top, right_top)
            if stepped:
                # measured, not budgeted: the front row's labels and width dimension end
                # ``_BELOW_FRONT`` under it, and the top view's ordinate rows and overall
                # dimension need ``top_need`` above it inside the border
                room = bottom - _BELOW_FRONT - gap - top_need
            else:
                room = bottom - top_margin - gap - 12.0     # the original budget, unchanged
            if (D + H) * s > room:
                continue
            if side:
                side_bottom = side_y0 + table_h
                # the top view (and the ordinate labels over it) stays left of the table,
                # the table stays above the title column, and above the front row unless
                # that row ends left of it
                front_top = bottom - _BELOW_FRONT - H * s - 4.0
                if (left_margin + W * s + 6.0 > side_x0 - 2.0
                        or side_bottom > right_top - 4.0
                        or (side_bottom > front_top and row_right > side_x0 - 4.0)):
                    continue
            found, limit = s, bottom
            break
        return {"stepped": stepped, "tol_left": tol_left, "side": side, "pages": pages,
                "draw_table": draw_table, "note": note, "left_top": left_top,
                "right_top": right_top, "limit": limit, "fits": found is not None,
                "scale": found if found is not None else STANDARD_SCALES[-1]}

    # The original arrangement is the reference: another is taken only for a strictly larger
    # scale, so a sheet that already reads well is not rearranged.  The order is preference
    # on a tie - table under the views, then beside them, then the tolerance block moved.
    arrangements = [candidate(False, False, MAX_TABLE_ROWS)]
    for stepped, tol_left, first_lines, side in (
            (True, False, MAX_TABLE_ROWS, False), (True, True, MAX_TABLE_ROWS, False),
            (True, False, MAX_TABLE_ROWS, True), (True, True, MAX_TABLE_ROWS, True),
            (True, False, 0, False), (True, True, 0, False)):
        if (first_lines == 0 or side) and not plan["rows"]:
            continue
        arrangements.append(candidate(stepped, tol_left, first_lines, side))
    chosen = choose_arrangement(arrangements)
    scale = chosen["scale"]
    pages = chosen["pages"]
    sheet_count = len(pages)
    left_top, right_top = chosen["left_top"], chosen["right_top"]

    svg = _Svg(sheet_w, sheet_h)
    svg.rect(margin / 2, margin / 2, sheet_w - margin, sheet_h - margin, width=0.5)
    side_page = table_page(plan, pages[0], 2 if sheet_count > 1 else None)
    if chosen["side"]:
        svg.reserve((side_x0 - 3.0, 0.0, sheet_w, side_y0 + side_page["height"] + 3.0))
    if chosen["stepped"]:
        svg.reserve((0.0, left_top - 2.0, tol_x - 2.0, sheet_h))
        svg.reserve((tol_x - 2.0, right_top - 2.0, sheet_w, sheet_h))
    else:
        svg.reserve((0.0, left_top - 2.0, sheet_w, sheet_h))
    svg.reserve((0.0, 0.0, sheet_w, margin / 2 + 1.0))
    svg.reserve((0.0, 0.0, margin / 2 + 1.0, sheet_h))
    svg.reserve((sheet_w - margin / 2 - 1.0, 0.0, sheet_w, sheet_h))

    # front view bottom-left of the view area; top above it; right to its right
    front_oy = chosen["limit"] - (_BELOW_FRONT if chosen["stepped"] else 22.0)
    fr_front = _ViewFrame(front, scale, left_margin, front_oy)
    fr_top = _ViewFrame(top, scale, left_margin, fr_front.top - gap)
    fr_right = _ViewFrame(right, scale, fr_front.right + gap + 6.0, front_oy)
    frames = {"top": fr_top, "front": fr_front, "right": fr_right}

    centres = {name: _circular_centres(fr, groups) for name, fr in frames.items()}
    drawn = _draw_geometry(svg, fr_top, [front, right], centres["top"])
    _draw_geometry(svg, fr_front, [top, right], centres["front"])
    _draw_geometry(svg, fr_right, [top, front], centres["right"])
    for fr in (fr_top, fr_front, fr_right):
        svg.text(fr.ox, fr.oy + 5.5, fr.view.name.upper(), size=3.0, bold=True)
        _draw_origin(svg, fr)

    xs, ys = ordinate_values(fr_top, groups)
    refused_x, refused_y = _draw_ordinates(svg, fr_top, xs, ys)
    over = fr_top.top - _ORD_LEAD - _ORD_MAX_ROWS * _ORD_ROW - 4.0
    _dim_h(svg, fr_top, fr_top.x0, fr_top.x1, over, fr_top.y1, fr_top.y1)
    beside = fr_top.ox - _ORD_LEAD - _ORD_MAX_ROWS * _ORD_ROW - 4.0
    _dim_v(svg, fr_top, fr_top.y0, fr_top.y1, beside, fr_top.x0, fr_top.x0)
    _dim_h(svg, fr_front, fr_front.x0, fr_front.x1, fr_front.oy + _DIM_STEP + 4, fr_front.y0,
           fr_front.y0)
    _dim_h(svg, fr_right, fr_right.x0, fr_right.x1, fr_right.oy + _DIM_STEP + 4, fr_right.y0,
           fr_right.y0)
    _draw_level_chain(svg, fr_right, top.levels, fr_right.right + _DIM_STEP)

    keep = len(svg.reserved)
    for other in (fr_front, fr_right):
        svg.reserve(_inflate(other.paper_box, 1.5))
    callouts, loops = _view_callouts(fr_top, groups, markings, pool)
    placed, dropped, over_part = _place_callouts(svg, fr_top, callouts, drawn, loops)
    del svg.reserved[keep:]

    table_y = left_top
    if chosen["tol_left"]:
        _draw_tolerance_block(svg, margin, left_top, table_w, tol_lines_l, tol_columns_l)
        table_y += tol_band_l
        title_y = right_top
    else:
        _draw_tolerance_block(svg, tol_x, right_top, title_w, tol_lines, tol_columns)
        title_y = right_top + tol_band
    table_bottom = table_y
    if chosen["side"]:
        _draw_table(svg, side_x0, side_y0, table_w, side_page)
    elif chosen["draw_table"]:
        table_bottom = _draw_table(svg, margin, table_y, table_w, table_page(
            plan, pages[0], 2 if sheet_count > 1 else None))
    ny = table_bottom + 3.5
    for note in sheet_notes(scale, views, markings, dropped, tol, chosen["note"]):
        for line in _wrap_text(note, table_w - 2.0, _NOTE_SIZE):
            if ny < footer_top - 1.0:
                svg.text(margin, ny, line, size=_NOTE_SIZE, color="#333")
                ny += _NOTE_STEP
    drawing_no = str(part or facts["part_name"] or Path(source_label).stem or title)
    source_name = Path(source_label).name or source_label
    tool = generator or "PhiMech drawing"
    # "PhiMech drawing 0.1.0" names the tool for the GENERATOR field; "PHIMECH 0.1.0" is the
    # signature that goes on the DRAWN BY line and in the footer, where it stands in for a
    # person's name and has to read like one.
    signature = tool.upper().replace(" DRAWING", "")
    date = time.strftime("%Y-%m-%d")
    fields: List[Tuple[str, str]] = [
        ("MATERIAL", material),
        ("SURFACE FINISH", finish),
        ("REVISION", str(facts["revision"]) if facts["revision"] else UNSPECIFIED),
        ("DATE", date),
        ("SCALE", _scale_text(scale)),
        ("UNITS", "mm"),
        ("SHEET", "1 / {}".format(sheet_count)),
        ("SHEET SIZE", "A3  {} x {}".format(_num(sheet_w), _num(sheet_h))),
        # DRAWN BY is a fact: this module drew it.  APPROVED BY is an attestation and stays
        # empty unless a human declared one for this drawing (see UNSIGNED).
        ("DRAWN BY", signature),
        ("APPROVED BY", approved_by),
        ("GENERATOR", tool),
        ("RENDERER", "OpenSCAD {}".format(openscad_version or UNSPECIFIED)),
        ("SOURCE FILE", source_name),
        ("SOURCE SHA-256", (source_sha256 or "")[:16] or UNSPECIFIED),
    ]
    footer_text = attribution_footer(signature, date, source_name, source_sha256)
    _draw_title_block(svg, tol_x, title_y, title_w,
                      title=str(facts["title"] or title), drawing_no=drawing_no, fields=fields,
                      banner=banner)
    _draw_footer(svg, margin, footer_top, sheet_w - 2 * margin, footer_text)

    # continuation sheets: the rest of the feature table, under the same title block
    sheets = [svg.render()]
    for number, rows in enumerate(pages[1:], start=2):
        more = _Svg(sheet_w, sheet_h)
        more.rect(margin / 2, margin / 2, sheet_w - margin, sheet_h - margin, width=0.5)
        starts_here = number == 2 and not pages[0]
        more.text(margin, margin + 4.5, "FEATURE TABLE" if starts_here else
                  "FEATURE TABLE — CONTINUED FROM SHEET {}".format(number - 1),
                  size=3.0, bold=True)
        _draw_table(more, margin, cont_top, table_w, table_page(
            plan, rows, number + 1 if number < sheet_count else None,
            None if starts_here else number - 1))
        sheet_fields = [(label, "{} / {}".format(number, sheet_count) if label == "SHEET"
                         else value) for label, value in fields]
        _draw_title_block(more, tol_x, footer_top - _BAND_PAD - title_h, title_w,
                          title=str(facts["title"] or title), drawing_no=drawing_no,
                          fields=sheet_fields, banner=banner)
        _draw_footer(more, margin, footer_top, sheet_w - 2 * margin, footer_text)
        sheets.append(more.render())
    layout = {
        "origin_mm": {name: [fr.ox_model, fr.oy_model] for name, fr in frames.items()},
        "origin_paper": {name: [round(fr.ox, 4), round(fr.oy, 4)] for name, fr in
                         frames.items()},
        "paper_scale": scale,
        "origin_frame": "each view's silhouette bbox lower-left corner, drawn as an origin "
                        "indicator; every ordinate and every table coordinate is a distance "
                        "from it",
        "ordinates": {"x": len(xs) - len(refused_x), "y": len(ys) - len(refused_y),
                      "x_in_table_only": [round(v, 3) for v in refused_x],
                      "y_in_table_only": [round(v, 3) for v in refused_y]},
        "callouts": {"placed": placed, "in_table_only": dropped,
                     "printed_over_the_view": over_part},
        "centre_marks": sum(len(v) for v in centres.values()),
        "modifiers": [{"of": g.group_id, "role": role, "diameter": round(diameter, 3),
                       "z": [round(low, 3), round(high, 3)]}
                      for g in groups
                      for role, diameter, low, high in _modifier_signatures(g)],
        "banner": banner,
        "approval": {"approved_by": facts["approved_by"],
                     "provenance": facts["approval_provenance"],
                     "printed": approved_by},
        "title_fields": [[label, value] for label, value in fields],
        "required_title_fields": list(REQUIRED_TITLE_FIELDS),
        "tolerance": tol,
        "scale_text": _scale_text(scale),
        "table_rows": (table_page(plan, pages[0], 2 if sheet_count > 1 else None)["height"]
                       / _TABLE_ROW) if chosen["draw_table"] else 0,
        # nothing is ever dropped from the table any more: rows past sheet 1 continue on the
        # next sheet, and `table_sheets` says which ids are on which
        "table_overflow": [],
        "sheets": sheet_count,
        "table_sheets": [[str(row["id"]) for row in rows] for rows in pages],
        "arrangement": _arrangement_facts(chosen),
        "scale_candidates": [dict(_arrangement_facts(option), fits=option["fits"],
                                  scale=option["scale"]) for option in arrangements],
        "min_text_gap_mm": MIN_TEXT_GAP_MM,
    }
    return sheets, scale, layout


def compose_sheet(
    views: Dict[str, ViewData], groups: Sequence["Group"], **kwargs: Any
) -> Tuple[str, float, Dict[str, Any]]:
    """Sheet 1 of :func:`compose_sheets` - (svg text, scale, layout facts).

    ``layout["sheets"]`` says whether the feature table continued past it."""
    sheets, scale, layout = compose_sheets(views, groups, **kwargs)
    return sheets[0], scale, layout


# --------------------------------------------------------------------------------------
# rasterising (best effort) and the public entry points
# --------------------------------------------------------------------------------------


#: Minimal HTML around the SVG for headless Chrome: ``page`` is an ``@page`` rule for PDF
#: printing (empty for a screenshot); ``w``/``h`` size the image.
_HTML_WRAPPER = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<style>{page}html,body{{margin:0;padding:0;background:#fff}}"
    "img{{display:block;width:{w};height:{h}}}</style></head>"
    "<body><img src='{src}'></body></html>"
)


def _find_chrome() -> Optional[str]:
    """See ``phimech.synth.raster``: the one environment read the drawing needs lives there."""
    from phimech.synth.raster import find_chrome

    return find_chrome()


def rasterize(svg_path: Path, png_path: Path, width_px: int = 2400) -> Optional[str]:
    """Write a PNG of the SVG with whatever rasteriser is installed; returns the tool name or
    None. Never required: the SVG is the deliverable, the PNG is for agents that read images."""
    tools: List[Tuple[str, List[str]]] = []
    if shutil.which("rsvg-convert"):
        tools.append(("rsvg-convert", ["rsvg-convert", "-w", str(width_px), "-b", "white",
                                       str(svg_path), "-o", str(png_path)]))
    chrome = _find_chrome()
    wrapper: Optional[Path] = None
    if chrome:
        height = int(width_px * SHEET_MM[1] / SHEET_MM[0])
        # An <img> at 100vw makes the sheet fill the window; a bare SVG renders at its mm size.
        wrapper = png_path.with_suffix(".shot.html")
        wrapper.write_text(_HTML_WRAPPER.format(w="100vw", h="auto", page="",
                                                src=svg_path.resolve().as_uri()), encoding="utf-8")
        tools.append(("chrome", [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                                 "--hide-scrollbars",
                                 "--window-size={},{}".format(width_px, height),
                                 "--screenshot={}".format(png_path), wrapper.as_uri()]))
    if shutil.which("qlmanage"):
        tools.append(("qlmanage", ["qlmanage", "-t", "-s", str(width_px), "-o", str(png_path.parent),
                                   str(svg_path)]))
    try:
        for name, command in tools:
            try:
                subprocess.run(command, capture_output=True, timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if name == "qlmanage":
                produced = png_path.parent / (svg_path.name + ".png")
                if produced.is_file():
                    produced.replace(png_path)
            if png_path.is_file() and png_path.stat().st_size > 0:
                return name
        return None
    finally:
        if wrapper is not None and wrapper.exists():
            wrapper.unlink()


def sheet_path(out_svg: Path, number: int) -> Path:
    """Where sheet ``number`` (2, 3, ...) of the drawing ``out_svg`` is written."""
    out_svg = Path(out_svg)
    return out_svg.with_name("{}-sheet{}{}".format(out_svg.stem, number, out_svg.suffix))


def glob_escape(text: str) -> str:
    return "".join("[{}]".format(ch) if ch in "*?[" else ch for ch in text)


def to_pdf(svg_path: Path, pdf_path: Path, extra: Sequence[Path] = ()) -> Optional[str]:
    """Write an A3-landscape PDF of the sheet; returns the tool name or None. Same contract as
    ``rasterize``: best effort, never required, never faked. A PDF is what a shop is *sent*; it
    is a picture of the sheet, where the DXF is the editable one. ``rsvg-convert`` prints the
    SVG's own mm size; Chrome needs an HTML wrapper with an ``@page`` rule or it would paginate
    the sheet onto Letter portrait."""
    tools: List[Tuple[str, List[str]]] = []
    if shutil.which("rsvg-convert"):
        # rsvg-convert writes one page per input: a continuation sheet is page 2, 3, ...
        tools.append(("rsvg-convert", ["rsvg-convert", "-f", "pdf", str(svg_path)]
                      + [str(path) for path in extra] + ["-o", str(pdf_path)]))
    chrome = _find_chrome()
    wrapper: Optional[Path] = None
    if chrome:
        wrapper = pdf_path.with_suffix(".print.html")
        w, h = _num(SHEET_MM[0]), _num(SHEET_MM[1])
        html = _HTML_WRAPPER.format(
            w=w + "mm", h=h + "mm", page="@page{{size:{}mm {}mm;margin:0}}".format(w, h),
            src=svg_path.resolve().as_uri())
        if extra:
            # one <img> per sheet; each is exactly a page tall, so each prints on its own page
            html = html.replace("</body>", "".join(
                "<img src='{}'>".format(Path(path).resolve().as_uri()) for path in extra)
                + "</body>")
        wrapper.write_text(html, encoding="utf-8")
        tools.append(("chrome", [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                                 "--no-pdf-header-footer", "--print-to-pdf={}".format(pdf_path),
                                 wrapper.as_uri()]))
    try:
        for name, command in tools:
            try:
                subprocess.run(command, capture_output=True, timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if pdf_path.is_file() and pdf_path.stat().st_size > 0:
                return name
        return None
    finally:
        if wrapper is not None and wrapper.exists():
            wrapper.unlink()


def _openscad_version() -> Optional[str]:
    from phimech.synth import preview

    return preview._openscad_version()


def doctor_section() -> Dict[str, Any]:
    """Health block for `phimech doctor`: can a drawing be produced, and will it get a PNG?"""
    try:
        from phimech.backends.openscad import OpenSCADBackend

        backend = OpenSCADBackend()
        available = bool(backend.available())
        version = backend.version() if available else None
    except Exception as exc:  # a broken backend import is a finding, not a crash
        return {"name": "drawing", "ok": False, "error": str(exc)}
    chrome = _find_chrome()
    rasteriser = next((n for n in ("rsvg-convert", "qlmanage") if shutil.which(n)), None) \
        or ("chrome" if chrome else None)
    pdf_tool = "rsvg-convert" if shutil.which("rsvg-convert") else ("chrome" if chrome else None)
    return {
        "name": "drawing",
        "ok": available,
        "openscad": version,
        "rasteriser": rasteriser,
        "pdf": pdf_tool,
        "chrome": chrome,
        "sheet_mm": list(SHEET_MM),
        "max_slices": MAX_SLICES,
        "note": ("SVG via OpenSCAD 2D projections; PNG copy via {}; PDF via {}".format(
                     rasteriser, pdf_tool or "nothing (no rsvg-convert/chrome)")
                 if available and rasteriser else
                 "SVG only: no rsvg-convert/qlmanage/chrome for a PNG or PDF copy "
                 "(set PHIMECH_CHROME to a headless Chrome)" if available else
                 "OpenSCAD not found: install it or set PHIMECH_OPENSCAD"),
    }



def file_sha256(path: Path, limit: int = 1 << 28) -> Optional[str]:
    """sha256 of a file, or None when it cannot be read.  Ties a sheet to its source bytes."""
    digest = hashlib.sha256()
    try:
        with open(str(path), "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                limit -= len(chunk)
                if limit <= 0:
                    return None
    except OSError:
        return None
    return digest.hexdigest()


def load_manifest(path: Any) -> Optional[Dict[str, Any]]:
    """Read a source manifest (SolidWorks ``source.json`` or a PhiMech ``design.json``).

    The manifest is the *only* place material, revision, finish, tolerance and the text of a
    marking may come from.  A file that will not parse yields ``None`` and the sheet prints
    UNSPECIFIED, which is the correct output: a half-read manifest must not become a default."""
    try:
        candidate = Path(str(path))
        if not candidate.is_file():
            return None
        record = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _generator_label() -> str:
    try:
        from phimech.version import __version__
    except Exception:                       # noqa: BLE001 - a missing version is not a fault
        return "PhiMech drawing"
    return "PhiMech drawing {}".format(__version__)


def render_drawing(
    source: Any,
    out_svg: Path,
    *,
    title: Optional[str] = None,
    part: Optional[str] = None,
    timeout: float = 180.0,
    max_slices: int = MAX_SLICES,
    png: bool = True,
    manifest: Optional[Mapping[str, Any]] = None,
    dxf: bool = False,
    pdf: bool = True,
) -> DrawingResult:
    """Render the dimensioned drawing of ``source`` (STL / CSG AST .json / .scad / CSG Node).

    ``pdf`` writes an A3-landscape PDF beside the SVG on the same best-effort terms as ``png``
    (rsvg-convert or headless Chrome; ``pdf`` is None in the result when neither exists).

    ``manifest`` is the source's own record - a SolidWorks handoff ``source.json`` or a PhiMech
    ``design.json``.  It is the only route by which material, revision, a general tolerance or
    the text of a marking can reach the sheet; without one every such field prints
    ``UNSPECIFIED`` and the sheet keeps its "not for manufacture" banner.

    ``dxf`` additionally writes the sheet beside the SVG as DXF (AC1015).  That is the
    *editable* artefact: SolidWorks opens it as real geometry with editable dimensions, where a
    PNG or PDF of the same sheet is only a picture of one.  It is written from the same
    measured features rather than from the SVG, so a hole is a true CIRCLE entity and not a
    tessellated polyline."""
    from phimech.backends.openscad import measure_stl
    from phimech.synth import preview

    started = time.monotonic()
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="phimech-drawing-"))
    try:
        stl, label = preview._resolve_source(source, workdir, timeout)
        mesh = Mesh.read(stl)
        vertices = mesh.vertices
        if not vertices:
            raise BackendError("the mesh has no vertices")
        views: Dict[str, ViewData] = {}
        for name in VIEW_ORDER:
            views[name] = extract_view(stl, name, mesh, workdir, timeout, max_slices,
                                       sections=(name == "top"))
        features = [f for v in views.values() for f in v.features]
        groups = group_features(features)
        facts = manifest_facts(manifest)
        markings = [replace(m, ident="M{}".format(n)) for n, m in
                    enumerate((m for v in views.values() for m in v.markings), start=1)]
        markings = apply_marking_text(markings, facts["text"])
        tolerance = resolve_tolerance_block(ir=manifest, source_manifest=manifest)
        volume = measure_stl(stl).volume
        version = _openscad_version()
        if not title:
            title = Path(label).stem if "." in label else label
        origin = source if isinstance(source, (str, Path)) else None
        digest = file_sha256(Path(str(origin))) if origin and Path(str(origin)).is_file() \
            else file_sha256(stl)
        svg_texts, scale, layout = compose_sheets(
            views, groups, title=title, source_label=label, volume=volume,
            openscad_version=version, part=part, markings=markings, manifest=manifest,
            tolerance=tolerance, source_sha256=digest, generator=_generator_label(),
        )
        # sheet 1 is the file asked for; a continuation sheet sits beside it as -sheetN
        sheet_paths = [out_svg] + [sheet_path(out_svg, n) for n in range(2, len(svg_texts) + 1)]
        for path, text in zip(sheet_paths, svg_texts):
            path.write_text(text, encoding="utf-8")
        for stale in out_svg.parent.glob(glob_escape(out_svg.stem) + "-sheet*.svg"):
            number = stale.stem[len(out_svg.stem) + len("-sheet"):]
            if number.isdigit() and stale not in sheet_paths:   # a re-draw needing fewer sheets
                for suffix in (".svg", ".png"):
                    stale.with_suffix(suffix).unlink(missing_ok=True)
        png_paths: List[Optional[Path]] = []
        tool: Optional[str] = None
        if png:
            for path in sheet_paths:
                made = rasterize(path, path.with_suffix(".png"))
                tool = tool or made
                png_paths.append(path.with_suffix(".png") if made else None)
        png_path = png_paths[0] if png_paths else None
        pdf_path: Optional[Path] = None
        pdf_tool: Optional[str] = None
        if pdf:
            pdf_path = out_svg.with_suffix(".pdf")
            pdf_tool = to_pdf(out_svg, pdf_path, extra=sheet_paths[1:])
            if pdf_tool is None:
                pdf_path = None
        dxf_path: Optional[Path] = None
        dxf_counts: Optional[Dict[str, int]] = None
        if dxf:
            from phimech.synth import dxf as dxf_mod

            written = dxf_mod.write_dxf(
                views, groups, out_svg.with_suffix(".dxf"), title=title or "",
                source_label=label, part=part, volume_mm3=volume,
            )
            dxf_path = Path(written.path)
            dxf_counts = dict(written.entity_counts)
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        return DrawingResult(
            path=str(out_svg),
            png=str(png_path) if png_path else None,
            png_tool=tool,
            sheets=[{"sheet": n, "svg": str(path),
                     "png": str(png_paths[n - 1]) if png_paths and png_paths[n - 1] else None}
                    for n, path in enumerate(sheet_paths, start=1)],
            scale=scale,
            sheet_mm=list(SHEET_MM),
            bbox_mm={"min": [round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)],
                     "max": [round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)]},
            volume_mm3=volume,
            levels_z=list(views["top"].levels),
            level_areas=[lv.face_area for lv in mesh.levels(2, max_slices + 1)],
            slivers=[s for v in views.values() for s in v.slivers],
            markings=[m.to_dict() for m in markings],
            tolerance=tolerance,
            layout=layout,
            source_sha256=digest,
            material=facts["material"],
            revision=facts["revision"],
            approved_by=facts["approved_by"],
            approval_provenance=facts["approval_provenance"],
            dxf=str(dxf_path) if dxf_path else None,
            pdf=str(pdf_path) if pdf_path else None,
            pdf_tool=pdf_tool,
            dxf_entities=dxf_counts,
            groups=[g.to_dict() for g in groups],
            views={n: {"bbox": [round(b, 3) for b in v.bbox], "features": len(v.features),
                       "slices": len(v.slices), "seconds": round(v.seconds, 3),
                       "transitions": sum(1 for c in v.cuts if c.transition),
                       "slivers": len(v.slivers), "markings": len(v.markings)}
                   for n, v in views.items()},
            render_s=time.monotonic() - started,
            source=label,
            part=part,
            openscad_version=version,
            bytes=out_svg.stat().st_size,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def for_build(
    out_dir: Path, *, part: Optional[str] = None, filename: str = "drawing.svg",
    out_svg: Optional[Path] = None, **kwargs: Any
) -> DrawingResult:
    """Draw one printable part of a synth build directory (the largest unless ``part`` is named),
    writing ``<out_dir>/<filename>`` (or ``out_svg``) — the pipeline-facing sibling of
    ``preview.for_build``."""
    from phimech.synth import preview

    out_dir = Path(out_dir)
    candidates = preview._build_stl_candidates(out_dir)
    if not candidates:
        raise FileNotFoundError("no STL in build directory {}".format(out_dir))
    if part:
        matches = [c for c in candidates if c[0] == part]
        if not matches:
            raise ValueError("part {!r} not in build; parts: {}".format(
                part, ", ".join(c[0] for c in candidates)))
        chosen = matches[0]
    else:
        chosen = max(candidates, key=lambda c: c[2])
    kwargs.setdefault("title", None)
    kwargs["title"] = kwargs["title"] or chosen[0]
    if kwargs.get("manifest") is None:
        # a build directory carries the design that produced it; that IR is the source of
        # record for revision, marking text and any declared tolerance, and reading it here
        # is the difference between a sheet that states them and one that says UNSPECIFIED.
        for name in ("source.json", "design.json"):
            found = load_manifest(out_dir / name)
            if found is not None:
                kwargs["manifest"] = found
                break
    target = Path(out_svg) if out_svg else out_dir / filename
    return render_drawing(chosen[1], target, part=chosen[0], **kwargs)


# --------------------------------------------------------------------------------------
# CLI hook: `phimech drawing FILE|DIR -o OUT.svg`
# --------------------------------------------------------------------------------------


def _handle_drawing(args: argparse.Namespace) -> int:
    from phimech.synth import preview as preview_mod

    source = Path(args.source).expanduser().resolve()
    out = Path(args.output).expanduser().resolve() if args.output else None
    manifest = load_manifest(args.manifest) if getattr(args, "manifest", None) else None
    if manifest is None and source.is_dir():
        manifest = load_manifest(source / "source.json")
    try:
        if source.is_dir() and not preview_mod.is_solidworks_bundle(source):
            result = for_build(source, part=args.part, out_svg=out, png=not args.no_png, pdf=not args.no_pdf,
                               title=args.title, dxf=args.dxf, manifest=manifest)
        else:
            if out is None:
                out = (source / "drawing.svg") if source.is_dir() else \
                    source.with_suffix(".drawing.svg")
            result = render_drawing(source, out, title=args.title, png=not args.no_png, pdf=not args.no_pdf,
                                    manifest=manifest, dxf=args.dxf)
    except Exception as exc:  # noqa: BLE001 - one line for the human, exit 1
        print("drawing failed: {}".format(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("wrote {} (scale {}, {} feature groups)".format(
            result.path, _scale_text(result.scale), len(result.groups)))
        for extra in result.sheets[1:]:
            print("sheet {} {}".format(extra["sheet"], extra["svg"]))
        if result.png:
            print("png   {} via {}".format(result.png, result.png_tool))
        if result.pdf:
            print("pdf   {} via {}".format(result.pdf, result.pdf_tool))
        for g in result.groups:
            print("  {:<4} {:>2}x {:<12} {}".format(
                g["id"], g["qty"], g["kind"] + ("" if g["hole"] else " solid"),
                "thru" if g["thru"] else "depth {}".format(_num(g["depth"]))))
    return 0


def register_cli(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "drawing",
        help="dimensioned inspection drawing (SVG, + PNG when a rasteriser exists) of any geometry",
        description="Third-angle drawing with hole table from an STL, STEP, CSG AST .json, "
                    ".scad, a synth build directory or a SolidWorks handoff bundle. Every "
                    "number is measured from the mesh.",
    )
    parser.add_argument("source", help="STL / .step / CSG AST .json / .scad file, a synth build "
                                       "dir, or a SolidWorks handoff bundle dir")
    parser.add_argument("-o", "--output", help="output .svg (default: beside the source / in the build)")
    parser.add_argument("--part", help="part name inside a build directory (default: largest)")
    parser.add_argument("--title", help="title block text (default: the source name)")
    parser.add_argument("--manifest", help="source manifest JSON (SolidWorks source.json or a "
                                           "PhiMech design.json): the ONLY route by which "
                                           "material, revision, a general tolerance or the "
                                           "text of a marking reaches the sheet")
    parser.add_argument("--dxf", action="store_true",
                        help="also write the sheet as DXF (AC1015) beside the SVG - the "
                             "editable artefact: SolidWorks opens it as real geometry with "
                             "editable dimensions, where a PNG or PDF is a picture of one")
    parser.add_argument("--no-png", action="store_true", help="skip the best-effort PNG copy")
    parser.add_argument("--no-pdf", action="store_true",
                        help="skip the best-effort A3 PDF copy (rsvg-convert or headless Chrome; "
                             "PHIMECH_CHROME names a binary that is on no PATH)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.set_defaults(handler=_handle_drawing)
