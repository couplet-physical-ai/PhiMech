"""Edit risk for an imported STEP/STL: know before you fillet.

Why this exists
---------------
An imported vendor model is *edited, never recreated* (the imported-geometry workflow). The trap in that rule is committing to an edit strategy before knowing whether the
geometry can survive one. agentcad's ``edit_risk.py`` (Apache-2.0; the *idea* is reimplemented
here, no code is copied) cites the canonical case: a 2157-face import with 526 free edges and
self-intersections, on which every fillet and boolean failed and the agent burned every turn
retrying. The numbers were knowable from the file before the first attempt.

So this module turns face / edge / free-edge / shell counts into a **risk level with a named
workflow**, from the file alone, in milliseconds, with no CAD kernel:

* ``low``      -> *boolean*: cut and union against it directly.
* ``medium``   -> *boolean, carefully*: cuts and unions only, expect slow or fragile results, no
  fillet across the import, verify ``manifold`` afterwards.
* ``high``     -> *repair first*: the shell is open or faceted; close it before any boolean, or
  fall back to wrapping.
* ``hostile``  -> *wrap*: build the enclosure envelope around the measured bbox and never touch
  the import's geometry.
* ``unknown``  -> the file could not be read as geometry; treated as *wrap* because "unknown"
  must never render as "fine" (the same tri-state rule the gates use).

Every level is decided by the numeric thresholds in ``THRESHOLDS`` below and every decision
records the triggering numbers in ``reasons`` -- a verdict that cannot say *why* is not a gate.

What is exact and what is a proxy
---------------------------------
STL: ``faces`` is the triangle count; edges are quantised to ``EDGE_TOL`` mm and counted by how
many triangles use them. ``free_edges`` (used once) and ``nonmanifold_edges`` (used by three or
more) are exact for the mesh as written. ``bbox_mm`` is exact; ``volume_mm3`` is the
divergence-theorem volume and is only reported for a closed mesh, because the number is
meaningless for an open one.

STEP: ``faces`` and ``edges`` are entity counts (``ADVANCED_FACE``/``FACE_SURFACE``,
``EDGE_CURVE``). ``free_edges`` counts ``EDGE_CURVE`` entities referenced by exactly one
``ORIENTED_EDGE``; this is exact for the topology *as written*. OpenCascade (FreeCAD, and most
exporters) writes a cylinder seam as one edge used twice, a full sphere with a ``VERTEX_LOOP``
and no edge at all, and a cone apex as a bare vertex -- so none of those produce a false free
edge. What no parse can see is *geometric* self-intersection between well-formed faces; that
needs a kernel, which is why the import path also folds in FreeCAD's ``isValid()`` when the
backend measurement is available (``measurement=`` argument). ``bbox_mm`` for a STEP is
computed from its vertex points only when every face is planar (then the vertices span the
shape exactly); for curved faces it is left to the kernel measurement.

stdlib only, imports under python 3.9 (the core boundary). The STL reader is the check layer's
dependency-free ``phimech.check.mesh``; edge accounting is done here because the gates' reader
reports "open edges" as one number and this module needs free and non-manifold edges apart.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phimech.interop.step import (
    StepFacts,
    _split_sections,
    _strip_strings_and_comments,
    inspect as inspect_step,
)

Vec3 = Tuple[float, float, float]

LEVELS: Tuple[str, ...] = ("low", "medium", "high", "hostile", "unknown")
WORKFLOWS: Tuple[str, ...] = ("boolean", "repair", "wrap")

#: Vertex quantisation for STL edge matching, mm. Same value the manifold gate uses, so the two
#: agree on what "the same vertex" means.
EDGE_TOL = 1.0e-6

#: The numeric thresholds. Keep these thresholds and their tests in sync.
THRESHOLDS: Dict[str, int] = {
    # hostile: an open shell at a size where nothing recovers (agentcad's 2157/526 case)
    "hostile_open_faces": 1000,   # free_edges > 0 and faces > this
    # hostile: enough free edges that the shell is torn, whatever its size
    "hostile_free_edges": 100,    # free_edges > this
    # high: a tessellated solid this large cannot be filleted and booleans crawl
    "high_faceted_faces": 500,    # faceted STEP and faces > this
    # medium: a true B-rep large enough that edits commonly fail or time out
    # (the same numbers `interop step` already warns at)
    "large_faces": 1000,
    "large_edges": 2000,
    # medium: a mesh (STL or faceted STEP) large enough that CGAL/OCC booleans get slow
    "large_triangles": 20000,
}

# --- STEP entity-graph regexes, applied to text with string literals blanked to '' -------
# ORIENTED_EDGE('', edge_start, edge_end, edge_element, orientation): the element is the
# EDGE_CURVE this oriented edge uses. Start/end are '*' (derived) in OCC output and explicit
# '#n' references elsewhere; both are accepted.
_ORIENTED_EDGE = re.compile(
    r"#\d+\s*=\s*ORIENTED_EDGE\s*\(\s*''\s*,\s*(?:\*|#\d+)\s*,\s*(?:\*|#\d+)\s*,\s*#(\d+)\s*,",
    re.I,
)
_EDGE_CURVE_ID = re.compile(r"#(\d+)\s*=\s*EDGE_CURVE\s*\(", re.I)
_OPEN_SHELL = re.compile(r"#\d+\s*=\s*OPEN_SHELL\s*\(", re.I)
_VERTEX_POINT = re.compile(r"#(\d+)\s*=\s*VERTEX_POINT\s*\(\s*''\s*,\s*#(\d+)\s*\)", re.I)
_CARTESIAN_POINT = re.compile(
    r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*''\s*,\s*\(([^()]*)\)\s*\)", re.I
)


@dataclass
class EditRisk:
    """The assessment of one imported file. ``to_dict()`` is the MCP/CLI/report shape."""

    path: Path
    format: str                      # "step" | "stl"
    size_bytes: int = 0
    level: str = "unknown"           # one of LEVELS
    workflow: str = "wrap"           # one of WORKFLOWS
    faces: int = 0
    edges: int = 0
    free_edges: int = 0
    nonmanifold_edges: int = 0
    shells: int = 0
    solids: int = 0
    is_closed: bool = False
    is_manifold: bool = False
    faceted: bool = False
    bbox_mm: Optional[Tuple[Vec3, Vec3]] = None
    volume_mm3: Optional[float] = None
    kernel_valid: Optional[bool] = None   # FreeCAD isValid() when a measurement was supplied
    reasons: List[str] = field(default_factory=list)
    guidance: List[str] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)  # metric -> exact / proxy note
    parse_error: Optional[str] = None
    step: Optional[Dict[str, Any]] = None  # StepFacts.to_dict() subset, for STEP inputs

    @property
    def bbox_size_mm(self) -> Optional[Vec3]:
        if self.bbox_mm is None:
            return None
        mn, mx = self.bbox_mm
        return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])

    def to_dict(self) -> Dict[str, Any]:
        bbox = None
        if self.bbox_mm is not None:
            bbox = {
                "min": list(self.bbox_mm[0]),
                "max": list(self.bbox_mm[1]),
                "size": list(self.bbox_size_mm or ()),
            }
        return {
            "path": str(self.path),
            "format": self.format,
            "size_bytes": self.size_bytes,
            "level": self.level,
            "workflow": self.workflow,
            "faces": self.faces,
            "edges": self.edges,
            "free_edges": self.free_edges,
            "nonmanifold_edges": self.nonmanifold_edges,
            "shells": self.shells,
            "solids": self.solids,
            "is_closed": self.is_closed,
            "is_manifold": self.is_manifold,
            "faceted": self.faceted,
            "bbox_mm": bbox,
            "volume_mm3": self.volume_mm3,
            "kernel_valid": self.kernel_valid,
            "reasons": list(self.reasons),
            "guidance": list(self.guidance),
            "provenance": dict(self.provenance),
            "thresholds": dict(THRESHOLDS),
            "parse_error": self.parse_error,
            "step": self.step,
            "units": "mm",
        }


# --------------------------------------------------------------------------------------
# STL
# --------------------------------------------------------------------------------------


def _vertex_key(point: Vec3) -> Tuple[int, int, int]:
    return (
        int(round(point[0] / EDGE_TOL)),
        int(round(point[1] / EDGE_TOL)),
        int(round(point[2] / EDGE_TOL)),
    )


def _assess_stl(risk: EditRisk) -> None:
    from phimech.check.mesh import bbox as mesh_bbox, read_stl, volume as mesh_volume

    try:
        triangles = read_stl(str(risk.path))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        risk.parse_error = "not a readable STL: {}".format(exc)
        return

    risk.faces = len(triangles)
    risk.faceted = True  # a mesh is faceted by definition

    # Undirected edge -> number of triangles using it. Also a union-find over triangles that
    # share a quantised vertex, so shells can be judged one by one: two closed shells and one
    # torn one must not be reported as "three shells, some free edges somewhere".
    parent = list(range(len(triangles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: Dict[Tuple[int, int, int], int] = {}
    edge_uses: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    edge_owner: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    for index, triangle in enumerate(triangles):
        keys = [_vertex_key(point) for point in triangle]
        for key in keys:
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                union(index, previous)
        for first, second in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
            edge = (first, second) if first <= second else (second, first)
            edge_uses[edge] = edge_uses.get(edge, 0) + 1
            edge_owner.setdefault(edge, index)

    roots = {find(index) for index in range(len(triangles))}
    free_by_shell: Dict[int, int] = {root: 0 for root in roots}
    bad_by_shell: Dict[int, int] = {root: 0 for root in roots}
    for edge, uses in edge_uses.items():
        root = find(edge_owner[edge])
        if uses == 1:
            free_by_shell[root] += 1
        elif uses > 2:
            bad_by_shell[root] += 1

    risk.edges = len(edge_uses)
    risk.free_edges = sum(free_by_shell.values())
    risk.nonmanifold_edges = sum(bad_by_shell.values())
    risk.shells = len(roots)
    risk.solids = sum(
        1 for root in roots if free_by_shell[root] == 0 and bad_by_shell[root] == 0
    )
    risk.is_closed = risk.free_edges == 0
    risk.is_manifold = risk.nonmanifold_edges == 0
    risk.bbox_mm = mesh_bbox(triangles)
    if risk.is_closed:
        risk.volume_mm3 = mesh_volume(triangles)
    risk.provenance.update({
        "faces": "exact: triangle count",
        "edges": "exact: distinct undirected edges after {:g} mm vertex quantisation".format(
            EDGE_TOL),
        "free_edges": "exact: edges used by exactly one triangle",
        "nonmanifold_edges": "exact: edges used by three or more triangles",
        "shells": "exact: triangle components connected through shared vertices",
        "solids": "exact: shells with no free and no non-manifold edge",
        "is_manifold": "proxy: edge manifoldness only; self-intersection between triangles "
                       "is not detected without a kernel",
        "bbox_mm": "exact: vertex extent",
        "volume_mm3": "exact for a closed mesh (divergence theorem); omitted when open",
    })


# --------------------------------------------------------------------------------------
# STEP
# --------------------------------------------------------------------------------------


def _step_text(path: Path, max_bytes: int) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_bytes)
    except OSError:
        return None


def _planar_vertex_bbox(data: str, facts: StepFacts) -> Optional[Tuple[Vec3, Vec3]]:
    """Vertex extent, exact only when every face is a plane; otherwise ``None``.

    A curved face bulges past its vertices (the far side of a cylinder has no vertex on it),
    so the vertex extent under-reports it and the honest answer is "ask the kernel".
    """
    if not facts.faces or facts.planar_fraction is None or facts.planar_fraction < 1.0:
        return None
    if facts.mm_per_unit is None:
        return None
    points: Dict[str, Vec3] = {}
    for match in _CARTESIAN_POINT.finditer(data):
        try:
            coords = [float(v) for v in match.group(2).split(",")]
        except ValueError:
            continue
        if len(coords) == 3:
            points[match.group(1)] = (coords[0], coords[1], coords[2])
    used: List[Vec3] = []
    for match in _VERTEX_POINT.finditer(data):
        point = points.get(match.group(2))
        if point is not None:
            used.append(point)
    if not used:
        return None
    scale = facts.mm_per_unit
    mn = tuple(min(p[i] for p in used) * scale for i in range(3))
    mx = tuple(max(p[i] for p in used) * scale for i in range(3))
    return mn, mx  # type: ignore[return-value]


def _assess_step(risk: EditRisk, max_bytes: int) -> None:
    facts = inspect_step(risk.path, max_bytes=max_bytes)
    if facts.parse_error and not facts.entity_counts:
        risk.parse_error = facts.parse_error
        return
    text = _step_text(risk.path, max_bytes)
    if text is None:
        risk.parse_error = "cannot read {}".format(risk.path)
        return
    _, data = _split_sections(_strip_strings_and_comments(text))

    edge_ids = {match.group(1) for match in _EDGE_CURVE_ID.finditer(data)}
    uses: Dict[str, int] = {}
    for match in _ORIENTED_EDGE.finditer(data):
        edge = match.group(1)
        uses[edge] = uses.get(edge, 0) + 1
    free = sum(1 for edge in edge_ids if uses.get(edge, 0) == 1)
    nonmanifold = sum(1 for edge in edge_ids if uses.get(edge, 0) > 2)
    unreferenced = sum(1 for edge in edge_ids if uses.get(edge, 0) == 0)
    open_shells = len(_OPEN_SHELL.findall(data))

    risk.faces = facts.faces
    risk.edges = len(edge_ids) or facts.edges
    risk.free_edges = free
    risk.nonmanifold_edges = nonmanifold
    risk.shells = facts.closed_shells + open_shells
    risk.solids = facts.solids
    risk.faceted = facts.looks_tessellated
    risk.is_closed = (
        facts.body_kind == "solid" and open_shells == 0 and free == 0 and facts.faces > 0
    )
    risk.is_manifold = nonmanifold == 0 and facts.body_kind == "solid"
    risk.bbox_mm = _planar_vertex_bbox(data, facts)
    if facts.parse_error:
        risk.parse_error = facts.parse_error  # truncated read: counts are partial, say so
    risk.step = {
        "protocol": facts.protocol,
        "body_kind": facts.body_kind,
        "length_unit": facts.length_unit,
        "mm_per_unit": facts.mm_per_unit,
        "originating_system": facts.originating_system,
        "planar_fraction": facts.planar_fraction,
        "analytic_surfaces": facts.analytic_surfaces,
        "spline_surfaces": facts.spline_surfaces,
        "curved_surfaces": facts.curved_surfaces,
        "closed_shells": facts.closed_shells,
        "open_shells": open_shells,
        "unreferenced_edges": unreferenced,
        "looks_tessellated": facts.looks_tessellated,
    }
    risk.provenance.update({
        "faces": "exact: ADVANCED_FACE + FACE_SURFACE entities",
        "edges": "exact: EDGE_CURVE entities",
        "free_edges": "exact for the topology as written: EDGE_CURVE referenced by exactly one "
                      "ORIENTED_EDGE (a seam is one edge used twice, so it does not count)",
        "nonmanifold_edges": "exact for the topology as written: EDGE_CURVE referenced by "
                             "three or more ORIENTED_EDGEs",
        "shells": "exact: CLOSED_SHELL + OPEN_SHELL entities",
        "solids": "exact: MANIFOLD_SOLID_BREP + BREP_WITH_VOIDS entities",
        "is_closed": "proxy: a MANIFOLD_SOLID_BREP/CLOSED_SHELL is present, no OPEN_SHELL, "
                     "no free edge; geometric gaps between well-formed faces are invisible",
        "is_manifold": "proxy: edge manifoldness only; self-intersection needs a kernel "
                       "(kernel_valid carries FreeCAD's isValid() when a measurement is given)",
        "faceted": "proxy: step.py looks_tessellated (>= 200 faces, >= 95% planar, "
                   "face/edge ratio of a triangulation)",
        "bbox_mm": ("exact: vertex extent, every face is planar" if risk.bbox_mm is not None
                    else "not computed: curved faces need a kernel measurement"),
        "volume_mm3": "not computed from STEP text; supplied by the kernel measurement",
    })


# --------------------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------------------

_GUIDANCE: Dict[str, List[str]] = {
    "wrap": [
        "wrap it: build the enclosure envelope around the measured bbox and do not fillet, cut "
        "or union the import itself; `source.kind: import` already treats it as a fixed "
        "envelope, so keep every edit on the enclosure body",
        "if the import must change, fix it at the source (re-export a sewn, valid solid) and "
        "re-run `phimech interop risk` before spending a turn on a boolean",
    ],
    "repair": [
        "repair first: the shell is open, so a boolean against it yields a shell or garbage, "
        "not a solid; sew/close it in the originating CAD (FreeCAD: Part sewShape + makeSolid) "
        "and re-assess - PhiMech's brep lane has no import/sew op yet, so it cannot do this "
        "for you",
        "until it is closed, wrap it: design around the measured bbox and do not fillet the "
        "import",
    ],
    "boolean": [
        "safe to boolean directly: cut and union against the import; keep fillets on the "
        "enclosure body, never across the import's own edges",
    ],
}


def _judge(risk: EditRisk) -> None:
    """Assign level, workflow, reasons and guidance from the metrics. Order is severity."""
    t = THRESHOLDS
    reasons: List[str] = []
    level = "low"

    if risk.parse_error and not risk.faces:
        risk.level = "unknown"
        risk.workflow = "wrap"
        risk.reasons = ["no geometry could be read: {}".format(risk.parse_error)]
        risk.guidance = [
            "unknown is not fine: treat as hostile until the file reads - wrap the backend-"
            "measured bbox or fix the file",
        ] + _GUIDANCE["wrap"][1:]
        return

    # hostile -------------------------------------------------------------------------
    if risk.kernel_valid is False:
        level = "hostile"
        reasons.append("kernel reports the shape invalid (FreeCAD isValid() is False): "
                       "self-intersecting or broken topology")
    if risk.nonmanifold_edges > 0:
        level = "hostile"
        reasons.append("{} non-manifold edge(s) shared by three or more faces (threshold 0)"
                       .format(risk.nonmanifold_edges))
    if risk.free_edges > 0 and risk.faces > t["hostile_open_faces"]:
        level = "hostile"
        reasons.append("open shell at scale: {} free edge(s) on {} faces (> {} faces)"
                       .format(risk.free_edges, risk.faces, t["hostile_open_faces"]))
    if risk.free_edges > t["hostile_free_edges"]:
        level = "hostile"
        reasons.append("{} free edges (> {}): the shell is torn, not merely unsewn"
                       .format(risk.free_edges, t["hostile_free_edges"]))

    # high ----------------------------------------------------------------------------
    if level != "hostile":
        if risk.free_edges > 0:
            level = "high"
            reasons.append("{} free edge(s) (threshold 0): the shell is open, so it is not a "
                           "solid and no boolean on it yields one"
                           .format(risk.free_edges))
        if risk.format == "step" and risk.faceted and risk.faces > t["high_faceted_faces"]:
            level = "high"
            reasons.append("faceted STEP: {} planar faces (> {}) - a mesh wearing a B-rep "
                           "costume; nothing on it can be filleted and booleans crawl"
                           .format(risk.faces, t["high_faceted_faces"]))
        if risk.format == "step" and risk.step and risk.step.get("body_kind") in (
                "surfaces", "mesh", "empty") and not risk.free_edges:
            level = "high"
            reasons.append("STEP body kind is {!r}: no MANIFOLD_SOLID_BREP/CLOSED_SHELL, so it "
                           "imports as loose faces or a graphics mesh, not a solid"
                           .format(risk.step.get("body_kind")))

    # medium --------------------------------------------------------------------------
    if level == "low":
        if risk.format == "step" and not risk.faceted and (
                risk.faces > t["large_faces"] or risk.edges > t["large_edges"]):
            level = "medium"
            reasons.append("large true B-rep: {} faces / {} edges (> {} / > {}): boolean and "
                           "fillet edits commonly fail or time out"
                           .format(risk.faces, risk.edges, t["large_faces"], t["large_edges"]))
        if risk.format == "step" and risk.faceted:
            level = "medium"
            reasons.append("faceted STEP: {} planar faces - cuts work, fillets on it are "
                           "impossible".format(risk.faces))
        if risk.faceted and risk.faces > t["large_triangles"]:
            level = "medium"
            reasons.append("{} triangles (> {}): mesh booleans get slow"
                           .format(risk.faces, t["large_triangles"]))
        if risk.solids > 1:
            level = "medium"
            reasons.append("{} solids in one file: an operation must target one body, and "
                           "placement treats the union bbox as the part".format(risk.solids))

    if level == "low":
        reasons.append(
            "closed, edge-manifold, {} faces, {} edges, 0 free edges, {} solid(s)".format(
                risk.faces, risk.edges, risk.solids))

    workflow = {"low": "boolean", "medium": "boolean", "high": "repair",
                "hostile": "wrap"}[level]
    # A faceted-but-closed STEP is not "open", so "repair" is the wrong verb for it: there is
    # nothing to sew, the answer is to design around it.
    if level == "high" and risk.free_edges == 0:
        workflow = "wrap"

    guidance = list(_GUIDANCE[workflow])
    if level == "medium":
        guidance = [
            "boolean with care: cuts and unions only, one at a time, and run the `manifold` "
            "gate after each; expect seconds-to-minutes and the odd failure",
            "do not fillet the import or any edge a boolean created on it; put fillets on the "
            "enclosure body",
        ] + guidance[1:]
    if risk.faceted and risk.format == "stl" and level in ("low", "medium"):
        guidance.append("it is a mesh: no feature edit (fillet, chamfer, shell) is possible on "
                        "it in any lane; cut/union is the whole vocabulary")
    if risk.kernel_valid is None and risk.format == "step":
        guidance.append("this is a file parse: geometric self-intersection is not checked; the "
                        "import path adds FreeCAD's isValid() as kernel_valid")

    risk.level = level
    risk.workflow = workflow
    risk.reasons = reasons
    risk.guidance = guidance


def _fold_measurement(risk: EditRisk, measurement: Dict[str, Any]) -> None:
    """Prefer the kernel's numbers where it has them: bbox, volume, solids, validity."""
    bbox = measurement.get("bbox")
    if isinstance(bbox, dict):
        mn, mx = bbox.get("min"), bbox.get("max")
        if isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple)) \
                and len(mn) == 3 and len(mx) == 3:
            try:
                risk.bbox_mm = (
                    (float(mn[0]), float(mn[1]), float(mn[2])),
                    (float(mx[0]), float(mx[1]), float(mx[2])),
                )
                risk.provenance["bbox_mm"] = "exact: kernel measurement (FreeCAD/OpenSCAD)"
            except (TypeError, ValueError):
                pass
    volume = measurement.get("volume_mm3", measurement.get("volume"))
    if isinstance(volume, (int, float)) and volume > 0:
        risk.volume_mm3 = float(volume)
        risk.provenance["volume_mm3"] = "exact: kernel measurement"
    solids = measurement.get("solids")
    if isinstance(solids, int) and solids >= 0 and risk.format == "step":
        risk.solids = solids
    extra = measurement.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("valid"), bool):
        risk.kernel_valid = extra["valid"]
        risk.provenance["kernel_valid"] = "exact: FreeCAD Shape.isValid() (BRepCheck)"


def assess(
    path: Any,
    *,
    measurement: Optional[Dict[str, Any]] = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> EditRisk:
    """Assess how safely an imported ``.step``/``.stp``/``.stl`` can be edited.

    Never raises for bad *content*: an unreadable or empty file comes back as level ``unknown``
    with ``parse_error`` set, because the import path must keep working and "could not tell"
    must never be reported as "low". A wrong *suffix* does raise ``ValueError`` -- that is a
    caller bug, not a property of the file.

    ``measurement`` is an optional backend ``Measurement.to_dict()`` (as cached by
    ``phimech.components.imports``); when given, its bbox/volume/solids/validity are folded in.
    ``max_bytes`` caps the STEP read so a mis-pointed multi-GB path cannot wedge a build.
    """
    resolved = Path(path).expanduser()
    suffix = resolved.suffix.lower()
    if suffix in (".step", ".stp"):
        fmt = "step"
    elif suffix == ".stl":
        fmt = "stl"
    else:
        raise ValueError(
            "edit risk is assessed for .step/.stp/.stl files only, not {!r}".format(suffix))

    risk = EditRisk(path=resolved, format=fmt)
    try:
        risk.size_bytes = resolved.stat().st_size
    except OSError as exc:
        risk.parse_error = "cannot stat: {}".format(exc)
        _judge(risk)
        return risk

    if fmt == "stl":
        _assess_stl(risk)
    else:
        _assess_step(risk, max_bytes)
    if measurement:
        _fold_measurement(risk, measurement)
    _judge(risk)
    return risk


# --------------------------------------------------------------------------------------
# hooks for the coordinator: CLI, doctor
# --------------------------------------------------------------------------------------

#: Exit codes for `phimech interop risk`, so a shell can gate on the verdict.
EXIT_CODES: Dict[str, int] = {"low": 0, "medium": 0, "high": 1, "hostile": 1, "unknown": 2}


def format_table(risk: EditRisk) -> str:
    """The human rendering: verdict first, numbers second, guidance last."""
    lines = ["{}  {}".format(risk.level.upper(), risk.path)]
    lines.append("  workflow: {}".format(risk.workflow))
    size = risk.bbox_size_mm
    bbox = ("{:g} x {:g} x {:g} mm".format(*size) if size else "bbox needs a kernel measurement")
    volume = "{:.1f} mm3".format(risk.volume_mm3) if risk.volume_mm3 is not None else "-"
    lines.append("  {} - {} bytes - {} - volume {}".format(
        risk.format.upper(), risk.size_bytes, bbox, volume))
    lines.append("  {} faces, {} edges, {} free, {} non-manifold; {} shell(s), {} solid(s)".format(
        risk.faces, risk.edges, risk.free_edges, risk.nonmanifold_edges, risk.shells,
        risk.solids))
    flags = [
        "closed" if risk.is_closed else "OPEN",
        "edge-manifold" if risk.is_manifold else "NON-MANIFOLD",
        "faceted" if risk.faceted else "true B-rep",
    ]
    if risk.kernel_valid is not None:
        flags.append("kernel valid" if risk.kernel_valid else "KERNEL INVALID")
    lines.append("  " + ", ".join(flags))
    if risk.parse_error:
        lines.append("  parse: {}".format(risk.parse_error))
    for reason in risk.reasons:
        lines.append("    because {}".format(reason))
    for hint in risk.guidance:
        lines.append("    -> {}".format(hint))
    return "\n".join(lines)


def _handle_risk(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        print("error: no such file: {}".format(path))
        return 2
    risk = assess(path)
    if getattr(args, "json", False):
        print(json.dumps(risk.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_table(risk))
    return EXIT_CODES[risk.level]


def register_cli(subparsers: Any) -> None:
    """Add ``risk FILE [--json]``. Pass the ``interop`` command's subparsers to get
    ``phimech interop risk``; exit code 0 low/medium, 1 high/hostile, 2 unknown."""
    parser = subparsers.add_parser(
        "risk",
        help="edit risk of an imported STEP/STL: level, free edges, and the workflow to use",
    )
    parser.add_argument("file", help="path to a .step, .stp or .stl file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.set_defaults(handler=_handle_risk)


def doctor_section() -> Dict[str, Any]:
    """Self-check for `phimech doctor`: the synthetic open-shell fixture must assess as ``hostile``."""
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "interop" / "hostile-open-shell.stl"
    section: Dict[str, Any] = {"name": "edit_risk", "fixture": str(fixture), "ok": False}
    if not fixture.is_file():
        section["error"] = "fixture missing"
        return section
    try:
        risk = assess(fixture)
    except Exception as exc:  # doctor must report, never crash
        section["error"] = str(exc)
        return section
    section.update(level=risk.level, faces=risk.faces, edges=risk.edges,
                   free_edges=risk.free_edges, ok=risk.level == "hostile")
    return section


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python3 -m phimech.interop.edit_risk FILE [--json]` for use without the launcher."""
    parser = argparse.ArgumentParser(prog="phimech.interop.edit_risk")
    parser.add_argument("file")
    parser.add_argument("--json", action="store_true")
    return _handle_risk(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_CODES",
    "EditRisk",
    "LEVELS",
    "THRESHOLDS",
    "WORKFLOWS",
    "assess",
    "doctor_section",
    "format_table",
    "register_cli",
]
