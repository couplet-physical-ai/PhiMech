"""Numeric gates on a generated inspection drawing.

Why this exists
---------------
``phimech/synth/drawing.py`` measures the mesh and lays out a sheet; until now nothing checked
the *sheet*.  A drawing can be geometrically perfect and still unusable: two labels printed on
top of each other, a callout off the paper, a feature with no dimension, a number nobody can
trace back to a measurement.  Those are the failures a machinist finds, and they are far too
cheap to detect to be left to a model: the AI review (``phimech/agent/drawing_review.py``) is
the *second* opinion, and it can never lift a failure recorded here.

Every check is ``{name, status, value, threshold, evidence}`` with
``status in pass | fail | skipped | not_applicable``, a numeric threshold in millimetres or a
count, and evidence naming the view, group, text id or sheet bbox that decided it.  The
summary is ``{status, first_failure, counts}``.

How the sheet is read back
--------------------------
The drawing module writes the SVG, so its own text positions are recoverable: this module
parses ``<text>`` and ``<rect>`` out of the sheet and reconstructs every label's paper box.
The viewBox is ``0 0 420 297`` for a 420 mm x 297 mm sheet, so one user unit is one
millimetre and no scaling is needed.  Glyph advances come from a small per-character table
rather than a font library (stdlib only), so collisions and clipping are judged with a
0.3 mm slack instead of pretending to be exact - the check reports overlaps a reader would
see, not sub-pixel touching.

Three checks here go beyond the accepted proposal, because the seven it lists can all
pass on a sheet that is wrong:

* ``dimension_traceability`` proves *self-consistency*, not truth - a tessellation ring the
  pipeline mistook for a face is a "measured" value and traces cleanly.  ``levels_are_faces``
  is the truth check: a printed z level must be a real horizontal face of at least
  ``MIN_FACE_AREA_MM2``, or the depth built on it was fabricated.
* ``feature_coverage`` at group level passes while an individual hole has no dimension
  anywhere on the sheet, so coverage is decided **per member**, following the chain of printed
  pitches out from the datum exactly as a machinist would.
* ``intent_vs_measured`` is the only check that can see a feature that was asked for and never
  built (three counterbores declared, none in the solid).  It needs a declared-intent source
  and is ``not_applicable`` without one; intent is never invented.

Two readings the spec left open, resolved here and stated on the sheet of every result:

* "every view >= 8 mm on the sheet" is read as the view's **largest** paper dimension.  A
  60 x 2 mm plate at 3:1 has a 6 mm tall front view that is perfectly readable; a view whose
  longest side is under 8 mm is a dot.
* "every printed value equals a measured value within 0.01 mm" is applied to the sheet's
  *dimension* texts (the blue ones the drawing module reserves for dimensions) and to the
  numeric tokens of the leader callouts.  The table and title block are projections of the
  same measured dict and carry non-dimensional text (dates, volumes, tool versions).

Units are millimetres.  Stdlib only and importable under python 3.9, because
``phimech/agent/drawing_review.py`` imports it and that package holds that line.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

PASS = "pass"
FAIL = "fail"
SKIPPED = "skipped"
NOT_APPLICABLE = "not_applicable"
STATUSES: Tuple[str, ...] = (PASS, FAIL, SKIPPED, NOT_APPLICABLE)

#: Summary verdicts, worst first.  ``needs_review`` is what a skipped check leaves behind: the
#: drawing was not proven wrong, and was not proven right either.
SUMMARY_FAIL = "fail"
SUMMARY_NEEDS_REVIEW = "needs_review"
SUMMARY_PASS = "pass"

CHECK_NAMES: Tuple[str, ...] = (
    "views_readable",
    "text_no_collision",
    "text_not_clipped",
    "label_view_overlap",
    "label_line_overlap",
    "feature_coverage",
    "dimension_traceability",
    "levels_are_faces",
    "intent_vs_measured",
    "source_translation_clean",
    "manufacturing_intent_present",
)

#: A view whose longest paper dimension is under this is unreadable at any scale.
MIN_VIEW_MM = 8.0
#: Two text boxes must overlap by more than this in *both* axes to count as a collision; the
#: glyph widths are estimates, so anything tighter would report font metrics, not a defect.
TEXT_OVERLAP_MM = 0.3
#: A text box may hang this far over a boundary before it counts as clipped (same reason).
CLIP_SLACK_MM = 0.5
#: "every printed value equals a measured value within 0.01 mm" (proposal section 3).
TRACE_TOLERANCE_MM = 0.01
#: Sheet values are printed rounded to 2 dp, so a position dimension matches its feature to
#: half of the last printed digit.
POSITION_TOLERANCE_MM = 0.005

#: A printed z level must be a real horizontal face of at least this area.  1 mm2 is well
#: below any feature worth dimensioning and well above a tessellation ring, whose horizontal
#: face area is exactly zero.
MIN_FACE_AREA_MM2 = 1.0
#: A facet counts as lying at a level when all three vertices are within this of it.
LEVEL_FACE_TOLERANCE_MM = 0.05
#: A declared feature is "found" when a measured feature matches its position and size to
#: within this.  0.05 mm is below any printable or machinable tolerance and far above the
#: repeatability of measuring the same solid twice.
INTENT_MATCH_TOLERANCE_MM = 0.05
#: Declared feature kinds that must show up as a measured *hole*.
HOLE_KINDS: Tuple[str, ...] = ("hole", "counterbore", "countersink", "tapped_hole", "pocket",
                               "slot", "bore")
#: Tapping-drill pitches, mirroring ``phimech.brep.program.THREAD_PITCH`` so this module does
#: not depend on the optional B-rep lane being installed.
THREAD_PITCH_MM: Dict[str, float] = {
    "M2": 0.4, "M2.5": 0.45, "M3": 0.5, "M4": 0.7, "M5": 0.8, "M6": 1.0, "M8": 1.25,
}

#: The colour ``phimech/synth/drawing.py`` reserves for dimensions (``_DIM``).  Anything drawn
#: in it is a dimension and its numbers must be traceable.
DIMENSION_FILL = "#1d4ed8"

#: The standard scale series the sheet must choose from (mirrors ``drawing.STANDARD_SCALES``;
#: duplicated rather than imported so this module stays independent of the synth package).
STANDARD_SCALES: Tuple[float, ...] = (5, 4, 3, 2.5, 2, 1.5, 1.25, 1, 0.75, 0.5, 0.4, 0.3, 0.25,
                                      0.2, 0.15, 0.1, 0.05)

#: Manufacturing intent a release drawing must carry and an inspection drawing must not invent.
INTENT_ITEMS: Tuple[str, ...] = ("material", "tolerance", "finish", "thread", "gdt")

#: Glyph advance as a fraction of the font size.  Helvetica-ish; exact values are impossible
#: without a font library and unnecessary given ``TEXT_OVERLAP_MM``.
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

_TEXT_RE = re.compile(r"<text\b([^>]*)>(.*?)</text>", re.S)
_RECT_RE = re.compile(r"<rect\b([^>]*?)/>", re.S)
_LINE_RE = re.compile(r"<line\b([^>]*?)/>", re.S)
_POLYGON_RE = re.compile(r"<polygon\b([^>]*?)/>", re.S)
_ATTR_RE = re.compile(r'([A-Za-z_][-A-Za-z0-9_]*)\s*=\s*"([^"]*)"')
_ROTATE_RE = re.compile(r"rotate\(\s*([-0-9.eE+]+)(?:\s+([-0-9.eE+]+)\s+([-0-9.eE+]+))?\s*\)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")
#: ``_group_callout`` prints "H1  2x %6 THRU": the id, two spaces, then the description.
_CALLOUT_RE = re.compile(r"^([HB]\d+)\s{2,}(\S.*)$")
#: The quantity prefix "2x " - a count, not a dimension.  "10 x 4" (a size) has spaces and is
#: deliberately not matched.
_QUANTITY_RE = re.compile(r"(?<![\d.])\d+[xX](?=\s|$)")


def _unescape(text: str) -> str:
    return (text.replace("&quot;", '"').replace("&gt;", ">").replace("&lt;", "<")
            .replace("&amp;", "&"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


# --------------------------------------------------------------------------------------
# what the sheet says: parsed SVG primitives
# --------------------------------------------------------------------------------------


@dataclass
class SheetText:
    """One ``<text>`` on the sheet with its reconstructed paper box, in millimetres.

    ``text_id`` is positional (``t0``, ``t1``, ...) because the drawing module writes no ids;
    it is stable for a given sheet and is what evidence points at.
    """

    text_id: str
    content: str
    x: float
    y: float
    size: float
    anchor: str
    baseline: str
    fill: str
    mono: bool
    rotate: float
    bbox: Tuple[float, float, float, float]

    @property
    def is_dimension(self) -> bool:
        return self.fill.lower() == DIMENSION_FILL

    @property
    def callout_group(self) -> Optional[str]:
        match = _CALLOUT_RE.match(self.content)
        return match.group(1) if match else None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.text_id, "text": self.content,
                "bbox": [round(v, 3) for v in self.bbox]}


@dataclass
class SheetRect:
    """One ``<rect>``; the title block is found among these rather than assumed."""

    x: float
    y: float
    w: float
    h: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def text_width_mm(content: str, size: float, mono: bool) -> float:
    """Estimated advance width of ``content`` in paper millimetres."""
    if mono:
        return len(content) * _MONO_ADVANCE * size
    return sum(_ADVANCE.get(ch, _ADVANCE_DEFAULT) for ch in content) * size


def _rotate_box(box: Tuple[float, float, float, float], degrees: float, cx: float,
                cy: float) -> Tuple[float, float, float, float]:
    """Axis-aligned hull of ``box`` rotated by ``degrees`` about ``(cx, cy)``."""
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    xs: List[float] = []
    ys: List[float] = []
    for px, py in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])):
        dx, dy = px - cx, py - cy
        xs.append(cx + dx * cos - dy * sin)
        ys.append(cy + dx * sin + dy * cos)
    return (min(xs), min(ys), max(xs), max(ys))


def parse_texts(svg_text: str) -> List[SheetText]:
    """Every label on the sheet with its paper box; document order gives the text ids."""
    texts: List[SheetText] = []
    for index, match in enumerate(_TEXT_RE.finditer(svg_text)):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        content = _unescape(match.group(2)).strip()
        if not content:
            continue
        x = _float(attrs.get("x"))
        y = _float(attrs.get("y"))
        size = _float(attrs.get("font-size"), 2.5)
        anchor = attrs.get("text-anchor", "start")
        baseline = attrs.get("dominant-baseline", "auto")
        mono = "mono" in attrs.get("font-family", "").lower()
        width = text_width_mm(content, size, mono)
        if anchor == "middle":
            x0 = x - width / 2.0
        elif anchor == "end":
            x0 = x - width
        else:
            x0 = x
        if baseline in ("middle", "central"):
            y0, y1 = y - size / 2.0, y + size / 2.0
        else:                       # "auto"/"alphabetic": the baseline sits near the bottom
            y0, y1 = y - 0.80 * size, y + 0.20 * size
        box = (x0, y0, x0 + width, y1)
        degrees = 0.0
        rotate = _ROTATE_RE.search(attrs.get("transform", ""))
        if rotate:
            degrees = _float(rotate.group(1))
            centre_x = _float(rotate.group(2), x) if rotate.group(2) else x
            centre_y = _float(rotate.group(3), y) if rotate.group(3) else y
            box = _rotate_box(box, degrees, centre_x, centre_y)
        texts.append(SheetText("t{}".format(index), content, x, y, size, anchor, baseline,
                               attrs.get("fill", "#000"), mono, degrees, box))
    return texts


#: The annotation lines a label must never be struck through by - the classes
#: ``phimech/synth/drawing.py`` writes on dimension, extension and leader segments.
ANNOTATION_LINE_CLASSES: Tuple[str, ...] = ("dim", "ext", "leader")


def parse_lines(svg_text: str) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    """Every classed ``<line>`` as ``(class, p0, p1)``; unclassed lines are geometry."""
    lines: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = []
    for match in _LINE_RE.finditer(svg_text):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        kind = attrs.get("class", "")
        if not kind:
            continue
        lines.append((kind, (_float(attrs.get("x1")), _float(attrs.get("y1"))),
                      (_float(attrs.get("x2")), _float(attrs.get("y2")))))
    return lines


def parse_outlines(svg_text: str) -> List[List[Tuple[float, float]]]:
    """Every view silhouette loop (``<polygon class="outline">``) in paper mm."""
    loops: List[List[Tuple[float, float]]] = []
    for match in _POLYGON_RE.finditer(svg_text):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        if attrs.get("class") != "outline":
            continue
        values = _NUMBER_RE.findall(attrs.get("points", ""))
        points = [(float(values[i]), float(values[i + 1]))
                  for i in range(0, len(values) - 1, 2)]
        if len(points) >= 3:
            loops.append(points)
    return loops


def parse_rects(svg_text: str) -> List[SheetRect]:
    """Every ``<rect>``; ``x``/``y`` default to 0 as SVG says (the sheet background omits them)."""
    rects: List[SheetRect] = []
    for match in _RECT_RE.finditer(svg_text):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        rects.append(SheetRect(_float(attrs.get("x")), _float(attrs.get("y")),
                               _float(attrs.get("width")), _float(attrs.get("height"))))
    return rects


def find_title_block(rects: Sequence[SheetRect], sheet: Tuple[float, float]) -> Optional[SheetRect]:
    """The title block: the largest bottom-right rectangle of plausible size.

    Found rather than assumed, so a layout change in the drawing module cannot silently turn
    this check into a no-op; ``None`` (title block not found) is reported as evidence.
    """
    width, height = sheet
    candidates = [
        rect for rect in rects
        if 60.0 <= rect.w <= 200.0 and 15.0 <= rect.h <= 100.0
        and (rect.x + rect.w / 2.0) > width * 0.5 and (rect.y + rect.h / 2.0) > height * 0.5
    ]
    return max(candidates, key=lambda r: r.w * r.h) if candidates else None


def _overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float],
             slack: float) -> bool:
    return (min(a[2], b[2]) - max(a[0], b[0]) > slack
            and min(a[3], b[3]) - max(a[1], b[1]) > slack)


def _inside(inner: Tuple[float, float, float, float],
            outer: Tuple[float, float, float, float], slack: float) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[2] <= outer[2] + slack and inner[3] <= outer[3] + slack)


def _shrink(box: Tuple[float, float, float, float],
            by: float) -> Tuple[float, float, float, float]:
    return (box[0] + by, box[1] + by, box[2] - by, box[3] - by)


def _point_in_loop(point: Tuple[float, float], loop: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def segment_hits_box(p0: Tuple[float, float], p1: Tuple[float, float],
                     box: Tuple[float, float, float, float]) -> bool:
    """Does the segment p0-p1 pass through the (open) box?  Liang-Barsky clipping."""
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


def box_hits_loop(box: Tuple[float, float, float, float],
                  loop: Sequence[Tuple[float, float]]) -> bool:
    """Does the box overlap the region the closed ``loop`` encloses?"""
    if box[2] <= box[0] or box[3] <= box[1] or len(loop) < 3:
        return False
    xs = [p[0] for p in loop]
    ys = [p[1] for p in loop]
    if box[2] <= min(xs) or max(xs) <= box[0] or box[3] <= min(ys) or max(ys) <= box[1]:
        return False
    corners = ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
    if any(_point_in_loop(c, loop) for c in corners):
        return True
    n = len(loop)
    return any(segment_hits_box(loop[i], loop[(i + 1) % n], box) for i in range(n))


# --------------------------------------------------------------------------------------
# what the geometry measured
# --------------------------------------------------------------------------------------


def measured_values(drawing: Mapping[str, Any]) -> List[float]:
    """Every length the drawing legitimately could print, from the measured result alone.

    A dimension on the sheet is traceable when it equals one of these within
    ``TRACE_TOLERANCE_MM``: an overall size, a view bbox coordinate, a z level or its height
    above the base, a feature size, corner radius, depth or depth bound, a feature centre or
    its offset from the view datum, the diameter or depth of a counterbore, countersink or
    skirt the parent's wall measured, or the pitch between two members of one group.  Anything
    else was not measured - it was typed.
    """
    values: List[float] = [0.0]
    bbox = drawing.get("bbox_mm") or {}
    low = [_float(v) for v in (bbox.get("min") or [])]
    high = [_float(v) for v in (bbox.get("max") or [])]
    values.extend(low)
    values.extend(high)
    for lo, hi in zip(low, high):
        values.append(hi - lo)
    views = drawing.get("views") or {}
    view_boxes: Dict[str, List[float]] = {}
    for name, view in views.items():
        box = [_float(v) for v in ((view or {}).get("bbox") or [])]
        if len(box) == 4:
            view_boxes[name] = box
            values.extend(box)
            values.append(box[2] - box[0])
            values.append(box[3] - box[1])
    levels = [_float(z) for z in (drawing.get("levels_z") or [])]
    if levels:
        base = min(levels)
        values.extend(levels)
        values.extend(level - base for level in levels)
        for i, a in enumerate(levels):
            for b in levels[i + 1:]:
                values.append(abs(b - a))
    for group in drawing.get("groups") or []:
        size = [_float(v) for v in (group.get("size") or [])]
        values.extend(size)
        values.append(_float(group.get("corner_r")))
        values.append(_float(group.get("depth")))
        values.append(_float(group.get("depth_from")))
        values.append(_float(group.get("depth_to")))
        # A counterbore or countersink is a modifier measured off the parent's own coaxial
        # wall, so its diameter and depth are measurements too. Without them the sheet cannot
        # print the modifier and stay traceable, and the only way to keep it traceable would
        # be to leave it standing as a second group at the same positions - a drawing that
        # says six holes where the part has three.
        for key in ("counterbore", "countersink"):
            modifier = group.get(key)
            if isinstance(modifier, Mapping):
                values.extend(_float(v) for v in (modifier.get("size") or []))
                for name in ("diameter", "depth", "from", "to"):
                    values.append(_float(modifier.get(name)))
        for skirt in group.get("skirts") or []:
            if isinstance(skirt, Mapping):
                for name in ("diameter", "reach", "from", "to"):
                    values.append(_float(skirt.get(name)))
        box = view_boxes.get(str(group.get("view")), [0.0, 0.0, 0.0, 0.0])
        positions = [[_float(c) for c in (p or [])] for p in (group.get("positions") or [])]
        for point in positions:
            if len(point) != 2:
                continue
            values.extend(point)
            values.append(point[0] - box[0])
            values.append(point[1] - box[1])
            values.append(box[2] - point[0])
            values.append(box[3] - point[1])
        for i, first in enumerate(positions):
            for second in positions[i + 1:]:
                if len(first) == 2 and len(second) == 2:
                    values.append(abs(second[0] - first[0]))
                    values.append(abs(second[1] - first[1]))
    return [v for v in values if math.isfinite(v)]


def _traceable(value: float, measured: Sequence[float],
               tolerance: float = TRACE_TOLERANCE_MM) -> bool:
    return any(abs(value - m) <= tolerance for m in measured)


def _numbers(content: str) -> List[float]:
    return [float(token) for token in _NUMBER_RE.findall(content)]


def callout_numbers(content: str) -> List[float]:
    """Dimensional tokens of a leader callout.

    The feature id ("H1") and the quantity prefix ("2x") are counts, not lengths, and are
    dropped before the numbers are read - otherwise every callout would claim to print a
    dimension of 1 or 2.
    """
    match = _CALLOUT_RE.match(content)
    body = match.group(2) if match else content
    return _numbers(_QUANTITY_RE.sub(" ", body))


# --------------------------------------------------------------------------------------
# one check
# --------------------------------------------------------------------------------------


@dataclass
class Check:
    """One gate decision.  ``value`` is what was measured, ``threshold`` what it had to meet."""

    name: str
    status: str
    value: Any
    threshold: Any
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("unknown check status {!r}; choose {}".format(self.status, STATUSES))

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "value": self.value,
                "threshold": self.threshold, "evidence": dict(self.evidence)}


def _evidence(note: str, **refs: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {"note": note}
    for key, value in refs.items():
        if value not in (None, [], {}, ""):
            record[key] = value
    return record


@dataclass
class _Sheet:
    """Everything the checks read: the measured result, the parsed sheet, the source manifest."""

    drawing: Mapping[str, Any]
    texts: List[SheetText]
    rects: List[SheetRect]
    svg_error: Optional[str]
    source_manifest: Optional[Mapping[str, Any]]
    intent: str
    stl_path: Optional[Any] = None
    declared_intent: Any = None
    #: Continuation sheets (2, 3, ...) as ``(texts, rects)``: where a feature table too long
    #: for sheet 1 carries on.  Read for what the drawing *says*; collisions are per sheet.
    continuation: List[Tuple[List[SheetText], List[SheetRect]]] = field(default_factory=list)
    #: Sheet 1's classed annotation lines and view silhouettes - what a label must stay off.
    lines: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list)
    outlines: List[List[Tuple[float, float]]] = field(default_factory=list)

    @property
    def all_texts(self) -> List[SheetText]:
        """Every label on every sheet - what the drawing as a whole prints."""
        return list(self.texts) + [t for texts, _ in self.continuation for t in texts]

    @property
    def pages(self) -> List[Tuple[List[SheetText], List[SheetRect]]]:
        return [(self.texts, self.rects)] + list(self.continuation)

    @property
    def sheet_mm(self) -> Tuple[float, float]:
        size = [_float(v) for v in (self.drawing.get("sheet_mm") or [])]
        if len(size) == 2 and size[0] > 0 and size[1] > 0:
            return (size[0], size[1])
        return (420.0, 297.0)

    @property
    def groups(self) -> List[Mapping[str, Any]]:
        return [g for g in (self.drawing.get("groups") or []) if isinstance(g, Mapping)]


# --------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------


def check_views_readable(sheet: _Sheet) -> Check:
    """Every view big enough to read, at a scale from the standard series."""
    scale = _float(sheet.drawing.get("scale"))
    views = sheet.drawing.get("views") or {}
    sizes: List[Tuple[str, float]] = []
    for name, view in sorted(views.items()):
        box = [_float(v) for v in ((view or {}).get("bbox") or [])]
        if len(box) == 4:
            sizes.append((name, max(box[2] - box[0], box[3] - box[1]) * scale))
    if scale <= 0 or not sizes:
        return Check("views_readable", SKIPPED, None, MIN_VIEW_MM,
                     _evidence("the result carries no scale or no view bboxes"))
    off_series = not any(abs(scale - s) <= 1e-9 for s in STANDARD_SCALES)
    smallest = min(sizes, key=lambda item: item[1])
    status = FAIL if (smallest[1] < MIN_VIEW_MM or off_series) else PASS
    note = "smallest view {} is {:.2f} mm across at {:g}:1".format(
        smallest[0].upper(), smallest[1], scale)
    if off_series:
        note += "; scale {:g} is not in the standard series".format(scale)
    return Check("views_readable", status, round(smallest[1], 3), MIN_VIEW_MM,
                 _evidence(note, views=[smallest[0]], scale=scale, off_series=off_series,
                           view_mm={name: round(value, 3) for name, value in sizes}))


def check_text_no_collision(sheet: _Sheet) -> Check:
    """No two labels overlap by more than ``TEXT_OVERLAP_MM`` in both axes."""
    if sheet.svg_error:
        return Check("text_no_collision", SKIPPED, None, 0,
                     _evidence("the sheet could not be read: {}".format(sheet.svg_error)))
    collisions: List[Dict[str, Any]] = []
    for texts, _ in sheet.pages:          # two sheets share coordinates, not paper
        for i, first in enumerate(texts):
            for second in texts[i + 1:]:
                if _overlap(first.bbox, second.bbox, TEXT_OVERLAP_MM):
                    collisions.append({"a": first.to_dict(), "b": second.to_dict()})
    texts = sheet.all_texts
    status = FAIL if collisions else PASS
    note = ("{} overlapping text pairs (slack {} mm)".format(len(collisions), TEXT_OVERLAP_MM)
            if collisions else "{} labels, none overlapping".format(len(texts)))
    return Check("text_no_collision", status, len(collisions), 0,
                 _evidence(note, texts=[pair["a"]["id"] for pair in collisions[:12]],
                           pairs=collisions[:12]))


def check_text_not_clipped(sheet: _Sheet) -> Check:
    """No label leaves the sheet, and no title-block label leaves the title block."""
    if sheet.svg_error:
        return Check("text_not_clipped", SKIPPED, None, 0,
                     _evidence("the sheet could not be read: {}".format(sheet.svg_error)))
    width, height = sheet.sheet_mm
    sheet_box = (0.0, 0.0, width, height)
    block = find_title_block(sheet.rects, sheet.sheet_mm)
    clipped: List[Dict[str, Any]] = []
    for texts, rects in sheet.pages:
        page_block = find_title_block(rects, sheet.sheet_mm)
        for text in texts:
            if not _inside(text.bbox, sheet_box, CLIP_SLACK_MM):
                entry = text.to_dict()
                entry["where"] = "sheet"
                clipped.append(entry)
            elif page_block is not None \
                    and _inside((text.x, text.y, text.x, text.y), page_block.bbox, 0.0) \
                    and not _inside(text.bbox, page_block.bbox, CLIP_SLACK_MM):
                entry = text.to_dict()
                entry["where"] = "title_block"
                clipped.append(entry)
    status = FAIL if clipped else PASS
    note = ("{} labels leave the sheet or the title block".format(len(clipped)) if clipped
            else "{} labels, all inside the sheet{}".format(
                len(sheet.all_texts), " and the title block" if block else
                " (no title block rectangle found)"))
    return Check("text_not_clipped", status, len(clipped), 0,
                 _evidence(note, texts=[entry["id"] for entry in clipped[:12]],
                           clipped=clipped[:12], bbox=[0.0, 0.0, width, height],
                           title_block=[round(v, 3) for v in block.bbox] if block else None))


def check_label_view_overlap(sheet: _Sheet) -> Check:
    """No label is printed over a view: every label sits outside every part silhouette.

    ``text_no_collision`` only compares labels with labels; a callout printed across the part
    it describes collides with *geometry* and hides the edges it names.  The silhouettes are
    read back from the sheet's own ``<polygon class="outline">`` loops, each taken as a filled
    region (a label inside a through-window is still over the view), and a label counts when
    its box, shrunk by ``TEXT_OVERLAP_MM``, overlaps one.  Count, threshold 0."""
    name = "label_view_overlap"
    if sheet.svg_error:
        return Check(name, SKIPPED, None, 0,
                     _evidence("the sheet could not be read: {}".format(sheet.svg_error)))
    if not sheet.outlines:
        return Check(name, NOT_APPLICABLE, 0, 0,
                     _evidence("the sheet marks no view outline (no polygon class=outline)"))
    over: List[Dict[str, Any]] = []
    for text in sheet.texts:
        box = _shrink(text.bbox, TEXT_OVERLAP_MM)
        if any(box_hits_loop(box, loop) for loop in sheet.outlines):
            over.append(text.to_dict())
    status = FAIL if over else PASS
    note = ("{} labels printed over a view".format(len(over)) if over else
            "{} labels, none over the {} view outline loops".format(len(sheet.texts),
                                                                     len(sheet.outlines)))
    return Check(name, status, len(over), 0,
                 _evidence(note, texts=[entry["id"] for entry in over[:12]], over=over[:12]))


def check_label_line_overlap(sheet: _Sheet) -> Check:
    """No label is struck through by a dimension, extension or leader line.

    Ordinate values stacked in columns used to be crossed by the extension line of the value
    one column further out, and a struck-through number is misread.  Every classed annotation
    segment is tested against every label box shrunk by ``TEXT_OVERLAP_MM``; a leader or
    extension line that *ends* beside its own label does not enter the shrunk box.
    Count of (label, line) pairs, threshold 0."""
    name = "label_line_overlap"
    if sheet.svg_error:
        return Check(name, SKIPPED, None, 0,
                     _evidence("the sheet could not be read: {}".format(sheet.svg_error)))
    lines = [entry for entry in sheet.lines if entry[0] in ANNOTATION_LINE_CLASSES]
    struck: List[Dict[str, Any]] = []
    for text in sheet.texts:
        box = _shrink(text.bbox, TEXT_OVERLAP_MM)
        for kind, p0, p1 in lines:
            if segment_hits_box(p0, p1, box):
                entry = text.to_dict()
                entry["line"] = {"class": kind, "from": [round(p0[0], 3), round(p0[1], 3)],
                                 "to": [round(p1[0], 3), round(p1[1], 3)]}
                struck.append(entry)
    status = FAIL if struck else PASS
    note = ("{} labels struck through by an annotation line".format(len(struck)) if struck
            else "{} labels clear of {} dimension, extension and leader lines".format(
                len(sheet.texts), len(lines)))
    return Check(name, status, len(struck), 0,
                 _evidence(note, texts=[entry["id"] for entry in struck[:12]],
                           struck=struck[:12]))


def locatable(offsets: Sequence[float], printed: Sequence[float],
              tolerance: float = POSITION_TOLERANCE_MM) -> "set":
    """Indices of ``offsets`` a caliper user can reach from the datum with ``printed`` values.

    A feature is located along an axis either directly - a printed dimension equals its offset
    from the view datum - or by a chain: a printed dimension equals the pitch from a feature
    that is already located.  That is exactly how the sheet dimensions a row of holes (one
    from the datum, then the pitch to its neighbour), so the check follows the same chain
    instead of demanding a datum dimension per hole.
    """
    located = {index for index, value in enumerate(offsets)
               if any(abs(value - p) <= tolerance for p in printed)}
    changed = True
    while changed:
        changed = False
        for index, value in enumerate(offsets):
            if index in located:
                continue
            for anchor in list(located):
                pitch = abs(value - offsets[anchor])
                if any(abs(pitch - p) <= tolerance for p in printed):
                    located.add(index)
                    changed = True
                    break
    return located


def check_feature_coverage(sheet: _Sheet) -> Check:
    """Every feature *member* is callable out and locatable, not just every group.

    Group-level coverage passes while an individual hole has no dimension anywhere on the
    sheet, which is the defect a machinist hits first, so the gate is member coverage == 100%:
    each member needs its group's leader callout **and** a printed path from the view datum in
    both axes (see :func:`locatable`).  Matching is on the printed value, because that is what
    the sheet actually shows.
    """
    groups = sheet.groups
    if not groups:
        return Check("feature_coverage", NOT_APPLICABLE, 0, 0,
                     _evidence("the part has no feature groups to call out"))
    if sheet.svg_error:
        return Check("feature_coverage", SKIPPED, None, 0,
                     _evidence("the sheet could not be read: {}".format(sheet.svg_error)))
    callouts = {text.callout_group for text in sheet.all_texts if text.callout_group}
    printed: List[float] = []
    for text in sheet.all_texts:
        if text.is_dimension:
            printed.extend(abs(value) for value in _numbers(text.content))
    views = sheet.drawing.get("views") or {}

    # Members are pooled per view: an offset is only meaningful against its own view's datum.
    by_view: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups:
        gid = str(group.get("id"))
        view = str(group.get("view"))
        box = [_float(v) for v in (((views.get(view) or {}).get("bbox")) or [])]
        for order, point in enumerate(group.get("positions") or []):
            values = [_float(c) for c in (point or [])]
            member = {
                "member": "{}#{}".format(gid, order),
                "group": gid,
                "callout": gid in callouts,
                "dx": values[0] - box[0] if len(values) == 2 and len(box) == 4 else None,
                "dy": values[1] - box[1] if len(values) == 2 and len(box) == 4 else None,
            }
            by_view.setdefault(view, []).append(member)

    uncovered: List[Dict[str, Any]] = []
    total = 0
    for members in by_view.values():
        total += len(members)
        known = [m for m in members if m["dx"] is not None and m["dy"] is not None]
        located_x = locatable([m["dx"] for m in known], printed)
        located_y = locatable([m["dy"] for m in known], printed)
        for index, member in enumerate(known):
            member["located_x"] = index in located_x
            member["located_y"] = index in located_y
        for member in members:
            if member["dx"] is None or member["dy"] is None:
                member["located_x"] = member["located_y"] = False
            if not (member["callout"] and member["located_x"] and member["located_y"]):
                uncovered.append(member)
    groups_missing = sorted({entry["group"] for entry in uncovered})
    status = FAIL if uncovered else PASS
    note = ("{} of {} members are not callable out and locatable ({})".format(
        len(uncovered), total, ", ".join(groups_missing))
        if uncovered else
        "all {} members of {} groups have a callout and a located position".format(
            total, len(groups)))
    return Check("feature_coverage", status, len(uncovered), 0,
                 _evidence(note, groups=groups_missing,
                           members=[entry["member"] for entry in uncovered[:12]],
                           detail=uncovered[:12],
                           member_coverage="{}/{}".format(total - len(uncovered), total)))


def check_dimension_traceability(sheet: _Sheet) -> Check:
    """Every printed dimension equals something the geometry measured, within 0.01 mm."""
    if sheet.svg_error:
        return Check("dimension_traceability", SKIPPED, None, TRACE_TOLERANCE_MM,
                     _evidence("the sheet could not be read: {}".format(sheet.svg_error)))
    measured = measured_values(sheet.drawing)
    if not measured:
        return Check("dimension_traceability", SKIPPED, None, TRACE_TOLERANCE_MM,
                     _evidence("the result carries no measurements to trace against"))
    untraceable: List[Dict[str, Any]] = []
    checked = 0
    for text in sheet.all_texts:
        if text.is_dimension:
            numbers = _numbers(text.content)
        elif text.callout_group:
            numbers = callout_numbers(text.content)
        else:
            continue
        for number in numbers:
            checked += 1
            # Both signs: a sheet prints lengths unsigned but a datum note prints the view's
            # bbox corner, which is negative for geometry modelled about the origin.
            if not (_traceable(number, measured) or _traceable(abs(number), measured)):
                entry = text.to_dict()
                entry["value"] = number
                entry["nearest_measured"] = round(
                    min(measured, key=lambda m: abs(abs(number) - m)), 4)
                untraceable.append(entry)
    status = FAIL if untraceable else PASS
    note = ("{} of {} printed values match no measurement".format(len(untraceable), checked)
            if untraceable else "all {} printed values trace to a measurement".format(checked))
    return Check("dimension_traceability", status, len(untraceable), 0,
                 _evidence(note, texts=[entry["id"] for entry in untraceable[:12]],
                           values=untraceable[:12], tolerance_mm=TRACE_TOLERANCE_MM,
                           checked=checked))


def check_source_translation_clean(sheet: _Sheet) -> Check:
    """No error-severity translation diagnostic, and the producer declared the export good."""
    manifest = sheet.source_manifest
    if not isinstance(manifest, Mapping):
        return Check("source_translation_clean", NOT_APPLICABLE, None, 0,
                     _evidence("no source manifest: the geometry did not come from a "
                               "translated export"))
    translation = manifest.get("translation")
    if translation is None and not manifest.get("errors") and "source" not in manifest:
        # A PhiMech design IR handed over as the build's manifest: the mesh came straight from
        # the CSG, nothing was translated, so there is no export to have gone wrong.  A
        # SolidWorks source.json always carries `translation` (and `source`), and one that
        # does not is still judged below.
        return Check("source_translation_clean", NOT_APPLICABLE, None, 0,
                     _evidence("manifest is a design IR, not a translated export: no "
                               "translation block to judge"))
    translation = translation if isinstance(translation, Mapping) else {}
    diagnostics = [d for d in (translation.get("diagnostics") or []) if isinstance(d, Mapping)]
    errors = [d for d in diagnostics if str(d.get("severity", "")).lower() == "error"]
    errors.extend({"severity": "error", "message": str(m)}
                  for m in (manifest.get("errors") or []))
    ok = translation.get("ok")
    status = FAIL if (errors or ok is not True) else PASS
    note = ("{} translation error(s); translation.ok={!r}".format(len(errors), ok) if
            status == FAIL else "translation clean: {} diagnostic(s), none an error".format(
                len(diagnostics)))
    return Check("source_translation_clean", status, len(errors), 0,
                 _evidence(note, translation_ok=ok, diagnostics=errors[:8],
                           warnings=[str(w) for w in (manifest.get("warnings") or [])][:8]))


def check_manufacturing_intent_present(sheet: _Sheet) -> Check:
    """Material / tolerance / finish: required for a release, listed as absent for inspection.

    An inspection drawing is explicitly *not* a manufacturing release (proposal section 2), so
    this is ``not_applicable`` and reports which intent items the sheet does not carry.  The
    same code becomes a gate the moment a caller declares ``intent="manufacturing"``, so the
    absence is recorded rather than assumed.
    """
    manifest = sheet.source_manifest if isinstance(sheet.source_manifest, Mapping) else {}
    component = manifest.get("component")
    component = component if isinstance(component, Mapping) else {}
    printed = " ".join(text.content.lower() for text in sheet.all_texts)
    # Tolerance and finish are read from the *resolved* block, never from the sheet's words.
    # Grepping for "tolerance" passes on a sheet whose tolerance region reads
    # "GENERAL TOLERANCES: UNSPECIFIED", and on the note warning that the measurement
    # resolution is NOT a manufacturing tolerance - i.e. the two texts that exist precisely
    # because nothing was specified would report it as specified.
    block = sheet.drawing.get("tolerance")
    block = block if isinstance(block, Mapping) else {}
    finish = block.get("finish")
    finish = finish.get("value") if isinstance(finish, Mapping) else finish
    present: List[str] = []
    absent: List[str] = []
    supplied = {
        "material": bool(component.get("material")) or bool(sheet.drawing.get("material")),
        "tolerance": bool(block.get("specified")),
        "finish": bool(finish),
        "thread": bool(re.search(r"\bm\d+(?:x[\d.]+)?\b", printed)),
        "gdt": bool(re.search(r"[⌖⏥⟂∥]", printed)),
    }
    for item in INTENT_ITEMS:
        (present if supplied[item] else absent).append(item)
    if sheet.intent != "manufacturing":
        return Check("manufacturing_intent_present", NOT_APPLICABLE, absent, [],
                     _evidence("inspection intent: the sheet carries no {} and says so; "
                               "nothing here is a manufacturing release".format(
                                   ", ".join(absent) or "gaps"),
                               present=present, absent=absent, intent=sheet.intent))
    status = FAIL if absent else PASS
    return Check("manufacturing_intent_present", status, absent, [],
                 _evidence("manufacturing intent: {} absent".format(", ".join(absent))
                           if absent else "material, tolerance and finish are all stated",
                           present=present, absent=absent, intent=sheet.intent))



def horizontal_face_area(triangles: Iterable[Any], level: float,
                         tolerance: float = LEVEL_FACE_TOLERANCE_MM) -> float:
    """Total area of horizontal facets lying at ``level``, in mm2.

    All three vertices within ``tolerance`` of the level *is* horizontality, so the projected
    XY area is the true area and no normal is needed.  A tessellation ring on a chamfer - the
    thing that makes a fabricated z level look measured - contributes exactly zero.
    """
    total = 0.0
    for triangle in triangles:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
        if max(abs(az - level), abs(bz - level), abs(cz - level)) > tolerance:
            continue
        total += abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
    return total


def _level_areas(sheet: _Sheet, levels: Sequence[float]) -> Tuple[Optional[List[float]], str]:
    """Face area per level, from a precomputed block or by reading the mesh."""
    precomputed = sheet.drawing.get("level_faces")
    if isinstance(precomputed, Sequence) and not isinstance(precomputed, (str, bytes)):
        table = {round(_float(entry.get("z")), 3): _float(entry.get("area_mm2"))
                 for entry in precomputed if isinstance(entry, Mapping)}
        if table:
            return [table.get(round(level, 3), 0.0) for level in levels], "level_faces"
    if sheet.stl_path is None:
        return None, "no mesh and no level_faces block"
    try:
        from phimech.check.mesh import read_stl

        triangles = read_stl(str(sheet.stl_path))
    except Exception as exc:                      # noqa: BLE001 - an unreadable mesh is a skip
        return None, "{}: {}".format(Path(str(sheet.stl_path)).name, exc)
    return [horizontal_face_area(triangles, level) for level in levels], str(sheet.stl_path)


def check_levels_are_faces(sheet: _Sheet) -> Check:
    """Every printed z level is a real horizontal face, not a vertex coincidence.

    This is the check ``dimension_traceability`` cannot be: a level clustered out of the
    vertices of a chamfer or a fillet is a value the pipeline genuinely "measured", so it
    traces perfectly while describing a face that does not exist - and every depth quoted
    against it is fabricated.  A real face has area; a tessellation ring has none.
    """
    levels = [_float(z) for z in (sheet.drawing.get("levels_z") or [])]
    if not levels:
        return Check("levels_are_faces", NOT_APPLICABLE, 0, MIN_FACE_AREA_MM2,
                     _evidence("the result prints no z levels"))
    areas, origin = _level_areas(sheet, levels)
    if areas is None:
        return Check("levels_are_faces", SKIPPED, None, MIN_FACE_AREA_MM2,
                     _evidence("cannot verify levels against geometry: {}".format(origin)))
    phantom = [{"z": round(level, 3), "face_area_mm2": round(area, 3)}
               for level, area in zip(levels, areas) if area < MIN_FACE_AREA_MM2]
    status = FAIL if phantom else PASS
    note = ("{} of {} printed z levels have no horizontal face >= {} mm2: {}".format(
        len(phantom), len(levels), MIN_FACE_AREA_MM2,
        ", ".join("z={}".format(entry["z"]) for entry in phantom))
        if phantom else
        "all {} printed z levels are real faces (smallest {:.2f} mm2)".format(
            len(levels), min(areas)))
    return Check("levels_are_faces", status, len(phantom), 0,
                 _evidence(note, levels=[entry["z"] for entry in phantom], phantom=phantom[:12],
                           face_area_threshold_mm2=MIN_FACE_AREA_MM2, source=origin,
                           areas=[round(area, 3) for area in areas]))


def declared_features(intent: Any) -> List[Dict[str, Any]]:
    """Normalise a declared-intent source into one record per feature *instance*.

    Understood without guessing: a B-rep feature program (``{"ops": [...]}``), any block with
    a ``features`` list (a SolidWorks ``source.json`` or a synth report), or that list on its
    own.  Anything else yields nothing, which makes the check ``not_applicable`` - intent is
    read, never inferred from the geometry it is supposed to be checked against.
    """
    if intent is None:
        return []
    if isinstance(intent, Mapping):
        if isinstance(intent.get("ops"), Sequence):
            return _features_from_ops(intent["ops"])
        raw = intent.get("features")
    else:
        raw = intent
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    features: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            continue
        kind = str(entry.get("kind") or entry.get("type") or "feature")
        diameter = entry.get("diameter")
        depth = entry.get("depth")
        positions = [[_float(c) for c in (p or [])] for p in (entry.get("positions") or [])
                     if isinstance(p, Sequence) and len(list(p)) == 2]
        count = int(entry.get("count") or len(positions) or 1)
        where = "features[{}]".format(index)
        if positions:
            features.extend(
                {"kind": kind, "diameter": _float(diameter) if diameter is not None else None,
                 "depth": _float(depth) if depth is not None else None,
                 "position": point, "source": where}
                for point in positions)
        else:
            features.extend(
                {"kind": kind, "diameter": _float(diameter) if diameter is not None else None,
                 "depth": _float(depth) if depth is not None else None,
                 "position": None, "source": where}
                for _ in range(max(count, 0)))
    return features


def _features_from_ops(ops: Sequence[Any]) -> List[Dict[str, Any]]:
    """Hole-family instances of a B-rep feature program, counterbores and pilots included."""
    features: List[Dict[str, Any]] = []
    for index, op in enumerate(ops):
        if not isinstance(op, Mapping):
            continue
        name = str(op.get("op"))
        points = [[_float(c) for c in p] for p in (op.get("at") or [])
                  if isinstance(p, Sequence) and not isinstance(p, (str, bytes))
                  and len(list(p)) == 2]
        if name == "hole":
            diameter = _float(op.get("diameter"))
            depth = _float(op["depth"]) if op.get("depth") is not None else None
            for point in points:
                features.append({"kind": "hole", "diameter": diameter, "depth": depth,
                                 "position": point, "source": "ops[{}]".format(index)})
            for extra in ("counterbore", "countersink"):
                spec = op.get(extra)
                if not isinstance(spec, Mapping):
                    continue
                for point in points:
                    features.append({
                        "kind": extra, "diameter": _float(spec.get("diameter")),
                        "depth": _float(spec.get("depth")), "position": point,
                        "source": "ops[{}].{}".format(index, extra)})
        elif name == "tapped_hole":
            size = str(op.get("size"))
            pitch = THREAD_PITCH_MM.get(size)
            # Same tapping-drill approximation the B-rep worker cuts with: no helix is
            # modelled, so the measurable feature is the minor-diameter pilot.
            diameter = (float(size[1:]) - 1.0825 * pitch) if pitch else None
            for point in points:
                features.append({
                    "kind": "tapped_hole", "diameter": diameter,
                    "depth": _float(op["depth"]) if op.get("depth") is not None else None,
                    "position": point, "source": "ops[{}]".format(index)})
    return features


def measured_members(drawing: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One record per measured feature instance, for matching against declared intent."""
    members: List[Dict[str, Any]] = []
    for group in drawing.get("groups") or []:
        if not isinstance(group, Mapping):
            continue
        size = [_float(v) for v in (group.get("size") or [0.0, 0.0])]
        for order, point in enumerate(group.get("positions") or []):
            values = [_float(c) for c in (point or [])]
            members.append({
                "member": "{}#{}".format(group.get("id"), order),
                "group": str(group.get("id")),
                "kind": str(group.get("kind")),
                "hole": bool(group.get("hole")),
                "diameter": size[0] if size else 0.0,
                "size": size,
                "depth": _float(group.get("depth")),
                "position": values if len(values) == 2 else None,
            })
    return members


def _matches(feature: Mapping[str, Any], member: Mapping[str, Any],
             tolerance: float) -> bool:
    if str(feature["kind"]) in HOLE_KINDS and not member["hole"]:
        return False
    position = feature.get("position")
    if position is not None and member["position"] is not None:
        if (abs(position[0] - member["position"][0]) > tolerance
                or abs(position[1] - member["position"][1]) > tolerance):
            return False
    diameter = feature.get("diameter")
    if diameter is not None:
        if min(abs(diameter - member["diameter"]),
               abs(diameter - (member["size"][1] if len(member["size"]) > 1 else
                               member["diameter"]))) > tolerance:
            return False
    return True


def check_intent_vs_measured(sheet: _Sheet) -> Check:
    """Every declared feature is present in the measured geometry, within 0.05 mm.

    The only check that can see a feature that was *asked for and never built*: three
    counterbores in the feature program, none in the solid, every other check green because
    the drawing faithfully describes the solid it was given.  Declared intent is optional and
    is never inferred - without a source this is ``not_applicable``.
    """
    features = declared_features(sheet.declared_intent)
    if not features:
        return Check("intent_vs_measured", NOT_APPLICABLE, None, 0,
                     _evidence("no declared-intent source (feature program, synth report or "
                               "source.json features[]) was supplied"))
    members = measured_members(sheet.drawing)
    available = list(range(len(members)))
    unmatched: List[Dict[str, Any]] = []
    tally: Dict[str, List[int]] = {}
    for feature in features:
        counts = tally.setdefault(str(feature["kind"]), [0, 0])
        counts[0] += 1
        hit = next((i for i in available
                    if _matches(feature, members[i], INTENT_MATCH_TOLERANCE_MM)), None)
        if hit is None:
            unmatched.append({"kind": feature["kind"], "diameter": feature["diameter"],
                              "position": feature["position"], "source": feature["source"]})
        else:
            counts[1] += 1
            available.remove(hit)
    status = FAIL if unmatched else PASS
    shortfalls = ["{} declared {}{}, {} found".format(
        declared, kind, "" if declared == 1 else "s", found)
        for kind, (declared, found) in sorted(tally.items()) if found < declared]
    note = ("; ".join(shortfalls) if shortfalls else
            "all {} declared features matched a measured feature within {} mm".format(
                len(features), INTENT_MATCH_TOLERANCE_MM))
    return Check("intent_vs_measured", status, len(unmatched), 0,
                 _evidence(note, declared=len(features), matched=len(features) - len(unmatched),
                           missing=unmatched[:12], tolerance_mm=INTENT_MATCH_TOLERANCE_MM,
                           by_kind={kind: {"declared": d, "found": f}
                                    for kind, (d, f) in sorted(tally.items())}))


_CHECKS = (
    check_views_readable,
    check_text_no_collision,
    check_text_not_clipped,
    check_label_view_overlap,
    check_label_line_overlap,
    check_feature_coverage,
    check_dimension_traceability,
    check_levels_are_faces,
    check_intent_vs_measured,
    check_source_translation_clean,
    check_manufacturing_intent_present,
)


# --------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------


def summarize(checks: Sequence[Check]) -> Dict[str, Any]:
    """``{status, first_failure, counts}``: one fail fails the sheet, one skip needs a human."""
    counts = {status: 0 for status in STATUSES}
    for check in checks:
        counts[check.status] += 1
    first_failure = next((c.name for c in checks if c.status == FAIL), None)
    if first_failure is not None:
        status = SUMMARY_FAIL
    elif counts[SKIPPED]:
        status = SUMMARY_NEEDS_REVIEW
    else:
        status = SUMMARY_PASS
    return {"status": status, "first_failure": first_failure, "counts": counts}


def _as_mapping(drawing: Any) -> Mapping[str, Any]:
    if isinstance(drawing, Mapping):
        return drawing
    to_dict = getattr(drawing, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("drawing must be a mapping or expose to_dict(); got {}".format(
        type(drawing).__name__))


def _load_svg(drawing: Mapping[str, Any], svg: Optional[str],
              svg_path: Optional[Any]) -> Tuple[str, Optional[str]]:
    if svg is not None:
        return svg, None
    path = svg_path if svg_path is not None else drawing.get("path")
    if not path:
        return "", "no SVG text and no path in the drawing result"
    try:
        return Path(str(path)).read_text(encoding="utf-8"), None
    except OSError as exc:
        return "", "{}: {}".format(Path(str(path)).name, exc)


def _load_continuation(drawing: Mapping[str, Any]) -> Tuple[
        List[Tuple[List[SheetText], List[SheetRect]]], Optional[str]]:
    """Sheets 2, 3, ... named in the result's ``sheets``; (pages, error)."""
    pages: List[Tuple[List[SheetText], List[SheetRect]]] = []
    for entry in list(drawing.get("sheets") or [])[1:]:
        path = entry.get("svg") if isinstance(entry, Mapping) else None
        if not path:
            continue
        try:
            text = Path(str(path)).read_text(encoding="utf-8")
        except OSError as exc:
            return pages, "{}: {}".format(Path(str(path)).name, exc)
        pages.append((parse_texts(text), parse_rects(text)))
    return pages, None


def run_checks(
    drawing: Any,
    *,
    svg: Optional[str] = None,
    svg_path: Optional[Any] = None,
    source_manifest: Optional[Mapping[str, Any]] = None,
    intent: str = "inspection",
    stl_path: Optional[Any] = None,
    declared_intent: Any = None,
) -> Dict[str, Any]:
    """Run every deterministic gate over one drawing and return the checks block.

    ``drawing`` is a :class:`phimech.synth.drawing.DrawingResult` or its ``to_dict()``.  The
    sheet is read from ``svg`` (text), ``svg_path``, or the result's own ``path``; when it
    cannot be read the sheet-dependent checks are ``skipped``, never silently passed, and the
    summary becomes ``needs_review``.

    ``stl_path`` is the mesh the drawing was measured from; without it (and without a
    ``level_faces`` block in the result) ``levels_are_faces`` is skipped rather than assumed.
    ``declared_intent`` is a B-rep feature program, a synth report or a ``source.json``
    ``features[]`` block; without one ``intent_vs_measured`` is ``not_applicable``.
    """
    record = _as_mapping(drawing)
    svg_text, svg_error = _load_svg(record, svg, svg_path)
    continuation, continuation_error = _load_continuation(record)
    # a continuation sheet that cannot be read is a sheet that cannot be checked
    svg_error = svg_error or continuation_error
    sheet = _Sheet(
        drawing=record,
        texts=parse_texts(svg_text) if not svg_error else [],
        rects=parse_rects(svg_text) if not svg_error else [],
        svg_error=svg_error,
        source_manifest=source_manifest,
        intent=intent,
        stl_path=stl_path,
        declared_intent=declared_intent,
        continuation=continuation if not svg_error else [],
        lines=parse_lines(svg_text) if not svg_error else [],
        outlines=parse_outlines(svg_text) if not svg_error else [],
    )
    checks = [check(sheet) for check in _CHECKS]
    return {
        "schema_version": SCHEMA_VERSION,
        "intent": intent,
        "checks": [check.to_dict() for check in checks],
        "summary": summarize(checks),
    }


def printed_inventory(drawing: Any, *, svg: Optional[str] = None,
                      svg_path: Optional[Any] = None) -> Dict[str, Any]:
    """What is actually printed on the sheet: feature ids, numbers, and the raw label text.

    ``phimech/agent/drawing_review.py`` uses this as the allowlist behind its invention filter:
    a model finding may reference these ids and these values, and nothing else.
    """
    record = _as_mapping(drawing)
    svg_text, svg_error = _load_svg(record, svg, svg_path)
    continuation, continuation_error = _load_continuation(record)
    svg_error = svg_error or continuation_error
    texts = parse_texts(svg_text) if not svg_error else []
    texts += [t for page, _ in continuation for t in page] if not svg_error else []
    ids = sorted({str(group.get("id")) for group in (record.get("groups") or [])
                  if isinstance(group, Mapping) and group.get("id")})
    printed_ids = sorted({text.callout_group for text in texts if text.callout_group})
    numbers: List[float] = []
    for text in texts:
        numbers.extend(abs(value) for value in _numbers(text.content))
    return {
        "feature_ids": sorted(set(ids) | set(printed_ids)),
        "numbers": sorted(set(round(value, 4) for value in numbers)),
        "measured": sorted(set(round(abs(value), 4) for value in measured_values(record))),
        "labels": [text.content for text in texts],
        "svg_error": svg_error,
    }


__all__ = [
    "CHECK_NAMES",
    "HOLE_KINDS",
    "INTENT_MATCH_TOLERANCE_MM",
    "LEVEL_FACE_TOLERANCE_MM",
    "MIN_FACE_AREA_MM2",
    "THREAD_PITCH_MM",
    "declared_features",
    "horizontal_face_area",
    "locatable",
    "measured_members",
    "CLIP_SLACK_MM",
    "Check",
    "DIMENSION_FILL",
    "FAIL",
    "INTENT_ITEMS",
    "MIN_VIEW_MM",
    "NOT_APPLICABLE",
    "PASS",
    "POSITION_TOLERANCE_MM",
    "SCHEMA_VERSION",
    "SKIPPED",
    "STATUSES",
    "SUMMARY_FAIL",
    "SUMMARY_NEEDS_REVIEW",
    "SUMMARY_PASS",
    "SheetRect",
    "SheetText",
    "TEXT_OVERLAP_MM",
    "TRACE_TOLERANCE_MM",
    "callout_numbers",
    "find_title_block",
    "measured_values",
    "ANNOTATION_LINE_CLASSES",
    "box_hits_loop",
    "parse_lines",
    "parse_outlines",
    "parse_rects",
    "parse_texts",
    "segment_hits_box",
    "printed_inventory",
    "run_checks",
    "summarize",
    "text_width_mm",
]
