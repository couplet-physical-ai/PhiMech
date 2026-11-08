"""DXF writer: PhiMech's drawings as *editable* CAD geometry, for SolidWorks and everything else.

Why this exists
---------------
``drawing.py`` produces an SVG inspection sheet — perfect to print, impossible to edit. A
mechanical engineer asked for "a SolidWorks native format to open the drawings so that it can
be edited there". ``.SLDDRW`` is a closed, undocumented OLE binary that cannot be written
without a licensed SolidWorks on Windows (``docs/README.md``), so **DXF is the
realistic editable target**: SolidWorks imports it directly into a drawing or a sketch, and so
does every other CAD, CAM and laser tool.

The decisive design rule: **true entities, never tessellation**
---------------------------------------------------------------
OpenSCAD's 2D projection output — the very thing the SVG sheet draws — is a polygon soup: a
Ø2.5 hole is a 48-sided polygon. Dumping that to DXF would hand the recipient 48 sketch
segments per hole, which is useless for manufacturing and miserable to edit. So this module
does **not** re-export the projection. It emits from the *already classified, already fitted*
feature data (``Shape``/``Feature``/``Group`` in ``drawing.py``), where a hole is a centre and a
diameter measured from the mesh:

* fitted circle   -> ``CIRCLE`` (exact analytic entity — this is the single hard requirement)
* rounded slot    -> 4x ``ARC`` + up to 4x ``LINE`` rebuilt from bbox and corner
  radius (an obround's straight ends are zero length and are not written)
* rectangle       -> closed ``LWPOLYLINE`` (a genuine polyline: it *is* four straight edges)
* generic profile -> greedy line/arc decomposition (:func:`decompose_loop`); a polyline only
  where no analytic fit is honest, and never a fit deviating by more than ``curve_tol``
* holes/bosses    -> ``CIRCLE`` plus ``CENTER``-layer centre marks and a leader callout

Every reconstruction is verified against the source points before it is written. If a rounded
rectangle does not actually lie on the arcs it claims, the raw polyline is emitted and a warning
is recorded — a wrong analytic entity is worse than an honest polyline.

Why AC1015 (R2000)
------------------
R12/AC1009 is maximally compatible and **disqualified**: it has no ``$INSUNITS``, so an import
wizard has nothing to default its units from and a silent 25.4x scale error is one click away
(trap #2 of ``docs/README.md``). It also has no ``LWPOLYLINE``, no ``MTEXT`` and no
lineweights. AC1015 is the oldest release with all of those, is what "DXF 2000" means in every
importer's dropdown, and is routinely fed to SolidWorks' DXF/DWG import wizard. It costs
handles, owner back-pointers, subclass markers, a ``BLOCK_RECORD`` table and an ``OBJECTS`` root
dictionary — all written here, because an R2000 file missing them is the same class of bug as a
referenced-but-undefined layer.

Why 1:1 in model space, with a frame instead of a scale
-------------------------------------------------------
A DXF imported into SolidWorks lands at the coordinates in the file, so a scaled sheet would
silently hand the recipient geometry at the wrong size. Everything here is therefore **1:1 in
millimetres**, and the **top view sits at true model coordinates**: a hole measured in the DXF
is the same number as the hole in the STEP and in the IR. Nothing is ever multiplied. To still
read as a drawing, the smallest ISO sheet (A4…A0) that *contains* the 1:1 content is drawn as a
border on the ``FRAME`` layer around it — the sheet is chosen to fit the geometry, never the
other way round. Annotation is sized from the part (ISO 3098 heights) and ``$LTSCALE`` follows.

Dimensions: real ``DIMENSION`` entities, no text overrides
-----------------------------------------------------------
Exploded dimension geometry would be lines and text a recipient must delete before they can
re-dimension anything. This module writes real ``DIMENSION`` entities (rotated/linear) against
an ``ISO-25`` ``DIMSTYLE``, each with its anonymous ``*D<n>`` geometry block in ``BLOCKS``, so a
reader that regenerates dimensions gets clean editable ones and a reader that does not still
shows the rendered picture. **No dimension carries a text override**: the text a reader shows is
the measurement it computes from the definition points, formatted by ``$DIMLUNIT``/``$DIMDEC``/
``$DIMZIN``/``$DIMDSEP``. That is also why hole *callouts* ("H2  4x %%c2.50 DEPTH 1.4") are
leader notes on the ``ANNOTATION`` layer rather than dimensions with an override: quantity,
depth and group ID are not measurements of one distance, and over-dimensioning a hole that the
callout and the hole table already describe is bad drafting, not extra information.

Imported dimensions arrive **editable, not associative**. Nothing in a DXF can be associative to
a SolidWorks model that does not exist yet; moving one converts it to SolidWorks' own style.

Honesty rules (the core boundary)
---------------------------------------
Millimetres and degrees, no unit fields, no conversions. Every number written comes from
measured geometry: no invented tolerance, thread, material or finish appears anywhere, and the
notes block says so on the sheet. :func:`inspect_dxf` and :func:`check_fidelity` re-read the
written file and return a list of numeric failures (radius mismatch, undefined layer or
linetype, dimension text disagreeing with its own definition points, extents not covering the
content) — a gate with thresholds, not a "looks fine".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only. drawing.py is expected to call *into* this
    # module, so this module must never import it at module scope (that would be a cycle).
    from phimech.synth.drawing import DrawingResult, Group, Shape, ViewData

Point = Tuple[float, float]

# --------------------------------------------------------------------------------------
# format constants
# --------------------------------------------------------------------------------------

#: DXF release written. See the module docstring for why R2000 and not R12.
DXF_VERSION = "AC1015"
#: $INSUNITS 4 == millimetres, $MEASUREMENT 1 == metric. Both, so no importer has to guess.
INSUNITS_MM = 4
MEASUREMENT_METRIC = 1
#: Dimension text formatting: decimal units, 2 decimals, trailing zeros suppressed, "." as the
#: decimal separator. Written both as header variables and into the DIMSTYLE record.
DIMLUNIT_DECIMAL = 2
DIMDEC = 2
DIMZIN_SUPPRESS_TRAILING = 8
DIMDSEP_POINT = ord(".")
DIMSTYLE_NAME = "ISO-25"
TEXT_STYLE_NAME = "Standard"
#: A real .shx font name: SolidWorks maps SHX fonts on import, an empty font name is a guess.
TEXT_STYLE_FONT = "txt.shx"

#: ISO 3098 text heights, extended downwards for small parts. Annotation is snapped to this
#: series so a drawing looks like a drawing rather than like an arbitrary scale factor.
TEXT_HEIGHTS: Tuple[float, ...] = (0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 1.8, 2.5, 3.5, 5.0,
                                   7.0, 10.0, 14.0, 20.0)
#: Annotation is this fraction of the larger layout extent before snapping to the series.
TEXT_FRACTION = 1.0 / 40.0

#: ISO A sheets, landscape, largest last. The frame is chosen to fit the 1:1 content.
SHEETS: Tuple[Tuple[str, float, float], ...] = (
    ("A4", 297.0, 210.0), ("A3", 420.0, 297.0), ("A2", 594.0, 420.0),
    ("A1", 841.0, 594.0), ("A0", 1189.0, 841.0),
)
SHEET_MARGIN_MM = 10.0

#: Default max deviation (mm) allowed before an analytic fit is refused and a polyline written.
CURVE_TOL_MM = 0.02
#: An arc must span at least this many degrees and this many segments to beat separate lines.
MIN_ARC_DEG = 8.0
MIN_ARC_SEGMENTS = 3
#: Longest run the greedy fit will consider; the fit is quadratic in the run length, and no
#: real projection loop has a single smooth run longer than this.
MAX_RUN_SEGMENTS = 300
#: Loops larger than this are written as polylines without decomposition (cost control).
MAX_DECOMPOSE_POINTS = 2000

#: Default caps, mirroring drawing.py's sheet policy so the DXF says what the SVG says.
MAX_POSITION_DIMS = 5
MAX_CALLOUTS = 14
MAX_TABLE_ROWS = 18
MAX_LEVEL_CHAIN = 8
MAX_FEATURE_LABELS = 60


@dataclass(frozen=True)
class LinetypeSpec:
    """One ``LTYPE`` record. ``pattern`` is in mm: +dash, -gap, 0 dot; empty means solid."""

    name: str
    description: str
    pattern: Tuple[float, ...] = ()

    @property
    def total(self) -> float:
        return sum(abs(value) for value in self.pattern)


#: Every linetype this module can reference, and nothing it cannot: a referenced-but-undefined
#: linetype is the same bug as a missing layer. ByBlock/ByLayer are required by AutoCAD.
LINETYPES: Tuple[LinetypeSpec, ...] = (
    LinetypeSpec("ByBlock", ""),
    LinetypeSpec("ByLayer", ""),
    LinetypeSpec("CONTINUOUS", "Solid line"),
    LinetypeSpec("HIDDEN", "Hidden __ __ __ __ __ __ __ __", (4.0, -2.0)),
    LinetypeSpec("CENTER", "Center ____ _ ____ _ ____ _ ____", (20.0, -2.0, 2.0, -2.0)),
)


@dataclass(frozen=True)
class LayerSpec:
    """One ``LAYER`` record: ACI colour, linetype, lineweight in 1/100 mm, and what it holds."""

    name: str
    color: int
    linetype: str
    lineweight: int
    purpose: str


#: The engineering layer scheme. SolidWorks carries layer, colour and line style through the
#: import, so deleting one layer must remove exactly one *kind* of information: a recipient can
#: strip all annotation and keep the geometry, or the reverse, in one click.
LAYERS: Tuple[LayerSpec, ...] = (
    LayerSpec("0", 7, "CONTINUOUS", -3, "DXF default layer, mandatory; PhiMech writes nothing "
                                        "on it"),
    LayerSpec("VISIBLE", 7, "CONTINUOUS", 50, "part outline (silhouette) of each view"),
    LayerSpec("HIDDEN", 8, "HIDDEN", 25, "hole extents projected from another view"),
    LayerSpec("CENTER", 1, "CENTER", 25, "centre marks and centre lines of circular features"),
    LayerSpec("SECTION", 9, "CONTINUOUS", 18,
              "feature edges at other levels; visibility not resolved (no hidden-line removal)"),
    LayerSpec("DIMENSIONS", 5, "CONTINUOUS", 18, "DIMENSION entities and their geometry blocks"),
    LayerSpec("ANNOTATION", 3, "CONTINUOUS", 18, "leader callouts, feature IDs and notes"),
    LayerSpec("HOLE_TABLE", 7, "CONTINUOUS", 25, "the feature (hole) table"),
    LayerSpec("TITLEBLOCK", 7, "CONTINUOUS", 35, "title block block reference and its content"),
    LayerSpec("FRAME", 7, "CONTINUOUS", 50, "sheet border and drawing frame"),
)

LAYER_BY_NAME: Dict[str, LayerSpec] = {layer.name: layer for layer in LAYERS}
#: The documented layer set a conforming PhiMech DXF defines; "0" is required by the format.
REQUIRED_LAYERS: Tuple[str, ...] = tuple(layer.name for layer in LAYERS)

TITLEBLOCK_BLOCK = "TITLEBLOCK"

#: Symbol tables written, in the order AutoCAD expects them.
_TABLE_ORDER: Tuple[str, ...] = ("VPORT", "LTYPE", "LAYER", "STYLE", "VIEW", "UCS", "APPID",
                                 "DIMSTYLE", "BLOCK_RECORD")


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------


def _n(value: float) -> str:
    """A DXF real: six decimals, trailing zeros stripped, never ``-0``."""
    number = float(value)
    if number == 0.0:
        return "0.0"
    text = "{:.6f}".format(number).rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0.0"


def _clean(text: str) -> str:
    """Annotation text with control characters removed.

    A DXF value is one line, so a newline inside a title or a source path would shift every
    following group code by one and corrupt the file. Labels come from user-supplied names, so
    this is a real path, not a theoretical one."""
    return "".join(" " if character in "\r\n\t" else character
                   for character in text if ord(character) >= 32 or character in "\r\n\t")


def _mm(value: float) -> str:
    """A length as annotation shows it: $DIMDEC 2 decimals, $DIMZIN 8 (no trailing zeros)."""
    text = "{:.2f}".format(value).rstrip("0").rstrip(".")
    return text or "0"


def _handle(value: int) -> str:
    return "{:X}".format(value)


def _norm(angle: float) -> float:
    """An angle in degrees mapped into [0, 360)."""
    return angle % 360.0


def _wrap(angle: float) -> float:
    """An angle in radians mapped into (-pi, pi]."""
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def choose_text_height(extent: float) -> float:
    """The largest ISO 3098 height at or below ``extent * TEXT_FRACTION``.

    Annotation must scale with the *part*, because the geometry is never scaled: 3.5 mm text is
    right on a 150 mm bracket and invisible on a 2 m frame."""
    target = max(extent, 0.0) * TEXT_FRACTION
    chosen = TEXT_HEIGHTS[0]
    for height in TEXT_HEIGHTS:
        if height <= target:
            chosen = height
    return chosen


def choose_sheet(width: float, height: float) -> Optional[Tuple[str, float, float]]:
    """Smallest ISO sheet whose printable area holds ``width`` x ``height`` at 1:1, or None.

    None means "larger than A0 at true size" — the frame then follows the content and the
    result carries a warning, because silently shrinking the geometry is the one thing this
    module must never do."""
    for name, sheet_w, sheet_h in SHEETS:
        if width <= sheet_w - 2 * SHEET_MARGIN_MM and height <= sheet_h - 2 * SHEET_MARGIN_MM:
            return name, sheet_w, sheet_h
    return None


# --------------------------------------------------------------------------------------
# polyline -> lines and arcs
# --------------------------------------------------------------------------------------


@dataclass
class Segment:
    """One piece of a decomposed loop: a straight ``line`` or a circular ``arc``.

    Angles are degrees; an arc always runs counter-clockwise from ``start_angle`` to
    ``end_angle`` about ``center``, which is exactly what a DXF ``ARC`` entity means."""

    kind: str
    p0: Point
    p1: Point
    center: Optional[Point] = None
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0
    points: int = 2

    @property
    def sweep(self) -> float:
        return _norm(self.end_angle - self.start_angle)


def _fit_circle(points: Sequence[Point]) -> Optional[Tuple[Point, float]]:
    """Kasa algebraic least-squares circle through ``points`` (needs 3 non-collinear).

    Solves ``x^2 + y^2 = a x + b y + c`` by normal equations — three unknowns, so a hand-rolled
    3x3 elimination keeps this stdlib-only (the core boundary)."""
    count = len(points)
    if count < 3:
        return None
    sx = sy = sxx = syy = sxy = sz = szx = szy = 0.0
    for x, y in points:
        z = x * x + y * y
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
        sz += z
        szx += z * x
        szy += z * y
    matrix = [[sxx, sxy, sx, szx], [sxy, syy, sy, szy], [sx, sy, float(count), sz]]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        factor = matrix[col][col]
        matrix[col] = [value / factor for value in matrix[col]]
        for row in range(3):
            if row != col and matrix[row][col]:
                scale = matrix[row][col]
                matrix[row] = [a - scale * b for a, b in zip(matrix[row], matrix[col])]
    a, b, c = matrix[0][3], matrix[1][3], matrix[2][3]
    cx, cy = a / 2.0, b / 2.0
    inner = c + cx * cx + cy * cy
    if inner <= 0.0:
        return None
    return (cx, cy), math.sqrt(inner)


def _deviation(points: Sequence[Point], center: Point, radius: float) -> float:
    return max(abs(math.hypot(x - center[0], y - center[1]) - radius) for x, y in points)


def _arc_deviation(run: Sequence[Point], center: Point, radius: float) -> float:
    """Worst distance from the fitted circle, sampling chord *midpoints* as well as vertices.

    Vertices alone are not enough, and this was a real bug: the four corners of any rectangle
    lie exactly on one circle, so a vertex-only check happily turned a rectangle into an arc.
    A chord midpoint is where a polyline is furthest from the curve it claims to be, so
    including it is what distinguishes "tessellated arc" from "polygon that happens to be
    cyclic"."""
    worst = _deviation(run, center, radius)
    for a, b in zip(run, run[1:]):
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        worst = max(worst, abs(math.hypot(mid[0] - center[0], mid[1] - center[1]) - radius))
    return worst


def _line_deviation(points: Sequence[Point], p0: Point, p1: Point) -> float:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return max((math.hypot(x - p0[0], y - p0[1]) for x, y in points), default=0.0)
    return max(abs((x - p0[0]) * dy - (y - p0[1]) * dx) / length for x, y in points)


def decompose_loop(loop: Sequence[Point], tol: float = CURVE_TOL_MM) -> List[Segment]:
    """Split a closed tessellated loop into straight runs and circular arcs.

    Greedy longest-run, verified: at each vertex it takes the longest straight run whose every
    intermediate point is within ``tol`` of the chord, then tries to beat it with a longer
    least-squares arc whose worst point is also within ``tol``. Whichever covers more segments
    wins, ties go to the line. An arc that is not really there is a manufacturing error rather
    than a simplification, so the deviation check — not the turn angle — is what decides.

    Turn angle alone is *not* enough, and that was a real bug: a rounded rectangle's straight
    sides meet its corner arcs tangentially, so the joint there turns by half a tessellation
    step and no joint in the whole loop is exactly flat. Fitting is the only honest test.

    The walk starts at the longest chord, which in tessellated output is always a straight edge
    and therefore never the middle of an arc that would then be split across the loop seam."""
    pts = [p for i, p in enumerate(loop) if i == 0 or _dist(p, loop[i - 1]) > 1e-9]
    if len(pts) > 1 and _dist(pts[0], pts[-1]) <= 1e-9:
        pts = pts[:-1]
    count = len(pts)
    if count < 3:
        return [Segment("line", pts[i], pts[i + 1]) for i in range(count - 1)]

    start = max(range(count), key=lambda i: _dist(pts[i], pts[(i + 1) % count]))
    segments: List[Segment] = []
    used = 0
    index = start
    while used < count:
        remaining = count - used
        line_length = _longest_line(pts, index, remaining, tol)
        arc = _longest_arc(pts, index, remaining, tol, line_length)
        if arc is not None:
            length, segment = arc
        else:
            length = line_length
            segment = Segment("line", pts[index % count], pts[(index + length) % count],
                              points=length + 1)
        segments.append(segment)
        used += length
        index = (index + length) % count
    return segments


def _longest_line(pts: Sequence[Point], index: int, remaining: int, tol: float) -> int:
    """How many consecutive segments from ``index`` stay within ``tol`` of one straight chord."""
    count = len(pts)
    best = 1
    for length in range(2, min(remaining, MAX_RUN_SEGMENTS) + 1):
        inner = [pts[(index + k) % count] for k in range(1, length)]
        if _line_deviation(inner, pts[index % count], pts[(index + length) % count]) > tol:
            break
        best = length
    return best


def _longest_arc(pts: Sequence[Point], index: int, remaining: int, tol: float,
                 beat: int) -> Optional[Tuple[int, Segment]]:
    """The longest arc from ``index`` that fits within ``tol`` and covers more than ``beat``
    segments, or None. Shorter than the straight run is not an improvement."""
    count = len(pts)
    found: Optional[Tuple[int, Segment]] = None
    first = max(MIN_ARC_SEGMENTS, beat + 1)
    for length in range(first, min(remaining, MAX_RUN_SEGMENTS) + 1):
        run = [pts[(index + k) % count] for k in range(length + 1)]
        fitted = _fit_circle(run)
        if fitted is None:
            break
        center, radius = fitted
        if radius <= 0.0 or _arc_deviation(run, center, radius) > tol:
            break
        segment = _arc_segment(run, center, radius)
        if segment is not None:
            found = (length, segment)
    return found


def _arc_segment(run: Sequence[Point], center: Point, radius: float) -> Optional[Segment]:
    """``run`` as one counter-clockwise DXF ``ARC``, or None if the sweep is too small to mean
    anything (a nearly straight run fits a huge circle: that is a line, not an arc)."""
    a0 = math.degrees(math.atan2(run[0][1] - center[1], run[0][0] - center[0]))
    a1 = math.degrees(math.atan2(run[-1][1] - center[1], run[-1][0] - center[0]))
    mid = run[len(run) // 2]
    am = math.degrees(math.atan2(mid[1] - center[1], mid[0] - center[0]))
    points = list(run)
    # DXF arcs always run counter-clockwise: if the midpoint is not inside [a0, a1] going CCW
    # then the run is clockwise, so the endpoints swap.
    if not _norm(am - a0) < _norm(a1 - a0):
        a0, a1 = a1, a0
        points = list(reversed(points))
    if _norm(a1 - a0) < MIN_ARC_DEG:
        return None
    return Segment("arc", points[0], points[-1], center, radius, _norm(a0), _norm(a1),
                   points=len(points))


def rounded_rect_segments(bbox: Tuple[float, float, float, float], radius: float) -> List[Segment]:
    """A rounded rectangle as 4 lines + 4 arcs — the analytic form of a slot or rounded pocket."""
    x0, y0, x1, y1 = bbox
    r = max(min(radius, (x1 - x0) / 2.0, (y1 - y0) / 2.0), 0.0)
    out: List[Segment] = []
    for p0, p1 in (((x0 + r, y0), (x1 - r, y0)), ((x1, y0 + r), (x1, y1 - r)),
                   ((x1 - r, y1), (x0 + r, y1)), ((x0, y1 - r), (x0, y0 + r))):
        if _dist(p0, p1) > 1e-9:
            out.append(Segment("line", p0, p1))
    if r > 1e-9:
        for center, a0, a1 in (((x1 - r, y0 + r), 270.0, 360.0), ((x1 - r, y1 - r), 0.0, 90.0),
                               ((x0 + r, y1 - r), 90.0, 180.0), ((x0 + r, y0 + r), 180.0, 270.0)):
            p0 = (center[0] + r * math.cos(math.radians(a0)),
                  center[1] + r * math.sin(math.radians(a0)))
            p1 = (center[0] + r * math.cos(math.radians(a1)),
                  center[1] + r * math.sin(math.radians(a1)))
            out.append(Segment("arc", p0, p1, center, r, _norm(a0), _norm(a1)))
    return out


def rounded_rect_deviation(loop: Sequence[Point], bbox: Tuple[float, float, float, float],
                           radius: float) -> float:
    """Worst distance from the loop's points to the rounded rectangle claimed for it.

    This is what stops a bad ``slot`` classification becoming a lie in the DXF: the signed
    distance function of a rounded box is exact, so the check is a threshold, not an opinion."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    r = max(min(radius, hx, hy), 0.0)
    worst = 0.0
    for x, y in loop:
        qx = abs(x - cx) - (hx - r)
        qy = abs(y - cy) - (hy - r)
        outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
        inside = min(max(qx, qy), 0.0)
        worst = max(worst, abs(outside + inside - r))
    return worst


# --------------------------------------------------------------------------------------
# the DXF document
# --------------------------------------------------------------------------------------


class _Space:
    """Somewhere entities can be written: model space, or one block definition.

    Every geometric point passes through :meth:`_grow`, which is how ``$EXTMIN``/``$EXTMAX`` end
    up honest instead of guessed."""

    def __init__(self, doc: "_Document", owner: str, track: bool = True) -> None:
        self.doc = doc
        self.owner = owner
        self.track = track
        self.parts: List[str] = []

    def tag(self, code: int, value: Any) -> None:
        self.parts.append("{}\n{}\n".format(code, value))

    def _head(self, kind: str, layer: str, subclass: str) -> None:
        self.doc.count(kind, model=self.owner == self.doc.msp_record)
        self.doc.use_layer(layer)
        self.tag(0, kind)
        self.tag(5, self.doc.next_handle())
        self.tag(330, self.owner)
        self.tag(100, "AcDbEntity")
        self.tag(8, layer)
        self.tag(100, subclass)

    def _grow(self, x: float, y: float) -> None:
        if self.track:
            self.doc.grow(x, y)

    # -- entities ------------------------------------------------------------------

    def line(self, p0: Point, p1: Point, layer: str) -> None:
        self._head("LINE", layer, "AcDbLine")
        self.tag(10, _n(p0[0]))
        self.tag(20, _n(p0[1]))
        self.tag(30, "0.0")
        self.tag(11, _n(p1[0]))
        self.tag(21, _n(p1[1]))
        self.tag(31, "0.0")
        self._grow(p0[0], p0[1])
        self._grow(p1[0], p1[1])

    def circle(self, center: Point, radius: float, layer: str) -> None:
        self._head("CIRCLE", layer, "AcDbCircle")
        self.tag(10, _n(center[0]))
        self.tag(20, _n(center[1]))
        self.tag(30, "0.0")
        self.tag(40, _n(radius))
        self._grow(center[0] - radius, center[1] - radius)
        self._grow(center[0] + radius, center[1] + radius)

    def arc(self, center: Point, radius: float, start: float, end: float, layer: str) -> None:
        self._head("ARC", layer, "AcDbCircle")
        self.tag(10, _n(center[0]))
        self.tag(20, _n(center[1]))
        self.tag(30, "0.0")
        self.tag(40, _n(radius))
        self.tag(100, "AcDbArc")
        self.tag(50, _n(start))
        self.tag(51, _n(end))
        self._grow(center[0] - radius, center[1] - radius)
        self._grow(center[0] + radius, center[1] + radius)

    def lwpolyline(self, points: Sequence[Point], layer: str, closed: bool = True) -> None:
        self._head("LWPOLYLINE", layer, "AcDbPolyline")
        self.tag(90, len(points))
        self.tag(70, 1 if closed else 0)
        for x, y in points:
            self.tag(10, _n(x))
            self.tag(20, _n(y))
            self._grow(x, y)

    def solid(self, points: Sequence[Point], layer: str) -> None:
        """A filled triangle or quad — how DXF draws an arrowhead."""
        pts = list(points)
        while len(pts) < 4:
            pts.append(pts[-1])
        self._head("SOLID", layer, "AcDbTrace")
        for index, (x, y) in enumerate(pts[:4]):
            self.tag(10 + index, _n(x))
            self.tag(20 + index, _n(y))
            self.tag(30 + index, "0.0")
            self._grow(x, y)

    def text(self, position: Point, content: str, layer: str, height: float, *,
             halign: int = 0, valign: int = 0, rotation: float = 0.0) -> None:
        self._head("TEXT", layer, "AcDbText")
        self.tag(10, _n(position[0]))
        self.tag(20, _n(position[1]))
        self.tag(30, "0.0")
        self.tag(40, _n(height))
        self.tag(1, _clean(content))
        if rotation:
            self.tag(50, _n(rotation))
        self.tag(7, TEXT_STYLE_NAME)
        self.tag(72, halign)
        self.tag(11, _n(position[0]))
        self.tag(21, _n(position[1]))
        self.tag(31, "0.0")
        self.tag(100, "AcDbText")
        self.tag(73, valign)
        self._grow_text(position, content, height, halign, valign, rotation)

    def mtext(self, position: Point, content: str, layer: str, height: float,
              width: float) -> None:
        """Multi-line note text; ``\\P`` is the MTEXT line break."""
        self._head("MTEXT", layer, "AcDbMText")
        self.tag(10, _n(position[0]))
        self.tag(20, _n(position[1]))
        self.tag(30, "0.0")
        self.tag(40, _n(height))
        self.tag(41, _n(width))
        self.tag(71, 1)      # attachment point: top left
        self.tag(72, 1)      # drawing direction: left to right
        chunks = _mtext_chunks(_clean(content))
        for chunk in chunks[:-1]:
            self.tag(3, chunk)
        self.tag(1, chunks[-1])
        self.tag(7, TEXT_STYLE_NAME)
        lines = content.count("\\P") + 1
        self._grow(position[0], position[1])
        self._grow(position[0] + width, position[1] - lines * height * 1.7)

    def insert(self, block: str, position: Point, layer: str) -> None:
        self._head("INSERT", layer, "AcDbBlockReference")
        self.tag(2, block)
        self.tag(10, _n(position[0]))
        self.tag(20, _n(position[1]))
        self.tag(30, "0.0")
        self.tag(41, "1.0")
        self.tag(42, "1.0")
        self.tag(43, "1.0")
        self.tag(50, "0.0")
        self._grow(position[0], position[1])

    def _grow_text(self, position: Point, content: str, height: float, halign: int, valign: int,
                   rotation: float) -> None:
        """Approximate text box, so extents cover annotation rather than only its anchor."""
        width = 0.7 * height * len(content)
        dx = {0: (0.0, width), 1: (-width / 2.0, width / 2.0), 2: (-width, 0.0)}.get(
            halign, (0.0, width))
        dy = {0: (0.0, height), 1: (0.0, height), 2: (-height / 2.0, height / 2.0),
              3: (-height, 0.0)}.get(valign, (0.0, height))
        for ex in dx:
            for ey in dy:
                if abs(rotation - 90.0) < 1e-6:
                    self._grow(position[0] - ey, position[1] + ex)
                else:
                    self._grow(position[0] + ex, position[1] + ey)


def _mtext_chunks(content: str, size: int = 250) -> List[str]:
    return [content[i:i + size] for i in range(0, max(len(content), 1), size)] or [""]


@dataclass
class _Block:
    name: str
    handle: str
    end_handle: str
    record: str
    space: _Space
    anonymous: bool


@dataclass
class DimensionPlan:
    """One dimension the sheet decided to place, and the measured value behind it.

    Kept so the fidelity gate can ask "does the file's dimension say what the geometry said",
    with a number rather than a shrug."""

    kind: str          # "horizontal" | "vertical"
    value: float
    p1: Point
    p2: Point
    block: str


class _Document:
    """Handle allocation, tables, blocks and the model space entity list for one DXF file."""

    def __init__(self) -> None:
        self._next = 0x20
        self.tables: Dict[str, str] = {name: self.next_handle() for name in _TABLE_ORDER}
        self.style_handle = self.next_handle()
        self.msp_record = self.next_handle()
        self.psp_record = self.next_handle()
        self.root_dict = self.next_handle()
        self.group_dict = self.next_handle()
        self.model = _Space(self, self.msp_record)
        self.blocks: List[_Block] = []
        self.dimensions: List[DimensionPlan] = []
        self.counts: Dict[str, int] = {}
        self.block_counts: Dict[str, int] = {}
        self.layers_used: Dict[str, int] = {}
        self.min_x = self.min_y = math.inf
        self.max_x = self.max_y = -math.inf

    def next_handle(self) -> str:
        self._next += 1
        return _handle(self._next)

    def count(self, kind: str, model: bool = True) -> None:
        """Model space and block content are counted apart: a reader queries model space, and
        every arrowhead and dimension text lives inside a ``*D`` block."""
        target = self.counts if model else self.block_counts
        target[kind] = target.get(kind, 0) + 1

    def use_layer(self, name: str) -> None:
        self.layers_used[name] = self.layers_used.get(name, 0) + 1

    def grow(self, x: float, y: float) -> None:
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def new_block(self, name: str, anonymous: bool = False, track: bool = True) -> _Block:
        record = self.next_handle()
        block = _Block(name, self.next_handle(), self.next_handle(), record,
                       _Space(self, record, track), anonymous)
        self.blocks.append(block)
        return block

    def new_dim_block(self) -> _Block:
        return self.new_block("*D{}".format(len(self.dimensions) + 1), anonymous=True)

    @property
    def extents(self) -> Tuple[float, float, float, float]:
        if self.min_x == math.inf:
            return (0.0, 0.0, 0.0, 0.0)
        return (self.min_x, self.min_y, self.max_x, self.max_y)


# --------------------------------------------------------------------------------------
# serialising the document
# --------------------------------------------------------------------------------------


def _header(doc: _Document, ltscale: float, text_height: float, arrow: float) -> List[str]:
    x0, y0, x1, y1 = doc.extents
    out: List[str] = ["0\nSECTION\n2\nHEADER\n"]

    def var(name: str, *pairs: Tuple[int, Any]) -> None:
        out.append("9\n{}\n".format(name))
        for code, value in pairs:
            out.append("{}\n{}\n".format(code, value))

    var("$ACADVER", (1, DXF_VERSION))
    var("$HANDSEED", (5, _handle(doc._next + 0x40)))
    var("$DWGCODEPAGE", (3, "ANSI_1252"))
    var("$INSUNITS", (70, INSUNITS_MM))
    var("$MEASUREMENT", (70, MEASUREMENT_METRIC))
    var("$LUNITS", (70, 2))          # decimal
    var("$LUPREC", (70, 3))
    var("$AUNITS", (70, 0))          # decimal degrees
    var("$AUPREC", (70, 2))
    var("$INSBASE", (10, "0.0"), (20, "0.0"), (30, "0.0"))
    var("$EXTMIN", (10, _n(x0)), (20, _n(y0)), (30, "0.0"))
    var("$EXTMAX", (10, _n(x1)), (20, _n(y1)), (30, "0.0"))
    var("$LIMMIN", (10, _n(x0)), (20, _n(y0)))
    var("$LIMMAX", (10, _n(x1)), (20, _n(y1)))
    var("$LTSCALE", (40, _n(ltscale)))
    var("$CELTYPE", (6, "ByLayer"))
    var("$CECOLOR", (62, 256))
    var("$CLAYER", (8, "0"))
    var("$TEXTSTYLE", (7, TEXT_STYLE_NAME))
    var("$TEXTSIZE", (40, _n(text_height)))
    var("$DIMSTYLE", (2, DIMSTYLE_NAME))
    var("$DIMTXT", (40, _n(text_height)))
    var("$DIMASZ", (40, _n(arrow)))
    var("$DIMSCALE", (40, "1.0"))
    var("$DIMLFAC", (40, "1.0"))     # geometry is 1:1: a dimension never multiplies its value
    var("$DIMLUNIT", (70, DIMLUNIT_DECIMAL))
    var("$DIMDEC", (70, DIMDEC))
    var("$DIMZIN", (70, DIMZIN_SUPPRESS_TRAILING))
    var("$DIMDSEP", (70, DIMDSEP_POINT))
    var("$PDMODE", (70, 0))
    var("$PDSIZE", (40, "0.0"))
    out.append("0\nENDSEC\n")
    return out


def _tables(doc: _Document, text_height: float, arrow: float,
            view_center: Point, view_height: float) -> List[str]:
    out: List[str] = ["0\nSECTION\n2\nTABLES\n"]

    def open_table(name: str, count: int, extra: Sequence[Tuple[int, Any]] = ()) -> None:
        out.append("0\nTABLE\n2\n{}\n5\n{}\n330\n0\n100\nAcDbSymbolTable\n70\n{}\n".format(
            name, doc.tables[name], count))
        for code, value in extra:
            out.append("{}\n{}\n".format(code, value))

    def record(kind: str, handle: str, subclass: str, pairs: Sequence[Tuple[int, Any]]) -> None:
        out.append("0\n{}\n5\n{}\n330\n{}\n100\nAcDbSymbolTableRecord\n100\n{}\n".format(
            kind, handle, doc.tables[kind], subclass))
        for code, value in pairs:
            out.append("{}\n{}\n".format(code, value))

    # VPORT — only so a viewer opens zoomed onto the drawing rather than at the origin.
    open_table("VPORT", 1)
    record("VPORT", doc.next_handle(), "AcDbViewportTableRecord", [
        (2, "*Active"), (70, 0), (10, "0.0"), (20, "0.0"), (11, "1.0"), (21, "1.0"),
        (12, _n(view_center[0])), (22, _n(view_center[1])), (13, "0.0"), (23, "0.0"),
        (14, "1.0"), (24, "1.0"), (15, "10.0"), (25, "10.0"),
        (16, "0.0"), (26, "0.0"), (36, "1.0"), (17, "0.0"), (27, "0.0"), (37, "0.0"),
        (40, _n(view_height)), (41, "1.5"), (42, "50.0"), (43, "0.0"), (44, "0.0"),
        (50, "0.0"), (51, "0.0"), (71, 0), (72, 100), (73, 1), (74, 3), (75, 0), (76, 0),
        (77, 0), (78, 0)])
    out.append("0\nENDTAB\n")

    open_table("LTYPE", len(LINETYPES))
    for linetype in LINETYPES:
        pairs: List[Tuple[int, Any]] = [(2, linetype.name), (70, 0), (3, linetype.description),
                                        (72, 65), (73, len(linetype.pattern)),
                                        (40, _n(linetype.total))]
        for dash in linetype.pattern:
            pairs.append((49, _n(dash)))
            pairs.append((74, 0))
        record("LTYPE", doc.next_handle(), "AcDbLinetypeTableRecord", pairs)
    out.append("0\nENDTAB\n")

    open_table("LAYER", len(LAYERS))
    for layer in LAYERS:
        record("LAYER", doc.next_handle(), "AcDbLayerTableRecord", [
            (2, layer.name), (70, 0), (62, layer.color), (6, layer.linetype),
            (370, layer.lineweight)])
    out.append("0\nENDTAB\n")

    open_table("STYLE", 1)
    out.append("0\nSTYLE\n5\n{}\n330\n{}\n100\nAcDbSymbolTableRecord\n"
               "100\nAcDbTextStyleTableRecord\n".format(doc.style_handle, doc.tables["STYLE"]))
    for code, value in [(2, TEXT_STYLE_NAME), (70, 0), (40, "0.0"), (41, "1.0"), (50, "0.0"),
                        (71, 0), (42, _n(text_height)), (3, TEXT_STYLE_FONT), (4, "")]:
        out.append("{}\n{}\n".format(code, value))
    out.append("0\nENDTAB\n")

    open_table("VIEW", 0)
    out.append("0\nENDTAB\n")
    open_table("UCS", 0)
    out.append("0\nENDTAB\n")

    open_table("APPID", 1)
    record("APPID", doc.next_handle(), "AcDbRegAppTableRecord", [(2, "ACAD"), (70, 0)])
    out.append("0\nENDTAB\n")

    # ISO-25: the ISO metric dimension style. Its values must agree with the geometry rendered
    # into the *D blocks, so a reader that regenerates a dimension draws what the block shows.
    dimstyle_handle = doc.next_handle()
    open_table("DIMSTYLE", 1, [(100, "AcDbDimStyleTable"), (71, 1), (340, dimstyle_handle)])
    out.append("0\nDIMSTYLE\n105\n{}\n330\n{}\n100\nAcDbSymbolTableRecord\n"
               "100\nAcDbDimStyleTableRecord\n".format(dimstyle_handle, doc.tables["DIMSTYLE"]))
    for code, value in [
        (2, DIMSTYLE_NAME), (70, 0),
        (40, "1.0"),                       # DIMSCALE: the sizes below are already absolute mm
        (41, _n(arrow)),                   # DIMASZ
        (42, _n(0.4 * text_height)),       # DIMEXO
        (43, _n(2.6 * text_height)),       # DIMDLI
        (44, _n(0.6 * text_height)),       # DIMEXE
        (140, _n(text_height)),            # DIMTXT
        (141, _n(0.9 * text_height)),      # DIMCEN
        (144, "1.0"),                      # DIMLFAC: 1:1, never a scale factor on the value
        (147, _n(0.35 * text_height)),     # DIMGAP
        (73, 0), (74, 0), (77, 1),         # text with the dimension line, placed above it
        (78, DIMZIN_SUPPRESS_TRAILING),    # DIMZIN
        (271, DIMDEC), (272, DIMDEC),      # DIMDEC / DIMTDEC
        (277, DIMLUNIT_DECIMAL),           # DIMLUNIT
        (278, DIMDSEP_POINT),              # DIMDSEP
        (280, 0),                          # DIMJUST: centred
        (340, doc.style_handle),           # DIMTXSTY
    ]:
        out.append("{}\n{}\n".format(code, value))
    out.append("0\nENDTAB\n")

    open_table("BLOCK_RECORD", 2 + len(doc.blocks))
    for name, handle in (("*Model_Space", doc.msp_record), ("*Paper_Space", doc.psp_record)):
        record("BLOCK_RECORD", handle, "AcDbBlockTableRecord",
               [(2, name), (70, 0), (280, 1), (281, 0)])
    for block in doc.blocks:
        record("BLOCK_RECORD", block.record, "AcDbBlockTableRecord",
               [(2, block.name), (70, 0), (280, 1), (281, 0)])
    out.append("0\nENDTAB\n0\nENDSEC\n")
    return out


def _block_definition(name: str, handle: str, end_handle: str, record: str, anonymous: bool,
                      body: Sequence[str]) -> List[str]:
    out = ["0\nBLOCK\n5\n{}\n330\n{}\n100\nAcDbEntity\n8\n0\n100\nAcDbBlockBegin\n"
           "2\n{}\n70\n{}\n10\n0.0\n20\n0.0\n30\n0.0\n3\n{}\n1\n\n".format(
               handle, record, name, 1 if anonymous else 0, name)]
    out.extend(body)
    out.append("0\nENDBLK\n5\n{}\n330\n{}\n100\nAcDbEntity\n8\n0\n100\nAcDbBlockEnd\n".format(
        end_handle, record))
    return out


def _blocks(doc: _Document) -> List[str]:
    out = ["0\nSECTION\n2\nBLOCKS\n"]
    out.extend(_block_definition("*Model_Space", doc.next_handle(), doc.next_handle(),
                                 doc.msp_record, False, []))
    out.extend(_block_definition("*Paper_Space", doc.next_handle(), doc.next_handle(),
                                 doc.psp_record, False, []))
    for block in doc.blocks:
        out.extend(_block_definition(block.name, block.handle, block.end_handle, block.record,
                                     block.anonymous, block.space.parts))
    out.append("0\nENDSEC\n")
    return out


def _objects(doc: _Document) -> List[str]:
    """The minimum OBJECTS section an R2000 reader expects: a named-object root dictionary."""
    return ["0\nSECTION\n2\nOBJECTS\n",
            "0\nDICTIONARY\n5\n{}\n330\n0\n100\nAcDbDictionary\n281\n1\n3\nACAD_GROUP\n350\n{}\n"
            .format(doc.root_dict, doc.group_dict),
            "0\nDICTIONARY\n5\n{}\n330\n{}\n100\nAcDbDictionary\n281\n1\n".format(
                doc.group_dict, doc.root_dict),
            "0\nENDSEC\n"]


def serialise(doc: _Document, text_height: float, arrow: float, ltscale: float) -> str:
    """The whole file as text, in the section order every DXF reader expects.

    The body is built before the header even though the header is written first: ``$HANDSEED``
    must be larger than every handle in the file, and the tables and block definitions are
    where the last handles get allocated."""
    x0, y0, x1, y1 = doc.extents
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    height = max(y1 - y0, 1.0) * 1.1
    body: List[str] = ["0\nSECTION\n2\nCLASSES\n0\nENDSEC\n"]
    body.extend(_tables(doc, text_height, arrow, center, height))
    body.extend(_blocks(doc))
    body.append("0\nSECTION\n2\nENTITIES\n")
    body.extend(doc.model.parts)
    body.append("0\nENDSEC\n")
    body.extend(_objects(doc))
    body.append("0\nEOF\n")
    return "".join(_header(doc, ltscale, text_height, arrow) + body)


# --------------------------------------------------------------------------------------
# dimensions
# --------------------------------------------------------------------------------------


@dataclass
class _Style:
    """Annotation sizes for one drawing, all derived from the part's own extent."""

    text: float
    arrow: float
    exo: float          # extension line offset from the measured point
    exe: float          # extension line overshoot past the dimension line
    gap: float          # text gap above the dimension line

    @property
    def step(self) -> float:
        """Spacing between stacked dimension lines (DIMDLI)."""
        return 2.6 * self.text


def _style_for(text_height: float) -> _Style:
    return _Style(text=text_height, arrow=1.1 * text_height, exo=0.4 * text_height,
                  exe=0.6 * text_height, gap=0.35 * text_height)


def _arrowhead(space: _Space, tip: Point, angle: float, size: float, layer: str) -> None:
    """A filled arrowhead with its point at ``tip``, pointing along ``angle`` (radians)."""
    back = (tip[0] - size * math.cos(angle), tip[1] - size * math.sin(angle))
    nx, ny = -math.sin(angle) * size / 6.0, math.cos(angle) * size / 6.0
    space.solid([tip, (back[0] + nx, back[1] + ny), (back[0] - nx, back[1] - ny)], layer)


def linear_dimension(doc: _Document, p1: Point, p2: Point, position: float, *,
                     horizontal: bool, style: _Style) -> DimensionPlan:
    """A rotated (horizontal or vertical) ``DIMENSION`` plus its rendered ``*D`` block.

    ``position`` is the paper y (horizontal) or x (vertical) the dimension line runs along. The
    entity carries **no text override**: the value a reader shows is the one it measures between
    the definition points, so the drawing cannot disagree with its own geometry."""
    layer = "DIMENSIONS"
    block = doc.new_dim_block()
    space = block.space
    if horizontal:
        a, b = (p1[0], position), (p2[0], position)
        measurement = abs(p2[0] - p1[0])
        rotation = 0.0
    else:
        a, b = (position, p1[1]), (position, p2[1])
        measurement = abs(p2[1] - p1[1])
        rotation = 90.0
    label = _mm(measurement)

    for origin, end in zip((p1, p2), (a, b)):
        if horizontal:
            sign = 1.0 if position > origin[1] else -1.0
            space.line((origin[0], origin[1] + sign * style.exo),
                       (origin[0], position + sign * style.exe), layer)
        else:
            sign = 1.0 if position > origin[0] else -1.0
            space.line((origin[0] + sign * style.exo, origin[1]),
                       (position + sign * style.exe, origin[1]), layer)
    space.line(a, b, layer)

    axis = 0.0 if horizontal else math.pi / 2.0
    low, high = (a, b) if (a[0] + a[1]) <= (b[0] + b[1]) else (b, a)
    if _dist(a, b) >= 3.2 * style.arrow:
        _arrowhead(space, low, axis + math.pi, style.arrow, layer)
        _arrowhead(space, high, axis, style.arrow, layer)
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        if horizontal:
            text_pos = (mid[0], mid[1] + style.gap)
            space.text(text_pos, label, layer, style.text, halign=1, valign=0)
        else:
            text_pos = (mid[0] - style.gap, mid[1])
            space.text(text_pos, label, layer, style.text, halign=1, valign=3, rotation=90.0)
    else:
        # too short to hold arrows and text between the extension lines: both go outside
        _arrowhead(space, low, axis, style.arrow, layer)
        _arrowhead(space, high, axis + math.pi, style.arrow, layer)
        if horizontal:
            text_pos = (high[0] + 1.2 * style.arrow, high[1] + style.gap)
            space.text(text_pos, label, layer, style.text, halign=0, valign=0)
        else:
            text_pos = (high[0] - style.gap, high[1] + 1.2 * style.arrow)
            space.text(text_pos, label, layer, style.text, halign=0, valign=3, rotation=90.0)

    model = doc.model
    doc.count("DIMENSION")
    doc.use_layer(layer)
    model.tag(0, "DIMENSION")
    model.tag(5, doc.next_handle())
    model.tag(330, model.owner)
    model.tag(100, "AcDbEntity")
    model.tag(8, layer)
    model.tag(100, "AcDbDimension")
    model.tag(2, block.name)
    model.tag(10, _n(b[0]))
    model.tag(20, _n(b[1]))
    model.tag(30, "0.0")
    model.tag(11, _n(text_pos[0]))
    model.tag(21, _n(text_pos[1]))
    model.tag(31, "0.0")
    model.tag(70, 32)            # rotated dimension + "block is used only by this dimension"
    model.tag(71, 5)             # attachment: middle centre
    model.tag(42, _n(measurement))
    model.tag(3, DIMSTYLE_NAME)
    model.tag(100, "AcDbAlignedDimension")
    model.tag(13, _n(p1[0]))
    model.tag(23, _n(p1[1]))
    model.tag(33, "0.0")
    model.tag(14, _n(p2[0]))
    model.tag(24, _n(p2[1]))
    model.tag(34, "0.0")
    model.tag(50, _n(rotation))
    model.tag(100, "AcDbRotatedDimension")

    plan = DimensionPlan("horizontal" if horizontal else "vertical", measurement, p1, p2,
                         block.name)
    doc.dimensions.append(plan)
    return plan


# --------------------------------------------------------------------------------------
# emitting the views
# --------------------------------------------------------------------------------------


def _model_bounds(views: Mapping[str, "ViewData"]) -> Tuple[List[float], List[float]]:
    """Per-model-axis min/max, taken from whichever views see that axis.

    One shared set of bounds is what keeps the three views *aligned*: the top view's x and the
    front view's x are the same model axis and must land on the same paper column."""
    from phimech.synth.drawing import VIEWS

    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    for name, view in views.items():
        spec = VIEWS[name]
        for axis, low, high in ((spec["ax"], view.bbox[0], view.bbox[2]),
                                (spec["ay"], view.bbox[1], view.bbox[3])):
            mins[axis] = min(mins[axis], low)
            maxs[axis] = max(maxs[axis], high)
    for axis in range(3):
        if mins[axis] == math.inf:
            mins[axis], maxs[axis] = 0.0, 0.0
    return mins, maxs


def view_origins(views: Mapping[str, "ViewData"], gap: float) -> Dict[str, Point]:
    """Translation applied to each view, third-angle, with the **top view at model coordinates**.

    A DXF imported into SolidWorks lands at the coordinates in the file, so keeping the plan at
    true model x/y means a hole picked in the sketch reads the same numbers as the STEP and the
    IR. The other two views are offset by known amounts, reported in the result."""
    mins, maxs = _model_bounds(views)
    height = maxs[2] - mins[2]
    below = mins[1] - gap - height - mins[2]
    return {
        "top": (0.0, 0.0),
        "front": (0.0, below),
        "right": (maxs[0] + gap - mins[1], below),
    }


def _shape_segments(shape: "Shape", curve_tol: float,
                    warnings: List[str]) -> Tuple[List[Segment], Optional[Tuple[Point, float]]]:
    """Analytic geometry for one classified loop, verified against its own points.

    Returns ``(segments, circle)`` where exactly one is populated: a fitted circle becomes a real
    ``CIRCLE`` entity, everything else becomes lines and arcs."""
    loop = list(shape.loop)
    if shape.kind == "circle":
        radius = shape.size[0] / 2.0
        deviation = _deviation(loop, shape.center, radius)
        if deviation <= max(curve_tol, 0.004 * shape.size[0]):
            return [], (shape.center, radius)
        warnings.append("circle at ({}, {}) deviates {} mm from its fit: written as a polyline"
                        .format(_mm(shape.center[0]), _mm(shape.center[1]), _mm(deviation)))
        return _closed_lines(loop), None
    if shape.kind == "rect":
        x0, y0, x1, y1 = shape.bbox
        return _closed_lines([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), None
    if shape.kind == "slot":
        deviation = rounded_rect_deviation(loop, shape.bbox, shape.corner_r)
        if deviation <= max(curve_tol, 0.01 * min(shape.size)):
            return rounded_rect_segments(shape.bbox, shape.corner_r), None
        warnings.append("rounded rect at ({}, {}) deviates {} mm from its fit: decomposed"
                        .format(_mm(shape.center[0]), _mm(shape.center[1]), _mm(deviation)))
    if len(loop) > MAX_DECOMPOSE_POINTS:
        warnings.append("profile at ({}, {}) has {} points: written as one polyline".format(
            _mm(shape.center[0]), _mm(shape.center[1]), len(loop)))
        return _closed_lines(loop), None
    return decompose_loop(loop, curve_tol), None


def _closed_lines(loop: Sequence[Point]) -> List[Segment]:
    return [Segment("line", loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]


def _emit_shape(space: _Space, shape: "Shape", layer: str, offset: Point, curve_tol: float,
                warnings: List[str]) -> None:
    """Write one classified loop as true entities, translated by ``offset``."""
    dx, dy = offset
    segments, circle = _shape_segments(shape, curve_tol, warnings)
    if circle is not None:
        center, radius = circle
        space.circle((center[0] + dx, center[1] + dy), radius, layer)
        return
    if segments and all(segment.kind == "line" for segment in segments):
        points = [(segment.p0[0] + dx, segment.p0[1] + dy) for segment in segments]
        if len(points) >= 2:
            space.lwpolyline(points, layer, closed=True)
            return
    for segment in segments:
        if segment.kind == "arc" and segment.center is not None:
            space.arc((segment.center[0] + dx, segment.center[1] + dy), segment.radius,
                      segment.start_angle, segment.end_angle, layer)
        else:
            space.line((segment.p0[0] + dx, segment.p0[1] + dy),
                       (segment.p1[0] + dx, segment.p1[1] + dy), layer)


def _dedupe(shapes: Sequence["Shape"], against: Sequence["Shape"] = ()) -> List["Shape"]:
    """Drop repeats of a loop already written: slices repeat the same wall at every level."""
    kept: List["Shape"] = []
    for shape in shapes:
        if any(shape.matches(other) for other in against) or \
                any(shape.matches(other) for other in kept):
            continue
        kept.append(shape)
    return kept


def _view_shapes(view: "ViewData") -> Tuple[List["Shape"], List["Shape"]]:
    """``(silhouette shapes, section shapes)`` for one view, deduplicated against each other."""
    from phimech.synth.drawing import classify_slice

    silhouette = classify_slice(list(view.silhouette))
    sections: List["Shape"] = []
    for _, _, _, loops in view.slices:
        sections.extend(classify_slice(list(loops)))
    return silhouette, _dedupe(sections, against=silhouette)


def planned_circles(views: Mapping[str, "ViewData"], origins: Mapping[str, Point],
                    curve_tol: float = CURVE_TOL_MM) -> List[Tuple[str, float, float, float]]:
    """Every ``CIRCLE`` this module will write, as ``(layer, x, y, radius)`` in DXF coordinates.

    Both the writer and :func:`check_fidelity` derive from this, so the gate compares what the
    *file* says against what the *classified geometry* says."""
    out: List[Tuple[str, float, float, float]] = []
    warnings: List[str] = []
    for name, view in views.items():
        dx, dy = origins.get(name, (0.0, 0.0))
        silhouette, sections = _view_shapes(view)
        for layer, shapes in (("VISIBLE", silhouette), ("SECTION", sections)):
            for shape in shapes:
                _, circle = _shape_segments(shape, curve_tol, warnings)
                if circle is not None:
                    out.append((layer, circle[0][0] + dx, circle[0][1] + dy, circle[1]))
    return out


def _emit_view(doc: _Document, view: "ViewData", others: Sequence["ViewData"], offset: Point,
               curve_tol: float, warnings: List[str], style: _Style) -> int:
    """Geometry of one view: outline, feature edges at other levels, projected hidden extents.

    Returns the number of hidden-edge boxes dropped for falling outside the view's silhouette."""
    from phimech.synth.drawing import box_in_silhouette, feature_extent_boxes

    dx, dy = offset
    silhouette, sections = _view_shapes(view)
    for shape in sections:
        _emit_shape(doc.model, shape, "SECTION", offset, curve_tol, warnings)
    for shape in silhouette:
        _emit_shape(doc.model, shape, "VISIBLE", offset, curve_tol, warnings)
    seen: List[Tuple[float, ...]] = []
    dropped = 0
    for other in others:
        for feature in other.features:
            layer = "HIDDEN" if feature.shape.is_hole else "SECTION"
            # One box per run of material the wall bounds, never one prism through the gap
            # between two slabs, and never a box reaching outside this view's own outline: a
            # hidden edge in empty air reads to a machinist as "there is material here".
            for box in feature_extent_boxes(feature, view.name):
                if not box_in_silhouette(box, view.silhouette):
                    dropped += 1
                    continue
                key = tuple(round(value, 4) for value in box)
                if key in seen:
                    continue
                seen.append(key)
                x0, y0, x1, y1 = box
                doc.model.lwpolyline([(x0 + dx, y0 + dy), (x1 + dx, y0 + dy),
                                      (x1 + dx, y1 + dy), (x0 + dx, y1 + dy)], layer,
                                     closed=True)
    for shape in silhouette + sections:
        if shape.kind != "circle":
            continue
        radius = shape.size[0] / 2.0
        arm = max(radius * 1.35, radius + 0.6 * style.text)
        cx, cy = shape.center[0] + dx, shape.center[1] + dy
        doc.model.line((cx - arm, cy), (cx + arm, cy), "CENTER")
        doc.model.line((cx, cy - arm), (cx, cy + arm), "CENTER")
    return dropped


# --------------------------------------------------------------------------------------
# annotation: callouts, table, title block
# --------------------------------------------------------------------------------------


def group_size_text(group: "Group") -> str:
    """The size of a group as annotation shows it: the group's own representative value.

    ``Group.size`` is the mean of its members' measured sizes, and it is the only size printed
    for the group anywhere. Printing ``members[0]``'s size instead would make a 7.28/7.32 pair
    read as whichever member happened to be found first — the order-dependent bug ``drawing.py``
    removed, which must not be reintroduced here. The text itself comes from ``_shape_label``
    so the DXF callout and the SVG row can never word the same feature differently; only the
    diameter sign changes, because DXF annotation is ASCII (``%%c``) — see :func:`group_callout`.
    """
    from phimech.synth.drawing import _shape_label

    return _shape_label(group.kind, group.size, group.corner_r).replace("\u2300", "%%c")


def group_counterbore(group: "Group") -> Optional[Dict[str, Any]]:
    """The counterbore every member of the group shares, or None.

    ``drawing.py`` recovers a counterbore as a *measured* coaxial wall, so it is a fact about
    the part and belongs on the callout. It is only printed when every member agrees within the
    grouping quantum: the group key does not include the counterbore, so one member having a
    wider mouth than the others would otherwise be printed for all of them."""
    from phimech.synth.drawing import SIZE_QUANTUM

    records = [getattr(member, "counterbore", None) for member in group.members]
    if not records or any(record is None for record in records):
        return None
    first = records[0]
    for record in records[1:]:
        if (abs(float(record["depth"]) - float(first["depth"])) > SIZE_QUANTUM
                or abs(max(record["size"]) - max(first["size"])) > SIZE_QUANTUM):
            return None
    return first


def counterbore_text(record: Mapping[str, Any]) -> str:
    """``C'BORE %%c9.5 DEPTH 4`` — the measured mouth of a hole, in ASCII."""
    if record.get("diameter") is not None:
        size = "%%c{}".format(_mm(float(record["diameter"])))
    else:
        size = "{} x {}".format(_mm(record["size"][0]), _mm(record["size"][1]))
    return "C'BORE {} DEPTH {}".format(size, _mm(float(record["depth"])))


def group_callout(group: "Group") -> str:
    """The manufacturing callout for a feature group, in ASCII.

    ``%%c`` is the DXF diameter symbol: Unicode in DXF depends on ``$DWGCODEPAGE`` and on MTEXT
    escapes, while ``%%c`` and ``%%d`` are understood by every reader including SolidWorks."""
    quantity = "{}x ".format(len(group.members)) if len(group.members) > 1 else ""
    size = group_size_text(group)
    if not group.is_hole:
        size = ("BOSS " if group.kind == "circle" else "STEP ") + size
    depth = "THRU" if group.thru else "DEPTH {}".format(_mm(group.depth))
    text = "{}  {}{} {}".format(group.group_id, quantity, size, depth)
    counterbore = group_counterbore(group)
    if counterbore is not None:
        text += ", " + counterbore_text(counterbore)
    # An unmeasured group's extent is the span between slice levels, not its own wall: say so
    # rather than let it read like a caliper reading.
    return text if group.measured else text + " (EST)"


def _emit_callouts(doc: _Document, groups: Sequence["Group"], origins: Mapping[str, Point],
                   column_x: float, y_top: float, y_min: float, style: _Style,
                   max_callouts: int) -> int:
    """Leader callouts stacked in a column right of the top view, as on the SVG sheet.

    A callout is a *note*, not a dimension: it carries quantity, depth and the group ID, none of
    which is the measurement of one distance. Sizes therefore appear here and in the hole table,
    and the DIMENSION entities stay free of text overrides."""
    written = 0
    y = y_top
    for group in groups:
        if written >= max_callouts or y < y_min:
            break
        member = group.members[0]
        dx, dy = origins.get(member.view, (0.0, 0.0))
        shape = member.shape
        if shape.kind == "circle":
            radius = shape.size[0] / 2.0
            angle = math.atan2(y - (shape.center[1] + dy), column_x - (shape.center[0] + dx))
            start = (shape.center[0] + dx + radius * math.cos(angle),
                     shape.center[1] + dy + radius * math.sin(angle))
        else:
            anchor_point = max(shape.loop, key=lambda p: p[0])
            start = (anchor_point[0] + dx, anchor_point[1] + dy)
        knee = (column_x - 2.0 * style.text, y)
        doc.model.line(start, knee, "ANNOTATION")
        doc.model.line(knee, (column_x, y), "ANNOTATION")
        _arrowhead(doc.model, start, math.atan2(start[1] - knee[1], start[0] - knee[0]),
                   style.arrow, "ANNOTATION")
        doc.model.text((column_x + style.gap, y + style.gap), group_callout(group),
                       "ANNOTATION", style.text)
        written += 1
        y -= 2.6 * style.text
    return written


def _emit_position_dimensions(doc: _Document, view: "ViewData", groups: Sequence["Group"],
                              offset: Point, mins: Sequence[float], maxs: Sequence[float],
                              style: _Style, limit: int) -> Tuple[int, int]:
    """Positional dimensions from the part datum to feature centres, stacked clear of the view.

    Mirrors ``drawing.py``'s policy — the same features get located here as on the SVG sheet, so
    the two documents never disagree about which datum a number came from."""
    from phimech.synth.drawing import VIEWS

    spec = VIEWS[view.name]
    dx, dy = offset
    x_datum, y_datum = mins[spec["ax"]] + dx, mins[spec["ay"]] + dy
    y_above, x_right = maxs[spec["ay"]] + dy, maxs[spec["ax"]] + dx
    x_stack = y_stack = 0
    done_x: List[float] = []
    done_y: List[float] = []

    def fresh(seen: List[float], value: float) -> bool:
        if any(abs(value - other) < 0.05 for other in seen):
            return False
        seen.append(value)
        return True

    dimensioned = 0
    for group in groups:
        if dimensioned >= limit:
            break
        member = group.members[0]
        if member.view != view.name:
            continue
        cx, cy = member.shape.center[0] + dx, member.shape.center[1] + dy
        need_x, need_y = fresh(done_x, cx), fresh(done_y, cy)
        if not (need_x or need_y):
            continue
        dimensioned += 1
        if need_x:
            x_stack += 1
            linear_dimension(doc, (x_datum, y_datum), (cx, cy), y_above + style.step * x_stack,
                             horizontal=True, style=style)
        if need_y:
            y_stack += 1
            linear_dimension(doc, (x_datum, y_datum), (cx, cy), x_datum - style.step * y_stack,
                             horizontal=False, style=style)
    x_stack += 1
    y_stack += 1
    linear_dimension(doc, (x_datum, y_datum), (x_right, y_datum),
                     y_above + style.step * x_stack, horizontal=True, style=style)
    linear_dimension(doc, (x_datum, y_datum), (x_datum, y_above),
                     x_datum - style.step * y_stack, horizontal=False, style=style)
    return x_stack, y_stack


def _emit_level_chain(doc: _Document, view: "ViewData", levels: Sequence[float], offset: Point,
                      mins: Sequence[float], maxs: Sequence[float], style: _Style) -> int:
    """Heights of the top view's z levels, stacked right of the right view (plate, boss, rim)."""
    from phimech.synth.drawing import VIEWS

    spec = VIEWS[view.name]
    dx, dy = offset
    base = mins[spec["ay"]] + dy
    x_line = maxs[spec["ax"]] + dx
    inner = list(levels[1:-1]) if len(levels) - 2 <= MAX_LEVEL_CHAIN else []
    written = 0
    for index, level in enumerate(inner):
        linear_dimension(doc, (x_line, base), (x_line, level + dy),
                         x_line + style.step * (index + 1), horizontal=False, style=style)
        written += 1
    linear_dimension(doc, (x_line, base), (x_line, maxs[spec["ay"]] + dy),
                     x_line + style.step * (written + 1), horizontal=False, style=style)
    return written + 1


def _emit_feature_labels(doc: _Document, groups: Sequence["Group"], origins: Mapping[str, Point],
                         style: _Style) -> int:
    """A small ID beside every instance, so the table's ``4x`` maps onto specific holes."""
    if sum(len(group.members) for group in groups) > MAX_FEATURE_LABELS:
        return 0
    written = 0
    for group in groups:
        for member in group.members:
            dx, dy = origins.get(member.view, (0.0, 0.0))
            shape = member.shape
            radius = max(shape.size) / 2.0
            doc.model.text((shape.center[0] + dx + radius * 0.75 + 0.3 * style.text,
                            shape.center[1] + dy + radius * 0.75),
                           group.group_id, "ANNOTATION", 0.8 * style.text)
            written += 1
    return written


def _table_rows(groups: Sequence["Group"], limit: int) -> List[List[str]]:
    from phimech.synth.drawing import VIEWS

    rows: List[List[str]] = [["ID", "QTY", "TYPE", "SIZE", "DEPTH", "POSITIONS (model mm)"]]
    for group in groups[:limit]:
        axes = "{}{}".format("xyz"[VIEWS[group.view]["ax"]], "xyz"[VIEWS[group.view]["ay"]])
        positions = ["({}, {})".format(_mm(m.shape.center[0]), _mm(m.shape.center[1]))
                     for m in group.members[:4]]
        if len(group.members) > 4:
            positions.append("+{} more".format(len(group.members) - 4))
        depth = "THRU" if group.thru else "{} deep".format(_mm(group.depth))
        if not group.measured:
            depth += " (est)"
        size = group_size_text(group)
        counterbore = group_counterbore(group)
        if counterbore is not None:
            size += ", " + counterbore_text(counterbore)
        rows.append([group.group_id, str(len(group.members)),
                     ("hole " if group.is_hole else "solid ") + group.kind, size, depth,
                     axes + ": " + "  ".join(positions)])
    if len(groups) > limit:
        rows.append(["", "", "+{} more groups".format(len(groups) - limit), "", "",
                     "see the drawing result JSON"])
    return rows


def _emit_table(doc: _Document, rows: Sequence[Sequence[str]], x: float, y_top: float,
                style: _Style) -> Tuple[float, float]:
    """The feature table as ruled lines and text; returns ``(width, bottom y)``."""
    text_height = 0.85 * style.text
    columns = [x]
    for width in (5.0, 4.0, 12.0, 15.0, 13.0, 34.0):
        columns.append(columns[-1] + width * style.text)
    row_h = 2.0 * style.text
    total_w = columns[-1] - columns[0]
    y = y_top
    for index, row in enumerate(rows):
        for column, cell in zip(columns, row):
            doc.model.text((column + 0.4 * style.text, y - row_h + 0.55 * style.text), cell,
                           "HOLE_TABLE", text_height)
        doc.model.line((x, y), (x + total_w, y), "HOLE_TABLE")
        y -= row_h
        if index == 0:
            doc.model.line((x, y), (x + total_w, y), "HOLE_TABLE")
    doc.model.line((x, y), (x + total_w, y), "HOLE_TABLE")
    for column in columns:
        doc.model.line((column, y_top), (column, y), "HOLE_TABLE")
    return total_w, y


def title_block_width(style: _Style) -> float:
    """How wide the title block will be, needed before it is placed so it can sit *beside* the
    feature table rather than on top of it."""
    return 46.0 * style.text


def _emit_title_block(doc: _Document, rows: Sequence[Tuple[str, str]], title: str, x_left: float,
                      y_top: float, style: _Style) -> Tuple[float, float]:
    """The title block as a reusable ``TITLEBLOCK`` block plus one ``INSERT``.

    A block is how every CAD tool expects a title block to arrive: it moves, it is one pick, and
    a recipient can redefine it once instead of editing loose lines. Returns ``(width, bottom)``."""
    width = title_block_width(style)
    title_h = 3.0 * style.text
    row_h = 2.0 * style.text
    height = title_h + row_h * len(rows)
    block = doc.new_block(TITLEBLOCK_BLOCK, track=False)
    space = block.space
    # drawn with its own origin at the bottom-left corner, so the INSERT point is that corner
    space.lwpolyline([(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)], "TITLEBLOCK")
    space.line((0.0, height - title_h), (width, height - title_h), "TITLEBLOCK")
    space.text((0.6 * style.text, height - title_h + 0.8 * style.text), title, "TITLEBLOCK",
               1.6 * style.text)
    y = height - title_h
    for key, value in rows:
        space.text((0.6 * style.text, y - row_h + 0.6 * style.text), key, "TITLEBLOCK",
                   0.8 * style.text)
        space.text((11.0 * style.text, y - row_h + 0.6 * style.text), value, "TITLEBLOCK",
                   0.95 * style.text)
        space.line((0.0, y - row_h), (width, y - row_h), "TITLEBLOCK")
        y -= row_h
    origin = (x_left, y_top - height)
    doc.model.insert(TITLEBLOCK_BLOCK, origin, "TITLEBLOCK")
    doc.grow(origin[0], origin[1])
    doc.grow(origin[0] + width, origin[1] + height)
    return width, origin[1]


def _emit_frame(doc: _Document, style: _Style,
                warnings: List[str]) -> Tuple[Optional[str], float, float]:
    """The sheet border around the finished 1:1 content; returns ``(sheet name, w, h)``.

    The sheet is chosen to *fit the geometry*, never the reverse — the geometry is never scaled,
    so the frame is placed around the content wherever the content happens to be."""
    x0, y0, x1, y1 = doc.extents
    content_w, content_h = x1 - x0, y1 - y0
    sheet = choose_sheet(content_w, content_h)
    if sheet is None:
        warnings.append(
            "content is {} x {} mm: larger than A0 at 1:1, so the frame follows the content "
            "instead of a standard sheet (the geometry is never scaled)".format(
                _mm(content_w), _mm(content_h)))
        name = None
        sheet_w = content_w + 2 * SHEET_MARGIN_MM
        sheet_h = content_h + 2 * SHEET_MARGIN_MM
    else:
        name, sheet_w, sheet_h = sheet
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = sheet_w / 2.0, sheet_h / 2.0
    doc.model.lwpolyline([(cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
                          (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)], "FRAME")
    inner_w, inner_h = half_w - SHEET_MARGIN_MM, half_h - SHEET_MARGIN_MM
    doc.model.lwpolyline([(cx - inner_w, cy - inner_h), (cx + inner_w, cy - inner_h),
                          (cx + inner_w, cy + inner_h), (cx - inner_w, cy + inner_h)], "FRAME")
    return name, sheet_w, sheet_h


# --------------------------------------------------------------------------------------
# results, inspection and the fidelity gate
# --------------------------------------------------------------------------------------


@dataclass
class DxfResult:
    """What :func:`write_dxf` produced. Lengths mm; nothing here is rounded away from measured."""

    path: str
    version: str = DXF_VERSION
    units: str = "mm"
    scale: str = "1:1"
    sheet: Optional[str] = None
    sheet_mm: List[float] = field(default_factory=list)
    bytes: int = 0
    text_height_mm: float = 0.0
    ltscale: float = 1.0
    extents: Dict[str, List[float]] = field(default_factory=dict)
    origins: Dict[str, List[float]] = field(default_factory=dict)
    layers: List[str] = field(default_factory=list)
    linetypes: List[str] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    block_entity_counts: Dict[str, int] = field(default_factory=dict)
    layers_used: Dict[str, int] = field(default_factory=dict)
    circles: int = 0
    dimensions: int = 0
    dimension_values: List[float] = field(default_factory=list)
    callouts: int = 0
    feature_labels: int = 0
    groups: int = 0
    #: Hidden-edge boxes refused because they fell outside the view's own silhouette.
    hidden_edges_dropped: int = 0
    #: Groups whose extent is a slice-level span because their own wall was not found.
    unmeasured_groups: int = 0
    counterbores: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "units": self.units,
            "scale": self.scale,
            "sheet": self.sheet,
            "sheet_mm": self.sheet_mm,
            "bytes": self.bytes,
            "text_height_mm": self.text_height_mm,
            "ltscale": round(self.ltscale, 4),
            "extents": self.extents,
            "origins": self.origins,
            "layers": self.layers,
            "linetypes": self.linetypes,
            "entity_counts": self.entity_counts,
            "block_entity_counts": self.block_entity_counts,
            "layers_used": self.layers_used,
            "circles": self.circles,
            "dimensions": self.dimensions,
            "dimension_values": self.dimension_values,
            "callouts": self.callouts,
            "feature_labels": self.feature_labels,
            "groups": self.groups,
            "hidden_edges_dropped": self.hidden_edges_dropped,
            "unmeasured_groups": self.unmeasured_groups,
            "counterbores": self.counterbores,
            "warnings": self.warnings,
            "note": "AC1015 (R2000) DXF, 1:1 in millimetres; the top view is at true model "
                    "coordinates. Holes are CIRCLE entities, not polygons. Dimensions are "
                    "editable on import, not associative to any model.",
        }


@dataclass
class DxfFacts:
    """Everything a re-read of a written DXF could establish, for gates and for `doctor`."""

    path: Path
    size_bytes: int
    version: Optional[str] = None
    insunits: Optional[int] = None
    measurement: Optional[int] = None
    dimlunit: Optional[int] = None
    dimdec: Optional[int] = None
    dimzin: Optional[int] = None
    dimdsep: Optional[int] = None
    ltscale: Optional[float] = None
    extmin: Optional[Tuple[float, float]] = None
    extmax: Optional[Tuple[float, float]] = None
    layers: Dict[str, Tuple[int, str]] = field(default_factory=dict)
    linetypes: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    dimstyles: List[str] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    layers_used: Dict[str, int] = field(default_factory=dict)
    circles: List[Tuple[str, float, float, float]] = field(default_factory=list)
    arcs: List[Tuple[str, float, float, float, float, float]] = field(default_factory=list)
    #: (layer, vertex count, bbox) per LWPOLYLINE: a large count where a circle was expected
    #: is exactly the tessellation failure this module exists to avoid, and the bbox is what
    #: says a hidden-edge box sits on material rather than in the gap between two slabs.
    polylines: List[Tuple[str, int, Tuple[float, float, float, float]]] = \
        field(default_factory=list)
    dimensions: List[Dict[str, Any]] = field(default_factory=list)
    content_bbox: Optional[Tuple[float, float, float, float]] = None


_HEADER_INT_VARS = {"$INSUNITS": "insunits", "$MEASUREMENT": "measurement",
                    "$DIMLUNIT": "dimlunit", "$DIMDEC": "dimdec", "$DIMZIN": "dimzin",
                    "$DIMDSEP": "dimdsep"}


def read_tags(text: str) -> List[Tuple[int, str]]:
    """Every ``(group code, value)`` pair of a DXF file, in order.

    DXF is line-oriented: an integer code line followed by a value line. Parsing it back is the
    cheapest possible independent check that the writer produced a well-formed stream."""
    lines = text.splitlines()
    tags: List[Tuple[int, str]] = []
    index = 0
    while index + 1 < len(lines):
        raw = lines[index].strip()
        try:
            code = int(raw)
        except ValueError:
            raise ValueError("line {}: {!r} is not a DXF group code".format(index + 1, raw))
        tags.append((code, lines[index + 1]))
        index += 2
    if index != len(lines) and lines[index:] != [""]:
        raise ValueError("odd number of lines: a group code has no value")
    return tags


def inspect_dxf(path: Any) -> DxfFacts:
    """Re-read a DXF and report what a CAD reader would find: tables, entities, extents."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    facts = DxfFacts(path=path, size_bytes=len(text.encode("utf-8")))
    section = ""
    table = ""
    entity: Optional[str] = None
    record: Dict[int, List[str]] = {}
    bbox = [math.inf, math.inf, -math.inf, -math.inf]
    variable = ""

    def flush() -> None:
        if entity is None:
            return
        layer = record.get(8, ["0"])[0]
        if section == "ENTITIES":
            facts.entity_counts[entity] = facts.entity_counts.get(entity, 0) + 1
            facts.layers_used[layer] = facts.layers_used.get(layer, 0) + 1
        radius = float(record.get(40, ["0"])[0]) if entity in ("CIRCLE", "ARC") else 0.0
        xs = [float(v) for code in (10, 11, 12, 13, 14, 15) for v in record.get(code, [])]
        ys = [float(v) for code in (20, 21, 22, 23, 24, 25) for v in record.get(code, [])]
        if section != "ENTITIES":
            # Only model space is reported: that is what a reader queries, and every arrowhead
            # and dimension text lives inside a *D block that belongs to its DIMENSION.
            return
        if entity == "CIRCLE" and xs and ys:
            facts.circles.append((layer, xs[0], ys[0], radius))
        if entity == "ARC" and xs and ys:
            facts.arcs.append((layer, xs[0], ys[0], radius, float(record.get(50, ["0"])[0]),
                               float(record.get(51, ["0"])[0])))
        if entity == "LWPOLYLINE" and xs and ys:
            facts.polylines.append((layer, int(record.get(90, ["0"])[0]),
                                    (min(xs), min(ys), max(xs), max(ys))))
        if entity == "DIMENSION":
            facts.dimensions.append({
                "block": record.get(2, [""])[0],
                "measurement": float(record.get(42, ["0"])[0]),
                "rotation": float(record.get(50, ["0"])[0]),
                "override": record.get(1, [None])[0],
                "p1": (float(record.get(13, ["0"])[0]), float(record.get(23, ["0"])[0])),
                "p2": (float(record.get(14, ["0"])[0]), float(record.get(24, ["0"])[0])),
                "style": record.get(3, [""])[0],
            })
        for x in xs:
            bbox[0] = min(bbox[0], x - radius)
            bbox[2] = max(bbox[2], x + radius)
        for y in ys:
            bbox[1] = min(bbox[1], y - radius)
            bbox[3] = max(bbox[3], y + radius)

    for code, value in read_tags(text):
        if code == 0:
            flush()
            entity = None
            record = {}
            if value in ("SECTION", "ENDSEC"):
                section = ""
            elif value in ("TABLE", "ENDTAB"):
                table = ""
            elif section == "TABLES" and value in ("LAYER", "LTYPE", "DIMSTYLE", "BLOCK_RECORD",
                                                   "STYLE", "APPID", "VPORT", "VIEW", "UCS"):
                entity = value
            elif section in ("ENTITIES", "BLOCKS") and value != "EOF":
                entity = value
            continue
        if not section and code == 2:
            section = value
            continue
        if section == "TABLES" and entity is None and code == 2 and not table:
            table = value
            continue
        if section == "HEADER":
            if code == 9:
                variable = value
                continue
            if variable == "$ACADVER" and code == 1:
                facts.version = value
            elif variable in _HEADER_INT_VARS and code == 70:
                setattr(facts, _HEADER_INT_VARS[variable], int(value))
            elif variable == "$LTSCALE" and code == 40:
                facts.ltscale = float(value)
            elif variable in ("$EXTMIN", "$EXTMAX") and code in (10, 20):
                current = facts.extmin if variable == "$EXTMIN" else facts.extmax
                pair = (float(value), current[1] if current else 0.0) if code == 10 else \
                    ((current[0] if current else 0.0), float(value))
                if variable == "$EXTMIN":
                    facts.extmin = pair
                else:
                    facts.extmax = pair
            continue
        record.setdefault(code, []).append(value)
        if section == "TABLES" and entity == "LAYER":
            if code == 2:
                facts.layers[value] = (0, "CONTINUOUS")
            elif code in (62, 6) and facts.layers:
                name = list(facts.layers)[-1]
                color, linetype = facts.layers[name]
                facts.layers[name] = (int(value) if code == 62 else color,
                                      value if code == 6 else linetype)
        elif section == "TABLES" and entity == "LTYPE" and code == 2:
            facts.linetypes.append(value)
        elif section == "TABLES" and entity == "DIMSTYLE" and code == 2:
            facts.dimstyles.append(value)
        elif section == "BLOCKS" and entity == "BLOCK" and code == 2:
            facts.blocks.append(value)
    flush()
    if bbox[0] != math.inf:
        facts.content_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
    return facts


def source_distances(views: Mapping[str, "ViewData"], groups: Sequence["Group"]) -> List[float]:
    """Every distance the source data can justify a dimension for.

    The gate uses this to prove no dimension value was invented: overall sizes of each view, the
    z levels, and each feature centre's offset from its view's datum. Recomputed from ``views``
    and ``groups``, not from what the writer happened to do."""
    from phimech.synth.drawing import VIEWS

    mins, maxs = _model_bounds(views)
    values: List[float] = []
    for name, view in views.items():
        spec = VIEWS[name]
        values.append(maxs[spec["ax"]] - mins[spec["ax"]])
        values.append(maxs[spec["ay"]] - mins[spec["ay"]])
    for group in groups:
        spec = VIEWS[group.view]
        for member in group.members:
            values.append(member.shape.center[0] - mins[spec["ax"]])
            values.append(member.shape.center[1] - mins[spec["ay"]])
    top = views.get("top")
    if top is not None:
        for level in top.levels:
            values.append(level - mins[2])
    return [abs(value) for value in values]


def check_fidelity(result: DxfResult, views: Mapping[str, "ViewData"],
                   groups: Optional[Sequence["Group"]] = None, *, radius_tol: float = 0.001,
                   position_tol: float = 0.001, dimension_tol: float = 0.01,
                   curve_tol: float = CURVE_TOL_MM) -> List[str]:
    """Numeric gate on a written DXF. An empty list is a pass; every entry is a measured failure.

    Each check has a threshold (the core boundary):

    * every fitted circle in the source appears once as a ``CIRCLE`` within ``radius_tol`` and
      ``position_tol`` — no hole may have degraded into a polygon;
    * the layer table is exactly the documented set with the documented colours and linetypes,
      and no entity references an undefined layer or linetype;
    * the header declares R2000, millimetres and the ISO dimension-text variables;
    * every ``DIMENSION`` carries no text override, names a block that exists, and its stated
      measurement equals the distance between its own definition points, and (when ``groups``
      is supplied) equals some distance the source data can justify, within ``dimension_tol``;
    * ``$EXTMIN``/``$EXTMAX`` cover every entity actually written."""
    problems: List[str] = []
    facts = inspect_dxf(result.path)
    if facts.version != DXF_VERSION:
        problems.append("header $ACADVER is {!r}, expected {!r}".format(facts.version,
                                                                        DXF_VERSION))
    for name, attribute, expected in (("$INSUNITS", "insunits", INSUNITS_MM),
                                      ("$MEASUREMENT", "measurement", MEASUREMENT_METRIC),
                                      ("$DIMLUNIT", "dimlunit", DIMLUNIT_DECIMAL),
                                      ("$DIMDEC", "dimdec", DIMDEC),
                                      ("$DIMZIN", "dimzin", DIMZIN_SUPPRESS_TRAILING),
                                      ("$DIMDSEP", "dimdsep", DIMDSEP_POINT)):
        actual = getattr(facts, attribute)
        if actual != expected:
            problems.append("header {} is {!r}, expected {}".format(name, actual, expected))
    if DIMSTYLE_NAME not in facts.dimstyles:
        problems.append("dimension style {!r} is not defined (found {})".format(
            DIMSTYLE_NAME, facts.dimstyles))

    origins = {name: (float(v[0]), float(v[1])) for name, v in result.origins.items()}
    remaining = list(facts.circles)
    for layer, x, y, radius in planned_circles(views, origins, curve_tol):
        match = None
        for candidate in remaining:
            if (candidate[0] == layer and abs(candidate[1] - x) <= position_tol
                    and abs(candidate[2] - y) <= position_tol
                    and abs(candidate[3] - radius) <= radius_tol):
                match = candidate
                break
        if match is None:
            near = [c for c in remaining if abs(c[1] - x) <= 0.5 and abs(c[2] - y) <= 0.5]
            problems.append(
                "fitted circle r={:.4f} at ({:.4f}, {:.4f}) on {} is not a CIRCLE in the file "
                "(nearest: {})".format(radius, x, y, layer, near[:1] or "none"))
        else:
            remaining.remove(match)

    defined = set(facts.layers)
    if defined != set(REQUIRED_LAYERS):
        problems.append("layer table is {} , expected exactly {}".format(
            sorted(defined), sorted(REQUIRED_LAYERS)))
    for name in facts.layers_used:
        if name not in facts.layers:
            problems.append("layer {!r} is used by an entity but not defined in TABLES".format(
                name))
    for name, (color, linetype) in facts.layers.items():
        if linetype not in facts.linetypes:
            problems.append("layer {!r} references linetype {!r}, which is not defined".format(
                name, linetype))
        spec = LAYER_BY_NAME.get(name)
        if spec is not None and (color, linetype) != (spec.color, spec.linetype):
            problems.append("layer {!r} is written as colour {} linetype {!r}, but the scheme "
                            "says colour {} linetype {!r}".format(name, color, linetype,
                                                                  spec.color, spec.linetype))

    # ``groups`` must be the same list write_dxf was given; None means "do not check this".
    legal = source_distances(views, groups) if groups is not None else []
    if len(facts.dimensions) != result.dimensions:
        problems.append("file has {} DIMENSION entities, the writer reported {}".format(
            len(facts.dimensions), result.dimensions))
    for dimension in facts.dimensions:
        if dimension["block"] not in facts.blocks:
            problems.append("DIMENSION names block {!r}, which is not in BLOCKS".format(
                dimension["block"]))
        if dimension["override"] is not None:
            problems.append("DIMENSION {} carries a text override {!r}: the value must be the "
                            "measurement".format(dimension["block"], dimension["override"]))
        p1, p2 = dimension["p1"], dimension["p2"]
        span = abs(p2[1] - p1[1]) if abs(dimension["rotation"] - 90.0) < 1e-6 else \
            abs(p2[0] - p1[0])
        if abs(span - dimension["measurement"]) > dimension_tol:
            problems.append(
                "DIMENSION {} states {:.4f} but its definition points are {:.4f} apart".format(
                    dimension["block"], dimension["measurement"], span))
        if legal and not any(abs(dimension["measurement"] - value) <= dimension_tol
                             for value in legal):
            problems.append("DIMENSION {} states {:.4f}, which no source measurement "
                            "justifies".format(dimension["block"], dimension["measurement"]))

    if facts.content_bbox and facts.extmin and facts.extmax:
        x0, y0, x1, y1 = facts.content_bbox
        slack = 1e-6
        if facts.extmin[0] > x0 + slack or facts.extmin[1] > y0 + slack:
            problems.append("$EXTMIN {} does not cover the content min ({:.4f}, {:.4f})".format(
                facts.extmin, x0, y0))
        if facts.extmax[0] < x1 - slack or facts.extmax[1] < y1 - slack:
            problems.append("$EXTMAX {} does not cover the content max ({:.4f}, {:.4f})".format(
                facts.extmax, x1, y1))
    return problems


# --------------------------------------------------------------------------------------
# the public entry points
# --------------------------------------------------------------------------------------


def write_dxf(
    views: Mapping[str, "ViewData"],
    groups: Sequence["Group"],
    out_path: Any,
    *,
    title: str = "",
    source_label: str = "",
    part: Optional[str] = None,
    volume_mm3: Optional[float] = None,
    date: Optional[str] = None,
    dimensions: bool = True,
    frame: bool = True,
    curve_tol: float = CURVE_TOL_MM,
    max_position_dims: int = MAX_POSITION_DIMS,
    max_callouts: int = MAX_CALLOUTS,
    max_table_rows: int = MAX_TABLE_ROWS,
    feature_labels: bool = True,
) -> DxfResult:
    """Write ``out_path`` as an editable AC1015 DXF of the geometry the SVG sheet draws.

    Pure function: it reads the already-measured ``ViewData``/``Group`` objects produced by
    ``drawing.py`` and writes one file. It runs no subprocess, needs no backend and imports
    nothing outside the standard library, so it is safe to call from anywhere in the pipeline.

    ``views`` maps ``"top" | "front" | "right"`` to ``ViewData``; ``groups`` is the feature
    grouping behind the hole table. Everything else is title-block metadata that must already be
    known — this function measures nothing of its own and invents nothing."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    mins, maxs = _model_bounds(views)
    width, depth, height = (maxs[axis] - mins[axis] for axis in range(3))
    text_height = choose_text_height(max(width + depth, height + depth, 1.0))
    style = _style_for(text_height)
    gap = max(8.0 * text_height, 0.12 * max(width, depth, height, 1.0))
    origins = view_origins(views, gap)

    doc = _Document()
    order = [name for name in ("top", "front", "right") if name in views]
    dropped = 0
    for name in order:
        others = [views[other] for other in order if other != name]
        dropped += _emit_view(doc, views[name], others, origins[name], curve_tol, warnings,
                              style)

    if dimensions:
        from phimech.synth.drawing import VIEWS

        for name in order:
            if name == "top":
                _emit_position_dimensions(doc, views[name], groups, origins[name], mins, maxs,
                                          style, max_position_dims)
            else:
                spec = VIEWS[name]
                dx, dy = origins[name]
                linear_dimension(doc, (mins[spec["ax"]] + dx, mins[spec["ay"]] + dy),
                                 (maxs[spec["ax"]] + dx, mins[spec["ay"]] + dy),
                                 mins[spec["ay"]] + dy - style.step, horizontal=True,
                                 style=style)
        if "right" in views and "top" in views and views["top"].levels:
            _emit_level_chain(doc, views["right"], views["top"].levels, origins["right"], mins,
                              maxs, style)

    # callouts stack right of the top view, above the right view's band (as on the SVG sheet)
    callouts = _emit_callouts(doc, groups, origins, maxs[0] + gap, maxs[1],
                              mins[1] + 2.0 * text_height, style, max_callouts)
    if callouts < len(groups):
        warnings.append(
            "{} of {} group callouts fitted beside the plan; every group is still a row in the "
            "feature table".format(callouts, len(groups)))
    labels = _emit_feature_labels(doc, groups, origins, style) if feature_labels else 0

    # the bottom band: feature table on the left, title block beside it on the right, notes
    # under both — measured in that order so nothing is written over anything else
    content_x0 = doc.extents[0]
    band_top = doc.extents[1] - gap
    table_w, table_bottom = _emit_table(doc, _table_rows(groups, max_table_rows), content_x0,
                                        band_top, style)
    sheet_name: Optional[str] = None
    sheet_mm: List[float] = []
    title_rows: List[Tuple[str, str]] = [
        ("SOURCE", source_label[:40] or "-"),
        ("PART", (part or "-")[:40]),
        ("SIZE mm", "{} x {} x {}".format(_mm(width), _mm(depth), _mm(height))),
        ("VOLUME", "{} mm3".format(_mm(volume_mm3)) if volume_mm3 is not None else "-"),
        ("UNITS / SCALE", "mm, 1:1 true size"),
        ("PROJECTION", "third angle"),
        ("DATE", date or time.strftime("%Y-%m-%d")),
        ("GENERATED BY", "PhiMech drawing/dxf"),
    ]
    title_left = max(content_x0 + table_w + 2.0 * text_height,
                     maxs[0] + gap + depth - title_block_width(style))
    _, title_bottom = _emit_title_block(doc, title_rows, (title or part or "DRAWING")[:34],
                                        title_left, band_top, style)
    notes = "\\P".join([
        "UNITS: MILLIMETRES. GEOMETRY IS 1:1 TRUE SIZE; THE TOP VIEW IS AT TRUE MODEL "
        "COORDINATES.",
        "THIRD-ANGLE PROJECTION. EVERY VALUE IS MEASURED FROM THE MESH; NONE IS HAND ENTERED.",
        "LAYERS: VISIBLE = OUTLINE, SECTION = FEATURE EDGES AT OTHER LEVELS (VISIBILITY NOT "
        "RESOLVED), HIDDEN = ONE BOX PER RUN OF MATERIAL A FEATURE PASSES THROUGH, PROJECTED "
        "FROM ANOTHER VIEW.",
        "(EST) MARKS AN EXTENT TAKEN FROM SLICE LEVELS BECAUSE THE FEATURE'S OWN WALL WAS NOT "
        "FOUND; EVERY OTHER NUMBER IS MEASURED OFF A WALL.",
        "NO TOLERANCE, THREAD, MATERIAL OR FINISH IS SPECIFIED: THE MODEL DOES NOT CARRY THEM.",
        "ADVISORY DRAWING, NOT A GATE.",
    ])
    doc.model.mtext((content_x0, min(table_bottom, title_bottom) - 1.5 * text_height), notes,
                    "ANNOTATION", 0.9 * text_height, max(table_w, 60.0 * text_height))
    if frame:
        sheet_name, sheet_w, sheet_h = _emit_frame(doc, style, warnings)
        sheet_mm = [sheet_w, sheet_h]

    ltscale = max(text_height / 3.5, 1e-3)
    text = serialise(doc, text_height, style.arrow, ltscale)
    out_path.write_text(text, encoding="utf-8")

    x0, y0, x1, y1 = doc.extents
    return DxfResult(
        path=str(out_path),
        sheet=sheet_name,
        sheet_mm=[round(value, 3) for value in sheet_mm],
        bytes=len(text.encode("utf-8")),
        text_height_mm=text_height,
        ltscale=ltscale,
        extents={"min": [round(x0, 4), round(y0, 4)], "max": [round(x1, 4), round(y1, 4)]},
        origins={name: [round(origin[0], 6), round(origin[1], 6)]
                 for name, origin in origins.items() if name in views},
        layers=[layer.name for layer in LAYERS],
        linetypes=[linetype.name for linetype in LINETYPES],
        entity_counts=dict(sorted(doc.counts.items())),
        block_entity_counts=dict(sorted(doc.block_counts.items())),
        layers_used=dict(sorted(doc.layers_used.items())),
        circles=doc.counts.get("CIRCLE", 0),
        dimensions=len(doc.dimensions),
        dimension_values=[round(plan.value, 4) for plan in doc.dimensions],
        callouts=callouts,
        feature_labels=labels,
        groups=len(groups),
        hidden_edges_dropped=dropped,
        unmeasured_groups=sum(1 for group in groups if not group.measured),
        counterbores=sum(1 for group in groups if group_counterbore(group) is not None),
        warnings=warnings,
    )


def write_dxf_for_result(
    result: "DrawingResult",
    views: Mapping[str, "ViewData"],
    groups: Sequence["Group"],
    out_path: Any = None,
    **kwargs: Any,
) -> DxfResult:
    """:func:`write_dxf` with the title-block metadata taken from a finished ``DrawingResult``.

    This is the wiring seam for ``drawing.render_drawing``: it already holds the views, the
    groups and the result, and the DXF is the same drawing in an editable container. The default
    output path is the SVG's path with a ``.dxf`` suffix."""
    target = Path(out_path) if out_path else Path(result.path).with_suffix(".dxf")
    kwargs.setdefault("title", result.part or Path(result.source or result.path).stem)
    kwargs.setdefault("source_label", result.source or "")
    kwargs.setdefault("part", result.part)
    kwargs.setdefault("volume_mm3", result.volume_mm3)
    return write_dxf(views, groups, target, **kwargs)
