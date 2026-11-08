#!/usr/bin/env python3
"""Build the committed drawing fixtures: one correct sheet and ten planted defects.

Why a generator, and why its output is committed
------------------------------------------------
The fixtures must test *the checker*, not whatever ``phimech/synth/drawing.py`` renders this
week - the drawing module's measurement internals are being changed in parallel, and a fixture
regenerated from the live pipeline would drift with it.  So the sheets here are frozen files.
This script is how they were made and is the only readable statement of what each defect *is*:
every defective fixture is the correct sheet plus exactly one surgical mutation, listed in
``DEFECTS`` below and repeated in each fixture's ``expect.json``.

Re-running it rewrites the committed files byte for byte (nothing here reads the clock).  A
fixture is only worth keeping if it fails exactly one check, which ``tests/test_drawing_checks.py``
asserts against every directory it finds.

The part: a 60 x 40 x 9 mm plate with three rectangular through-windows, all on y = 20 -
6 x 6 at x = 10 and x = 45 (one group, two members) and 10 x 6 at x = 30 (a second group).
Two members in one group is the point: it is what makes member-level coverage differ from
group-level coverage, and what the datum-plus-pitch dimension chain has to reach.

Units are millimetres.  Stdlib only, python 3.9.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# the part
# --------------------------------------------------------------------------------------

PLATE = (60.0, 40.0, 9.0)
#: (x0, x1, y0, y1) of each rectangular through-window.
WINDOWS: Tuple[Tuple[float, float, float, float], ...] = (
    (7.0, 13.0, 17.0, 23.0),     # 6 x 6 at (10, 20)
    (25.0, 35.0, 17.0, 23.0),    # 10 x 6 at (30, 20)
    (42.0, 48.0, 17.0, 23.0),    # 6 x 6 at (45, 20)
)
VOLUME_MM3 = PLATE[0] * PLATE[1] * PLATE[2] - sum(
    (x1 - x0) * (y1 - y0) * PLATE[2] for x0, x1, y0, y1 in WINDOWS)

SCALE = 2.0
SHEET = (420.0, 297.0)

GROUPS: List[Dict[str, Any]] = [
    {"id": "H1", "qty": 1, "kind": "rect", "hole": True, "thru": True, "size": [10.0, 6.0],
     "corner_r": 0.0, "depth": 9.0, "view": "top", "positions": [[30.0, 20.0]],
     "depth_from": 0.0, "depth_to": 9.0},
    {"id": "H2", "qty": 2, "kind": "rect", "hole": True, "thru": True, "size": [6.0, 6.0],
     "corner_r": 0.0, "depth": 9.0, "view": "top", "positions": [[10.0, 20.0], [45.0, 20.0]],
     "depth_from": 0.0, "depth_to": 9.0},
]

VIEWS: Dict[str, Dict[str, Any]] = {
    "top": {"bbox": [0.0, 0.0, 60.0, 40.0], "features": 3, "slices": 1, "seconds": 0.0},
    "front": {"bbox": [0.0, 0.0, 60.0, 9.0], "features": 3, "slices": 0, "seconds": 0.0},
    "right": {"bbox": [0.0, 0.0, 40.0, 9.0], "features": 0, "slices": 0, "seconds": 0.0},
}

# --------------------------------------------------------------------------------------
# the sheet dialect (the same elements phimech/synth/drawing.py writes)
# --------------------------------------------------------------------------------------

_FONT = "Helvetica, Arial, sans-serif"
_MONO = "Menlo, Consolas, monospace"
_INK = "#111"
_DIM = "#1d4ed8"
_HIDDEN = "#666"
_TABLE_BG = "#f3f4f6"
_TEXT_MM = 2.6


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


class Svg:
    """Minimal SVG builder in paper millimetres, y down; element-for-element the drawing's."""

    def __init__(self, width: float, height: float) -> None:
        self.w, self.h = width, height
        self.parts: List[str] = []

    def line(self, x0: float, y0: float, x1: float, y1: float, *, width: float = 0.35,
             color: str = _INK, dash: Optional[str] = None, arrows: bool = False,
             cls: Optional[str] = None) -> None:
        extra = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        if arrows:
            extra += ' marker-start="url(#arw)" marker-end="url(#arw)"'
        if cls:
            extra += ' class="{}"'.format(cls)
        self.parts.append(
            '<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}" stroke="{}" '
            'stroke-width="{:.2f}" stroke-linecap="round"{}/>'.format(
                x0, y0, x1, y1, color, width, extra))

    def poly(self, pts: Sequence[Tuple[float, float]], *, width: float = 0.35,
             color: str = _INK, dash: Optional[str] = None, fill: str = "none",
             cls: Optional[str] = None) -> None:
        d = " ".join("{:.3f},{:.3f}".format(x, y) for x, y in pts)
        extra = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        if cls:
            extra += ' class="{}"'.format(cls)
        self.parts.append(
            '<polygon points="{}" fill="{}" stroke="{}" stroke-width="{:.2f}" '
            'stroke-linejoin="round"{}/>'.format(d, fill, color, width, extra))

    def rect(self, x: float, y: float, w: float, h: float, *, width: float = 0.35,
             color: str = _INK, fill: str = "none") -> None:
        self.parts.append(
            '<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="{}" stroke="{}" '
            'stroke-width="{:.2f}"/>'.format(x, y, w, h, fill, color, width))

    def text(self, x: float, y: float, s: str, *, size: float = _TEXT_MM, anchor: str = "start",
             color: str = _INK, bold: bool = False, mono: bool = False, rotate: float = 0.0,
             baseline: str = "auto") -> None:
        weight = ' font-weight="bold"' if bold else ""
        tr = ' transform="rotate({:.1f} {:.3f} {:.3f})"'.format(rotate, x, y) if rotate else ""
        self.parts.append(
            '<text x="{:.3f}" y="{:.3f}" font-family="{}" font-size="{:.2f}" text-anchor="{}" '
            'fill="{}" dominant-baseline="{}"{}{}>{}</text>'.format(
                x, y, _MONO if mono else _FONT, size, anchor, color, baseline, weight, tr,
                _esc(s)))

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


def _num(v: float) -> str:
    return "{:.2f}".format(v).rstrip("0").rstrip(".") or "0"


class Frame:
    """Model mm -> paper mm for one view; the datum (bbox min) sits at (ox, oy)."""

    def __init__(self, bbox: Sequence[float], ox: float, oy: float, scale: float = SCALE) -> None:
        self.x0, self.y0, self.x1, self.y1 = [float(v) for v in bbox]
        self.ox, self.oy, self.s = ox, oy, scale

    def px(self, x: float) -> float:
        return self.ox + self.s * (x - self.x0)

    def py(self, y: float) -> float:
        return self.oy - self.s * (y - self.y0)

    @property
    def top(self) -> float:
        return self.py(self.y1)

    @property
    def right(self) -> float:
        return self.px(self.x1)


def dim_h(svg: Svg, fr: Frame, xa: float, xb: float, y_line: float,
          text: Optional[str] = None) -> None:
    pa, pb = fr.px(xa), fr.px(xb)
    svg.line(pa, y_line, pb, y_line, width=0.25, color=_DIM, arrows=abs(pb - pa) > 4,
             cls="dim")
    svg.text((pa + pb) / 2.0, y_line - 0.8, text if text is not None else _num(abs(xb - xa)),
             anchor="middle", color=_DIM)


def dim_v(svg: Svg, fr: Frame, ya: float, yb: float, x_line: float,
          text: Optional[str] = None) -> None:
    pa, pb = fr.py(ya), fr.py(yb)
    svg.line(x_line, pa, x_line, pb, width=0.25, color=_DIM, arrows=abs(pb - pa) > 4,
             cls="dim")
    svg.text(x_line - 0.8, (pa + pb) / 2.0, text if text is not None else _num(abs(yb - ya)),
             anchor="middle", color=_DIM, rotate=-90)


# --------------------------------------------------------------------------------------
# the correct sheet
# --------------------------------------------------------------------------------------

BAND_TOP = 241.0
TITLE_X, TITLE_W, TITLE_H = 292.0, 118.0, 42.0
TABLE_X, TABLE_W = 10.0, 278.0
LEVEL_CHAIN_X = 267.0


def build_sheet(levels: Sequence[float] = (0.0, 9.0)) -> str:
    """The correct sheet.  ``levels`` is the z-level chain the right view prints."""
    svg = Svg(*SHEET)
    svg.rect(5, 5, 410, 287, width=0.5)

    top = Frame(VIEWS["top"]["bbox"], 40.0, 178.0)
    front = Frame(VIEWS["front"]["bbox"], 40.0, 210.0)
    right = Frame(VIEWS["right"]["bbox"], 180.0, 210.0)

    # geometry: outline plus the windows, hidden edges dashed in the side views
    for fr, boxes, dash in (
        (top, [(x0, y0, x1, y1) for x0, x1, y0, y1 in WINDOWS], None),
        (front, [(x0, 0.0, x1, 9.0) for x0, x1, _, _ in WINDOWS], "1.2,0.8"),
        (right, [(y0, 0.0, y1, 9.0) for _, _, y0, y1 in WINDOWS], "1.2,0.8"),
    ):
        svg.poly([(fr.px(fr.x0), fr.py(fr.y0)), (fr.px(fr.x1), fr.py(fr.y0)),
                  (fr.px(fr.x1), fr.py(fr.y1)), (fr.px(fr.x0), fr.py(fr.y1))], width=0.5,
                 cls="outline")
        for x0, y0, x1, y1 in boxes:
            svg.poly([(fr.px(x0), fr.py(y0)), (fr.px(x1), fr.py(y0)),
                      (fr.px(x1), fr.py(y1)), (fr.px(x0), fr.py(y1))],
                     width=0.2, color=_HIDDEN, dash=dash)

    for fr, name in ((top, "TOP"), (front, "FRONT"), (right, "RIGHT")):
        svg.text(fr.ox, fr.oy + 5.5, name, size=3.0, bold=True)
    svg.text(top.ox - 2, top.oy + 3.5, "datum ({}, {})".format(_num(top.x0), _num(top.y0)),
             size=2.0, anchor="end", color=_DIM)

    # position dimensions: H1 from the datum, H2 from the datum plus the pitch to its twin
    dim_h(svg, top, 0.0, 30.0, top.top - 7.0)                      # H1 x
    dim_h(svg, top, 0.0, 10.0, top.top - 14.0)                     # H2 x
    dim_h(svg, top, 10.0, 45.0, top.top - 14.0)                    # H2 pitch  [PITCH]
    dim_h(svg, top, 0.0, 60.0, top.top - 21.0)                     # overall width
    dim_v(svg, top, 0.0, 20.0, top.ox - 7.0)                       # H1/H2 y
    dim_v(svg, top, 0.0, 40.0, top.ox - 14.0)                      # overall depth
    dim_h(svg, front, 0.0, 60.0, front.oy + 11.0)                  # front overall  [FRONTDIM]
    dim_h(svg, right, 0.0, 40.0, right.oy + 11.0)                  # right overall
    inner = list(levels[1:-1])
    for index, level in enumerate(inner):
        dim_v(svg, right, 0.0, level, LEVEL_CHAIN_X + 7.0 * index)
    dim_v(svg, right, 0.0, 9.0, LEVEL_CHAIN_X + 7.0 * len(inner))

    # leader callouts, stacked to the right of the top view
    column = top.right + 12.0
    for index, (group, anchor) in enumerate(zip(GROUPS, ((30.0, 23.0), (13.0, 23.0)))):
        y = 102.0 + 5.5 * index
        svg.line(top.px(anchor[0]), top.py(anchor[1]), column - 1.5, y, width=0.2, color=_INK,
                 cls="leader")
        svg.line(column - 1.5, y, column, y, width=0.2, color=_INK, cls="leader")
        svg.text(column + 0.8, y, callout(group), baseline="middle")

    # feature table
    row_h = 4.6
    cols = [(0, "ID"), (9, "QTY"), (17, "TYPE"), (42, "SIZE"), (72, "DEPTH"),
            (114, "POSITIONS (model mm)")]
    svg.rect(TABLE_X, BAND_TOP, TABLE_W, row_h, fill=_TABLE_BG, width=0.25)
    for dx, name in cols:
        svg.text(TABLE_X + dx + 1.2, BAND_TOP + row_h - 1.3, name, size=2.4, bold=True)
    yy = BAND_TOP + row_h
    for group in GROUPS:
        cells = [group["id"], str(group["qty"]), "hole " + group["kind"],
                 "{} x {}".format(_num(group["size"][0]), _num(group["size"][1])), "THRU",
                 "xy: " + "  ".join("({}, {})".format(_num(p[0]), _num(p[1]))
                                    for p in group["positions"])]
        for (dx, _), cell in zip(cols, cells):
            svg.text(TABLE_X + dx + 1.2, yy + row_h - 1.3, cell, size=2.3, mono=True)
        svg.line(TABLE_X, yy + row_h, TABLE_X + TABLE_W, yy + row_h, width=0.15, color="#999")
        yy += row_h
    svg.rect(TABLE_X, BAND_TOP, TABLE_W, yy - BAND_TOP, width=0.3)

    notes = [
        "Units mm. Third-angle projection, scale 2:1 on A3. Every number measured from the "
        "mesh; no hand-entered values.",
        "Inspection drawing: no tolerance, GD&T, thread, finish or material is stated or "
        "implied. Advisory, not a release.",
    ]
    ny = yy + 3.5
    for note in notes:
        svg.text(TABLE_X, ny, note, size=2.1, color="#333")
        ny += 3.2

    # title block
    svg.rect(TITLE_X, BAND_TOP, TITLE_W, TITLE_H, width=0.5)
    svg.text(TITLE_X + 2, BAND_TOP + 6.5, "plate-3window", size=4.2, bold=True)
    svg.line(TITLE_X, BAND_TOP + 9, TITLE_X + TITLE_W, BAND_TOP + 9, width=0.3)
    rows = [("SOURCE", "plate.stl"), ("PART", "plate"), ("BBOX mm", "60 x 40 x 9"),
            ("VOLUME", "{} mm3".format(_num(VOLUME_MM3))), ("SCALE / SHEET", "2:1  A3 landscape"),
            ("OPENSCAD", "2026.09.07 (fixture)"), ("MATERIAL", "-")]
    ty = BAND_TOP + 9
    step = min((TITLE_H - 9) / len(rows), 5.5)
    for key, value in rows:
        svg.text(TITLE_X + 2, ty + step - 1.2, key, size=2.2, color="#555")
        svg.text(TITLE_X + 24, ty + step - 1.2, value, size=2.4, mono=True)
        ty += step
    return svg.render()


def callout(group: Dict[str, Any]) -> str:
    qty = "{}x ".format(group["qty"]) if group["qty"] > 1 else ""
    return "{}  {}{} x {} THRU".format(group["id"], qty, _num(group["size"][0]),
                                       _num(group["size"][1]))


# --------------------------------------------------------------------------------------
# the mesh (binary STL: the house rule, and 5x smaller than ASCII)
# --------------------------------------------------------------------------------------


def plate_triangles() -> List[Tuple[Tuple[float, float, float], ...]]:
    """The plate as a closed triangle soup: 56 facets, horizontal faces only at z=0 and z=9."""
    width, depth, height = PLATE
    bands = [(0.0, 17.0, [(0.0, width)]),
             (17.0, 23.0, [(0.0, 7.0), (13.0, 25.0), (35.0, 42.0), (48.0, width)]),
             (23.0, depth, [(0.0, width)])]
    tris: List[Tuple[Tuple[float, float, float], ...]] = []
    for y0, y1, spans in bands:
        for x0, x1 in spans:
            tris.append(((x0, y0, height), (x1, y0, height), (x1, y1, height)))
            tris.append(((x0, y0, height), (x1, y1, height), (x0, y1, height)))
            tris.append(((x0, y0, 0.0), (x1, y1, 0.0), (x1, y0, 0.0)))
            tris.append(((x0, y0, 0.0), (x0, y1, 0.0), (x1, y1, 0.0)))
    walls = [((0.0, 0.0), (width, 0.0)), ((width, 0.0), (width, depth)),
             ((width, depth), (0.0, depth)), ((0.0, depth), (0.0, 0.0))]
    for x0, x1, y0, y1 in WINDOWS:     # hole walls, wound the other way round
        walls.extend([((x0, y0), (x0, y1)), ((x0, y1), (x1, y1)),
                      ((x1, y1), (x1, y0)), ((x1, y0), (x0, y0))])
    for (ax, ay), (bx, by) in walls:
        tris.append(((ax, ay, 0.0), (bx, by, 0.0), (bx, by, height)))
        tris.append(((ax, ay, 0.0), (bx, by, height), (ax, ay, height)))
    return tris


def write_stl(path: Path) -> None:
    tris = plate_triangles()
    blob = [b"PhiMech drawing fixture: 60x40x9 plate, three through-windows".ljust(80, b" "),
            struct.pack("<I", len(tris))]
    for a, b, c in tris:
        ux, uy, uz = (b[i] - a[i] for i in range(3))
        vx, vy, vz = (c[i] - a[i] for i in range(3))
        normal = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        length = sum(n * n for n in normal) ** 0.5 or 1.0
        blob.append(struct.pack("<3f", *(n / length for n in normal)))
        for vertex in (a, b, c):
            blob.append(struct.pack("<3f", *vertex))
        blob.append(struct.pack("<H", 0))
    path.write_bytes(b"".join(blob))


# --------------------------------------------------------------------------------------
# the planted defects
# --------------------------------------------------------------------------------------


def _replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError("mutation anchor is not unique: {!r}".format(old[:60]))
    return text.replace(old, new)


def _text_element(svg: str, needle: str) -> str:
    """The whole ``<text ...>needle</text>`` element, so a mutation can move or delete it."""
    end = svg.index("</text>", svg.index(">" + needle + "</text>")) + len("</text>")
    start = svg.rindex("<text ", 0, end)
    return svg[start:end]


def defect_text_collision(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Print the second callout on top of the first."""
    element = _text_element(svg, callout(GROUPS[1]))
    return _replace(svg, element, element.replace('y="107.500"', 'y="102.300"')), drawing


def defect_text_clipped(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Push the second callout off the right edge of the sheet."""
    element = _text_element(svg, callout(GROUPS[1]))
    return _replace(svg, element, element.replace('x="172.800"', 'x="402.000"')), drawing


def defect_label_over_view(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Print H1's leader label inside the top view, across the plate it describes."""
    element = _text_element(svg, callout(GROUPS[0]))
    moved = element.replace('x="172.800"', 'x="100.000"').replace('y="102.000"', 'y="170.000"')
    return _replace(svg, element, moved), drawing


def defect_label_struck(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Run H1's leader shoulder on through its own label: a struck-through callout."""
    old = 'x1="170.500" y1="102.000" x2="172.000" y2="102.000"'
    return _replace(svg, old, 'x1="170.500" y1="102.000" x2="190.000" y2="102.000"'), drawing


def defect_missing_callout(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Drop H2's leader label: the group is in the table but never called out on the view."""
    return _replace(svg, _text_element(svg, callout(GROUPS[1])) + "\n", ""), drawing


def defect_unlocated_member(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Drop the pitch dimension: H2's second window cannot be reached from the datum.

    Group-level coverage still passes - H2 has a callout and a position dimension - which is
    exactly why coverage has to be decided per member.
    """
    element = _text_element(svg, "35")
    return _replace(svg, element + "\n", ""), drawing


def defect_untraceable_dimension(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Retype the front view's overall width as a value nothing measured."""
    element = _text_element(svg, "60")            # the front view's, not the top view's
    second = svg.rindex(element)
    return svg[:second] + element.replace(">60<", ">77.77<") + svg[second + len(element):], drawing


def defect_phantom_level(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Print a z level that is a tessellation coincidence, not a face.

    The level is in ``levels_z``, so it is a "measured" value and traces perfectly; only a
    face-area check can see that nothing at z=4.33 exists.
    """
    drawing = dict(drawing)
    drawing["levels_z"] = [0.0, 4.33, 9.0]
    return build_sheet(levels=(0.0, 4.33, 9.0)), drawing


def defect_dirty_translation(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Same sheet, but the STEP that fed it reported an error the exporter still completed."""
    return svg, drawing


def defect_missing_counterbore(svg: str, drawing: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Same sheet, but the feature program declared counterbores the solid does not have."""
    return svg, drawing


#: name -> (mutation, what it is, the one check it must fail, extra files it needs)
DEFECTS: List[Tuple[str, Any, str, str, Dict[str, Any]]] = [
    ("text-collision", defect_text_collision,
     "H2's leader label is printed on top of H1's", "text_no_collision", {}),
    ("text-clipped", defect_text_clipped,
     "H2's leader label runs off the right edge of the sheet", "text_not_clipped", {}),
    ("label-over-view", defect_label_over_view,
     "H1's leader label is printed inside the top view, over the plate", "label_view_overlap",
     {}),
    ("label-struck", defect_label_struck,
     "H1's leader runs on through its own label", "label_line_overlap", {}),
    ("missing-callout", defect_missing_callout,
     "H2 has no leader callout on the view", "feature_coverage", {}),
    ("unlocated-member", defect_unlocated_member,
     "the pitch dimension is gone, so H2's second window cannot be located from the datum",
     "feature_coverage", {}),
    ("untraceable-dimension", defect_untraceable_dimension,
     "the front view's overall width is printed as 77.77, which nothing measured",
     "dimension_traceability", {}),
    ("phantom-level", defect_phantom_level,
     "z=4.33 is printed as a level but no horizontal face exists there", "levels_are_faces", {}),
    ("dirty-translation", defect_dirty_translation,
     "the source manifest reports a translation error the exporter completed anyway",
     "source_translation_clean", {"source.json": "dirty"}),
    ("missing-counterbore", defect_missing_counterbore,
     "the feature program declares two counterbores that are absent from the geometry",
     "intent_vs_measured", {"intent.json": "counterbored"}),
]


# --------------------------------------------------------------------------------------
# side files
# --------------------------------------------------------------------------------------

CLEAN_MANIFEST: Dict[str, Any] = {
    "schema_version": 1,
    "source": {"kind": "step", "native_path": None, "native_sha256": None,
               "configuration": "fixture", "exported_by": None, "exported_at": None,
               "exporter": {"name": "offline_fixture", "version": None}},
    "step": {"path": "model.step", "sha256": "0" * 64, "protocol": "AP214", "units": "mm"},
    "component": {"name": "plate", "revision": None, "material": None, "mass_g": None},
    "translation": {"ok": True, "diagnostics": []},
    "errors": [], "warnings": [],
}

DIRTY_MANIFEST: Dict[str, Any] = json.loads(json.dumps(CLEAN_MANIFEST))
DIRTY_MANIFEST["translation"] = {
    "ok": False,
    "diagnostics": [
        {"severity": "warning", "message": "1 face was rebuilt during translation"},
        {"severity": "error", "message": "entity #1842 could not be translated; the export "
                                         "completed with a gap in the solid"},
    ],
}

MATCHING_INTENT: Dict[str, Any] = {
    "version": 1, "units": "mm",
    "ops": [
        {"op": "box", "size": [60, 40, 9]},
        {"op": "hole", "at": [[10, 20], [45, 20]], "diameter": 6},
        {"op": "hole", "at": [[30, 20]], "diameter": 10},
    ],
}

COUNTERBORED_INTENT: Dict[str, Any] = json.loads(json.dumps(MATCHING_INTENT))
COUNTERBORED_INTENT["ops"][1]["counterbore"] = {"diameter": 9.5, "depth": 4}


def drawing_record(svg_text: str) -> Dict[str, Any]:
    """The ``drawing.json`` beside a sheet: the measured result plus the schema-1 envelope."""
    return {
        "schema_version": 1,
        "intent": "inspection",
        "path": "drawing.svg",
        "png": None,
        "png_tool": None,
        "scale": SCALE,
        "sheet_mm": list(SHEET),
        "bbox_mm": {"min": [0.0, 0.0, 0.0], "max": list(PLATE)},
        "volume_mm3": VOLUME_MM3,
        "levels_z": [0.0, 9.0],
        "groups": json.loads(json.dumps(GROUPS)),
        "views": json.loads(json.dumps(VIEWS)),
        "render_s": 0.0,
        "source": "plate.stl",
        "part": "plate",
        "backend": "openscad",
        "openscad_version": "2026.09.07 (fixture)",
        "bytes": len(svg_text.encode("utf-8")),
        "units": "mm; positions and depths are model coordinates; dimensions on the sheet run "
                 "from the part's lower-left bbox corner (datum) in each view",
        "hashes": {"source_sha256": None, "step_sha256": None,
                   "svg_sha256": hashlib.sha256(svg_text.encode("utf-8")).hexdigest(),
                   "png_sha256": None},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    write_stl(HERE / "plate.stl")
    _write_json(HERE / "source.json", CLEAN_MANIFEST)
    _write_json(HERE / "intent.json", MATCHING_INTENT)

    correct_svg = build_sheet()
    written = []
    cases: List[Tuple[str, str, str, Dict[str, Any], Any]] = [
        ("correct", "the sheet as it should be: every check green", "", {}, None)]
    cases.extend((name, why, fails, extra, mutate)
                 for name, mutate, why, fails, extra in DEFECTS)
    for name, why, fails, extra, mutate in cases:
        directory = HERE / name
        directory.mkdir(exist_ok=True)
        svg_text, record = correct_svg, drawing_record(correct_svg)
        if mutate is not None:
            svg_text, record = mutate(correct_svg, record)
            record = dict(record)
            record["bytes"] = len(svg_text.encode("utf-8"))
            record["hashes"] = dict(record["hashes"])
            record["hashes"]["svg_sha256"] = hashlib.sha256(
                svg_text.encode("utf-8")).hexdigest()
        (directory / "drawing.svg").write_text(svg_text, encoding="utf-8")
        _write_json(directory / "drawing.json", record)
        expect: Dict[str, Any] = {
            "name": name,
            "defect": why,
            "fails": [fails] if fails else [],
            "svg": "drawing.svg",
            "drawing": "drawing.json",
            "stl": "../plate.stl",
            "source_manifest": "../source.json",
            "declared_intent": "../intent.json",
        }
        for filename, flavour in extra.items():
            payload = {"dirty": DIRTY_MANIFEST, "counterbored": COUNTERBORED_INTENT}[flavour]
            _write_json(directory / filename, payload)
            expect["source_manifest" if filename == "source.json" else
                   "declared_intent"] = filename
        _write_json(directory / "expect.json", expect)
        written.append(name)
    print("wrote {} fixtures: {}".format(len(written), ", ".join(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
