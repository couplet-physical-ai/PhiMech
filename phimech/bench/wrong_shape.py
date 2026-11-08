"""The wrong-shape benchmark: right dimensions, passing gates, wrong geometry.

Why this exists
---------------
The generation benchmark (``suite.json``) scores designs with numeric gates, and numeric gates
have a blind spot: a part can be manifold, have a positive volume, fit its bbox to the tenth of a
millimetre - and be a box where a cylinder was asked for. This suite measures that blind spot,
and measures how much of it the preview sheet closes.

Every case pairs a **reference** (what the brief asked for) with a **deliberately wrong**
candidate that has the *same bounding box*, is manifold, and has a positive volume. Each candidate
is graded twice with the ordinary grader (``grade.grade_case``):

* ``numeric``: ``builds`` + ``manifold`` + ``part_volume_positive`` + ``outer_size_mm`` - the
  checks an enclosure-agnostic gate set can run on a bare part. Expected to *pass* the wrong
  shape (that is the blind spot).
* ``full``: the same plus ``preview_min_iou`` - the silhouette IoU of the candidate's 7-view
  sheet against the reference sheet in ``fixtures/bench/preview``. Expected to *fail* it.

The recorded numbers live in ``fixtures/bench/preview/results.json``; the Phase A gate is that
numeric gates catch at most 1 in 6 and the preview proxy catches at least 5 in 6.

Cases are Python, not JSON, on purpose: the stand's geometry needs trigonometry to keep the
bbox identical at two angles, and a reader should see ``cyl(30, 40)`` next to ``box(30, 30, 40)``
rather than two pages of nested dicts. ``write_fixtures`` exports the wrong candidates as CSG AST
JSON (the recorded "submission", in the same ``<case-id>.json`` shape as ``sol-baseline``) and
renders the reference sheets.

One case, ``tray-open-front``, is kept although the proxy **cannot** catch it: an open-top tray
and an open-front tray have identical silhouettes from every direction. It is the concrete,
numbered statement of the proxy's limitation, and the reason the sheet itself - not the number -
is what an agent must look at.

stdlib only, like the rest of the bench; the geometry kernel is OpenSCAD as a subprocess.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from phimech.bench.grade import grade_case
from phimech.csg import ops
from phimech.csg.ast import Cylinder, Node, Rotate, Translate, Union as CsgUnion

SUITE_NAME = "phimech-wrong-shape-v1"

#: Tile size of the committed reference sheets. Small on purpose: eight 640x320 PNGs.
FIXTURE_TILE_PX = 160

#: The Phase A gate, as ratios so a suite of any size is judged the same way.
NUMERIC_CATCH_MAX = 1.0 / 6.0
PREVIEW_CATCH_MIN = 5.0 / 6.0

#: Tolerance for the shared bbox. CSG-exact geometry, so half a millimetre is generous.
SIZE_TOLERANCE_MM = 0.5

_SEGMENTS = 64  # a multiple of 4 puts vertices on the axes, so a cylinder's bbox is exact


def _repository() -> Path:
    configured = os.environ.get("PHIMECH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def fixtures_dir() -> Path:
    return _repository() / "fixtures" / "bench" / "preview"


def wrong_dir() -> Path:
    """The recorded submission of deliberately wrong candidates, one ``<case-id>.json`` each."""
    return fixtures_dir() / "wrong"


# --------------------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WrongShapeCase:
    id: str
    prompt: str
    why: str
    size_mm: Tuple[float, float, float]
    reference: Node
    wrong: Node
    wrong_summary: str
    silhouette_visible: bool = True

    def expect(self, *, preview_threshold: Optional[float]) -> Dict[str, Any]:
        expect: Dict[str, Any] = {
            "builds": True,
            "gates_pass": ["manifold", "part_volume_positive"],
            "outer_size_mm": {"value": list(self.size_mm), "tolerance": SIZE_TOLERANCE_MM},
        }
        if preview_threshold is not None:
            expect["preview_min_iou"] = float(preview_threshold)
        return expect

    def to_dict(self, preview_threshold: float) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "why": self.why,
            "expect": self.expect(preview_threshold=preview_threshold),
            "size_mm": list(self.size_mm),
            "wrong": self.wrong_summary,
            "silhouette_visible": self.silhouette_visible,
        }


def _through_hole(d: float, length: float, axis: str, at: Tuple[float, float, float]) -> Node:
    """A cylinder that starts before one face and ends past the other - no co-planar skins."""
    return ops.cyl(d, length, at=at, axis=axis, segments=_SEGMENTS)


def _stand(angle_deg: float, *, width: float = 60.0, depth: float = 50.0, height: float = 40.0,
           plate: float = 4.0) -> Node:
    """A base plate plus a back rest leaning ``angle_deg`` back from vertical, height fixed.

    The rest's length is solved so its top lands at ``height`` for *any* angle, and it is sunk
    1 mm into the plate so the union is a clean overlap rather than an edge contact. The bbox is
    therefore ``width x depth x height`` at every angle - which is the whole point.
    """
    theta = math.radians(angle_deg)
    sink = 1.0
    length = (height - (plate - sink) - plate * math.sin(theta)) / math.cos(theta)
    rest_depth = length * math.sin(theta) + plate * math.cos(theta)
    rest = Rotate((-angle_deg, 0.0, 0.0), [ops.box(width, plate, length)])
    rest = Translate((0.0, depth - rest_depth, plate - sink + plate * math.sin(theta)), [rest])
    return CsgUnion([ops.box(width, depth, plate), rest])


def _bracket(holes_in: str) -> Node:
    horizontal = ops.box(40.0, 40.0, 4.0)
    vertical = ops.box(40.0, 4.0, 30.0, at=(0.0, 36.0, 0.0))
    body = CsgUnion([horizontal, vertical])
    if holes_in == "vertical":
        cutters = [_through_hole(10.0, 6.0, "y", (x, 35.0, 17.0)) for x in (12.0, 28.0)]
    else:
        cutters = [_through_hole(10.0, 6.0, "z", (x, 18.0, -1.0)) for x in (12.0, 28.0)]
    return ops.diff_all(body, cutters)


def _lid(with_bezel: bool) -> Node:
    plate = ops.box(60.0, 40.0, 3.0)
    shroud = ops.box(20.0, 10.0, 8.0, at=(5.0, 15.0, 3.0))
    parts = [plate, shroud]
    if with_bezel:
        parts.append(ops.cyl(14.0, 8.0, at=(45.0, 20.0, 3.0), segments=_SEGMENTS))
    return CsgUnion(parts)


def _tray(open_side: str) -> Node:
    shell = ops.box(50.0, 40.0, 20.0)
    if open_side == "top":
        cavity = ops.box(46.0, 36.0, 19.0, at=(2.0, 2.0, 2.0))  # 1 mm past the rim
    else:  # open front: the cavity runs out through the -Y wall, the top stays closed
        cavity = ops.box(46.0, 37.0, 16.0, at=(2.0, -1.0, 2.0))
    return ops.diff_all(shell, [cavity])


CASES: Tuple[WrongShapeCase, ...] = (
    WrongShapeCase(
        id="spacer-box-for-cylinder",
        prompt="A solid cylindrical spacer, 30 mm diameter and 40 mm tall.",
        why="The purest form of the blind spot: a box has the same bbox, is manifold and has a "
            "positive volume. Only the top view (a square where a disc belongs) tells them apart.",
        size_mm=(30.0, 30.0, 40.0),
        reference=ops.cyl(30.0, 40.0, segments=_SEGMENTS),
        wrong=ops.box(30.0, 30.0, 40.0),
        wrong_summary="a 30 x 30 x 40 box instead of a cylinder",
    ),
    WrongShapeCase(
        id="funnel-straight-for-tapered",
        prompt="A conical funnel adapter 40 mm tall, 30 mm diameter at the base tapering to "
               "16 mm at the top.",
        why="A taper is a shape, not a size: the straight cylinder shares the bbox. Every side "
            "view shows a rectangle where a trapezoid belongs.",
        size_mm=(30.0, 30.0, 40.0),
        reference=Cylinder(h=40.0, r=15.0, r2=8.0, segments=_SEGMENTS),
        wrong=ops.cyl(30.0, 40.0, segments=_SEGMENTS),
        wrong_summary="a straight Ø30 cylinder instead of a Ø30 -> Ø16 cone",
    ),
    WrongShapeCase(
        id="lid-no-cutout",
        prompt="A flat lid 60 x 40 x 3 mm with a 30 x 16 mm rectangular display window cut "
               "through, centred.",
        why="The most common real feature request is a cutout, and forgetting it changes no "
            "outer dimension. The window reads as background in the top and bottom views.",
        size_mm=(60.0, 40.0, 3.0),
        reference=ops.diff_all(
            ops.box(60.0, 40.0, 3.0), [ops.box(30.0, 16.0, 5.0, at=(15.0, 12.0, -1.0))]
        ),
        wrong=ops.box(60.0, 40.0, 3.0),
        wrong_summary="a plain plate: the display window is missing",
    ),
    WrongShapeCase(
        id="stand-wrong-angle",
        prompt="A phone stand 60 mm wide: a 50 mm deep, 4 mm thick base plate and a 4 mm thick "
               "back rest leaning 15 degrees back from vertical, 40 mm tall overall.",
        why="An angle is the classic thing a model gets wrong while every dimension is right. "
            "The rest is solved to the same 60 x 50 x 40 bbox at both angles.",
        size_mm=(60.0, 50.0, 40.0),
        reference=_stand(15.0),
        wrong=_stand(45.0),
        wrong_summary="the back rest leans 45 degrees instead of 15",
    ),
    WrongShapeCase(
        id="bracket-holes-wrong-leg",
        prompt="An L-bracket 40 mm wide: a 40 x 40 mm horizontal leg and a 30 mm tall vertical "
               "leg along its back edge, both 4 mm thick, with two 10 mm mounting holes through "
               "the vertical leg.",
        why="Features on the wrong face keep the bbox and nearly the volume. The holes show as "
            "background in the front view of the right part and in the top view of the wrong one.",
        size_mm=(40.0, 40.0, 30.0),
        reference=_bracket("vertical"),
        wrong=_bracket("horizontal"),
        wrong_summary="the two holes go through the horizontal leg instead of the vertical one",
    ),
    WrongShapeCase(
        id="lid-bezel-missing",
        prompt="A 60 x 40 x 3 mm lid with a 20 x 10 x 8 mm connector shroud standing on top "
               "near one end and a 14 mm diameter, 8 mm tall round LED bezel near the other, "
               "both centred across the width.",
        why="A missing boss whose neighbour sets the height: bbox unchanged, volume off by a "
            "few percent that no gate targets. The bezel is a separate lump in the front view.",
        size_mm=(60.0, 40.0, 11.0),
        reference=_lid(with_bezel=True),
        wrong=_lid(with_bezel=False),
        wrong_summary="the LED bezel boss is missing",
    ),
    WrongShapeCase(
        id="plate-tab-mirrored",
        prompt="A 50 x 30 x 4 mm mounting plate with a 15 x 10 mm tab extending from the middle "
               "of its right-hand (+X) edge.",
        why="Handedness. A mirrored part has identical size and volume and fits the same bed; "
            "it just does not fit the assembly. The tab sits on the wrong side in the top view.",
        size_mm=(65.0, 30.0, 4.0),
        reference=CsgUnion([ops.box(50.0, 30.0, 4.0), ops.box(15.0, 10.0, 4.0, at=(50.0, 10.0, 0.0))]),
        wrong=CsgUnion([ops.box(50.0, 30.0, 4.0), ops.box(15.0, 10.0, 4.0, at=(-15.0, 10.0, 0.0))]),
        wrong_summary="the tab is on the -X edge: a mirror image",
    ),
    WrongShapeCase(
        id="tray-open-front",
        prompt="An open-top tray 50 x 40 x 20 mm with 2 mm walls and a 2 mm floor.",
        why="Kept as the documented limit of the proxy. An open-front tray with a closed top has "
            "the same silhouette from every direction - the opening never reaches the outline - "
            "so IoU is 1.0 for a wrong part. Only looking at the sheet catches this one.",
        size_mm=(50.0, 40.0, 20.0),
        reference=_tray("top"),
        wrong=_tray("front"),
        wrong_summary="open on the front face with a closed top, instead of open on top",
        silhouette_visible=False,
    ),
)


def case_by_id(case_id: str) -> Optional[WrongShapeCase]:
    return next((case for case in CASES if case.id == case_id), None)


def suite(preview_threshold: Optional[float] = None) -> Dict[str, Any]:
    """The suite in the same shape as ``suite.json``, so the grader and its tests can read it."""
    from phimech.synth.preview import DEFAULT_IOU_THRESHOLD

    threshold = DEFAULT_IOU_THRESHOLD if preview_threshold is None else float(preview_threshold)
    return {
        "suite": SUITE_NAME,
        "description": "Deliberately wrong shapes with the right dimensions. Each case is "
                       "graded with numeric gates alone and with the preview-sheet silhouette "
                       "IoU added; the gap between the two is what the sheet buys. Dimensions "
                       "are mm.",
        "iou_threshold": threshold,
        "cases": [case.to_dict(threshold) for case in CASES],
    }


# --------------------------------------------------------------------------------------
# building and measuring one candidate
# --------------------------------------------------------------------------------------


def _mesh_report(stl: Path) -> Dict[str, Any]:
    """A ``synth build``-shaped report for a bare part, from the check layer's mesh metrics.

    ``manifold`` and ``part_volume_positive`` are recomputed here with the same thresholds the
    real gates use (one watertight shell; volume > 0), so the numbers match a pipeline build.
    """
    from phimech.check import mesh

    triangles = mesh.read_stl(str(stl))
    (lo, hi) = mesh.bbox(triangles)
    volume = mesh.volume(triangles)
    watertight, open_edges = mesh.is_watertight(triangles)
    shells = mesh.count_shells(triangles)
    manifold = watertight and shells == 1
    size = [hi[i] - lo[i] for i in range(3)]
    return {
        "ok": True,
        "errors": [],
        "gates": {
            "results": [
                {
                    "name": "manifold",
                    "passed": manifold,
                    "severity": "error",
                    "value": 1 if manifold else 0,
                    "threshold": 1,
                    "message": "shells={} open_edges={} threshold=1".format(shells, open_edges),
                },
                {
                    "name": "part_volume_positive",
                    "passed": volume > 0.0,
                    "severity": "error",
                    "value": volume,
                    "threshold": 0.0,
                    "message": "volume_mm3={:.6g} threshold>0".format(volume),
                },
            ]
        },
        "measurements": {
            "part": {
                "bbox": {"min": list(lo), "max": list(hi), "size": size},
                "volume_mm3": volume,
                "area_mm2": mesh.area(triangles),
                "solids": shells,
                "extra": {"stl": str(stl)},
            }
        },
    }


def _load_candidate(path: Path) -> Node:
    from phimech.csg.ast import from_dict

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "kind" not in document:
        raise ValueError("{} is not a CSG AST (no `kind`)".format(path))
    return from_dict(document)


def _render_stl(node: Node, stl: Path) -> float:
    from phimech.backends.openscad import OpenSCADBackend

    return float(OpenSCADBackend().render(node, stl, "stl").seconds)


def grade_candidate(
    case: WrongShapeCase,
    candidate: Union[Node, Path],
    out_dir: Path,
    *,
    reference_sheet: Path,
    tile_px: int = FIXTURE_TILE_PX,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Build, measure and sheet one candidate, then grade it numeric-only and with the preview."""
    from phimech.synth.preview import DEFAULT_IOU_THRESHOLD, compare_sheets, render_sheet

    iou_threshold = DEFAULT_IOU_THRESHOLD if threshold is None else float(threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    node = candidate if isinstance(candidate, Node) else _load_candidate(candidate)
    started = time.monotonic()
    stl = out_dir / "{}.stl".format(case.id)
    seconds: Dict[str, float] = {"stl": round(_render_stl(node, stl), 3)}
    report = _mesh_report(stl)
    sheet = render_sheet(stl, out_dir / "{}.png".format(case.id), tile_px=tile_px)
    seconds["sheet"] = sheet.render_s
    comparison = compare_sheets(reference_sheet, sheet.path, threshold=iou_threshold)
    comparison["sheet"] = sheet.path
    comparison["reference"] = str(reference_sheet)
    report["preview"] = comparison

    numeric = grade_case({"id": case.id, "expect": case.expect(preview_threshold=None)}, report)
    full = grade_case({"id": case.id, "expect": case.expect(preview_threshold=iou_threshold)},
                      report)
    return {
        "id": case.id,
        "wrong": case.wrong_summary,
        "silhouette_visible": case.silhouette_visible,
        "numeric_passed": bool(numeric["passed"]),
        "full_passed": bool(full["passed"]),
        "caught_by_numeric": not numeric["passed"],
        "caught_by_preview": not comparison["same_shape"],
        "min_iou": comparison["min_iou"],
        "worst_view": comparison["worst_view"],
        "per_view": comparison["per_view"],
        "measured_size_mm": [round(v, 3) for v in report["measurements"]["part"]["bbox"]["size"]],
        "volume_mm3": round(report["measurements"]["part"]["volume_mm3"], 1),
        "numeric_checks": numeric["checks"],
        "full_checks": full["checks"],
        "seconds": dict(seconds, total=round(time.monotonic() - started, 3)),
        "sheet": sheet.path,
    }


# --------------------------------------------------------------------------------------
# fixtures: reference sheets + the recorded wrong submission
# --------------------------------------------------------------------------------------


def write_fixtures(
    directory: Optional[Path] = None, *, tile_px: int = FIXTURE_TILE_PX
) -> Dict[str, Any]:
    """Render every reference sheet and write every wrong candidate as ``wrong/<id>.json``."""
    from phimech.synth.preview import render_sheet

    target = Path(directory) if directory else fixtures_dir()
    (target / "wrong").mkdir(parents=True, exist_ok=True)
    written: Dict[str, Any] = {"directory": str(target), "tile_px": tile_px, "cases": {}}
    for case in CASES:
        started = time.monotonic()
        sheet = render_sheet(case.reference, target / "{}.png".format(case.id), tile_px=tile_px)
        wrong_path = target / "wrong" / "{}.json".format(case.id)
        wrong_path.write_text(
            json.dumps(case.wrong.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written["cases"][case.id] = {
            "sheet": sheet.path,
            "wrong": str(wrong_path),
            "bbox_mm": sheet.bbox_mm,
            "volume_mm3": round(sheet.volume_mm3, 1),
            "seconds": round(time.monotonic() - started, 3),
        }
    return written


# --------------------------------------------------------------------------------------
# the run, the gate, and the record
# --------------------------------------------------------------------------------------


def _controls(out_dir: Path, *, tile_px: int, threshold: float,
              references: Path) -> Dict[str, Any]:
    """Two same-shape controls, so the threshold is shown to be tolerant as well as strict.

    A fresh render of a reference must match its committed sheet (IoU 1.0 on the same machine),
    and a dimension tweak of the kind a gate would argue about (+2 mm on a 40 mm height) must
    still read as the same shape - otherwise the proxy would flag every tolerance discussion.
    """
    from phimech.synth.preview import compare_sheets, render_sheet

    case = case_by_id("spacer-box-for-cylinder")
    assert case is not None
    controls_dir = out_dir / "_controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    rerender = render_sheet(case.reference, controls_dir / "rerender.png", tile_px=tile_px)
    same = compare_sheets(references / "{}.png".format(case.id), rerender.path,
                          threshold=threshold)
    tweaked = render_sheet(ops.cyl(30.0, 42.0, segments=_SEGMENTS),
                           controls_dir / "tweaked.png", tile_px=tile_px)
    tweak = compare_sheets(references / "{}.png".format(case.id), tweaked.path,
                           threshold=threshold)
    return {
        "same_shape_rerender": {"min_iou": same["min_iou"], "same_shape": same["same_shape"]},
        "same_shape_height_plus_2mm": {"min_iou": tweak["min_iou"],
                                       "same_shape": tweak["same_shape"],
                                       "worst_view": tweak["worst_view"]},
    }


def run(
    candidates_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    tile_px: int = FIXTURE_TILE_PX,
    threshold: Optional[float] = None,
    references: Optional[Path] = None,
    controls: bool = True,
) -> Dict[str, Any]:
    """Grade a submission of ``<case-id>.json`` CSG ASTs both ways and judge the Phase A gate.

    Defaults to the recorded wrong submission, so ``phimech preview bench`` with no arguments
    reproduces ``results.json``. A case with no candidate is reported as missing rather than
    silently dropped.
    """
    from phimech.synth.preview import DEFAULT_IOU_THRESHOLD

    iou_threshold = DEFAULT_IOU_THRESHOLD if threshold is None else float(threshold)
    candidates = Path(candidates_dir) if candidates_dir else wrong_dir()
    reference_dir = Path(references) if references else fixtures_dir()
    build_root = Path(out_dir) if out_dir else _repository() / "build" / "preview-bench"
    build_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    missing: List[str] = []
    for case in CASES:
        candidate = candidates / "{}.json".format(case.id)
        reference_sheet = reference_dir / "{}.png".format(case.id)
        if not candidate.is_file():
            missing.append(case.id)
            continue
        if not reference_sheet.is_file():
            raise FileNotFoundError(
                "reference sheet missing: {} (run `phimech preview bench --write-fixtures`)"
                .format(reference_sheet)
            )
        results.append(grade_candidate(case, candidate, build_root, reference_sheet=reference_sheet,
                                       tile_px=tile_px, threshold=iou_threshold))

    graded = len(results)
    numeric_caught = sum(1 for r in results if r["caught_by_numeric"])
    preview_caught = sum(1 for r in results if r["caught_by_preview"])
    visible = [r for r in results if r["silhouette_visible"]]
    gate_ok = (
        graded > 0
        and not missing
        and numeric_caught <= NUMERIC_CATCH_MAX * graded
        and preview_caught >= PREVIEW_CATCH_MIN * graded
    )
    outcome: Dict[str, Any] = {
        "suite": SUITE_NAME,
        "candidates_dir": str(candidates),
        "references_dir": str(reference_dir),
        "tile_px": tile_px,
        "iou_threshold": iou_threshold,
        "openscad": _openscad_version(),
        "results": results,
        "missing": missing,
        "summary": {
            "cases": graded,
            "numeric_caught": numeric_caught,
            "preview_caught": preview_caught,
            "preview_caught_of_silhouette_visible": "{}/{}".format(
                sum(1 for r in visible if r["caught_by_preview"]), len(visible)),
            "either_caught": sum(
                1 for r in results if r["caught_by_numeric"] or r["caught_by_preview"]),
            "closest_wrong_shape_min_iou": max(
                (r["min_iou"] for r in results if r["caught_by_preview"]), default=None),
        },
        "gate": {
            "numeric_catch_max": "{}/{} (<= 1/6)".format(
                math.floor(NUMERIC_CATCH_MAX * graded + 1e-9), graded),
            "preview_catch_min": "{}/{} (>= 5/6)".format(
                math.ceil(PREVIEW_CATCH_MIN * graded - 1e-9), graded),
            "ok": gate_ok,
        },
    }
    if controls and graded:
        outcome["controls"] = _controls(build_root, tile_px=tile_px, threshold=iou_threshold,
                                        references=reference_dir)
    outcome["table"] = format_table(outcome)
    return outcome


def _openscad_version() -> Optional[str]:
    try:
        from phimech.backends.openscad import OpenSCADBackend

        return OpenSCADBackend().version()
    except Exception:
        return None


def format_table(outcome: Mapping[str, Any]) -> str:
    """One line per case: did the numbers notice, did the picture notice, and by how much."""
    lines = ["%-28s %-8s %-8s %8s  %-7s %s" % (
        "case", "numeric", "preview", "min IoU", "worst", "what is wrong")]
    for result in outcome.get("results", []):
        lines.append("%-28s %-8s %-8s %8.3f  %-7s %s" % (
            str(result["id"])[:28],
            "CAUGHT" if result["caught_by_numeric"] else "passed",
            "CAUGHT" if result["caught_by_preview"] else ("blind" if not result["silhouette_visible"]
                                                          else "MISSED"),
            float(result["min_iou"]),
            str(result["worst_view"]),
            str(result["wrong"])[:44],
        ))
    for case_id in outcome.get("missing", []):
        lines.append("%-28s %-8s %-8s %8s  %-7s %s" % (case_id, "-", "-", "-", "-",
                                                        "NO CANDIDATE"))
    summary = outcome.get("summary", {})
    gate = outcome.get("gate", {})
    lines.append("")
    lines.append("numeric gates caught %s/%s (gate: at most %s)   preview IoU < %.2f caught %s/%s "
                 "(gate: at least %s)   -> %s" % (
                     summary.get("numeric_caught"), summary.get("cases"),
                     gate.get("numeric_catch_max"), float(outcome.get("iou_threshold", 0.0)),
                     summary.get("preview_caught"), summary.get("cases"),
                     gate.get("preview_catch_min"), "GATE OK" if gate.get("ok") else "GATE FAILED"))
    controls = outcome.get("controls")
    if controls:
        lines.append("controls: same shape re-rendered IoU %.3f; height +2 mm IoU %.3f (%s)" % (
            controls["same_shape_rerender"]["min_iou"],
            controls["same_shape_height_plus_2mm"]["min_iou"],
            "same shape" if controls["same_shape_height_plus_2mm"]["same_shape"]
            else "FLAGGED - threshold too tight"))
    return "\n".join(lines)


def record(outcome: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    """Write ``results.json``: the evidence for the Phase A gate, minus machine-local paths."""
    target = Path(path) if path else fixtures_dir() / "results.json"
    keep = {
        "suite": outcome["suite"],
        "recorded": time.strftime("%Y-%m-%d"),
        "openscad": outcome.get("openscad"),
        "tile_px": outcome["tile_px"],
        "iou_threshold": outcome["iou_threshold"],
        "gate": outcome["gate"],
        "summary": outcome["summary"],
        "controls": outcome.get("controls"),
        "cases": [
            {
                "id": r["id"],
                "wrong": r["wrong"],
                "silhouette_visible": r["silhouette_visible"],
                "caught_by_numeric": r["caught_by_numeric"],
                "caught_by_preview": r["caught_by_preview"],
                "min_iou": r["min_iou"],
                "worst_view": r["worst_view"],
                "per_view": r["per_view"],
                "measured_size_mm": r["measured_size_mm"],
                "volume_mm3": r["volume_mm3"],
                "numeric_checks": [c["name"] + ("" if c["passed"] else " FAIL")
                                   for c in r["numeric_checks"]],
                "seconds": r["seconds"],
            }
            for r in outcome["results"]
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(keep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------------------
# CLI hook: `phimech preview bench`
# --------------------------------------------------------------------------------------


def _handle_bench(args: argparse.Namespace) -> int:
    if args.write_fixtures:
        written = write_fixtures(Path(args.fixtures) if args.fixtures else None,
                                 tile_px=args.tile_px)
        if not args.json:
            for case_id, entry in written["cases"].items():
                print("wrote {} and {} ({:.1f}s)".format(
                    entry["sheet"], entry["wrong"], entry["seconds"]))
    outcome = run(
        Path(args.candidates) if args.candidates else None,
        Path(args.out) if args.out else None,
        tile_px=args.tile_px,
        threshold=args.threshold,
        references=Path(args.fixtures) if args.fixtures else None,
    )
    if args.record:
        where = record(outcome, Path(args.record) if isinstance(args.record, str) else None)
        if not args.json:
            print("recorded {}".format(where))
    if args.json:
        print(json.dumps({k: v for k, v in outcome.items() if k != "table"}, indent=2,
                         sort_keys=True))
    else:
        print(outcome["table"])
    return 0 if outcome["gate"]["ok"] else 1


def register_cli(commands: Any) -> None:
    """Add ``bench`` under the ``preview`` command (called by ``preview.register_cli``)."""
    from phimech.synth.preview import DEFAULT_IOU_THRESHOLD

    bench = commands.add_parser(
        "bench",
        help="grade the wrong-shape suite with numeric gates alone and with the preview IoU",
    )
    bench.add_argument("candidates", nargs="?", default=None,
                       help="directory of <case-id>.json CSG ASTs (default: the recorded wrong "
                            "submission in fixtures/bench/preview/wrong)")
    bench.add_argument("-o", "--out", default=None, help="build directory")
    bench.add_argument("--tile-px", type=int, default=FIXTURE_TILE_PX, dest="tile_px",
                       help="tile size; must match the reference sheets (default %d)"
                       % FIXTURE_TILE_PX)
    bench.add_argument("--threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
                       help="min IoU to call a shape the same (default %.2f)" % DEFAULT_IOU_THRESHOLD)
    bench.add_argument("--fixtures", default=None,
                       help="reference sheet directory (default fixtures/bench/preview)")
    bench.add_argument("--write-fixtures", action="store_true", dest="write_fixtures",
                       help="re-render the reference sheets and rewrite wrong/<id>.json first")
    bench.add_argument("--record", nargs="?", const=True, default=False,
                       help="write results.json (optionally to this path)")
    bench.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    bench.set_defaults(handler=_handle_bench)


__all__ = [
    "CASES",
    "FIXTURE_TILE_PX",
    "NUMERIC_CATCH_MAX",
    "PREVIEW_CATCH_MIN",
    "SUITE_NAME",
    "WrongShapeCase",
    "case_by_id",
    "fixtures_dir",
    "format_table",
    "grade_candidate",
    "record",
    "register_cli",
    "run",
    "suite",
    "wrong_dir",
    "write_fixtures",
]
