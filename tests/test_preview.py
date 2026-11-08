"""Preview sheet, silhouette comparison, wrong-shape bench and the preview MCP tools.

Runs directly under the system python (``bin/phimech check`` does exactly that) and under
pytest. Tests that need OpenSCAD skip cleanly when it is absent; everything about the PNG codec,
the font, the sheet layout, the IoU arithmetic, the grader check and the *recorded* benchmark
evidence runs without it.
"""

from __future__ import annotations

import argparse
import ast as _ast
import json
import os as _os
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by the system-python runner
    pytest = None  # type: ignore[assignment]

from phimech.bench import grade
from phimech.bench import wrong_shape
from phimech.csg import ops
from phimech.csg.ast import from_dict
from phimech.csg.validate import validate as validate_ast
from phimech.mcp import tools_preview
from phimech.synth import preview

REPO = Path(__file__).resolve().parents[1]
FIXTURE_BASE_STL = REPO / "fixtures" / "viewer" / "viewer-fixture-base.stl"
FIXTURE_LID_STL = REPO / "fixtures" / "viewer" / "viewer-fixture-lid.stl"
PREVIEW_FIXTURES = REPO / "fixtures" / "bench" / "preview"

BG = (248, 248, 248)
FG = (52, 84, 136)
TILE = 96


class _Skipped(Exception):
    pass


def _skip(message: str) -> None:
    # pytest.skip raises a BaseException that would crash the direct runner; PYTEST_CURRENT_TEST
    # tells "pytest is driving" from "pytest is merely installed" (see test_backend_openscad).
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _require_openscad() -> None:
    from phimech.backends.openscad import OpenSCADBackend

    if not OpenSCADBackend().available():
        _skip("openscad is not installed")


# --------------------------------------------------------------------------------------
# synthetic sheets: tiles with a filled rectangle, composed exactly like the real thing
# --------------------------------------------------------------------------------------


def _tile(shape: Tuple[int, int, int, int], colour=FG, size: int = TILE) -> preview.Canvas:
    rows = preview._blank(size, size, BG)
    x, y, w, h = shape
    preview._fill_rect(rows, x, y, w, h, colour)
    return rows


def _sheet(path: Path, shapes: Dict[str, Tuple[int, int, int, int]], size: int = TILE) -> Path:
    tiles = {name: _tile(shapes.get(name, (30, 30, 40, 40)), size=size) for name in
             preview.DEFAULT_VIEWS}
    canvas = preview.compose_sheet(tiles, preview.DEFAULT_VIEWS, size, ["TEST"])
    preview.write_png(path, size * preview.GRID_COLUMNS, size * preview.GRID_ROWS, canvas)
    return path


def _pixel(rows: preview.Canvas, x: int, y: int) -> Tuple[int, int, int]:
    return rows[y][x * 3], rows[y][x * 3 + 1], rows[y][x * 3 + 2]


# --------------------------------------------------------------------------------------
# PNG codec
# --------------------------------------------------------------------------------------


def test_png_round_trip_preserves_every_pixel() -> None:
    width, height = 37, 11
    rows = [bytearray((x * 7 + y * 13 + c * 31) & 0xFF for x in range(width) for c in range(3))
            for y in range(height)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rt.png"
        size = preview.write_png(path, width, height, rows)
        assert size == path.stat().st_size
        w, h, back = preview.read_png(path)
    assert (w, h) == (width, height)
    assert back == rows


def _filtered_png(rows: List[bytearray], width: int, kind: int, bpp: int = 3) -> bytes:
    """Encode rows with one PNG filter type applied forwards - the decoder must undo it."""
    out = bytearray()
    prev = bytearray(width * bpp)
    for cur in rows:
        out.append(kind)
        for i in range(len(cur)):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = a
            elif kind == 2:
                pred = b
            elif kind == 3:
                pred = (a + b) >> 1
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            out.append((cur[i] - pred) & 0xFF)
        prev = cur

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(
            ">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, len(rows), 8, 2 if bpp == 3 else 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(out))) + chunk(b"IEND", b""))


def test_png_decoder_reverses_all_five_filters() -> None:
    """OpenSCAD's lodepng uses Sub, Up and Paeth on real tiles; Average and None for completeness."""
    width, height = 23, 9
    rows = [bytearray((x * 5 + y * 17 + c * 29 + (x * y) % 7) & 0xFF
                      for x in range(width) for c in range(3)) for y in range(height)]
    with tempfile.TemporaryDirectory() as tmp:
        for kind in range(5):
            path = Path(tmp) / "f{}.png".format(kind)
            path.write_bytes(_filtered_png(rows, width, kind))
            w, h, back = preview.read_png(path)
            assert (w, h) == (width, height)
            assert back == rows, "filter type {} decoded wrongly".format(kind)


def test_png_decoder_drops_alpha_and_refuses_16_bit() -> None:
    width, height = 5, 2
    rgba = [bytearray(v & 0xFF for v in range(y * 40, y * 40 + width * 4))
            for y in range(height)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rgba.png"
        path.write_bytes(_filtered_png(rgba, width, 0, bpp=4))
        _, _, back = preview.read_png(path)
        for y in range(height):
            expected = bytearray()
            for x in range(width):
                expected += rgba[y][x * 4:x * 4 + 3]
            assert back[y] == expected
        deep = Path(tmp) / "deep.png"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)
        body = struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(
            ">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
        deep.write_bytes(b"\x89PNG\r\n\x1a\n" + body)
        try:
            preview.read_png(deep)
        except ValueError as exc:
            assert "16-bit" in str(exc)
        else:
            raise AssertionError("a 16-bit PNG must be refused, not mis-decoded")


# --------------------------------------------------------------------------------------
# font, labels, layout
# --------------------------------------------------------------------------------------


def test_font_covers_letters_digits_and_is_5x7() -> None:
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-:/()=+%_":
        glyph = preview._FONT[char]
        assert len(glyph) == 7, char
        assert all(len(row) == 5 and set(row) <= {"0", "1"} for row in glyph), char
    assert preview.text_width_px("ISO", 1) == 3 * 6 - 1
    assert preview.text_width_px("ISO", 3) == 3 * 18 - 3


def test_label_band_and_scale_are_pure_functions_of_tile_size() -> None:
    assert preview.font_scale(160) == 1
    assert preview.font_scale(400) == 3
    assert preview.label_band_px(160) == 13
    assert preview.label_band_px(400) == 39
    assert preview.label_band_px(400) < 400 // 8, "the label must not eat the view"


def test_compose_sheet_places_tiles_labels_and_info() -> None:
    tiles = {name: _tile((10, 40, 30, 30)) for name in preview.DEFAULT_VIEWS}
    canvas = preview.compose_sheet(tiles, preview.DEFAULT_VIEWS, TILE,
                                   ["PHIMECH", "X 1.0"])
    assert len(canvas) == 2 * TILE and all(len(r) == 4 * TILE * 3 for r in canvas)
    # tile 5 ("top") sits at column 1, row 1: its shape pixel must land at the offset.
    x, y = preview._slot_origin(5, TILE)
    assert _pixel(canvas, x + 20, y + 50) == FG
    assert _pixel(canvas, x + 5, y + 90) == BG
    # every view tile carries a label box (dark pixels in the band); the info tile has text.
    for index in range(7):
        x, y = preview._slot_origin(index, TILE)
        band = [_pixel(canvas, x + dx, y + dy) for dx in range(2, 30) for dy in
                range(2, preview.label_band_px(TILE))]
        assert preview._LABEL_BG in band, "no label on tile {}".format(index)
    x, y = preview._slot_origin(preview.INFO_SLOT, TILE)
    info = [_pixel(canvas, x + dx, y + dy) for dx in range(2, 60) for dy in range(2, 12)]
    assert preview._INFO_FG in info
    # borders are drawn on every tile edge
    assert _pixel(canvas, 0, 0) == preview._BORDER_RGB
    assert _pixel(canvas, TILE, TILE + 5) == preview._BORDER_RGB


def test_view_validation() -> None:
    for bad in ((), ("iso", "iso"), ("nope",), tuple(preview.DEFAULT_VIEWS) + ("iso",)):
        try:
            preview._validate_views(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("views {!r} must be refused".format(bad))
    assert preview._validate_views(["top", "iso"]) == ("top", "iso")


# --------------------------------------------------------------------------------------
# silhouette IoU
# --------------------------------------------------------------------------------------


def test_mask_iou_edge_cases() -> None:
    empty = [bytearray(4) for _ in range(4)]
    full = [bytearray(b"\x01" * 4) for _ in range(4)]
    half = [bytearray(b"\x01\x01\x00\x00") for _ in range(4)]
    assert preview.mask_iou(empty, empty)[0] == 1.0
    assert preview.mask_iou(full, empty)[0] == 0.0
    assert preview.mask_iou(full, half) == (0.5, 8, 16, 8)


def test_compare_sheets_identical_and_per_view_difference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = _sheet(Path(tmp) / "ref.png", {})
        same = _sheet(Path(tmp) / "same.png", {})
        # "top" gets a shape half as tall; every other view is untouched
        cand = _sheet(Path(tmp) / "cand.png", {"top": (30, 30, 40, 20)})
        identical = preview.compare_sheets(ref, same)
        assert identical["min_iou"] == 1.0 and identical["same_shape"] is True
        result = preview.compare_sheets(ref, cand)
        assert result["per_view"]["top"] == 0.5, result["per_view"]
        assert all(result["per_view"][v] == 1.0 for v in preview.DEFAULT_VIEWS if v != "top")
        assert result["worst_view"] == "top"
        assert result["min_iou"] == 0.5 and result["same_shape"] is False
        assert result["threshold"] == preview.DEFAULT_IOU_THRESHOLD
        assert preview.compare_sheets(ref, cand, threshold=0.4)["same_shape"] is True


def test_compare_sheets_ignores_label_band_and_refuses_mismatched_tiles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = _sheet(Path(tmp) / "ref.png", {})
        # paint garbage across the whole label band of the "iso" tile of a copy
        w, h, rows = preview.read_png(ref)
        preview._fill_rect(rows, 0, 0, TILE, preview.label_band_px(TILE) - 1, (255, 0, 0))
        noisy = Path(tmp) / "noisy.png"
        preview.write_png(noisy, w, h, rows)
        assert preview.compare_sheets(ref, noisy)["min_iou"] == 1.0
        small = _sheet(Path(tmp) / "small.png", {}, size=64)
        try:
            preview.compare_sheets(ref, small)
        except ValueError as exc:
            assert "tile" in str(exc)
        else:
            raise AssertionError("sheets with different tile sizes must be refused")


# --------------------------------------------------------------------------------------
# grader check and the recorded wrong-shape evidence (no OpenSCAD needed)
# --------------------------------------------------------------------------------------


def test_grade_preview_min_iou_check() -> None:
    case = {"id": "x", "expect": {"preview_min_iou": 0.92}}
    ok = grade.grade_case(case, {"ok": True, "preview": {"min_iou": 0.95, "worst_view": "top"}})
    bad = grade.grade_case(case, {"ok": True, "preview": {"min_iou": 0.7, "worst_view": "iso"}})
    none = grade.grade_case(case, {"ok": True})
    assert ok["passed"] and not bad["passed"] and not none["passed"]
    assert "0.700" in bad["checks"][0]["detail"] and "iso" in bad["checks"][0]["detail"]
    assert "no preview comparison" in none["checks"][0]["detail"]


def test_wrong_shape_suite_is_well_formed() -> None:
    suite = wrong_shape.suite()
    ids = [case["id"] for case in suite["cases"]]
    assert len(ids) >= 6 and len(set(ids)) == len(ids)
    assert suite["iou_threshold"] == preview.DEFAULT_IOU_THRESHOLD
    for case in wrong_shape.CASES:
        assert case.prompt and case.why and case.wrong_summary
        assert validate_ast(case.reference) == [], case.id
        assert validate_ast(case.wrong) == [], case.id
        expect = case.expect(preview_threshold=None)
        assert "preview_min_iou" not in expect
        assert case.expect(preview_threshold=0.5)["preview_min_iou"] == 0.5
    try:
        from phimech.check.gates import gate_catalog
    except Exception as exc:  # pragma: no cover - only if check deps are absent
        _skip("gate catalogue unavailable: %s" % exc)
    known = {entry["name"] for entry in gate_catalog()}
    for case in suite["cases"]:
        for name in case["expect"]["gates_pass"]:
            assert name in known, "unknown gate {} in {}".format(name, case["id"])


def test_wrong_shape_fixtures_are_committed_for_every_case() -> None:
    for case in wrong_shape.CASES:
        sheet = PREVIEW_FIXTURES / "{}.png".format(case.id)
        wrong = PREVIEW_FIXTURES / "wrong" / "{}.json".format(case.id)
        assert sheet.is_file(), "missing reference sheet {}".format(sheet)
        assert wrong.is_file(), "missing wrong candidate {}".format(wrong)
        width, height, _ = preview.read_png(sheet)
        assert (width, height) == (4 * wrong_shape.FIXTURE_TILE_PX,
                                   2 * wrong_shape.FIXTURE_TILE_PX)
        node = from_dict(json.loads(wrong.read_text(encoding="utf-8")))
        assert node.to_dict() == case.wrong.to_dict(), (
            "wrong/{}.json drifted from the case definition; run "
            "`phimech preview bench --write-fixtures`".format(case.id))


def test_recorded_results_meet_the_phase_a_gate() -> None:
    """The committed evidence: numeric gates catch <= 1/6, the preview proxy catches >= 5/6."""
    results = json.loads((PREVIEW_FIXTURES / "results.json").read_text(encoding="utf-8"))
    cases = results["cases"]
    assert len(cases) >= 6
    assert {c["id"] for c in cases} == {case.id for case in wrong_shape.CASES}
    numeric = sum(1 for c in cases if c["caught_by_numeric"])
    caught = sum(1 for c in cases if c["caught_by_preview"])
    assert numeric <= len(cases) * wrong_shape.NUMERIC_CATCH_MAX, numeric
    assert caught >= len(cases) * wrong_shape.PREVIEW_CATCH_MIN, caught
    assert results["gate"]["ok"] is True
    assert results["iou_threshold"] == preview.DEFAULT_IOU_THRESHOLD
    for c in cases:
        assert c["caught_by_preview"] == (c["min_iou"] < results["iou_threshold"]), c["id"]
        assert not any(name.endswith("FAIL") for name in c["numeric_checks"]), c["id"]
    blind = [c for c in cases if not c["silhouette_visible"]]
    assert blind and all(c["min_iou"] == 1.0 for c in blind), "the documented miss must be 1.0"
    controls = results["controls"]
    assert controls["same_shape_rerender"]["min_iou"] == 1.0
    assert controls["same_shape_height_plus_2mm"]["same_shape"] is True


def test_format_table_reads_back() -> None:
    results = json.loads((PREVIEW_FIXTURES / "results.json").read_text(encoding="utf-8"))
    outcome = {"results": [dict(c, per_view=c["per_view"]) for c in results["cases"]],
               "summary": results["summary"], "gate": results["gate"],
               "iou_threshold": results["iou_threshold"], "controls": results["controls"]}
    table = wrong_shape.format_table(outcome)
    assert "GATE OK" in table and "blind" in table and "CAUGHT" in table


# --------------------------------------------------------------------------------------
# MCP tool docstrings (same bar as tests/test_mcp_tool_docs.py) and the CLI hook
# --------------------------------------------------------------------------------------


def _tool_docs() -> Dict[str, Tuple[List[str], str]]:
    """The nested ``@mcp.tool`` functions inside ``register`` - what the guide and agents read."""
    source = (REPO / "phimech" / "mcp" / "tools_preview.py").read_text(encoding="utf-8")
    tree = _ast.parse(source)
    found: Dict[str, Tuple[List[str], str]] = {}
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.FunctionDef) or node.name not in tools_preview.TOOLS:
            continue
        decorated = any(
            isinstance(d.func if isinstance(d, _ast.Call) else d, _ast.Attribute)
            and (d.func if isinstance(d, _ast.Call) else d).attr == "tool"
            for d in node.decorator_list
        )
        assert decorated, "{} must carry @mcp.tool".format(node.name)
        found[node.name] = ([a.arg for a in node.args.args], _ast.get_docstring(node) or "")
    return found


def test_preview_tools_state_paths_units_and_limits() -> None:
    docs = _tool_docs()
    assert set(docs) == set(tools_preview.TOOLS)
    for name, (args, doc) in docs.items():
        low = doc.lower()
        assert doc.strip(), name
        assert "absolute" in low, "{}: path arguments must be documented as absolute".format(name)
        assert "limits:" in low, "{}: needs a Limits: sentence".format(name)
        assert any(m in low for m in (" mm", "mm3", "pixel")), "{}: name the units".format(name)
        assert any(a.endswith("_png") or a == "path" for a in args), name


def test_preview_tools_register_and_fail_closed() -> None:
    class FakeMCP:
        def __init__(self) -> None:
            self.tools: Dict[str, object] = {}

        def tool(self, fn):  # bare @mcp.tool form, as server.py uses it
            self.tools[fn.__name__] = fn
            return fn

    fake = FakeMCP()
    tools_preview.register(fake)
    assert set(fake.tools) == set(tools_preview.TOOLS)
    # the registered wrappers must reach the impls, not just exist
    assert fake.tools["preview_compare"]("a.png", "/tmp/b.png")["ok"] is False
    relative = tools_preview.preview_compare_impl("a.png", "/tmp/b.png")
    assert relative["ok"] is False and "absolute" in relative["error"]
    missing = tools_preview.preview_sheet_impl("/definitely/not/here.stl", "/tmp/out.png")
    assert missing["ok"] is False and "not found" in missing["error"]
    bad_ext = tools_preview.preview_sheet_impl(str(FIXTURE_BASE_STL), "/tmp/out.jpg")
    assert bad_ext["ok"] is False and ".png" in bad_ext["error"]


def test_register_cli_adds_preview_commands() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    preview.register_cli(sub)
    args = parser.parse_args(["preview", "compare", "/a.png", "/b.png", "--json"])
    assert args.command == "preview" and args.preview_command == "compare"
    assert args.threshold == preview.DEFAULT_IOU_THRESHOLD and callable(args.handler)
    args = parser.parse_args(["preview", "sheet", "/m.stl", "-o", "/s.png", "--tile-px", "200"])
    assert args.tile_px == 200 and args.keep_views is False
    args = parser.parse_args(["preview", "bench", "--record"])
    assert args.record is True and args.tile_px == wrong_shape.FIXTURE_TILE_PX


def test_doctor_section_is_json() -> None:
    section = preview.doctor_section()
    json.dumps(section)
    assert section["name"] == "preview" and "ok" in section


# --------------------------------------------------------------------------------------
# with OpenSCAD: real renders
# --------------------------------------------------------------------------------------


def test_render_sheet_from_stl_and_keep_views() -> None:
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sheet.png"
        result = preview.render_sheet(FIXTURE_BASE_STL, out, tile_px=160, keep_views=True)
        assert Path(result.path) == out and out.is_file()
        width, height, _ = preview.read_png(out)
        assert (width, height) == (640, 320)
        size = result.bbox_mm["size"]
        assert abs(size[0] - 33.0) < 1e-6 and abs(size[1] - 39.0) < 1e-6
        assert abs(size[2] - 14.5) < 1e-6
        assert result.volume_mm3 > 0 and result.render_s > 0
        assert set(result.views) == set(preview.DEFAULT_VIEWS)
        for name, entry in result.views.items():
            assert Path(entry["path"]).is_file(), name
            assert entry["camera"] == list(preview.CAMERAS[name])
        again = preview.render_sheet(FIXTURE_BASE_STL, Path(tmp) / "again.png", tile_px=160)
        same = preview.compare_sheets(out, again.path)
        assert same["min_iou"] == 1.0, "a re-render must be pixel-identical: {}".format(same)
        other = preview.render_sheet(FIXTURE_LID_STL, Path(tmp) / "lid.png", tile_px=160)
        assert preview.compare_sheets(out, other.path)["same_shape"] is False


def _window_box():
    """A closed hollow box (40 x 30 x 20, 2 mm walls) with one window, in the +X wall only."""
    return ops.diff_all(ops.box(40, 30, 20), [
        ops.box(36, 26, 16, at=(2, 2, 2)),
        ops.box(5, 12, 8, at=(37, 9, 6)),
    ])


def _foreground_colour_change(a: preview.Canvas, b: preview.Canvas, size: int,
                              mirror_b: bool) -> Tuple[int, int, Tuple[float, float]]:
    """Pixels foreground in both tiles whose colours differ, the foreground count, and the
    centroid (fraction of tile) of the differing pixels. ``mirror_b`` flips b left-right."""
    differing = foreground = 0
    sx = sy = 0.0
    for y in range(preview.label_band_px(size), size):
        for x in range(size):
            pa = _pixel(a, x, y)
            pb = _pixel(b, size - 1 - x if mirror_b else x, y)
            fa = any(abs(c - k) > preview.FOREGROUND_DELTA for c, k in zip(pa, BG))
            fb = any(abs(c - k) > preview.FOREGROUND_DELTA for c, k in zip(pb, BG))
            if fa and fb:
                foreground += 1
                if any(abs(c - k) > 30 for c, k in zip(pa, pb)):
                    differing += 1
                    sx += x
                    sy += y
    centroid = (sx / differing / size, sy / differing / size) if differing else (0.0, 0.0)
    return differing, foreground, centroid


def test_left_and_right_views_show_which_wall_has_the_window() -> None:
    """A window on +X must show on RIGHT and not on LEFT.

    OpenSCAD shades a face by its normal only, so face-on, the window in the +X wall of a hollow
    box showed the inner face of the -X wall - same normal, same colour - and LEFT and RIGHT were
    pixel-for-pixel mirror images of one blank wall. The sheet now shades by depth.
    """
    _require_openscad()
    size = 160
    with tempfile.TemporaryDirectory() as tmp:
        result = preview.render_sheet(_window_box(), Path(tmp) / "sheet.png", tile_px=size,
                                      views=("left", "right", "front"), keep_views=True)
        tiles = {name: preview.read_png(entry["path"])[2]
                 for name, entry in result.views.items()}
        # Silhouettes are still exact mirror images (the outline of a box is a rectangle)...
        left_mask, _ = preview._tile_mask(tiles["left"], 0, 0, size)
        right_mask, _ = preview._tile_mask(tiles["right"], 0, 0, size)
        mirrored = [bytearray(reversed(row)) for row in left_mask]
        assert preview.mask_iou(mirrored, right_mask)[0] > 0.98
        # ...but what is inside them is not: the window reads as a deeper region on RIGHT only.
        changed, foreground, (cx, cy) = _foreground_colour_change(
            tiles["right"], tiles["left"], size, mirror_b=True)
        assert foreground > 0 and changed / foreground > 0.05, (changed, foreground)
        # 12 x 8 mm window centred on a 30 x 20 mm face: the change sits at the tile centre.
        assert abs(cx - 0.5) < 0.1 and abs(cy - 0.5) < 0.1, (cx, cy)
        # LEFT is a solid wall: one colour across the face.
        left_colours = {_pixel(tiles["left"], x, y) for y in range(size) for x in range(size)}
        left_face = {c for c in left_colours
                     if any(abs(v - k) > preview.FOREGROUND_DELTA for v, k in zip(c, BG))}
        right_colours = {_pixel(tiles["right"], x, y) for y in range(size) for x in range(size)}
        right_face = {c for c in right_colours
                      if any(abs(v - k) > preview.FOREGROUND_DELTA for v, k in zip(c, BG))}
        assert len(right_face) > len(left_face), (left_face, right_face)


def test_render_sheet_accepts_ast_json_and_scad_but_not_ir() -> None:
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        ast_path = Path(tmp) / "part.json"
        ast_path.write_text(json.dumps(ops.cyl(20, 10).to_dict()), encoding="utf-8")
        result = preview.render_sheet(ast_path, Path(tmp) / "a.png", tile_px=96, views=("top",))
        assert abs(result.bbox_mm["size"][2] - 10.0) < 1e-6
        scad = Path(tmp) / "part.scad"
        scad.write_text("cube([10, 20, 30]);\n", encoding="utf-8")
        result = preview.render_sheet(scad, Path(tmp) / "s.png", tile_px=96, views=("iso",))
        assert [round(v, 3) for v in result.bbox_mm["size"]] == [10.0, 20.0, 30.0]
        ir = Path(tmp) / "design.json"
        ir.write_text(json.dumps({"components": []}), encoding="utf-8")
        try:
            preview.render_sheet(ir, Path(tmp) / "ir.png", tile_px=96)
        except ValueError as exc:
            assert "synth build" in str(exc)
        else:
            raise AssertionError("a design IR must be refused with a hint")


def test_for_build_picks_the_largest_printable_part() -> None:
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp).resolve()  # for_build resolves; macOS tempdirs live under /private
        report = {
            "printable_parts": ["base", "lid"],
            "outputs": [
                {"part": "base", "fmt": "stl", "path": str(FIXTURE_BASE_STL)},
                {"part": "lid", "fmt": "stl", "path": str(FIXTURE_LID_STL)},
                {"part": "assembly", "fmt": "png", "path": "/nowhere/assembly.png"},
            ],
            "measurements": {"base": {"volume_mm3": 6375.4}, "lid": {"volume_mm3": 2616.2}},
        }
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        result = preview.for_build(out, tile_px=96)
        assert result.part == "base" and Path(result.path) == out / "preview.png"
        lid = preview.for_build(out, part="lid", tile_px=96, filename="lid.png")
        assert lid.part == "lid" and abs(lid.bbox_mm["size"][2] - 3.5) < 1e-6


def test_wrong_shape_case_end_to_end_and_fixture_drift() -> None:
    """One wrong candidate through the real pipeline, and the committed reference still holds."""
    _require_openscad()
    case = wrong_shape.case_by_id("spacer-box-for-cylinder")
    assert case is not None
    reference = PREVIEW_FIXTURES / "{}.png".format(case.id)
    with tempfile.TemporaryDirectory() as tmp:
        result = wrong_shape.grade_candidate(case, case.wrong, Path(tmp),
                                             reference_sheet=reference)
        assert result["numeric_passed"] is True, result["numeric_checks"]
        assert result["caught_by_preview"] is True and result["full_passed"] is False
        assert result["min_iou"] < preview.DEFAULT_IOU_THRESHOLD
        fresh = preview.render_sheet(case.reference, Path(tmp) / "fresh.png",
                                     tile_px=wrong_shape.FIXTURE_TILE_PX)
        drift = preview.compare_sheets(reference, fresh.path)
        assert drift["same_shape"] is True, (
            "committed reference sheet no longer matches a fresh render: {}".format(drift))


def test_preview_tools_end_to_end() -> None:
    _require_openscad()
    with tempfile.TemporaryDirectory() as tmp:
        first = tools_preview.preview_sheet_impl(str(FIXTURE_LID_STL), str(Path(tmp) / "a.png"),
                                                 tile_px=96)
        assert first["ok"] is True and Path(first["path"]).is_file()
        assert first["bbox_mm"]["size"][2] == 3.5 and first["render_s"] > 0
        second = tools_preview.preview_sheet_impl(str(FIXTURE_LID_STL), str(Path(tmp) / "b.png"),
                                                  tile_px=96)
        compared = tools_preview.preview_compare_impl(first["path"], second["path"])
        assert compared["ok"] is True and compared["same_shape"] is True
        assert compared["min_iou"] == 1.0
        too_small = tools_preview.preview_sheet_impl(str(FIXTURE_LID_STL), str(Path(tmp) / "c.png"),
                                                     tile_px=8)
        assert too_small["ok"] is False and "tile_px" in too_small["error"]


def _run_direct() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(test.__name__, exc))
        except Exception as exc:  # pragma: no cover - only used by direct invocation
            failures += 1
            print("FAIL {}: {!r}".format(test.__name__, exc))
        else:
            print("PASS {}".format(test.__name__))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
