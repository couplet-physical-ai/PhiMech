"""Tests for the batch drawing surface (``phimech/mcp/tools_drawing_batch.py``).

Runs both ways, like the rest of the suite::

    /usr/bin/python3 tests/test_drawing_batch.py     # direct, exit code = failure count
    pytest tests/test_drawing_batch.py

What is gated here
------------------
The crux of this tool is *discrimination*: a SolidWorks handoff bundle and a ``synth build`` are
each ONE source, while any other directory is a folder of parts. Those tests are pure - they
build directory shapes out of empty files and never touch a CAD kernel, so they run on a machine
with nothing installed. The same goes for output naming: two sources with the same stem must not
overwrite each other, which is checked by driving the whole batch with a stub renderer rather
than by rendering twenty real sheets.

Only the end-to-end test needs OpenSCAD, and it is gated with ``unittest.skipUnless``, which
both pytest and the direct runner below report as a skip rather than a failure.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phimech.mcp import tools_drawing_batch as b  # noqa: E402
from phimech.synth import drawing as drawing_mod  # noqa: E402


def _openscad_available() -> bool:
    try:
        from phimech.backends.openscad import OpenSCADBackend

        return bool(OpenSCADBackend().available())
    except Exception:  # noqa: BLE001 - a broken backend is "not available" for this file
        return False


HAS_OPENSCAD = _openscad_available()


# --------------------------------------------------------------------------------------
# fixtures: directory shapes, built from empty files
# --------------------------------------------------------------------------------------


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_bundle(directory: Path) -> Path:
    """The three files ``preview.is_solidworks_bundle`` looks for, plus a decoy part inside."""
    for name in ("manifest.json", "model.step", "source.json"):
        _touch(directory / name, "{}")
    _touch(directory / "decoy-part.stl")
    return directory


def _make_build(directory: Path) -> Path:
    """A finished ``synth build``: the report marker plus the STLs it wrote."""
    _touch(directory / "report.json", json.dumps({"outputs": [], "printable_parts": ["base"]}))
    _touch(directory / "base.stl")
    _touch(directory / "lid.stl")
    return directory


CSG_AST = json.dumps({"kind": "cube", "size": [10, 10, 10]})
DESIGN_IR = json.dumps({"components": [{"type": "pcb", "id": "PCB1"}], "enclosure": {}})


# --------------------------------------------------------------------------------------
# a stub renderer, so the batch machinery is testable with no CAD installed
# --------------------------------------------------------------------------------------


class _StubRenderer:
    """Replaces ``drawing.render_drawing`` / ``drawing.for_build`` for the duration of a test.

    It writes a real file at the path it is handed, which is what makes the collision test
    meaningful: an overwrite shows up as one file with the wrong contents, not as a missing one.
    """

    def __init__(self, fail_for: Optional[str] = None) -> None:
        self.fail_for = fail_for
        self.calls: List[Dict[str, Any]] = []

    def _result(self, source: Path, out_svg: Path, title: Optional[str]) -> Any:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_svg.write_text(str(source), encoding="utf-8")
        return drawing_mod.DrawingResult(
            path=str(out_svg), png=None, png_tool=None, scale=1.0, sheet_mm=[420.0, 297.0],
            bbox_mm={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}, volume_mm3=1.0,
            levels_z=[0.0, 1.0], groups=[{"id": "H1"}, {"id": "H2"}], views={}, render_s=0.0,
            source=Path(source).name, part=title,
        )

    def render_drawing(self, source: Any, out_svg: Any, **kwargs: Any) -> Any:
        self.calls.append({"source": str(source), "out_svg": str(out_svg), "kwargs": kwargs})
        if self.fail_for and self.fail_for in str(source):
            raise RuntimeError("stub failure for {}".format(source))
        return self._result(Path(str(source)), Path(str(out_svg)), kwargs.get("title"))

    def for_build(self, out_dir: Any, **kwargs: Any) -> Any:
        self.calls.append({"source": str(out_dir), "out_svg": str(kwargs.get("out_svg")),
                           "kwargs": kwargs})
        if self.fail_for and self.fail_for in str(out_dir):
            raise RuntimeError("stub failure for {}".format(out_dir))
        return self._result(Path(str(out_dir)), Path(str(kwargs["out_svg"])), kwargs.get("title"))


class _stubbed:
    """Context manager swapping the drawing entry points for a stub, restoring them after."""

    def __init__(self, stub: _StubRenderer) -> None:
        self.stub = stub
        self.saved: Dict[str, Any] = {}

    def __enter__(self) -> _StubRenderer:
        self.saved = {"render_drawing": drawing_mod.render_drawing,
                      "for_build": drawing_mod.for_build}
        drawing_mod.render_drawing = self.stub.render_drawing  # type: ignore[assignment]
        drawing_mod.for_build = self.stub.for_build  # type: ignore[assignment]
        return self.stub

    def __exit__(self, *exc: Any) -> None:
        drawing_mod.render_drawing = self.saved["render_drawing"]  # type: ignore[assignment]
        drawing_mod.for_build = self.saved["for_build"]  # type: ignore[assignment]


class _workspace:
    """``with _workspace() as root:`` - a throwaway tree, cleaned up however the test exits."""

    def __enter__(self) -> Path:
        # resolved, because the tool resolves every path it is given and on macOS the temp
        # root is /var -> /private/var; comparing an unresolved fixture path would fail there.
        self.directory = Path(tempfile.mkdtemp(prefix="phimech-batch-test-")).resolve()
        return self.directory

    def __exit__(self, *exc: Any) -> None:
        shutil.rmtree(str(self.directory), ignore_errors=True)


def _kinds(sources: List[b.Source]) -> List[str]:
    return [source.kind for source in sources]


def _names(sources: List[b.Source]) -> List[str]:
    return [source.path.name for source in sources]


# --------------------------------------------------------------------------------------
# source discrimination
# --------------------------------------------------------------------------------------


def test_mixed_file_list_keeps_every_geometry_file() -> None:
    """A hand-written list of files is taken as given, in the order it was given."""
    with _workspace() as root:
        files = [_touch(root / "a.stl"), _touch(root / "b.step"), _touch(root / "c.stp"),
                 _touch(root / "d.scad"), _touch(root / "e.json", CSG_AST)]
        sources = b.collect_sources(files)
        assert _kinds(sources) == [b.KIND_FILE] * 5, _kinds(sources)
        assert _names(sources) == ["a.stl", "b.step", "c.stp", "d.scad", "e.json"]


def test_named_file_of_an_unknown_type_is_reported_not_dropped() -> None:
    with _workspace() as root:
        sources = b.collect_sources([_touch(root / "notes.txt")])
        assert _kinds(sources) == [b.KIND_UNSUPPORTED]
        assert "unsupported file" in (sources[0].reason or "")


def test_missing_path_is_a_row_not_an_exception() -> None:
    with _workspace() as root:
        sources = b.collect_sources([root / "gone.stl"])
        assert _kinds(sources) == [b.KIND_MISSING]
        assert "not found" in (sources[0].reason or "")


def test_directory_expansion_collects_cad_files_and_ignores_the_rest() -> None:
    with _workspace() as root:
        folder = root / "parts"
        _touch(folder / "bracket.stl")
        _touch(folder / "plate.step")
        _touch(folder / "README.md")
        _touch(folder / "sheet.svg")
        _touch(folder / "notes.json", json.dumps({"hello": "world"}))
        sources = b.collect_sources([folder])
        assert _names(sources) == ["bracket.stl", "plate.step"], _names(sources)


def test_directory_expansion_is_shallow_unless_recursive() -> None:
    with _workspace() as root:
        folder = root / "parts"
        _touch(folder / "top.stl")
        _touch(folder / "sub" / "deep.stl")
        assert _names(b.collect_sources([folder])) == ["top.stl"]
        assert _names(b.collect_sources([folder], recursive=True)) == ["deep.stl", "top.stl"]


def test_solidworks_bundle_is_one_source_not_the_parts_inside_it() -> None:
    """The crux: a bundle carries a model.step and may carry other geometry - it is still one."""
    with _workspace() as root:
        bundle = _make_bundle(root / "handoff")
        sources = b.collect_sources([bundle])
        assert _kinds(sources) == [b.KIND_BUNDLE]
        assert sources[0].path == bundle


def test_build_directory_is_one_source_not_every_stl_in_it() -> None:
    with _workspace() as root:
        build = _make_build(root / "out")
        sources = b.collect_sources([build])
        assert _kinds(sources) == [b.KIND_BUILD]
        assert sources[0].path == build


def test_a_folder_of_stls_is_a_batch_not_a_build() -> None:
    """No ``report.json`` means it is a folder of parts, and every part gets its own sheet."""
    with _workspace() as root:
        folder = root / "loose"
        _touch(folder / "one.stl")
        _touch(folder / "two.stl")
        sources = b.collect_sources([folder])
        assert _kinds(sources) == [b.KIND_FILE, b.KIND_FILE]
        assert _names(sources) == ["one.stl", "two.stl"]


def test_bundles_and_builds_inside_a_scanned_folder_stay_whole() -> None:
    with _workspace() as root:
        folder = root / "everything"
        _make_bundle(folder / "handoff")
        _make_build(folder / "build")
        _touch(folder / "loose.stl")
        sources = b.collect_sources([folder])
        assert sorted(zip(_names(sources), _kinds(sources))) == [
            ("build", b.KIND_BUILD), ("handoff", b.KIND_BUNDLE), ("loose.stl", b.KIND_FILE)]
        # ...and the same when the walk reaches them one level down
        nested = b.collect_sources([root], recursive=True)
        assert sorted(zip(_names(nested), _kinds(nested))) == [
            ("build", b.KIND_BUILD), ("handoff", b.KIND_BUNDLE), ("loose.stl", b.KIND_FILE)]


def test_sources_are_deduplicated_by_resolved_path() -> None:
    with _workspace() as root:
        folder = root / "parts"
        part = _touch(folder / "a.stl")
        sources = b.collect_sources([part, folder])
        assert _names(sources) == ["a.stl"]


def test_design_ir_is_refused_with_build_it_first_guidance() -> None:
    """Refused, not skipped: "build it first" is actionable, silence is not."""
    with _workspace() as root:
        ir = _touch(root / "design.json", DESIGN_IR)
        sources = b.collect_sources([ir])
        assert _kinds(sources) == [b.KIND_REFUSED]
        reason = sources[0].reason or ""
        assert "design IR" in reason and "synth build" in reason
        # and the same inside a directory scan, where silence would be easiest
        folder = root / "designs"
        _touch(folder / "one.json", DESIGN_IR)
        _touch(folder / "two.json", CSG_AST)
        scanned = b.collect_sources([folder])
        assert sorted(zip(_names(scanned), _kinds(scanned))) == [
            ("one.json", b.KIND_REFUSED), ("two.json", b.KIND_FILE)]


def test_design_ir_reaches_the_caller_as_a_failed_row() -> None:
    with _workspace() as root:
        ir = _touch(root / "design.json", DESIGN_IR)
        out = root / "sheets"
        with _stubbed(_StubRenderer()):
            summary = b.drawing_batch_impl(str(out), paths=[str(ir)])
        assert summary["ok"] is False
        assert summary["total"] == 1 and summary["failed"] == 1 and summary["succeeded"] == 0
        row = summary["results"][0]
        assert row["status"] == "failed" and row["path"] is None
        assert "synth build" in row["error"]


# --------------------------------------------------------------------------------------
# output naming
# --------------------------------------------------------------------------------------


def test_output_name_suffixes_collisions_deterministically() -> None:
    taken: set = set()
    assert b.output_name("part", taken) == "part.svg"
    assert b.output_name("part", taken) == "part-2.svg"
    assert b.output_name("part", taken) == "part-3.svg"
    assert b.output_name("other", taken) == "other.svg"
    # case-insensitive, because the filesystems this runs on are
    assert b.output_name("PART", taken) == "PART-4.svg"


def test_output_name_falls_back_for_an_unusable_stem() -> None:
    taken: set = set()
    assert b.output_name("", taken) == "drawing.svg"
    assert b.output_name("a/b", taken) == "a_b.svg"


def test_same_stem_sources_never_overwrite_each_other() -> None:
    """Two ``part.*`` files must leave two files on disk, not one written twice."""
    with _workspace() as root:
        first = _touch(root / "a" / "part.stl")
        second = _touch(root / "b" / "part.step")
        out = root / "sheets"
        with _stubbed(_StubRenderer()):
            summary = b.drawing_batch_impl(str(out), paths=[str(first), str(second)])
        assert summary["ok"] is True and summary["succeeded"] == 2
        written = sorted(p.name for p in out.glob("*.svg"))
        assert written == ["part-2.svg", "part.svg"], written
        # each sheet came from its own source: an overwrite would leave one of these unmatched
        assert (out / "part.svg").read_text(encoding="utf-8") == str(first)
        assert (out / "part-2.svg").read_text(encoding="utf-8") == str(second)
        assert sorted(row["path"] for row in summary["results"]) == [
            str(out / "part-2.svg"), str(out / "part.svg")]


def test_a_failed_source_keeps_the_name_it_reserved() -> None:
    """A name is reserved before the render, and never handed to a later source.

    The failing source here is the *first* ``part.*``, so it holds ``part.svg`` and the one that
    succeeds writes ``part-2.svg``. A reserved-but-unwritten name costs nothing; reusing it would
    mean a half-written sheet from the failure could masquerade as the successful result.
    """
    with _workspace() as root:
        failing = _touch(root / "a" / "part.stl")
        working = _touch(root / "b" / "part.step")
        out = root / "sheets"
        with _stubbed(_StubRenderer(fail_for=str(root / "a"))):
            summary = b.drawing_batch_impl(str(out), paths=[str(failing), str(working)])
        assert summary["succeeded"] == 1 and summary["failed"] == 1
        ok_row = [row for row in summary["results"] if row["status"] == "ok"][0]
        assert Path(ok_row["path"]).name == "part-2.svg"
        assert [p.name for p in out.glob("*.svg")] == ["part-2.svg"]


# --------------------------------------------------------------------------------------
# failure isolation, ok semantics, caps and empty input
# --------------------------------------------------------------------------------------


def test_one_bad_file_never_loses_the_good_ones() -> None:
    with _workspace() as root:
        files = [_touch(root / "good1.stl"), _touch(root / "bad.stl"),
                 _touch(root / "good2.stl")]
        out = root / "sheets"
        with _stubbed(_StubRenderer(fail_for="bad.stl")):
            summary = b.drawing_batch_impl(str(out), paths=[str(f) for f in files])
        assert summary["ok"] is True
        assert (summary["total"], summary["succeeded"], summary["failed"], summary["skipped"]) \
            == (3, 2, 1, 0)
        statuses = {Path(row["source"]).name: row["status"] for row in summary["results"]}
        assert statuses == {"good1.stl": "ok", "bad.stl": "failed", "good2.stl": "ok"}
        failed = [row for row in summary["results"] if row["status"] == "failed"][0]
        assert "stub failure" in failed["error"]


def test_stop_on_error_reports_the_rest_as_skipped_and_is_not_ok() -> None:
    with _workspace() as root:
        files = [_touch(root / "good1.stl"), _touch(root / "bad.stl"),
                 _touch(root / "good2.stl")]
        out = root / "sheets"
        with _stubbed(_StubRenderer(fail_for="bad.stl")) as stub:
            summary = b.drawing_batch_impl(str(out), paths=[str(f) for f in files],
                                           continue_on_error=False)
        assert summary["ok"] is False
        assert (summary["total"], summary["succeeded"], summary["failed"], summary["skipped"]) \
            == (3, 1, 1, 1)
        assert summary["results"][2]["status"] == "skipped"
        assert "not attempted" in summary["results"][2]["error"]
        assert [Path(call["source"]).name for call in stub.calls] == ["good1.stl", "bad.stl"]


def test_every_result_row_carries_the_full_key_set() -> None:
    with _workspace() as root:
        out = root / "sheets"
        with _stubbed(_StubRenderer()):
            summary = b.drawing_batch_impl(
                str(out),
                paths=[str(_touch(root / "a.stl")), str(_touch(root / "notes.txt"))],
            )
        for row in summary["results"]:
            assert set(row) == {"source", "kind", "status", "path", "png", "groups", "render_s",
                                "error"}, sorted(row)
        assert summary["results"][0]["groups"] == 2
        assert summary["results"][1]["status"] == "skipped"
        assert summary["skipped"] == 1


def test_build_and_bundle_are_drawn_through_their_own_entry_points() -> None:
    with _workspace() as root:
        build = _make_build(root / "out")
        bundle = _make_bundle(root / "handoff")
        sheets = root / "sheets"
        with _stubbed(_StubRenderer()) as stub:
            summary = b.drawing_batch_impl(str(sheets), paths=[str(build), str(bundle)])
        assert summary["succeeded"] == 2
        kinds = {row["kind"] for row in summary["results"]}
        assert kinds == {b.KIND_BUILD, b.KIND_BUNDLE}
        # named from the directory, not from a file inside it
        assert sorted(p.name for p in sheets.glob("*.svg")) == ["handoff.svg", "out.svg"]
        assert len(stub.calls) == 2


def test_title_prefix_is_prepended_to_the_source_stem() -> None:
    with _workspace() as root:
        out = root / "sheets"
        with _stubbed(_StubRenderer()) as stub:
            b.drawing_batch_impl(str(out), paths=[str(_touch(root / "bracket.stl"))],
                                 title_prefix="REV-A ")
        assert stub.calls[0]["kwargs"]["title"] == "REV-A bracket"
        with _stubbed(_StubRenderer()) as stub:
            b.drawing_batch_impl(str(out), paths=[str(root / "bracket.stl")])
        assert stub.calls[0]["kwargs"]["title"] is None


def test_empty_input_is_refused_with_a_hint() -> None:
    with _workspace() as root:
        summary = b.drawing_batch_impl(str(root / "sheets"))
        assert summary["ok"] is False
        assert "paths" in summary["error"] and "directory" in summary["error"]
        assert summary.get("hint")


def test_a_directory_with_no_cad_files_returns_an_empty_ledger() -> None:
    with _workspace() as root:
        folder = root / "empty"
        _touch(folder / "README.md")
        summary = b.drawing_batch_impl(str(root / "sheets"), directory=str(folder))
        assert summary["ok"] is False
        assert (summary["total"], summary["succeeded"], summary["failed"], summary["skipped"]) \
            == (0, 0, 0, 0)
        assert summary["results"] == []
        assert "no CAD sources" in summary["error"] and summary.get("hint")


def test_max_files_refuses_rather_than_truncating() -> None:
    with _workspace() as root:
        folder = root / "parts"
        for index in range(4):
            _touch(folder / "p{}.stl".format(index))
        summary = b.drawing_batch_impl(str(root / "sheets"), directory=str(folder), max_files=2)
        assert summary["ok"] is False
        assert "4 sources exceed max_files=2" in summary["error"]
        assert not (root / "sheets").exists(), "nothing may be written when the batch is refused"
        for bad in (0, b.MAX_MAX_FILES + 1):
            refused = b.drawing_batch_impl(str(root / "sheets"), directory=str(folder),
                                           max_files=bad)
            assert refused["ok"] is False and "max_files must be between" in refused["error"]


def test_relative_paths_are_refused_like_every_other_tool() -> None:
    summary = b.drawing_batch_impl("relative/out", paths=["/tmp/a.stl"])
    assert summary["ok"] is False and "absolute" in summary["error"]
    summary = b.drawing_batch_impl("/tmp", paths=["relative.stl"])
    assert summary["ok"] is False and "absolute" in summary["error"]


def test_out_dir_that_is_a_file_is_refused() -> None:
    with _workspace() as root:
        blocker = _touch(root / "sheets")
        summary = b.drawing_batch_impl(str(blocker), paths=[str(_touch(root / "a.stl"))])
        assert summary["ok"] is False and "not a directory" in summary["error"]


# --------------------------------------------------------------------------------------
# the agent-facing contract
# --------------------------------------------------------------------------------------


def test_the_tool_docstring_quotes_the_real_caps() -> None:
    """The caps are spelled out in prose (a %-formatted docstring is invisible to ``ast``)."""
    import ast

    source = Path(b.__file__).with_suffix(".py").read_text(encoding="utf-8")
    doc = ""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "drawing_batch":
            doc = ast.get_docstring(node) or ""
    assert doc, "drawing_batch must have a docstring `ast.get_docstring` can read"
    assert "at {} by default".format(b.DEFAULT_MAX_FILES) in doc
    assert "hard ceiling {}".format(b.MAX_MAX_FILES) in doc
    assert "mm and degrees" in doc and "absolute" in doc.lower()


def test_register_wires_exactly_the_declared_tools() -> None:
    """``register(mcp)`` needs no fastmcp: a stand-in with a ``tool`` decorator is enough."""
    names: List[str] = []

    class _FakeMCP:
        @staticmethod
        def tool(fn: Any) -> Any:
            names.append(fn.__name__)
            return fn

    b.register(_FakeMCP())
    assert names == b.TOOLS == ["drawing_batch"]


def test_cli_hook_registers_the_drawings_command() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    b.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["drawings", "/tmp/a.stl", "/tmp/b.stl", "-o", "/tmp/out", "-r"])
    assert args.sources == ["/tmp/a.stl", "/tmp/b.stl"]
    assert args.output == "/tmp/out" and args.recursive is True
    assert args.max_files == b.DEFAULT_MAX_FILES and args.stop_on_error is False


# --------------------------------------------------------------------------------------
# end to end: real renders, only where a backend exists
# --------------------------------------------------------------------------------------

_PART = """
difference() {
  cube([40, 25, 6]);
  translate([10, 12.5, -1]) cylinder(h = 8, d = 5, $fn = 48);
}
"""


@unittest.skipUnless(HAS_OPENSCAD, "openscad is not installed")
def test_end_to_end_draws_every_source_and_reports_the_failures() -> None:
    """One real batch: two drawable sources plus a design IR that must be refused."""
    with _workspace() as root:
        folder = root / "parts"
        _touch(folder / "plate.scad", _PART)
        _touch(folder / "plate2.scad", _PART)
        _touch(folder / "design.json", DESIGN_IR)
        out = root / "sheets"
        summary = b.drawing_batch_impl(str(out), directory=str(folder), png=False)
        assert summary["ok"] is True, summary
        assert (summary["total"], summary["succeeded"], summary["failed"]) == (3, 2, 1)
        for row in summary["results"]:
            if row["status"] != "ok":
                assert "synth build" in row["error"]
                continue
            sheet = Path(row["path"])
            assert sheet.is_file() and sheet.stat().st_size > 0
            assert sheet.read_text(encoding="utf-8").lstrip().startswith("<?xml")
            assert row["groups"] >= 1
        assert sorted(p.name for p in out.glob("*.svg")) == ["plate.svg", "plate2.svg"]


if __name__ == "__main__":  # direct runner for the system python
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except unittest.SkipTest as exc:
                print("skip ", name, exc)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, exc)
            else:
                print("ok   ", name)
    raise SystemExit(1 if failures else 0)
