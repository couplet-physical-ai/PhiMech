"""Viewer server + asset-bundle tests.

Runs both ways, like the rest of the suite::

    /usr/bin/python3 tests/test_viewer.py     # direct, prints PASS/FAIL, exit code = failures
    pytest tests/test_viewer.py

Stdlib only (``urllib``). Geometry-dependent tests prefer a real build under ``out/`` and fall back
to the committed fixture in ``fixtures/viewer`` — so a fresh clone with **no CAD toolchain
installed** still exercises every geometry assertion instead of skipping it. They only skip when
even the fixture is missing.
"""

from __future__ import annotations

import os as _os

import json
import sys
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by the system-python runner
    pytest = None  # type: ignore[assignment]

from phimech.viewer.server import (
    ASSETS_DIR,
    build_manifest,
    discover_builds,
    serve,
    stl_triangle_count,
)

REPO = Path(__file__).resolve().parents[1]

#: The build the task validates against. Any build works; this one is preferred when present.
PREFERRED_BUILD = REPO / "out" / "final" / "box"

#: Committed fallback so the geometry tests run on a clone with no OpenSCAD/FreeCAD installed.
#: Binary STL, ~760 KB total. Regenerate with ``fixtures/viewer/README.md``.
FIXTURE_BUILD = REPO / "fixtures" / "viewer"


class _Skipped(Exception):
    pass


def _skip(message: str) -> "None":
    # pytest.skip() raises Skipped, which derives from BaseException, so it sails through the
    # `except Exception` in _run_direct and turns a skip into a crash with a traceback and exit 1.
    # Delegating merely because pytest is IMPORTABLE was the bug: `bin/phimech check` runs each test
    # file directly with the venv python, where pytest is installed but is not driving. pytest sets
    # PYTEST_CURRENT_TEST for the duration of each test, so it tells "installed" from "running".
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _find_build_root() -> Tuple[Path, str]:
    """Return ``(out_dir, build_id)`` for a real build, falling back to the committed fixture.

    A generated build under ``out/`` wins when present, so these tests keep validating real
    pipeline output on a developer machine. The fixture is the floor, not the preference: without
    it a clone that has no CAD backends installed would skip every geometry assertion and report a
    green run that checked nothing.
    """
    if PREFERRED_BUILD.is_dir() and any(PREFERRED_BUILD.glob("*.stl")):
        return PREFERRED_BUILD.parent, PREFERRED_BUILD.name
    for parent in (REPO / "out" / "final", REPO / "out"):
        if not parent.is_dir():
            continue
        for build in discover_builds(parent):
            if build["part_count"]:
                return parent, build["id"]
    if FIXTURE_BUILD.is_dir() and any(FIXTURE_BUILD.glob("*.stl")):
        return FIXTURE_BUILD.parent, FIXTURE_BUILD.name
    _skip("no build output under out/ and no fixture at fixtures/viewer")
    raise AssertionError("unreachable")


class _Server:
    """Context manager running the viewer on an ephemeral port in a background thread."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.httpd: Any = None
        self.url = ""
        self.thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_Server":
        # port=0 asks the OS for a free port: parallel test runs must not collide.
        self.httpd, self.url = serve(self.out_dir, port=0, quiet=True)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)

    # -- helpers ---------------------------------------------------------------------------
    def get(self, path: str) -> Tuple[int, bytes, Dict[str, str]]:
        """GET, returning ``(status, body, headers)`` and never raising on 4xx/5xx."""
        try:
            with urllib.request.urlopen(self.url.rstrip("/") + path, timeout=15) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers or {})

    def get_json(self, path: str) -> Dict[str, Any]:
        status, body, _ = self.get(path)
        assert status == 200, "{} -> HTTP {}".format(path, status)
        return json.loads(body.decode("utf-8"))

    def raw_get(self, raw_path: str) -> int:
        """GET a path with NO client-side normalisation, so ``..`` reaches the server verbatim.

        ``urllib`` collapses ``../`` before sending, which would make a traversal test pass
        without the server ever being asked the dangerous question.
        """
        host, _, port = self.url[len("http://") :].rstrip("/").partition(":")
        connection = HTTPConnection(host, int(port), timeout=15)
        try:
            connection.putrequest("GET", raw_path, skip_host=False, skip_accept_encoding=True)
            connection.endheaders()
            return connection.getresponse().status
        finally:
            connection.close()


# ------------------------------------------------------------------------------------------
# manifest
# ------------------------------------------------------------------------------------------


def test_manifest_has_expected_keys() -> None:
    out_dir, build_id = _find_build_root()
    with _Server(out_dir) as server:
        manifest = server.get_json("/api/manifest?build={}".format(build_id))

    for key in (
        "schema", "project", "title", "rev", "generated", "ok", "gates", "gates_summary",
        "parts", "enclosure", "components", "stages", "totals", "material", "bed",
        "files", "warnings", "build_id",
    ):
        assert key in manifest, "manifest is missing {!r}".format(key)

    assert manifest["parts"], "manifest lists no parts"
    for part in manifest["parts"]:
        for key in ("name", "stl", "color", "volume_mm3", "size_mm", "mass_g", "triangles"):
            assert key in part, "part {} missing {!r}".format(part.get("name"), key)
        assert part["stl"].startswith("/files/"), "stl url must be under /files/"
        assert part["size_mm"] is None or len(part["size_mm"]) == 3

    summary = manifest["gates_summary"]
    assert summary["total"] == len(manifest["gates"])
    assert summary["total"] >= 1, "expected at least one gate"


def test_manifest_gate_shape_is_tristate() -> None:
    """`passed` must stay True/False/None — a skipped gate is not a failed gate."""
    out_dir, build_id = _find_build_root()
    with _Server(out_dir) as server:
        manifest = server.get_json("/api/manifest?build={}".format(build_id))

    for gate in manifest["gates"]:
        for key in ("name", "passed", "value", "threshold", "message"):
            assert key in gate, "gate {} missing {!r}".format(gate.get("name"), key)
        assert gate["passed"] in (True, False, None), "passed must be tri-state"
    assert manifest["gates_summary"]["skipped"] == len(
        [g for g in manifest["gates"] if g["passed"] is None]
    )


def test_manifest_totals_are_numeric_and_consistent() -> None:
    out_dir, build_id = _find_build_root()
    with _Server(out_dir) as server:
        manifest = server.get_json("/api/manifest?build={}".format(build_id))

    totals = manifest["totals"]
    parts = manifest["parts"]
    assert totals["parts"] == len(parts)
    volume = sum(p["volume_mm3"] for p in parts if p["volume_mm3"] is not None)
    if volume:
        assert abs(totals["volume_mm3"] - volume) < 1.0, "totals.volume_mm3 disagrees with parts"
        density = manifest["material"]["density_g_cm3"]
        expected_mass = volume / 1000.0 * density
        assert abs(totals["mass_g"] - expected_mass) < 0.5, "mass must be volume x density"
    assert manifest["bed"]["size_mm"] == [360.0, 360.0], "bed must be the 360x360 Max4 plate"


def test_manifest_files_always_carry_the_drawing_keys() -> None:
    """The UI contract: every files key is present; drawing_* are URLs only when the file exists."""
    import shutil
    import tempfile

    out_dir, build_id = _find_build_root()
    with _Server(out_dir) as server:
        manifest = server.get_json("/api/manifest?build={}".format(build_id))
    files = manifest["files"]
    for key in ("report_json", "report_md", "design_json",
                "drawing_svg", "drawing_pdf", "drawing_png", "drawing_dxf"):
        assert key in files, "files is missing {!r}".format(key)
    for key in ("drawing_svg", "drawing_pdf", "drawing_png", "drawing_dxf"):
        assert files[key] is None or files[key].startswith("/files/")

    # Drop a drawing beside the report of a copy and it must be advertised and served.
    with tempfile.TemporaryDirectory(prefix="phimech-viewer-drawing-") as directory:
        copy = Path(directory) / "with-drawing"
        shutil.copytree(out_dir / build_id if (out_dir / build_id).is_dir() else out_dir, copy)
        (copy / "drawing.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        (copy / "drawing.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
        with _Server(Path(directory)) as server:
            manifest = server.get_json("/api/manifest?build=with-drawing")
            files = manifest["files"]
            assert files["drawing_svg"] == "/files/with-drawing/drawing.svg", files
            assert files["drawing_pdf"] == "/files/with-drawing/drawing.pdf", files
            assert files["drawing_png"] is None and files["drawing_dxf"] is None
            status, body, _ = server.get(files["drawing_pdf"])
            assert status == 200 and body.startswith(b"%PDF"), status


def test_manifest_tolerates_missing_report() -> None:
    """An empty directory must yield a 200 with warnings, never a 500."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="phimech-viewer-empty-") as directory:
        empty = Path(directory)
        with _Server(empty) as server:
            status, body, _ = server.get("/api/manifest")
        assert status == 200, "empty out_dir must not error, got HTTP {}".format(status)
        manifest = json.loads(body.decode("utf-8"))
        assert manifest["warnings"], "a missing report must be reported in warnings"
        assert manifest["parts"] == []
        assert manifest["ok"] is None


def test_manifest_tolerates_corrupt_report() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="phimech-viewer-bad-") as directory:
        build = Path(directory)
        (build / "report.json").write_text("{not json at all", encoding="utf-8")
        manifest = build_manifest(build)
    assert manifest["parts"] == []
    assert any("unreadable" in w for w in manifest["warnings"]), manifest["warnings"]


def test_unknown_build_falls_back_with_warning() -> None:
    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        manifest = server.get_json("/api/manifest?build=does-not-exist")
    assert any("unknown build" in w for w in manifest["warnings"]), manifest["warnings"]


# ------------------------------------------------------------------------------------------
# builds
# ------------------------------------------------------------------------------------------


def test_builds_lists_the_build() -> None:
    out_dir, build_id = _find_build_root()
    with _Server(out_dir) as server:
        payload = server.get_json("/api/builds")

    assert payload["count"] == len(payload["builds"])
    ids = [b["id"] for b in payload["builds"]]
    assert build_id in ids, "expected {!r} in {}".format(build_id, ids)
    for build in payload["builds"]:
        for key in ("id", "name", "title", "ok", "part_count", "path"):
            assert key in build, "build entry missing {!r}".format(key)
        assert build["ok"] in (True, False, None)


def test_single_build_dir_is_also_served() -> None:
    """Pointing the viewer straight at one build dir must work, not just at a parent."""
    out_dir, build_id = _find_build_root()
    build_dir = out_dir / build_id
    with _Server(build_dir) as server:
        payload = server.get_json("/api/builds")
        manifest = server.get_json("/api/manifest")
    assert payload["count"] >= 1, "a build dir must report itself"
    assert manifest["parts"], "manifest from a direct build dir must still list parts"


# ------------------------------------------------------------------------------------------
# static files
# ------------------------------------------------------------------------------------------


def test_stl_is_served_with_sane_type_and_length() -> None:
    out_dir, build_id = _find_build_root()
    with _Server(out_dir) as server:
        manifest = server.get_json("/api/manifest?build={}".format(build_id))
        url = manifest["parts"][0]["stl"]
        status, body, headers = server.get(url)

    assert status == 200, "{} -> HTTP {}".format(url, status)
    assert len(body) > 1000, "STL suspiciously small: {} bytes".format(len(body))
    assert headers.get("Content-Type") == "model/stl", headers.get("Content-Type")
    assert int(headers["Content-Length"]) == len(body)

    # OpenSCAD emits ASCII STL; FreeCAD emits binary. Accept either, but insist the payload is
    # a real mesh with a positive facet count rather than a truncated or empty file.
    if body[:5].lower().lstrip().startswith(b"solid") and b"facet normal" in body[:4096]:
        facets = body.count(b"facet normal")
        assert facets > 0, "ASCII STL contains no facets"
        assert b"endsolid" in body[-200:], "ASCII STL is truncated (no endsolid)"
    else:
        count = int.from_bytes(body[80:84], "little")
        assert count > 0, "binary STL header reports zero triangles"
        assert len(body) == 84 + 50 * count, "not a well-formed binary STL"


def test_path_traversal_is_refused() -> None:
    """`../../etc/passwd` must NOT return the file. Sent raw, bypassing urllib normalisation."""
    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        for attack in (
            "/files/../../etc/passwd",
            "/files/../../../../../../etc/passwd",
            "/files/..%2f..%2f..%2fetc%2fpasswd",
            "/files/%2e%2e/%2e%2e/etc/passwd",
            "/files//etc/passwd",
            "/files/subdir/../../../etc/passwd",
        ):
            status = server.raw_get(attack)
            assert status in (403, 404), "{} returned HTTP {}, expected 403/404".format(
                attack, status
            )

        # And prove the guard is not simply refusing everything: a real file still serves.
        status, body, _ = server.get("/api/health")
        assert status == 200 and b"ok" in body


def test_traversal_does_not_leak_passwd_contents() -> None:
    """Belt and braces: whatever the status code, the body must never be /etc/passwd."""
    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        host_port = server.url[len("http://") :].rstrip("/")
        host, _, port = host_port.partition(":")
        connection = HTTPConnection(host, int(port), timeout=15)
        try:
            connection.putrequest(
                "GET", "/files/../../../../etc/passwd", skip_host=False,
                skip_accept_encoding=True,
            )
            connection.endheaders()
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
    assert b"root:" not in body, "path traversal leaked /etc/passwd"


def test_missing_file_404s() -> None:
    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        status, _, _ = server.get("/files/definitely-not-here.stl")
        assert status == 404, "expected 404, got {}".format(status)
        status, _, _ = server.get("/no-such-asset.js")
        assert status == 404, "expected 404 for a missing asset, got {}".format(status)


def test_png_and_json_get_correct_mime() -> None:
    out_dir, build_id = _find_build_root()
    build_dir = out_dir / build_id
    with _Server(out_dir) as server:
        if (build_dir / "report.json").is_file():
            _, _, headers = server.get("/files/{}/report.json".format(build_id))
            assert headers.get("Content-Type", "").startswith("application/json"), headers
        pngs = sorted(build_dir.glob("*.png"))
        if pngs:
            _, body, headers = server.get("/files/{}/{}".format(build_id, pngs[0].name))
            assert headers.get("Content-Type") == "image/png", headers.get("Content-Type")
            assert body[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG payload"


# ------------------------------------------------------------------------------------------
# asset bundle
# ------------------------------------------------------------------------------------------


def test_index_html_carries_the_import_map() -> None:
    """Regression for the bare-'three' trap.

    OrbitControls.js and STLLoader.js both `import ... from 'three'`. Without an import map the
    browser cannot resolve that specifier and the viewer dies silently with a black canvas.
    """
    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        status, body, headers = server.get("/viewer")
    assert status == 200
    html = body.decode("utf-8")
    assert headers.get("Content-Type", "").startswith("text/html"), headers
    assert 'type="importmap"' in html, "index.html must declare an import map"
    assert '"three"' in html, "the import map must define the bare 'three' specifier"
    assert "vendor/three.module.js" in html, "'three' must map to the vendored module"

    map_at = html.index('type="importmap"')
    first_module = html.index('type="module"')
    assert map_at < first_module, "the import map must precede every module script"


def test_served_html_and_js_reference_no_cdn() -> None:
    """The viewer must work offline: nothing may be fetched from a third-party host."""
    out_dir, _ = _find_build_root()
    banned = (
        "unpkg.com", "cdn.jsdelivr", "jsdelivr.net", "cdnjs", "cdn.skypack",
        "esm.sh", "googleapis.com", "https://threejs.org", "//cdn.",
    )
    with _Server(out_dir) as server:
        for path in ("/", "/app.js", "/style.css"):
            status, body, _ = server.get(path)
            assert status == 200, "{} -> HTTP {}".format(path, status)
            text = body.decode("utf-8", "replace").lower()
            for host in banned:
                assert host.lower() not in text, "{} references CDN host {!r}".format(path, host)


def test_vendor_modules_are_present_and_served() -> None:
    for name in ("three.module.js", "OrbitControls.js", "STLLoader.js"):
        path = ASSETS_DIR / "vendor" / name
        assert path.is_file(), "missing vendored module: {}".format(path)

    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        for name in ("three.module.js", "OrbitControls.js", "STLLoader.js"):
            status, body, headers = server.get("/vendor/{}".format(name))
            assert status == 200, "/vendor/{} -> HTTP {}".format(name, status)
            assert len(body) > 1000, "{} is suspiciously small".format(name)
            assert "javascript" in headers.get("Content-Type", ""), headers.get("Content-Type")


def test_app_js_exposes_the_agent_contract() -> None:
    """`window.__phimech` is how an agent (and a Forca eval) verifies viewer state."""
    source = (ASSETS_DIR / "app.js").read_text(encoding="utf-8")
    assert "window.__phimech" in source
    for key in ("ready", "parts", "tris", "error"):
        assert key in source, "the __phimech contract must expose {!r}".format(key)
    # Z-up is not optional for CAD geometry.
    assert "camera.up.set(0, 0, 1)" in source, "the camera must be Z-up for CAD coordinates"


def test_assets_do_not_escape_via_traversal() -> None:
    out_dir, _ = _find_build_root()
    with _Server(out_dir) as server:
        for attack in ("/../../etc/passwd", "/..%2f..%2fetc%2fpasswd"):
            status = server.raw_get(attack)
            assert status in (403, 404), "{} returned {}".format(attack, status)


# ------------------------------------------------------------------------------------------
# units
# ------------------------------------------------------------------------------------------


def test_stl_triangle_count_matches_header() -> None:
    out_dir, build_id = _find_build_root()
    stls = sorted((out_dir / build_id).glob("*.stl"))
    if not stls:
        _skip("no STL in the chosen build")
    path = stls[0]
    counted = stl_triangle_count(path)
    assert counted is not None, "could not count triangles in {}".format(path.name)
    assert counted > 0

    data = path.read_bytes()
    if data[:5].lower().lstrip().startswith(b"solid") and b"facet normal" in data[:4096]:
        expected = data.count(b"facet normal")  # ASCII, as produced by OpenSCAD
    else:
        expected = (len(data) - 84) // 50  # binary, as produced by FreeCAD
    assert counted == expected, "counted {} triangles, file implies {}".format(counted, expected)


def test_discover_builds_ignores_non_builds() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="phimech-viewer-mixed-") as directory:
        root = Path(directory)
        (root / "notabuild").mkdir()
        (root / "notabuild" / "readme.txt").write_text("hi", encoding="utf-8")
        real = root / "realbuild"
        real.mkdir()
        (real / "report.json").write_text(json.dumps({"project": "realbuild"}), encoding="utf-8")
        builds = discover_builds(root)
    ids = [b["id"] for b in builds]
    assert "realbuild" in ids
    assert "notabuild" not in ids


def test_serve_picks_a_free_port_when_busy() -> None:
    """Two viewers must coexist; the returned URL must name the port actually bound."""
    out_dir, _ = _find_build_root()
    first, first_url = serve(out_dir, port=0, quiet=True)
    try:
        busy_port = first.server_address[1]
        second, second_url = serve(out_dir, port=busy_port, quiet=True)
        try:
            assert second.server_address[1] != busy_port, "should not rebind the busy port"
            assert str(second.server_address[1]) in second_url, second_url
            assert str(busy_port) in first_url
        finally:
            second.server_close()
    finally:
        first.server_close()


# -- the committed fixture -----------------------------------------------------------------
#
# These tests guard the fixture itself, not the viewer. It exists so a clone with no CAD backends
# can still render something and run every geometry assertion, and each property below is load
# bearing for that job.


def test_fixture_is_committed_and_complete() -> None:
    """The fixture must carry everything the manifest builder reads."""
    assert FIXTURE_BUILD.is_dir(), "fixtures/viewer is missing"
    for required in ("report.json", "design.json", "REPORT.md"):
        assert (FIXTURE_BUILD / required).is_file(), "fixture is missing {}".format(required)
    stls = sorted(FIXTURE_BUILD.glob("*.stl"))
    assert len(stls) >= 2, "need >=2 parts to exercise assembly centring, found {}".format(stls)
    assert sorted(FIXTURE_BUILD.glob("*.png")), "need a PNG so previews[] is not all null"


def test_fixture_stls_are_binary_and_within_budget() -> None:
    """Binary STL, and small enough that the fixture cannot quietly bloat the repo.

    The same geometry as ASCII is ~5x larger. A size ceiling is the only thing that stops a
    future "nicer" fixture from adding megabytes to every clone.
    """
    total = 0
    for stl in sorted(FIXTURE_BUILD.glob("*.stl")):
        head = stl.read_bytes()[:5]
        assert head != b"solid", "{} is ASCII STL; commit binary".format(stl.name)
        size = stl.stat().st_size
        count = stl_triangle_count(stl)
        assert count > 0, "{} has no triangles".format(stl.name)
        # Binary STL is a fixed 84-byte header plus 50 bytes per triangle, exactly.
        assert size == 84 + 50 * count, "{}: {} bytes for {} triangles".format(
            stl.name, size, count)
        total += size
    budget = 1_000_000
    assert total <= budget, "fixture STLs total {} bytes, budget {}".format(total, budget)


def test_fixture_report_has_no_absolute_paths() -> None:
    """A committed fixture must not leak the authoring machine's layout."""
    blob = (FIXTURE_BUILD / "report.json").read_text()
    for needle in ("/Users/", "/home/", "/private/", "/tmp/", "C:\\"):
        assert needle not in blob, "fixture report.json leaks {}".format(needle)


def test_fixture_exercises_all_three_gate_states() -> None:
    """pass + fail + skip must all appear, because the UI renders each differently.

    No bundled example produces a failing gate, so without this fixture a UI could never be shown
    against a red gate. If someone "fixes" the fixture's failing gate, the UI loses its only
    example of the state it most needs to get right - hence this assertion.
    """
    # build_manifest's second argument is a `/files/` URL prefix, not a build id: pass the build
    # directory itself. (The HTTP layer is what resolves `?build=<subdir>`.)
    manifest = build_manifest(FIXTURE_BUILD)
    states = {}
    for gate in manifest["gates"]:
        states.setdefault(gate["passed"], []).append(gate["name"])
    assert True in states, "fixture has no passing gate"
    assert False in states, "fixture has no FAILING gate; the UI needs one to render red"
    assert None in states, "fixture has no SKIPPED gate; skipped must not render as passed"


def test_fixture_failing_gate_does_not_flip_the_verdict() -> None:
    """A warn-severity gate can fail while ``ok`` stays true - the UI must not conflate them.

    ``ok`` is the build verdict (stages + errors + *error*-severity gates), not a gate tally. The
    fixture is deliberately in this state so a UI that renders "ok" as "all gates passed" is caught.
    """
    manifest = build_manifest(FIXTURE_BUILD)
    failed = [g for g in manifest["gates"] if g["passed"] is False]
    assert failed, "fixture should have a failing gate"
    assert all(g["severity"] != "error" for g in failed), (
        "fixture's failing gate(s) must be warn-severity: {}".format(
            [(g["name"], g["severity"]) for g in failed]))
    assert manifest["ok"] is True, "warn-severity failures must not flip ok to false"
    assert manifest["gates_summary"]["failed"] == len(failed)


def _run_direct() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(test.__name__, exc))
        except Exception as exc:  # pragma: no cover - only used by direct invocation
            failures += 1
            print("FAIL {}: {}".format(test.__name__, exc))
        else:
            print("PASS {}".format(test.__name__))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
