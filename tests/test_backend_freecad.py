"""FreeCAD backend tests — analytic volumes, not golden bytes.

Every geometric assertion here compares against a closed-form volume/area, because that is
what makes the B-rep claim ("true solid volume, not a tessellation") checkable. Mesh exports
are the only place tolerances get loose, and only for the mesh, never for the shape.

Runs under pytest and, for the coordinator's benefit, directly with the system python 3.9:
`/usr/bin/python3 tests/test_backend_freecad.py`.
"""

from __future__ import annotations

import os as _os

import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:  # pragma: no cover - direct system-python runs
    pytest = None  # type: ignore[assignment]

from phimech.backends.base import BackendError
from phimech.backends.freecad import FreeCADBackend, _parse_output
from phimech.csg.ast import (
    Circle,
    Color,
    Cube,
    Cylinder,
    Difference,
    Group,
    Hull,
    Import,
    Intersection,
    LinearExtrude,
    Minkowski,
    Mirror,
    Polygon,
    Polyhedron,
    Rotate,
    RotateExtrude,
    Scale,
    Sphere,
    Square,
    Translate,
    Union,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "freecad"
TIMINGS = []  # type: list


class _Skipped(Exception):
    pass


def _skip(message: str) -> None:
    # pytest.skip() raises Skipped, which derives from BaseException, so it sails through the
    # `except Exception` in _run_direct and turns a skip into a crash with a traceback and exit 1.
    # Delegating merely because pytest is IMPORTABLE was the bug: `bin/phimech check` runs each test
    # file directly with the venv python, where pytest is installed but is not driving. pytest sets
    # PYTEST_CURRENT_TEST for the duration of each test, so it tells "installed" from "running".
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _require_freecad() -> FreeCADBackend:
    backend = FreeCADBackend()
    if backend.available():
        return backend
    _skip("freecadcmd is not installed")
    raise AssertionError("unreachable")


def _timed(label, fn):
    started = time.time()
    out = fn()
    elapsed = time.time() - started
    TIMINGS.append((label, elapsed))
    print("    [timing] {}: {:.2f}s".format(label, elapsed))
    return out


def _close(actual: float, expected: float, tol: float, what: str) -> None:
    assert abs(actual - expected) <= tol, "{}: got {!r}, expected {!r} +/- {}".format(
        what, actual, expected, tol)


# --------------------------------------------------------------------------------------
# discovery / plumbing (no subprocess)
# --------------------------------------------------------------------------------------


def test_backend_metadata() -> None:
    backend = FreeCADBackend()
    assert backend.name == "freecad"
    assert "step" in backend.supported_formats
    assert "png" not in backend.supported_formats
    assert "minkowski" in backend.UNSUPPORTED


def test_registered_in_registry() -> None:
    from phimech.backends import base

    backend = base.get("freecad")
    assert isinstance(backend, FreeCADBackend)


def test_can_render_routes_unsupported_nodes() -> None:
    backend = FreeCADBackend()
    assert backend.can_render(Cube((1, 2, 3))) is None
    assert backend.can_render(Hull([Cube(), Sphere()])) is None  # scipy hull is supported
    assert "minkowski" in (backend.can_render(Minkowski([Cube(), Sphere()])) or "")
    twisted = LinearExtrude(10, [Square((5, 5))], twist=90)
    assert "twist" in (backend.can_render(twisted) or "")
    assert backend.can_render(Union([Cube(), Translate((5, 0, 0), [Minkowski([Cube()])])]))


def test_parse_output_ignores_freecad_noise() -> None:
    noisy = "\n".join([
        "FreeCAD 1.1.3, Libs: 1.1.3R20260725 (Git shallow)",
        "saving......\t(5 %)\t(100 %)",
        'PHIMECH_RESULT {"outputs": {}, "measurement": {"volume": 42.0}}',
    ])
    payload = _parse_output(noisy, "", 0, ["freecadcmd"])
    assert payload["measurement"]["volume"] == 42.0


def test_parse_output_surfaces_worker_error() -> None:
    err = 'PHIMECH_ERROR {"error": "cube size must be positive", "traceback": "T"}'
    try:
        _parse_output("banner\n" + err, "", 1, ["freecadcmd"])
    except BackendError as exc:
        assert "cube size must be positive" in str(exc)
    else:
        raise AssertionError("expected BackendError")


def test_parse_output_without_marker_is_an_error() -> None:
    try:
        _parse_output("banner only", "segfault", 139, ["freecadcmd"])
    except BackendError as exc:
        assert "no result marker" in str(exc)
        assert exc.returncode == 139
    else:
        raise AssertionError("expected BackendError")


def test_unsupported_format_rejected_before_launch() -> None:
    backend = FreeCADBackend()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            backend.render_many(Cube(), {"png": Path(tmp) / "x.png"})
        except BackendError as exc:
            assert "cannot write .png" in str(exc)
        else:
            raise AssertionError("expected BackendError")


# --------------------------------------------------------------------------------------
# version / availability
# --------------------------------------------------------------------------------------


def test_version_and_doctor() -> None:
    backend = _require_freecad()
    version = _timed("version()", backend.version)
    assert version and version[0].isdigit(), version
    info = backend.doctor()
    assert info["available"] is True
    assert info["backend"] == "freecad"
    assert Path(str(info["worker"])).is_file()
    print("    freecad {} via {} ({})".format(version, info["binary_source"], info["binary"]))


# --------------------------------------------------------------------------------------
# geometry: analytic checks
# --------------------------------------------------------------------------------------


def test_measure_cube_exact() -> None:
    backend = _require_freecad()
    m = _timed("measure(cube 10x20x30)", lambda: backend.measure(Cube((10, 20, 30))))
    assert m.bbox.min == (0.0, 0.0, 0.0)
    assert m.bbox.max == (10.0, 20.0, 30.0)
    _close(m.volume, 6000.0, 1e-3, "cube volume")
    _close(m.area, 2200.0, 1e-6, "cube area")
    assert m.solids == 1


def test_render_step_and_measure_in_one_pass() -> None:
    backend = _require_freecad()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cube.step"
        result = _timed("render(step)", lambda: backend.render(Cube((10, 20, 30)), out))
        assert out.exists()
        assert result.bytes > 1024, "STEP should be > 1KB, got {}".format(result.bytes)
        assert result.fmt == "step" and result.backend == "freecad"
        assert result.measurement is not None, "render() must fill in the measurement"
        _close(result.measurement.volume, 6000.0, 1e-3, "step volume")
        assert result.measurement.bbox.max == (10.0, 20.0, 30.0)
        assert "ISO-10303" in out.read_text(errors="replace")[:400]


def test_cube_minus_cylinder_through_hole_is_analytic() -> None:
    """The headline B-rep claim: a real pi*r^2*h hole, not a 32-gon's worth of it."""
    backend = _require_freecad()
    node = Difference([
        Cube((30, 20, 10)),
        Translate((15, 10, -1), [Cylinder(h=12, r=2)]),  # over-cut, with a small epsilon to clear both faces
    ])
    m = _timed("measure(cube - through hole)", lambda: backend.measure(node))
    expected = 30 * 20 * 10 - math.pi * 4 * 10
    _close(m.volume, expected, 0.5, "cube minus through-hole volume")
    assert m.solids == 1
    assert m.bbox.min == (0.0, 0.0, 0.0) and m.bbox.max == (30.0, 20.0, 10.0)


def test_cylinder_volume_is_analytic_not_faceted() -> None:
    backend = _require_freecad()
    m = _timed("measure(cylinder r5 h10)", lambda: backend.measure(Cylinder(h=10, r=5, segments=8)))
    _close(m.volume, math.pi * 25 * 10, 1e-6, "cylinder volume (segments must be ignored)")


def test_cone_and_centered_cylinder() -> None:
    backend = _require_freecad()
    cone = Cylinder(h=10, r=5, r2=0, center=True)
    m = _timed("measure(cone centered)", lambda: backend.measure(cone))
    _close(m.volume, math.pi * 25 * 10 / 3.0, 1e-6, "cone volume")
    _close(m.bbox.min[2], -5.0, 1e-9, "centered cylinder ZMin")
    _close(m.bbox.max[2], 5.0, 1e-9, "centered cylinder ZMax")


def test_rotate_extrude_matches_analytic_torus() -> None:
    """Pins the OpenSCAD revolve convention: profile-x -> radius, profile-y -> world z."""
    backend = _require_freecad()
    node = RotateExtrude([Translate((10, 0, 0), [Circle(r=2)])])
    m = _timed("measure(torus via rotate_extrude)", lambda: backend.measure(node))
    expected = 2 * math.pi**2 * 10 * 2**2  # 789.5684
    _close(m.volume, expected, 1e-3, "torus volume")
    _close(m.bbox.max[0], 12.0, 1e-6, "torus outer radius")
    _close(m.bbox.max[2], 2.0, 1e-6, "torus half-height (profile y became z)")
    _close(m.bbox.min[2], -2.0, 1e-6, "torus half-height (profile y became z)")


def test_bbox_is_tight_not_occ_conservative() -> None:
    """Regression: `Shape.BoundBox` reports 12.9887 for a torus of outer radius 12, because OCC
    bounds the control polygon. The fit gate compares bbox to bed size, so the worker must use
    `optimalBoundingBox()`. This is the test that would catch a revert."""
    backend = _require_freecad()
    m = _timed("measure(torus bbox tightness)",
               lambda: backend.measure(RotateExtrude([Translate((10, 0, 0), [Circle(r=2)])])))
    _close(m.bbox.max[0], 12.0, 1e-6, "tight torus XMax")
    _close(m.bbox.min[1], -12.0, 1e-6, "tight torus YMin")
    assert m.bbox.max[0] < 12.9, "got OCC's conservative bbox ({}), not the optimal one".format(
        m.bbox.max[0])


def test_mesh_export_does_not_contaminate_the_measured_bbox() -> None:
    """Regression: `MeshPart.meshFromShape` caches a triangulation on the shape, after which
    `optimalBoundingBox()` measures the *facets* — reporting ZMax 12.00497 for an exactly-12
    box. The worker must measure before exporting (and clamp to the conservative box)."""
    backend = _require_freecad()
    node = Difference([Cube((40, 30, 12)), Translate((20, 15, -1), [Cylinder(h=14, r=4)])])
    with tempfile.TemporaryDirectory() as tmp:
        results = _timed("render_many(mesh+step, bbox purity)", lambda: backend.render_many(
            node, {"stl": Path(tmp) / "m.stl", "step": Path(tmp) / "m.step"}))
    m = results["step"].measurement
    assert m is not None
    for axis, want in enumerate((40.0, 30.0, 12.0)):
        _close(m.bbox.max[axis], want, 1e-9, "exact bbox max axis {}".format(axis))
        _close(m.bbox.min[axis], 0.0, 1e-9, "exact bbox min axis {}".format(axis))
    _close(m.volume, 40 * 30 * 12 - math.pi * 16 * 12, 1e-6, "volume beside a mesh export")


def test_text_group_becomes_solid_geometry() -> None:
    """An empty Group named `text:...` is the AST's escape hatch for lid labels."""
    backend = _require_freecad()
    m = _timed("measure(text group)", lambda: backend.measure(Group([], name="text:8:1.5:PHI")))
    if m.volume == 0.0:
        _skip("no usable system font for makeWireString")
    assert m.volume > 0.0 and m.solids >= 1
    _close(m.bbox.max[2] - m.bbox.min[2], 1.5, 1e-6, "text extrusion depth")
    assert 2.0 < m.bbox.max[0] < 60.0, "three glyphs at size 8 should be a few mm wide"


def test_rotate_extrude_partial_angle() -> None:
    backend = _require_freecad()
    node = RotateExtrude([Translate((10, 0, 0), [Circle(r=2)])], angle=90)
    m = _timed("measure(quarter torus)", lambda: backend.measure(node))
    _close(m.volume, 2 * math.pi**2 * 10 * 4 / 4.0, 1e-3, "quarter torus volume")


def test_polygon_with_hole_extruded() -> None:
    backend = _require_freecad()
    node = LinearExtrude(3, [Polygon(
        points=((0, 0), (20, 0), (20, 20), (0, 20), (5, 5), (15, 5), (15, 15), (5, 15)),
        paths=((0, 1, 2, 3), (4, 5, 6, 7)),
    )])
    m = _timed("measure(polygon with hole, extruded)", lambda: backend.measure(node))
    _close(m.volume, (400 - 100) * 3, 1e-6, "frame volume")
    # two 20x20 faces minus the window, plus outer and inner side walls
    _close(m.area, 2 * 300 + 4 * 20 * 3 + 4 * 10 * 3, 1e-6, "frame area")
    assert m.solids == 1


def test_linear_extrude_centered_and_scaled() -> None:
    backend = _require_freecad()
    centered = _timed("measure(linear_extrude centered)",
                      lambda: backend.measure(LinearExtrude(10, [Square((4, 6))], center=True)))
    _close(centered.volume, 240.0, 1e-9, "centered extrude volume")
    _close(centered.bbox.min[2], -5.0, 1e-9, "centered extrude ZMin")
    # frustum via loft: h/3 * (A1 + A2 + sqrt(A1*A2)); 10x10 -> 5x5 over h=6
    tapered = _timed("measure(linear_extrude scale=0.5 -> loft)",
                     lambda: backend.measure(LinearExtrude(6, [Square((10, 10))], scale=0.5)))
    expected = 6 / 3.0 * (100 + 25 + math.sqrt(100 * 25))
    _close(tapered.volume, expected, 1e-6, "tapered extrude (frustum) volume")


def test_polyhedron_tetrahedron() -> None:
    backend = _require_freecad()
    node = Polyhedron(
        points=((0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)),
        faces=((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)),
    )
    m = _timed("measure(polyhedron tetra)", lambda: backend.measure(node))
    _close(m.volume, 1000 / 6.0, 1e-6, "tetrahedron volume")
    assert m.solids == 1


def test_hull_of_two_boxes_is_the_convex_span() -> None:
    """FreeCAD 1.1.3 has no Part.makeConvexHull; the worker uses the bundled scipy instead."""
    backend = _require_freecad()
    node = Hull([Cube((10, 10, 10)), Translate((30, 0, 0), [Cube((10, 10, 10))])])
    m = _timed("measure(hull of 2 boxes)", lambda: backend.measure(node))
    _close(m.volume, 40 * 10 * 10, 1e-3, "hull volume")
    assert m.bbox.min == (0.0, 0.0, 0.0) and m.bbox.max == (40.0, 10.0, 10.0)


def test_transforms_translate_rotate_scale_mirror() -> None:
    backend = _require_freecad()
    node = Union([
        Translate((100, 0, 0), [Cube((2, 2, 2))]),
        Rotate((0, 0, 90), [Translate((10, 0, 0), [Cube((4, 2, 2))])]),
        Scale((2, 3, 4), [Cube((1, 1, 1))]),
        Mirror((1, 0, 0), [Translate((50, 0, 0), [Cube((2, 2, 2))])]),
    ])
    m = _timed("measure(transform battery)", lambda: backend.measure(node))
    _close(m.volume, 8 + 16 + 24 + 8, 1e-6, "disjoint transform volumes")
    assert m.solids == 4, "four disjoint bodies expected, got {}".format(m.solids)
    _close(m.bbox.max[0], 102.0, 1e-6, "translate reached x=102")
    _close(m.bbox.min[0], -52.0, 1e-6, "mirror reached x=-52")
    _close(m.bbox.max[1], 14.0, 1e-6, "rotate([0,0,90]) put the bar on +Y")


def test_intersection_and_color_and_group_passthrough() -> None:
    backend = _require_freecad()
    node = Group([Color((1, 0, 0, 1), [Intersection([
        Cube((10, 10, 10)),
        Translate((5, 5, 5), [Cube((10, 10, 10))]),
    ])])], name="base")
    m = _timed("measure(intersection in color+group)", lambda: backend.measure(node))
    _close(m.volume, 125.0, 1e-9, "intersection volume")


def test_empty_group_is_tolerated() -> None:
    backend = _require_freecad()
    m = _timed("measure(empty group)", lambda: backend.measure(Group([], name="nothing")))
    assert m.volume == 0.0 and m.solids == 0


def test_sphere_volume() -> None:
    backend = _require_freecad()
    m = _timed("measure(sphere r7)", lambda: backend.measure(Sphere(r=7)))
    _close(m.volume, 4 / 3.0 * math.pi * 343, 1e-6, "sphere volume")
    _close(m.area, 4 * math.pi * 49, 1e-6, "sphere area")


# --------------------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------------------


def test_render_many_writes_step_and_stl_in_one_launch() -> None:
    backend = _require_freecad()
    node = Difference([Cube((30, 20, 10)), Translate((15, 10, -1), [Cylinder(h=12, r=3)])])
    with tempfile.TemporaryDirectory() as tmp:
        outputs = {
            "step": Path(tmp) / "part.step",
            "stl": Path(tmp) / "part.stl",
            "3mf": Path(tmp) / "part.3mf",
        }
        results = _timed("render_many(step+stl+3mf)", lambda: backend.render_many(node, outputs))
        assert set(results) == {"step", "stl", "3mf"}
        for fmt, result in sorted(results.items()):
            assert result.path.exists() and result.bytes > 0
            print("      {} -> {} bytes in {:.2f}s".format(fmt, result.bytes, result.seconds))
        expected = 30 * 20 * 10 - math.pi * 9 * 10
        _close(results["step"].measurement.volume, expected, 0.5, "render_many volume")
        # one launch: the whole call must cost about one startup, not three
        assert results["step"].measurement is results["stl"].measurement


def test_stl_roundtrip_import_measures_within_one_percent() -> None:
    """Meshes a solid, then imports the mesh back through the `import` node and re-measures."""
    backend = _require_freecad()
    node = Cube((10, 20, 30))
    with tempfile.TemporaryDirectory() as tmp:
        stl = Path(tmp) / "cube.stl"
        _timed("render(stl)", lambda: backend.render(node, stl))
        assert stl.stat().st_size > 0
        m = _timed("measure(import stl)", lambda: backend.measure(Import(str(stl))))
        assert abs(m.volume - 6000.0) / 6000.0 < 0.01, "imported mesh volume {}".format(m.volume)
        assert m.bbox.max == (10.0, 20.0, 30.0)


def test_step_roundtrip_import() -> None:
    backend = _require_freecad()
    with tempfile.TemporaryDirectory() as tmp:
        step = Path(tmp) / "cyl.step"
        _timed("render(step for reimport)", lambda: backend.render(Cylinder(h=10, r=5), step))
        node = Union([Import(str(step)), Translate((20, 0, 0), [Cube((2, 2, 2))])])
        m = _timed("measure(import step)", lambda: backend.measure(node))
        _close(m.volume, math.pi * 25 * 10 + 8, 1e-3, "imported STEP + cube volume")


def test_fcstd_export_labels_top_level_groups() -> None:
    backend = _require_freecad()
    node = Group([
        Group([Cube((20, 20, 5))], name="base"),
        Group([Translate((0, 0, 10), [Cube((20, 20, 2))])], name="lid"),
    ], name="assembly")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "asm.FCStd"
        result = _timed("render(fcstd)", lambda: backend.render(node, out, "fcstd"))
        assert out.exists() and result.bytes > 512
        import zipfile

        with zipfile.ZipFile(out) as zf:
            document = zf.read("Document.xml").decode("utf-8", "replace")
        assert "base" in document and "lid" in document, "group names must survive as labels"


def test_obj_and_off_exports() -> None:
    backend = _require_freecad()
    with tempfile.TemporaryDirectory() as tmp:
        results = _timed("render_many(obj+off)", lambda: backend.render_many(
            Sphere(r=5), {"obj": Path(tmp) / "s.obj", "off": Path(tmp) / "s.off"}))
        assert results["obj"].path.read_text(errors="replace").startswith(("#", "v", "g"))
        assert results["off"].path.read_text(errors="replace").startswith("OFF")


# --------------------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------------------


def test_bad_kind_raises_backend_error() -> None:
    backend = _require_freecad()

    class Bogus(Cube):
        pass

    node = Bogus((1, 1, 1))
    object.__setattr__(node, "kind", "banana")
    try:
        _timed("measure(bad kind)", lambda: backend.measure(node))
    except BackendError as exc:
        assert "banana" in str(exc), str(exc)
        assert "unknown CSG node kind" in str(exc) or "unknown" in str(exc)
    else:
        raise AssertionError("expected BackendError for an unknown node kind")


def test_invalid_geometry_message_is_actionable() -> None:
    backend = _require_freecad()
    try:
        backend.measure(Cube((10, 0, 5)))
    except BackendError as exc:
        assert "must be > 0" in str(exc), str(exc)
    else:
        raise AssertionError("expected BackendError for a zero-size cube")


def test_zero_area_polygon_is_rejected_before_binary_lookup() -> None:
    node = LinearExtrude(1, [Polygon(((0, 0), (1, 0), (2, 0)))])
    backend = FreeCADBackend(binary="/nonexistent/freecadcmd")
    try:
        backend.measure(node)
    except BackendError as exc:
        assert str(exc) == (
            "invalid CSG AST: root.children[0] polygon outline encloses zero area"
        )
    else:
        raise AssertionError("expected fail-fast polygon validation")


def test_missing_import_file_raises() -> None:
    backend = _require_freecad()
    try:
        backend.measure(Import("/tmp/phimech-does-not-exist-4f2a.step"))
    except BackendError as exc:
        assert "does not exist" in str(exc).lower()
    else:
        raise AssertionError("expected BackendError for a missing import")


def test_minkowski_is_rejected_without_launching_freecad() -> None:
    backend = FreeCADBackend()
    started = time.time()
    try:
        backend.measure(Minkowski([Cube(), Sphere()]))
    except BackendError as exc:
        assert "minkowski" in str(exc)
        assert time.time() - started < 0.5, "should be rejected before any subprocess"
    else:
        raise AssertionError("expected BackendError for minkowski")


def test_missing_binary_is_a_clear_error() -> None:
    backend = FreeCADBackend(binary="/nonexistent/freecadcmd")
    assert backend.available() is False
    assert backend.version() is None
    try:
        backend.measure(Cube())
    except BackendError as exc:
        assert "freecadcmd not found" in str(exc)
    else:
        raise AssertionError("expected BackendError when the binary is missing")


# --------------------------------------------------------------------------------------
# golden
# --------------------------------------------------------------------------------------


def test_golden_cube_measurement() -> None:
    """Golden files hold measurements, never bytes: OCC tessellation and STEP headers move."""
    backend = _require_freecad()
    node_file = GOLDEN / "cube.json"
    golden_file = GOLDEN / "cube.measurement.json"
    if not node_file.exists():
        _skip("golden fixture missing: {}".format(node_file))
    from phimech.csg.ast import from_dict

    node = from_dict(json.loads(node_file.read_text()))
    m = _timed("measure(golden cube)", lambda: backend.measure(node))
    golden = json.loads(golden_file.read_text())
    _close(m.volume, golden["volume_mm3"], 1e-3, "golden volume")
    _close(m.area, golden["area_mm2"], 1e-3, "golden area")
    assert m.solids == golden["solids"]
    for got, want in zip(m.bbox.min, golden["bbox"]["min"]):
        _close(got, want, 1e-6, "golden bbox min")
    for got, want in zip(m.bbox.max, golden["bbox"]["max"]):
        _close(got, want, 1e-6, "golden bbox max")


def test_golden_bracket_measurement() -> None:
    backend = _require_freecad()
    node_file = GOLDEN / "bracket.json"
    golden_file = GOLDEN / "bracket.measurement.json"
    if not node_file.exists():
        _skip("golden fixture missing: {}".format(node_file))
    from phimech.csg.ast import from_dict

    node = from_dict(json.loads(node_file.read_text()))
    m = _timed("measure(golden bracket)", lambda: backend.measure(node))
    golden = json.loads(golden_file.read_text())
    _close(m.volume, golden["volume_mm3"], 0.05, "golden bracket volume")
    assert m.solids == golden["solids"]
    for got, want in zip(m.bbox.max, golden["bbox"]["max"]):
        _close(got, want, 1e-6, "golden bracket bbox max")


def _run_direct() -> int:
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    started = time.time()
    for name, test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(name, exc))
        except Exception as exc:  # pragma: no cover - direct invocation only
            failures += 1
            import traceback

            print("FAIL {}: {}".format(name, exc))
            traceback.print_exc()
        else:
            print("PASS {}".format(name))
    print("\n{} tests, {} failures, {:.1f}s wall".format(
        len(tests), failures, time.time() - started))
    if TIMINGS:
        slowest = sorted(TIMINGS, key=lambda pair: -pair[1])[:5]
        print("slowest freecad calls: " + ", ".join(
            "{} {:.2f}s".format(label, seconds) for label, seconds in slowest))
        print("total freecad subprocess time: {:.1f}s over {} launches".format(
            sum(s for _, s in TIMINGS), len(TIMINGS)))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
