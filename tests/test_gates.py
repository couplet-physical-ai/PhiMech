"""Mesh and DFM gate tests; runnable with pytest or /usr/bin/python3."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phimech.backends.base import BBox, Measurement
from phimech.check.gates import gate_catalog, run_gates
from phimech.check.mesh import (
    area,
    count_bodies,
    count_shells,
    is_watertight,
    min_wall_estimate,
    overhang_fraction,
    read_stl,
    volume,
)


def _close(actual, expected, tolerance=1.0e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def _gate(report, name):
    """One gate by name.

    Named rather than positional: these assertions used to index `report.results[8]`, which
    silently retargets a different gate the moment one is inserted, and gives a bare `False`
    when it breaks. By name, a missing gate is an explicit failure and the assertion always
    tests the gate it claims to.
    """
    for result in report.results:
        if result.name == name:
            return result
    raise AssertionError(
        "no gate named {!r}; have {}".format(name, [r.name for r in report.results])
    )


def cube(x0=0.0, y0=0.0, z0=0.0, size=1.0):
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    a = (x0, y0, z0)
    b = (x1, y0, z0)
    c = (x1, y1, z0)
    d = (x0, y1, z0)
    e = (x0, y0, z1)
    f = (x1, y0, z1)
    g = (x1, y1, z1)
    h = (x0, y1, z1)
    return [
        (a, c, b), (a, d, c),  # bottom, -Z
        (e, f, g), (e, g, h),  # top, +Z
        (a, b, f), (a, f, e),  # y=0, -Y
        (b, c, g), (b, g, f),  # x=1, +X
        (c, d, h), (c, h, g),  # y=1, +Y
        (d, a, e), (d, e, h),  # x=0, -X
    ]


def tetrahedron():
    a = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)
    d = (0.0, 0.0, 1.0)
    return [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]


def ascii_stl(triangles):
    lines = ["solid test"]
    for first, second, third in triangles:
        lines += [
            " facet normal 0 0 0",
            "  outer loop",
            "   vertex {} {} {}".format(*first),
            "   vertex {} {} {}".format(*second),
            "   vertex {} {} {}".format(*third),
            "  endloop",
            " endfacet",
        ]
    lines.append("endsolid test")
    return "\n".join(lines) + "\n"


def _rotated_x45(triangles):
    factor = math.sqrt(0.5)

    def rotate(point):
        x, y, z = point
        return (x, y * factor - z * factor, y * factor + z * factor)

    return [tuple(rotate(point) for point in triangle) for triangle in triangles]


def _measurement(volume_value=100.0, size=(10.0, 10.0, 10.0), solids=1, extra=None):
    return Measurement(
        bbox=BBox((0.0, 0.0, 0.0), tuple(float(value) for value in size)),
        volume=volume_value,
        area=60.0,
        solids=solids,
        extra=extra or {},
    )


def _objects():
    placement = SimpleNamespace(
        errors=[],
        keepouts=[SimpleNamespace(world_bbox=lambda: ((1.0, 1.0, 1.0), (9.0, 9.0, 9.0)))],
    )
    summary = {
        "standoffs": [{"od": 4.0, "hole": 2.0}],
    }
    parts = SimpleNamespace(
        inner_cavity=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
        wall=1.6,
        floor=1.6,
        cutouts=[
            {
                "face": "front",
                "rect": (2.0, 2.0, 4.0, 4.0),
                "face_rect": (0.0, 0.0, 10.0, 10.0),
            }
        ],
        summary=lambda: summary,
    )
    design = {
        "project": "gate-test",
        "_errors": [],
        "manufacturing": {
            "min_wall": 1.2,
            "min_feature": 0.8,
            "nozzle": 0.4,
            "max_overhang_deg": 45.0,
            "printer_bed": [20.0, 20.0, 20.0],
        },
        "enclosure": {
            "wall": 1.6,
            "floor": 1.6,
            "lid": {"fastening": "screws", "screw_count": 4},
            "text": {"size": 6.0},
        },
    }
    return design, placement, parts


def test_read_ascii_tetrahedron():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tetra.stl"
        path.write_text(ascii_stl(tetrahedron()))
        assert len(read_stl(path)) == 4
        _close(volume(read_stl(path)), 1.0 / 6.0)


def test_ascii_cube_metrics_and_watertightness():
    triangles = cube()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cube.stl"
        path.write_text(ascii_stl(triangles))
        parsed = read_stl(path)
    _close(volume(parsed), 1.0)
    _close(area(parsed), 6.0)
    assert is_watertight(parsed) == (True, 0)
    assert count_shells(parsed) == 1


def test_binary_stl_reader():
    triangles = cube()
    data = bytearray(b"binary cube".ljust(80, b" "))
    data += struct.pack("<I", len(triangles))
    for triangle in triangles:
        normal = (0.0, 0.0, 0.0)
        data += struct.pack("<12fH", *(normal + triangle[0] + triangle[1] + triangle[2]), 0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cube-binary.stl"
        path.write_bytes(bytes(data))
        parsed = read_stl(path)
    assert len(parsed) == 12
    _close(volume(parsed), 1.0)


def test_missing_cube_triangle_is_not_watertight():
    watertight, open_edges = is_watertight(cube()[:-1])
    assert not watertight
    assert open_edges == 3


def test_disjoint_cubes_have_two_shells():
    triangles = cube() + cube(3.0, 0.0, 0.0)
    assert count_shells(triangles) == 2
    assert is_watertight(triangles)[0]


def test_axis_cube_has_no_overhang_at_45_degrees():
    triangles = cube()
    _close(overhang_fraction(triangles, 45.0), 0.0)
    # With a limit just below zero, only the downward bottom face is counted: 1/6 area.
    _close(overhang_fraction(triangles, -0.01), 1.0 / 6.0)


def test_rotated_cube_has_overhang():
    assert overhang_fraction(_rotated_x45(cube()), 45.0) > 0.0


def test_min_wall_estimate_cube_is_one():
    estimate = min_wall_estimate(cube(), samples=12)
    assert estimate is not None
    _close(estimate, 1.0, 1.0e-5)


def test_gate_catalog_matches_the_gates_that_actually_run():
    """The catalog is the documented surface; it must not drift from `run_gates`."""
    design, placement, parts = _objects()
    catalog = [entry["name"] for entry in gate_catalog()]
    executed = [result.name for result in run_gates(design, placement, parts, {}).results]
    assert catalog == executed, (catalog, executed)
    assert len(catalog) == 19
    assert catalog[-1] == "material_estimate"


def test_all_gates_pass_with_valid_fake_objects():
    design, placement, parts = _objects()
    report = run_gates(design, placement, parts, {"base": _measurement()})
    assert report.passed
    assert len(report.results) == 19
    assert report.results[0].passed is True


def test_ir_valid_flips_on_errors():
    design, placement, parts = _objects()
    design["_errors"] = ["bad source"]
    result = run_gates(design, placement, parts, {})
    assert result.results[0].passed is False
    assert not result.passed


def test_ir_valid_validates_when_nobody_pre_populated_errors():
    """The gate must validate itself when `_errors` is absent, not assume the design is clean.

    `_errors` was read in exactly one place and written in NONE, so for every real caller the gate
    reported `errors=0 PASS` unconditionally. The test above did not catch it because it sets
    `_errors` by hand - it exercised the only path no caller used.

    Here the key is removed, which is the shape every real design arrives in, and the design is
    given a genuinely invalid `source.kind`. The gate has to come back FAIL.
    """
    design, placement, parts = _objects()
    del design["_errors"]
    design["components"] = [
        {"ref": "U1", "source": {"kind": "inline", "size": [1.0, 1.0, 1.0]}},
    ]
    gate = run_gates(design, placement, parts, {}).results[0]
    assert gate.name == "ir_valid", gate.name
    assert gate.passed is False, gate.message
    assert float(gate.value) >= 1, gate.message
    assert "validated in-gate" in gate.message, gate.message


def test_ir_valid_passes_a_clean_design_without_the_errors_key():
    """The other half: validating in-gate must not invent failures for a valid design.

    Uses a real shipped example rather than the `_objects()` stub, because the stub is deliberately
    partial - it has no `components`, so honest validation rightly reports it invalid. That is worth
    knowing on its own: before this gate validated anything, `run_gates` on a partial design
    reported `ir_valid PASS`.
    """
    from phimech.ir import ir as _ir

    _, placement, parts = _objects()
    design = _ir.load(str(Path(__file__).resolve().parents[1] / "examples" / "bare-assembly.json"))
    design.pop("_errors", None)
    gate = run_gates(design, placement, parts, {}).results[0]
    assert gate.name == "ir_valid", gate.name
    assert gate.passed is True, gate.message
    assert "errors=0" in gate.message, gate.message


def test_placement_overlap_flips_gate():
    design, placement, parts = _objects()
    placement.errors = ["A/B"]
    result = run_gates(design, placement, parts, {})
    assert _gate(result, "placement_no_overlap").passed is False


def test_clearance_flips_when_box_exits_cavity():
    design, placement, parts = _objects()
    placement.keepouts = [SimpleNamespace(world_bbox=lambda: ((-1.0, 1.0, 1.0), (9.0, 9.0, 9.0)))]
    result = run_gates(design, placement, parts, {})
    assert _gate(result, "clearance_positive").passed is False


def test_wall_and_floor_thresholds_flip():
    design, placement, parts = _objects()
    parts.wall = 1.0
    parts.floor = 1.0
    result = run_gates(design, placement, parts, {})
    assert _gate(result, "wall_thickness").passed is False
    assert _gate(result, "floor_thickness").passed is False


def test_non_nozzle_multiple_is_warning_only():
    design, placement, parts = _objects()
    parts.wall = 1.7
    result = run_gates(design, placement, parts, {})
    wall = _gate(result, "wall_thickness")
    assert wall.passed is False and wall.severity == "warn"
    assert result.passed


def test_bed_fit_and_volume_flip():
    design, placement, parts = _objects()
    result = run_gates(design, placement, parts, {"base": _measurement(size=(21.0, 10.0, 10.0))})
    assert _gate(result, "fits_bed").passed is False
    result = run_gates(design, placement, parts, {"base": _measurement(volume_value=0.0)})
    assert _gate(result, "part_volume_positive").passed is False


def test_manifold_checks_solids():
    design, placement, parts = _objects()
    result = run_gates(design, placement, parts, {"base": _measurement(solids=2)})
    assert _gate(result, "manifold").passed is False


def _inverted(triangles):
    return [(a, c, b) for a, b, c in triangles]


def test_count_bodies_ignores_voids_but_counts_floating_parts():
    """A sealed glyph void inside a plate is not a body; a tab floating beside it is."""
    plate = cube(0.0, 0.0, 0.0, 10.0)
    void = _inverted(cube(4.0, 4.0, 4.0, 1.0))
    assert count_bodies(plate + void) == 1
    assert count_bodies(plate + cube(20.0, 0.0, 0.0, 2.0)) == 2
    assert count_bodies(_inverted(plate)) == 1, "an inward-wound mesh is still one body"


def test_manifold_fails_disjoint_bodies_the_kernel_called_one_solid():
    """2026-10-08: an open_frame's wall-mount tabs floated beside the plate; the kernel still
    reported solids=1 and the mesh was watertight. The mesh body count catches it."""
    design, placement, parts = _objects()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "base.stl"
        path.write_text(ascii_stl(cube(0.0, 0.0, 0.0, 10.0) + cube(20.0, 0.0, 0.0, 2.0)))
        measurement = _measurement(solids=1, extra={"stl": str(path)})
        gate = _gate(run_gates(design, placement, parts, {"base": measurement}), "manifold")
    assert gate.passed is False, gate.message
    assert "bodies=2" in gate.message, gate.message


def test_text_legible_fails_a_label_cut_away_by_a_cutout():
    """A 6 mm 'FAN' label centred over a d35 fan hole has no glyph left; its size is 0."""
    design, placement, parts = _objects()
    parts.style = "open_frame"
    parts.cutouts = [{"kind": "cutout", "face": "bottom", "center": (30.0, 25.0, 0.0),
                      "size": (35.0, 35.0), "parts": ["base"], "face_rect": (0, 0, 60, 50),
                      "rect": (12.5, 7.5, 47.5, 42.5)}]
    parts.labels = [{"label": "FAN", "part": "base", "face": "top", "size": 6.0,
                     "rect": (22.8, 20.2, 37.2, 29.8)}]
    gate = _gate(run_gates(design, placement, parts, {}), "text_legible")
    assert gate.passed is False and gate.value == 0.0, gate.message
    assert "'FAN' overlaps a cutout" in gate.message, gate.message
    parts.labels = [{"label": "FAN", "part": "base", "face": "top", "size": 6.0,
                     "rect": None, "skipped": True}]
    gate = _gate(run_gates(design, placement, parts, {}), "text_legible")
    assert gate.passed is False and "was not placed" in gate.message, gate.message
    parts.labels = [{"label": "FAN", "part": "base", "face": "top", "size": 6.0,
                     "rect": (22.8, 0.5, 37.2, 6.1)}]
    assert _gate(run_gates(design, placement, parts, {}), "text_legible").passed is True


def test_min_feature_and_screw_count_flip():
    design, placement, parts = _objects()
    parts.summary = lambda: {"standoffs": [{"od": 3.0, "hole": 2.0}]}
    design["enclosure"]["lid"]["screw_count"] = 1
    result = run_gates(design, placement, parts, {})
    assert _gate(result, "min_feature").passed is False
    assert _gate(result, "screw_count_sane").passed is False


def test_cutout_and_text_are_warning_or_error_as_configured():
    design, placement, parts = _objects()
    parts.cutouts = [{"face": "front", "rect": (-1.0, 2.0, 4.0, 4.0), "face_rect": (0, 0, 10, 10)}]
    design["enclosure"]["text"]["size"] = 3.0
    result = run_gates(design, placement, parts, {})
    assert _gate(result, "cutout_within_face").passed is False
    text = _gate(result, "text_legible")
    assert text.passed is False
    assert text.severity == "warn"
    assert result.passed is False


def test_measurement_gates_skip_without_render():
    design, placement, parts = _objects()
    report = run_gates(design, placement, parts, {})
    for name in ("fits_bed", "part_volume_positive", "manifold", "overhang"):
        gate = _gate(report, name)
        assert gate.passed is None, (name, gate.message)
        assert gate.message == "skipped: no render", (name, gate.message)


def test_broken_enclosure_returns_crash_messages():
    design, placement, _ = _objects()
    report = run_gates(design, placement, None, {})
    assert report.results
    assert any(result.message.startswith("gate crashed:") for result in report.results)


def test_material_estimate_and_json_serialisation():
    design, placement, parts = _objects()
    report = run_gates(design, placement, parts, {"base": _measurement(volume_value=1000.0)})
    material = _gate(report, "material_estimate")
    assert material.passed is True
    _close(material.value, 1.24)
    json.dumps(report.to_dict())


# --------------------------------------------------------------------------------------
# assembly-interference gates
#
# Each of these three was written against a design the other fourteen gates passed clean, so
# each test builds the *real* failure it was written for rather than a stub.
# --------------------------------------------------------------------------------------


def _interference_objects():
    """A minimal placement + enclosure carrying the fields the interference gates read.

    One 10 x 10 x 10 mm component with two mounting holes, inside a cavity that spans the same
    footprint. `world_offset` is non-zero so a gate that forgets to apply it is caught: standoff
    records live in the enclosure datum frame, hole positions in the placement frame.
    """
    from phimech.components.envelope import Envelope, Hole, Placed

    envelope = Envelope(
        ref="U1",
        size=(10.0, 10.0, 10.0),
        holes=(Hole(at=(2.0, 2.0), d=3.2, ref="H1"), Hole(at=(8.0, 8.0), d=3.2, ref="H2")),
    )
    placed = Placed(envelope=envelope, position=(0.0, 0.0, 0.0))
    placement = SimpleNamespace(items=[placed], errors=[], keepouts=[placed])
    parts = SimpleNamespace(
        inner_cavity=((0.0, 0.0, 2.0), (10.0, 10.0, 14.0)),
        world_offset=(0.0, 0.0, 2.0),
        wall=1.6,
        floor=1.6,
        cutouts=[],
        # Both bosses under real holes: hole (2, 2) + world offset (0, 0) -> (2, 2).
        standoff_positions=[
            {"ref": "U1", "at": (2.0, 2.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 2.0},
            {"ref": "U1", "at": (8.0, 8.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 2.0},
        ],
        screw_positions=[],
        summary=lambda: {},
    )
    design = {
        "project": "interference-test",
        "_errors": [],
        "enclosure": {"wall": 1.6, "floor": 1.6, "lid": {"fastening": "screws", "screw_count": 4}},
    }
    return design, placement, parts


def test_interference_gates_pass_on_a_consistent_assembly():
    design, placement, parts = _interference_objects()
    report = run_gates(design, placement, parts, {})
    for name in ("standoff_within_cavity", "standoff_lands_on_hole"):
        gate = _gate(report, name)
        assert gate.passed is True, (name, gate.message)
    # The fixture records no lid posts, so this gate had nothing to judge: it must say so
    # (`not_applicable`) rather than claim a pass it never earned, and it must not block.
    posts = _gate(report, "fastener_clears_components")
    assert posts.passed is None and posts.effective_status == "not_applicable", posts.message


def test_standoff_outside_the_cavity_fails():
    """The exact defect found on `dev`: a mount hole far outside the part.

    A `mount.holes` entry of [900, 900] on a 60 x 40 mm component produced a boss 830 mm
    outside the enclosure, floating unsupported, and all fourteen prior gates reported PASS.
    """
    design, placement, parts = _interference_objects()
    parts.standoff_positions = [
        {"ref": "U1", "at": (900.0, 900.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 2.0}
    ]
    report = run_gates(design, placement, parts, {})
    gate = _gate(report, "standoff_within_cavity")
    assert gate.passed is False, gate.message
    assert gate.value < -800.0, gate.message
    assert report.passed is False


def test_lid_post_inside_the_cavity_but_through_a_component_fails():
    """The `lid.screw_count > 4` defect: `_reserve_screw_corners` reserves corners only.

    A mid-edge post is placed with no clearance analysis, so on a component that spans the
    cavity it drives straight through it. The post here is fully inside the cavity, so only
    the component-interference gate can catch it — `standoff_within_cavity` must still pass.
    """
    design, placement, parts = _interference_objects()
    parts.screw_positions = [
        {"at": (5.0, 5.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 12.0, "screw": "m3"}
    ]
    report = run_gates(design, placement, parts, {})
    clearance = _gate(report, "fastener_clears_components")
    assert clearance.passed is False, clearance.message
    assert clearance.value > 1.0, clearance.message
    assert "U1" in clearance.message, clearance.message
    assert _gate(report, "standoff_within_cavity").passed is True
    assert report.passed is False


def test_lid_post_beside_a_component_passes():
    """The other half: a post that genuinely clears must not be reported as interference."""
    design, placement, parts = _interference_objects()
    # Cavity is widened in X, the way `_reserve_screw_corners()` widens it, and the post sits
    # in the new space rather than over the component.
    parts.inner_cavity = ((-10.0, 0.0, 2.0), (10.0, 10.0, 14.0))
    parts.screw_positions = [
        {"at": (-5.0, 5.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 12.0, "screw": "m3"}
    ]
    report = run_gates(design, placement, parts, {})
    gate = _gate(report, "fastener_clears_components")
    assert gate.passed is True, gate.message


def test_standoff_that_matches_no_declared_hole_fails():
    """A boss is only a mount if the referenced component has a hole there."""
    design, placement, parts = _interference_objects()
    parts.standoff_positions = [
        # Inside the cavity, so the other two gates pass; 3 mm from the nearest real hole.
        {"ref": "U1", "at": (5.0, 2.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 2.0}
    ]
    report = run_gates(design, placement, parts, {})
    gate = _gate(report, "standoff_lands_on_hole")
    assert gate.passed is False, gate.message
    assert gate.value > 2.9, gate.message
    assert _gate(report, "standoff_within_cavity").passed is True
    assert report.passed is False


def test_standoff_naming_an_unplaced_component_fails():
    design, placement, parts = _interference_objects()
    parts.standoff_positions = [
        {"ref": "U9", "at": (2.0, 2.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 2.0}
    ]
    gate = _gate(run_gates(design, placement, parts, {}), "standoff_lands_on_hole")
    assert gate.passed is False, gate.message
    assert "U9" in gate.message, gate.message


def test_standoff_hole_match_accounts_for_the_world_offset():
    """A gate that ignored `world_offset` would still pass the happy case here by accident.

    So shift the offset in X, which the boss records already include, and confirm the gate
    follows: with a 3 mm X offset the boss over hole (2, 2) is recorded at (5, 2).
    """
    design, placement, parts = _interference_objects()
    parts.world_offset = (3.0, 0.0, 2.0)
    parts.inner_cavity = ((3.0, 0.0, 2.0), (13.0, 10.0, 14.0))
    parts.standoff_positions = [
        {"ref": "U1", "at": (5.0, 2.0), "od": 7.0, "hole_d": 2.5, "z0": 2.0, "h": 2.0}
    ]
    gate = _gate(run_gates(design, placement, parts, {}), "standoff_lands_on_hole")
    assert gate.passed is True, gate.message
    _close(gate.value, 0.0, 1e-9)


# --------------------------------------------------------------------------------------
# cutout_through: a declared opening is open on the exported mesh
# --------------------------------------------------------------------------------------

#: A 2.8 mm floor with an opening at (5, 5) declared on its bottom face.
_FLOOR_T = 2.8


def _plate_with_opening(floor_z):
    """A 10 x 10 floor plate drawn as two slabs either side of x in [4, 6], plus - when
    `floor_z` is given - a pocket floor across the gap at that height: what a blind hole
    leaves behind. The gate only reads triangles, so the mesh need not be closed."""
    tris = []
    for x0, x1 in ((0.0, 4.0), (6.0, 10.0)):
        t = _FLOOR_T
        a, b, c, d = (x0, 0.0, t), (x1, 0.0, t), (x1, 10.0, t), (x0, 10.0, t)
        tris += [(a, b, c), (a, c, d)]
        tris += [tuple((p[0], p[1], 0.0) for p in t) for t in ((a, c, b), (a, d, c))]
    if floor_z is not None:
        z = floor_z
        a, b, c, d = (4.0, 4.0, z), (6.0, 4.0, z), (6.0, 6.0, z), (4.0, 6.0, z)
        tris += [(a, c, b), (a, d, c)]
    return tris


def _through_parts(face="bottom", surface=(5.0, 5.0, 0.0), wall_t=_FLOOR_T):
    return SimpleNamespace(cutouts=[{
        "kind": "cable_gland", "face": face, "center": surface, "surface_center": surface,
        "size": (4.4, 4.4), "shape": "round", "parts": ["base"], "through": True,
        "wall_t": wall_t,
    }])


def _through_gate(tris, parts=None):
    from phimech.check.gates import _gate_cutout_through

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "base.stl"
        path.write_text(ascii_stl(tris))
        measurement = _measurement(extra={"stl": str(path)})
        return _gate_cutout_through(parts or _through_parts(), {"base": measurement})


def test_cutout_through_fails_a_blind_hole():
    """The 2026-10-08 regression, measured: a d4.4 'through' hole that stops 2.01 mm into a
    2.8 mm floor (what a wall-sized, surface-centred cutter left) must fail, by 0.79 mm."""
    gate = _through_gate(_plate_with_opening(2.01))
    assert gate.passed is False, gate.message
    _close(gate.value, 2.01 - _FLOOR_T, 1e-6)
    assert "BLIND" in gate.message and "clear 2.010 of 2.800" in gate.message, gate.message


def test_cutout_through_passes_an_open_hole():
    gate = _through_gate(_plate_with_opening(None))
    assert gate.passed is True, gate.message
    _close(gate.value, 0.0)


def test_cutout_through_fails_an_opening_that_was_never_cut():
    gate = _through_gate(cube(0.0, 0.0, 0.0, 10.0), _through_parts(wall_t=2.0))
    assert gate.passed is False
    _close(gate.value, -2.0)


def test_cutout_through_probes_along_the_face_normal():
    """A side-wall opening is probed horizontally: the plate below it is not in the way."""
    parts = _through_parts(face="front", surface=(5.0, 0.0, 5.0), wall_t=2.0)
    gate = _through_gate(_plate_with_opening(None), parts)
    assert gate.passed is True, gate.message


def test_cutout_through_is_not_applicable_without_declared_openings():
    design, placement, parts = _objects()  # a cutout record with no `through` claim
    gate = _gate(run_gates(design, placement, parts, {"base": _measurement()}), "cutout_through")
    assert gate.status == "not_applicable", gate.message


def test_cutout_through_skips_before_a_render():
    gate = _gate(run_gates(*_objects()[:2], _through_parts(), {}), "cutout_through")
    assert gate.passed is None and gate.message == "skipped: no render", gate.message


def _run_without_pytest():
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("{} gate and mesh tests passed".format(len(tests)))


if __name__ == "__main__":
    _run_without_pytest()
