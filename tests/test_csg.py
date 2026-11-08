"""Focused tests for the CSG builders and structural validator.

The file intentionally has no pytest-only fixtures: it is both a pytest module and a
standalone Python 3.9 smoke test for the core, which must also import inside FreeCAD.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phimech.csg.ast import (  # noqa: E402
    Cube,
    Cylinder,
    Difference,
    Import,
    LinearExtrude,
    Polygon,
    Polyhedron,
    Scale,
    Sphere,
    Union,
    from_dict,
    is_2d,
)
from phimech.csg.ops import (  # noqa: E402
    HEAT_SET,
    EPS,
    box,
    countersunk_hole,
    cut_through,
    cyl,
    diff_all,
    ensure_ccw,
    heat_set_boss,
    is_ccw,
    offset_polygon,
    place,
    polygon_area,
    polygon_bbox,
    rounded_box,
    rounded_rect,
    screw_boss,
    slot,
    standoff,
    text_emboss,
    tube,
    union_all,
    vent_slots,
)
from phimech.csg.validate import assert_valid, stats, validate  # noqa: E402


def _round_trip(node):
    assert from_dict(node.to_dict()) == node
    assert_valid(node)
    return node


def _has_error(node, text):
    messages = validate(node)
    assert any(text in message for message in messages), messages


def _raises(function, text=None):
    try:
        function()
    except ValueError as exc:
        if text is not None:
            assert text in str(exc), str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_box_round_trip():
    _round_trip(box(10, 20, 3, center=True, at=(1, 2, 3)))


def test_cylinder_axes_round_trip():
    for axis in ("x", "y", "z"):
        _round_trip(cyl(4, 8, axis=axis, at=(1, 2, 3), segments=24))


def test_cylinder_rejects_bad_axis():
    _raises(lambda: cyl(4, 8, axis="q"), "axis")


def test_tube_round_trip():
    _round_trip(tube(12, 6, 4, axis="y", at=(2, 3, 4)))


def test_rounded_box_round_trip():
    _round_trip(rounded_box(20, 10, 5, 2))


def test_rounded_box_zero_radius_is_valid():
    node = rounded_box(20, 10, 5, 0)
    assert node.kind == "cube"
    _round_trip(node)


def test_rounded_box_edge_rounding_round_trip():
    _round_trip(rounded_box(20, 10, 5, 2, edge_r=1))
    _round_trip(rounded_box(20, 10, 5, 0, edge_r=1))


def test_rounded_rect_is_two_dimensional():
    node = rounded_rect(20, 10, 2)
    # Asserts the property the name promises. This used to check `kind == "hull"`, which was a
    # proxy for "2D" and became wrong when rounded_rect started returning a `rounded_square` so
    # that FreeCAD could fillet it instead of failing on a planar hull.
    assert node.kind == "rounded_square"
    assert is_2d(node)
    _round_trip(node)


def test_slot_round_trip():
    node = _round_trip(slot(20, 3, 4))
    assert node.kind == "linear_extrude"


def test_countersunk_hole_round_trip():
    _round_trip(countersunk_hole(3, 8, 6, 2))


def test_screw_boss_round_trip():
    _round_trip(screw_boss(10, 4, 6))
    _round_trip(screw_boss(10, 4, 6, fillet_r=1))


def test_standoff_round_trip():
    _round_trip(standoff(5, 8, 3))
    _round_trip(standoff(5, 8, 3, base_d=12))


def test_heat_set_boss_table_and_round_trip():
    assert HEAT_SET["m3"] == (4.0, 4.0)
    _round_trip(heat_set_boss("m3", 6))


def test_heat_set_boss_rejects_unknown_insert():
    _raises(lambda: heat_set_boss("m5"), "unknown")


def test_heat_set_boss_rejects_shallow_boss():
    _raises(lambda: heat_set_boss("m3", 3), "at least")


def test_vent_slots_round_trip():
    node = _round_trip(vent_slots(50, 25, 2, 4, 6, margin=3))
    assert node.kind == "union"
    assert len(node.children) == 5


def test_cut_through_has_eps_overcut():
    node = cut_through(box(10, 10, 2), box(10, 10, 3))
    assert node.kind == "difference"
    assert node.children[1].kind == "minkowski"
    assert node.children[1].children[1].r == EPS
    _round_trip(node)


def test_union_and_difference_helpers():
    a, b = box(1, 1, 1), box(2, 2, 2)
    assert union_all((a, b)).kind == "union"
    assert diff_all(a, ()) is a
    _round_trip(diff_all(a, (b,)))


def test_union_all_requires_a_node():
    _raises(lambda: union_all(()), "at least one")


def test_place_round_trip():
    node = place(box(1, 2, 3), at=(10, 20, 30), rot=(0, 0, 90))
    assert node.kind == "translate"
    _round_trip(node)


def test_text_emboss_convention_round_trip():
    node = _round_trip(text_emboss("PhiMech", 6, 0.6))
    assert node.kind == "group"
    assert node.name.startswith("text:")
    assert '"valign":"center"' in node.name


def test_polygon_offset_square_area():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    grown = offset_polygon(square, 1)
    assert abs(polygon_area(grown) - 144.0) < 1e-6


def test_polygon_helpers():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_area(square) == 100.0
    assert is_ccw(square)
    assert not is_ccw(list(reversed(square)))
    assert polygon_bbox(square) == ((0.0, 0.0), (10.0, 10.0))
    assert ensure_ccw(list(reversed(square))) == square


def test_offset_polygon_rejects_clockwise_input():
    _raises(lambda: offset_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], 1), "counter-clockwise")


def test_validate_rejects_bad_cube_size():
    _has_error(Cube(size=(1, 0, 2)), "size[1]")


def test_validate_rejects_bad_cylinder_height_and_radius():
    _has_error(Cylinder(h=0, r=0), ".h")
    _has_error(Cylinder(h=0, r=0), ".r")


def test_validate_accepts_cone_with_one_zero_radius():
    assert validate(Cylinder(h=2, r=1, r2=0)) == []
    assert validate(Cylinder(h=2, r=0, r2=1)) == []
    _has_error(Cylinder(h=2, r=0, r2=0), "at least one positive radius")


def test_validate_rejects_bad_sphere_radius():
    _has_error(Sphere(r=-1), ".r")


def test_validate_rejects_empty_boolean_and_transform():
    _has_error(Union([]), "at least one child")
    _has_error(Scale(v=(1, 1, 1), children=[]), "at least one child")


def test_validate_warns_on_one_child_difference():
    messages = validate(Difference([box(1, 1, 1)]))
    assert any(message.startswith("warn:") for message in messages), messages
    assert_valid(Difference([box(1, 1, 1)]))


def test_validate_rejects_three_dimensional_extrusion_child():
    _has_error(LinearExtrude(height=2, children=[box(1, 1, 1)]), "must be 2D")


def test_validate_rejects_bad_polygon():
    _has_error(Polygon(points=((0, 0), (1, 0))), "at least three")


def test_validate_rejects_nested_zero_area_polygon_with_exact_path():
    node = LinearExtrude(
        height=1,
        children=[Polygon(points=((0, 0), (1, 0), (2, 0)))],
    )
    expected = "root.children[0] polygon outline encloses zero area"
    assert validate(node) == [expected]
    _raises(lambda: assert_valid(node), expected)


def test_validate_rejects_short_and_zero_area_explicit_paths():
    node = Polygon(
        points=((0, 0), (1, 0), (2, 0), (0, 1)),
        paths=((0, 1), (0, 1, 2)),
    )
    assert validate(node) == [
        "root paths[0] needs at least three indices",
        "root polygon paths[1] encloses zero area",
    ]


def test_validate_accepts_both_polygon_windings():
    counter_clockwise = Polygon(points=((0, 0), (2, 0), (0, 1)))
    clockwise = Polygon(points=tuple(reversed(counter_clockwise.points)))
    assert validate(counter_clockwise) == []
    assert validate(clockwise) == []


def test_validate_rejects_bad_polyhedron_face_index():
    node = Polyhedron(points=((0, 0, 0), (1, 0, 0), (0, 1, 0)), faces=((0, 1, 9),))
    _has_error(node, "invalid point")


def test_validate_rejects_zero_scale_factor():
    _has_error(Scale(v=(1, 0, 1), children=[box(1, 1, 1)]), "nonzero")


def test_validate_rejects_nonfinite_numbers():
    _has_error(Cube(size=(1, math.nan, 1)), "finite")
    _has_error(Cube(size=(1, math.inf, 1)), "finite")


def test_validate_rejects_import_path_shape():
    _has_error(Import(path="relative.stl"), "absolute")
    _has_error(Import(path="/tmp/not-a-model.bad"), "not allowed")
    _has_error(Import(path="/tmp/not-a-model.stl"), "does not exist")


def test_validate_accepts_existing_import():
    with tempfile.NamedTemporaryFile(suffix=".stl") as handle:
        assert validate(Import(path=os.path.abspath(handle.name))) == []


def test_stats_counts_depth_and_imports():
    with tempfile.NamedTemporaryFile(suffix=".stl") as handle:
        imported = Import(path=os.path.abspath(handle.name))
        node = Union([box(1, 1, 1), place(imported, (1, 2, 3))])
        summary = stats(node)
    assert summary["counts"]["union"] == 1
    assert summary["counts"]["cube"] == 1
    assert summary["counts"]["import"] == 1
    assert summary["depth"] == 3
    assert len(summary["imports"]) == 1


def test_ast_json_round_trip_for_complex_builder():
    node = diff_all(rounded_box(30, 20, 4, 2), [vent_slots(20, 10, 1.5, 3, 5)])
    assert from_dict(node.to_dict()) == node


def test_standalone_suite():
    assert EPS == 0.01


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print("PASS: {} CSG tests".format(len(tests)))
