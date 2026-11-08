"""Tests for relative placement, runnable with pytest or /usr/bin/python3."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phimech.components.envelope import Envelope
from phimech.synth.place import auto_arrange, place


def box(ref, size):
    return Envelope(ref=ref, size=tuple(float(value) for value in size))


def design(components, placement=None):
    return {"components": components, "placement": placement or []}


def component(ref, clearance=0.0, orientation=None):
    result = {"ref": ref, "clearance": clearance, "source": {"kind": "box"}}
    if orientation is not None:
        result["orientation"] = list(orientation)
    return result


def test_beside_exact_x():
    envelopes = {"A": box("A", (10, 4, 2)), "B": box("B", (5, 3, 1))}
    result = place(
        design(
            [component("A"), component("B")],
            [{"ref": "B", "relation": "beside", "to": "A", "gap": 2, "align": "min"}],
        ),
        envelopes,
    )
    assert result.by_ref["B"].position == (12.0, 0.0, 0.0)
    assert result.keepout_by_ref["B"].world_bbox()[0][0] == 12.0


def test_gap_accounts_for_both_clearances():
    envelopes = {"A": box("A", (10, 4, 2)), "B": box("B", (5, 3, 1))}
    result = place(
        design(
            [component("A", 1), component("B", 2)],
            [{"ref": "B", "relation": "beside", "to": "A", "gap": 3}],
        ),
        envelopes,
    )
    # A keep-out ends at 11 and B keep-out begins at 14; B's body origin is 2 mm
    # inside its keep-out, so its origin is 16 mm from the assembly origin.
    assert result.by_ref["B"].position[0] == 16.0
    a_max = result.keepout_by_ref["A"].world_bbox()[1][0]
    b_min = result.keepout_by_ref["B"].world_bbox()[0][0]
    assert b_min - a_max == 3.0


def test_align_min():
    envelopes = {"A": box("A", (10, 20, 30)), "B": box("B", (2, 4, 6))}
    result = place(
        design(
            [component("A"), component("B")],
            [{"ref": "B", "relation": "beside", "to": "A", "align": "min"}],
        ),
        envelopes,
    )
    assert result.by_ref["B"].position == (12.0, 0.0, 0.0)


def test_align_center():
    envelopes = {"A": box("A", (10, 20, 30)), "B": box("B", (2, 4, 6))}
    result = place(
        design(
            [component("A"), component("B")],
            [{"ref": "B", "relation": "beside", "to": "A", "align": "center"}],
        ),
        envelopes,
    )
    assert result.by_ref["B"].position == (12.0, 8.0, 12.0)


def test_align_max():
    envelopes = {"A": box("A", (10, 20, 30)), "B": box("B", (2, 4, 6))}
    result = place(
        design(
            [component("A"), component("B")],
            [{"ref": "B", "relation": "beside", "to": "A", "align": "max"}],
        ),
        envelopes,
    )
    assert result.by_ref["B"].position == (12.0, 16.0, 24.0)


def test_above_is_positive_z():
    envelopes = {"A": box("A", (10, 10, 10)), "B": box("B", (4, 4, 3))}
    result = place(
        design([component("A"), component("B")], [{"ref": "B", "relation": "above", "to": "A"}]),
        envelopes,
    )
    assert result.by_ref["B"].position[2] == 12.0


def test_below_is_negative_z():
    envelopes = {"A": box("A", (10, 10, 10)), "B": box("B", (4, 4, 3))}
    result = place(
        design([component("A"), component("B")], [{"ref": "B", "relation": "below", "to": "A"}]),
        envelopes,
    )
    assert result.by_ref["B"].position[2] == -5.0


def test_behind_is_positive_y():
    envelopes = {"A": box("A", (10, 10, 10)), "B": box("B", (4, 4, 3))}
    result = place(
        design([component("A"), component("B")], [{"ref": "B", "relation": "behind", "to": "A"}]),
        envelopes,
    )
    assert result.by_ref["B"].position[1] == 12.0


def test_in_front_is_negative_y():
    envelopes = {"A": box("A", (10, 10, 10)), "B": box("B", (4, 4, 3))}
    result = place(
        design([component("A"), component("B")], [{"ref": "B", "relation": "in_front", "to": "A"}]),
        envelopes,
    )
    assert result.by_ref["B"].position[1] == -6.0


def test_stacked_on_has_zero_gap_and_center_alignment():
    envelopes = {"A": box("A", (10, 10, 10)), "B": box("B", (4, 6, 3))}
    result = place(
        design(
            [component("A"), component("B")],
            [{"ref": "B", "relation": "stacked_on", "to": "A", "gap": 99, "align": "max"}],
        ),
        envelopes,
    )
    assert result.by_ref["B"].position == (3.0, 2.0, 10.0)
    assert result.keepout_by_ref["B"].world_bbox()[0][2] == 10.0


def test_centered_in_centers_all_axes():
    envelopes = {"A": box("A", (20, 20, 20)), "B": box("B", (4, 6, 8))}
    result = place(
        design(
            [component("A"), component("B")],
            [{"ref": "B", "relation": "centered_in", "to": "A"}],
        ),
        envelopes,
    )
    assert result.by_ref["B"].position == (8.0, 7.0, 6.0)


def test_orientation_90_swaps_x_and_y_extents():
    envelopes = {"A": box("A", (10, 4, 3))}
    result = place(design([component("A", orientation=(0, 0, 90))]), envelopes)
    minimum, maximum = result.by_ref["A"].world_bbox()
    assert maximum[0] - minimum[0] == 4.0
    assert maximum[1] - minimum[1] == 10.0


def test_chain_resolves_in_dependency_order():
    envelopes = {ref: box(ref, (1, 1, 1)) for ref in ("A", "B", "C")}
    result = place(
        design(
            [component("C"), component("B"), component("A")],
            [
                {"ref": "C", "relation": "beside", "to": "B", "gap": 1},
                {"ref": "B", "relation": "beside", "to": "A", "gap": 1},
            ],
        ),
        envelopes,
    )
    assert result.order == ["A", "B", "C"]
    assert result.by_ref["B"].position[0] == 2.0
    assert result.by_ref["C"].position[0] == 4.0


def test_overlap_reports_both_refs_without_raising():
    envelopes = {"A": box("A", (5, 5, 5)), "B": box("B", (5, 5, 5))}
    result = place(
        design(
            [component("A"), component("B")],
            [
                {"ref": "A", "relation": "at", "position": [0, 0, 0]},
                {"ref": "B", "relation": "at", "position": [0, 0, 0]},
            ],
        ),
        envelopes,
    )
    assert result.errors
    assert "A" in result.errors[0] and "B" in result.errors[0]


def test_unconstrained_components_use_non_overlapping_x_shelf():
    envelopes = {"A": box("A", (1, 2, 3)), "B": box("B", (2, 2, 3)), "C": box("C", (3, 2, 3))}
    result = place(design([component("A"), component("B"), component("C")]), envelopes)
    assert result.errors == []
    assert [result.by_ref[ref].position[0] for ref in ("A", "B", "C")] == [0.0, 3.0, 7.0]


def test_origin_is_not_normalized_and_bboxes_are_distinct():
    result = place(design([component("A", clearance=1)]), {"A": box("A", (10, 10, 10))})
    assert result.by_ref["A"].position == (0.0, 0.0, 0.0)
    assert result.bbox_bodies() == ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    assert result.bbox() == ((-1.0, -1.0, -1.0), (11.0, 11.0, 11.0))


def test_invalid_orientation_raises():
    try:
        place(design([component("A", orientation=(0, 15, 0))]), {"A": box("A", (1, 1, 1))})
    except ValueError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("non-right-angle orientation did not raise")


def test_at_position_and_offset():
    result = place(
        design(
            [component("A")],
            [{"ref": "A", "relation": "at", "position": [4, 5, 6], "offset": [1, -1, 2]}],
        ),
        {"A": box("A", (1, 1, 1))},
    )
    assert result.by_ref["A"].position == (5.0, 4.0, 8.0)


def test_cycle_is_rejected():
    envelopes = {"A": box("A", (1, 1, 1)), "B": box("B", (1, 1, 1))}
    try:
        place(
            design(
                [component("A"), component("B")],
                [
                    {"ref": "A", "relation": "beside", "to": "B"},
                    {"ref": "B", "relation": "beside", "to": "A"},
                ],
            ),
            envelopes,
        )
    except ValueError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("cyclic constraints did not raise")


def test_auto_arrange_is_a_beside_shelf():
    envelopes = {"A": box("A", (4, 2, 1)), "B": box("B", (3, 2, 1)), "C": box("C", (2, 2, 1))}
    constraints = auto_arrange(envelopes, gap=1)
    assert [constraint["relation"] for constraint in constraints] == ["at", "beside", "beside"]
    assert constraints[1]["to"] == "A"
    assert constraints[2]["to"] == "B"


def test_auto_arrange_wraps_rows_along_y():
    envelopes = {"A": box("A", (4, 2, 1)), "B": box("B", (4, 2, 1)), "C": box("C", (3, 2, 1))}
    constraints = auto_arrange(envelopes, gap=1, max_width=9)
    assert [constraint["relation"] for constraint in constraints] == ["at", "beside", "behind"]
    assert constraints[2]["to"] == "A"
    result = place(design([component(ref) for ref in envelopes], constraints), envelopes)
    assert result.by_ref["C"].position[1] == 3.0
    assert result.errors == []


def test_explain_contains_every_ref():
    envelopes = {ref: box(ref, (1, 1, 1)) for ref in ("A", "B", "C")}
    result = place(
        design(
            [component("A"), component("B"), component("C")],
            [
                {"ref": "B", "relation": "beside", "to": "A"},
                {"ref": "C", "relation": "above", "to": "B"},
            ],
        ),
        envelopes,
    )
    explanation = result.explain()
    assert all(ref in explanation for ref in envelopes)
    assert "B beside A" in explanation
    assert "C above B" in explanation


def test_to_dict_reports_body_and_keepout_bboxes():
    result = place(design([component("A", clearance=1)]), {"A": box("A", (2, 3, 4))})
    data = result.to_dict()
    assert data["bbox"]["min"] == [-1.0, -1.0, -1.0]
    assert data["bbox_bodies"]["max"] == [2.0, 3.0, 4.0]
    assert data["items"][0]["keepout_bbox"]["max"] == [3.0, 4.0, 5.0]


def _run_without_pytest():
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("{} placement tests passed".format(len(tests)))


if __name__ == "__main__":
    _run_without_pytest()
