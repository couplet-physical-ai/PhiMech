"""Unit tests for the general-tolerance and finish resolver.

The explicit main block follows the house dual-run convention: this slice is stdlib-only and is
also smoke-tested with the macOS system Python, which does not necessarily have pytest.

The test this file exists for is ``test_nothing_declared_produces_no_values_at_all``.  Every
other test guards a value the resolver is allowed to produce; that one guards the case where it
must produce none, because a tolerance PhiMech invented would be printed on a drawing somebody
manufactures to.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phimech.ir import tolerance as T
from phimech.ir.ir import apply_defaults, load_and_validate, validate
from phimech.ir.tolerance import (
    ISO_2768_ANGULAR,
    ISO_2768_BROKEN_EDGES,
    ISO_2768_CLASSES,
    ISO_2768_LINEAR,
    PRECEDENCE,
    UNBOUNDED,
    ToleranceBlock,
    format_angle,
    resolve_tolerance,
)



def _design(manufacturing: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project": "tolerance-test",
        "components": [{"ref": "A", "source": {"kind": "box", "size": [10, 10, 5]}}],
        "manufacturing": manufacturing,
    }


def _ir(**manufacturing: Any) -> ToleranceBlock:
    return resolve_tolerance(ir=_design(dict(manufacturing)))


def _manifest(**component: Any) -> Dict[str, Any]:
    return {"schema_version": 1, "component": dict({"name": "part.step"}, **component)}


def _raises(call: Any, fragment: str) -> str:
    try:
        call()
    except ValueError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return str(exc)
    raise AssertionError("expected ValueError containing %r" % fragment)


# -- nothing declared ----------------------------------------------------------------------


def test_nothing_declared_produces_no_values_at_all() -> None:
    """The load-bearing test: absence must never become a default.

    Checked through every entry point that could leak one - no arguments, empty documents, a
    design that has been through ``apply_defaults``, and a manufacturing block that declares
    other policy - because a default that appeared in only one of them would still reach a
    drawing.
    """
    candidates = [
        resolve_tolerance(),
        resolve_tolerance(ir=None, source_manifest=None),
        resolve_tolerance(ir={}),
        resolve_tolerance(ir=_design({})),
        resolve_tolerance(ir=apply_defaults(_design({}))),
        resolve_tolerance(ir=_design({"process": "cnc", "material": "6061-T6", "min_wall": 1.2})),
        resolve_tolerance(source_manifest={}),
        resolve_tolerance(source_manifest=_manifest()),
        resolve_tolerance(source_manifest=_manifest(material=None, mass_g=None,
                                                    general_tolerance=None, finish=None)),
        resolve_tolerance(ir=_design({}), source_manifest=_manifest(general_tolerance=None)),
    ]
    for index, block in enumerate(candidates):
        assert block.specified is False, index
        assert block.provenance == "none", index
        assert block.standard is None and block.standard_class is None, index
        assert block.linear == () and block.angular == () and block.broken_edges == (), index
        assert block.finish is None, index
        assert block.notes == () and block.conflicts == () and block.considered == (), index
        assert block.linear_for(10) is None and block.angular_for(10) is None, index
        assert block.broken_edge_for(1) is None, index
        payload = block.to_dict()
        assert payload["specified"] is False and payload["provenance"] == "none", index
        assert payload["linear"] == [] and payload["angular"] == [], index
        assert payload["broken_edges"] == [] and payload["standard"] is None, index
        assert payload["finish"] is None, index
        assert payload["unspecified"] == ["general_tolerance", "finish"], index
        assert payload["unspecified_text"] == "UNSPECIFIED", index
        assert json.loads(json.dumps(payload)) == payload, index


def test_the_schema_declares_no_default_for_either_field() -> None:
    """A JSON-schema default would be filled in by ``apply_defaults`` behind the resolver."""
    from phimech.ir.ir import SCHEMA

    manufacturing = SCHEMA["properties"]["manufacturing"]["properties"]
    for key in ("general_tolerance", "finish"):
        assert key in manufacturing, key
        assert "default" not in manufacturing[key], key
    general = SCHEMA["$defs"]["general_tolerance"]
    for name, sub in general["properties"].items():
        assert "default" not in sub, name
    for name, sub in SCHEMA["$defs"]["tolerance_band"]["properties"].items():
        assert "default" not in sub, name
    design = apply_defaults(_design({}))
    assert "general_tolerance" not in design["manufacturing"]
    assert "finish" not in design["manufacturing"]


def test_the_cutout_tolerance_is_not_a_drawing_tolerance() -> None:
    """`access[].tolerance` is a fit clearance; it must not reach the drawing block."""
    design = _design({})
    design["components"][0]["access"] = [
        {"name": "USB", "face": "front", "size": [9.5, 3.5], "tolerance": 0.5}]
    assert validate(design) == []
    assert resolve_tolerance(ir=design).specified is False
    # ...and the name is not silently accepted on the manufacturing block either.
    assert "additional property 'tolerance' is not allowed" in " ".join(
        validate(_design({"tolerance": 0.2})))


# -- the published ISO 2768-1 tables -------------------------------------------------------


def test_every_iso_2768_class_expands_to_its_published_table() -> None:
    for cls in ("f", "m", "c", "v"):
        block = _ir(general_tolerance={"standard": "ISO 2768-%s" % cls})
        assert block.specified is True and block.provenance == "ir"
        assert block.standard == "ISO 2768-%s" % cls
        assert block.standard_class == cls
        assert block.linear == ISO_2768_LINEAR[cls], cls
        assert block.angular == ISO_2768_ANGULAR[cls], cls
        assert block.broken_edges == ISO_2768_BROKEN_EDGES[cls], cls
        assert block.derivation == "ISO 2768-%s" % cls
        assert block.finish is None  # a class says nothing about finish


def test_spot_checked_published_linear_values() -> None:
    """Individual cells, written out by hand against the standard rather than recomputed."""
    fine = _ir(general_tolerance="ISO 2768-f")
    medium = _ir(general_tolerance="ISO 2768-m")
    coarse = _ir(general_tolerance="ISO 2768-c")
    very = _ir(general_tolerance="ISO 2768-v")

    assert fine.linear_for(2) == 0.05          # 0.5 up to 3
    assert fine.linear_for(100) == 0.15        # over 30 up to 120
    assert fine.linear_for(1500) == 0.5        # over 1000 up to 2000
    assert medium.linear_for(2) == 0.1
    assert medium.linear_for(20) == 0.2        # over 6 up to 30
    assert medium.linear_for(300) == 0.5       # over 120 up to 400
    assert medium.linear_for(3000) == 2.0      # over 2000 up to 4000
    assert coarse.linear_for(2) == 0.2
    assert coarse.linear_for(5) == 0.3         # over 3 up to 6 - NOT the DIN 7168 0.2
    assert coarse.linear_for(100) == 0.8
    assert coarse.linear_for(700) == 2.0       # over 400 up to 1000
    assert very.linear_for(5) == 0.5
    assert very.linear_for(20) == 1.0
    assert very.linear_for(3000) == 8.0

    # DIN 7168 is printed under an ISO 2768 title all over the internet; it differs from ISO
    # exactly here, so these two assertions are the guard against re-importing its values.
    assert coarse.linear_for(2) != 0.15
    assert coarse.linear_for(5) != 0.2


def test_band_boundaries_are_over_and_up_to_and_including() -> None:
    medium = _ir(general_tolerance="ISO 2768-m")
    assert medium.linear_for(3.0) == 0.1        # "up to 3" includes 3
    assert medium.linear_for(6.0) == 0.1        # "over 3 up to 6" includes 6
    assert medium.linear_for(6.0001) == 0.2     # ...and 6+ has moved to the next band
    assert medium.linear_for(0.5) == 0.1        # the first band's lower bound is inclusive
    assert medium.linear_for(0.49) is None      # below 0.5 mm: individually toleranced
    assert medium.linear_for(4000.0) == 2.0
    assert medium.linear_for(4000.1) is None    # past the end of the published table


def test_the_blank_cells_of_the_standard_stay_blank() -> None:
    """Class f defines nothing over 2000 mm and class v nothing below 3 mm."""
    fine = _ir(general_tolerance="ISO 2768-f")
    assert fine.linear_for(2000.0) == 0.5
    assert fine.linear_for(2000.1) is None
    assert fine.linear_for(3000) is None
    assert all(band[1] <= 2000.0 for band in fine.linear)

    very = _ir(general_tolerance="ISO 2768-v")
    assert very.linear_for(1) is None
    assert very.linear_for(3.0) is None         # the 0.5-3 cell is blank, so 3 is not covered
    assert very.linear_for(3.0001) == 0.5
    assert all(band[0] >= 3.0 for band in very.linear)

    # Both classes still have a full angular and broken-edge table; the blanks are linear only.
    assert len(fine.angular) == 5 and len(very.angular) == 5
    assert len(fine.broken_edges) == 3 and len(very.broken_edges) == 3


def test_spot_checked_published_angular_values() -> None:
    fine = _ir(general_tolerance="ISO 2768-f")
    medium = _ir(general_tolerance="ISO 2768-m")
    coarse = _ir(general_tolerance="ISO 2768-c")
    very = _ir(general_tolerance="ISO 2768-v")

    assert fine.angular == medium.angular       # the standard prints one merged f/m row
    assert medium.angular_for(5) == 1.0                 # up to 10:        1 deg
    assert medium.angular_for(30) == 30 / 60.0          # over 10 to 50:   0 deg 30'
    assert medium.angular_for(100) == 20 / 60.0         # over 50 to 120:  0 deg 20'
    assert medium.angular_for(300) == 10 / 60.0         # over 120 to 400: 0 deg 10'
    assert medium.angular_for(900) == 5 / 60.0          # over 400:        0 deg 5'
    assert coarse.angular_for(5) == 90 / 60.0           # 1 deg 30'
    assert coarse.angular_for(300) == 15 / 60.0         # 0 deg 15' - NOT the 0 deg 20' that one
    assert coarse.angular_for(300) != 20 / 60.0         #   widely mirrored vendor PDF prints
    assert very.angular_for(5) == 3.0
    assert very.angular_for(900) == 20 / 60.0

    assert medium.angular_for(10.0) == 1.0              # "up to 10" includes 10
    assert medium.angular_for(10.1) == 30 / 60.0
    assert medium.angular_for(0.0) == 1.0               # the angular table has no lower cut-off
    assert medium.angular_for(1e6) == 5 / 60.0          # "over 400" is unbounded


def test_angles_render_the_way_the_standard_prints_them() -> None:
    assert format_angle(1.0) == "1°"
    assert format_angle(30 / 60.0) == "0°30'"
    assert format_angle(20 / 60.0) == "0°20'"
    assert format_angle(90 / 60.0) == "1°30'"
    assert format_angle(3.0) == "3°"
    assert format_angle(0.123) == "0.123°"  # not a whole arc-minute: shown as declared
    rows = _ir(general_tolerance="ISO 2768-m").to_dict()["angular"]
    assert [row["plus_minus_display"] for row in rows] == [
        "1°", "0°30'", "0°20'", "0°10'", "0°5'"]


def test_broken_edges_are_a_separate_table_from_the_linear_one() -> None:
    """A 1 mm chamfer is +-0.2 under class m, not the +-0.1 the linear table would give it."""
    medium = _ir(general_tolerance="ISO 2768-m")
    assert medium.broken_edge_for(1.0) == 0.2
    assert medium.linear_for(1.0) == 0.1
    assert medium.broken_edge_for(5.0) == 0.5
    assert medium.broken_edge_for(50.0) == 1.0      # "over 6" is unbounded
    assert medium.broken_edge_for(0.4) is None      # below 0.5 mm, as for linear
    coarse = _ir(general_tolerance="ISO 2768-c")
    assert coarse.broken_edge_for(1.0) == 0.4
    assert coarse.broken_edge_for(50.0) == 2.0
    # The standard merges f with m and c with v in this table.
    assert ISO_2768_BROKEN_EDGES["f"] == ISO_2768_BROKEN_EDGES["m"]
    assert ISO_2768_BROKEN_EDGES["c"] == ISO_2768_BROKEN_EDGES["v"]
    assert medium.to_dict()["broken_edges"][-1]["range_max_mm"] is None  # unbounded -> null


def test_designation_spellings_the_standard_allows() -> None:
    for text in ("ISO 2768-m", "iso 2768 m", "ISO2768-M", "ISO 2768-1-m", "2768-m"):
        block = _ir(general_tolerance=text)
        assert block.standard_class == "m", text
        assert block.standard == text, text  # carried verbatim; never rewritten for the sheet
        assert block.linear == ISO_2768_LINEAR["m"], text

    combined = _ir(general_tolerance="ISO 2768-mK")
    assert combined.standard_class == "m" and combined.linear == ISO_2768_LINEAR["m"]
    assert any("geometrical class K" in note and "NOT expanded" in note
               for note in combined.notes)

    envelope = _ir(general_tolerance="ISO 2768-mK-E")
    assert envelope.standard_class == "m"
    assert any("envelope requirement" in note for note in envelope.notes)

    # ISO 2768-2 clause 6.2: the part-1 class may be omitted. Then there is no size tolerance.
    geometric_only = _ir(general_tolerance="ISO 2768-K")
    assert geometric_only.specified is True and geometric_only.standard_class is None
    assert geometric_only.linear == () and geometric_only.angular == ()
    assert any("no ISO 2768-1 size class" in note for note in geometric_only.notes)


def test_an_unknown_standard_is_carried_but_never_approximated() -> None:
    block = _ir(general_tolerance={"standard": "ASME Y14.5-2018"})
    assert block.specified is True and block.provenance == "ir"
    assert block.standard == "ASME Y14.5-2018" and block.standard_class is None
    assert block.linear == () and block.angular == () and block.broken_edges == ()
    assert block.linear_for(10) is None
    assert any("not a class this resolver can expand" in note for note in block.notes)
    assert block.to_dict()["unspecified"] == ["finish"]


# -- explicit values -----------------------------------------------------------------------


def test_an_explicit_table_passes_through_verbatim() -> None:
    declared = {
        "linear": [{"range_min": 0, "range_max": 10, "plus_minus": 0.05},
                   {"range_min": 10, "range_max": 100, "plus_minus": 0.15}],
        "angular": [{"range_min": 0, "range_max": 50, "plus_minus": 0.25}],
        "broken_edges": [{"range_min": 0, "range_max": 5, "plus_minus": 0.3}],
    }
    block = _ir(general_tolerance=copy.deepcopy(declared))
    assert block.specified is True and block.provenance == "ir"
    assert block.standard is None and block.standard_class is None
    assert block.linear == ((0.0, 10.0, 0.05), (10.0, 100.0, 0.15))
    assert block.angular == ((0.0, 50.0, 0.25),)
    assert block.broken_edges == ((0.0, 5.0, 0.3),)
    assert block.derivation == "declared"
    assert block.notes == ()  # nothing was expanded, so there is nothing to explain
    assert block.linear_for(5) == 0.05 and block.linear_for(50) == 0.15
    assert block.linear_for(101) is None
    assert block.angular_for(10) == 0.25


def test_a_single_plus_minus_applies_to_every_linear_dimension() -> None:
    block = _ir(general_tolerance={"plus_minus": 0.2})
    assert block.linear == ((0.0, UNBOUNDED, 0.2),)
    assert block.linear_for(0.1) == 0.2 and block.linear_for(10_000) == 0.2
    assert block.angular == () and block.angular_for(10) is None
    assert block.broken_edges == ()
    payload = block.to_dict()
    assert payload["linear"][0]["range_max_mm"] is None
    assert json.loads(json.dumps(payload)) == payload


def test_a_finish_can_be_declared_on_its_own() -> None:
    block = _ir(finish="bead blast, then anodise clear")
    assert block.specified is True and block.provenance == "ir"
    assert block.finish == "bead blast, then anodise clear"
    assert block.standard is None and block.linear == ()
    assert block.to_dict()["unspecified"] == ["general_tolerance"]


def test_declarations_the_resolver_refuses_rather_than_guesses() -> None:
    _raises(lambda: _ir(general_tolerance={"standard": "ISO 2768-m", "plus_minus": 0.2}),
            "cannot print both")
    _raises(lambda: _ir(general_tolerance={"standard": "ISO 2768-m",
                                           "linear": [{"range_min": 0, "range_max": 1,
                                                       "plus_minus": 0.1}]}),
            "cannot print both")
    _raises(lambda: _ir(general_tolerance={"plus_minus": 0.2,
                                           "linear": [{"range_min": 0, "range_max": 1,
                                                       "plus_minus": 0.1}]}),
            "mutually exclusive")
    _raises(lambda: _ir(general_tolerance={}), "omit it entirely")
    _raises(lambda: _ir(general_tolerance={"plus_minus": 0}), "greater than zero")
    _raises(lambda: _ir(general_tolerance={"plus_minus": -0.1}), "greater than zero")
    _raises(lambda: _ir(general_tolerance={"linear": []}), "non-empty list")
    _raises(lambda: _ir(general_tolerance={"linear": [{"range_min": 0, "range_max": 1}]}),
            "missing plus_minus")
    _raises(lambda: _ir(general_tolerance={"linear": [{"range_min": 5, "range_max": 1,
                                                       "plus_minus": 0.1}]}),
            "greater than range_min")
    _raises(lambda: _ir(general_tolerance={"linear": [
        {"range_min": 0, "range_max": 10, "plus_minus": 0.1},
        {"range_min": 5, "range_max": 20, "plus_minus": 0.2}]}), "overlaps")
    _raises(lambda: _ir(general_tolerance={"linear": [{"range_min": 0, "range_max": 1,
                                                       "plus_minus": "0.1"}]}),
            "must be a number")
    _raises(lambda: _ir(general_tolerance={"nonsense": 1}), "unknown key")
    _raises(lambda: _ir(general_tolerance=0.2), "must be an object or a published class name")
    _raises(lambda: _ir(finish=""), "non-empty string")
    _raises(lambda: _ir(finish=3.2), "non-empty string")


# -- precedence and provenance ---------------------------------------------------------------


def test_precedence_is_ir_over_manifest_and_the_loser_stays_visible() -> None:
    assert PRECEDENCE == ("ir", "manifest")
    block = resolve_tolerance(
        ir=_design({"general_tolerance": "ISO 2768-f", "finish": "Ra 1.6"}),
        source_manifest=_manifest(general_tolerance="ISO 2768-c", finish="as machined"),
    )
    assert block.provenance == "ir"
    assert block.standard == "ISO 2768-f" and block.linear == ISO_2768_LINEAR["f"]
    assert block.finish == "Ra 1.6"
    sources = {entry.source: entry for entry in block.considered}
    assert sources["ir"].used is True
    assert sources["manifest"].used is False
    assert "overridden" in sources["manifest"].reason
    assert len(block.conflicts) == 1
    conflict = block.conflicts[0]
    assert "ISO 2768-f" in conflict and "ISO 2768-c" in conflict
    assert "as machined" in conflict and "Ra 1.6" in conflict
    assert block.to_dict()["conflicts"] == list(block.conflicts)


def test_the_manifest_is_used_when_the_ir_declares_nothing() -> None:
    block = resolve_tolerance(
        ir=_design({"process": "cnc"}),
        source_manifest=_manifest(material="6061-T6", general_tolerance="ISO 2768-m",
                                  finish="Ra 3.2"),
    )
    assert block.specified is True and block.provenance == "manifest"
    assert block.linear == ISO_2768_LINEAR["m"]
    assert block.finish == "Ra 3.2"
    assert [entry.source for entry in block.considered] == ["manifest"]
    assert block.conflicts == ()
    assert all(row["source"] == "manifest" for row in block.to_dict()["linear"])


def test_agreeing_declarations_are_not_reported_as_a_conflict() -> None:
    block = resolve_tolerance(
        ir=_design({"general_tolerance": {"standard": "ISO 2768-m"}, "finish": "Ra 3.2"}),
        source_manifest=_manifest(general_tolerance={"standard": "ISO 2768-m"}, finish="Ra 3.2"),
    )
    assert block.conflicts == ()
    sources = {entry.source: entry for entry in block.considered}
    assert sources["manifest"].used is False and "agrees" in sources["manifest"].reason


def test_the_winning_document_supplies_the_whole_block_and_says_so() -> None:
    """Precedence is block-level: a title block must describe one document, not a blend."""
    block = resolve_tolerance(
        ir=_design({"general_tolerance": "ISO 2768-m"}),
        source_manifest=_manifest(finish="Ra 3.2"),
    )
    assert block.provenance == "ir"
    assert block.finish is None, "the manifest finish must not be merged in silently"
    assert block.to_dict()["unspecified"] == ["finish"]
    assert len(block.conflicts) == 1 and "Ra 3.2" in block.conflicts[0]


def test_no_value_is_ever_produced_without_provenance() -> None:
    """Walk every value a drawing could print and insist each one names its source."""
    blocks = [
        _ir(general_tolerance="ISO 2768-m", finish="Ra 3.2"),
        _ir(general_tolerance={"plus_minus": 0.2}),
        _ir(general_tolerance={"linear": [{"range_min": 0, "range_max": 9, "plus_minus": 0.1}],
                               "angular": [{"range_min": 0, "range_max": 9, "plus_minus": 0.5}]}),
        resolve_tolerance(source_manifest=_manifest(general_tolerance="ISO 2768-c",
                                                    finish="as machined")),
        _ir(finish="as printed"),
        _ir(general_tolerance="ASME Y14.5-2018"),
    ]
    for block in blocks:
        payload = block.to_dict()
        assert payload["provenance"] in ("ir", "manifest"), payload
        assert payload["considered"], payload
        for table in ("linear", "angular", "broken_edges"):
            for row in payload[table]:
                assert row["source"] in ("ir", "manifest"), row
                assert row["derivation"], row
        for key in ("standard", "finish"):
            if payload[key] is not None:
                assert payload[key]["source"] in ("ir", "manifest"), payload[key]
                assert payload[key]["value"], payload[key]
        assert json.loads(json.dumps(payload)) == payload


def test_the_unspecified_block_is_shared_and_cannot_be_mutated_into_a_value() -> None:
    first, second = resolve_tolerance(), resolve_tolerance()
    assert first is T.UNSPECIFIED_BLOCK and second is T.UNSPECIFIED_BLOCK
    try:
        first.specified = True  # type: ignore[misc]
    except Exception as exc:  # frozen dataclass
        assert exc.__class__.__name__ == "FrozenInstanceError", exc
    else:
        raise AssertionError("ToleranceBlock must be frozen")
    assert resolve_tolerance().specified is False


# -- round trips ---------------------------------------------------------------------------


def test_round_trip_through_the_ir_loader() -> None:
    """A design written to disk, loaded, validated and defaulted must resolve to what it said."""
    design = _design({
        "process": "cnc",
        "general_tolerance": {"standard": "ISO 2768-m"},
        "finish": "Ra 3.2 unless otherwise stated",
    })
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "design.json"
        path.write_text(json.dumps(design, indent=2), encoding="utf-8")
        loaded, errors = load_and_validate(path)
    assert errors == [], errors
    block = resolve_tolerance(ir=loaded)
    assert block.provenance == "ir" and block.standard_class == "m"
    assert block.linear == ISO_2768_LINEAR["m"]
    assert block.finish == "Ra 3.2 unless otherwise stated"


def test_round_trip_through_the_loader_with_an_explicit_table_and_the_string_shorthand() -> None:
    design = _design({
        "general_tolerance": {
            "linear": [{"range_min": 0, "range_max": 30, "plus_minus": 0.1},
                       {"range_min": 30, "range_max": 200, "plus_minus": 0.25}],
        },
    })
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "design.json"
        path.write_text(json.dumps(design), encoding="utf-8")
        loaded, errors = load_and_validate(path)
        assert errors == [], errors
        assert resolve_tolerance(ir=loaded).linear == ((0.0, 30.0, 0.1), (30.0, 200.0, 0.25))

        # The alias layer accepts the shorthand a human actually writes and the loader
        # normalises it to the object form before the resolver ever sees it.
        shorthand = _design({"general_tolerance": "ISO 2768-m", "surface_finish": "Ra 3.2"})
        path.write_text(json.dumps(shorthand), encoding="utf-8")
        loaded, errors = load_and_validate(path)
    assert errors == [], errors
    assert loaded["manufacturing"]["general_tolerance"] == {"standard": "ISO 2768-m"}
    assert loaded["manufacturing"]["finish"] == "Ra 3.2"
    block = resolve_tolerance(ir=loaded)
    assert block.standard_class == "m" and block.finish == "Ra 3.2"


def test_a_bare_manufacturing_block_or_component_block_is_accepted() -> None:
    """The drawing lane often holds only the sub-object, not the whole document."""
    assert resolve_tolerance(ir={"general_tolerance": "ISO 2768-m"}).standard_class == "m"
    assert resolve_tolerance(source_manifest={"finish": "Ra 3.2"}).finish == "Ra 3.2"
    assert resolve_tolerance(ir={"manufacturing": {}}).specified is False


def test_the_tables_agree_with_the_class_list() -> None:
    assert sorted(ISO_2768_CLASSES) == ["c", "f", "m", "v"]
    for cls in ISO_2768_CLASSES:
        assert len(ISO_2768_ANGULAR[cls]) == 5, cls
        assert len(ISO_2768_BROKEN_EDGES[cls]) == 3, cls
    assert len(ISO_2768_LINEAR["m"]) == len(ISO_2768_LINEAR["c"]) == 8
    assert len(ISO_2768_LINEAR["f"]) == len(ISO_2768_LINEAR["v"]) == 7  # one blank cell each


def _run_standalone() -> None:
    tests: List[Any] = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("%d tolerance tests passed" % len(tests))


if __name__ == "__main__":
    _run_standalone()
