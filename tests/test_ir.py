"""Unit tests for the stdlib PhiMech IR loader and validator.

The explicit main block is intentional: this slice is also smoke-tested with the macOS
system Python, which does not necessarily have pytest installed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phimech.ir.ir import SCHEMA, apply_defaults, dumps, load, load_and_validate, validate


EXAMPLES = REPO / "examples"


def _box_design() -> Dict[str, Any]:
    return {
        "project": "test-design",
        "components": [
            {"ref": "A", "source": {"kind": "box", "size": [10, 10, 5]}},
            {"ref": "B", "source": {"kind": "box", "size": [5, 5, 5]}},
        ],
    }


def _assert_has(errors: List[str], text: str) -> None:
    assert any(text in error for error in errors), (text, errors)


def test_schema_is_loaded() -> None:
    assert SCHEMA["$schema"].endswith("2020-12/schema")
    assert "component" in SCHEMA["$defs"]


def test_battery_example_validates() -> None:
    _, errors = load_and_validate(EXAMPLES / "battery-box.json")
    assert errors == [], errors


def test_bare_example_validates_without_enclosure() -> None:
    design, errors = load_and_validate(EXAMPLES / "bare-assembly.json")
    assert errors == [], errors
    assert "enclosure" not in design


def test_rpi_example_validates() -> None:
    _, errors = load_and_validate(EXAMPLES / "rpi-zero-case.json")
    assert errors == [], errors


def test_defaults_apply_to_enclosure_and_outputs() -> None:
    design = _box_design()
    design["enclosure"] = {"wall": 2.25}
    design["outputs"] = {}
    apply_defaults(design)
    assert design["enclosure"]["floor"] == 2.25
    assert design["outputs"]["formats"] == ["stl", "step", "scad", "png"]
    assert design["outputs"]["backend"] == "auto"


def test_defaults_apply_to_omitted_output_and_manufacturing_objects() -> None:
    design = apply_defaults(_box_design())
    assert design["outputs"]["formats"] == ["stl", "step", "scad", "png"]
    assert design["manufacturing"]["process"] == "fdm"
    assert design["manufacturing"]["min_wall"] == 1.2


def test_defaults_follow_source_one_of() -> None:
    design = _box_design()
    design["components"][0]["source"] = {"kind": "cylinder", "d": 8, "h": 12}
    apply_defaults(design)
    assert design["components"][0]["source"]["axis"] == "z"


def test_load_sets_metadata_and_resolves_source_path() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        fixture = root / "part.step"
        fixture.write_text("placeholder", encoding="utf-8")
        design_path = root / "design.json"
        design_path.write_text(
            json.dumps(
                {
                    "project": "path-test",
                    "components": [
                        {
                            "ref": "PART1",
                            "source": {"kind": "import", "path": "part.step"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        design = load(design_path)
        assert design["_meta"]["path"] == str(design_path.resolve())
        assert design["_meta"]["base_dir"] == str(root.resolve())
        assert design["components"][0]["source"]["path"] == str(fixture.resolve())
        assert validate(design) == []


def test_title_defaults_to_project() -> None:
    design = apply_defaults(_box_design())
    assert design["title"] == "test-design"


def test_duplicate_component_ref_is_rejected() -> None:
    design = _box_design()
    design["components"][1]["ref"] = "A"
    _assert_has(validate(design), "duplicate component ref")


def test_dangling_placement_ref_is_rejected() -> None:
    design = _box_design()
    design["placement"] = [{"ref": "MISSING", "relation": "beside", "to": "A"}]
    _assert_has(validate(design), "placement[0].ref: unknown component ref")


def test_dangling_placement_to_is_rejected() -> None:
    design = _box_design()
    design["placement"] = [{"ref": "A", "relation": "beside", "to": "MISSING"}]
    _assert_has(validate(design), "placement[0].to: unknown component ref")


def test_placement_cycle_is_rejected() -> None:
    design = _box_design()
    design["placement"] = [
        {"ref": "A", "relation": "beside", "to": "B"},
        {"ref": "B", "relation": "beside", "to": "A"},
    ]
    _assert_has(validate(design), "placement cycle")


def test_missing_at_position_is_rejected() -> None:
    design = _box_design()
    design["placement"] = [{"ref": "A", "relation": "at"}]
    _assert_has(validate(design), "position: required for an 'at' relation")


def test_missing_external_source_is_rejected() -> None:
    design = _box_design()
    design["components"][0]["source"] = {"kind": "import", "path": "/not/a/real/file.step"}
    _assert_has(validate(design), "source.path: file does not exist")


def test_non_kicad_standoff_requires_holes() -> None:
    design = _box_design()
    design["components"][0]["mount"] = {"style": "standoff"}
    _assert_has(validate(design), "mount.holes: required for standoff")


def test_non_kicad_standoff_with_holes_is_valid() -> None:
    design = _box_design()
    design["components"][0]["mount"] = {
        "style": "standoff",
        "holes": [[2, 2], [8, 8]],
    }
    assert validate(design) == []


def test_bad_project_slug_is_rejected() -> None:
    design = _box_design()
    design["project"] = "Not valid"
    _assert_has(validate(design), "project: does not match pattern")


def test_bad_component_ref_is_rejected() -> None:
    design = _box_design()
    design["components"][0]["ref"] = "1BAD"
    _assert_has(validate(design), "components[0].ref: does not match pattern")


def test_extra_property_is_rejected() -> None:
    design = _box_design()
    design["components"][0]["typo"] = True
    _assert_has(validate(design), "additional property 'typo' is not allowed")


def test_wrong_source_kind_is_rejected() -> None:
    design = _box_design()
    design["components"][0]["source"] = {"kind": "wat", "size": [1, 1, 1]}
    _assert_has(validate(design), "source kind must be one of")


def test_wrong_enum_is_rejected() -> None:
    design = _box_design()
    design["components"][0]["source"] = {"kind": "cylinder", "d": 8, "h": 10, "axis": "diagonal"}
    _assert_has(validate(design), "axis: must be one of")


def test_wrong_type_is_rejected() -> None:
    design = _box_design()
    design["components"][0]["source"]["size"] = "10x10x5"
    _assert_has(validate(design), "source.size: expected type array")


def test_units_const_is_rejected() -> None:
    design = _box_design()
    design["units"] = "inch"
    _assert_has(validate(design), "units: must be 'mm'")


def test_wall_warning_is_reported() -> None:
    design = _box_design()
    design["enclosure"] = {"wall": 0.5}
    design["manufacturing"] = {"min_wall": 1.2}
    _assert_has(validate(design), "warn: enclosure.wall")


def test_screw_count_below_two_is_rejected() -> None:
    design = _box_design()
    design["enclosure"] = {"lid": {"fastening": "screws", "screw_count": 1}}
    _assert_has(validate(design), "screw_count")


def test_dumps_strips_metadata_and_round_trips() -> None:
    design = load(EXAMPLES / "bare-assembly.json")
    encoded = dumps(design)
    assert "_meta" not in encoded
    decoded = json.loads(encoded)
    assert "_meta" not in decoded
    assert validate(decoded) == []
    assert json.loads(dumps(decoded)) == decoded


def test_private_keys_are_not_schema_properties() -> None:
    design = _box_design()
    design["_meta"] = {"path": "/tmp/design.json", "base_dir": "/tmp"}
    design["components"][0]["_note"] = "loader note"
    assert validate(design) == []


def test_required_root_property_is_rejected() -> None:
    design = _box_design()
    del design["components"]
    _assert_has(validate(design), "required property 'components' is missing")


def test_load_and_validate_returns_same_loaded_design() -> None:
    design, errors = load_and_validate(EXAMPLES / "bare-assembly.json")
    assert design["_meta"]["base_dir"] == str(EXAMPLES.resolve())
    assert errors == []


def test_enclosure_cutout_accepts_the_shapes_synth_cuts() -> None:
    """`cutout` features are cut by `features.cutout_node`, which takes rect|round|slot and a
    tolerance; the schema used to reject both keys, so a round cutout could not be asked for."""
    design = _box_design()
    design["enclosure"] = {"style": "open_frame", "floor": 2.8, "features": [
        {"kind": "cutout", "face": "bottom", "at": [5, 5], "size": [4.4, 4.4],
         "shape": "round", "tolerance": 0.1},
        {"kind": "cutout", "face": "left", "size": [8, 3], "shape": "slot"},
    ]}
    assert validate(design) == []
    design["enclosure"]["features"][0]["shape"] = "hexagon"
    _assert_has(validate(design), "enclosure.features[0].shape")


def _run_standalone() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("%d IR tests passed" % len(tests))


if __name__ == "__main__":
    _run_standalone()
