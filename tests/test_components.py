"""Component-library and generic-provider tests.

The file is deliberately runnable with the system Python because the component core has no
third-party dependency.  Pytest will also collect each function normally.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from phimech.components.envelope import FACES
from phimech.components.imports import envelope_from_import
from phimech.components.library import get, list_parts, make
from phimech.components.providers import envelope_for, envelopes_for
from phimech.csg.ast import Cube


def _numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numbers(child)


def test_registry_has_at_least_twenty_five_parts() -> None:
    assert len(list_parts()) >= 25
    assert all({"key", "name", "category", "params", "size"} <= set(item) for item in list_parts())


def test_every_default_part_builds() -> None:
    for item in list_parts():
        envelope = make(item["key"], ref="X1")
        assert envelope.ref == "X1"
        assert envelope.body is not None
        assert envelope.source == "library:" + item["key"]
        assert all(value > 0.0 for value in envelope.size)


def test_every_library_body_is_finite() -> None:
    for item in list_parts():
        envelope = make(item["key"])
        assert envelope.body is not None
        assert all(math.isfinite(number) for number in _numbers(envelope.body.to_dict()))
        assert all(value > 0.0 for value in envelope.size)


def test_library_access_faces_are_valid() -> None:
    for item in list_parts():
        envelope = make(item["key"])
        assert all(access.face in FACES for access in envelope.access)


def test_library_hole_centres_are_inside_footprint() -> None:
    for item in list_parts():
        envelope = make(item["key"])
        width, depth, _ = envelope.size
        for hole in envelope.holes:
            assert 0.0 <= hole.at[0] <= width
            assert 0.0 <= hole.at[1] <= depth


def test_raspberry_pi_zero_mounting_and_ports() -> None:
    envelope = make("raspberry-pi-zero-2w", ref="PCB1")
    assert len(envelope.holes) == 4
    assert len(envelope.access) >= 3
    assert {access.name for access in envelope.access} >= {"USB1", "USB2", "HDMI"}


def test_18650_default_orientation_is_x() -> None:
    assert make("18650").size == (65.2, 18.6, 18.6)


def test_18650_y_orientation_changes_order() -> None:
    assert make("18650", orientation="y").size == (18.6, 65.2, 18.6)


def test_18650_z_orientation_changes_order() -> None:
    assert make("18650", orientation="z").size == (18.6, 18.6, 65.2)


def test_display_windows_are_top_access() -> None:
    for key in ("ssd1306-0.96", "ssd1306-1.3", "ili9341-2.4", "lcd1602", "epaper-2.13"):
        windows = [access for access in make(key).access if access.name == "WINDOW"]
        assert len(windows) == 1
        assert windows[0].face == "top"
        assert windows[0].through_lid


def test_ili9341_has_four_corner_holes() -> None:
    assert len(make("ili9341-2.4").holes) == 4
    assert all(hole.d == 3.0 for hole in make("ili9341-2.4").holes)


def test_lcd1602_hole_pattern_is_four_places() -> None:
    envelope = make("lcd1602")
    assert len(envelope.holes) == 4
    assert {(hole.at[0], hole.at[1]) for hole in envelope.holes} == {
        (2.5, 2.5),
        (77.5, 2.5),
        (2.5, 33.5),
        (77.5, 33.5),
    }


def test_fan_holes_and_top_access() -> None:
    envelope = make("40mm-fan")
    assert len(envelope.holes) == 4
    assert envelope.holes[0].d == 3.5
    assert envelope.access[0].face == "top"
    assert envelope.access[0].shape == "round"


def test_parameterised_lipo_size() -> None:
    assert make("lipo-pouch", w=60, d=40, h=8).size == (60.0, 40.0, 8.0)


def test_parameterised_screw_length() -> None:
    assert make("m3-screw", l=20).size == (5.5, 5.5, 22.4)


def test_unknown_part_has_close_match() -> None:
    try:
        make("1865")
    except KeyError as exc:
        assert "18650" in str(exc)
    else:
        raise AssertionError("unknown part did not raise KeyError")


def test_get_returns_part_spec() -> None:
    spec = get("raspberry-pi-zero-2w")
    assert spec.key == "raspberry-pi-zero-2w"
    assert callable(spec.build)


def test_provider_box_path() -> None:
    envelope = envelope_for({"ref": "B1", "source": {"kind": "box", "size": [10, 20, 3]}})
    assert envelope.size == (10.0, 20.0, 3.0)
    assert envelope.body is not None and envelope.body.kind == "cube"
    assert envelope.source == "box"


def test_provider_cylinder_axis_paths() -> None:
    for axis, expected in (
        ("x", (8.0, 4.0, 4.0)),
        ("y", (4.0, 8.0, 4.0)),
        ("z", (4.0, 4.0, 8.0)),
    ):
        envelope = envelope_for(
            {"ref": "J1", "source": {"kind": "cylinder", "d": 4, "h": 8, "axis": axis}}
        )
        assert envelope.size == expected


def test_provider_library_path() -> None:
    envelope = envelope_for(
        {"ref": "BAT1", "source": {"kind": "library", "part": "18650", "params": {}}}
    )
    assert envelope.source == "library:18650"
    assert envelope.ref == "BAT1"


def test_provider_explicit_mount_holes() -> None:
    envelope = envelope_for(
        {
            "ref": "M1",
            "source": {"kind": "box", "size": [20, 10, 4]},
            "mount": {"holes": [[3, 3], [17, 7]], "hole_d": 2.5},
        }
    )
    assert [(hole.at, hole.d) for hole in envelope.holes] == [
        ((3.0, 3.0), 2.5),
        ((17.0, 7.0), 2.5),
    ]


def test_provider_access_defaults_to_face_centre() -> None:
    envelope = envelope_for(
        {
            "ref": "P1",
            "source": {"kind": "box", "size": [10, 20, 30]},
            "access": [{"name": "PORT", "face": "right", "size": [4, 5]}],
        }
    )
    assert envelope.access[0].at == (10.0, 15.0)


def test_provider_applies_name_and_thermal_power() -> None:
    envelope = envelope_for(
        {
            "ref": "U1",
            "name": "controller",
            "thermal_w": 3.5,
            "source": {"kind": "box", "size": [2, 3, 4]},
        }
    )
    assert envelope.name == "controller"
    assert envelope.thermal_w == 3.5


def test_envelopes_for_returns_reference_map() -> None:
    design = {
        "components": [
            {"ref": "A1", "source": {"kind": "box", "size": [1, 2, 3]}},
            {"ref": "A2", "source": {"kind": "library", "part": "aaa"}},
        ]
    }
    result = envelopes_for(design)
    assert sorted(result) == ["A1", "A2"]
    assert result["A2"].source == "library:aaa"


def test_provider_csg_path_uses_backend_measurement() -> None:
    import phimech.backends.base as backend_base

    original = getattr(backend_base, "for_format", None)

    class FakeBackend:
        def measure(self, node: Any) -> Any:
            assert node.kind == "cube"
            return SimpleNamespace(bbox=SimpleNamespace(size=(7.0, 8.0, 9.0)))

    backend_base.for_format = lambda fmt: FakeBackend()
    try:
        envelope = envelope_for(
            {"ref": "C1", "source": {"kind": "csg", "csg": Cube((1, 1, 1)).to_dict()}}
        )
    finally:
        if original is None:
            del backend_base.for_format
        else:
            backend_base.for_format = original
    assert envelope.size == (7.0, 8.0, 9.0)
    assert envelope.source == "csg"


def test_import_measurement_cache() -> None:
    import phimech.backends.base as backend_base

    original = getattr(backend_base, "for_format", None)
    calls = []

    class FakeMeasurement:
        def to_dict(self) -> Dict[str, Any]:
            return {
                "bbox": {"min": [0, 0, 0], "max": [3, 4, 5]},
                "volume_mm3": 60,
                "area_mm2": 94,
                "solids": 1,
            }

    class FakeBackend:
        def measure(self, node: Any) -> FakeMeasurement:
            calls.append(node)
            return FakeMeasurement()

    backend_base.for_format = lambda fmt: FakeBackend()
    try:
        with TemporaryDirectory() as temporary:
            model = Path(temporary) / "part.stl"
            model.write_bytes(b"solid test\nendsolid test\n")
            first = envelope_from_import(model, "I1")
            second = envelope_from_import(model, "I2", scale=2)
            assert first.size == (3.0, 4.0, 5.0)
            assert second.size == (6.0, 8.0, 10.0)
            assert len(calls) == 1
            assert (Path(str(model) + ".phimech-measure.json")).is_file()
    finally:
        if original is None:
            del backend_base.for_format
        else:
            backend_base.for_format = original


# --------------------------------------------------------------------------------------
# the IR -> board-file referencing contract
# --------------------------------------------------------------------------------------

TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_")]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print("{}: ok".format(test.__name__))
    print("{} component tests passed".format(len(TESTS)))
