"""A library part that declares its mounting holes needs no ``mount.holes`` in the IR.

`ir.validate` demanded ``mount.holes`` for every non-KiCad standoff mount, although synth already
takes a library envelope's own holes first (`enclosure._holes_of`). So
``{"kind": "library", "part": "raspberry-pi-4b"}`` with ``mount: {"style": "standoff"}`` was
refused, and the only way through was to retype the Pi's hole pattern - a second copy that can
only drift from the library's.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phimech.components import library  # noqa: E402
from phimech.ir.ir import validate  # noqa: E402

_HOLES_ERROR = "mount.holes"


def _design(source: dict, mount: dict) -> dict:
    return {
        "project": "library-holes",
        "title": "Library holes",
        "components": [{"ref": "U1", "name": "part", "source": source, "mount": mount}],
        "enclosure": {"style": "tray_lid"},
    }


def _hole_errors(design: dict) -> list:
    return [error for error in validate(design) if _HOLES_ERROR in error]


def test_library_part_with_declared_holes_needs_no_mount_holes() -> None:
    assert library.make("raspberry-pi-4b").holes, "fixture assumption: the Pi 4 declares holes"
    design = _design({"kind": "library", "part": "raspberry-pi-4b"}, {"style": "standoff"})
    assert _hole_errors(design) == [], validate(design)


def test_library_part_without_holes_still_needs_them() -> None:
    assert not library.make("18650").holes, "fixture assumption: a cell has no mounting holes"
    design = _design({"kind": "library", "part": "18650"}, {"style": "standoff"})
    assert _hole_errors(design), "a hole-less library part must still ask for mount.holes"


def test_box_source_still_needs_holes_and_explicit_holes_still_satisfy() -> None:
    box = {"kind": "box", "size": [40, 30, 5]}
    assert _hole_errors(_design(box, {"style": "standoff"}))
    explicit = {"style": "standoff", "holes": [[3, 3], [37, 27]]}
    assert _hole_errors(_design(box, explicit)) == []


def test_unknown_library_part_does_not_crash_the_hole_check() -> None:
    design = _design({"kind": "library", "part": "no-such-part"}, {"style": "standoff"})
    assert _hole_errors(design), "an unknown part declares no holes"


if __name__ == "__main__":
    test_library_part_with_declared_holes_needs_no_mount_holes()
    test_library_part_without_holes_still_needs_them()
    test_box_source_still_needs_holes_and_explicit_holes_still_satisfy()
    test_unknown_library_part_does_not_crash_the_hole_check()
    print("ok")
