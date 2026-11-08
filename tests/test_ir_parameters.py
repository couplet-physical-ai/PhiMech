"""Parameters: the one supported way to edit a design without regenerating it.

Runs both ways, like the rest of the suite::

    /usr/bin/python3 tests/test_ir_parameters.py
    pytest tests/test_ir_parameters.py

Why these tests exist
---------------------
A UI slider and a hand edit must not be able to produce a design that its own IR cannot reproduce.
That is the whole reason parameters exist rather than letting a client patch ``enclosure.wall`` or,
worse, a coordinate. So the tests below pin the *boundaries* of the feature, not its happy path:
bounds are enforced on both routes in, a typo'd reference fails loudly instead of silently reading
as a string, and the ``parameters`` block itself is never substituted into.
"""

from __future__ import annotations

import json
import os as _os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:  # pragma: no cover - exercised by the system-python runner
    pytest = None  # type: ignore[assignment]

from phimech.ir import ir

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "parametric-box.json"


class _Skipped(Exception):
    pass


def _skip(message: str) -> None:
    if pytest is not None and _os.environ.get("PYTEST_CURRENT_TEST"):
        pytest.skip(message)
    raise _Skipped(message)


def _temp_design(mutate: Any = None) -> Path:
    """Copy the example into a temp file, optionally mutating it first."""
    if not EXAMPLE.is_file():
        _skip("examples/parametric-box.json is absent")
    design = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(design)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(design, handle)
    handle.close()
    return Path(handle.name)


# -- resolution ----------------------------------------------------------------------------


def test_example_resolves_and_validates() -> None:
    design, errors = ir.load_and_validate(EXAMPLE)
    assert errors == [], errors
    # Every reference must be gone by the time anything downstream sees the design.
    assert design["enclosure"]["wall"] == 2
    assert design["components"][0]["source"]["size"] == [40, 50, 8]


def test_no_reference_survives_loading() -> None:
    """A leftover ``\"$name\"`` string would reach a backend as a dimension and fail obscurely."""
    design, _ = ir.load_and_validate(EXAMPLE)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "parameters" or (isinstance(key, str) and key.startswith("_")):
                    continue
                walk(item, "{}.{}".format(path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, "{}[{}]".format(path, index))
        elif isinstance(node, str):
            assert not node.startswith("$"), "unresolved reference at {}: {!r}".format(path, node)

    walk(design, "design")


def test_parameters_block_is_not_substituted_into() -> None:
    """A parameter cannot reference a parameter; the block must survive verbatim for the UI."""
    design, _ = ir.load_and_validate(EXAMPLE)
    spec = design["parameters"]["wall"]
    assert spec["value"] == 2 and spec["min"] == 1.2 and spec["max"] == 5
    assert spec["description"]


def test_unknown_reference_is_a_named_error() -> None:
    path = _temp_design(lambda d: d["enclosure"].__setitem__("wall", "$wal"))
    try:
        _, errors = ir.load_and_validate(path)
        assert errors, "a typo'd reference must not validate"
        joined = " ".join(errors)
        assert "$wal" in joined and "enclosure.wall" in joined, joined
        # The message has to list what IS declared, or the caller cannot fix it in one step.
        assert "wall" in joined
    finally:
        path.unlink(missing_ok=True)


def test_design_without_parameters_still_works() -> None:
    """The feature must be invisible to every existing design."""
    plain = REPO / "examples" / "bare-assembly.json"
    if not plain.is_file():
        _skip("examples/bare-assembly.json is absent")
    design, errors = ir.load_and_validate(plain)
    assert errors == [], errors
    assert "parameters" not in design or design["parameters"] == {}


# -- bounds --------------------------------------------------------------------------------


def test_out_of_bounds_value_in_the_file_is_caught() -> None:
    """Bounds are checked on load, not only on write - a design can be hand-edited or generated."""
    path = _temp_design(lambda d: d["parameters"]["wall"].__setitem__("value", 0.4))
    try:
        _, errors = ir.load_and_validate(path)
        assert any("below min" in e for e in errors), errors
    finally:
        path.unlink(missing_ok=True)


def test_inverted_bounds_are_caught() -> None:
    def mutate(design: Dict[str, Any]) -> None:
        design["parameters"]["wall"]["min"] = 9
        design["parameters"]["wall"]["max"] = 2

    path = _temp_design(mutate)
    try:
        _, errors = ir.load_and_validate(path)
        assert any("greater than max" in e for e in errors), errors
    finally:
        path.unlink(missing_ok=True)


def test_write_refuses_out_of_bounds_and_leaves_the_file_untouched() -> None:
    """A refused edit must not partially write: the design has to stay buildable."""
    path = _temp_design()
    try:
        before = path.read_text(encoding="utf-8")
        for bad in (0.4, 99.0):
            try:
                ir.set_parameter(path, "wall", bad)
            except ValueError as exc:
                assert "min" in str(exc) or "max" in str(exc), exc
            else:
                raise AssertionError("wall={} should have been refused".format(bad))
        assert path.read_text(encoding="utf-8") == before, "refused edit modified the file"
        assert not list(path.parent.glob(path.name + ".tmp-*")), "temp file left behind"
    finally:
        path.unlink(missing_ok=True)


def test_write_refuses_an_undeclared_name_unless_asked() -> None:
    path = _temp_design()
    try:
        try:
            ir.set_parameter(path, "not_a_parameter", 1)
        except KeyError as exc:
            assert "not_a_parameter" in str(exc)
        else:
            raise AssertionError("an undeclared parameter must not be created silently")
        # Explicit opt-in is allowed, because authoring a new parameter is a legitimate act.
        result = ir.set_parameter(path, "not_a_parameter", 1, allow_new=True)
        assert result["value"] == 1 and result["previous"] is None
    finally:
        path.unlink(missing_ok=True)


def test_step_is_not_enforced() -> None:
    """Off-grid but in-bounds must be accepted: a gate-passing value is not refused for tidiness."""
    path = _temp_design()
    try:
        result = ir.set_parameter(path, "wall", 2.37)  # step is 0.2
        assert result["value"] == 2.37
        _, errors = ir.load_and_validate(path)
        assert errors == [], errors
    finally:
        path.unlink(missing_ok=True)


# -- the round trip ------------------------------------------------------------------------


def test_edit_changes_the_resolved_design_everywhere_at_once() -> None:
    """One parameter, many referencing fields - that is the point of the indirection."""
    path = _temp_design()
    try:
        ir.set_parameter(path, "board_width", 70)
        design, errors = ir.load_and_validate(path)
        assert errors == [], errors
        assert design["components"][0]["source"]["size"][0] == 70
        assert design["parameters"]["board_width"]["value"] == 70
    finally:
        path.unlink(missing_ok=True)


def test_write_is_atomic_and_canonical() -> None:
    path = _temp_design()
    try:
        ir.set_parameter(path, "wall", 3)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n"), "canonical output must end in a newline"
        reloaded = json.loads(text)
        assert reloaded["parameters"]["wall"]["value"] == 3
        keys = list(reloaded.keys())
        assert keys == sorted(keys), "keys must be sorted so diffs stay reviewable"
    finally:
        path.unlink(missing_ok=True)


def _run_direct() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except _Skipped as exc:
            print("SKIP {}: {}".format(test.__name__, exc))
        except Exception as exc:  # pragma: no cover - direct invocation only
            failures += 1
            print("FAIL {}: {}".format(test.__name__, exc))
        else:
            print("PASS {}".format(test.__name__))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_direct())
