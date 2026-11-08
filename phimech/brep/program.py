"""The feature program — what crosses into the B-rep kernel.

Why a declarative op list and not a Python script
-------------------------------------------------
The obvious way to drive a B-rep kernel is to let the model write kernel code and `exec` it,
which is what agentcad does. We deliberately do not, for three reasons:

* **Rule 4.** The agent writes *intent*; numbers are computed by us. A script is not intent,
  it is an implementation, and it can compute its own coordinates behind our back.
* **A script cannot be validated before it runs.** An op list can: every op is checked for
  known name, required keys, numeric ranges and unit sanity *before* a subprocess starts, so a
  malformed request costs milliseconds instead of a kernel timeout.
* **`exec` of model-written code has no sandbox.** agentcad says so itself. A data program has
  no import, no file IO and no arbitrary call surface, so the worker's attack surface is the
  op table below and nothing else.

The cost is expressiveness: anything not in `OPS` cannot be built. That is the intended
trade — the vocabulary grows deliberately, one reviewed op at a time.

Why this lane exists at all
---------------------------
`ops.rounded_box()` builds a rounded box as `Hull([cylinders])`, and a hull has no curved
faces: measured, a 60x40x25 rounded box exports as **130 planar faces and zero cylindrical
surfaces**, and a real enclosure reaches 9,174. Rounding an edge is a *fillet*, and a fillet
needs a kernel that knows what an edge is. This is that lane.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

VERSION = 1

# Edge and face selectors, kept as a closed vocabulary so a typo is an error rather than an
# empty selection that silently rounds nothing.
EDGE_SELECTORS: Tuple[str, ...] = ("vertical", "top", "bottom", "all", "top_and_bottom")
FACE_SELECTORS: Tuple[str, ...] = ("top", "bottom", "front", "back", "left", "right")

# Metric coarse pitches, used to derive a tapping pilot diameter. We do NOT model a helix:
# build123d 0.10 has no IsoThread, sweeping a profile along a Helix is slow and fragile, and a
# printed part is tapped or takes a heat-set insert anyway. So the op is named for what it
# actually does -- cut the pilot -- rather than for what it might be mistaken for.
THREAD_PITCH: Dict[str, float] = {
    "M2": 0.4, "M2.5": 0.45, "M3": 0.5, "M4": 0.7, "M5": 0.8, "M6": 1.0, "M8": 1.25,
}

_POSITIVE = "must be a positive number"


class ProgramError(ValueError):
    """A feature program that cannot be built. Message names the op index and the problem."""


def _num(value: Any, name: str, where: str, *, positive: bool = True,
         allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError("{}: {} must be a number, got {!r}".format(where, name, value))
    out = float(value)
    if positive and (out < 0 or (out == 0 and not allow_zero)):
        raise ProgramError("{}: {} {}, got {}".format(where, name, _POSITIVE, out))
    return out


def _vec(value: Any, name: str, where: str, length: int = 3) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProgramError(
            "{}: {} must be a list of {} numbers, got {!r}".format(where, name, length, value)
        )
    return [_num(v, "{}[{}]".format(name, i), where, positive=False) for i, v in enumerate(value)]


def _points(value: Any, where: str) -> List[List[float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProgramError("{}: at must be a non-empty list of [x, y] points".format(where))
    return [_vec(p, "at[{}]".format(i), where, length=2) for i, p in enumerate(value)]


def _one_of(value: Any, options: Sequence[str], name: str, where: str) -> str:
    if value not in options:
        raise ProgramError(
            "{}: {} must be one of {}, got {!r}".format(where, name, ", ".join(options), value)
        )
    return str(value)


# --------------------------------------------------------------------------------------
# op validators
# --------------------------------------------------------------------------------------

def _v_box(op: Dict[str, Any], where: str) -> None:
    _vec(op.get("size"), "size", where)
    for i, v in enumerate(op["size"]):
        _num(v, "size[{}]".format(i), where)
    if "at" in op:
        _vec(op.get("at"), "at", where)


def _v_cylinder(op: Dict[str, Any], where: str) -> None:
    _num(op.get("diameter"), "diameter", where)
    _num(op.get("height"), "height", where)
    if "at" in op:
        _vec(op.get("at"), "at", where)


def _v_fillet(op: Dict[str, Any], where: str) -> None:
    _num(op.get("radius"), "radius", where)
    _one_of(op.get("select", "vertical"), EDGE_SELECTORS, "select", where)


def _v_chamfer(op: Dict[str, Any], where: str) -> None:
    _num(op.get("length"), "length", where)
    _one_of(op.get("select", "vertical"), EDGE_SELECTORS, "select", where)


def _v_shell(op: Dict[str, Any], where: str) -> None:
    _num(op.get("thickness"), "thickness", where)
    _one_of(op.get("open_face", "top"), FACE_SELECTORS, "open_face", where)


def _v_hole(op: Dict[str, Any], where: str) -> None:
    """Everything about a hole that the numbers alone can settle.

    What they cannot settle is whether the hole meets any *material*: that needs the solid, and
    the solid only exists inside the kernel. So the check is split, and both halves are numeric
    (rule 7). Here: a counterbore no wider than its hole, or as deep as the blind hole it is
    supposed to sit on top of, cuts nothing that the hole does not already cut. There: the
    worker anchors each cutter on the face it actually enters and refuses any that removes less
    than `MIN_REMOVED_MM3` (0.5 mm3), naming the same op index. A counterbore hanging in free
    air used to be a silent no-op in both halves; it is now an error in the second.
    """
    _num(op.get("diameter"), "diameter", where)
    _points(op.get("at"), where)
    depth = op.get("depth")
    if "depth" in op and depth is not None:
        _num(depth, "depth", where)
    for extra in ("counterbore", "countersink"):
        spec = op.get(extra)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise ProgramError("{}: {} must be an object".format(where, extra))
        _num(spec.get("diameter"), "{}.diameter".format(extra), where)
        _num(spec.get("depth"), "{}.depth".format(extra), where)
        if spec["diameter"] <= op["diameter"]:
            raise ProgramError(
                "{}: {}.diameter ({}) must exceed the hole diameter ({}) or it cuts "
                "nothing".format(where, extra, spec["diameter"], op["diameter"])
            )
        if depth is not None and spec["depth"] >= depth:
            raise ProgramError(
                "{}: {}.depth ({}) must be less than the hole depth ({}); both are measured "
                "down from the same face, so the hole would be entirely inside the {} and the "
                "hole itself would cut nothing".format(where, extra, spec["depth"], depth, extra)
            )


def _v_tapped_hole(op: Dict[str, Any], where: str) -> None:
    """`depth` is DRILL depth, and the usable thread is necessarily shorter than it.

    The worker cuts the pilot flat-bottomed to exactly `depth`, so as modelled the drill and the
    thread are the same length. No real tapped hole is: a tap cannot form a full thread in the
    last few pitches of a flat-bottomed hole, so a shop drills roughly one to two diameters
    deeper than the thread it wants, plus the drill point. Separating the two needs a second
    number in this vocabulary (`thread_depth` beside `depth`, or a `pilot_extra`) — a reviewed
    change to the op table, not a default invented in the worker. Until it is made, read this
    number as drill depth and specify a shorter usable thread than it says.
    """
    _one_of(op.get("size"), tuple(THREAD_PITCH), "size", where)
    _num(op.get("depth"), "depth", where)
    _points(op.get("at"), where)


OPS: Dict[str, Any] = {
    "box": _v_box,
    "cylinder": _v_cylinder,
    "cut_box": _v_box,
    "cut_cylinder": _v_cylinder,
    "fillet": _v_fillet,
    "chamfer": _v_chamfer,
    "shell": _v_shell,
    "hole": _v_hole,
    "tapped_hole": _v_tapped_hole,
}

# Ops that need an existing solid to act on; a program cannot start with one.
_MODIFIERS = ("fillet", "chamfer", "shell", "hole", "tapped_hole", "cut_box",
              "cut_cylinder")


def validate(program: Any) -> List[Dict[str, Any]]:
    """Check a feature program and return its op list. Raises `ProgramError` with a reason.

    Validation is total: every op is checked, and the first failure names the op index so a
    caller (or a model) can fix exactly one thing.
    """
    if not isinstance(program, dict):
        raise ProgramError("program must be an object with an 'ops' list")
    version = program.get("version", VERSION)
    if version != VERSION:
        raise ProgramError(
            "unsupported program version {!r}; this build understands {}".format(version, VERSION)
        )
    units = program.get("units", "mm")
    if units != "mm":
        raise ProgramError(
            "units must be 'mm' ({!r} given): PhiMech is millimetres throughout".format(units)
        )
    ops = program.get("ops")
    if not isinstance(ops, list) or not ops:
        raise ProgramError("program.ops must be a non-empty list")

    for index, op in enumerate(ops):
        where = "ops[{}]".format(index)
        if not isinstance(op, dict):
            raise ProgramError("{}: each op must be an object, got {!r}".format(where, op))
        name = op.get("op")
        if name not in OPS:
            raise ProgramError(
                "{}: unknown op {!r}; known ops are {}".format(
                    where, name, ", ".join(sorted(OPS))
                )
            )
        if index == 0 and name in _MODIFIERS:
            raise ProgramError(
                "{}: a program cannot start with {!r} because there is no solid yet; "
                "start with 'box' or 'cylinder'".format(where, name)
            )
        OPS[name](op, "{} ({})".format(where, name))
    return ops


def describe(program: Dict[str, Any]) -> str:
    """One line per op, for a human or a build report."""
    lines = []
    for index, op in enumerate(validate(program)):
        name = op["op"]
        if name in ("box", "cut_box"):
            detail = "size {}".format(op["size"])
        elif name in ("cylinder", "cut_cylinder"):
            detail = "d{} x {}".format(op["diameter"], op["height"])
        elif name == "fillet":
            detail = "r{} on {} edges".format(op["radius"], op.get("select", "vertical"))
        elif name == "chamfer":
            detail = "{} on {} edges".format(op["length"], op.get("select", "vertical"))
        elif name == "shell":
            detail = "{} mm wall, {} open".format(op["thickness"], op.get("open_face", "top"))
        elif name == "hole":
            detail = "d{} x{}".format(op["diameter"], len(op["at"]))
            if op.get("counterbore"):
                detail += " counterbored"
            if op.get("countersink"):
                detail += " countersunk"
        elif name == "tapped_hole":
            detail = "{} pilot x{}, {} deep".format(op["size"], len(op["at"]), op["depth"])
        else:
            detail = ""
        lines.append("{:>3}. {:<14} {}".format(index, name, detail))
    return "\n".join(lines)
