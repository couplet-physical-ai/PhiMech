"""The B-rep worker — the only file in PhiMech that imports a geometry kernel.

It is never imported by the PhiMech process. `kernel.py` runs it as a subprocess under a
python that has `build123d` installed (`build/brep-venv`), which is what keeps rule 5 intact:
no OCP in our interpreter, no ABI roulette, and a kernel crash kills a child, not a build.

Contract
--------
* **stdin** — one JSON object, in one of two shapes:
  * `{"program": {...}, "outputs": {"step": "/abs/path", ...}}` — build one.
  * `{"programs": [{"program": {...}, "outputs": {...}}, ...]}` — build several in ONE kernel
    start. `import build123d` costs ~3 s, more than most builds, so N programs should not pay
    it N times. The reply is `{"ok": ..., "results": [...]}`, one verdict per program in order;
    a refusal in the middle does not cost the rest their build. Still one bounded subprocess
    that exits when it is done — rule 5 bans a kernel in *our* interpreter and bans a daemon,
    not more than one program per invocation.
* **stdout** — exactly one JSON object, and nothing else, ever.
* **stderr** — human-readable progress and diagnostics.
* exit 0 on success, 1 on a handled failure (the JSON says why), 2 on a broken request.

Protecting stdout is not optional here. OCCT's `Message_DefaultMessenger` writes to **file
descriptor 1 directly**, bypassing `sys.stdout` entirely, so a kernel warning lands in the
middle of our JSON and the parse fails. `_redirect_native_stdout()` moves fd 1 to fd 2 for the
duration of every kernel call. The idea is from agentcad's `native_io.py`
(https://github.com/jdilla1277/agentcad, Copyright James Dillard, Apache-2.0); this is an
independent implementation, and the reasoning is recorded because it is invisible until it
bites.

Where a blind feature is cut
----------------------------
Every blind cutter -- a ``depth``-limited hole, a counterbore pocket, a countersink cone -- is
anchored on **the highest material face on that feature's own axis**, found by intersecting the
solid with a hair-thin vertical probe at ``(x, y)`` *before* anything is cut there. The obvious
shortcut, anchoring on the whole solid's bounding-box ``z_max``, is right only when the feature
happens to sit on the topmost face of the entire part. On the shipped L-bracket -- a 60x40x8
plate under a 45 mm upstand -- it put all three counterbore pockets at z 41..45, 33 mm above the
plate they belong to, in free air. They removed nothing, and the part measured identically with
and without them, because a boolean whose result is empty is not an error.

So it is one now. ``MIN_REMOVED_MM3`` is the gate (the core boundary: every gate is numeric and
has a threshold): each individual cutter must remove at least 0.5 mm3 or the program is refused
with exit 2 and the op index. Per *cutter*, not per op -- the bug above hid inside a ``hole`` op
that removed 510 mm3 of through-holes while its three counterbores removed exactly zero.

The opposite failure has a gate too. Anchoring a recess correctly is only half the job: a
counterbore deeper than the material it enters stops being a recess and becomes a wider through
hole. The shipped enclosure declared exactly that -- ``shell`` thickness 2 and a counterbore
``depth`` 3 -- and got away with it for as long as the pocket was being cut in mid-air. The
moment the pocket landed on the floor, four M3 mounting holes would have quietly become 6 mm
clearance holes. ``MIN_REMAINING_WALL_MM`` refuses it, reporting the measured remaining wall.

Known limitation: ``tapped_hole`` pilot depth
---------------------------------------------
``tapped_hole`` cuts the pilot flat-bottomed to exactly ``depth``, so modelled drill depth
equals modelled thread depth. No real tapped hole is like that: a tap cannot form a full thread
in the last few pitches of a flat-bottomed hole, so the drill has to run deeper than the usable
thread -- the shop allowance is roughly one to two diameters of extra drill, plus the drill
point itself. Representing that needs a second number in the vocabulary (``thread_depth``
alongside ``depth``, or a ``pilot_extra``), which is a program-vocabulary change and therefore a
reviewed decision, not a default this file should invent. Until it is made, read
``tapped_hole.depth`` as *drill* depth and specify a shorter usable thread than the number says.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


#: A single cutter that removes less than this much material is refused rather than silently
#: ignored. Rule 7: a gate that prints "looks fine" is not a gate.
MIN_REMOVED_MM3 = 0.5

#: A counterbore or countersink must leave at least this much material between its floor and
#: the far side of the run it enters. Zero is not enough: a recess whose floor is coplanar with
#: the far face is a through hole wearing a counterbore's name. This is a *breach* gate, not a
#: strength one -- minimum printable wall is `phimech/check`'s job, and it is a larger number.
MIN_REMAINING_WALL_MM = 0.01

#: Radius of the vertical probe that finds the face a blind feature enters. A point has no
#: volume and OCCT cannot intersect one, so the probe is a hair-thin cylinder: narrow enough to
#: sample the axis rather than its neighbourhood, wide enough to keep the boolean conditioned.
_PROBE_RADIUS_MM = 0.01

#: How far a cutter over-runs the face it enters. A cut that stops exactly on a face leaves a
#: coplanar skin that some kernels keep, so every blind cutter pokes out by this much.
_OVERSHOOT_MM = 1.0


class GateError(RuntimeError):
    """A cutter that does not meet material, or a feature with no face to enter.

    Distinct from a kernel failure: the *program* is wrong, not the build. It leaves the worker
    by exit 2 with a ``gate`` key in the JSON, and `kernel.py` re-raises it as a `ProgramError`
    so the CLI reports it the same way it reports any other refused program.
    """


@contextlib.contextmanager
def _redirect_native_stdout():
    """Point fd 1 at fd 2 so C-level writes cannot corrupt the JSON on stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _fail(message: str, *, code: int = 1, **extra: Any) -> int:
    payload: Dict[str, Any] = {"ok": False, "error": message}
    payload.update(extra)
    print(json.dumps(payload))
    return code


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def _select_edges(shape: Any, how: str, bd: Any) -> Any:
    """Resolve an edge selector to an edge list.

    `vertical` means edges parallel to Z — on a box those are the four corner edges, which is
    what "rounded corners" means. It is emphatically NOT the top and bottom rims; picking those
    by mistake is the classic build123d footgun, so each selector is spelled out.
    """
    edges = shape.edges()
    if how == "all":
        return edges
    if how == "vertical":
        return edges.filter_by(bd.Axis.Z)
    zs = [e.center().Z for e in edges]
    if not zs:
        return edges
    top_z, bottom_z = max(zs), min(zs)
    tol = 1e-6
    if how == "top":
        return bd.ShapeList([e for e in edges if abs(e.center().Z - top_z) < tol])
    if how == "bottom":
        return bd.ShapeList([e for e in edges if abs(e.center().Z - bottom_z) < tol])
    if how == "top_and_bottom":
        return bd.ShapeList(
            [e for e in edges
             if abs(e.center().Z - top_z) < tol or abs(e.center().Z - bottom_z) < tol]
        )
    raise ValueError("unknown edge selector {!r}".format(how))


def _face_normal_axis(how: str, bd: Any) -> Tuple[Any, str]:
    table = {
        "top": (bd.Axis.Z, "max"), "bottom": (bd.Axis.Z, "min"),
        "right": (bd.Axis.X, "max"), "left": (bd.Axis.X, "min"),
        "back": (bd.Axis.Y, "max"), "front": (bd.Axis.Y, "min"),
    }
    if how not in table:
        raise ValueError("unknown face selector {!r}".format(how))
    return table[how]


def _pick_face(shape: Any, how: str, bd: Any) -> Any:
    axis, end = _face_normal_axis(how, bd)
    faces = shape.faces().filter_by(axis)
    if not faces:
        faces = shape.faces()
    grouped = faces.group_by(axis)
    return (grouped[-1] if end == "max" else grouped[0])[0]


def _entry_run_at(shape: Any, x: float, y: float, bd: Any) -> Optional[Tuple[float, float]]:
    """`(top_z, bottom_z)` of the topmost *contiguous* run of material on the axis at `(x, y)`.

    `top_z` is the face a blind feature actually enters, and it is *measured* rather than
    assumed, because the cheap assumption -- "the top of the part" -- is wrong for every feature
    that is not on the part's topmost face, and wrong without any symptom.

    `bottom_z` is the far side of that same run, and it is what makes a breach checkable: a
    3 mm counterbore in a 2 mm floor is not a recess, it is a wider hole. Contiguity matters --
    a thin vertical probe through a lid and the boss below it returns two solids, and only the
    one the feature enters bounds it. Returns None when the axis misses the solid entirely.
    """
    bbox = shape.bounding_box()
    probe = bd.Cylinder(radius=_PROBE_RADIUS_MM, height=(bbox.max.Z - bbox.min.Z) + 2.0,
                        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    probe = probe.translate((x, y, bbox.min.Z - 1.0))
    hit = shape.intersect(probe)
    if hit is None:
        return None
    boxes = [solid.bounding_box() for solid in hit.solids()]
    if not boxes:
        return None
    entry = max(boxes, key=lambda box: box.max.Z)
    return float(entry.max.Z), float(entry.min.Z)


def _check_no_breach(spec: Dict[str, Any], kind: str, x: float, y: float,
                     face_z: float, far_z: float) -> None:
    """Refuse a recess that reaches the far side of the material it enters.

    Kernel-side and not in `program.py` on purpose. The declared numbers cannot say how much
    material sits under a feature: that is the accumulated solid's property, set by the shell
    thickness, the fillets, which face the axis happens to enter and whatever earlier ops cut
    away. Deriving it statically from `shell.thickness` would be right only for a feature
    entering a flat, uniform, previously-uncut shelled face, and silently wrong for a boss, a
    rib, a corner or anything already cut -- and silently wrong is the failure mode this whole
    change exists to remove. The kernel has probed the actual run; it can simply measure.

    Scoped to counterbores and countersinks, which are recesses *by definition*. A blind `hole`
    whose `depth` exceeds the material is a through hole spelled out the long way, which is
    legible rather than wrong, so it is left alone.
    """
    material = face_z - far_z
    remaining = material - spec["depth"]
    if remaining < MIN_REMAINING_WALL_MM:
        raise GateError(
            "{} at ({}, {}) is {:g} mm deep into the {:.4f} mm of material it enters at "
            "z={:g}, leaving {:.4f} mm of wall at z={:g}: it reaches the far face, so it is a "
            "wider through hole and not a recess".format(
                kind, x, y, spec["depth"], material, face_z, remaining, far_z)
        )


def _remove(shape: Any, cutter: Any, what: str) -> Any:
    """Subtract `cutter`, and prove it did something.

    `what` names the individual cutter rather than only the op, because one `hole` op can hold a
    through-hole that works and a counterbore that misses the part entirely -- which is exactly
    the failure this gate exists for, and an op-level total would have hidden it.
    """
    before = float(shape.volume)
    result = shape - cutter
    removed = before - float(result.volume)
    if removed < MIN_REMOVED_MM3:
        raise GateError(
            "{} removed {:.4f} mm3, under the {} mm3 minimum: the cutter does not meet "
            "material".format(what, removed, MIN_REMOVED_MM3)
        )
    return result


def _build(ops: List[Dict[str, Any]], bd: Any) -> Any:
    """Apply the op list in order to one evolving solid."""
    shape = None
    for index, op in enumerate(ops):
        name = op["op"]
        try:
            if name == "box":
                w, d, h = op["size"]
                solid = bd.Box(w, d, h, align=(bd.Align.MIN,) * 3)
                at = op.get("at")
                if at:
                    solid = solid.translate(tuple(at))
                shape = solid if shape is None else shape + solid
            elif name == "cylinder":
                solid = bd.Cylinder(radius=op["diameter"] / 2.0, height=op["height"],
                                    align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
                at = op.get("at")
                if at:
                    solid = solid.translate(tuple(at))
                shape = solid if shape is None else shape + solid
            elif name == "cut_box":
                w, d, h = op["size"]
                solid = bd.Box(w, d, h, align=(bd.Align.MIN,) * 3)
                at = op.get("at")
                if at:
                    solid = solid.translate(tuple(at))
                shape = _remove(shape, solid, "cut_box {} at {}".format(
                    op["size"], at or [0, 0, 0]))
            elif name == "cut_cylinder":
                solid = bd.Cylinder(radius=op["diameter"] / 2.0, height=op["height"],
                                    align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
                at = op.get("at")
                if at:
                    solid = solid.translate(tuple(at))
                shape = _remove(shape, solid, "cut_cylinder d{} at {}".format(
                    op["diameter"], at or [0, 0, 0]))
            elif name == "fillet":
                edges = _select_edges(shape, op.get("select", "vertical"), bd)
                if not edges:
                    raise ValueError("selector {!r} matched no edges".format(
                        op.get("select", "vertical")))
                shape = bd.fillet(edges, radius=op["radius"])
            elif name == "chamfer":
                edges = _select_edges(shape, op.get("select", "vertical"), bd)
                if not edges:
                    raise ValueError("selector {!r} matched no edges".format(
                        op.get("select", "vertical")))
                shape = bd.chamfer(edges, length=op["length"])
            elif name == "shell":
                face = _pick_face(shape, op.get("open_face", "top"), bd)
                shape = bd.offset(shape, amount=-abs(op["thickness"]), openings=face)
            elif name in ("hole", "tapped_hole"):
                shape = _cut_holes(shape, op, bd)
            else:  # pragma: no cover - validate() rejects unknown ops before we get here
                raise ValueError("unhandled op {!r}".format(name))
        except GateError as exc:
            # Re-raised, not wrapped in RuntimeError: the op index belongs in the message but
            # the *kind* of failure has to survive as far as main(), which exits 2 on it.
            raise GateError("ops[{}] ({}): {}".format(index, name, exc))
        except Exception as exc:
            raise RuntimeError(
                "ops[{}] ({}) failed: {}: {}".format(index, name, type(exc).__name__, exc)
            )
        if shape is None:
            raise RuntimeError("ops[{}] ({}) produced no solid".format(index, name))
    return shape


def _countersink_cone(cs: Dict[str, Any], hole_diameter: float, x: float, y: float,
                      face_z: float, bd: Any) -> Any:
    """A cone WIDE at the entry face, narrowing down to the hole — the screw-head shape.

    `bd.Cone(bottom_radius=..., top_radius=...)` puts `bottom_radius` at the bottom, so the
    obvious spelling (`bottom_radius=cs["diameter"] / 2`) builds the cone upside down: an
    undercut that flares *below* the surface, which no drill cuts and no screw head sits in.
    Measured on a 40x40x10 plate the flipped cone also leaves an extra downward-facing annulus
    where it meets the drill, which is how the tests tell the two apart.

    The cone is extended `_OVERSHOOT_MM` above the face at the same half-angle, so the taper
    inside the material is untouched and no coplanar skin can survive at the surface.
    """
    top_r = cs["diameter"] / 2.0
    bottom_r = hole_diameter / 2.0
    if top_r <= bottom_r:
        raise GateError(
            "countersink diameter {} does not exceed the hole diameter {:.4g}: there is no cone "
            "to cut".format(cs["diameter"], hole_diameter)
        )
    flare = (top_r - bottom_r) / cs["depth"]      # radius gained per mm of depth
    cone = bd.Cone(bottom_radius=bottom_r, top_radius=top_r + flare * _OVERSHOOT_MM,
                   height=cs["depth"] + _OVERSHOOT_MM,
                   align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX))
    return cone.translate((x, y, face_z + _OVERSHOOT_MM))


def _cut_holes(shape: Any, op: Dict[str, Any], bd: Any) -> Any:
    """Cut plain, counterbored, countersunk or tapping-pilot holes at a list of XY points.

    A hole with no `depth` goes through: the cutter spans the solid's whole Z extent plus a
    margin at each end, because a cut that stops exactly on a face leaves a coplanar skin that
    some kernels keep. Everything else is blind, and its depth is measured down from the face
    the feature enters — the highest material on that feature's own axis, not the top of the
    part. The module docstring records what the second reading cost.
    """
    bbox = shape.bounding_box()
    z_min, z_max = bbox.min.Z, bbox.max.Z
    span = (z_max - z_min) + 2.0
    is_tapped = op["op"] == "tapped_hole"

    if is_tapped:
        from phimech.brep.program import THREAD_PITCH
        pitch = THREAD_PITCH[op["size"]]
        nominal = float(op["size"][1:])
        # Standard tapping-drill approximation: minor diameter = d - 1.0825 * pitch. No helix is
        # modelled -- see THREAD_PITCH in program.py for why that is deliberate. `depth` is the
        # DRILL depth; see the module docstring on why the thread is not modelled shorter.
        diameter = nominal - 1.0825 * pitch
        depth = op["depth"]
        kind = "{} pilot".format(op["size"])
    else:
        diameter = op["diameter"]
        depth = op.get("depth")
        kind = "d{} hole".format(diameter)

    for point in op["at"]:
        x, y = point
        # Probe before cutting anything at this point: once the hole is drilled there is no
        # material left on the axis, so the counterbore and countersink would find no face.
        run = _entry_run_at(shape, x, y, bd)
        if run is None:
            raise GateError(
                "nothing to drill at ({}, {}): the axis misses the solid entirely, so every "
                "cutter placed there would be cut in free space".format(x, y)
            )
        face_z, far_z = run
        where = "at ({}, {}) entering the face at z={:g}".format(x, y, face_z)

        if depth is None:
            cutter = bd.Cylinder(radius=diameter / 2.0, height=span,
                                 align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            cutter = cutter.translate((x, y, z_min - 1.0))
        else:
            cutter = bd.Cylinder(radius=diameter / 2.0, height=depth + _OVERSHOOT_MM,
                                 align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX))
            cutter = cutter.translate((x, y, face_z + _OVERSHOOT_MM))
        shape = _remove(shape, cutter, "{} {}".format(kind, where))

        cb = op.get("counterbore")
        if cb:
            _check_no_breach(cb, "counterbore", x, y, face_z, far_z)
            pocket = bd.Cylinder(radius=cb["diameter"] / 2.0,
                                 height=cb["depth"] + _OVERSHOOT_MM,
                                 align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX))
            pocket = pocket.translate((x, y, face_z + _OVERSHOOT_MM))
            shape = _remove(shape, pocket, "counterbore {}".format(where))
        cs = op.get("countersink")
        if cs:
            _check_no_breach(cs, "countersink", x, y, face_z, far_z)
            cone = _countersink_cone(cs, diameter, x, y, face_z, bd)
            shape = _remove(shape, cone, "countersink {}".format(where))
    return shape


def _measure(shape: Any) -> Dict[str, Any]:
    bbox = shape.bounding_box()
    return {
        "volume_mm3": round(float(shape.volume), 4),
        "area_mm2": round(float(shape.area), 4),
        "bbox": {
            "min": [round(bbox.min.X, 4), round(bbox.min.Y, 4), round(bbox.min.Z, 4)],
            "max": [round(bbox.max.X, 4), round(bbox.max.Y, 4), round(bbox.max.Z, 4)],
            "size": [round(bbox.size.X, 4), round(bbox.size.Y, 4), round(bbox.size.Z, 4)],
        },
        "faces": len(shape.faces()),
        "edges": len(shape.edges()),
        "solids": len(shape.solids()),
    }


#: Output formats the worker can write, checked before any geometry runs so a typo costs
#: nothing and a batch never dies half-written.
WRITABLE_FORMATS = ("step", "stp", "stl", "brep")


def _write(shape: Any, fmt: str, target: str, bd: Any) -> None:
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if fmt in ("step", "stp"):
        bd.export_step(shape, target)
    elif fmt == "stl":
        bd.export_stl(shape, target)
    else:
        bd.export_brep(shape, target)


def _run_one(ops: List[Dict[str, Any]], outputs: Dict[str, Any], bd: Any) -> Dict[str, Any]:
    """Build one op list, write its outputs, and return that program's own verdict.

    Never raises. In a batch every program needs a verdict of its own: one refusal must not
    cost the programs after it the kernel start they have already paid for, and the caller
    still has to be able to tell a gate refusal from a kernel failure per program.
    """
    try:
        with _redirect_native_stdout():
            shape = _build(ops, bd)
            measurements = _measure(shape)
            written: Dict[str, str] = {}
            for fmt, path in sorted(outputs.items()):
                _write(shape, fmt, str(path), bd)
                written[fmt] = str(path)
    except GateError as exc:
        return {"ok": False, "error": str(exc), "gate": "min_removed_mm3"}
    except Exception as exc:  # noqa: BLE001 - the verdict carries the reason
        return {"ok": False, "error": str(exc),
                "traceback": "".join(traceback.format_exc()).splitlines()[-6:]}
    return {"ok": True, "measurements": measurements, "written": written}


def main(argv: Optional[List[str]] = None) -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        return _fail("request is not valid JSON: {}".format(exc), code=2)

    # Two request shapes. `program` is the original one and its reply is unchanged. `programs`
    # is a batch: several builds in ONE kernel start, because `import build123d` costs ~3 s --
    # more than most builds -- and paying it per program is what turned a 47-test suite into
    # ten minutes of startup. Still one bounded subprocess that exits when it is done; rule 5
    # forbids importing a kernel into our interpreter and keeping a daemon, not building more
    # than one program per invocation.
    batch = request.get("programs")
    if batch is None:
        program = request.get("program")
        if not isinstance(program, dict):
            return _fail("request.program must be an object", code=2)
        items = [(program, request.get("outputs") or {})]
    elif isinstance(batch, list) and batch:
        if not all(isinstance(item, dict) and isinstance(item.get("program"), dict)
                   for item in batch):
            return _fail("each entry of request.programs must be {program, outputs}", code=2)
        items = [(item["program"], item.get("outputs") or {}) for item in batch]
    else:
        return _fail("request.programs must be a non-empty list", code=2)

    def _where(index: int) -> str:
        return "" if batch is None else " [{}]".format(index)

    try:
        sys.path.insert(0, str(request.get("repo") or os.getcwd()))
        from phimech.brep.program import validate
    except Exception as exc:
        return _fail("invalid program: {}".format(exc), code=2)
    validated: List[List[Dict[str, Any]]] = []
    for index, (program, outputs) in enumerate(items):
        try:
            validated.append(validate(program))
        except Exception as exc:
            return _fail("invalid program{}: {}".format(_where(index), exc), code=2)
        for fmt in outputs:
            if fmt not in WRITABLE_FORMATS:
                return _fail("cannot write format {!r}{}".format(fmt, _where(index)), code=2)

    try:
        with _redirect_native_stdout():
            import build123d as bd
    except Exception as exc:
        return _fail("build123d is not importable in this interpreter: {}".format(exc), code=2,
                     hint="run `bin/phimech brep setup`")

    results = [_run_one(ops, outputs, bd)
               for ops, (_, outputs) in zip(validated, items)]
    version = getattr(bd, "__version__", "unknown")

    if batch is not None:
        print(json.dumps({
            "ok": all(item["ok"] for item in results), "results": results,
            "kernel": "build123d", "kernel_version": version,
        }))
        # Per-item `ok`/`gate`/`error` are the authoritative verdicts -- one exit code cannot
        # express N of them. This only says "something in there did not build".
        return 0 if all(item["ok"] for item in results) else 1

    result = results[0]
    if not result["ok"]:
        extra = {k: result[k] for k in ("gate", "traceback") if k in result}
        # Exit 2, the same code a malformed request gets: the program is wrong, not the kernel.
        # The `gate` key is what `kernel.py` keys off to re-raise this as a `ProgramError`.
        return _fail(result["error"], code=2 if "gate" in result else 1, **extra)
    print(json.dumps({
        "ok": True,
        "measurements": result["measurements"],
        "written": result["written"],
        "kernel": "build123d",
        "kernel_version": version,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
