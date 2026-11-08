"""MCP tools for the BOSL2 lane: what the vendored library offers and how to call it from the AST.

Coordinator hook: ``register(mcp)`` decorates the tools onto a ``FastMCP`` instance; nothing here
imports fastmcp itself, so ``bosl2_modules_impl()`` is testable from the system python and the
module can be imported by ``phimech.mcp.server`` without changing its import order.

Why a discovery tool rather than more docs: models guess module names and argument spellings, and
OpenSCAD's response to a wrong guess is a *warning* about an unknown module followed by an empty
render. The table here is short, hand-curated, and checked against the vendored source by
``tests/test_bosl2.py``, so what the tool says exists, exists.
"""

from __future__ import annotations

from typing import Any, Dict, List

from phimech.vendor import bosl2

_HINT = "call with no query for the whole table, or a substring of a module name or summary"

#: The one-node example every answer carries, so a caller has a correct shape to copy.
_EXAMPLE_NODE: Dict[str, Any] = {
    "kind": "scad_module",
    "module": "threaded_rod",
    "args": {"d": 3, "l": 12, "pitch": 0.5, "$fn": 48},
    "include": "BOSL2/threading.scad",
    "children": [],
}

_RULES: List[str] = [
    "Units are millimetres and degrees, like the rest of the CSG AST.",
    "OpenSCAD-only: the freecad backend refuses any tree containing a scad_module before "
    "launching, so no STEP or FCStd export exists for this lane; render stl/3mf/obj/png.",
    "`args` values must be JSON scalars or arrays. Profiles and paths are point lists "
    "([[x, y], ...]); function calls such as circle(d=5) or arc(r=15) cannot appear in arguments.",
    "`$fn` is an ordinary named argument (\"$fn\": 48) and controls a module's tessellation.",
    "`include` names a vendored file relative to phimech/vendor (e.g. BOSL2/screws.scad); "
    "BOSL2/std.scad is added automatically. Anything outside phimech/vendor is rejected.",
    "Children become the module's child block, which BOSL2 uses for attachments - a sweep's "
    "profile is an argument, not a child.",
    "Subtract screw_hole()/threaded_rod(internal=true) with a Difference node; the core Hull "
    "node hulls children, BOSL2's hull_points() takes a point list.",
]


def bosl2_modules_impl(query: str = "") -> Dict[str, Any]:
    """The tool body: includes, curated modules (optionally filtered), the pin, and the rules."""
    needle = (query or "").strip().lower()
    modules = [dict(entry) for entry in bosl2.CURATED_MODULES]
    if needle:
        modules = [
            entry for entry in modules
            if needle in entry["module"].lower() or needle in entry["summary"].lower()
        ]
    return {
        "ok": True,
        "library": bosl2.LIBRARY,
        "licence": "BSD-2-Clause (THIRD_PARTY_NOTICES.md)",
        "pinned": bosl2.pinned(),
        "openscad_only": True,
        "step_export": False,
        "units": "mm, degrees",
        "query": needle,
        "includes": bosl2.include_files(),
        "modules": modules,
        "example": dict(_EXAMPLE_NODE),
        "rules": list(_RULES),
    }


def register(mcp: Any) -> None:
    """Attach the BOSL2 tools to a FastMCP server (``phimech/mcp/server.py`` calls this)."""

    @mcp.tool
    def bosl2_modules(query: str = "") -> Dict[str, Any]:
        """List the vendored BOSL2 library and the curated modules an agent should reach for.

        Returns every include file under ``phimech/vendor/BOSL2`` (as the ``include`` strings a
        ``scad_module`` CSG node takes), a curated table of the most useful modules - threaded_rod,
        screw, screw_hole, nut, cuboid with rounding, cyl with chamfer, tube, prismoid,
        path_sweep, offset_sweep, linear_sweep, skin, rounded_prism - with their key arguments, the
        pinned upstream commit, a copy-ready example node, and the rules for writing arguments.
        ``query`` filters the table by a substring of the module name or summary; empty lists all.

        Units: every dimension is millimetres and every angle degrees; the M3 rod example
        measures 3.0 mm across and 12.0 mm long. Limits: OpenSCAD-only - a tree containing a
        ``scad_module`` renders on the openscad backend (stl, 3mf, obj, png) and the freecad
        backend refuses it, so **no STEP export** exists from this lane. Read-only: builds no
        geometry and launches no subprocess; the table is curated, not the whole library, and
        argument values must be JSON literals (point lists), never OpenSCAD function calls.
        """
        try:
            return bosl2_modules_impl(query)
        except Exception as exc:
            return {"ok": False, "error": str(exc) or exc.__class__.__name__, "hint": _HINT}


TOOLS = ["bosl2_modules"]

__all__ = ["TOOLS", "bosl2_modules_impl", "register"]
