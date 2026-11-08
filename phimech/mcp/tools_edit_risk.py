"""MCP tools for the import edit-risk assessment.

Wired by the coordinator from ``phimech/mcp/server.py`` with one line::

    from phimech.mcp import tools_edit_risk
    tools_edit_risk.register(mcp)

The module deliberately imports nothing from ``server.py`` (no shared ``_absolute_path``), so
it can be registered on any FastMCP instance and unit-tested with a stand-in that only has a
``tool`` decorator.  The assessment itself is stdlib and kernel-free, so the tool answers in
milliseconds and never spawns FreeCAD or OpenSCAD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

#: Tool names this module registers, for the handshake/registration tests.
TOOLS = ["import_edit_risk"]


def _absolute_file(raw: str, label: str) -> Path:
    """Require an absolute, existing file so a client cannot address another checkout."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("%s is required" % label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "%s must be an absolute path (hint: pass /full/path/to/model.step)" % label
        )
    path = path.resolve()
    if not path.is_file():
        if path.is_dir():
            raise ValueError("%s must be a file, but %s is a directory" % (label, path))
        raise FileNotFoundError("%s not found: %s" % (label, path))
    return path


def register(mcp: Any) -> None:
    """Decorate the edit-risk tools onto ``mcp`` (a FastMCP instance or a stand-in)."""

    @mcp.tool
    def import_edit_risk(path: str) -> Dict[str, Any]:
        """Assess how safely an imported STEP/STL can be edited - call this BEFORE filleting,
        cutting or unioning an imported vendor model, and act on ``workflow``.

        ``path`` must be an **absolute** path to an existing ``.step``, ``.stp`` or ``.stl``
        file. Returns ``level`` (``low`` | ``medium`` | ``high`` | ``hostile`` | ``unknown``),
        ``workflow`` (``boolean`` = cut/union directly; ``repair`` = close the open shell first;
        ``wrap`` = design the enclosure around ``bbox_mm`` and never touch the import),
        ``guidance`` sentences, the ``reasons`` with the triggering numbers, the metrics
        (``faces``, ``edges``, ``free_edges``, ``nonmanifold_edges``, ``shells``, ``solids``,
        ``is_closed``, ``is_manifold``, ``faceted``, ``bbox_mm``, ``volume_mm3``) and the
        ``thresholds`` that decided the level.

        Units: millimetres throughout (``bbox_mm``, ``volume_mm3``); STEP files declaring
        another unit are scaled to mm for the bbox and reported in ``step.length_unit``.

        Limits: this is a file parse, not a kernel check. It **cannot** detect geometric
        self-intersection between well-formed faces - only the import path folds in FreeCAD's
        ``isValid()`` as ``kernel_valid`` - and ``provenance`` says per metric what is exact and
        what is a proxy. STL ``volume_mm3`` is omitted for an open mesh, STEP ``bbox_mm`` is
        omitted when any face is curved, and ``level: unknown`` means the file did not read as
        geometry and must be treated as hostile, never as fine. Read-only: writes nothing and
        spawns no CAD process. Refuses relative paths and any other suffix.
        """
        try:
            resolved = _absolute_file(path, "path")
            from phimech.interop.edit_risk import assess

            result = assess(resolved).to_dict()
            result["ok"] = True
            return result
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc) or exc.__class__.__name__,
                "hint": "pass an absolute path to an existing .step, .stp or .stl file",
            }

    # Exposed for tests and for a coordinator that wants to list what was registered.
    register.registered = {"import_edit_risk": import_edit_risk}  # type: ignore[attr-defined]


__all__ = ["TOOLS", "register"]
