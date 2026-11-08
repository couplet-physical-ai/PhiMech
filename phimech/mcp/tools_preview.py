"""MCP tools for the preview loop: ``preview_sheet`` and ``preview_compare``.

Why a separate module
---------------------
``server.py`` is the one file every MCP tool lives in, and several workstreams add tools to it
at once. This module owns its two tools and exposes a single ``register(mcp)`` hook, so wiring it
is one import and one call in ``server.py`` (plus the names in ``TOOLS``), with no merge surface
inside the tool bodies. It deliberately does not import ``fastmcp`` or ``server`` itself, so the
``*_impl`` bodies are testable from the system python.

The layout follows the house pattern for an optional tool family (see ``tools_bosl2.py``): the
body is a module-level ``*_impl`` function, and ``register(mcp)`` decorates a nested wrapper with
``@mcp.tool``. That nesting is what ``phimech/guide`` indexes, so the tools appear in the agent
guide as soon as the coordinator wires the hook; a module-level ``mcp.tool(fn)`` call would work
at runtime but leave both tools invisible to the guide.

The path policy, units sentences and ``{"ok": ...}`` return shape follow ``server.py`` exactly,
because the agent reading these tools has learned those conventions from the other thirty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

#: Names to append to ``server.TOOLS`` when wiring.
TOOLS = ["preview_sheet", "preview_compare"]

_MIN_TILE_PX = 64
_MAX_TILE_PX = 2000


def _failure(error: Any, hint: str) -> Dict[str, Any]:
    """Agent-readable failure instead of an escaping exception (same shape as server.py)."""
    message = str(error) or error.__class__.__name__
    return {"ok": False, "error": message, "hint": hint}


def _absolute(raw: str, label: str, *, must_exist: bool = False) -> Path:
    """Require an absolute path so a client cannot accidentally address another checkout."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("%s is required" % label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "%s must be an absolute path (hint: pass /full/path/to/%s)"
            % (label, label.replace("_", "."))
        )
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError("%s not found: %s" % (label, path))
    return path


def _absolute_png_output(raw: str, label: str) -> Path:
    path = _absolute(raw, label)
    if path.exists() and path.is_dir():
        raise IsADirectoryError("%s is a directory: %s" % (label, path))
    if path.suffix.lower() != ".png":
        raise ValueError("%s must end in .png" % label)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def preview_sheet_impl(
    path: str,
    out_png: str,
    tile_px: int = 400,
    keep_views: bool = False,
    part: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``preview_sheet`` body: validate the paths, render the sheet, return it as data."""
    try:
        source = _absolute(path, "path", must_exist=True)
        output = _absolute_png_output(out_png, "out_png")
        size = int(tile_px)
        if not _MIN_TILE_PX <= size <= _MAX_TILE_PX:
            raise ValueError(
                "tile_px must be between %d and %d pixels" % (_MIN_TILE_PX, _MAX_TILE_PX)
            )
        from phimech.synth import preview

        if source.is_dir():
            result = preview.for_build(
                source, part=part, tile_px=size, keep_views=bool(keep_views),
                filename=output.name,
            )
            if Path(result.path) != output:
                import shutil

                shutil.copyfile(result.path, str(output))
                result.path = str(output)
        else:
            result = preview.render_sheet(
                source, output, tile_px=size, keep_views=bool(keep_views)
            )
        payload = result.to_dict()
        payload["ok"] = True
        payload["next"] = (
            "read the image at `path` and check each view against the brief; use "
            "preview_compare against a reference sheet for a numeric silhouette check"
        )
        return payload
    except Exception as exc:  # noqa: BLE001 - every failure must reach the agent as data
        return _failure(
            exc,
            "pass absolute paths; `path` is an .stl, a CSG AST .json, a .scad file or a synth "
            "build directory (a design IR must be built first); run `phimech doctor` if "
            "OpenSCAD is missing",
        )


def preview_compare_impl(
    reference_png: str,
    candidate_png: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """The ``preview_compare`` body: per-view silhouette IoU between two sheets."""
    try:
        reference = _absolute(reference_png, "reference_png", must_exist=True)
        candidate = _absolute(candidate_png, "candidate_png", must_exist=True)
        from phimech.synth import preview

        minimum = preview.DEFAULT_IOU_THRESHOLD if threshold is None else float(threshold)
        if not 0.0 <= minimum <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        result = preview.compare_sheets(reference, candidate, threshold=minimum)
        result["ok"] = True
        result["reference_png"] = str(reference)
        result["candidate_png"] = str(candidate)
        return result
    except Exception as exc:  # noqa: BLE001 - every failure must reach the agent as data
        return _failure(
            exc,
            "both sheets must be absolute paths to PNGs written by preview_sheet with the same "
            "tile_px; the silhouette check cannot compare sheets of different tile sizes",
        )


def register(mcp: Any) -> None:
    """Attach the preview tools to a FastMCP server (``phimech/mcp/server.py`` calls this).

    The tools are nested so ``phimech/guide`` indexes them along with the server's own; each is a
    thin wrapper whose docstring is the agent-facing contract and whose body is the ``*_impl``
    function above.
    """

    @mcp.tool
    def preview_sheet(
        path: str,
        out_png: str,
        tile_px: int = 400,
        keep_views: bool = False,
        part: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Render the 7-view contact sheet (iso, front, back, left, right, top, bottom) of a part.

        ``path`` must be an **absolute** path to an existing ``.stl``, CSG AST ``.json`` or
        ``.scad`` file, or to a ``synth build`` output directory - then the enclosure's largest
        printable part is used, or the one named by ``part``. ``out_png`` must be absolute and end
        in ``.png``; the sheet is written there (4 x 2 tiles, view names burnt in, last tile =
        measured bbox and volume). With ``keep_views`` each view is also written beside it as
        ``<stem>-<view>.png``.

        Units: ``bbox_mm`` and ``volume_mm3`` are millimetres and cubic millimetres measured from
        the rendered mesh; ``render_s`` is seconds; ``tile_px`` is pixels per view (64-2000,
        default 400). Limits: **advisory, never a gate** - no gate verdict is derived from this
        image and none is returned; its purpose is that you *look at the image at* ``path``
        *before deciding the part is right*, because every numeric gate passes on a box where a
        cylinder was asked for. Renders through OpenSCAD only (about 1-3 s for an enclosure),
        writes only the files named above, and refuses a design IR: build it first and pass the
        build directory.
        """
        return preview_sheet_impl(path, out_png, tile_px, keep_views, part)

    @mcp.tool
    def preview_compare(
        reference_png: str,
        candidate_png: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Silhouette IoU per view between two preview sheets: the deterministic shape check.

        ``reference_png`` and ``candidate_png`` must be **absolute** paths to existing sheets made
        by ``preview_sheet`` with the same ``tile_px`` and default views. ``threshold`` is the
        minimum per-view IoU (0-1) for ``same_shape`` to be true; the default is the calibrated
        0.92.

        Units: IoU is a dimensionless ratio of silhouette pixels (1.0 = identical outline); the
        label band and tile border are excluded. Limits: **silhouette-only** - because each view
        is scaled to fit, this compares shape, not size (size is the numeric gates' job), and a
        concavity that never reaches the outline (an open-front tray versus an open-top tray) is
        invisible to it; ``same_shape`` is therefore a floor for "looks right", not a proof. Reads
        the two PNGs, writes nothing, runs in well under a second.
        """
        return preview_compare_impl(reference_png, candidate_png, threshold)


__all__ = ["TOOLS", "preview_compare_impl", "preview_sheet_impl", "register"]
