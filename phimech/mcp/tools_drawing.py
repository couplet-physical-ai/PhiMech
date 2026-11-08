"""MCP tool for the inspection drawing: ``drawing_sheet``.

Same shape as ``tools_preview.py`` for the same reasons: one ``register(mcp)`` hook wired by
``server.py``'s ``tools_*`` glob, a module-level ``*_impl`` body that is testable from the
system python, and the ``{"ok": ...}`` / absolute-path conventions every other tool uses. The
docstring on the nested wrapper is what the agent guide indexes, so it states units and limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

#: Names to append to ``server.TOOLS`` when wiring.
TOOLS = ["drawing_sheet"]


def _failure(error: Any, hint: str) -> Dict[str, Any]:
    message = str(error) or error.__class__.__name__
    return {"ok": False, "error": message, "hint": hint}


def _absolute(raw: str, label: str, *, must_exist: bool = False) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("%s is required" % label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("%s must be an absolute path (hint: pass /full/path/to/%s)"
                         % (label, label.replace("_", ".")))
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError("%s not found: %s" % (label, path))
    return path


def drawing_sheet_impl(
    path: str,
    out_svg: str,
    part: Optional[str] = None,
    title: str = "",
    png: bool = True,
    dxf: bool = False,
    pdf: bool = True,
) -> Dict[str, Any]:
    """The ``drawing_sheet`` body: validate paths, render, return the result as data."""
    try:
        source = _absolute(path, "path", must_exist=True)
        output = _absolute(out_svg, "out_svg")
        if output.exists() and output.is_dir():
            raise IsADirectoryError("out_svg is a directory: %s" % output)
        if output.suffix.lower() != ".svg":
            raise ValueError("out_svg must end in .svg (.png/.pdf copies are written beside it)")
        output.parent.mkdir(parents=True, exist_ok=True)
        from phimech.synth import drawing

        from phimech.synth import preview as preview_mod

        if source.is_dir() and not preview_mod.is_solidworks_bundle(source):
            result = drawing.for_build(source, part=part, out_svg=output, png=bool(png),
                                       title=title or None, dxf=bool(dxf), pdf=bool(pdf))
        else:
            result = drawing.render_drawing(source, output, title=title or None, part=part,
                                            png=bool(png), dxf=bool(dxf), pdf=bool(pdf))
        payload = result.to_dict()
        payload["ok"] = True
        payload["next"] = (
            "open `path` (or read the image at `png` when it is set) and check the callouts "
            "against the brief; `groups` is the same hole table as data - compare its sizes and "
            "positions with the intended ones numerically"
        )
        return payload
    except Exception as exc:  # noqa: BLE001 - every failure must reach the agent as data
        return _failure(
            exc,
            "pass absolute paths; `path` is an .stl, a .step/.stp, a CSG AST .json, a .scad "
            "file, a synth build directory or a SolidWorks handoff bundle (a design IR must be "
            "built first); run `phimech doctor` if OpenSCAD or FreeCAD is missing",
        )


def register(mcp: Any) -> None:
    """Attach ``drawing_sheet`` to a FastMCP server instance."""

    @mcp.tool
    def drawing_sheet(
        path: str,
        out_svg: str,
        part: Optional[str] = None,
        title: str = "",
        png: bool = True,
        dxf: bool = False,
        pdf: bool = True,
    ) -> Dict[str, Any]:
        """Dimensioned third-angle inspection drawing (SVG, + PDF/PNG) of a part, with a hole table.

        ``path`` must be an **absolute** path to an existing ``.stl``, ``.step``/``.stp``, CSG
        AST ``.json`` or ``.scad`` file, or to a ``synth build`` directory - then the largest
        printable part is drawn, or the one named by ``part`` - or to a SolidWorks handoff
        bundle directory, which is validated before it is drawn. **STEP is the SolidWorks
        route:** export STEP in mm from SolidWorks and pass it here. A STEP is tessellated
        through FreeCAD first and refused, not converted, if it declares anything but
        millimetres; a mesh-in-STEP and a translation whose kernel and mesh volumes disagree by
        over 1% are refused too. ``out_svg`` must be absolute and end in ``.svg``; a
        ``.png`` copy is written beside it when a rasteriser exists (``png`` in the result is
        null otherwise - the SVG is the deliverable), and an A3-landscape ``.pdf`` when
        rsvg-convert or a headless Chrome exists (``pdf`` null otherwise; the PDF is what a
        shop is sent, a picture of the sheet). ``title`` overrides the title block text.
        ``dxf`` additionally writes an AC1015 DXF beside the SVG: that is the **editable**
        artefact, which SolidWorks opens as real geometry with editable (though not
        associative) dimensions, where a PNG or a PDF of the same sheet is only a picture of
        one. Holes are true CIRCLE entities, mm is declared in ``$INSUNITS``, and geometry is
        1:1.

        What is on the sheet: top, front and right views (A3, standard scale chosen to fit);
        overall dimensions; one x/y position dimension per feature group from the part's
        lower-left bbox corner (the datum) plus the pitch to its nearest twin; leader callouts;
        the z-level heights beside the right view; a table of every hole, boss and window with
        qty, size, depth (``THRU`` or the z range) and model-coordinate positions. Every number
        is **measured from the mesh**: silhouettes and stepped sections are OpenSCAD 2D
        projections, circles are fitted to the polygon vertices, so a hole a build script
        over-cuts by EPS reads 0.02 larger, which is the truth. A STEP arrives through the same
        mesh path, so its analytic surfaces are tessellated before measurement. A counterbore
        *is* recovered, from the coaxial wall bands it leaves in the mesh. A **thread** is not,
        and cannot be: a tapped hole measures as its pilot diameter, so `M4x0.7` reads as
        `⌀2.5` unless the source supplies the thread. A chamfer reads as the shape it
        projects to, not as an angle.

        Units: mm throughout; ``render_s`` seconds. Limits: **advisory, never a gate** - nothing
        is asserted from it. Features are found as prisms along z (holes, bores, bosses, windows,
        steps) plus thru-holes seen in the side silhouettes; a freeform or angled feature shows
        in the silhouette but gets no callout. Thin lines are section edges at every z level and
        dashed hidden edges, not true hidden-line removal. OpenSCAD only; 1-3 s for a bracket,
        longer for a mesh with hundreds of z levels (capped at 14 slices). Refuses a design IR:
        build it first and pass the build directory.
        """
        return drawing_sheet_impl(path, out_svg, part, title, png, dxf, pdf)


__all__ = ["TOOLS", "drawing_sheet_impl", "register"]
