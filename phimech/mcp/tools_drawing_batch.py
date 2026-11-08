"""MCP tool and CLI for the batch drawing surface: ``drawing_batch``.

Why this exists
---------------
``drawing_sheet`` draws one part. The thing an agent is actually asked to do is *"here are the
files, go read them and create drawings"* - a folder of STEP exports, a handful of STLs, a
finished ``synth build``, a SolidWorks handoff - and get back one dimensioned sheet per part plus
a machine-readable ledger of what worked. Doing that with N ``drawing_sheet`` calls means the
agent has to invent the discrimination rules itself (is this directory a *build*, a *handoff
bundle*, or a *folder of parts*?) and one bad file aborts the run. This module owns those rules
once, and turns every failure into a row rather than an exception.

Design decisions, and why
-------------------------
* **The directory discrimination is the whole job.** Three directory shapes mean three different
  things and must never be confused: a **SolidWorks handoff bundle** (validated, one source - see
  ``preview.is_solidworks_bundle``), a **synth build** (one source, the largest printable part -
  ``drawing.for_build``), and a plain **folder of CAD files** (one source *per file*). Get it
  wrong and a bundle is drawn as three unrelated parts, or a folder of twenty brackets collapses
  to one sheet.
* **A design IR is refused, never skipped.** An IR is not geometry; silently dropping it would
  read as "that file has no drawing", when the truth is "build it first". Same wording as
  ``preview._resolve_source``, and the same ``"kind"`` / ``"components"`` test to tell a CSG AST
  from an IR.
* **Output names never collide silently.** Two sources with the same stem (``a/part.stl`` and
  ``b/part.step``) would otherwise overwrite each other and the batch would report two successes
  with one file on disk - a data-loss bug wearing a green tick. The second gets ``part-2.svg``.
  Names are reserved case-insensitively because macOS filesystems are.
* **stdlib only.** Wired into ``bin/phimech`` via ``CLI_HOOKS``, so it has to import under
  ``/usr/bin/python3`` 3.9 with no fastmcp and no CAD installed; every heavy import is deferred
  into the function that needs it. ``register(mcp)`` takes the server as an argument, exactly
  like ``tools_drawing.py`` and ``tools_preview.py``.
* **Units are mm and degrees**, because every number on the sheet is measured from the mesh by
  ``phimech/synth/drawing.py``; this module adds no geometry of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Names to append to ``server.TOOLS`` when wiring.
TOOLS = ["drawing_batch"]

#: File suffixes a directory scan collects. ``.json`` is admitted here and *then* discriminated
#: by content, because a CSG AST and a design IR share the suffix and only one is geometry.
CAD_SUFFIXES: Tuple[str, ...] = (".step", ".stp", ".stl", ".scad", ".json")

#: Default and hard ceilings on how many sources one call will draw. A sheet is seconds, not
#: milliseconds, so an unbounded batch is a client timeout waiting to happen. Over the cap the
#: call is **refused** rather than truncated: silently drawing the first N of M files and
#: reporting success is the same class of bug as silently overwriting an output.
DEFAULT_MAX_FILES = 25
MAX_MAX_FILES = 500

#: Source kinds. The first three are drawn; the last three are reported without rendering.
KIND_FILE = "file"
KIND_BUILD = "build"
KIND_BUNDLE = "bundle"
KIND_REFUSED = "refused"        # geometry-shaped but not geometry (a design IR) -> failure
KIND_UNSUPPORTED = "unsupported"  # named explicitly but not something we can draw -> skipped
KIND_MISSING = "missing"        # named explicitly but not on disk -> failure

DRAWABLE_KINDS = (KIND_FILE, KIND_BUILD, KIND_BUNDLE)

#: Marker that a directory is a finished ``synth build`` and therefore ONE source. ``report.json``
#: is what the pipeline always writes and what ``drawing.for_build`` reads back. Deliberately not
#: "contains any .stl": a folder of STLs is a *batch*, and collapsing it to a single sheet would
#: throw away every other part in it.
BUILD_MARKER = "report.json"


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
        raise ValueError("%s must be an absolute path (hint: pass /full/path/to/%s)"
                         % (label, label.replace("_", ".")))
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError("%s not found: %s" % (label, path))
    return path


# --------------------------------------------------------------------------------------
# discrimination: what is one source, and what is a folder of them
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One thing to draw - or one thing that cannot be drawn and must still be reported.

    ``reason`` is set only for the non-drawable kinds; it is the text the caller sees in the
    result row's ``error``, so it has to say what to do next, not just what went wrong.
    """

    path: Path
    kind: str
    reason: Optional[str] = None


def is_build_dir(path: Path) -> bool:
    """True for a finished ``synth build`` output directory (one source, not a folder of parts)."""
    return path.is_dir() and (path / BUILD_MARKER).is_file()


def is_bundle_dir(path: Path) -> bool:
    """True for a SolidWorks handoff bundle; delegated so there is one definition in the tree."""
    from phimech.synth import preview

    return preview.is_solidworks_bundle(path)


def _json_source(path: Path, *, explicit: bool) -> Optional[Source]:
    """Tell a CSG AST apart from a design IR exactly as ``preview._resolve_source`` does.

    A design IR always becomes a row (``KIND_REFUSED``) even inside a directory scan: "build it
    first" is actionable, and dropping it would look like the file had no features worth drawing.
    Any other JSON is noise in a build folder (``manifest.json``, a gate report) and is only
    reported when the caller named it.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if explicit:
            return Source(path, KIND_UNSUPPORTED, "not readable as JSON: %s" % exc)
        return None
    if isinstance(document, dict) and "kind" in document:
        return Source(path, KIND_FILE)
    if isinstance(document, dict) and "components" in document:
        return Source(path, KIND_REFUSED,
                      "%s is a design IR, not geometry: build it with `phimech synth build` "
                      "and pass the build directory" % path.name)
    if explicit:
        return Source(path, KIND_UNSUPPORTED,
                      "%s is neither a CSG AST nor a design IR" % path.name)
    return None


def _file_source(path: Path, *, explicit: bool) -> Optional[Source]:
    """Classify one file. ``None`` means "not CAD, and nobody asked about it"."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _json_source(path, explicit=explicit)
    if suffix in CAD_SUFFIXES:
        return Source(path, KIND_FILE)
    if explicit:
        return Source(path, KIND_UNSUPPORTED,
                      "unsupported file %s: pass an .stl, .step/.stp, .scad or CSG AST .json"
                      % path.name)
    return None


def _expand_directory(
    directory: Path, *, recursive: bool, seen_dirs: Set[Path], out: List[Source]
) -> None:
    """Append the sources inside ``directory``, keeping bundles and builds whole.

    A sub-directory that is itself a bundle or a build is one source and is **not** descended
    into, at either depth: that is what makes ``drawing_batch`` over a folder of handoffs do the
    obvious thing. ``seen_dirs`` breaks symlink cycles, which a recursive walk would otherwise
    follow forever.
    """
    try:
        entries = sorted(directory.iterdir())
    except OSError as exc:
        out.append(Source(directory, KIND_UNSUPPORTED, "cannot list directory: %s" % exc))
        return
    for entry in entries:
        if entry.is_dir():
            if is_bundle_dir(entry):
                out.append(Source(entry, KIND_BUNDLE))
            elif is_build_dir(entry):
                out.append(Source(entry, KIND_BUILD))
            elif recursive:
                resolved = entry.resolve()
                if resolved in seen_dirs:
                    continue
                seen_dirs.add(resolved)
                _expand_directory(entry, recursive=recursive, seen_dirs=seen_dirs, out=out)
        elif entry.is_file():
            found = _file_source(entry, explicit=False)
            if found is not None:
                out.append(found)


def collect_sources(inputs: Sequence[Path], *, recursive: bool = False) -> List[Source]:
    """Resolve absolute files and directories into the ordered, de-duplicated list of sources.

    Order is input order, then name order within a directory, so two runs over the same tree
    produce the same sheets with the same names. De-duplication is by resolved path, so naming a
    file *and* the folder containing it draws it once.
    """
    found: List[Source] = []
    seen_dirs: Set[Path] = set()
    for raw in inputs:
        if raw.is_dir():
            if is_bundle_dir(raw):
                found.append(Source(raw, KIND_BUNDLE))
            elif is_build_dir(raw):
                found.append(Source(raw, KIND_BUILD))
            else:
                seen_dirs.add(raw.resolve())
                _expand_directory(raw, recursive=recursive, seen_dirs=seen_dirs, out=found)
        elif raw.is_file():
            explicit = _file_source(raw, explicit=True)
            if explicit is not None:
                found.append(explicit)
        else:
            found.append(Source(raw, KIND_MISSING, "not found: %s" % raw))
    unique: List[Source] = []
    seen: Set[Path] = set()
    for source in found:
        if source.path in seen:
            continue
        seen.add(source.path)
        unique.append(source)
    return unique


# --------------------------------------------------------------------------------------
# output naming: deterministic, and never an overwrite
# --------------------------------------------------------------------------------------

_UNSAFE_STEM_CHARS = set("/\\:\0")


def _safe_stem(raw: str) -> str:
    """A filename-safe stem; empty or separator-only names fall back to ``drawing``."""
    cleaned = "".join(
        "_" if char in _UNSAFE_STEM_CHARS or ord(char) < 32 else char for char in raw
    ).strip().strip(".")
    return cleaned or "drawing"


def output_name(stem: str, taken: Set[str]) -> str:
    """``<stem>.svg``, or ``<stem>-2.svg``, ``<stem>-3.svg``... when that name is already used.

    ``taken`` is keyed case-insensitively: on the case-insensitive filesystems this runs on,
    ``Part.svg`` and ``part.svg`` are the same file, and treating them as different is exactly
    how a batch loses a result. Mutated in place so the caller's ledger stays authoritative.
    """
    base = _safe_stem(stem)
    candidate = "%s.svg" % base
    index = 2
    while candidate.lower() in taken:
        candidate = "%s-%d.svg" % (base, index)
        index += 1
    taken.add(candidate.lower())
    return candidate


def _source_stem(source: Source) -> str:
    """The name a source's sheet is derived from: a file's stem, a directory's own name."""
    if source.kind in (KIND_BUILD, KIND_BUNDLE):
        return source.path.name
    return source.path.stem


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


def _row(source: Source, status: str, **extra: Any) -> Dict[str, Any]:
    """One result row. Every key is always present so a caller can index without guarding."""
    row: Dict[str, Any] = {
        "source": str(source.path),
        "kind": source.kind,
        "status": status,
        "path": None,
        "png": None,
        "groups": None,
        "render_s": None,
        "error": source.reason,
    }
    row.update(extra)
    return row


def run_batch(
    sources: Sequence[Source],
    out_dir: Path,
    *,
    png: bool = True,
    title_prefix: str = "",
    continue_on_error: bool = True,
    dxf: bool = False,
) -> List[Dict[str, Any]]:
    """Draw every source into ``out_dir``, one row per source, never raising for one bad file.

    With ``continue_on_error`` false the run stops at the first failure and every remaining
    source is reported ``skipped`` with the reason, so ``len(results) == len(sources)`` always
    holds and the caller can see exactly where the batch stopped.
    """
    from phimech.synth import drawing

    results: List[Dict[str, Any]] = []
    taken: Set[str] = set()
    stopped = False
    for source in sources:
        if stopped:
            results.append(_row(source, "skipped",
                                error="not attempted: an earlier source failed and "
                                      "continue_on_error is false"))
            continue
        if source.kind in (KIND_REFUSED, KIND_MISSING):
            results.append(_row(source, "failed"))
            stopped = not continue_on_error
            continue
        if source.kind == KIND_UNSUPPORTED:
            results.append(_row(source, "skipped"))
            continue
        out_svg = out_dir / output_name(_source_stem(source), taken)
        title = (title_prefix + _source_stem(source)) if title_prefix else None
        try:
            if source.kind == KIND_BUILD:
                result = drawing.for_build(source.path, out_svg=out_svg, png=bool(png),
                                           title=title, dxf=bool(dxf))
            else:
                result = drawing.render_drawing(source.path, out_svg, title=title,
                                                png=bool(png), dxf=bool(dxf))
        except Exception as exc:  # noqa: BLE001 - one bad file must not lose the good ones
            results.append(_row(source, "failed",
                                error=str(exc) or exc.__class__.__name__))
            stopped = not continue_on_error
            continue
        results.append(_row(source, "ok", path=result.path, png=result.png,
                            groups=len(result.groups), render_s=round(result.render_s, 3),
                            error=None))
    return results


def _summary(out_dir: Path, results: Sequence[Dict[str, Any]], *,
             continue_on_error: bool) -> Dict[str, Any]:
    """Fold the rows into the ledger. ``ok`` is deliberately strict; see the tool docstring."""
    succeeded = sum(1 for row in results if row["status"] == "ok")
    failed = sum(1 for row in results if row["status"] == "failed")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    return {
        "ok": succeeded > 0 and (continue_on_error or failed == 0),
        "out_dir": str(out_dir),
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "results": list(results),
        "units": "mm; every dimension on every sheet is measured from the mesh",
        "next": (
            "open each `path` (or read the image at `png` where it is set) and check the callouts "
            "against the brief; `groups` is how many feature groups that sheet tables - call "
            "`drawing_sheet` on one source for its full hole table; re-run with only the sources "
            "whose `error` you have fixed"
        ),
    }


def drawing_batch_impl(
    out_dir: str,
    paths: Optional[Sequence[str]] = None,
    directory: Optional[str] = None,
    recursive: bool = False,
    png: bool = True,
    title_prefix: str = "",
    continue_on_error: bool = True,
    max_files: int = DEFAULT_MAX_FILES,
    dxf: bool = False,
) -> Dict[str, Any]:
    """The ``drawing_batch`` body: resolve sources, draw each, return the ledger as data."""
    try:
        destination = _absolute(out_dir, "out_dir")
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError("out_dir exists and is not a directory: %s" % destination)
        cap = int(max_files)
        if not 1 <= cap <= MAX_MAX_FILES:
            raise ValueError("max_files must be between 1 and %d" % MAX_MAX_FILES)
        raw_inputs: List[str] = [str(item) for item in (paths or []) if str(item).strip()]
        if directory is not None and str(directory).strip():
            raw_inputs.append(str(directory))
        if not raw_inputs:
            raise ValueError(
                "pass `paths` (absolute files and/or directories) or `directory`; both were empty"
            )
        inputs = [_absolute(item, "paths[%d]" % index) for index, item in enumerate(raw_inputs)]
        sources = collect_sources(inputs, recursive=bool(recursive))
        if len(sources) > cap:
            raise ValueError(
                "%d sources exceed max_files=%d; raise max_files (up to %d) or split the batch - "
                "refusing rather than drawing only the first %d"
                % (len(sources), cap, MAX_MAX_FILES, cap)
            )
        if not sources:
            payload = _summary(destination, [], continue_on_error=bool(continue_on_error))
            payload["error"] = "no CAD sources found in the given paths"
            payload["hint"] = (
                "a directory is scanned for .stl, .step/.stp, .scad and CSG AST .json files "
                "(pass recursive=True for sub-directories); a SolidWorks handoff bundle and a "
                "`synth build` directory each count as one source"
            )
            return payload
        destination.mkdir(parents=True, exist_ok=True)
        results = run_batch(sources, destination, png=bool(png),
                            title_prefix=str(title_prefix or ""),
                            continue_on_error=bool(continue_on_error), dxf=bool(dxf))
        return _summary(destination, results, continue_on_error=bool(continue_on_error))
    except Exception as exc:  # noqa: BLE001 - every failure must reach the agent as data
        return _failure(
            exc,
            "pass absolute paths; each is an .stl, a .step/.stp, a CSG AST .json, a .scad file, "
            "a synth build directory, a SolidWorks handoff bundle, or a directory holding them "
            "(a design IR must be built first); run `phimech doctor` if OpenSCAD or FreeCAD is "
            "missing",
        )


def register(mcp: Any) -> None:
    """Attach ``drawing_batch`` to a FastMCP server instance."""

    @mcp.tool
    def drawing_batch(
        out_dir: str,
        paths: Optional[List[str]] = None,
        directory: Optional[str] = None,
        recursive: bool = False,
        png: bool = True,
        title_prefix: str = "",
        continue_on_error: bool = True,
        max_files: int = DEFAULT_MAX_FILES,
        dxf: bool = False,
    ) -> Dict[str, Any]:
        """Dimensioned inspection drawings for a whole set of CAD files, one sheet per source.

        Set ``dxf`` to also write an AC1015 DXF beside each sheet - the editable artefact
        SolidWorks opens as real geometry with editable dimensions, rather than a picture.

        Hand it the files: ``paths`` is a list of **absolute** files and/or directories, or pass a
        single ``directory``; sheets land in ``out_dir`` (**absolute**, created if needed). Each
        sheet is what ``drawing_sheet`` produces - third-angle top/front/right views on A3 with
        overall dimensions, leader callouts and a hole table, every number measured from the mesh.

        What counts as one source. A ``.stl``, ``.step``/``.stp``, ``.scad`` or CSG AST ``.json``
        file is one. A **SolidWorks handoff bundle** directory is one (validated first, drawn
        under its native document name), *not* the parts inside it. A **synth build** directory
        (one holding ``report.json``) is one - its largest printable part. Any other directory is
        a folder of parts: every CAD file directly inside it gets its own sheet, and with
        ``recursive`` its sub-directories too, except that a bundle or build found in the walk is
        still taken whole. Sheets are named from the source stem; a second source with the same
        stem becomes ``<stem>-2.svg``, so a result is **never** silently overwritten.

        Units: mm and degrees throughout - positions, diameters and depths are model millimetres,
        ``render_s`` is seconds. Limits: **advisory, never a gate** - nothing is asserted from a
        sheet. Per source the ledger reports ``status`` (``ok`` / ``failed`` / ``skipped``),
        ``path``, ``png`` (null when no rasteriser is installed - the SVG is the deliverable),
        ``groups`` (the *number* of feature groups tabled, not the table; call ``drawing_sheet``
        for that) and ``error``. One bad file never loses the others: it becomes a ``failed`` row
        and the run continues unless ``continue_on_error`` is false, after which the remaining
        sources are reported ``skipped`` rather than drawn. ``ok`` is true only when at least one
        source succeeded and, with ``continue_on_error`` false, none failed. A design IR is
        **refused** as a failed row, not skipped: build it first and pass the build directory.
        Sources are drawn one at a time through OpenSCAD (seconds each, longer for a dense mesh),
        so ``max_files`` caps the batch at 25 by default (hard ceiling 500) and the call is
        refused outright rather than truncated when the cap is exceeded. ``title_prefix``, when
        set, overrides each title block with ``<prefix><source stem>`` (include your own
        separator); left empty, every sheet keeps its own default title. Writes only into
        ``out_dir``, reads everything else.
        """
        # The two caps are spelled out above rather than interpolated: a docstring built with %
        # is a BinOp, and `ast.get_docstring` - which both the agent guide and the tool-docs gate
        # use - would read it as no docstring at all. `tests/test_drawing_batch.py` asserts the
        # numbers in the prose still match DEFAULT_MAX_FILES and MAX_MAX_FILES.
        return drawing_batch_impl(out_dir, paths, directory, recursive, png, title_prefix,
                                  continue_on_error, max_files, dxf)


# --------------------------------------------------------------------------------------
# CLI hook: `phimech drawings PATH [PATH...] -o DIR`
# --------------------------------------------------------------------------------------


def _handle_drawings(args: argparse.Namespace) -> int:
    result = drawing_batch_impl(
        str(Path(args.output).expanduser().resolve()),
        paths=[str(Path(p).expanduser().resolve()) for p in args.sources],
        recursive=bool(args.recursive),
        png=not args.no_png,
        title_prefix=args.title_prefix or "",
        continue_on_error=not args.stop_on_error,
        max_files=args.max_files,
        dxf=bool(args.dxf),
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    if "results" not in result:
        print("drawing batch failed: {}".format(result.get("error")), file=sys.stderr)
        print("hint: {}".format(result.get("hint")), file=sys.stderr)
        return 1
    for row in result["results"]:
        if row["status"] == "ok":
            print("ok    {}  ->  {}  ({} feature groups, {:.1f}s)".format(
                row["source"], row["path"], row["groups"], row["render_s"] or 0.0))
        else:
            print("{:<5} {}  {}".format(row["status"].upper(), row["source"], row["error"] or ""))
    print("{} of {} drawn ({} failed, {} skipped) in {}".format(
        result["succeeded"], result["total"], result["failed"], result["skipped"],
        result["out_dir"]))
    if result.get("error"):
        print("error: {}".format(result["error"]), file=sys.stderr)
    return 0 if result["ok"] else 1


def register_cli(subparsers: Any) -> None:
    """Add ``phimech drawings`` to the CLI (wired from ``phimech/cli.py``'s ``CLI_HOOKS``)."""
    parser = subparsers.add_parser(
        "drawings",
        help="batch: one dimensioned drawing per CAD file / bundle / build, into one directory",
        description="Point it at files and/or folders and get a dimensioned A3 sheet for each. "
                    "A SolidWorks handoff bundle and a synth build directory each count as ONE "
                    "source; any other directory is a folder of parts. Every number is measured "
                    "from the mesh, in mm. Exit 1 unless at least one drawing succeeded.",
    )
    parser.add_argument("sources", nargs="+",
                        help="files and/or directories: .stl, .step/.stp, .scad, CSG AST .json, "
                             "a synth build dir, or a SolidWorks handoff bundle dir")
    parser.add_argument("-o", "--output", required=True, metavar="DIR",
                        help="directory the sheets are written into (created if needed)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="walk sub-directories (bundles and builds are still taken whole)")
    parser.add_argument("--title-prefix", default="", dest="title_prefix",
                        help="prepend this to every title block (include your own separator)")
    parser.add_argument("--no-png", action="store_true", dest="no_png",
                        help="skip the best-effort PNG copy beside each SVG")
    parser.add_argument("--stop-on-error", action="store_true", dest="stop_on_error",
                        help="stop at the first failure instead of drawing the rest")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, dest="max_files",
                        help="refuse a batch larger than this (default %d, ceiling %d)"
                             % (DEFAULT_MAX_FILES, MAX_MAX_FILES))
    parser.add_argument("--dxf", action="store_true",
                        help="also write each sheet as DXF (AC1015) - the editable artefact "
                             "SolidWorks opens as real geometry, not a picture of a drawing")
    parser.add_argument("--json", action="store_true", help="print the full ledger as JSON")
    parser.set_defaults(handler=_handle_drawings)


__all__ = [
    "CAD_SUFFIXES",
    "DEFAULT_MAX_FILES",
    "MAX_MAX_FILES",
    "Source",
    "TOOLS",
    "collect_sources",
    "drawing_batch_impl",
    "is_build_dir",
    "is_bundle_dir",
    "output_name",
    "register",
    "register_cli",
    "run_batch",
]
