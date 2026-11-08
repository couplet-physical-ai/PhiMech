"""Where a drawing's general tolerance and surface finish come from - and why never from here.

Why this module exists
----------------------
A drawing that prints a tolerance is a manufacturing instruction: a machinist holds whatever
number is on the sheet.  Choosing +-0.1 rather than +-0.3 is an engineering decision about fit,
function, cost and process, and **none of that information is in the geometry**.  PhiMech can
measure a 24.00 mm bore; it cannot know whether that bore is a press fit for a bearing or a
clearance hole for a cable.  So this module does exactly three things, and deliberately not a
fourth:

* **carry** - a tolerance or finish a *human* declared, in the design IR
  (``manufacturing.general_tolerance`` / ``manufacturing.finish``) or on an imported part's
  handoff manifest (``component.general_tolerance`` / ``component.finish``);
* **resolve** - one function, one documented precedence, provenance attached to every value, so
  the drawing can trace each printed number back to the document that declared it;
* **expand** - turn a *published* class such as ``ISO 2768-m`` into that standard's own defined
  per-range table.  Expanding a class is a service; **choosing** the class is not ours to do.
  The tables below are transcribed from the standard, not derived, interpolated or rounded.

It never originates a value.  When nothing is declared, ``resolve_tolerance()`` returns
``specified=False`` with empty tables and ``finish=None``, and the drawing prints UNSPECIFIED.
There is no default tolerance anywhere in this file, and the tests assert that.

Sources for the ISO 2768-1 tables
---------------------------------
Transcribed from **ISO 2768-1:1989** (first edition 1989-11-15), *General tolerances - Part 1:
Tolerances for linear and angular dimensions without individual tolerance indications*, its
table 1 (linear), table 2 (broken edges) and table 3 (angular), read from the published
standard text itself and then cross-checked cell by cell against a spread of independent
reproductions (Engineers Edge, AmesWeb, Fictiv, Hubs, SR Tech, Runsom, RivCut, Noramark).
Every cell is confirmed by the standard plus at least two independent secondary sources; no
value here is single-sourced, interpolated or rounded.

Cells the standard leaves **blank are encoded as absent rows**, never as a number: class ``f``
defines no linear deviation over 2000 mm and class ``v`` defines none below 3 mm, so those rows
simply do not exist and a lookup there correctly returns None.  Where the standard prints one
merged row for two classes (f with m in tables 2 and 3, c with v in table 2) the two columns
hold equal values, and ``_self_check()`` asserts that they stay equal.

Four traps that were checked for deliberately, recorded so a future edit does not "correct"
these values back to something wrong:

* **DIN 7168 is not ISO 2768-1.**  Several widely mirrored PDFs titled "DIN ISO 2768-1" in fact
  print DIN 7168: coarse linear +-0,15 / +-0,2 in the first two bands (ISO says +-0.2 / +-0.3),
  class letters g and sg instead of c and v, an extra 4000-8000 band, and a five-band broken-edge
  table.  The two standards agree for f and m and diverge for the coarse classes, which is
  exactly why they get confused.  The ISO values are what is encoded here.
* **Angular class c over 120 up to 400 is +-0 deg 15', not +-0 deg 20'.**  One vendor PDF, mirrored
  on two sites and therefore easy to mistake for two sources, prints 20'.  The standard prints 15'.
* ISO 2768 **part 2** covers geometrical tolerances (classes H, K, L) and is a different
  document.  PhiMech emits no GD&T, so ``ISO 2768-mK`` has its part-1 class expanded and its
  part-2 class carried onto the sheet verbatim, flagged as not expanded.  Per part 2 clause 6,
  ``ISO 2768-K`` (part-1 class omitted) and ``ISO 2768-mK-E`` (with the envelope requirement)
  are both legal designations, and both are recognised here.
* **ISO 2768-1:1989 is current, not withdrawn** - ISO stage 90.92, "standard to be revised", as
  of 2022-10-04.  ISO 22081:2021 cancelled and replaced ISO 2768-*2*, not part 1, and a second
  edition of part 1 is in progress.  This module expands the class a human declared; it does not
  advise which standard to declare.

ISO's own footnote 1 to tables 1 and 2 governs the bottom of the range: *"For nominal sizes
below 0,5 mm, the deviations shall be indicated adjacent to the relevant nominal size(s)."*
That is why no table starts at zero and why None from a lookup means "this dimension needs its
own tolerance", never "zero" and never "use the nearest band".

Units are mm and degrees throughout, like the rest of PhiMech.  stdlib only, python 3.9.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: One row of a tolerance table: ``(range_min, range_max, plus_minus)``.  The bounds are always
#: millimetres - for an angular row they bound the *shorter side of the angle*, which is how
#: ISO 2768-1 keys its angular table - and ``plus_minus`` is mm for a linear or broken-edge row
#: and degrees for an angular one.  A band covers ``range_min < x <= range_max``; only the bound
#: where the *table* starts is inclusive (the standard's first linear band reads "0,5 up to 3",
#: so 0.5 mm is inside it).  See ``_lookup`` for why that is not simply "the first band".
Band = Tuple[float, float, float]

#: ``range_max`` of a band with no upper bound.  ``to_dict()`` serialises it as ``null`` because
#: ``Infinity`` is not valid JSON, and a reader of the sidecar must not mistake it for a number.
UNBOUNDED = float("inf")

#: What a drawing prints where nothing was declared.  Exported so the drawing lane and this
#: module cannot drift into saying it two different ways.
UNSPECIFIED = "UNSPECIFIED"


# --------------------------------------------------------------------------------------
# the published ISO 2768-1 tables
# --------------------------------------------------------------------------------------

ISO_2768_CLASSES: Dict[str, str] = {
    "f": "fine", "m": "medium", "c": "coarse", "v": "very coarse",
}

#: Where the ISO 2768-1 size tables start, from footnote 1 to tables 1 and 2: below 0.5 mm the
#: deviations "shall be indicated adjacent to the relevant nominal size(s)".  Held separately
#: from the bands because a class whose first cell is blank starts *above* this - class v's
#: first row is "over 3", not "3 and up" - and only this constant can tell the two apart.
ISO_2768_APPLIES_FROM_MM = 0.5

#: ISO 2768-1 table 1 - permissible deviations for linear dimensions except broken edges, mm.
#: Nominal size bands are (over, up to and including); the first band starts at 0.5 inclusive.
#: **Blanks are omissions, not zeroes**: the standard defines no value for class ``f`` over
#: 2000 mm, and none for class ``v`` below 3 mm, so neither row exists here.  A dimension that
#: falls in no band has no general tolerance and must be toleranced individually - ISO 2768-1
#: says as much for nominal sizes below 0.5 mm, which is why no table starts at zero.
ISO_2768_LINEAR: Dict[str, Tuple[Band, ...]] = {
    "f": (
        (0.5, 3.0, 0.05), (3.0, 6.0, 0.05), (6.0, 30.0, 0.1), (30.0, 120.0, 0.15),
        (120.0, 400.0, 0.2), (400.0, 1000.0, 0.3), (1000.0, 2000.0, 0.5),
    ),
    "m": (
        (0.5, 3.0, 0.1), (3.0, 6.0, 0.1), (6.0, 30.0, 0.2), (30.0, 120.0, 0.3),
        (120.0, 400.0, 0.5), (400.0, 1000.0, 0.8), (1000.0, 2000.0, 1.2),
        (2000.0, 4000.0, 2.0),
    ),
    "c": (
        (0.5, 3.0, 0.2), (3.0, 6.0, 0.3), (6.0, 30.0, 0.5), (30.0, 120.0, 0.8),
        (120.0, 400.0, 1.2), (400.0, 1000.0, 2.0), (1000.0, 2000.0, 3.0),
        (2000.0, 4000.0, 4.0),
    ),
    "v": (
        (3.0, 6.0, 0.5), (6.0, 30.0, 1.0), (30.0, 120.0, 1.5), (120.0, 400.0, 2.5),
        (400.0, 1000.0, 4.0), (1000.0, 2000.0, 6.0), (2000.0, 4000.0, 8.0),
    ),
}

#: ISO 2768-1 table 2 - permissible deviations for broken edges (external radii and chamfer
#: heights), mm.  A **separate** table from the linear one and materially looser at small sizes:
#: class m tolerates a 1 mm chamfer at +-0.2, not the +-0.1 the linear table would give it.
#: Reading a chamfer out of the linear table is a real way to print a tolerance nobody specified,
#: so the resolver keeps this table distinct and ``broken_edge_for()`` is the only lookup for it.
#: Classes f and m share a column, as do c and v.
ISO_2768_BROKEN_EDGES: Dict[str, Tuple[Band, ...]] = {
    "f": ((0.5, 3.0, 0.2), (3.0, 6.0, 0.5), (6.0, UNBOUNDED, 1.0)),
    "m": ((0.5, 3.0, 0.2), (3.0, 6.0, 0.5), (6.0, UNBOUNDED, 1.0)),
    "c": ((0.5, 3.0, 0.4), (3.0, 6.0, 1.0), (6.0, UNBOUNDED, 2.0)),
    "v": ((0.5, 3.0, 0.4), (3.0, 6.0, 1.0), (6.0, UNBOUNDED, 2.0)),
}

#: ISO 2768-1 table 3 - permissible deviations of angular dimensions, keyed by the length in mm
#: of the **shorter side** of the angle.  Held in whole arc-minutes because that is how the
#: standard publishes them (0 deg 20', not 0.333 deg): storing degrees first and rendering back
#: would print a repeating decimal where the standard prints an exact value.  Classes f and m
#: share a column here too.
ISO_2768_ANGULAR_RANGES: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0), (10.0, 50.0), (50.0, 120.0), (120.0, 400.0), (400.0, UNBOUNDED),
)
ISO_2768_ANGULAR_MINUTES: Dict[str, Tuple[int, ...]] = {
    "f": (60, 30, 20, 10, 5),
    "m": (60, 30, 20, 10, 5),
    "c": (90, 60, 30, 15, 10),
    "v": (180, 120, 60, 30, 20),
}


def _angular_bands(cls: str) -> Tuple[Band, ...]:
    """Arc-minutes to degrees, at the last possible moment so the published value stays exact."""
    return tuple(
        (low, high, minutes / 60.0)
        for (low, high), minutes in zip(ISO_2768_ANGULAR_RANGES, ISO_2768_ANGULAR_MINUTES[cls])
    )


ISO_2768_ANGULAR: Dict[str, Tuple[Band, ...]] = {
    cls: _angular_bands(cls) for cls in ISO_2768_ANGULAR_MINUTES
}


def format_angle(degrees: float) -> str:
    """Render an angular deviation the way a drawing does: ``1 deg``, ``0 deg 20'``.

    Whole arc-minutes come back exactly; anything else falls back to decimal degrees rather than
    rounding a value the author gave us to a minute we then present as theirs.
    """
    minutes = degrees * 60.0
    nearest = round(minutes)
    if abs(minutes - nearest) > 1e-9:
        return "%g°" % degrees
    whole, remainder = divmod(int(nearest), 60)
    return "%d°" % whole if remainder == 0 else "%d°%d'" % (whole, remainder)


@dataclass(frozen=True)
class _Designation:
    """A parsed ISO 2768 title-block designation.

    ``size_class`` is the part-1 class (f/m/c/v) and the only thing with a table here.  It is
    legitimately None: ISO 2768-2 clause 6.2 allows ``ISO 2768-K``, which declares a geometrical
    class and no general size tolerance at all.
    """

    size_class: Optional[str]
    geometrical: Optional[str]
    envelope: bool


# ``ISO 2768-m``, ``iso 2768 m``, ``ISO 2768-1-m``, ``ISO 2768-mK``, ``ISO 2768-K``,
# ``ISO 2768-mK-E``.  Lower-cased and de-spaced first, so the H/K/L case the standard prints is
# not relied on - the 1989 printing of part 2 is itself inconsistent about it.
_ISO_2768_RE = re.compile(r"^(?:iso)?2768(?:-?[12])?-?([fmcv])?([hkl])?(?:-?(e))?$")


def _designation_of(standard: str) -> Optional[_Designation]:
    """Parse a designation, or None when this is not an ISO 2768 class at all.

    Unrecognised is a first-class outcome, not an error: the text is still carried onto the
    drawing verbatim, just without a table beside it.  Guessing that an unknown class "is
    probably medium" is precisely the invention this module exists to prevent.
    """
    match = _ISO_2768_RE.match(re.sub(r"[\s–—]+", "", standard).lower())
    if match is None:
        return None
    size_class, geometrical, envelope = match.groups()
    if size_class is None and geometrical is None:
        return None  # a bare "ISO 2768" declares no class and expands to nothing
    return _Designation(size_class, geometrical.upper() if geometrical else None,
                        envelope is not None)


# --------------------------------------------------------------------------------------
# the resolved block
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Considered:
    """A declaration the resolver saw, whether or not it won.

    Recorded so a losing source is *visible* rather than silently dropped: a reviewer can see
    that the manifest said something and why the IR was printed instead.
    """

    source: str
    used: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "used": self.used, "reason": self.reason}


@dataclass(frozen=True)
class ToleranceBlock:
    """Everything a drawing needs in order to print - or refuse to print - a tolerance block.

    ``specified`` is the only question the drawing has to ask first.  When it is False every
    table is empty, ``standard`` and ``finish`` are None, ``provenance`` is ``"none"``, and the
    sheet prints UNSPECIFIED.  There is no partially-filled state where a default has leaked in.
    """

    specified: bool
    provenance: str
    standard: Optional[str] = None
    standard_class: Optional[str] = None
    linear: Tuple[Band, ...] = ()
    angular: Tuple[Band, ...] = ()
    broken_edges: Tuple[Band, ...] = ()
    finish: Optional[str] = None
    derivation: Optional[str] = None
    applies_from_mm: Optional[float] = None
    notes: Tuple[str, ...] = ()
    conflicts: Tuple[str, ...] = ()
    considered: Tuple[Considered, ...] = ()

    # -- lookups ---------------------------------------------------------------------------

    def linear_for(self, nominal_mm: float) -> Optional[float]:
        """The general +- in mm for a linear nominal size, or None if no band covers it.

        None is a meaningful answer and must not be smoothed over: ISO 2768-1 leaves sizes below
        0.5 mm (and, per class, the ends of the range) to individual tolerancing, so a drawing
        that gets None must say the dimension needs its own tolerance.
        """
        return _lookup(self.linear, nominal_mm, self._floor(self.linear))

    def angular_for(self, shorter_side_mm: float) -> Optional[float]:
        """The general +- in degrees for an angle, keyed by its shorter side in mm.

        ISO 2768-1 table 3 has no blank cells, so its first row is always the table's start and
        the angular floor is simply that row's lower bound.
        """
        return _lookup(self.angular, shorter_side_mm,
                       self.angular[0][0] if self.angular else None)

    def broken_edge_for(self, size_mm: float) -> Optional[float]:
        """The general +- in mm for a chamfer height or external radius - table 2, not table 1."""
        return _lookup(self.broken_edges, size_mm, self._floor(self.broken_edges))

    def _floor(self, bands: Tuple[Band, ...]) -> Optional[float]:
        """Where a size table starts: the standard's documented floor, else the first band."""
        if self.applies_from_mm is not None:
            return self.applies_from_mm
        return bands[0][0] if bands else None

    # -- serialisation ---------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The drawing sidecar: every printed number carries the source that declared it.

        Shapes stay stable whether or not anything was declared, so a consumer never has to
        branch on key presence - only on ``specified``.
        """
        unspecified: List[str] = []
        if not (self.standard or self.linear or self.angular or self.broken_edges):
            unspecified.append("general_tolerance")
        if self.finish is None:
            unspecified.append("finish")
        return {
            "specified": self.specified,
            "provenance": self.provenance,
            "standard": self._traced(self.standard),
            "standard_class": self.standard_class,
            "applies_from_mm": self.applies_from_mm,
            "finish": self._traced(self.finish),
            "linear": [self._row(band, "plus_minus_mm") for band in self.linear],
            "angular": [self._row(band, "plus_minus_deg") for band in self.angular],
            "broken_edges": [self._row(band, "plus_minus_mm") for band in self.broken_edges],
            "notes": list(self.notes),
            "conflicts": list(self.conflicts),
            "considered": [entry.to_dict() for entry in self.considered],
            "unspecified": unspecified,
            "unspecified_text": UNSPECIFIED,
        }

    def _traced(self, value: Optional[str]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        return {"value": value, "source": self.provenance, "derivation": "declared"}

    def _row(self, band: Band, unit_key: str) -> Dict[str, Any]:
        low, high, deviation = band
        row: Dict[str, Any] = {
            "range_min_mm": low,
            "range_max_mm": None if high == UNBOUNDED else high,
            unit_key: deviation,
            "source": self.provenance,
            "derivation": self.derivation,
        }
        if unit_key == "plus_minus_deg":
            row["plus_minus_display"] = format_angle(deviation)
        return row


#: The one block every caller gets when nobody declared anything.  A module-level constant so
#: "nothing declared" is a single object that cannot accidentally acquire a value.
UNSPECIFIED_BLOCK = ToleranceBlock(specified=False, provenance="none")


def _lookup(bands: Sequence[Band], value: float, inclusive_at: Optional[float]) -> Optional[float]:
    """Find the band covering ``value``: ``low < value <= high``, ``low <= value`` at the floor.

    ``inclusive_at`` is the one lower bound that is inclusive - the nominal size the table
    *starts* at.  It cannot be "whichever band happens to be first", because a class with a
    blank leading cell starts at the upper bound of that blank: ISO 2768-1 class v has no value
    for 0.5 up to 3, so its first row is "**over** 3 up to 6" and a 3.00 mm dimension has no
    general tolerance at all.  Treating that row's 3.0 as inclusive would silently hand a
    machinist +-0.5 on a dimension the standard says must be toleranced individually.
    """
    for low, high, deviation in bands:
        matched = value >= low if (inclusive_at is not None and low == inclusive_at) \
            else value > low
        if matched and value <= high:
            return deviation
    return None


# --------------------------------------------------------------------------------------
# reading a declaration out of a document
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Declaration:
    """What one document said, normalised but not yet expanded."""

    source: str
    standard: Optional[str]
    plus_minus: Optional[float]
    linear: Tuple[Band, ...]
    angular: Tuple[Band, ...]
    broken_edges: Tuple[Band, ...]
    finish: Optional[str]

    def empty(self) -> bool:
        return not (self.standard or self.plus_minus or self.linear or self.angular
                    or self.broken_edges or self.finish)

    def key(self) -> Tuple[Any, ...]:
        """What two declarations are compared on to decide whether they disagree."""
        return (self.standard, self.plus_minus, self.linear, self.angular, self.broken_edges,
                self.finish)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number, got %r" % (where, value))
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("%s must be finite, got %r" % (where, value))
    return number


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string, got %r" % (where, value))
    return value.strip()


def _bands(value: Any, where: str) -> Tuple[Band, ...]:
    """Read an author-supplied table, refusing anything that would make a lookup ambiguous.

    Overlapping or unordered bands are rejected rather than sorted into shape: if two rows claim
    the same nominal size, only the author knows which one they meant to hold.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("%s must be a non-empty list of tolerance bands" % where)
    bands: List[Band] = []
    for index, row in enumerate(value):
        at = "%s[%d]" % (where, index)
        if not isinstance(row, Mapping):
            raise ValueError("%s must be an object with range_min, range_max, plus_minus" % at)
        unknown = sorted(set(row) - {"range_min", "range_max", "plus_minus"})
        if unknown:
            raise ValueError("%s has unknown key(s): %s" % (at, ", ".join(unknown)))
        missing = sorted({"range_min", "range_max", "plus_minus"} - set(row))
        if missing:
            raise ValueError("%s is missing %s" % (at, ", ".join(missing)))
        low = _number(row["range_min"], at + ".range_min")
        high = _number(row["range_max"], at + ".range_max")
        deviation = _number(row["plus_minus"], at + ".plus_minus")
        if low < 0:
            raise ValueError("%s.range_min must not be negative" % at)
        if high <= low:
            raise ValueError("%s.range_max must be greater than range_min" % at)
        if deviation <= 0:
            raise ValueError("%s.plus_minus must be greater than zero" % at)
        if bands and low < bands[-1][1]:
            raise ValueError(
                "%s overlaps the band before it (%g..%g); bands must be ordered and disjoint"
                % (at, bands[-1][0], bands[-1][1]))
        bands.append((low, high, deviation))
    return tuple(bands)


def _read_general_tolerance(value: Any, source: str, where: str) -> Dict[str, Any]:
    """Normalise the two accepted spellings of a declaration into one dict of parts."""
    if isinstance(value, str):
        return {"standard": _text(value, where)}
    if not isinstance(value, Mapping):
        raise ValueError(
            "%s must be an object or a published class name, got %r" % (where, value))
    unknown = sorted(set(value) - {"standard", "plus_minus", "linear", "angular",
                                   "broken_edges"})
    if unknown:
        raise ValueError("%s has unknown key(s): %s" % (where, ", ".join(unknown)))
    parts: Dict[str, Any] = {}
    if value.get("standard") is not None:
        parts["standard"] = _text(value["standard"], where + ".standard")
    if value.get("plus_minus") is not None:
        parts["plus_minus"] = _number(value["plus_minus"], where + ".plus_minus")
        if parts["plus_minus"] <= 0:
            raise ValueError("%s.plus_minus must be greater than zero" % where)
    for key in ("linear", "angular", "broken_edges"):
        if value.get(key) is not None:
            parts[key] = _bands(value[key], where + "." + key)

    # A class *and* explicit numbers in one declaration is refused rather than merged: the sheet
    # would then read "ISO 2768-m" beside a table that is not ISO 2768-m, and a machinist cannot
    # tell which one governs.  Declaring the whole table explicitly says it unambiguously.
    explicit = [key for key in ("plus_minus", "linear", "angular", "broken_edges")
                if key in parts]
    if "standard" in parts and explicit:
        raise ValueError(
            "%s declares the class %r and also %s; a drawing cannot print both. Declare the "
            "class alone, or write the whole table out explicitly."
            % (where, parts["standard"], ", ".join(explicit)))
    if "plus_minus" in parts and "linear" in parts:
        raise ValueError(
            "%s declares both plus_minus and a linear table; they are mutually exclusive" % where)
    if not parts:
        raise ValueError(
            "%s is empty; omit it entirely to mean UNSPECIFIED rather than declaring nothing"
            % where)
    return parts


def _declaration(document: Any, source: str) -> Optional[_Declaration]:
    """Pull a declaration out of a design IR or a handoff source manifest.

    The IR is read at ``manufacturing.general_tolerance`` / ``manufacturing.finish``; a bare
    manufacturing block is accepted too, because the drawing lane often holds only that.  The
    manifest is read at ``component.general_tolerance`` / ``component.finish``, the same place
    ``material`` and ``mass_g`` live.
    """
    if document is None:
        return None
    if not isinstance(document, Mapping):
        raise ValueError("%s document must be a mapping, got %r" % (source, type(document)))
    if source == "manifest":
        holder = document.get("component")
        prefix = "manifest.component"
        if holder is None and ("general_tolerance" in document or "finish" in document):
            holder, prefix = document, "manifest"  # already the component block
    else:
        holder = document.get("manufacturing")
        prefix = "ir.manufacturing"
        if holder is None and ("general_tolerance" in document or "finish" in document):
            holder, prefix = document, "ir"  # already the manufacturing block
    if not isinstance(holder, Mapping):
        return None
    raw_tolerance = holder.get("general_tolerance")
    raw_finish = holder.get("finish")
    if raw_tolerance is None and raw_finish is None:
        return None
    parts: Dict[str, Any] = {}
    if raw_tolerance is not None:
        parts = _read_general_tolerance(raw_tolerance, source, prefix + ".general_tolerance")
    return _Declaration(
        source=source,
        standard=parts.get("standard"),
        plus_minus=parts.get("plus_minus"),
        linear=parts.get("linear", ()),
        angular=parts.get("angular", ()),
        broken_edges=parts.get("broken_edges", ()),
        finish=_text(raw_finish, prefix + ".finish") if raw_finish is not None else None,
    )


# --------------------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------------------

#: Precedence, highest first.  The design IR beats the handoff manifest because the IR is the
#: document the local engineer edits for *this* build and can change; the manifest is an
#: attestation about somebody else's upstream export that PhiMech must not rewrite.  Precedence
#: exists to make the function deterministic, NOT to say the losing declaration is wrong - which
#: is why the loser is always recorded in ``considered`` and any disagreement in ``conflicts``.
PRECEDENCE: Tuple[str, ...] = ("ir", "manifest")


def resolve_tolerance(
    ir: Optional[Mapping[str, Any]] = None,
    source_manifest: Optional[Mapping[str, Any]] = None,
) -> ToleranceBlock:
    """Resolve the declared general tolerance and finish for a drawing.  Never invent one.

    ``ir`` is a design IR (or just its ``manufacturing`` block); ``source_manifest`` is a
    handoff ``source.json`` (or just its ``component`` block).  Either may be None.

    Precedence is ``PRECEDENCE``: IR, then manifest, applied to the declaration **as a whole**
    rather than field by field.  Taking the tolerance from one document and the finish from
    another would produce a title block describing a specification that no single document ever
    stated, so the winning document supplies both and the other is reported.

    Raises ``ValueError`` when a document *did* declare something the resolver cannot use - a
    malformed band, a class plus an overriding table, an empty object.  Silence there would turn
    a human's stated intent into a blank sheet.  Absence is never an error: no declaration
    anywhere returns ``UNSPECIFIED_BLOCK``.
    """
    found: Dict[str, _Declaration] = {}
    for source, document in (("ir", ir), ("manifest", source_manifest)):
        declaration = _declaration(document, source)
        if declaration is not None and not declaration.empty():
            found[source] = declaration

    if not found:
        return UNSPECIFIED_BLOCK

    winner = next(found[source] for source in PRECEDENCE if source in found)
    losers = [found[source] for source in PRECEDENCE
              if source in found and source != winner.source]

    conflicts: List[str] = []
    considered: List[Considered] = [
        Considered(winner.source, True, "highest-precedence declaration")]
    for loser in losers:
        agrees = loser.key() == winner.key()
        considered.append(Considered(
            loser.source, False,
            "agrees with the %s declaration" % winner.source if agrees
            else "overridden by the %s declaration, which takes precedence" % winner.source))
        if not agrees:
            conflicts.append(
                "%s and %s declare different general tolerances or finishes (%s vs %s); the %s "
                "declaration is printed and the %s one is not"
                % (winner.source, loser.source, _summary(winner), _summary(loser),
                   winner.source, loser.source))

    block = _expand(winner)
    return replace(block, conflicts=tuple(conflicts), considered=tuple(considered))


def _summary(declaration: _Declaration) -> str:
    parts: List[str] = []
    if declaration.standard:
        parts.append(declaration.standard)
    if declaration.plus_minus is not None:
        parts.append("+-%g mm" % declaration.plus_minus)
    for name, table in (("linear", declaration.linear), ("angular", declaration.angular),
                        ("broken_edges", declaration.broken_edges)):
        if table:
            parts.append("%d explicit %s band(s)" % (len(table), name))
    if declaration.finish:
        parts.append("finish %r" % declaration.finish)
    return "; ".join(parts) if parts else "nothing"


def _expand(declaration: _Declaration) -> ToleranceBlock:
    """Turn one winning declaration into printable tables, expanding a published class only."""
    notes: List[str] = []
    standard = declaration.standard
    designation = _designation_of(standard) if standard else None
    cls = designation.size_class if designation else None

    linear, angular, broken_edges = declaration.linear, declaration.angular, \
        declaration.broken_edges
    derivation = "declared"
    applies_from: Optional[float] = None

    if standard is not None:
        if cls is not None:
            linear = ISO_2768_LINEAR[cls]
            angular = ISO_2768_ANGULAR[cls]
            broken_edges = ISO_2768_BROKEN_EDGES[cls]
            derivation = standard
            applies_from = ISO_2768_APPLIES_FROM_MM
            notes.append(
                "linear, angular and broken-edge tables expanded from the published ISO 2768-1 "
                "class %s (%s); no value here was chosen by PhiMech"
                % (cls, ISO_2768_CLASSES[cls]))
            notes.append(
                "ISO 2768-1 leaves nominal sizes below 0.5 mm to individual tolerancing, so the "
                "linear table starts at 0.5 mm")
            if cls == "f":
                notes.append(
                    "ISO 2768-1 class f defines no linear deviation over 2000 mm; a dimension "
                    "beyond that band has no general tolerance and must be toleranced "
                    "individually")
            if cls == "v":
                notes.append(
                    "ISO 2768-1 class v defines no linear deviation below 3 mm; a dimension "
                    "under that band has no general tolerance and must be toleranced "
                    "individually")
        elif designation is not None:
            notes.append(
                "%r declares no ISO 2768-1 size class, so there is no general linear or angular "
                "tolerance to print; sizes must be toleranced individually" % standard)
        else:
            notes.append(
                "%r is not a class this resolver can expand; it is printed as declared and no "
                "table is shown beside it" % standard)
        if designation is not None and designation.geometrical is not None:
            notes.append(
                "the ISO 2768-2 geometrical class %s is carried onto the drawing as declared "
                "but is NOT expanded: PhiMech emits no geometrical tolerances"
                % designation.geometrical)
        if designation is not None and designation.envelope:
            notes.append(
                "the envelope requirement (E) is carried onto the drawing as declared but is "
                "NOT evaluated: PhiMech checks no envelope condition")
    elif declaration.plus_minus is not None:
        linear = ((0.0, UNBOUNDED, declaration.plus_minus),)
        notes.append(
            "a single declared +-%g mm applies to every linear dimension; no angular or "
            "broken-edge general tolerance was declared" % declaration.plus_minus)

    return ToleranceBlock(
        specified=bool(standard or linear or angular or broken_edges or declaration.finish),
        provenance=declaration.source,
        standard=standard,
        standard_class=cls,
        linear=linear,
        angular=angular,
        broken_edges=broken_edges,
        finish=declaration.finish,
        derivation=derivation,
        applies_from_mm=applies_from,
        notes=tuple(notes),
    )


def _self_check() -> None:
    """Import-time guard on the transcribed tables: shape, ordering and the deliberate blanks.

    Cheap, and it fails loudly at import rather than silently printing a malformed table on a
    drawing if somebody edits a row above.
    """
    for name, table in (("linear", ISO_2768_LINEAR), ("angular", ISO_2768_ANGULAR),
                        ("broken_edges", ISO_2768_BROKEN_EDGES)):
        assert set(table) == set(ISO_2768_CLASSES), name
        for cls, bands in table.items():
            assert bands, (name, cls)
            previous = -1.0
            for low, high, deviation in bands:
                assert 0.0 <= low < high, (name, cls, low, high)
                assert deviation > 0.0, (name, cls, deviation)
                assert low >= previous, (name, cls, low, previous)
                previous = high
    assert ISO_2768_LINEAR["f"][-1][1] == 2000.0, "class f must stop at 2000 mm"
    assert ISO_2768_LINEAR["v"][0][0] == 3.0, "class v must start over 3 mm"
    for cls in ("m", "c", "v"):
        assert ISO_2768_LINEAR[cls][-1][1] == 4000.0, cls
    assert ISO_2768_ANGULAR["f"] == ISO_2768_ANGULAR["m"], "classes f and m share the angles"
    assert ISO_2768_BROKEN_EDGES["f"] == ISO_2768_BROKEN_EDGES["m"], "f and m share the edges"
    assert ISO_2768_BROKEN_EDGES["c"] == ISO_2768_BROKEN_EDGES["v"], "c and v share the edges"
    assert UNSPECIFIED_BLOCK.to_dict()["unspecified"] == ["general_tolerance", "finish"]
    # The blank leading cell of class v must stay outside the table: 3.00 mm is covered for
    # every class that has a 0.5-3 row and for none that does not.
    for cls in ISO_2768_CLASSES:
        covered = _lookup(ISO_2768_LINEAR[cls], 3.0, ISO_2768_APPLIES_FROM_MM) is not None
        assert covered == (ISO_2768_LINEAR[cls][0][0] == ISO_2768_APPLIES_FROM_MM), cls


_self_check()


__all__ = [
    "Band",
    "Considered",
    "ISO_2768_ANGULAR",
    "ISO_2768_APPLIES_FROM_MM",
    "ISO_2768_ANGULAR_MINUTES",
    "ISO_2768_ANGULAR_RANGES",
    "ISO_2768_BROKEN_EDGES",
    "ISO_2768_CLASSES",
    "ISO_2768_LINEAR",
    "PRECEDENCE",
    "ToleranceBlock",
    "UNBOUNDED",
    "UNSPECIFIED",
    "UNSPECIFIED_BLOCK",
    "format_angle",
    "resolve_tolerance",
]
