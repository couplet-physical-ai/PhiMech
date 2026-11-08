"""STEP inspection and a SolidWorks-readiness verdict, using only the standard library.

Why this exists
---------------
"Can SolidWorks open this?" is a question about the *file*, not about our kernel, and STEP
(ISO 10303) is plain ASCII. So the whole check is a parse — no OpenCascade, no FreeCAD, no
902 MB of OCP wheels. That keeps it inside the stdlib-only core (the core boundary) and makes
it fast enough to run as a gate on every build.

The three facts that decide whether an import is useful
-------------------------------------------------------
1. **Schema.** `FILE_SCHEMA` names the application protocol. SolidWorks reads AP203, AP214
   and AP242. Anything else is a coin toss, so we report it rather than guess.
2. **Solid or salad.** A STEP file can carry a real B-rep solid (`MANIFOLD_SOLID_BREP` over
   `ADVANCED_FACE`), a tessellated approximation (`TRIANGULATED_*`, `FACETED_BREP`), or loose
   surfaces with no closed shell. Only the first imports as a *solid body* you can cut, mate
   and measure; the second lands as a graphics blob. Both "open fine", which is exactly why
   file size and a successful import tell you nothing.
3. **Units.** STEP states its own length unit. Getting this wrong is the classic 25.4×
   disaster, and it is silent — the geometry is valid, just the wrong size.

What this module deliberately does NOT do
-----------------------------------------
It does not validate topology. A file can be a perfectly formed `MANIFOLD_SOLID_BREP` and
still be self-intersecting or unorientable; proving otherwise needs a geometry kernel. So
`readiness()` reports *transferability*, never correctness, and says so in its own output —
a check that overstates its reach is worse than no check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Application protocols SolidWorks imports natively. AP242 is the current one and the only
# one carrying PMI/tolerance data; AP214 is the common CAD interchange default; AP203 is
# geometry-only and still ubiquitous in older vendor downloads.
KNOWN_PROTOCOLS: Dict[str, str] = {
    "203": "AP203 (configuration-controlled design) — geometry only, no colour or PMI",
    "214": "AP214 (automotive design) — geometry, colour and assembly structure",
    "242": "AP242 (managed model-based 3D engineering) — adds PMI and tolerances",
}

# Entities that prove a closed solid body is present.
SOLID_ENTITIES: Tuple[str, ...] = ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "CLOSED_SHELL")
# Entities that mean the geometry is an approximation, not a solid.
MESH_ENTITIES: Tuple[str, ...] = (
    "TRIANGULATED_FACE_SET",
    "TRIANGULATED_SURFACE_SET",
    "TESSELLATED_SHELL",
    "TESSELLATED_SOLID",
    "FACETED_BREP",
)
# Surface types worth reporting: analytic surfaces survive a round trip cleanly and are what
# a fillet or a hole should look like; B-splines are legitimate but lossy to re-approximate.
SURFACE_ENTITIES: Tuple[str, ...] = (
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "B_SPLINE_SURFACE",
    "SURFACE_OF_REVOLUTION",
    "SURFACE_OF_LINEAR_EXTRUSION",
)
TOPOLOGY_ENTITIES: Tuple[str, ...] = ("ADVANCED_FACE", "FACE_SURFACE", "EDGE_CURVE", "VERTEX_POINT")

# STEP spells SI prefixes as enumeration literals; anything not millimetre needs a scale on
# import, and inch files are the usual source of a silent 25.4x error.
_SI_LENGTH = re.compile(r"SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.METRE\.\s*\)", re.I)
_CONVERSION_UNIT = re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']*)'", re.I)
_SCHEMA = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", re.I | re.S)
_NAME = re.compile(r"FILE_NAME\s*\(\s*'([^']*)'", re.I | re.S)
_DESCRIPTION = re.compile(r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'", re.I | re.S)
_FILE_NAME = re.compile(r"FILE_NAME\s*\((.*?)\)\s*;", re.I | re.S)
# An entity instance: `#12 = NAME(...)`. Complex instances nest several names in parentheses,
# so we capture the whole right-hand side and pull every bare identifier out of it.
_INSTANCE = re.compile(r"#\d+\s*=\s*(.+?);", re.S)
_IDENT = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*\(")

# Below this many faces, an all-planar solid is far more likely to be a genuine prismatic or
# faceted part than a tessellation. An imported sample may exceed that number; a machined bracket
# sits in the tens.
_TESSELLATION_FACE_FLOOR = 200

_PREFIX_TO_MM = {
    ".MILLI.": 1.0,
    ".CENTI.": 10.0,
    ".DECI.": 100.0,
    "$": 1000.0,  # bare metre
    ".KILO.": 1_000_000.0,
    ".MICRO.": 0.001,
}


@dataclass
class StepFacts:
    """Everything the parse could establish about one STEP file."""

    path: Path
    size_bytes: int
    protocol: Optional[str] = None           # "214", "242", ...
    schema_raw: Optional[str] = None
    description: Optional[str] = None
    originating_system: Optional[str] = None
    preprocessor: Optional[str] = None
    length_unit: Optional[str] = None        # "mm", "in", "m", ...
    mm_per_unit: Optional[float] = None
    entity_counts: Dict[str, int] = field(default_factory=dict)
    parse_error: Optional[str] = None

    # --- derived ------------------------------------------------------------------

    @property
    def solids(self) -> int:
        return sum(self.entity_counts.get(e, 0) for e in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS"))

    @property
    def closed_shells(self) -> int:
        return self.entity_counts.get("CLOSED_SHELL", 0)

    @property
    def mesh_entities(self) -> int:
        return sum(self.entity_counts.get(e, 0) for e in MESH_ENTITIES)

    @property
    def faces(self) -> int:
        return sum(self.entity_counts.get(e, 0) for e in ("ADVANCED_FACE", "FACE_SURFACE"))

    @property
    def edges(self) -> int:
        return self.entity_counts.get("EDGE_CURVE", 0)

    @property
    def analytic_surfaces(self) -> int:
        keys = ("PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
                "TOROIDAL_SURFACE")
        return sum(self.entity_counts.get(k, 0) for k in keys)

    @property
    def revolved_surfaces(self) -> int:
        """Surfaces of revolution, which OCC uses for a *variable-radius* fillet blend.

        Deliberately not counted as analytic: a surface of revolution revolves an arbitrary
        curve, which may itself be a spline, so the name alone does not promise an exact radius.
        It is equally not a spline surface, and leaving it out of both totals made a filleted
        box read as "26 faces, 18 analytic" with eight faces unaccounted for.

        Seen in practice when two fillets of *different* radii meet: equal radii give a
        TOROIDAL_SURFACE, unequal ones give this.
        """
        return self.entity_counts.get("SURFACE_OF_REVOLUTION", 0)

    @property
    def curved_surfaces(self) -> int:
        """Every non-planar surface, however it is represented. The honest "is it round?"."""
        return sum(count for name, count in self.entity_counts.items()
                   if name in SURFACE_ENTITIES and name != "PLANE")

    @property
    def spline_surfaces(self) -> int:
        keys = ("B_SPLINE_SURFACE_WITH_KNOTS", "B_SPLINE_SURFACE")
        return sum(self.entity_counts.get(k, 0) for k in keys)

    @property
    def planar_fraction(self) -> Optional[float]:
        """Share of faces lying on a plane. ~1.0 on a tessellated shell."""
        if not self.faces:
            return None
        return self.entity_counts.get("PLANE", 0) / float(self.faces)

    @property
    def looks_tessellated(self) -> bool:
        """True when this is a triangle mesh wearing a B-rep costume.

        A STEP written by converting a mesh face-by-face is a genuine `MANIFOLD_SOLID_BREP`,
        so `body_kind` says `solid` and the import succeeds -- but every face is one triangle.
        It cannot be filleted, feature recognition finds nothing, and the file is enormous.
        Two signatures together, because either alone has honest counterexamples: nearly every
        face is planar (a real prismatic part has curves, a real faceted part has few faces),
        and the face/edge ratio approaches the 2/3 of a closed triangulation.
        """
        if self.faces < _TESSELLATION_FACE_FLOOR or not self.edges:
            return False
        fraction = self.planar_fraction or 0.0
        ratio = self.faces / float(self.edges)
        return fraction >= 0.95 and 0.60 <= ratio <= 0.70

    @property
    def body_kind(self) -> str:
        """`solid`, `mesh`, `surfaces` or `empty` — what an import will actually yield."""
        if self.solids or self.closed_shells:
            return "solid"
        if self.mesh_entities:
            return "mesh"
        if self.faces:
            return "surfaces"
        return "empty"

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "protocol": self.protocol,
            "schema_raw": self.schema_raw,
            "description": self.description,
            "originating_system": self.originating_system,
            "preprocessor": self.preprocessor,
            "length_unit": self.length_unit,
            "mm_per_unit": self.mm_per_unit,
            "body_kind": self.body_kind,
            "planar_fraction": self.planar_fraction,
            "curved_surfaces": self.curved_surfaces,
            "revolved_surfaces": self.revolved_surfaces,
            "looks_tessellated": self.looks_tessellated,
            "solids": self.solids,
            "closed_shells": self.closed_shells,
            "faces": self.faces,
            "edges": self.edges,
            "analytic_surfaces": self.analytic_surfaces,
            "spline_surfaces": self.spline_surfaces,
            "mesh_entities": self.mesh_entities,
            "entity_counts": dict(sorted(self.entity_counts.items())),
            "parse_error": self.parse_error,
        }


def _split_top_level(args: str) -> List[str]:
    """Split a STEP argument list on commas that are not nested or inside a literal."""
    out: List[str] = []
    depth = 0
    in_str = False
    buf: List[str] = []
    i = 0
    while i < len(args):
        ch = args[i]
        if in_str:
            if ch == "'":
                # A doubled quote is an escaped quote, not the end of the literal.
                if i + 1 < len(args) and args[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_str = False
            buf.append(ch)
        elif ch == "'":
            in_str = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf).strip())
    return out


def _unquote(field: str) -> Optional[str]:
    field = field.strip()
    if field.startswith("'") and field.endswith("'") and len(field) >= 2:
        value = field[1:-1].replace("''", "'").strip()
        return value or None
    return None


def _split_sections(text: str) -> Tuple[str, str]:
    """Return (header, data). Either may be empty if the file is truncated."""
    up = text.upper()
    h = up.find("HEADER")
    d = up.find("DATA")
    e = up.rfind("ENDSEC")
    if h == -1 or d == -1:
        return text, text
    header = text[h:d]
    data = text[d:e if e > d else len(text)]
    return header, data


def _strip_strings_and_comments(s: str) -> str:
    """Remove `/* ... */` comments and `'...'` literals so identifiers cannot be faked.

    A part named `'TRIANGULATED_BRACKET'` must not make the file look tessellated.
    """
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    # STEP escapes a quote by doubling it, so consume '' inside a literal.
    return re.sub(r"'(?:[^']|'')*'", "''", s)


def _length_unit(header_and_data: str) -> Tuple[Optional[str], Optional[float]]:
    m = _SI_LENGTH.search(header_and_data)
    if m:
        prefix = m.group(1).upper()
        mm = _PREFIX_TO_MM.get(prefix)
        name = {".MILLI.": "mm", ".CENTI.": "cm", ".DECI.": "dm", "$": "m",
                ".KILO.": "km", ".MICRO.": "um"}.get(prefix)
        if mm is not None:
            return name, mm
        return prefix.strip("."), None
    c = _CONVERSION_UNIT.search(header_and_data)
    if c:
        raw = c.group(1).strip().lower()
        inch = {"inch": 25.4, "in": 25.4, "'inch'": 25.4}
        if raw in inch:
            return "in", 25.4
        if raw in ("foot", "ft"):
            return "ft", 304.8
        return raw, None
    return None, None


def inspect(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> StepFacts:
    """Parse a STEP file into `StepFacts`. Never raises for bad content.

    `max_bytes` caps the read so a mis-pointed path at a multi-GB file cannot wedge a build.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return StepFacts(path=path, size_bytes=0, parse_error=f"cannot stat: {exc}")

    facts = StepFacts(path=path, size_bytes=size)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(max_bytes)
    except OSError as exc:
        facts.parse_error = f"cannot read: {exc}"
        return facts

    if size > max_bytes:
        facts.parse_error = f"read truncated at {max_bytes} bytes of {size}; counts are partial"

    clean = _strip_strings_and_comments(text)
    header, data = _split_sections(clean)

    # Entity *counting* uses the cleaned text so a part named 'TRIANGULATED_BRACKET' cannot
    # fake a tessellation. Unit and provenance parsing needs the raw text, because the unit
    # name in CONVERSION_BASED_UNIT('INCH',...) is itself a string literal.
    raw_header, raw_data = _split_sections(text)
    m = _SCHEMA.search(raw_header) or _SCHEMA.search(text)
    if m:
        facts.schema_raw = m.group(1).strip()
        proto = re.search(r"10303[^0-9]+(\d{3})", facts.schema_raw)
        if proto:
            facts.protocol = proto.group(1)
        elif "AUTOMOTIVE_DESIGN" in facts.schema_raw.upper():
            facts.protocol = "214"
        elif "CONFIG_CONTROL_DESIGN" in facts.schema_raw.upper():
            facts.protocol = "203"
        elif "AP242" in facts.schema_raw.upper():
            facts.protocol = "242"
    d = _DESCRIPTION.search(raw_header)
    if d:
        facts.description = d.group(1).strip() or None
    # FILE_NAME fields, in ISO 10303-21 order: name, time_stamp, author[], organization[],
    # preprocessor_version, originating_system, authorisation. Index 5 is the CAD system that
    # produced the file, which is the one worth reporting.
    fn = _FILE_NAME.search(raw_header)
    if fn:
        fields = _split_top_level(fn.group(1))
        if len(fields) > 5:
            facts.originating_system = _unquote(fields[5])
        if len(fields) > 4:
            facts.preprocessor = _unquote(fields[4])

    facts.length_unit, facts.mm_per_unit = _length_unit(raw_data or text)

    counts: Dict[str, int] = {}
    watch = set(SOLID_ENTITIES + MESH_ENTITIES + SURFACE_ENTITIES + TOPOLOGY_ENTITIES)
    for inst in _INSTANCE.finditer(data or clean):
        for ident in _IDENT.finditer(inst.group(1)):
            name = ident.group(1).upper()
            if name in watch:
                counts[name] = counts.get(name, 0) + 1
    facts.entity_counts = counts
    return facts


# --------------------------------------------------------------------------------------
# Readiness verdict
# --------------------------------------------------------------------------------------

@dataclass
class Finding:
    """One numeric or categorical check, in the tri-state shape the gates use."""

    name: str
    ok: Optional[bool]        # None = could not be determined, never rendered as a pass
    severity: str             # "fail" | "warn" | "info"
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "ok": self.ok, "severity": self.severity, "detail": self.detail}


def readiness(facts: StepFacts, *, expect_mm: bool = True) -> Dict[str, object]:
    """Judge whether this file transfers usefully into SolidWorks.

    Returns `{ready, findings[], summary}`. `ready` is tri-state: `None` when a required
    fact could not be read, because "unknown" must never render as "fine".
    """
    findings: List[Finding] = []

    if facts.parse_error and not facts.entity_counts:
        findings.append(Finding("parses", False, "fail", facts.parse_error))
        return {"ready": False, "findings": [f.to_dict() for f in findings],
                "summary": "not a readable STEP file"}

    # 1. Protocol
    if facts.protocol in KNOWN_PROTOCOLS:
        findings.append(Finding("protocol_supported", True, "info",
                                f"AP{facts.protocol}: {KNOWN_PROTOCOLS[facts.protocol]}"))
    elif facts.protocol:
        findings.append(Finding("protocol_supported", False, "fail",
                                f"AP{facts.protocol} is not a protocol SolidWorks imports "
                                f"natively (schema: {facts.schema_raw})"))
    else:
        findings.append(Finding("protocol_supported", None, "warn",
                                "no FILE_SCHEMA found; the importer will have to guess"))

    # 2. Body kind — the check that actually predicts a usable import
    kind = facts.body_kind
    if kind == "solid":
        findings.append(Finding("imports_as_solid", True, "info",
                                f"{facts.solids} manifold solid(s), {facts.closed_shells} closed "
                                f"shell(s), {facts.faces} faces — imports as a solid body"))
    elif kind == "mesh":
        findings.append(Finding("imports_as_solid", False, "fail",
                                f"{facts.mesh_entities} tessellated entities and no closed shell: "
                                "SolidWorks will import a graphics/mesh body that cannot be cut, "
                                "mated or measured as a solid"))
    elif kind == "surfaces":
        findings.append(Finding("imports_as_solid", False, "fail",
                                f"{facts.faces} faces but no closed shell — imports as loose "
                                "surfaces needing a manual knit"))
    else:
        findings.append(Finding("imports_as_solid", False, "fail",
                                "no faces, shells or solids found: the file carries no geometry"))

    # 3. Units
    if facts.length_unit is None:
        findings.append(Finding("units_declared", None, "warn",
                                "no length unit found; import scale is a guess"))
    elif expect_mm and facts.length_unit != "mm":
        factor = facts.mm_per_unit
        scale = f" (×{factor:g} to reach mm)" if factor else ""
        findings.append(Finding("units_declared", False, "fail",
                                f"file is in {facts.length_unit}, not mm{scale}: PhiMech is mm "
                                "throughout, so this needs an explicit conversion on import"))
    else:
        findings.append(Finding("units_declared", True, "info",
                                f"length unit is {facts.length_unit}"))

    # 4. A mesh wearing a B-rep costume. Reported separately from `imports_as_solid` because
    #    the import genuinely succeeds -- what fails is every downstream CAD operation.
    if facts.looks_tessellated:
        pct = (facts.planar_fraction or 0.0) * 100.0
        findings.append(Finding(
            "true_brep", False, "warn",
            f"{facts.faces} faces, {pct:.0f}% planar: this is a tessellated mesh converted "
            "face-by-face into a solid, not true B-rep. It imports as a solid body but cannot "
            "be filleted or edited, feature recognition will find nothing, and the file is "
            f"{facts.size_bytes // 1024} KB. Emitted when a part uses geometry FreeCAD cannot "
            "build natively, so the pipeline falls back to the step_via_stl route"))
    elif facts.body_kind == "solid" and facts.faces:
        findings.append(Finding("true_brep", True, "info",
                                f"true B-rep: {facts.faces} faces, "
                                f"{facts.analytic_surfaces} analytic surfaces"))

    # 5. Surface quality — advisory: splines transfer, they just re-approximate
    if facts.spline_surfaces and facts.analytic_surfaces:
        findings.append(Finding("analytic_surfaces", True, "warn",
                                f"{facts.analytic_surfaces} analytic + {facts.spline_surfaces} "
                                "spline surfaces; splines survive transfer but are re-fitted, so "
                                "exact radii may drift"))
    elif facts.spline_surfaces:
        findings.append(Finding("analytic_surfaces", None, "warn",
                                f"{facts.spline_surfaces} spline surfaces and no analytic ones — "
                                "feature recognition in SolidWorks will likely find nothing"))

    # 6. Edit risk, borrowed as a *threshold idea* from agentcad's edit_risk.py (Apache-2.0)
    if facts.faces > 1000 or facts.edges > 2000:
        findings.append(Finding("edit_risk", False, "warn",
                                f"{facts.faces} faces / {facts.edges} edges: large enough that "
                                "boolean and fillet edits commonly fail or time out"))

    fails = [f for f in findings if f.ok is False and f.severity == "fail"]
    unknown = [f for f in findings if f.ok is None and f.severity != "info"]
    if fails:
        ready: Optional[bool] = False
        summary = "; ".join(f.detail.split(":")[0] for f in fails)
    elif unknown:
        ready = None
        summary = "could not confirm: " + "; ".join(f.name for f in unknown)
    else:
        ready = True
        unit = facts.length_unit or "?"
        if facts.looks_tessellated:
            summary = (f"AP{facts.protocol} in {unit}, but FACETED ({facts.faces} planar faces): "
                       "imports as a solid body that cannot be edited")
        else:
            curved = facts.curved_surfaces
            shape = f"{facts.faces} faces"
            if curved:
                shape += f" ({curved} curved)"
            summary = (f"AP{facts.protocol} solid in {unit}, {shape} — "
                       "imports into SolidWorks as a solid body")

    return {
        "ready": ready,
        "findings": [f.to_dict() for f in findings],
        "summary": summary,
        "caveat": ("transferability only: this is a file-format check, not a topology check. "
                   "A well-formed STEP can still be self-intersecting."),
    }
