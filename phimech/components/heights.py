"""Typical component heights by KiCad footprint name — the "how tall is this thing" oracle.

Why a table and not the 3D model: a `.kicad_pcb` names a `.step`/`.wrl` model per footprint, but
resolving `${KICAD10_3DMODEL_DIR}` and measuring a STEP costs a FreeCAD subprocess *per part*
(seconds), and half of a real board's models are missing on a headless box anyway. Enclosure
synthesis only needs the *tallest* part above and below the board to a fraction of a millimetre,
so a regex table over footprint names gets the right answer in microseconds and degrades
gracefully (`DEFAULT_HEIGHT`) instead of failing. `components/pcb.py` still records the model
path so a later pass can measure the few parts that matter.

Numbers are body heights in mm from the board surface, taken from the manufacturer drawings the
KiCad libraries are generated from (JEDEC/IPC nominal + typical package max). Two values per
part:

* **above** — body height on the side the footprint is placed.
* **below** — how far leads protrude out the *other* side of the board. For through-hole parts
  this is the clipped-lead allowance (headers ship with ~3 mm of pin below the board before
  trimming); for SMD it is 0. It matters because an enclosure floor sits under the board and a
  standoff shorter than the protrusion means the board does not seat.

Frame note: "above"/"below" here are relative to the *board surface the footprint sits on*.
`height_for()` converts to the PhiMech mounting frame (board bottom at Z = 0, +Z up) by swapping
the two values for back-side (`B.Cu`) footprints.

stdlib only; must import under python 3.9.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Pattern, Tuple, Union

#: Height used when nothing in the table matches — a 2.5 mm generic SMD/THT body. Chosen so a
#: mystery part does not silently create a collision, but also does not blow up the enclosure.
DEFAULT_HEIGHT = 2.5

#: Lead protrusion under the board for an untrimmed through-hole part.
THT_PROTRUSION = 3.0


@dataclass(frozen=True)
class FromCapture:
    """A height derived from a number captured out of the footprint name.

    Electrolytic cans (`CP_Radial_D6.3mm_H11.0mm`) and other parameterised generators encode
    their own dimensions, so a single pattern covers the whole family: `D * scale + offset`.
    """

    group: int = 1
    scale: float = 1.0
    offset: float = 0.0

    def value(self, m: "re.Match") -> float:
        try:
            captured = float(m.group(self.group))
        except (IndexError, TypeError, ValueError):
            return DEFAULT_HEIGHT
        return captured * self.scale + self.offset


HeightSpec = Union[float, FromCapture]

#: footprint-name regex (searched, case-insensitive) -> body height above the board, mm.
#: Order matters: the FIRST match wins, so specific patterns come before generic ones.
HEIGHT_TABLE: Dict[str, HeightSpec] = {
    # --- nothing there at all --------------------------------------------------------
    r"^MountingHole": 0.0,
    r"^TestPoint": 0.5,
    r"^Fiducial": 0.0,
    r"^NetTie": 0.0,
    r"^Solder(Jumper|Wire)": 0.6,
    # --- connectors (tall, and the reason enclosures need cutouts) -------------------
    r"USB_C_Receptacle": 3.2,
    r"USB_(Micro|Mini)": 2.9,
    r"USB_[AB]_": 11.0,
    r"RJ45": 13.5,
    r"Barrel_Jack": 11.0,
    r"^Jack_|Audio_Jack|PJ3": 6.5,
    r"microSD|SD_(Card|TF)": 2.0,
    r"JST_XH": 7.0,
    r"JST_(PH|SH|ZH)": 6.0,
    r"Molex_(Micro|Pico)": 4.5,
    r"TerminalBlock|Screw_Terminal|ScrewTerminal": 10.0,
    r"IDC[-_]Header|Box.?Header": 9.0,
    r"PinSocket_": 8.5,
    r"PinHeader_.*_P2\.54mm_(Vertical|Horizontal)": 8.5,
    r"PinHeader_.*_P1\.27mm": 4.5,
    r"PinHeader_": 8.5,
    r"Conn_.*Header": 8.5,
    # --- modules -------------------------------------------------------------------
    r"Teensy": 12.0,
    r"Arduino|Nucleo|Feather|Pro_?Micro|Pro_?Mini": 12.0,
    r"RaspberryPi|RPi_?Zero": 12.0,
    r"ESP32|ESP-?12|ESP-?WROOM": 3.5,
    # --- discretes: metric imperial code in the name -------------------------------
    r"_0402(Metric)?|_1005Metric": 0.5,
    r"_0603(Metric)?|_1608Metric": 0.6,
    r"LED_0805|_0805.*LED": 1.1,
    r"_0805(Metric)?|_2012Metric": 0.7,
    r"_1206(Metric)?|_3216Metric": 1.0,
    r"_1210(Metric)?|_3225Metric": 1.4,
    r"_2010(Metric)?|_5025Metric": 1.2,
    r"_2512(Metric)?|_6332Metric": 1.2,
    # --- capacitors / inductors with encoded diameter ------------------------------
    r"CP_Radial_D(\d+\.?\d*)mm": FromCapture(1, 1.6),
    r"C_Radial_D(\d+\.?\d*)mm": FromCapture(1, 1.4),
    r"C_Disc_D(\d+\.?\d*)mm": FromCapture(1, 1.1),
    r"L_Toroid_D(\d+\.?\d*)mm": FromCapture(1, 0.8),
    r"L_(Wuerth|Bourns|Coilcraft|Taiyo)": 4.0,
    # --- semiconductor packages ----------------------------------------------------
    r"SOT-?23": 1.1,
    r"SOT-?89|SOT-?223": 1.8,
    r"SOD-?(123|323|523|882)": 1.1,
    r"DFN|QFN|LGA|BGA": 1.0,
    r"LQFP|TQFP|QFP": 1.0,
    r"SOIC|SO-?\d+|MSOP|TSSOP|SSOP": 1.75,
    r"DIP-?\d+": 4.5,
    r"TO-?220": 16.0,
    r"TO-?247": 21.0,
    r"TO-?92": 5.2,
    r"TO-?252|DPAK": 2.3,
    r"TO-?263|D2PAK": 4.8,
    # --- everything else with a body worth naming ----------------------------------
    r"Crystal_SMD|Crystal_HC49|^Crystal": 3.5,
    r"Oscillator": 2.0,
    r"Buzzer|Speaker": 9.0,
    r"^SW_(Push|SPST|DIP|Tactile)|Tactile|Push_?Button": 4.3,
    r"^SW_(Slide|Rocker|Toggle)": 8.0,
    r"Relay": 15.5,
    r"Fuse(Holder)?": 5.0,
    r"BatteryHolder_.*(2032|1220)": 4.5,
    r"BatteryHolder": 20.0,
    r"Heatsink": 12.0,
    r"Inductor_SMD": 3.0,
    r"Transformer": 12.0,
    r"Display|OLED|LCD": 3.5,
    r"Potentiometer": 7.0,
    r"^LED_D(\d+\.?\d*)mm": FromCapture(1, 1.0, 0.6),
    r"^LED_": 1.1,
    r"^D_": 1.1,
}

#: Patterns that mean "through hole": their leads stick out the far side of the board.
THT_PATTERNS = (
    r"PinHeader_",
    r"PinSocket_",
    r"IDC[-_]Header|Box.?Header",
    r"TerminalBlock|Screw_Terminal|ScrewTerminal",
    r"^Jack_|Barrel_Jack|Audio_Jack",
    r"USB_[AB]_",
    r"RJ45",
    r"DIP-?\d+",
    r"TO-?(92|220|247)",
    r"CP_Radial|C_Disc|C_Radial",
    r"JST_(XH|VH|EH)",
    r"Molex_KK",
    r"Crystal_HC49",
    r"^SW_(Slide|Rocker|Toggle)",
    r"Relay",
    r"BatteryHolder",
    r"Buzzer",
    r"Potentiometer",
)

#: Connector families, and how they mate: 'top' = plug arrives along +Z (vertical header),
#: 'edge' = plug arrives horizontally, so the cutout belongs on the nearest board edge.
#: `opening` is a (width, height) hint in mm for parts whose size is not derivable from the pad
#: row; None means "measure the pads" (pcb.py does that and is more accurate).
CONNECTOR_TABLE: Dict[str, Tuple[str, Optional[Tuple[float, float]], float]] = {
    # pattern: (mating face, opening hint (w, h) or None, protrusion past the body)
    r"USB_C_Receptacle.*(Horizontal|_16P|_24P)": ("edge", (9.2, 3.3), 0.0),
    r"USB_C_Receptacle": ("edge", (9.2, 3.3), 0.0),
    r"USB_(Micro|Mini)": ("edge", (8.0, 3.0), 0.0),
    r"USB_[AB]_": ("edge", (13.0, 6.0), 0.0),
    r"RJ45": ("edge", (16.0, 13.5), 0.0),
    r"Barrel_Jack": ("edge", (9.0, 11.0), 0.0),
    r"^Jack_|Audio_Jack|PJ3": ("edge", (7.0, 6.5), 0.0),
    r"microSD|SD_(Card|TF)": ("edge", (14.0, 2.0), 0.0),
    r"HDMI|DisplayPort": ("edge", (15.0, 6.0), 0.0),
    r"_Horizontal\b": ("edge", None, 0.0),
    r"TerminalBlock|Screw_Terminal|ScrewTerminal": ("top", None, 0.0),
    r"JST_": ("top", None, 0.0),
    r"Molex_(Micro|Pico)": ("top", None, 0.0),
    r"IDC[-_]Header|Box.?Header": ("top", None, 0.0),
    r"PinSocket_": ("top", None, 0.0),
    r"PinHeader_": ("top", None, 0.0),
    r"Conn_.*Header": ("top", None, 0.0),
}

_HEIGHT_COMPILED: Tuple[Tuple[Pattern[str], HeightSpec], ...] = tuple(
    (re.compile(p, re.IGNORECASE), v) for p, v in HEIGHT_TABLE.items()
)
_THT_COMPILED: Tuple[Pattern[str], ...] = tuple(re.compile(p, re.IGNORECASE) for p in THT_PATTERNS)
ConnSpec = Tuple[str, Optional[Tuple[float, float]], float]
_CONNECTOR_COMPILED: Tuple[Tuple[Pattern[str], ConnSpec], ...] = tuple(
    (re.compile(p, re.IGNORECASE), v) for p, v in CONNECTOR_TABLE.items()
)


def _bare(footprint_name: str) -> str:
    """Strip the `library:` prefix — the table is written against footprint names alone."""
    return footprint_name.rsplit(":", 1)[-1] if footprint_name else ""


def body_height(footprint_name: str) -> float:
    """Body height above the board surface, mm. `DEFAULT_HEIGHT` when nothing matches."""
    name = _bare(footprint_name)
    if not name:
        return DEFAULT_HEIGHT
    for rx, spec in _HEIGHT_COMPILED:
        m = rx.search(name)
        if m:
            return spec.value(m) if isinstance(spec, FromCapture) else float(spec)
    return DEFAULT_HEIGHT


def matched_pattern(footprint_name: str) -> Optional[str]:
    """Which table entry decided the height — for `inspect()` output and debugging."""
    name = _bare(footprint_name)
    for rx, _spec in _HEIGHT_COMPILED:
        if rx.search(name):
            return rx.pattern
    return None


def is_through_hole(footprint_name: str) -> bool:
    """True when the part's leads protrude out the far side of the board."""
    name = _bare(footprint_name)
    return any(rx.search(name) for rx in _THT_COMPILED)


def height_for(footprint_name: str, layer: str = "F") -> Tuple[float, float]:
    """`(above_board, below_board)` in the PhiMech mounting frame, mm.

    `layer` is `'F'`/`'F.Cu'` for the top side, `'B'`/`'B.Cu'` for the bottom. A back-side part
    has its body *below* the board and its lead protrusion *above*, so the pair is swapped —
    doing that here is what lets pcb.py stay frame-agnostic.
    """
    body = body_height(footprint_name)
    lead = THT_PROTRUSION if is_through_hole(footprint_name) else 0.0
    if body <= 0.0 and lead <= 0.0:
        return (0.0, 0.0)
    back = str(layer).upper().startswith("B")
    return (lead, body) if back else (body, lead)


def is_connector(footprint_name: str) -> bool:
    """True for anything a cable, plug or mating header attaches to."""
    name = _bare(footprint_name)
    return any(rx.search(name) for rx, _info in _CONNECTOR_COMPILED)


def connector_face_info(footprint_name: str) -> Optional[Dict[str, object]]:
    """How a connector mates, or None when the footprint is not a connector.

    Keys:
      * `face`        — `'top'` (plug arrives along +Z: vertical headers, JST, terminal blocks)
                        or `'edge'` (plug arrives horizontally: USB, barrel, RJ45, any
                        `_Horizontal` variant). `'edge'` means pcb.py must pick the nearest
                        board edge for the cutout.
      * `opening`     — `(w, h)` hint in mm, or None when the pad row is a better measure.
      * `protrusion`  — how far the mating face already sticks out past the body, mm.
      * `body_height` — the part's own height above the board, mm (cutout height for 'top').
      * `pattern`     — the CONNECTOR_TABLE entry that matched, for traceability.
    """
    name = _bare(footprint_name)
    if not name:
        return None
    for rx, (face, opening, protrusion) in _CONNECTOR_COMPILED:
        if rx.search(name):
            return {
                "face": face,
                "opening": tuple(opening) if opening else None,
                "protrusion": float(protrusion),
                "body_height": body_height(footprint_name),
                "pattern": rx.pattern,
            }
    return None


def is_mounting_hole(footprint_name: str) -> bool:
    """True for `MountingHole*` footprints (no body, but a hole the enclosure must respect)."""
    return _bare(footprint_name).lower().startswith("mountinghole")


#: `MountingHole_3.2mm_M3` -> 3.2. Applied to the bare footprint name.
HOLE_DIAMETER_RE = re.compile(r"_(\d+\.?\d*)mm")


def hole_diameter_from_name(footprint_name: str) -> Optional[float]:
    """Nominal hole diameter encoded in a MountingHole footprint name, mm, or None."""
    m = HOLE_DIAMETER_RE.search(_bare(footprint_name))
    return float(m.group(1)) if m else None


__all__ = [
    "DEFAULT_HEIGHT",
    "THT_PROTRUSION",
    "FromCapture",
    "HEIGHT_TABLE",
    "THT_PATTERNS",
    "CONNECTOR_TABLE",
    "HOLE_DIAMETER_RE",
    "body_height",
    "matched_pattern",
    "is_through_hole",
    "height_for",
    "is_connector",
    "connector_face_info",
    "is_mounting_hole",
    "hole_diameter_from_name",
]
