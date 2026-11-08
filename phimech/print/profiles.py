"""Resolution of friendly printer/material/quality names to real OrcaSlicer profile files.

PhiMech never ships slicer settings of its own: the vendor profile pack that comes with
OrcaSlicer is the ground truth for nozzle temperatures, flow, and machine limits.  This module
discovers that pack by globbing and reading each profile's JSON ``name`` /
``compatible_printers`` so a profile-pack update cannot silently break us.

Two verified facts drive the layout of the code:

* Filament profiles for the X-Max 4 live in the ``filament/X4/`` **subfolder**.  The identically
  named files in ``filament/`` target other printers and are not ``compatible_printers`` for the
  Max 4, so a naive ``filament/*.json`` glob picks the wrong preset.
* The profiles are an ``inherits`` chain, and **the OrcaSlicer CLI does not resolve it**.  Passing
  ``filament/X4/QIDI PETG Basic @Qidi X-Max 4 0.4 nozzle.json`` (7 keys, all overrides) directly
  yields a slice at PLA's 200/45 °C with ``filament_density = 0`` — which is exactly why
  ``; total filament used [g]`` printed ``0.00`` in the original hand-run recipe.  ``flatten()``
  merges the chain child-over-parent, and the same slice then reports PETG 240/70 °C, density
  1.24 and 3.40 g.  See :mod:`phimech.print.slicer`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_PROFILES_FALLBACK = "/Applications/OrcaSlicer.app/Contents/Resources/profiles/Qidi"

# Friendly printer id -> the vendor's ``printer_model`` string.  Written out rather than derived
# because the vendor names are inconsistent ("X-Max 4" the process suffix vs "Qidi X-Max 4" the
# model), and a wrong guess here silently slices for the wrong machine.
_PRINTER_ALIASES = {
    "qidi-max4": "Qidi X-Max 4",
    "qidi-x-max-4": "Qidi X-Max 4",
    "qidi-xmax4": "Qidi X-Max 4",
    "x-max-4": "Qidi X-Max 4",
    "xmax4": "Qidi X-Max 4",
    "max4": "Qidi X-Max 4",
    "qidi-plus4": "Qidi X-Plus 4",
    "qidi-x-plus-4": "Qidi X-Plus 4",
    "plus4": "Qidi X-Plus 4",
    "qidi-q1-pro": "Qidi Q1 Pro",
    "q1-pro": "Qidi Q1 Pro",
}

# Density (g/cm3) of last resort, only used when the resolved profile chain has no
# ``filament_density`` at all.  Values are the vendor's published figures for the stock presets.
_FALLBACK_DENSITY = {
    "PLA": 1.24,
    "PETG": 1.27,
    "ABS": 1.04,
    "ASA": 1.07,
    "PC": 1.20,
    "TPU": 1.21,
    "PA": 1.14,
    "PVA": 1.23,
}

_DEFAULT_DENSITY = 1.24


class ProfileError(ValueError):
    """Raised when a printer/material/quality name cannot be resolved to a profile file."""


@dataclass
class Profiles:
    """The three profile files one OrcaSlicer invocation needs, plus their resolved names."""

    printer: str
    machine_path: Path
    process_path: Path
    filament_path: Path
    machine: str
    process: str
    filament: str
    nozzle_mm: float
    layer_height_mm: Optional[float] = None
    filament_type: Optional[str] = None
    filament_density: Optional[float] = None
    filament_diameter_mm: Optional[float] = None
    nozzle_temp_c: Optional[int] = None
    bed_temp_c: Optional[int] = None
    root: Optional[Path] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "printer": self.printer,
            "machine": self.machine,
            "process": self.process,
            "filament": self.filament,
            "machine_path": str(self.machine_path),
            "process_path": str(self.process_path),
            "filament_path": str(self.filament_path),
            "nozzle_mm": self.nozzle_mm,
            "layer_height_mm": self.layer_height_mm,
            "filament_type": self.filament_type,
            "filament_density": self.filament_density,
            "filament_diameter_mm": self.filament_diameter_mm,
            "nozzle_temp_c": self.nozzle_temp_c,
            "bed_temp_c": self.bed_temp_c,
        }


@dataclass
class _Profile:
    """One profile JSON as found on disk."""

    name: str
    path: Path
    data: Dict[str, object]

    @property
    def kind(self) -> str:
        return str(self.data.get("type", ""))

    @property
    def instantiable(self) -> bool:
        return str(self.data.get("instantiation", "true")).lower() != "false"

    @property
    def short_name(self) -> str:
        """The part of the name before the ``@printer`` suffix ('QIDI PETG Basic')."""
        return self.name.split("@")[0].strip()

    def compatible_with(self, machine_name: str) -> bool:
        raw = self.data.get("compatible_printers")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return False
        return machine_name in [str(item) for item in raw]


def profiles_root() -> Optional[Path]:
    """Locate the vendor profile directory, honouring ``PHIMECH_ORCA_PROFILES``."""
    configured = os.environ.get("PHIMECH_ORCA_PROFILES")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else None
    path = Path(_PROFILES_FALLBACK)
    return path if path.is_dir() else None


def _require_root() -> Path:
    root = profiles_root()
    if root is None:
        # No str.format() here: the message names ${PHIMECH_ORCA_PROFILES}, and a brace-wrapped
        # token in a format string is read as a field name, so formatting raised KeyError and
        # this error path crashed instead of reporting a missing install.
        raise ProfileError(
            "OrcaSlicer profiles not found. Looked at ${PHIMECH_ORCA_PROFILES} then "
            + str(_PROFILES_FALLBACK)
            + ". Install OrcaSlicer or set PHIMECH_ORCA_PROFILES."
        )
    return root


_INDEX_CACHE: Dict[str, Dict[str, _Profile]] = {}


def _index(root: Path) -> Dict[str, _Profile]:
    """Map profile ``name`` -> profile for every JSON under ``root`` (cached per root).

    Roughly 1.5k files for the Qidi pack; reading them all costs ~120 ms once, and every later
    lookup (including the ``inherits`` walk) is a dict hit.
    """
    key = str(root)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index: Dict[str, _Profile] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name not in index:
            index[name] = _Profile(name=name, path=path, data=data)
    _INDEX_CACHE[key] = index
    return index


def clear_cache() -> None:
    """Drop the profile index. Only needed by tests that point at a synthetic pack."""
    _INDEX_CACHE.clear()


def flatten(name: str, root: Optional[Path] = None) -> Dict[str, object]:
    """Merge a profile with every ancestor in its ``inherits`` chain, child overriding parent.

    The CLI needs this: it treats a ``--load-settings`` file as the complete preset.
    """
    root = root or _require_root()
    index = _index(root)
    chain: List[_Profile] = []
    seen = set()
    current: Optional[str] = name
    while current and current in index and current not in seen:
        seen.add(current)
        profile = index[current]
        chain.append(profile)
        inherited = profile.data.get("inherits")
        current = inherited if isinstance(inherited, str) and inherited else None
    if not chain:
        raise ProfileError("unknown profile {!r}".format(name))
    if current:
        raise ProfileError(
            "profile {!r} inherits {!r}, which is not in {}".format(name, current, root)
        )
    merged: Dict[str, object] = {}
    for profile in reversed(chain):
        for key, value in profile.data.items():
            if key == "inherits":
                continue
            merged[key] = value
    # `from` and `instantiation` must survive: the CLI rejects a preset whose `from` is missing
    # ("file <x>.json's from  unsupported") and refuses to instantiate one marked false.
    merged["name"] = name
    merged["from"] = "system"
    merged["instantiation"] = "true"
    return merged


def _scalar(data: Dict[str, object], key: str) -> Optional[str]:
    """Read a profile value that the vendor stores as a 1-element list of strings."""
    value = data.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(data: Dict[str, object], key: str) -> Optional[float]:
    text = _scalar(data, key)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _integer(data: Dict[str, object], key: str) -> Optional[int]:
    value = _number(data, key)
    return None if value is None else int(round(value))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _machine_model(printer: str) -> str:
    """Friendly printer id -> vendor ``printer_model``."""
    slug = _slug(printer)
    if slug in _PRINTER_ALIASES:
        return _PRINTER_ALIASES[slug]
    return printer


def _machines(root: Path) -> List[_Profile]:
    return [
        profile
        for profile in _index(root).values()
        if profile.kind == "machine" and profile.instantiable and "printer_model" in profile.data
    ]


def list_printers(root: Optional[Path] = None) -> List[str]:
    """Every instantiable machine model in the pack, e.g. ``Qidi X-Max 4``."""
    root = root or _require_root()
    models = {str(profile.data.get("printer_model")) for profile in _machines(root)}
    return sorted(models)


def _find_machine(printer: str, nozzle: str, root: Path) -> _Profile:
    model = _machine_model(printer)
    machines = _machines(root)
    wanted_nozzle = "{:g}".format(float(nozzle))
    matches = [
        profile
        for profile in machines
        if str(profile.data.get("printer_model")) == model
        and wanted_nozzle in [str(n) for n in _as_list(profile.data.get("nozzle_diameter"))]
    ]
    if not matches:
        by_model = [p for p in machines if str(p.data.get("printer_model")) == model]
        if by_model:
            nozzles = sorted(
                {str(n) for p in by_model for n in _as_list(p.data.get("nozzle_diameter"))}
            )
            raise ProfileError(
                "printer {!r} ({}) has no {} mm nozzle profile. Available nozzles: {}".format(
                    printer, model, wanted_nozzle, ", ".join(nozzles)
                )
            )
        raise ProfileError(
            "unknown printer {!r} (resolved to model {!r}). Known models: {}. "
            "Known aliases: {}.".format(
                printer, model, ", ".join(list_printers(root)), ", ".join(sorted(_PRINTER_ALIASES))
            )
        )
    return sorted(matches, key=lambda p: p.name)[0]


def _as_list(value: object) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, list):
        return value
    return (value,)


def _compatible(root: Path, kind: str, machine_name: str) -> List[_Profile]:
    return [
        profile
        for profile in _index(root).values()
        if profile.kind == kind and profile.instantiable and profile.compatible_with(machine_name)
    ]


def _filament_candidates(machine: _Profile, root: Path) -> List[_Profile]:
    """Filament profiles declaring this machine — this is what puts us in ``filament/X4/``."""
    return sorted(_compatible(root, "filament", machine.name), key=lambda p: p.name)


def _process_candidates(machine: _Profile, root: Path) -> List[_Profile]:
    return sorted(_compatible(root, "process", machine.name), key=lambda p: p.name)


def _filament_type_of(profile: _Profile, root: Path) -> Optional[str]:
    """``filament_type`` after flattening — the leaf almost never carries it."""
    try:
        merged = flatten(profile.name, root)
    except ProfileError:
        merged = profile.data
    return _scalar(merged, "filament_type")


def list_materials(
    printer: str = "qidi-max4", nozzle: str = "0.4", root: Optional[Path] = None
) -> List[str]:
    """Friendly material keys accepted by :func:`resolve` for this printer."""
    root = root or _require_root()
    machine = _find_machine(printer, nozzle, root)
    keys = set()
    for profile in _filament_candidates(machine, root):
        keys.add(profile.short_name)
        material = _filament_type_of(profile, root)
        if material:
            keys.add(material)
    return sorted(keys)


def list_qualities(
    printer: str = "qidi-max4", nozzle: str = "0.4", root: Optional[Path] = None
) -> List[str]:
    """Friendly quality keys accepted by :func:`resolve` for this printer."""
    root = root or _require_root()
    machine = _find_machine(printer, nozzle, root)
    keys = set()
    for profile in _process_candidates(machine, root):
        height, style = _split_process(profile.short_name)
        if style:
            keys.add(_slug(style))
        if height is not None and style:
            keys.add("{:.2f}mm-{}".format(height, _slug(style)))
        if height is not None:
            keys.add("{:.2f}mm".format(height))
    keys.update({"draft", "fastest"})
    return sorted(keys)


_PROCESS_RE = re.compile(r"^\s*(?P<height>[0-9]*\.?[0-9]+)\s*mm\s*(?P<style>.*)$", re.IGNORECASE)


def _split_process(short_name: str) -> Tuple[Optional[float], str]:
    """``0.20mm Balanced Strength`` -> ``(0.20, 'Balanced Strength')``."""
    match = _PROCESS_RE.match(short_name)
    if not match:
        return None, short_name.strip()
    try:
        height = float(match.group("height"))
    except ValueError:
        return None, match.group("style").strip()
    return height, match.group("style").strip()


def _pick_material(candidates: List[_Profile], material: str, root: Path) -> Optional[_Profile]:
    """Resolve a material name, preferring the vendor's own stock preset.

    Accepts an exact profile name, the short name (``QIDI PETG Basic``), or a bare material type
    (``PETG``).  For a bare type the order is: QIDI ... Basic, any QIDI, Generic <type>, then
    alphabetical — so ``PETG`` lands on the vendor's stock preset rather than on ``PETG-CF``.
    """
    wanted = material.strip().lower()
    for profile in candidates:
        if profile.name.lower() == wanted or profile.short_name.lower() == wanted:
            return profile
    typed = [p for p in candidates if (_filament_type_of(p, root) or "").lower() == wanted]
    if not typed:
        return None

    def rank(profile: _Profile) -> Tuple[int, str]:
        short = profile.short_name
        upper = short.upper()
        if upper.startswith("QIDI") and "BASIC" in upper:
            return (0, short)
        if upper.startswith("QIDI"):
            return (1, short)
        if upper.startswith("GENERIC"):
            return (2, short)
        return (3, short)

    return sorted(typed, key=rank)[0]


def _pick_process(candidates: List[_Profile], quality: str) -> Optional[_Profile]:
    """Resolve a quality name to a process profile.

    Accepts an exact/short profile name (``0.20mm Standard @X-Max 4``, ``0.20mm Standard``), a
    style slug (``standard``, ``balanced-strength``, ``fine``), a bare layer height (``0.20mm``,
    ``0.2``), a ``height-style`` pair (``0.12mm-fine``), or ``draft``/``fastest`` for the
    coarsest layer available.
    """
    wanted = quality.strip().lower()
    for profile in candidates:
        if profile.name.lower() == wanted or profile.short_name.lower() == wanted:
            return profile
    parsed = [(p,) + _split_process(p.short_name) for p in candidates]

    if wanted in ("draft", "fastest"):
        with_height = [item for item in parsed if item[1] is not None]
        if not with_height:
            return None
        return sorted(with_height, key=lambda item: (-item[1], item[0].short_name))[0][0]

    # Numeric forms are matched on the raw text: _slug turns '0.08mm' into '0-08mm'.
    pair = re.match(r"^([0-9]*\.?[0-9]+)\s*mm[\s\-_]+(.+)$", wanted)
    if pair:
        height = float(pair.group(1))
        style = _slug(pair.group(2))
        exact = [
            item
            for item in parsed
            if item[1] is not None
            and abs(item[1] - height) < 1e-9
            and _slug(item[2]) == style
        ]
        if exact:
            return exact[0][0]
        return None

    height_only = re.match(r"^([0-9]*\.?[0-9]+)\s*(?:mm)?$", wanted)
    if height_only:
        height = float(height_only.group(1))
        same = [item for item in parsed if item[1] is not None and abs(item[1] - height) < 1e-9]
        if not same:
            return None
        # A bare height means "the standard style at that height" when one exists.
        standard = [item for item in same if _slug(item[2]) == "standard"]
        chosen = standard or same
        return sorted(chosen, key=lambda item: item[0].short_name)[0][0]

    styled = [item for item in parsed if _slug(item[2]) == _slug(wanted)]
    if not styled:
        return None
    # Deterministic pick for a bare style: closest to 0.20 mm, then the thinner layer.
    return sorted(
        styled,
        key=lambda item: (
            abs((item[1] if item[1] is not None else 99.0) - 0.20),
            item[1] if item[1] is not None else 99.0,
            item[0].short_name,
        ),
    )[0][0]


def resolve(
    printer: str = "qidi-max4",
    material: str = "PETG",
    quality: str = "standard",
    nozzle: str = "0.4",
    root: Optional[Path] = None,
) -> Profiles:
    """Resolve friendly names to the machine/process/filament profile files.

    Raises :class:`ProfileError` listing the valid options on any miss.
    """
    root = root or _require_root()
    machine = _find_machine(printer, nozzle, root)

    process_candidates = _process_candidates(machine, root)
    process = _pick_process(process_candidates, quality)
    if process is None:
        raise ProfileError(
            "unknown quality {!r} for {}. Valid keys: {}. Profiles: {}.".format(
                quality,
                machine.name,
                ", ".join(list_qualities(printer, nozzle, root)),
                ", ".join(p.short_name for p in process_candidates),
            )
        )

    filament_candidates = _filament_candidates(machine, root)
    filament = _pick_material(filament_candidates, material, root)
    if filament is None:
        raise ProfileError(
            "unknown material {!r} for {}. Valid keys: {}.".format(
                material, machine.name, ", ".join(list_materials(printer, nozzle, root))
            )
        )

    merged_filament = flatten(filament.name, root)
    merged_process = flatten(process.name, root)
    merged_machine = flatten(machine.name, root)

    filament_type = _scalar(merged_filament, "filament_type")
    density = _number(merged_filament, "filament_density")
    if not density:
        density = _FALLBACK_DENSITY.get((filament_type or "").upper(), _DEFAULT_DENSITY)
    diameter = _number(merged_filament, "filament_diameter") or 1.75
    nozzle_mm = _number(merged_machine, "nozzle_diameter") or float(nozzle)

    return Profiles(
        printer=printer,
        machine_path=machine.path,
        process_path=process.path,
        filament_path=filament.path,
        machine=machine.name,
        process=process.name,
        filament=filament.name,
        nozzle_mm=nozzle_mm,
        layer_height_mm=_number(merged_process, "layer_height"),
        filament_type=filament_type,
        filament_density=density,
        filament_diameter_mm=diameter,
        nozzle_temp_c=_integer(merged_filament, "nozzle_temperature"),
        bed_temp_c=_integer(merged_filament, "hot_plate_temp"),
        root=root,
    )


def write_flattened(
    profiles: Profiles,
    out_dir: Path,
    extra_settings: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    """Write the flattened machine/process/filament presets into ``out_dir``.

    Returns the three paths in the order the CLI wants them.  ``extra_settings`` keys are routed
    to whichever flattened preset already defines them (process wins for unknown keys, since
    that is where slicing knobs live).
    """
    root = profiles.root or _require_root()
    merged = {
        "machine": flatten(profiles.machine, root),
        "process": flatten(profiles.process, root),
        "filament": flatten(profiles.filament, root),
    }
    for key, value in (extra_settings or {}).items():
        target = "process"
        for candidate in ("machine", "filament", "process"):
            if key in merged[candidate]:
                target = candidate
                break
        merged[target][key] = value
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, data in merged.items():
        path = out_dir / "{}.json".format(kind)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1, sort_keys=True)
        paths[kind] = path
    return paths


__all__ = [
    "ProfileError",
    "Profiles",
    "clear_cache",
    "flatten",
    "list_materials",
    "list_printers",
    "list_qualities",
    "profiles_root",
    "resolve",
    "write_flattened",
]
