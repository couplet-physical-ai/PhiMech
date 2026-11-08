"""OrcaSlicer CLI wrapper: STL -> G-code, with real time/filament numbers.

OrcaSlicer is a subprocess, never an import — same rule as the CAD backends.  The invocation is
an optional locally configured slicer invocation::

    QT_QPA_PLATFORM=offscreen <orca> --load-settings "<machine.json>;<process.json>" \\
        --load-filaments "<filament.json>" --slice 0 --export-3mf <name>.gcode.3mf \\
        --outputdir <dir> <in.stl>

Three traps this module exists to close, all measured on a 20 mm cube:

1. **Grams read 0.00.**  ``; total filament used [g]`` was ``0.00`` because the CLI does not
   resolve a profile's ``inherits`` chain, so ``filament_density`` stayed ``0`` and the slice ran
   at PLA's 200/45 °C.  :func:`phimech.print.profiles.write_flattened` flattens the chain first;
   the same cube then reports PETG 240/70 °C, density 1.24, **3.40 g**.  Grams are still read
   from ``Metadata/slice_info.config`` inside the 3MF first (it is a zip), then the G-code header,
   then a length x cross-section x density estimate — and an estimate always sets a warning.
   Zero grams is never reported silently.
2. **A relative ``--outputdir`` fails the 3MF export** while still exiting 0 ("Project export to
   out/w2/cube.gcode.3mf failed").  Every path handed to the CLI is absolute.
3. **Exit 0 does not mean success.**  Success is the ``.gcode.3mf`` existing on disk.
"""

from __future__ import annotations

import json
import math
import os
import plistlib
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from phimech.print import profiles as _profiles
from phimech.print import thumbnails as _thumbnails
from phimech.print.profiles import ProfileError, Profiles


_ORCA_FALLBACK = "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"
_ORCA_NAMES = ("orcaslicer", "orca-slicer", "OrcaSlicer")

# Header keys of interest. Values arrive as `; key = value` or `; key: value`.
_HEADER_KEYS = (
    "total layer number",
    "total layers count",
    "estimated printing time (normal mode)",
    "filament used [mm]",
    "filament used [cm3]",
    "total filament used [g]",
    "filament_type",
    "filament_density",
    "filament_diameter",
    "nozzle_temperature",
    "hot_plate_temp",
    "chamber_temperature",
    "layer_height",
    "max_z_height",
    "printer_settings_id",
    "print_settings_id",
    "filament_settings_id",
)

# The header block sits at the top; the summary + CONFIG_BLOCK sit at the very end (~26 KB in the
# verified run). Reading only the two ends keeps a 500 MB G-code file cheap to parse.
_HEAD_BYTES = 64 * 1024
_TAIL_BYTES = 512 * 1024

_AVAILABLE: Optional[bool] = None
_VERSION_CACHE: Optional[str] = None
_VERSION_DONE = False


class SliceError(RuntimeError):
    """Raised when slicing cannot be performed or produced no G-code."""


def find_orcaslicer() -> Optional[Path]:
    """Locate the OrcaSlicer binary: ``$PHIMECH_ORCASLICER``, the app bundle, then ``PATH``."""
    configured = os.environ.get("PHIMECH_ORCASLICER")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
        found = shutil.which(configured)
        if found:
            return Path(found)
    fallback = Path(_ORCA_FALLBACK)
    if fallback.is_file() and os.access(str(fallback), os.X_OK):
        return fallback
    for name in _ORCA_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def available() -> bool:
    """True when an OrcaSlicer binary can be found."""
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = find_orcaslicer() is not None
    return _AVAILABLE


def version() -> Optional[str]:
    """OrcaSlicer's version string, cached.

    Read from the app bundle's ``Info.plist`` when possible: OrcaSlicer has no ``--version`` flag
    (it answers "Invalid option --version"), and launching it costs ~1 s.
    """
    global _VERSION_CACHE, _VERSION_DONE
    if _VERSION_DONE:
        return _VERSION_CACHE
    _VERSION_DONE = True
    binary = find_orcaslicer()
    if binary is None:
        return None
    plist = binary.parent.parent / "Info.plist"
    if plist.is_file():
        try:
            with plist.open("rb") as handle:
                data = plistlib.load(handle)
            text = data.get("CFBundleShortVersionString") or data.get("CFBundleVersion")
            if text:
                _VERSION_CACHE = str(text).strip()
                return _VERSION_CACHE
        except (OSError, ValueError):
            pass
    # Fall back to the banner the CLI prints when it rejects an unknown flag.
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_slicer_env(),
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"OrcaSlicer[- ]v?([0-9][0-9.]*)", completed.stdout.decode(errors="replace"))
    if match:
        _VERSION_CACHE = match.group(1)
    return _VERSION_CACHE


def reset_cache() -> None:
    """Forget the cached binary/version lookups. For tests that move the binary."""
    global _AVAILABLE, _VERSION_CACHE, _VERSION_DONE
    _AVAILABLE = None
    _VERSION_CACHE = None
    _VERSION_DONE = False


def _slicer_env() -> Dict[str, str]:
    """Environment for the CLI: headless Qt, and no proxy in the way of its startup checks."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(name, None)
    return env


@dataclass
class SliceResult:
    """Everything PhiMech knows about one sliced plate."""

    gcode_3mf: Path
    gcode: Optional[Path] = None
    print_time_s: Optional[int] = None
    filament_g: Optional[float] = None
    filament_mm: Optional[float] = None
    layer_count: Optional[int] = None
    layer_height: Optional[float] = None
    nozzle_temp_c: Optional[int] = None
    bed_temp_c: Optional[int] = None
    filament_type: Optional[str] = None
    machine: Optional[str] = None
    process: Optional[str] = None
    filament: Optional[str] = None
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def print_time_text(self) -> Optional[str]:
        """``1688`` -> ``28m 8s``."""
        return None if self.print_time_s is None else format_duration(self.print_time_s)

    def to_dict(self) -> Dict[str, object]:
        return {
            "gcode_3mf": str(self.gcode_3mf),
            "gcode": None if self.gcode is None else str(self.gcode),
            "print_time_s": self.print_time_s,
            "print_time": self.print_time_text,
            "filament_g": self.filament_g,
            "filament_mm": self.filament_mm,
            "layer_count": self.layer_count,
            "layer_height": self.layer_height,
            "nozzle_temp_c": self.nozzle_temp_c,
            "bed_temp_c": self.bed_temp_c,
            "filament_type": self.filament_type,
            "machine": self.machine,
            "process": self.process,
            "filament": self.filament,
            "seconds": round(self.seconds, 3),
            "warnings": list(self.warnings),
        }


_DURATION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(d|h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(text: str) -> Optional[int]:
    """Parse OrcaSlicer's duration strings into seconds.

    Handles ``18m 18s``, ``1h 2m 3s``, ``2d 1h``, ``28m8s``, ``45s``, and a bare ``1688``.
    """
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    total = 0.0
    matched = False
    for value, unit in _DURATION_RE.findall(cleaned):
        try:
            total += float(value) * _UNIT_SECONDS[unit.lower()]
        except (ValueError, KeyError):
            continue
        matched = True
    if matched:
        return int(round(total))
    try:  # a bare number is seconds
        return int(round(float(cleaned)))
    except ValueError:
        return None


def format_duration(seconds: int) -> str:
    """Inverse of :func:`parse_duration`, in OrcaSlicer's own style."""
    seconds = int(max(0, seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if days:
        parts.append("{}d".format(days))
    if hours or days:
        parts.append("{}h".format(hours))
    if minutes or hours or days:
        parts.append("{}m".format(minutes))
    parts.append("{}s".format(secs))
    return " ".join(parts)


def _float(text: object) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _int(text: object) -> Optional[int]:
    value = _float(text)
    return None if value is None else int(round(value))


def _parse_header_text(text: str) -> Dict[str, str]:
    """Collect the `; key = value` / `; key: value` comment pairs we care about."""
    found: Dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(";"):
            continue
        body = line[1:].strip()
        for separator in (" = ", ": ", "="):
            if separator not in body:
                continue
            key, _, value = body.partition(separator)
            key = key.strip()
            if key in _HEADER_KEYS and key not in found:
                found[key] = value.strip().strip('"')
            break
    return found


def parse_gcode_metadata(path: Path) -> Dict[str, object]:
    """Read the metadata OrcaSlicer writes into a G-code file's comment blocks.

    Only the first 64 KB and last 512 KB are read: the header block is at the top and the
    summary + ``CONFIG_BLOCK`` at the bottom (~26 KB in the verified run), so this stays cheap
    for large plates.  Returns normalized types; missing keys are simply absent.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(min(size, _HEAD_BYTES))
        if size > _HEAD_BYTES:
            handle.seek(max(_HEAD_BYTES, size - _TAIL_BYTES))
            tail = handle.read()
        else:
            tail = b""
    raw = _parse_header_text(head.decode("utf-8", "replace"))
    for key, value in _parse_header_text(tail.decode("utf-8", "replace")).items():
        raw.setdefault(key, value)

    out: Dict[str, object] = {}
    layers = raw.get("total layer number") or raw.get("total layers count")
    if layers is not None:
        value = _int(layers)
        if value is not None:
            out["layer_count"] = value
    duration = raw.get("estimated printing time (normal mode)")
    if duration is not None:
        value = parse_duration(duration)
        if value is not None:
            out["print_time_s"] = value
    for source, target in (
        ("filament used [mm]", "filament_mm"),
        ("filament used [cm3]", "filament_cm3"),
        ("total filament used [g]", "filament_g"),
        ("filament_density", "filament_density"),
        ("filament_diameter", "filament_diameter"),
        ("layer_height", "layer_height"),
        ("max_z_height", "max_z_height"),
    ):
        if source in raw:
            value = _float(raw[source])
            if value is not None:
                out[target] = value
    for source, target in (
        ("nozzle_temperature", "nozzle_temp_c"),
        ("hot_plate_temp", "bed_temp_c"),
        ("chamber_temperature", "chamber_temp_c"),
    ):
        if source in raw:
            value = _int(raw[source])
            if value is not None:
                out[target] = value
    for source, target in (
        ("filament_type", "filament_type"),
        ("printer_settings_id", "machine"),
        ("print_settings_id", "process"),
        ("filament_settings_id", "filament"),
    ):
        if source in raw:
            text = raw[source].strip()
            if text:
                out[target] = text
    return out


_SLICE_INFO_FILAMENT = re.compile(r"<filament\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_3mf_metadata(path: Path) -> Dict[str, object]:
    """Read ``Metadata/slice_info.config`` (and ``plate_1.json``) from inside a ``.gcode.3mf``.

    A ``.gcode.3mf`` is a zip.  ``slice_info.config`` carries ``used_g`` / ``used_m`` per filament
    and a ``prediction`` (seconds) per plate — the authoritative grams source, because the G-code
    header's ``total filament used [g]`` is 0.00 whenever the density did not resolve.
    """
    out: Dict[str, object] = {}
    try:
        with zipfile.ZipFile(str(path)) as archive:
            names = set(archive.namelist())
            if "Metadata/slice_info.config" in names:
                text = archive.read("Metadata/slice_info.config").decode("utf-8", "replace")
                grams = 0.0
                metres = 0.0
                types: List[str] = []
                for tag in _SLICE_INFO_FILAMENT.findall(text):
                    attrs = dict(_ATTR.findall(tag))
                    grams += _float(attrs.get("used_g")) or 0.0
                    metres += _float(attrs.get("used_m")) or 0.0
                    kind = (attrs.get("type") or "").strip()
                    if kind and kind not in types:
                        types.append(kind)
                if grams > 0:
                    out["filament_g"] = round(grams, 3)
                if metres > 0:
                    out["filament_mm"] = round(metres * 1000.0, 3)
                if types:
                    out["filament_type"] = "+".join(types)
                for key, target in (("prediction", "print_time_s"), ("weight", "plate_weight_g")):
                    match = re.search(
                        r'<metadata\s+key="{}"\s+value="([^"]*)"'.format(key), text
                    )
                    if match:
                        value = _float(match.group(1))
                        if value:
                            out[target] = int(round(value)) if target.endswith("_s") else value
            if "Metadata/plate_1.json" in names:
                try:
                    plate = json.loads(
                        archive.read("Metadata/plate_1.json").decode("utf-8", "replace")
                    )
                except ValueError:
                    plate = {}
                if isinstance(plate, dict):
                    objects = plate.get("bbox_objects")
                    if isinstance(objects, list) and objects:
                        height = _float(objects[0].get("layer_height"))
                        if height:
                            out.setdefault("layer_height", round(height, 4))
    except (OSError, zipfile.BadZipFile, KeyError):
        return out
    return out


def estimate_grams(filament_mm: float, diameter_mm: float, density_g_cm3: float) -> float:
    """length x cross-section x density, in grams. ``mm3 / 1000 = cm3``."""
    area_mm2 = math.pi * (diameter_mm / 2.0) ** 2
    volume_cm3 = filament_mm * area_mm2 / 1000.0
    return round(volume_cm3 * density_g_cm3, 3)


def slice_stl(
    stl_path,
    out_dir,
    *,
    material: str = "PETG",
    quality: str = "standard",
    printer: str = "qidi-max4",
    nozzle: str = "0.4",
    extra_settings: Optional[Dict[str, object]] = None,
    name: Optional[str] = None,
    timeout: int = 600,
) -> SliceResult:
    """Slice ``stl_path`` for ``printer`` and return the measured result.

    Writes ``<name>.gcode.3mf``, a per-part ``<name>.gcode`` (OrcaSlicer's ``plate_1.gcode``
    renamed) and ``<name>.slicer.log`` into ``out_dir``, and the
    flattened presets actually used into ``out_dir/profiles/`` so a slice is reproducible.
    """
    binary = find_orcaslicer()
    if binary is None:
        raise SliceError(
            "OrcaSlicer not found. Looked at $PHIMECH_ORCASLICER, {}, then PATH. "
            "Install with: brew install --cask orcaslicer".format(_ORCA_FALLBACK)
        )
    source = Path(stl_path).expanduser().resolve()
    if not source.is_file():
        raise SliceError("input model not found: {}".format(source))
    # Absolute: a relative --outputdir makes the 3MF export fail while still exiting 0.
    directory = Path(out_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    resolved = _profiles.resolve(
        printer=printer, material=material, quality=quality, nozzle=nozzle
    )
    flat = _profiles.write_flattened(
        resolved, directory / "profiles", extra_settings=extra_settings
    )

    stem = name or source.stem
    if stem.endswith(".gcode"):
        stem = stem[: -len(".gcode")]
    target = directory / "{}.gcode.3mf".format(stem)
    gcode_target = directory / "{}.gcode".format(stem)
    log_path = directory / "{}.slicer.log".format(stem)
    for stale in (target, gcode_target):
        if stale.exists():
            stale.unlink()

    # OrcaSlicer's side outputs are NOT per-model: it always writes ``plate_1.gcode`` into
    # --outputdir and ``00000.log`` into the CWD.  Sharing one directory across parts made each
    # slice overwrite the last and left both files in the build root.  Slice in a private scratch
    # directory, then move out only the per-part files.
    with tempfile.TemporaryDirectory(prefix="phimech-slice-") as scratch_name:
        scratch = Path(scratch_name).resolve()
        command = [
            str(binary),
            "--load-settings",
            "{};{}".format(flat["machine"], flat["process"]),
            "--load-filaments",
            str(flat["filament"]),
            "--slice",
            "0",
            "--export-3mf",
            target.name,
            "--outputdir",
            str(scratch),
            str(source),
        ]
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_slicer_env(),
                # Headless, OrcaSlicer cannot make a GL context for the plate thumbnail and drops
                # a `00000.log` in the CWD. The scratch directory keeps that out of the build.
                cwd=str(scratch),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SliceError(
                "OrcaSlicer timed out after {} s slicing {}".format(timeout, source.name)
            ) from exc
        elapsed = time.time() - started
        log = completed.stdout.decode("utf-8", "replace") if completed.stdout else ""
        # Headless OrcaSlicer logs to ``00000.log`` rather than stdout; fold those messages in so
        # the per-part log is the whole story.
        for side_log in sorted(scratch.glob("*.log")):
            text = side_log.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                log = "{}{}[{}]\n{}\n".format(
                    log, "" if not log or log.endswith("\n") else "\n", side_log.name, text
                )
        log_path.write_text(log, encoding="utf-8")

        scratch_3mf = scratch / target.name
        scratch_gcode = scratch / "plate_1.gcode"
        # Exit 0 is not success: the CLI exits 0 after "Project export ... failed".
        if not scratch_3mf.is_file():
            tail = "\n".join([line for line in log.splitlines() if line.strip()][-12:])
            raise SliceError(
                "OrcaSlicer produced no {} (exit {}). Last output:\n{}".format(
                    target.name, completed.returncode, tail or "<empty>"
                )
            )
        shutil.move(str(scratch_3mf), str(target))
        if scratch_gcode.is_file():
            shutil.move(str(scratch_gcode), str(gcode_target))

    warnings: List[str] = []
    for line in log.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if "warning" in lowered or "error" in lowered or "failed" in lowered:
            if stripped not in warnings:
                warnings.append(stripped)
    if completed.returncode != 0:
        warnings.append("orcaslicer exit code {}".format(completed.returncode))

    gcode_path: Optional[Path] = gcode_target if gcode_target.is_file() else None

    meta: Dict[str, object] = {}
    if gcode_path is not None:
        meta.update(parse_gcode_metadata(gcode_path))
    archive_meta = parse_3mf_metadata(target)

    # Grams: 3MF slice_info first, then the header, then an estimate. Never a silent zero.
    grams = _float(archive_meta.get("filament_g"))
    if not grams:
        grams = _float(meta.get("filament_g")) or None
    filament_mm = _float(meta.get("filament_mm")) or _float(archive_meta.get("filament_mm"))
    if not grams:
        density = _float(meta.get("filament_density")) or resolved.filament_density
        diameter = (
            _float(meta.get("filament_diameter")) or resolved.filament_diameter_mm or 1.75
        )
        if filament_mm and density:
            grams = estimate_grams(filament_mm, diameter, density)
            warnings.append(
                "filament mass is an ESTIMATE ({:.3f} g from {:.1f} mm x {:.2f} mm dia x "
                "{:.3f} g/cm3): neither the 3MF slice_info nor the G-code header reported a "
                "non-zero mass".format(grams, filament_mm, diameter, density)
            )
        else:
            warnings.append(
                "filament mass unknown: no mass in the 3MF or G-code header and no length to "
                "estimate from"
            )
            grams = None

    print_time = meta.get("print_time_s")
    if print_time is None:
        print_time = archive_meta.get("print_time_s")

    result = SliceResult(
        gcode_3mf=target,
        gcode=gcode_path,
        print_time_s=None if print_time is None else int(print_time),
        filament_g=grams,
        filament_mm=filament_mm,
        layer_count=_int(meta.get("layer_count")),
        layer_height=_float(meta.get("layer_height"))
        or _float(archive_meta.get("layer_height"))
        or resolved.layer_height_mm,
        nozzle_temp_c=_int(meta.get("nozzle_temp_c")) or resolved.nozzle_temp_c,
        bed_temp_c=_int(meta.get("bed_temp_c")) or resolved.bed_temp_c,
        filament_type=(
            meta.get("filament_type")
            or archive_meta.get("filament_type")
            or resolved.filament_type
        ),
        machine=resolved.machine,
        process=resolved.process,
        filament=resolved.filament,
        seconds=elapsed,
        warnings=warnings,
    )
    if result.layer_count is None or result.layer_count <= 0:
        result.warnings.append("layer count missing from the G-code header")
    # Headless OrcaSlicer has no GL context, so it writes no plate thumbnail: render one with
    # OpenSCAD and inject it into the G-code header and the 3MF (phimech/print/thumbnails.py).
    result.warnings.extend(
        _thumbnails.add_to_slice(source, target, result.gcode, flat.get("machine"))
    )
    return result


__all__ = [
    # Re-exported so a caller can catch everything slice_stl raises from one import.
    "ProfileError",
    "Profiles",
    "SliceError",
    "SliceResult",
    "available",
    "estimate_grams",
    "find_orcaslicer",
    "format_duration",
    "parse_3mf_metadata",
    "parse_duration",
    "parse_gcode_metadata",
    "reset_cache",
    "slice_stl",
    "version",
]
