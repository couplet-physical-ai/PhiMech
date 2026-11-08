#!/usr/bin/env python3
"""Register PhiMech's stdio MCP server in a client configuration.

The stdio client deliberately sanitises the child environment.  Consequently the repository
and CAD binary paths must travel in this entry's ``env`` block; shell exports alone are not
reliable.  This module is stdlib-only so it can run with macOS's Python before the PhiMech
venv exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_CLIENT = Path.home() / ".pi" / "agent" / "mcp.json"
_COMMENT = [
    "PhiMech: component-driven enclosure synthesis over OpenSCAD + FreeCAD.",
    "Written by phimech/mcp/register.py; the PHIMECH_* env block is REQUIRED "
    "(stdio client sanitises env).",
]


def _repo_default() -> Path:
    """Derive the checkout root from this file, independent of the current directory."""
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else fallback


def _resolve_binary(
    env_name: str, names: Tuple[str, ...], fallbacks: Tuple[str, ...]
) -> Optional[Path]:
    """Resolve an optional CAD binary, omitting it when the machine does not have it."""
    configured = os.environ.get(env_name, "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            found = shutil.which(configured)
            candidate = Path(found) if found else candidate
        if candidate.is_file():
            return candidate.resolve()
        return None
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    for fallback in fallbacks:
        candidate = Path(fallback)
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_defaults() -> Dict[str, Any]:
    """Return registration paths with the same overrides used by the server and backends."""
    repo = _env_path("PHIMECH_REPO", _repo_default()).resolve()
    venv = _env_path("PHIMECH_VENV", repo / "build" / "venv").resolve()
    return {
        "repo": repo,
        "venv": venv,
        "python": venv / "bin" / "python",
        "server": repo / "phimech" / "mcp" / "server.py",
        "openscad": _resolve_binary(
            "PHIMECH_OPENSCAD",
            ("openscad",),
            ("/opt/homebrew/bin/openscad", "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"),
        ),
        "freecadcmd": _resolve_binary(
            "PHIMECH_FREECADCMD",
            ("freecadcmd",),
            ("/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",),
        ),
    }


#: Config keys whose values are credentials. `--dry-run` prints the *whole* client config so a
#: caller can see their other servers were preserved, and those other servers are not ours --
#: on a real machine they carry OAuth bearer tokens. Printing them to stdout puts them in shell
#: history, CI logs, screen shares and agent transcripts, which is a disclosure this command has
#: no reason to cause. The names are matched, not the values, so a rotated or unusual token
#: shape cannot slip past.
SECRET_KEY_RE = re.compile(
    r"authorization|bearer|token|secret|password|passwd|api[-_]?key|credential|cookie",
    re.IGNORECASE,
)

REDACTED = "<redacted by --dry-run>"


def redact_secrets(value: Any) -> Any:
    """Copy ``value`` with every credential-named field replaced.

    Structure, key order and every non-secret value are preserved, because the point of the
    dry-run is to show the caller exactly what the file will look like -- including that their
    other servers survived. Only the secret *values* are withheld.
    """
    if isinstance(value, dict):
        return {k: (REDACTED if SECRET_KEY_RE.search(str(k)) else redact_secrets(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value


def build_entry(
    repo: Path,
    python: Path,
    server: Path,
    openscad: Optional[Path] = None,
    freecadcmd: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the exact absolute-path entry consumed by an MCP stdio client."""
    environment: Dict[str, str] = {
        "PHIMECH_REPO": str(repo),
        "QT_QPA_PLATFORM": "offscreen",
    }
    if openscad is not None:
        environment["PHIMECH_OPENSCAD"] = str(openscad)
    if freecadcmd is not None:
        environment["PHIMECH_FREECADCMD"] = str(freecadcmd)
    return {
        "_comment": list(_COMMENT),
        "type": "stdio",
        "command": str(python),
        "args": [str(server)],
        "env": environment,
    }


def load_config(path: Path) -> Dict[str, Any]:
    """Load an object config, refusing to destroy malformed client credentials."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError("cannot read %s: %s" % (path, exc)) from exc
    if not text.strip():
        return {}
    try:
        config = json.loads(text)
    except ValueError as exc:
        raise ValueError("%s is not valid JSON; refusing to overwrite it" % path) from exc
    if not isinstance(config, dict):
        raise ValueError("%s must contain a JSON object" % path)
    return config


def place_entry(config: Dict[str, Any], entry: Dict[str, Any]) -> str:
    """Change only ``phimech`` while preserving flat versus ``mcpServers`` layout."""
    servers = config.get("mcpServers")
    if isinstance(servers, dict):
        servers["phimech"] = entry
        return "mcpServers.phimech"
    if "mcpServers" in config and servers is not None:
        raise ValueError("mcpServers must be a JSON object to preserve the existing layout")
    config["phimech"] = entry
    return "phimech"


def backup(path: Path) -> Path:
    """Create a mode-600 timestamped backup before replacing an existing config."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(path.name + ".bak-" + stamp)
    suffix = 1
    while destination.exists():
        destination = path.with_name(path.name + ".bak-%s-%d" % (stamp, suffix))
        suffix += 1
    shutil.copy2(str(path), str(destination))
    os.chmod(str(destination), stat.S_IRUSR | stat.S_IWUSR)
    return destination


def write_config(path: Path, config: Dict[str, Any]) -> None:
    """Atomically replace the client config and force credentials-safe mode 600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(config, target, indent=2, ensure_ascii=False)
            target.write("\n")
        os.chmod(str(temporary), stat.S_IRUSR | stat.S_IWUSR)
        os.replace(str(temporary), str(path))
        os.chmod(str(path), stat.S_IRUSR | stat.S_IWUSR)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _missing_paths(entry: Dict[str, Any]) -> list:
    """Return required entry paths that are absent; optional CAD paths are omitted upstream."""
    paths = [
        ("venv python", Path(str(entry["command"]))),
        ("server.py", Path(str(entry["args"][0]))),
    ]
    return ["%s missing: %s" % (label, path) for label, path in paths if not path.exists()]


def main(argv: Optional[list] = None) -> int:
    """Parse arguments, print an entry for dry-run, or update one client config."""
    defaults = resolve_defaults()
    parser = argparse.ArgumentParser(
        description="Register/update only the PhiMech MCP entry in a client config."
    )
    parser.add_argument("--repo", default=str(defaults["repo"]), help="PhiMech checkout root")
    parser.add_argument(
        "--venv",
        default=str(defaults["venv"]),
        help="PhiMech venv root; its bin/python is the stdio command",
    )
    parser.add_argument(
        "--openscad",
        default=None,
        help="OpenSCAD binary override (default: PHIMECH_OPENSCAD/PATH/known path)",
    )
    parser.add_argument(
        "--freecadcmd",
        default=None,
        help="FreeCAD command override (default: PHIMECH_FREECADCMD/PATH/known path)",
    )
    parser.add_argument(
        "--client",
        "--config",
        dest="client",
        default=str(DEFAULT_CLIENT),
        help="MCP client JSON path (default: ~/.pi/agent/mcp.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the entry without writing")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="write despite a missing venv python or server.py (use setup afterward)",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    venv = Path(args.venv).expanduser().resolve()
    python = venv / "bin" / "python"
    server = repo / "phimech" / "mcp" / "server.py"

    if args.openscad:
        openscad = Path(args.openscad).expanduser()
        if not openscad.is_absolute():
            found = shutil.which(args.openscad)
            openscad = Path(found) if found else openscad
        openscad = openscad.resolve() if openscad.is_file() else None
    else:
        openscad = _resolve_binary(
            "PHIMECH_OPENSCAD",
            ("openscad",),
            ("/opt/homebrew/bin/openscad", "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"),
        )
    if args.freecadcmd:
        freecadcmd = Path(args.freecadcmd).expanduser()
        if not freecadcmd.is_absolute():
            found = shutil.which(args.freecadcmd)
            freecadcmd = Path(found) if found else freecadcmd
        freecadcmd = freecadcmd.resolve() if freecadcmd.is_file() else None
    else:
        freecadcmd = _resolve_binary(
            "PHIMECH_FREECADCMD",
            ("freecadcmd",),
            ("/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",),
        )

    entry = build_entry(repo, python, server, openscad, freecadcmd)
    missing = _missing_paths(entry)
    client = Path(args.client).expanduser().resolve()

    if args.dry_run:
        # Print the prospective full config so a caller can verify both layout and preservation,
        # with other servers' credentials redacted -- see SECRET_KEY_RE.
        try:
            preview = load_config(client)
            place_entry(preview, entry)
        except (OSError, ValueError) as exc:
            print("error: %s" % exc, file=sys.stderr)
            return 2
        print(json.dumps(redact_secrets(preview), indent=2, ensure_ascii=False))
        for problem in missing:
            print("warning: %s (dry-run; no file was written)" % problem, file=sys.stderr)
        return 0

    if missing and not args.allow_missing:
        for problem in missing:
            print("error: %s" % problem, file=sys.stderr)
        print(
            "error: run `bin/phimech setup` to create the venv, or pass --allow-missing",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_config(client)
        if client.exists():
            saved = backup(client)
            print("backup: %s" % saved)
        where = place_entry(config, entry)
        write_config(client, config)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    if missing:
        for problem in missing:
            print("warning: %s (--allow-missing)" % problem, file=sys.stderr)
    print("wrote %s (%s), mode 600" % (client, where))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
