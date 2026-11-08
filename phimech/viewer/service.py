"""Keep the PhiMech app running at one stable URL, across logouts and reboots.

Why this exists
---------------
`bin/phimech view` is a foreground server you stop with Ctrl-C, and its port is a *request* rather
than a promise -- if 8765 is busy it walks upward. Both are right for looking at one build. Neither
gives you a link you can bookmark, put in a text file, or send to someone, because tomorrow the
server is gone and the port may have moved.

This installs a macOS **LaunchAgent**: a per-user service launchd starts at login and restarts if it
dies. The port is pinned, so the URL never changes:

    http://127.0.0.1:8787/            the home page
    http://127.0.0.1:8787/docs        the documentation
    http://127.0.0.1:8787/viewer      the 3D workspace

Deliberate limits
-----------------
It binds **127.0.0.1**, so it is reachable only from this machine. That is not a hedge -- the server
reads files from a repository checkout and serves a `docs/` tree, and nothing in it was designed to
face a network. Do not expose this unauthenticated server on a network.

It serves the **repository**, not a copy. Editing the public docs is live without reinstall.

Only macOS is implemented. On Linux the equivalent is a systemd user unit; `install()` refuses
rather than pretending -- a service that silently did not install is worse than an error.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Reverse-DNS label, the macOS convention, and the name shown by `launchctl list`.
LABEL = "io.github.couplet-physical-ai.phimech.app"
#: Pinned so the URL is quotable. 8787 is outside the range `view` walks through by default.
DEFAULT_PORT = 8787
DEFAULT_HOST = "127.0.0.1"

AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = AGENTS_DIR / "{}.plist".format(LABEL)
LOG_DIR = Path.home() / "Library" / "Logs" / "PhiMech"


def repo_root() -> Path:
    """The checkout this module lives in."""
    return Path(__file__).resolve().parents[2]


def url(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> str:
    return "http://{}:{}".format(host, port)


def links(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> "List[Tuple[str, str, str]]":
    """``(title, url, blurb)`` for every page worth bookmarking."""
    base = url(port, host)
    return [
        ("Home", base + "/", "the local landing page"),
        ("Docs", base + "/docs", "the shipped public guide"),
        ("Workspace", base + "/viewer", "the 3D build viewer"),
    ]


def _plist(out_dir: Path, port: int, host: str) -> Dict[str, object]:
    launcher = repo_root() / "bin" / "phimech"
    return {
        "Label": LABEL,
        # bin/phimech is the launcher that strips the proxy env and sets QT_QPA_PLATFORM; going
        # through it rather than python -m keeps one definition of how PhiMech starts.
        "ProgramArguments": [
            str(launcher), "view", str(out_dir),
            "--port", str(port), "--host", host, "--no-open",
        ],
        "WorkingDirectory": str(repo_root()),
        "RunAtLoad": True,
        # KeepAlive without SuccessfulExit: restart it however it died, including a clean exit,
        # because a serving process exiting cleanly still means the URL stopped working.
        "KeepAlive": True,
        # A crash-loop should not spin: launchd's floor is 10s, and being explicit documents it.
        "ThrottleInterval": 10,
        "StandardOutPath": str(LOG_DIR / "app.log"),
        "StandardErrorPath": str(LOG_DIR / "app.err.log"),
        "ProcessType": "Background",
        "EnvironmentVariables": {"PHIMECH_SERVICE": "1"},
    }


def _launchctl(*args: str) -> "subprocess.CompletedProcess":
    return subprocess.run(["launchctl"] + list(args), capture_output=True, text=True)


def _uid() -> str:
    import os

    return str(os.getuid())


def install(out_dir: Optional[Path] = None, port: int = DEFAULT_PORT,
            host: str = DEFAULT_HOST) -> "Tuple[bool, List[str]]":
    """Write the plist and start the service. Returns ``(ok, log_lines)``."""
    notes: List[str] = []
    if sys.platform != "darwin":
        return False, ["only macOS LaunchAgents are implemented; on Linux use a systemd user unit"]
    if shutil.which("launchctl") is None:
        return False, ["launchctl not found -- cannot manage services on this machine"]

    target = Path(out_dir).expanduser().resolve() if out_dir else repo_root() / "out"
    if not target.is_dir():
        # Serving a directory that does not exist yet is fine -- the manifest degrades to warnings,
        # and the home page and docs do not need a build at all.
        notes.append("note: {} does not exist yet; the pages work, builds will appear when you "
                     "make one".format(target))

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with PLIST_PATH.open("wb") as handle:
        plistlib.dump(_plist(target, port, host), handle)
    notes.append("wrote {}".format(PLIST_PATH))

    # Replace any previous incarnation first; bootstrap fails outright on an already-loaded label.
    _launchctl("bootout", "gui/{}/{}".format(_uid(), LABEL))
    booted = _launchctl("bootstrap", "gui/{}".format(_uid()), str(PLIST_PATH))
    if booted.returncode != 0:
        # Fall back to the older verb: `bootstrap` is unavailable on some systems, and on a machine
        # whose launchd is unhealthy neither may work -- in which case login will still load it.
        legacy = _launchctl("load", "-w", str(PLIST_PATH))
        if legacy.returncode != 0:
            notes.append("could not start it now: {}".format(
                (booted.stderr or legacy.stderr or "launchctl refused").strip()))
            notes.append("the plist is installed, so it will start at your next login anyway")
            return False, notes
    notes.append("service {} started".format(LABEL))
    return True, notes


def uninstall() -> "Tuple[bool, List[str]]":
    notes: List[str] = []
    if sys.platform != "darwin":
        return False, ["nothing to do: no LaunchAgent is installed on this platform"]
    _launchctl("bootout", "gui/{}/{}".format(_uid(), LABEL))
    _launchctl("unload", "-w", str(PLIST_PATH))
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
        notes.append("removed {}".format(PLIST_PATH))
    else:
        notes.append("no plist at {} -- nothing to remove".format(PLIST_PATH))
    return True, notes


def status(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> Dict[str, object]:
    """Describe it: installed, loaded, and -- the only answer that matters -- responding."""
    import json
    import urllib.error
    import urllib.request

    info: Dict[str, object] = {
        "label": LABEL,
        "plist": str(PLIST_PATH),
        "installed": PLIST_PATH.is_file(),
        "url": url(port, host),
        "loaded": False,
        "responding": False,
        "out_dir": None,
    }
    if sys.platform == "darwin" and shutil.which("launchctl"):
        listed = _launchctl("list", LABEL)
        info["loaded"] = listed.returncode == 0

    # An HTTP 200 is the real test: launchd can happily report a loaded job that is crash-looping.
    try:
        request = urllib.request.Request(url(port, host) + "/api/health")
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        info["responding"] = True
        info["out_dir"] = payload.get("out_dir")
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return info


def write_link_folder(folder: Path, port: int = DEFAULT_PORT,
                      host: str = DEFAULT_HOST) -> "List[Path]":
    """Create a folder of clickable links plus a plain-text fallback.

    Write a native browser shortcut, HTML redirect and readable text links.
    """
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    entries = links(port, host)
    primary = entries[0][1]

    webloc = folder / "PhiMech home.webloc"
    with webloc.open("wb") as handle:
        plistlib.dump({"URL": primary}, handle)
    written.append(webloc)

    opener = folder / "Open PhiMech.html"
    opener.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\" />\n"
        "<meta http-equiv=\"refresh\" content=\"0; url={url}\" />\n"
        "<title>Opening PhiMech\u2026</title></head>\n<body style=\"font-family:-apple-system,"
        "sans-serif;padding:40px\">\n<p>Opening <a href=\"{url}\">{url}</a>\u2026</p>\n"
        "<p>If nothing happens the service is not running. Start it with:<br />\n"
        "<code>bin/phimech service install</code></p>\n</body></html>\n".format(url=primary))
    written.append(opener)

    lines = [
        "PhiMech \u2014 permanent links",
        "=" * 30,
        "",
        "These URLs do not change. The service starts itself at login and restarts if it dies,",
        "so you can bookmark them or paste them to yourself.",
        "",
    ]
    width = max(len(title) for title, _, _ in entries)
    for title, link, blurb in entries:
        lines.append("  {}  {}".format(title.ljust(width), link))
        lines.append("  {}  {}".format(" " * width, blurb))
        lines.append("")
    # Every command is prefixed with the checkout path: a reader of this file has no reason to
    # know where the repository lives, and a command they cannot paste is not instructions.
    root = repo_root()
    lines += [
        "Reachable from this Mac only (127.0.0.1), on purpose: the server reads a repository",
        "checkout and is not intended for network exposure.",,
        "",
        "If a link does not load",
        "-" * 23,
        "  cd {}".format(root),
        "  bin/phimech service status      is it installed, loaded, and answering?",
        "  bin/phimech service install     (re)install and start it",
        "  bin/phimech service uninstall   remove it",
        "",
        "Logs: ~/Library/Logs/PhiMech/app.log",
        "",
    ]
    readme = folder / "LINKS.txt"
    readme.write_text("\n".join(lines))
    written.append(readme)

    return written
