"""Where a headless Chrome is found, for the drawing's PNG and PDF copies.

Kept out of ``phimech/synth/drawing.py`` on purpose: that module's source is forbidden from
touching ``os.environ`` at all (``tests/test_drawing.py`` checks it), because an approver's
name must never be able to arrive from the machine's environment. A rasteriser's path is the
one environment fact the drawing legitimately needs, so it is read here and only here.

``PHIMECH_CHROME`` names the binary when it is on no PATH (a Playwright-cached
``chrome-headless-shell``, say); otherwise the candidates below are probed in order. A value
that is not a file yields ``None`` - a converter is never invented.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

#: Headless-Chrome-compatible binaries, probed in order when ``PHIMECH_CHROME`` is unset.
CHROME_CANDIDATES: Tuple[str, ...] = (
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome",
    "chrome-headless-shell", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_chrome() -> Optional[str]:
    """Absolute path of a headless-capable Chrome, or None."""
    override = os.environ.get("PHIMECH_CHROME")
    if override:
        return override if Path(override).is_file() else None
    for name in CHROME_CANDIDATES:
        binary = shutil.which(name) or (name if Path(name).is_file() else None)
        if binary:
            return binary
    return None


__all__ = ["CHROME_CANDIDATES", "find_chrome"]
