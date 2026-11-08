"""PhiMech package version helpers.

The source checkout is intentionally usable without installation, so version information
lives here instead of relying on package metadata.  ``git_describe`` is best effort because
installed wheels and source archives do not necessarily contain a Git checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


__version__ = '0.1.0'


def git_describe() -> Optional[str]:
    """Return ``git describe`` output, or ``None`` when this is not a Git checkout.

    The command is deliberately quiet and bounded: version display must never make the CLI
    fail just because Git is absent, the checkout is shallow, or the repository is unreadable.
    """
    repository = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


__all__ = ["__version__", "git_describe"]
