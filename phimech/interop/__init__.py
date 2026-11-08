"""Interop with the CAD tools PhiMech does not own — SolidWorks first.

Everything here is stdlib-only and file-based on purpose: the question "will this open
correctly in SolidWorks?" is answerable from the bytes we wrote, and answering it that way
means it can run as a gate on every build without a kernel or a licence.
"""

from __future__ import annotations

from typing import Any

from phimech.interop.step import Finding, StepFacts, inspect, readiness

__all__ = ["EditRisk", "Finding", "StepFacts", "assess", "inspect", "readiness"]


def __getattr__(name: str) -> Any:
    """Export `assess` / `EditRisk` lazily (PEP 562, python >= 3.7).

    Importing `edit_risk` eagerly here would make `python3 -m phimech.interop.edit_risk` warn
    that the module was already in `sys.modules`; resolving it on first use keeps that entry
    point quiet while `from phimech.interop import assess` still works.
    """
    if name in ("EditRisk", "assess"):
        from phimech.interop import edit_risk

        return getattr(edit_risk, name)
    raise AttributeError("module 'phimech.interop' has no attribute {!r}".format(name))
