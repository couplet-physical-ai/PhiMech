"""Slicing and printer transport: STL -> G-code -> the printer's file system.

The layer is deliberately thin and read-mostly.  :mod:`phimech.print.profiles` resolves friendly
names to the slicer's own vendor profiles, :mod:`phimech.print.slicer` drives OrcaSlicer as a
subprocess, and :mod:`phimech.print.printer` talks to Moonraker over stdlib ``urllib``.

PhiMech never starts a print.  See the safety contract in :mod:`phimech.print.printer`.

Submodules are imported lazily so that ``import phimech.print`` stays free for callers that only
need one of them (and so a broken OrcaSlicer install cannot break an unrelated import).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phimech.print import printer, profiles, slicer  # noqa: F401


__all__ = ["printer", "profiles", "slicer"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module("phimech.print.{}".format(name))
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
