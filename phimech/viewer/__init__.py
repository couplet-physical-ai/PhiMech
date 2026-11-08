"""Interactive 3D web viewer for PhiMech build output.

A stdlib-only ``http.server`` that serves a vendored three.js single-page app over a build
directory: the STLs render in the browser, the gates and measurements come from ``report.json``.
No third-party dependency and no build step, so it imports on python 3.9 and runs anywhere the
rest of PhiMech runs.

Usage::

    python -m phimech.viewer.server out/final --port 8765 --open

The re-exports below are resolved lazily (PEP 562) so that ``python -m phimech.viewer.server``
does not warn about the submodule already being in ``sys.modules``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_manifest", "discover_builds", "serve", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__() -> list:
    return sorted(__all__)
