#!/usr/bin/env python3
"""Run a real stdio MCP initialize/list/call handshake against PhiMech."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).resolve().with_name("server.py")


def _server_python() -> Optional[Path]:
    """Prefer the registered/build venv, falling back only when the current Python has fastmcp."""
    configured = os.environ.get("PHIMECH_VENV", "").strip()
    venv = Path(configured).expanduser() if configured else REPO / "build" / "venv"
    candidate = venv / "bin" / "python"
    if candidate.is_file():
        # Do not resolve the venv's interpreter symlink: its path is what activates the venv.
        return candidate.absolute()
    if importlib.util.find_spec("fastmcp") is not None:
        return Path(sys.executable).absolute()
    return None


def _instructions() -> None:
    print(
        "MCP handshake unavailable: install fastmcp==2.13.1 and mcp in build/venv "
        "(run bin/phimech setup, then use the project venv; do not install into system Python).",
        file=sys.stderr,
    )


async def _handshake(python: Path) -> int:
    """Initialize a real client session and invoke two representative tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env["PHIMECH_REPO"] = str(REPO)
    parameters = StdioServerParameters(
        command=str(python),
        args=[str(SERVER)],
        env=env,
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = [tool.name for tool in listed.tools]
            await session.call_tool("phimech_doctor", arguments={})
            await session.call_tool("ir_schema", arguments={})
            print("tool count: %d" % len(names))
            print("handshake: initialize/list_tools/phimech_doctor/ir_schema ok")
    return 0


def main() -> int:
    """Return 3 for a missing client dependency, 1 for a failed handshake, else 0."""
    try:
        # Import only for the availability check; the actual classes are imported in the
        # async function so a missing optional package remains a clean status-3 exit.
        if importlib.util.find_spec("mcp") is None:
            raise ImportError("mcp client package is not importable")
    except (ImportError, ModuleNotFoundError, ValueError):
        _instructions()
        return 3

    python = _server_python()
    if python is None:
        _instructions()
        return 3
    try:
        return asyncio.run(_handshake(python))
    except Exception as exc:
        print("handshake failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
