"""Stdlib tests for PhiMech MCP registration and the server tool inventory."""

from __future__ import annotations

import ast
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REGISTER = REPO / "phimech" / "mcp" / "register.py"
SERVER = REPO / "phimech" / "mcp" / "server.py"


def _run_register(*args: str) -> subprocess.CompletedProcess:
    """Invoke the stdlib registration script as a client would."""
    return subprocess.run(
        [sys.executable, str(REGISTER)] + list(args),
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )


def test_dry_run_preserves_wrapper_and_other_server() -> None:
    """Dry-run should be machine-readable and should not mutate the seeded config."""
    with tempfile.TemporaryDirectory(prefix="phimech-register-") as temporary:
        client = Path(temporary) / "mcp.json"
        original = {"mcpServers": {"other": {"command": "keep-me"}}}
        client.write_text(json.dumps(original), encoding="utf-8")
        result = _run_register("--dry-run", "--client", str(client))
        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert set(printed["mcpServers"]) == {"other", "phimech"}
        assert printed["mcpServers"]["phimech"]["type"] == "stdio"
        assert "PHIMECH_REPO" in printed["mcpServers"]["phimech"]["env"]
        assert json.loads(client.read_text(encoding="utf-8")) == original


def test_write_backup_mode_layout_and_idempotence() -> None:
    """A real write keeps the wrapper, creates a backup, and replaces one key on repeat."""
    with tempfile.TemporaryDirectory(prefix="phimech-register-") as temporary:
        root = Path(temporary)
        client = root / "mcp.json"
        client.write_text(json.dumps({"mcpServers": {"other": {}}}), encoding="utf-8")
        result = _run_register(
            "--client",
            str(client),
            "--repo",
            str(REPO),
            "--venv",
            str(root / "missing-venv"),
            "--allow-missing",
        )
        assert result.returncode == 0, result.stderr
        assert stat.S_IMODE(client.stat().st_mode) == 0o600
        assert list(root.glob("mcp.json.bak-*")), "expected timestamped backup"
        config = json.loads(client.read_text(encoding="utf-8"))
        assert set(config["mcpServers"]) == {"other", "phimech"}
        assert config["mcpServers"]["phimech"]["env"]["PHIMECH_REPO"] == str(REPO)

        second = _run_register(
            "--client",
            str(client),
            "--repo",
            str(REPO),
            "--venv",
            str(root / "missing-venv"),
            "--allow-missing",
        )
        assert second.returncode == 0, second.stderr
        config_again = json.loads(client.read_text(encoding="utf-8"))
        assert set(config_again["mcpServers"]) == {"other", "phimech"}
        assert len(list(root.glob("mcp.json.bak-*"))) >= 2


def test_flat_layout_without_wrapper() -> None:
    """A client config that is flat stays flat rather than being restructured."""
    with tempfile.TemporaryDirectory(prefix="phimech-register-") as temporary:
        root = Path(temporary)
        client = root / "flat.json"
        client.write_text(json.dumps({"other": {}}), encoding="utf-8")
        result = _run_register(
            "--client",
            str(client),
            "--repo",
            str(REPO),
            "--venv",
            str(root / "missing-venv"),
            "--allow-missing",
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(client.read_text(encoding="utf-8"))
        assert "mcpServers" not in config
        assert set(config) == {"other", "phimech"}


def test_server_decorated_tools_match_tools_constant() -> None:
    """The public MCP inventory stays synchronized without importing fastmcp."""
    tree = ast.parse(SERVER.read_text(encoding="utf-8"), filename=str(SERVER))
    decorated: List[str] = []
    tools_value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOLS":
                    tools_value = ast.literal_eval(node.value)
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "tool"
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "mcp"
            ):
                decorated.append(node.name)
    assert isinstance(tools_value, list)
    assert sorted(decorated) == sorted(tools_value)



def test_dry_run_redacts_other_servers_credentials() -> None:
    """`--dry-run` prints the whole config to prove other servers survive; those servers are
    not ours and on a real machine carry OAuth bearer tokens. Printing them would put a live
    credential into shell history, CI logs and agent transcripts."""
    from phimech.mcp.register import REDACTED, redact_secrets

    config = {
        "mcpServers": {
            "other": {
                "type": "http",
                "url": "https://example.invalid/mcp",
                "headers": {"Authorization": "Bearer eyJhbGciOi.SECRET.value"},
            },
            "stdio-one": {
                "command": "/bin/true",
                "env": {"SOME_API_KEY": "sk-live-000", "PATH": "/usr/bin"},
            },
            "phimech": {"command": "/bin/true", "args": ["server.py"]},
        }
    }
    out = redact_secrets(config)
    rendered = json.dumps(out)

    # the secrets are gone, by value not just by key
    assert "SECRET" not in rendered and "sk-live-000" not in rendered
    assert out["mcpServers"]["other"]["headers"]["Authorization"] == REDACTED
    assert out["mcpServers"]["stdio-one"]["env"]["SOME_API_KEY"] == REDACTED

    # ...and everything the dry-run exists to show is still there
    assert sorted(out["mcpServers"]) == ["other", "phimech", "stdio-one"]
    assert out["mcpServers"]["other"]["url"] == "https://example.invalid/mcp"
    assert out["mcpServers"]["stdio-one"]["env"]["PATH"] == "/usr/bin"
    assert out["mcpServers"]["phimech"]["args"] == ["server.py"]

    # the input is not mutated - a redacting helper that edited the config in place would make
    # the non-dry-run path write `<redacted>` into the user's real file
    assert config["mcpServers"]["other"]["headers"]["Authorization"].startswith("Bearer ")


def _run_all() -> None:
    """Permit the coordinator's system-Python command to run this file directly."""
    tests = [
        test_dry_run_preserves_wrapper_and_other_server,
        test_write_backup_mode_layout_and_idempotence,
        test_flat_layout_without_wrapper,
        test_server_decorated_tools_match_tools_constant,
        test_dry_run_redacts_other_servers_credentials,
    ]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)


if __name__ == "__main__":
    _run_all()
