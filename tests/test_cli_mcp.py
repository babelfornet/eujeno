import json, os, sys
import pytest
from typer.testing import CliRunner
from synapse.cli import app

runner = CliRunner()
_ECHO = os.path.join(os.path.dirname(__file__), "_mcp_echo_server.py")


def test_mcp_add_list_remove(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))
    r = runner.invoke(app, ["--json", "mcp", "--add", "x", "--command", "echo", "--args", "hello world"])
    assert r.exit_code == 0
    assert "x" in json.loads(r.stdout)["data"]["servers"]
    r = runner.invoke(app, ["--json", "mcp"])   # default = list
    assert "x" in json.loads(r.stdout)["data"]["servers"]
    r = runner.invoke(app, ["--json", "mcp", "--remove", "x"])
    assert json.loads(r.stdout)["data"]["servers"] == []


def test_mcp_add_requires_command(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))
    r = runner.invoke(app, ["mcp", "--add", "x"])
    assert r.exit_code == 2


@pytest.mark.slow
def test_mcp_list_discovers_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))
    runner.invoke(app, ["mcp", "--add", "echo", "--command", sys.executable, "--args", _ECHO])
    r = runner.invoke(app, ["--json", "mcp"])
    tools = [t["name"] for t in json.loads(r.stdout)["data"]["tools"]]
    assert any(n.endswith("echo") for n in tools)
