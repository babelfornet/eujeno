import sys, os
import pytest
from eujeno.ui.mcp import McpRegistry

_SERVER = os.path.join(os.path.dirname(__file__), "_mcp_echo_server.py")


@pytest.mark.slow
def test_list_and_call_tool_via_stdio():
    reg = McpRegistry()
    reg.add("echo", sys.executable, [_SERVER])
    tools = reg.list_tools()
    names = [t["function"]["name"] for t in tools]
    assert any(n.endswith("echo") for n in names)
    full = next(n for n in names if n.endswith("echo"))
    out = reg.call_tool(full, {"text": "ciao"})
    assert "echo: ciao" in out
    assert reg.list_servers() == ["echo"]
    reg.remove("echo")
    assert reg.list_servers() == []
