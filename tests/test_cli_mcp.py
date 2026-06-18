import json, os, sys
import pytest
from typer.testing import CliRunner
from axyn.cli import app

runner = CliRunner()
_ECHO = os.path.join(os.path.dirname(__file__), "_mcp_echo_server.py")


def test_mcp_add_list_remove(monkeypatch, tmp_path):
    monkeypatch.setenv("AXYN_HOME", str(tmp_path))
    r = runner.invoke(app, ["--json", "mcp", "--add", "x", "--command", "echo", "--args", "hello world"])
    assert r.exit_code == 0
    assert "x" in json.loads(r.stdout)["data"]["servers"]
    r = runner.invoke(app, ["--json", "mcp"])   # default = list
    assert "x" in json.loads(r.stdout)["data"]["servers"]
    r = runner.invoke(app, ["--json", "mcp", "--remove", "x"])
    assert json.loads(r.stdout)["data"]["servers"] == []


def test_mcp_add_requires_command(monkeypatch, tmp_path):
    monkeypatch.setenv("AXYN_HOME", str(tmp_path))
    r = runner.invoke(app, ["mcp", "--add", "x"])
    assert r.exit_code == 2


@pytest.mark.slow
def test_mcp_list_discovers_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("AXYN_HOME", str(tmp_path))
    runner.invoke(app, ["mcp", "--add", "echo", "--command", sys.executable, "--args", _ECHO])
    r = runner.invoke(app, ["--json", "mcp"])
    tools = [t["name"] for t in json.loads(r.stdout)["data"]["tools"]]
    assert any(n.endswith("echo") for n in tools)


@pytest.mark.slow
def test_infer_mcp_runs_tool_loop(monkeypatch, tmp_path):
    import socket, threading, time
    import uvicorn
    from fastapi import FastAPI, Request

    monkeypatch.setenv("AXYN_HOME", str(tmp_path))
    runner.invoke(app, ["mcp", "--add", "echo", "--command", sys.executable, "--args", _ECHO])

    state = {"n": 0}
    fake = FastAPI()

    @fake.post("/v1/chat/completions")
    async def v1(request: Request):
        state["n"] += 1
        if state["n"] == 1:
            return {"choices": [{"message": {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c0", "type": "function",
                                    "function": {"name": "echo__echo", "arguments": json.dumps({"text": "ciao"})}}]},
                    "finish_reason": "tool_calls"}]}
        return {"choices": [{"message": {"role": "assistant", "content": "Il tool ha risposto."},
                "finish_reason": "stop"}]}

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started: break
        time.sleep(0.05)
    try:
        r = runner.invoke(app, ["--json", "infer", "--coordinator", f"http://127.0.0.1:{port}",
                                "--mcp", "--prompt", "usa echo"])
        assert r.exit_code == 0, r.stdout
        data = json.loads(r.stdout)["data"]
        assert data["text"] == "Il tool ha risposto."
        assert any(run["name"] == "echo__echo" for run in data.get("tool_runs", []))
    finally:
        srv.should_exit = True
