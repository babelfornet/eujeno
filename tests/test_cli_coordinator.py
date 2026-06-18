import json
import socket
import threading
import time
import asyncio

import pytest
import uvicorn

from typer.testing import CliRunner
from axyn.cli import app as cli_app
from axyn.net.coordinator import create_coordinator_app
from axyn.net.node import run_node
from axyn.net.node_exec import NodeState
from axyn.net.topology import StageSpec
from axyn.model.generate import reference_generate

runner = CliRunner()


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve_uvicorn(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


def _run_node_thread(ws_url, state):
    threading.Thread(target=lambda: asyncio.run(run_node(ws_url, state)), daemon=True).start()


@pytest.mark.slow
def test_cli_infer_via_coordinator(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    port = _free_port()
    server = _serve_uvicorn(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    try:
        _run_node_thread(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
        _run_node_thread(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))
        import httpx
        with httpx.Client(timeout=30.0) as client:
            for _ in range(200):
                if len(client.get(f"http://127.0.0.1:{port}/registry").json()["nodes"]) == 2:
                    break
                time.sleep(0.05)

        result = runner.invoke(cli_app, ["--json", "infer", "--coordinator", f"http://127.0.0.1:{port}",
                                         "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        server.should_exit = True
