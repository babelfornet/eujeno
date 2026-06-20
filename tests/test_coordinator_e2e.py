import socket
import threading
import time
import asyncio

import pytest
import httpx
import uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


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
def test_two_nodes_via_coordinator_match_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA dei NodeState

    port = _free_port()
    app = create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", num_layers=24, tokenizer=tokenizer)
    server = _serve_uvicorn(app, port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    try:
        _run_node_thread(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
        _run_node_thread(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))
        with httpx.Client(timeout=30.0) as client:
            for _ in range(200):
                reg = client.get(f"http://127.0.0.1:{port}/registry").json()
                if len(reg["nodes"]) == 2:
                    break
                time.sleep(0.05)
            r = client.post(f"http://127.0.0.1:{port}/infer",
                            json={"prompt": "La capitale dell'Italia è", "max_new_tokens": 6})
            data = r.json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference
    finally:
        server.should_exit = True
