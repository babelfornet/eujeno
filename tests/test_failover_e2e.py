import socket
import threading
import time
import asyncio

import pytest
import httpx
import uvicorn
import websockets

from synapse.net.coordinator import create_coordinator_app
from synapse.net.node import run_node
from synapse.net.node_exec import NodeState, handle_request
from synapse.net.framing import pack, unpack
from synapse.net.topology import StageSpec
from synapse.model.generate import reference_generate


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


async def _run_flaky_node(ws_url, state):
    """Annuncia, serve gli hop, ma CHIUDE la connessione alla prima 'decode' (crash simulato)."""
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_running_loop()
        async for message in ws:
            header, payload = unpack(message)
            if header["op"] == "decode":
                await ws.close()
                return
            rh, rp = await loop.run_in_executor(None, handle_request, state, header, payload)
            await ws.send(pack({**rh, "req_id": header.get("req_id")}, rp))


def _thread(coro_factory):
    threading.Thread(target=lambda: asyncio.run(coro_factory()), daemon=True).start()


def _registry_count(client, base):
    return len(client.get(f"{base}/registry").json()["nodes"])


@pytest.mark.slow
def test_failover_completes_via_redundant_node(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=30.0) as client:
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)]))))
            for _ in range(200):
                if _registry_count(client, base) == 1:
                    break
                time.sleep(0.05)
            _thread(lambda: _run_flaky_node(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)]))))
            for _ in range(200):
                if _registry_count(client, base) == 2:
                    break
                time.sleep(0.05)
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)]))))
            for _ in range(200):
                if _registry_count(client, base) == 3:
                    break
                time.sleep(0.05)

            r = client.post(f"{base}/infer", json={"prompt": "La capitale dell'Italia è", "max_new_tokens": 6})
            data = r.json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference
        assert data["failovers"] >= 1
    finally:
        server.should_exit = True
