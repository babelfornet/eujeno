import socket, threading, time, asyncio
import pytest, httpx, uvicorn, websockets

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState, handle_request
from eujeno.net.framing import pack, unpack
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


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


def _thread(coro_factory):
    threading.Thread(target=lambda: asyncio.run(coro_factory()), daemon=True).start()


def _wait_count(client, base, n):
    for _ in range(200):
        if len(client.get(f"{base}/registry").json()["nodes"]) == n:
            return
        time.sleep(0.05)
    raise AssertionError(f"registry never reached {n} nodes")


async def _run_flaky_after(ws_url, state, die_after_decodes):
    """Serve normally, but close the connection on the Nth 'decode' op (crash after some tokens)."""
    seen = {"n": 0}
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_running_loop()
        async for message in ws:
            header, payload = unpack(message)
            if header["op"] == "decode":
                seen["n"] += 1
                if seen["n"] >= die_after_decodes:
                    await ws.close()
                    return
            rh, rp = await loop.run_in_executor(None, handle_request, state, header, payload)
            await ws.send(pack({**rh, "req_id": header.get("req_id")}, rp))


@pytest.mark.slow
def test_failover_resumes_from_persisted_tokens(full_model, tmp_path):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    db = str(tmp_path / "jobs.db")
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer, db_path=db), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=60.0) as client:
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)]))))
            _wait_count(client, base, 1)
            # tail node that dies on the 4th decode -> ~3 tokens already persisted when it fails
            _thread(lambda: _run_flaky_after(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])), 4))
            _wait_count(client, base, 2)
            # redundant tail node to fail over to
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)]))))
            _wait_count(client, base, 3)

            data = client.post(f"{base}/infer", json={"prompt": prompt, "max_new_tokens": 6}).json()
            jobs = client.get(f"{base}/jobs").json()["jobs"]
            detail = client.get(f"{base}/jobs/{jobs[0]['job_id']}").json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference          # resume reproduced the golden sequence
        assert data["failovers"] >= 1
        assert detail["status"] == "DONE"
        assert detail["tokens"] == reference         # durable log drove a correct resume
    finally:
        server.should_exit = True
