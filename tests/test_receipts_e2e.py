import socket, threading, time, asyncio
import pytest, httpx, uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


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


@pytest.mark.slow
def test_receipts_recorded_for_completed_job(full_model, tmp_path):
    model, tokenizer = full_model
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer,
                                           db_path=str(tmp_path / "j.db")), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, head=True, decoders=[(0, 24)]))))
        with httpx.Client(timeout=120.0) as client:
            for _ in range(200):
                if client.get(f"{base}/registry").json()["nodes"]:
                    break
                time.sleep(0.05)
            r = client.post(f"{base}/infer", json={"prompt": "The capital of France is", "max_new_tokens": 5}).json()
            assert r["ok"] is True
            jid = client.get(f"{base}/jobs").json()["jobs"][0]["job_id"]
            receipts = client.get(f"{base}/jobs/{jid}/receipts").json()["receipts"]
        assert len(receipts) >= 1
        total_hops = sum(x["hops"] for x in receipts)
        assert total_hops > 0
        assert all(x["bytes"] > 0 for x in receipts)
        assert all(x["t_compute"] >= 0 for x in receipts)
    finally:
        server.should_exit = True
