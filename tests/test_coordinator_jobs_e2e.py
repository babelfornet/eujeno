import socket, threading, time, asyncio
import pytest, httpx, uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.net.jobstore import JobStore


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started: break
        time.sleep(0.05)
    assert server.started
    return server


def _run_node_thread(ws_url, state):
    threading.Thread(target=lambda: asyncio.run(run_node(ws_url, state)), daemon=True).start()


@pytest.mark.slow
def test_job_is_persisted_and_reconstructible(full_model, tmp_path):
    model, tokenizer = full_model
    db = str(tmp_path / "jobs.db")
    port = _free_port()
    app = create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", num_layers=24, tokenizer=tokenizer, db_path=db)
    _serve(app, port)
    # one node covering the whole model
    state = NodeState(model, StageSpec(embed=True, head=True, decoders=[(0, 24)]))
    _run_node_thread(f"ws://127.0.0.1:{port}/node", state)
    for _ in range(200):
        r = httpx.get(f"http://127.0.0.1:{port}/registry").json()
        if r["nodes"]: break
        time.sleep(0.05)

    resp = httpx.post(f"http://127.0.0.1:{port}/infer",
                      json={"prompt": "The capital of France is", "max_new_tokens": 5}, timeout=120).json()
    assert resp["ok"] is True
    tokens = resp["tokens"]

    # reconstructible via the read API
    api = httpx.get(f"http://127.0.0.1:{port}/jobs").json()
    assert len(api["jobs"]) >= 1
    jid = api["jobs"][0]["job_id"]
    one = httpx.get(f"http://127.0.0.1:{port}/jobs/{jid}").json()
    assert one["status"] == "DONE"
    assert one["tokens"] == tokens

    reg = httpx.get(f"http://127.0.0.1:{port}/registry").json()
    assert all("load" in n for n in reg["nodes"]), reg
    assert all(n["load"] == 0 for n in reg["nodes"]), reg   # decremented back after completion

    # reconstructible from a freshly-opened DB (durability)
    s2 = JobStore(db)
    assert s2.get_job(jid)["tokens"] == tokens
    assert s2.get_job(jid)["status"] == "DONE"
