import socket, threading, time, asyncio
import pytest, httpx, uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


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


def _wait_count(client, base, n):
    for _ in range(200):
        if len(client.get(f"{base}/registry").json()["nodes"]) == n:
            return
        time.sleep(0.05)
    raise AssertionError(f"registry never reached {n} nodes")


@pytest.mark.slow
def test_job_parks_waiting_coverage_then_resumes(full_model, tmp_path):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)
    db = str(tmp_path / "jobs.db")
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer, db_path=db), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    result = {}

    def _infer():
        with httpx.Client(timeout=120.0) as c:
            result["data"] = c.post(f"{base}/infer", json={"prompt": prompt, "max_new_tokens": 6}).json()

    try:
        # only embed+decoders -> the head block is uncovered
        _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 24)]))))
        with httpx.Client(timeout=10.0) as client:
            _wait_count(client, base, 1)
            t = threading.Thread(target=_infer, daemon=True); t.start()
            parked = False
            for _ in range(400):
                jobs = client.get(f"{base}/jobs").json()["jobs"]
                if jobs and jobs[0]["status"] == "WAITING_COVERAGE":
                    parked = True; break
                time.sleep(0.05)
            assert parked, "job never entered WAITING_COVERAGE"
            # provide the head -> coverage completes -> job resumes
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(head=True))))
            t.join(timeout=120)
            jid = client.get(f"{base}/jobs").json()["jobs"][0]["job_id"]
            detail = client.get(f"{base}/jobs/{jid}").json()
        assert result["data"]["ok"] is True, result
        assert result["data"]["tokens"] == reference
        assert detail["status"] == "DONE"
    finally:
        server.should_exit = True


@pytest.mark.slow
def test_coverage_timeout_fails(full_model, tmp_path):
    model, tokenizer = full_model
    db = str(tmp_path / "jobs.db")
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer,
                                           db_path=db, coverage_timeout=2.0), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 24)]))))  # no head, ever
        with httpx.Client(timeout=30.0) as client:
            _wait_count(client, base, 1)
            data = client.post(f"{base}/infer", json={"prompt": "ciao", "max_new_tokens": 4}).json()
            jobs = client.get(f"{base}/jobs").json()["jobs"]
        assert data["ok"] is False
        assert "coverage timeout" in data.get("error", "")
        assert jobs[0]["status"] == "FAILED"
    finally:
        server.should_exit = True
