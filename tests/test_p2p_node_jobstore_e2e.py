import socket, threading, time
import pytest, httpx, uvicorn

from eujeno.net.server import create_app
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


@pytest.mark.slow
def test_p2p_node_logs_job_and_receipts(full_model, tmp_path):
    model, tokenizer = full_model
    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    # A is the entry node (has a durable db); both cover the model together
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3,
                           db_path=str(tmp_path / "nodeA.db")), pA)
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    try:
        with httpx.Client(timeout=120.0) as client:
            for _ in range(100):
                if set(client.get(f"{uA}/registry").json()["nodes"].keys()) == {uA, uB}:
                    break
                time.sleep(0.1)
            resp = client.post(f"{uA}/v1/chat/completions",
                               json={"messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 5}).json()
            assert resp["choices"][0]["message"] is not None, resp
            jobs = client.get(f"{uA}/jobs").json()["jobs"]
            assert len(jobs) >= 1
            jid = jobs[0]["job_id"]
            detail = client.get(f"{uA}/jobs/{jid}").json()
            receipts = client.get(f"{uA}/jobs/{jid}/receipts").json()["receipts"]
        assert detail["status"] == "DONE"
        assert detail["result"] is not None
        assert len(receipts) >= 1
        assert sum(r["hops"] for r in receipts) > 0
        assert all(r["bytes"] > 0 for r in receipts)
    finally:
        sA.should_exit = sB.should_exit = True
