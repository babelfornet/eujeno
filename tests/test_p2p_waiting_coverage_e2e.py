import socket, threading, time
import pytest, httpx, uvicorn

from eujeno.net.server import create_app
from eujeno.net.topology import StageSpec
from eujeno.net.orchestrator import distributed_generate_resilient
from eujeno.net.generation import stop_token_ids
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


@pytest.mark.slow
def test_p2p_waits_for_coverage_then_completes(full_model):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 24)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3), pA)
    holder = {}

    def _run():
        with httpx.Client(timeout=90.0) as client:
            holder["r"] = distributed_generate_resilient(
                {uA: {"embed": True, "head": False, "decoders": ["0-24"]}}, 24, prompt, 6, client, tokenizer,
                stop_ids=stop_token_ids(tokenizer), coverage_timeout=30,
                refresh=lambda: httpx.get(f"{uA}/registry", timeout=10).json()["nodes"])

    t = threading.Thread(target=_run, daemon=True); t.start()
    time.sleep(2.0)
    assert t.is_alive(), "should be waiting for coverage (head uncovered), not returned"
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    t.join(timeout=60)
    try:
        assert not t.is_alive(), "did not complete after coverage arrived"
        assert holder["r"]["ok"] is True, holder
        assert holder["r"]["tokens"] == reference
    finally:
        sA.should_exit = sB.should_exit = True


@pytest.mark.slow
def test_p2p_coverage_timeout(full_model):
    model, tokenizer = full_model
    pA = _free_port()
    uA = f"http://127.0.0.1:{pA}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 24)]),
                           node_url=uA, peers=[], num_layers=24, gossip_interval=0.3), pA)
    try:
        with httpx.Client(timeout=30.0) as client:
            result = distributed_generate_resilient(
                {uA: {"embed": True, "head": False, "decoders": ["0-24"]}}, 24, "ciao", 4, client, tokenizer,
                stop_ids=set(), coverage_timeout=2,
                refresh=lambda: httpx.get(f"{uA}/registry", timeout=5).json()["nodes"])
        assert result["ok"] is False
        assert "coverage timeout" in result["error"]
    finally:
        sA.should_exit = True
