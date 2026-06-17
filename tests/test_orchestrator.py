import socket
import threading
import time

import pytest
import httpx
import uvicorn

from synapse.net.topology import StageSpec, Topology
from synapse.net.server import create_app
from synapse.net.orchestrator import distributed_generate
from synapse.model.generate import reference_generate


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "il server uvicorn non è partito"
    return server


@pytest.mark.slow
def test_two_node_distributed_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA dei create_app

    p1, p2 = _free_port(), _free_port()
    app1 = create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]))
    app2 = create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]))
    s1, s2 = _serve(app1, p1), _serve(app2, p2)
    try:
        topo = Topology(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            embed=f"http://127.0.0.1:{p1}",
            head=f"http://127.0.0.1:{p2}",
            decoders=[("0-12", f"http://127.0.0.1:{p1}"), ("12-24", f"http://127.0.0.1:{p2}")],
        )
        with httpx.Client(timeout=60.0) as client:
            result = distributed_generate(topo, "La capitale dell'Italia è", 6, client, tokenizer)
        assert result["tokens"] == reference
        assert isinstance(result["text"], str) and result["text"]
    finally:
        s1.should_exit = True
        s2.should_exit = True
