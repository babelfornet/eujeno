import socket
import threading
import time

import pytest
import httpx
import uvicorn

from eujeno.net.topology import StageSpec
from eujeno.net.server import create_app


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


@pytest.mark.slow
def test_registry_converges_via_gossip(full_model):
    model, tokenizer = full_model
    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    app1 = create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                      node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3, ttl=30.0)
    app2 = create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                      node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3, ttl=30.0)
    s1, s2 = _serve(app1, p1), _serve(app2, p2)
    try:
        with httpx.Client(timeout=10.0) as client:
            converged = False
            for _ in range(100):
                reg = client.get(f"{u1}/registry").json()
                if set(reg["nodes"].keys()) == {u1, u2}:
                    converged = True
                    break
                time.sleep(0.1)
            assert converged, reg
            assert reg["num_layers"] == 24
            assert reg["nodes"][u2]["head"] is True
    finally:
        s1.should_exit = True
        s2.should_exit = True
