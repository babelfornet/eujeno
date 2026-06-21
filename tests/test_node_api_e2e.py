# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
import socket, threading, time
import pytest, httpx, uvicorn

from eujeno.net.server import create_app
from eujeno.net.topology import StageSpec


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def _serve(app, port):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started: break
        time.sleep(0.05)
    assert srv.started
    return srv


@pytest.mark.slow
def test_node_api_real_data(full_model, tmp_path):
    model, tokenizer = full_model
    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3,
                           db_path=str(tmp_path/"a.db"), config_path=str(tmp_path/"a.json")), pA)
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    try:
        with httpx.Client(timeout=120.0) as client:
            for _ in range(100):
                if set(client.get(f"{uA}/registry").json()["nodes"].keys()) == {uA, uB}: break
                time.sleep(0.1)
            node = client.get(f"{uA}/api/node").json()
            assert node["peerId"].startswith("node·")
            assert node["ramTotalGb"] > 0 and node["numLayers"] == 24
            # settings round-trip
            assert client.put(f"{uA}/api/settings", json={"name": "alpha"}).json()["name"] == "alpha"
            assert client.get(f"{uA}/api/settings").json()["name"] == "alpha"
            # a chat call: routing field present + requests counter grows
            before = client.get(f"{uA}/api/metrics").json()["requestsServed"]
            chat = client.post(f"{uA}/v1/chat/completions",
                               json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).json()
            assert "eujeno" in chat and chat["eujeno"]["layers"] == 24
            after = client.get(f"{uA}/api/metrics").json()["requestsServed"]
            assert after > before
            peers = client.get(f"{uA}/api/peers").json()["peers"]
            assert any(p["url"] == uB for p in peers)
    finally:
        sA.should_exit = sB.should_exit = True
