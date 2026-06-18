import socket, threading, time
import pytest, httpx, uvicorn
from axyn.net.topology import StageSpec
from axyn.net.server import create_app


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
def test_query_a_peer_directly(full_model):
    model, tokenizer = full_model
    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    a = create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                   node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3)
    b = create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                   node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3)
    s1, s2 = _serve(a, p1), _serve(b, p2)
    try:
        with httpx.Client(timeout=60.0) as c:
            for _ in range(100):
                if set(c.get(f"{u1}/registry").json()["nodes"].keys()) == {u1, u2}:
                    break
                time.sleep(0.1)
            r = c.post(f"{u1}/v1/chat/completions", json={
                "model": "axyn", "messages": [{"role": "user", "content": "Di' ciao."}], "max_tokens": 8})
            body = r.json()
        assert body["object"] == "chat.completion"
        assert isinstance(body["choices"][0]["message"]["content"], str)
    finally:
        s1.should_exit = True; s2.should_exit = True
