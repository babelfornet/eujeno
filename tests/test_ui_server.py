import socket, threading, time
import pytest, uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from synapse.ui.server import create_ui_app


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _stub_coordinator():
    app = FastAPI()

    @app.get("/registry")
    async def reg():
        return {"num_layers": 24, "model": "stub",
                "nodes": [{"conn": "c1", "stages": {"embed": True, "head": True, "decoders": ["0-24"]}}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        return {"choices": [{"message": {"role": "assistant", "content": "ciao!"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}
    return app


def _serve(app, port):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started: break
        time.sleep(0.05)
    assert srv.started
    return srv


def test_config_endpoint():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    assert c.get("/api/config").json()["coordinator_url"] == "http://example:9000"


def test_serves_index_html():
    app = create_ui_app("http://example:9000")
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "Synapse" in r.text


@pytest.mark.slow
def test_proxies_registry_and_chat():
    port = _free_port()
    srv = _serve(_stub_coordinator(), port)
    try:
        app = create_ui_app(f"http://127.0.0.1:{port}")
        c = TestClient(app)
        reg = c.get("/api/registry").json()
        assert reg["num_layers"] == 24 and len(reg["nodes"]) == 1
        chat = c.post("/api/chat", json={"messages": [{"role": "user", "content": "ciao"}]}).json()
        assert chat["choices"][0]["message"]["content"] == "ciao!"
    finally:
        srv.should_exit = True
