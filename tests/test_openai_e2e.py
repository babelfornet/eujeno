import socket, threading, time, asyncio
import pytest, httpx, uvicorn
from synapse.net.coordinator import create_coordinator_app
from synapse.net.node import run_node
from synapse.net.node_exec import NodeState
from synapse.net.topology import StageSpec


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


def _node(ws, state):
    threading.Thread(target=lambda: asyncio.run(run_node(ws, state)), daemon=True).start()


def _two_node_coordinator(full_model):
    model, tokenizer = full_model
    port = _free_port()
    srv = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws = f"ws://127.0.0.1:{port}/node"
    _node(ws, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
    _node(ws, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=30.0) as c:
        for _ in range(200):
            if len(c.get(f"{base}/registry").json()["nodes"]) == 2: break
            time.sleep(0.05)
    return srv, base


@pytest.mark.slow
def test_infer_sampling_seeded_is_reproducible(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            body = {"prompt": "Ciao", "max_new_tokens": 6, "temperature": 0.8, "top_p": 0.9, "seed": 42}
            a = c.post(f"{base}/infer", json=body).json()
            b = c.post(f"{base}/infer", json=body).json()
        assert a["ok"] and b["ok"]
        assert a["tokens"] == b["tokens"]
    finally:
        srv.should_exit = True


@pytest.mark.slow
def test_openai_chat_completions(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            models = c.get(f"{base}/v1/models").json()
            assert models["object"] == "list" and len(models["data"]) >= 1
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "synapse",
                "messages": [{"role": "user", "content": "Di' ciao in una parola"}],
                "max_tokens": 8,
            })
            body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)
        assert body["usage"]["completion_tokens"] >= 1
    finally:
        srv.should_exit = True


@pytest.mark.slow
def test_chat_output_has_no_special_tokens(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "synapse",
                "messages": [{"role": "user", "content": "Di' ciao."}],
                "max_tokens": 64,
            }).json()
        content = r["choices"][0]["message"]["content"]
        assert "<|im_end|>" not in content and "<|endoftext|>" not in content
        assert r["choices"][0]["finish_reason"] in ("stop", "length")
    finally:
        srv.should_exit = True
