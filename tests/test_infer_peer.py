import json
import socket
import threading
import time

import pytest
import uvicorn

from typer.testing import CliRunner
from axyn.cli import app as cli_app
from axyn.net.topology import StageSpec
from axyn.net.server import create_app
from axyn.model.generate import reference_generate

runner = CliRunner()


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
def test_infer_peer_autodiscovers_and_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    s1 = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3), p1)
    s2 = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3), p2)
    try:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            for _ in range(100):
                if set(client.get(f"{u1}/registry").json()["nodes"].keys()) == {u1, u2}:
                    break
                time.sleep(0.1)
        result = runner.invoke(cli_app, ["--json", "infer", "--peer", u1,
                                         "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        s1.should_exit = True
        s2.should_exit = True
