import json
import socket
import threading
import time

import pytest
import uvicorn

from typer.testing import CliRunner
from synapse.cli import app as cli_app
from synapse.net.topology import StageSpec
from synapse.net.server import create_app
from synapse.model.generate import reference_generate

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
def test_cli_infer_against_two_nodes(full_model, tmp_path):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    p1, p2 = _free_port(), _free_port()
    s1 = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)])), p1)
    s2 = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)])), p2)
    try:
        topo = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "embed": f"http://127.0.0.1:{p1}",
            "decoders": [{"block": "0-12", "url": f"http://127.0.0.1:{p1}"},
                         {"block": "12-24", "url": f"http://127.0.0.1:{p2}"}],
            "head": f"http://127.0.0.1:{p2}",
        }
        topo_file = tmp_path / "topo.json"
        topo_file.write_text(json.dumps(topo))

        result = runner.invoke(cli_app, ["--json", "infer", "--topology", str(topo_file),
                                         "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        s1.should_exit = True
        s2.should_exit = True
