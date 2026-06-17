import json
from typer.testing import CliRunner
from synapse.cli import app

runner = CliRunner()


def test_models_lists_examples():
    r = runner.invoke(app, ["--json", "models"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    assert "qwen2" in data["supported_architectures"]
    assert any("Qwen2.5" in m for m in data["examples"])
