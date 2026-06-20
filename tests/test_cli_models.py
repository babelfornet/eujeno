import json
from typer.testing import CliRunner
from eujeno.cli import app

runner = CliRunner()


def test_models_lists_examples():
    r = runner.invoke(app, ["--json", "models"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    assert "qwen2" in data["supported_architectures"]
    assert any("Qwen2.5" in m for m in data["examples"])


def test_up_dry_run_prints_commands(monkeypatch):
    # evita il download config: stub model_config_dims
    import eujeno.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: {"num_layers": 24})
    r = runner.invoke(app, ["--json", "up", "--model", "X", "--dtype", "bfloat16", "--dry-run"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    cmds = " ".join(" ".join(c) for c in data["commands"])
    assert "coordinator" in cmds and "serve" in cmds
    assert "embed,decoder:0-24,head" in cmds
    assert "bfloat16" in cmds
