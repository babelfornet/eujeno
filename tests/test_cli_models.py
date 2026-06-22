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
    # the serve command now always carries a resolved --device
    assert "--device" in cmds


def test_up_dry_run_explicit_device(monkeypatch):
    import eujeno.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: {"num_layers": 24})
    r = runner.invoke(app, ["--json", "up", "--model", "X", "--device", "cpu", "--dry-run"])
    assert r.exit_code == 0
    cmds = " ".join(" ".join(c) for c in json.loads(r.stdout)["data"]["commands"])
    assert "--device cpu" in cmds
    # CPU -> dtype defaults to float32
    assert "--dtype float32" in cmds


def test_up_dry_run_gpu_defaults_to_bf16(monkeypatch):
    # explicit GPU device -> dtype auto-defaults to bfloat16
    import eujeno.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: {"num_layers": 24})
    r = runner.invoke(app, ["--json", "up", "--model", "X", "--device", "mps", "--dry-run"])
    cmds = " ".join(" ".join(c) for c in json.loads(r.stdout)["data"]["commands"])
    assert "--device mps" in cmds and "--dtype bfloat16" in cmds


def test_up_dry_run_explicit_dtype_wins(monkeypatch):
    # an explicit --dtype overrides the per-device default
    import eujeno.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: {"num_layers": 24})
    r = runner.invoke(app, ["--json", "up", "--model", "X", "--device", "mps",
                            "--dtype", "float16", "--dry-run"])
    cmds = " ".join(" ".join(c) for c in json.loads(r.stdout)["data"]["commands"])
    assert "--dtype float16" in cmds


def test_up_dry_run_auto_device(monkeypatch):
    # no --device -> the serve cmd carries the auto-detected device
    import eujeno.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: {"num_layers": 24})
    monkeypatch.setattr(cli, "resolve_device", lambda d: "mps" if d is None else d)
    r = runner.invoke(app, ["--json", "up", "--model", "X", "--dry-run"])
    cmds = " ".join(" ".join(c) for c in json.loads(r.stdout)["data"]["commands"])
    assert "--device mps" in cmds
