import json
from typer.testing import CliRunner
from axyn.cli import app

runner = CliRunner()

STUB = {
    "num_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
    "num_key_value_heads": 4, "intermediate_size": 18944, "vocab_size": 152064,
    "model_type": "qwen2",
}


def test_fit_suggests_stage(monkeypatch):
    import axyn.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: STUB)
    r = runner.invoke(app, ["--json", "fit", "--model", "X", "--ram", "4", "--dtype", "bfloat16"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    assert data["max_decoder_layers"] > 0
    assert "decoder:" in data["suggested_stages"]
    assert data["ram_per_layer_gb"] > 0


def test_fit_bf16_fits_more_than_fp32(monkeypatch):
    import axyn.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: STUB)
    fp32 = json.loads(runner.invoke(app, ["--json", "fit", "--model", "X", "--ram", "8", "--dtype", "float32"]).stdout)["data"]["max_decoder_layers"]
    bf16 = json.loads(runner.invoke(app, ["--json", "fit", "--model", "X", "--ram", "8", "--dtype", "bfloat16"]).stdout)["data"]["max_decoder_layers"]
    assert bf16 >= 2 * fp32 - 1
