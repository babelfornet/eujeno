import json
import pytest
from typer.testing import CliRunner
from synapse.cli import app

runner = CliRunner()


def test_version_json_is_valid_envelope():
    result = runner.invoke(app, ["--json", "version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)          # stdout deve essere JSON puro
    assert payload["ok"] is True
    assert payload["command"] == "version"
    assert "version" in payload["data"]


def test_version_text_mode():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "synapse" in result.stdout


@pytest.mark.slow
def test_model_info_json():
    result = runner.invoke(app, ["--json", "model", "--info", "--blocks", "2"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["num_layers"] == 24
    assert payload["data"]["boundaries"] == [0, 12, 24]


@pytest.mark.slow
def test_model_invalid_blocks_returns_error_envelope():
    result = runner.invoke(app, ["--json", "model", "--info", "--blocks", "999"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_BOUNDARIES"
