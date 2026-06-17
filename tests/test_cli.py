import json
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
