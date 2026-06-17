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


@pytest.mark.slow
def test_generate_json_produces_text():
    result = runner.invoke(app, ["--json", "generate", "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "8"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert isinstance(payload["data"]["text"], str) and payload["data"]["text"]
    assert len(payload["data"]["tokens"]) == 8


@pytest.mark.slow
def test_generate_reads_prompt_from_stdin():
    result = runner.invoke(app, ["--json", "generate", "--prompt", "-", "--max-new-tokens", "4"], input="La capitale dell'Italia è")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data"]["prompt"] == "La capitale dell'Italia è"


@pytest.mark.slow
def test_selfcheck_reports_match():
    result = runner.invoke(app, ["--json", "selfcheck", "--max-new-tokens", "8"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data"]["match"] is True
    assert payload["data"]["reference"] == payload["data"]["pipeline"]


def test_schema_lists_commands():
    result = runner.invoke(app, ["--json", "schema"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = {c["name"] for c in payload["data"]["commands"]}
    assert {"version", "model", "generate", "selfcheck", "schema"} <= names
    # ogni comando elenca le sue opzioni con nome
    model_cmd = next(c for c in payload["data"]["commands"] if c["name"] == "model")
    opt_names = {o["name"] for o in model_cmd["options"]}
    assert "info" in opt_names and "blocks" in opt_names


@pytest.mark.slow
def test_json_stdout_is_single_pure_json_object():
    # In modalità --json, stdout deve essere ESATTAMENTE un envelope JSON,
    # senza barre di progresso o warning di transformers mescolati.
    result = runner.invoke(app, ["--json", "model", "--info"])
    assert result.exit_code == 0
    out = result.stdout.strip()
    payload = json.loads(out)              # solleva se c'è rumore non-JSON su stdout
    assert payload["ok"] is True
    assert out.count("\n") == 0            # una sola riga JSON, nessuna riga extra
