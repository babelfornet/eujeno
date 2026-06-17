import json as _json
import os
import sys

import typer

from synapse.config import DEFAULT_MODEL_ID
from synapse.model.blocks import compute_boundaries
from synapse.model.loader import model_config_dims

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Synapse — rete di inferenza LLM decentralizzata.")

# Stato globale impostato dal callback (modalità output).
_state = {"json": False}


def _json_enabled(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("SYNAPSE_JSON", "").lower() in ("1", "true", "yes")


@app.callback()
def _main(json_out: bool = typer.Option(False, "--json", "-j", help="Output come envelope JSON")):
    """Opzioni globali. Imposta la modalità output e silenzia il rumore su stdout."""
    _state["json"] = _json_enabled(json_out)
    # Disciplina degli stream: niente progress bar / log su stdout.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


def _emit_ok(command: str, data: dict, human: str) -> None:
    if _state["json"]:
        typer.echo(_json.dumps({"ok": True, "command": command, "data": data}))
    else:
        typer.echo(human)


def _fail(command: str, code: str, message: str, exit_code: int = 1):
    if _state["json"]:
        typer.echo(_json.dumps({"ok": False, "command": command, "error": {"code": code, "message": message}}))
    else:
        typer.echo(f"error[{code}]: {message}", err=True)
    raise typer.Exit(exit_code)


@app.command()
def version():
    """Stampa la versione del pacchetto."""
    from importlib.metadata import version as _pkg_version
    try:
        v = _pkg_version("synapse")
    except Exception:
        v = "0.0.1"
    _emit_ok("version", {"version": v}, human=f"synapse {v}")


@app.command()
def model(
    info: bool = typer.Option(False, "--info", help="Mostra dimensioni e split proposto"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    blocks: int = typer.Option(2, "--blocks", help="Numero di blocchi decoder"),
):
    """Operazioni sul modello. In v1: --info (default) mostra dims + split proposto."""
    try:
        dims = model_config_dims(model_id)
    except Exception as e:
        _fail("model", "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(dims["num_layers"], blocks)
    except ValueError as e:
        _fail("model", "INVALID_BOUNDARIES", str(e))
    data = {"model": model_id, **dims, "blocks": blocks, "boundaries": boundaries}
    human = (
        f"model: {model_id}\n"
        f"layers: {dims['num_layers']}  hidden: {dims['hidden_size']}  "
        f"heads: {dims['num_attention_heads']} (kv {dims['num_key_value_heads']})\n"
        f"blocchi: {blocks}  confini: {boundaries}"
    )
    _emit_ok("model", data, human)
