import json as _json
import os
import sys

import typer

from synapse.config import DEFAULT_MODEL_ID

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
