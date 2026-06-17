import json as _json
import os
import sys

import typer

from synapse.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from synapse.model.blocks import compute_boundaries, split_into_blocks
from synapse.model.loader import model_config_dims, load_full_model, model_dims
from synapse.model.generate import reference_generate, pipeline_generate
from synapse.net.topology import parse_stages, load_topology
from synapse.net.server import create_app
from synapse.net.orchestrator import distributed_generate

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


def _read_prompt(prompt: str) -> str:
    """'-' legge il prompt da stdin (pipe da un agente)."""
    if prompt == "-":
        return sys.stdin.read().strip()
    return prompt


def _prepare_pipeline(model_id: str, blocks: int, command: str):
    """Carica il modello, calcola i confini, splitta. Solleva via _fail su errore."""
    try:
        model, tokenizer = load_full_model(model_id, DTYPE, DEVICE)
        model.eval()
    except Exception as e:
        _fail(command, "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(model_dims(model)["num_layers"], blocks)
    except ValueError as e:
        _fail(command, "INVALID_BOUNDARIES", str(e))
    embed, decoders, head = split_into_blocks(model, boundaries)
    return model, tokenizer, embed, decoders, head


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", help="Testo del prompt ('-' legge da stdin)"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
    blocks: int = typer.Option(2, "--blocks", help="Numero di blocchi decoder"),
):
    """Genera testo eseguendo la pipeline splittata in-process."""
    prompt = _read_prompt(prompt)
    model, tokenizer, embed, decoders, head = _prepare_pipeline(model_id, blocks, "generate")
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        tokens = pipeline_generate(embed, decoders, head, ids, max_new_tokens)
        text = tokenizer.decode(tokens)
    except Exception as e:
        _fail("generate", "GENERATION_FAILED", str(e))
    data = {"model": model_id, "prompt": prompt, "text": text, "tokens": tokens}
    _emit_ok("generate", data, human=text)


@app.command()
def selfcheck(
    prompt: str = typer.Option("La capitale dell'Italia è", "--prompt", help="Prompt di verifica ('-' = stdin)"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
    blocks: int = typer.Option(2, "--blocks", help="Numero di blocchi decoder"),
):
    """Confronta la pipeline splittata col modello intero (golden equivalence)."""
    prompt = _read_prompt(prompt)
    try:
        model, tokenizer = load_full_model(model_id, DTYPE, DEVICE)
        model.eval()
    except Exception as e:
        _fail("selfcheck", "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(model_dims(model)["num_layers"], blocks)
    except ValueError as e:
        _fail("selfcheck", "INVALID_BOUNDARIES", str(e))
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        reference = reference_generate(model, ids, max_new_tokens)        # PRIMA dello split (split muta layer_idx)
        embed, decoders, head = split_into_blocks(model, boundaries)
        pipeline = pipeline_generate(embed, decoders, head, ids, max_new_tokens)
    except Exception as e:
        _fail("selfcheck", "GENERATION_FAILED", str(e))
    data = {"model": model_id, "match": reference == pipeline, "reference": reference, "pipeline": pipeline}
    human = f"match: {data['match']}\nreference: {reference}\npipeline: {pipeline}"
    _emit_ok("selfcheck", data, human)


@app.command()
def serve(
    stages: str = typer.Option(..., "--stages", help="Stage serviti, es. 'embed,decoder:0-12'"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host di ascolto"),
    port: int = typer.Option(8001, "--port", help="Porta di ascolto"),
):
    """Avvia un BlockServer che ospita gli stage indicati (processo a lunga durata)."""
    import uvicorn
    try:
        spec = parse_stages(stages)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    try:
        model, tokenizer = load_full_model(model_id, DTYPE, DEVICE)
        model.eval()
    except Exception as e:
        _fail("serve", "MODEL_LOAD_FAILED", str(e))
    fastapi_app = create_app(model, tokenizer, spec)
    typer.echo(f"synapse serve: stages={stages} su http://{host}:{port}  (model={model_id})", err=True)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command()
def infer(
    topology: str = typer.Option(..., "--topology", help="Path al file JSON di topologia"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt ('-' legge da stdin)"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
):
    """Esegue inferenza distribuita su una topologia di BlockServer."""
    import httpx
    from transformers import AutoTokenizer

    prompt = _read_prompt(prompt)
    try:
        with open(topology) as f:
            topo = load_topology(_json.loads(f.read()))
    except Exception as e:
        _fail("infer", "USAGE_ERROR", f"topologia non leggibile: {e}", exit_code=2)
    try:
        tokenizer = AutoTokenizer.from_pretrained(topo.model)
    except Exception as e:
        _fail("infer", "MODEL_LOAD_FAILED", str(e))
    try:
        with httpx.Client(timeout=120.0) as client:
            result = distributed_generate(topo, prompt, max_new_tokens, client, tokenizer)
    except Exception as e:
        _fail("infer", "GENERATION_FAILED", str(e))
    data = {"model": topo.model, "prompt": prompt, **result}
    _emit_ok("infer", data, human=result["text"])


@app.command()
def schema():
    """Stampa l'albero comandi+opzioni in forma machine-readable (per agenti AI)."""
    import click
    import typer.main

    root = typer.main.get_command(app)
    commands = []
    for name, cmd in sorted(root.commands.items()):
        options = []
        for param in cmd.params:
            if isinstance(param, click.Argument):
                continue
            # Esponi il flag CLI reale (es. "--model"), non il nome della
            # variabile Python (es. "model_id"): un agente lo usa verbatim.
            flag = param.opts[0] if param.opts else param.name
            options.append({
                "name": flag,
                "type": getattr(param.type, "name", str(param.type)),
                "default": param.default,
                "required": bool(param.required),
            })
        commands.append({"name": name, "help": (cmd.help or "").strip(), "options": options})
    _emit_ok("schema", {"commands": commands}, human=_json.dumps({"commands": commands}, indent=2))
