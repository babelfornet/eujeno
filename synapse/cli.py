import json as _json
import os
import sys

import typer

from synapse.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from synapse.model.blocks import compute_boundaries, split_into_blocks
from synapse.model.loader import model_config_dims, load_full_model, load_partial_model, model_dims
from synapse.model.generate import reference_generate, pipeline_generate
from synapse.net.topology import parse_stages, load_topology, Topology
from synapse.net.server import create_app
from synapse.net.orchestrator import distributed_generate
from synapse.net.discovery import build_chain
from synapse.net.node_exec import NodeState
from synapse.net.node import run_node
from synapse.net.coordinator import create_coordinator_app

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Synapse — rete di inferenza LLM decentralizzata.")

# Pre-import MCP stdio transport so its sys.stderr default is captured before any test
# runner patches sys.stderr (otherwise stdio_client fails with 'fileno' inside CliRunner).
try:
    import mcp.client.stdio as _mcp_stdio  # noqa: F401
except ImportError:
    pass

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
    from synapse.config import SUPPORTED_ARCHS
    data = {"model": model_id, **dims, "blocks": blocks, "boundaries": boundaries}
    data["architecture"] = dims.get("model_type", "?")
    data["compatible"] = data["architecture"] in SUPPORTED_ARCHS
    human = (
        f"model: {model_id}\n"
        f"layers: {dims['num_layers']}  hidden: {dims['hidden_size']}  "
        f"heads: {dims['num_attention_heads']} (kv {dims['num_key_value_heads']})\n"
        f"blocchi: {blocks}  confini: {boundaries}"
    )
    _emit_ok("model", data, human)


@app.command()
def models():
    """Elenca i modelli/famiglie compatibili (architettura Llama/Qwen2)."""
    from synapse.config import SUPPORTED_ARCHS
    examples = [
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
    ]
    data = {"supported_architectures": sorted(SUPPORTED_ARCHS), "examples": examples,
            "note": "Architettura Llama/Qwen2 (decoder-only). Verifica con 'synapse model --info --model <id>'. "
                    "Per modelli grandi usa --dtype bfloat16 e/o splitta su piu nodi."}
    human = "Modelli compatibili (Llama/Qwen2):\n" + "\n".join(f"  - {m}" for m in examples)
    _emit_ok("models", data, human=human)


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
    peers: str = typer.Option(None, "--peers", help="Seed peer per la discovery gossip, separati da virgola"),
    advertise: str = typer.Option(None, "--advertise", help="URL con cui il nodo si annuncia (es. http://IP:8001). Default http://<host>:<port>"),
    num_layers: int = typer.Option(None, "--num-layers", help="Numero totale layer (per coverage). Default: dal config."),
    coordinator: str = typer.Option(None, "--coordinator", help="URL WS del coordinator (es. ws://host:9000/node). Se presente, il nodo si connette in uscita invece di esporre un server diretto."),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16 (bf16 per modelli grandi)"),
):
    """Avvia un BlockServer che ospita gli stage indicati (processo a lunga durata).

    Carica in RAM SOLO i layer assegnati (partial loading): un nodo non deve avere
    risorse per il modello intero, basta per i suoi stage."""
    import uvicorn
    try:
        spec = parse_stages(stages)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    from synapse.config import parse_dtype
    try:
        _dtype = parse_dtype(dtype)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    try:
        model, tokenizer = load_partial_model(model_id, spec, _dtype, DEVICE)
    except Exception as e:
        _fail("serve", "MODEL_LOAD_FAILED", str(e))
    if coordinator:
        import asyncio
        typer.echo(f"synapse serve→coordinator {coordinator}: stages={stages} (model={model_id})", err=True)
        asyncio.run(run_node(coordinator, NodeState(model, spec)))
        return
    own_url = advertise or f"http://{host}:{port}"
    seeds = [p.strip() for p in peers.split(",")] if peers else []
    nl = num_layers if num_layers is not None else model_config_dims(model_id)["num_layers"]
    fastapi_app = create_app(model, tokenizer, spec, node_url=own_url, peers=seeds, num_layers=nl)
    typer.echo(f"synapse serve (P2P): stages={stages} su http://{host}:{port} advertise={own_url} peers={seeds}", err=True)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command()
def coordinator(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello (per tokenizer + num_layers)"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host di ascolto"),
    port: int = typer.Option(9000, "--port", help="Porta di ascolto"),
):
    """Avvia il coordinator-relay (deve essere raggiungibile dai nodi)."""
    import uvicorn
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        num_layers = model_config_dims(model_id)["num_layers"]
    except Exception as e:
        _fail("coordinator", "MODEL_LOAD_FAILED", str(e))
    coord_app = create_coordinator_app(model_id, num_layers, tokenizer)
    typer.echo(f"synapse coordinator: model={model_id} layers={num_layers} su http://{host}:{port}", err=True)
    uvicorn.run(coord_app, host=host, port=port, log_level="info")


@app.command()
def up(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host del coordinator"),
    port: int = typer.Option(9000, "--port", help="Porta del coordinator"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Stampa i comandi senza avviare nulla"),
):
    """Avvia in un colpo una rete operativa a nodo singolo (coordinator + un nodo che copre tutto il modello)."""
    nl = model_config_dims(model_id)["num_layers"]
    coord_cmd = [sys.executable, "-m", "synapse", "coordinator", "--model", model_id,
                 "--host", host, "--port", str(port)]
    ws = f"ws://{host}:{port}/node"
    serve_cmd = [sys.executable, "-m", "synapse", "serve", "--coordinator", ws,
                 "--stages", f"embed,decoder:0-{nl},head", "--model", model_id, "--dtype", dtype]
    if dry_run:
        _emit_ok("up", {"commands": [coord_cmd, serve_cmd],
                        "coordinator_url": f"http://{host}:{port}"},
                 human="comandi:\n  " + "\n  ".join(" ".join(c) for c in [coord_cmd, serve_cmd]))
        return
    import subprocess
    import time as _t
    import httpx
    procs = [subprocess.Popen(coord_cmd)]
    _t.sleep(4)
    procs.append(subprocess.Popen(serve_cmd))
    base = f"http://{host}:{port}"
    typer.echo(f"synapse up: avvio rete per {model_id} (dtype={dtype})…", err=True)
    operational = False
    for _ in range(120):
        try:
            reg = httpx.get(f"{base}/registry", timeout=5).json()
            if reg.get("nodes"):
                operational = True
                break
        except Exception:
            pass
        _t.sleep(2)
    if operational:
        typer.echo(f"PRONTO. Interroga: synapse infer --coordinator {base} --prompt \"...\"", err=True)
        typer.echo(f"Frontend:        synapse ui --coordinator {base}", err=True)
    else:
        typer.echo("rete non ancora operativa (controlla i log).", err=True)
    try:
        procs[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


@app.command()
def ui(
    coordinator: str = typer.Option("http://127.0.0.1:9000", "--coordinator", help="URL HTTP del coordinator a cui collegarsi"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host della UI"),
    port: int = typer.Option(8500, "--port", help="Porta della UI"),
):
    """Avvia il frontend di controllo locale (dashboard rete + chat)."""
    import uvicorn
    from synapse.ui.server import create_ui_app
    typer.echo(f"synapse ui: http://{host}:{port}  (coordinator={coordinator})", err=True)
    uvicorn.run(create_ui_app(coordinator), host=host, port=port, log_level="info")


@app.command()
def infer(
    topology: str = typer.Option(None, "--topology", help="Path al file JSON di topologia"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt ('-' legge da stdin)"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
    peer: str = typer.Option(None, "--peer", help="[P2P] URL di un nodo qualsiasi: scopre la topologia via gossip ed esegue diretto"),
    coordinator: str = typer.Option(None, "--coordinator", help="[coordinator] URL HTTP del coordinator: client sottile"),
    mcp: bool = typer.Option(False, "--mcp", help="[coordinator/peer] usa i tool MCP configurati (loop tool-calling)"),
):
    """Esegue inferenza distribuita su una topologia di BlockServer."""
    import httpx
    from transformers import AutoTokenizer

    prompt = _read_prompt(prompt)
    if mcp:
        import httpx as _httpx
        from synapse.mcp_config import load_servers
        from synapse.ui.mcp import McpRegistry
        from synapse.ui.agent import run_tool_loop
        target = coordinator or peer
        if not target:
            _fail("infer", "USAGE_ERROR", "--mcp richiede --coordinator o --peer", exit_code=2)
        target = target.rstrip("/")
        reg = McpRegistry()
        for name, cfg in load_servers().items():
            reg.add(name, cfg["command"], cfg.get("args", []))
        if not reg.list_servers():
            _fail("infer", "USAGE_ERROR", "nessun server MCP configurato (usa 'synapse mcp --add')", exit_code=2)
        try:
            tools = reg.list_tools()
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", f"errore MCP: {e}")
        clean_tools = [{"type": t["type"], "function": t["function"]} for t in tools]

        def call_model(messages, tls):
            with _httpx.Client(timeout=300.0) as client:
                rr = client.post(f"{target}/v1/chat/completions",
                                 json={"messages": messages, "tools": tls, "max_tokens": max_new_tokens})
            return rr.json()["choices"][0]["message"]

        try:
            out = run_tool_loop([{"role": "user", "content": prompt}], clean_tools,
                                call_model, lambda n, a: reg.call_tool(n, a), 6)
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", str(e))
        _emit_ok("infer", {"model": "synapse", "prompt": prompt, "text": out["content"],
                           "tool_runs": out["tool_runs"]}, human=out["content"])
        return
    if coordinator:
        try:
            with httpx.Client(timeout=300.0) as client:
                r = client.post(f"{coordinator}/infer", json={"prompt": prompt, "max_new_tokens": max_new_tokens})
                r.raise_for_status()
                body = r.json()
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", str(e))
        if not body.get("ok"):
            _fail("infer", "NOT_OPERATIONAL", body.get("error", "coordinator non pronto"))
        _emit_ok("infer", body, human=body["text"])
        return
    if peer:
        try:
            reg = httpx.get(f"{peer}/registry", timeout=30.0).json()
        except Exception as e:
            _fail("infer", "USAGE_ERROR", f"peer non raggiungibile: {e}", exit_code=2)
        chain = build_chain(reg["nodes"], reg["num_layers"])
        if chain is None:
            _fail("infer", "NOT_OPERATIONAL", "coverage incompleta: il modello non è ancora operativo sulla rete")
        embed_url, decoders, head_url = chain
        topo = Topology(model=reg["model"], embed=embed_url, head=head_url, decoders=decoders)
        try:
            tokenizer = AutoTokenizer.from_pretrained(topo.model)
            with httpx.Client(timeout=120.0) as client:
                result = distributed_generate(topo, prompt, max_new_tokens, client, tokenizer)
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", str(e))
        _emit_ok("infer", {"model": topo.model, "prompt": prompt, **result}, human=result["text"])
        return
    if not topology:
        _fail("infer", "USAGE_ERROR", "specificare --topology o --peer", exit_code=2)
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
def mcp(
    add: str = typer.Option(None, "--add", help="Nome di un server MCP da aggiungere"),
    command: str = typer.Option(None, "--command", help="Comando del server MCP (con --add)"),
    args: str = typer.Option(None, "--args", help="Argomenti del comando, separati da spazio"),
    remove: str = typer.Option(None, "--remove", help="Nome di un server MCP da rimuovere"),
):
    """Configura i server MCP (tool) usabili da 'synapse infer --mcp'. Senza switch: elenca."""
    from synapse.mcp_config import load_servers, add_server, remove_server
    if add:
        if not command:
            _fail("mcp", "USAGE_ERROR", "--command è obbligatorio con --add", exit_code=2)
        servers = add_server(add, command, (args or "").split())
        _emit_ok("mcp", {"servers": list(servers.keys())}, human=f"aggiunto server MCP: {add}")
        return
    if remove:
        servers = remove_server(remove)
        _emit_ok("mcp", {"servers": list(servers.keys())}, human=f"rimosso: {remove}")
        return
    servers = load_servers()
    tools = []
    if servers:
        from synapse.ui.mcp import McpRegistry
        reg = McpRegistry()
        for name, cfg in servers.items():
            reg.add(name, cfg["command"], cfg.get("args", []))
        try:
            tools = [{"name": t["function"]["name"], "description": t["function"]["description"]}
                     for t in reg.list_tools()]
        except Exception as e:
            _emit_ok("mcp", {"servers": list(servers.keys()), "tools": [], "error": str(e)},
                     human=f"server: {list(servers.keys())}  (tool non elencabili: {e})")
            return
    human = "\n".join([f"server MCP: {list(servers.keys())}"] + [f"  🔧 {t['name']} — {t['description']}" for t in tools])
    _emit_ok("mcp", {"servers": list(servers.keys()), "tools": tools}, human=human or "nessun server MCP configurato")


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
