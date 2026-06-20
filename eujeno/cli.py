# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json as _json
import os
import sys

import typer

from eujeno.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from eujeno.model.blocks import compute_boundaries, split_into_blocks
from eujeno.model.loader import model_config_dims, load_full_model, load_partial_model, model_dims
from eujeno.model.generate import reference_generate, pipeline_generate
from eujeno.net.topology import parse_stages, load_topology, Topology
from eujeno.net.server import create_app
from eujeno.net.orchestrator import distributed_generate
from eujeno.net.discovery import build_chain
from eujeno.net.node_exec import NodeState
from eujeno.net.node import run_node
from eujeno.net.coordinator import create_coordinator_app

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Eujeno — decentralized LLM inference network.")

# Pre-import MCP stdio transport so its sys.stderr default is captured before any test
# runner patches sys.stderr (otherwise stdio_client fails with 'fileno' inside CliRunner).
try:
    import mcp.client.stdio as _mcp_stdio  # noqa: F401
except ImportError:
    pass

# Global state set by the callback (output mode).
_state = {"json": False}


def _json_enabled(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("EUJENO_JSON", "").lower() in ("1", "true", "yes")


@app.callback()
def _main(json_out: bool = typer.Option(False, "--json", "-j", help="Output as a JSON envelope")):
    """Global options. Sets the output mode and silences noise on stdout."""
    _state["json"] = _json_enabled(json_out)
    # Stream discipline: no progress bars / logs on stdout.
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


def plan_auto_stages(dims: dict, bytes_per: int, ram_gb: float, reserve: float,
                     stages_by_url: dict, target: int) -> str:
    """Decide the stage spec to claim by combining capacity (fit) and coverage gaps."""
    from eujeno.net.capacity import fit_layers
    from eujeno.net.discovery import coverage_gaps
    from eujeno.net.allocator import choose_stages
    nl = dims["num_layers"]
    fit = fit_layers(dims, bytes_per, ram_gb, reserve)
    gaps = coverage_gaps(stages_by_url, nl, target=target)
    take_eh = fit["fits_whole_model"] or fit["max_decoder_layers"] >= nl
    return choose_stages(gaps, fit["max_decoder_layers"], nl, take_embed_head=take_eh)


@app.command()
def version():
    """Print the package version."""
    from importlib.metadata import version as _pkg_version
    try:
        v = _pkg_version("eujeno")
    except Exception:
        v = "0.0.1"
    _emit_ok("version", {"version": v}, human=f"eujeno {v}")


@app.command()
def model(
    info: bool = typer.Option(False, "--info", help="Show dimensions and proposed split"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    blocks: int = typer.Option(2, "--blocks", help="Number of decoder blocks"),
):
    """Model operations. In v1: --info (default) shows dims + proposed split."""
    try:
        dims = model_config_dims(model_id)
    except Exception as e:
        _fail("model", "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(dims["num_layers"], blocks)
    except ValueError as e:
        _fail("model", "INVALID_BOUNDARIES", str(e))
    from eujeno.config import SUPPORTED_ARCHS
    data = {"model": model_id, **dims, "blocks": blocks, "boundaries": boundaries}
    data["architecture"] = dims.get("model_type", "?")
    data["compatible"] = data["architecture"] in SUPPORTED_ARCHS
    human = (
        f"model: {model_id}\n"
        f"layers: {dims['num_layers']}  hidden: {dims['hidden_size']}  "
        f"heads: {dims['num_attention_heads']} (kv {dims['num_key_value_heads']})\n"
        f"blocks: {blocks}  boundaries: {boundaries}"
    )
    _emit_ok("model", data, human)


@app.command()
def models():
    """List compatible models/families (Llama/Qwen2 architecture)."""
    from eujeno.config import SUPPORTED_ARCHS
    examples = [
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
    ]
    data = {"supported_architectures": sorted(SUPPORTED_ARCHS), "examples": examples,
            "note": "Llama/Qwen2 architecture (decoder-only). Check with 'eujeno model --info --model <id>'. "
                    "For large models use --dtype bfloat16 and/or split across more nodes."}
    human = "Compatible models (Llama/Qwen2):\n" + "\n".join(f"  - {m}" for m in examples)
    _emit_ok("models", data, human=human)


@app.command()
def fit(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    ram: float = typer.Option(..., "--ram", help="RAM available on the node, in GB"),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16"),
    reserve: float = typer.Option(0.2, "--reserve", help="Fraction of RAM reserved for activations/KV-cache/OS (0-0.9)"),
):
    """How many layers can you host with the given RAM? Suggests a stage spec for --stages."""
    from eujeno.config import parse_dtype
    try:
        _dt = parse_dtype(dtype)
    except ValueError as e:
        _fail("fit", "USAGE_ERROR", str(e), exit_code=2)
        return
    import torch
    from eujeno.net.capacity import fit_layers
    bytes_per = torch.finfo(_dt).bits // 8
    d = model_config_dims(model_id)
    nl = d["num_layers"]
    r = fit_layers(d, bytes_per, ram, reserve)
    k = r["max_decoder_layers"]
    fits_whole = r["fits_whole_model"]
    if fits_whole:
        suggested = f"embed,decoder:0-{nl},head"
    elif k > 0:
        suggested = f"decoder:0-{k}"
    else:
        suggested = ""
    data = {
        "model": model_id, "dtype": dtype, "num_layers": nl, "hidden_size": d["hidden_size"],
        "ram_gb": ram, "reserve": reserve,
        "ram_per_layer_gb": r["ram_per_layer_gb"],
        "ram_embed_head_gb": r["ram_embed_head_gb"],
        "max_decoder_layers": k, "fits_whole_model": fits_whole,
        "suggested_stages": suggested,
    }
    if not suggested:
        human = (f"With {ram} GB ({dtype}) it can't even fit a single layer "
                 f"(~{data['ram_per_layer_gb']} GB/layer). Use --dtype bfloat16 or more RAM.")
    elif fits_whole:
        human = (f"With {ram} GB ({dtype}) it fits the WHOLE model ({nl} layers, "
                 f"~{data['ram_per_layer_gb']} GB/layer). Stage: {suggested}")
    else:
        human = (f"With {ram} GB ({dtype}) you can host ~{k}/{nl} decoder layers "
                 f"(~{data['ram_per_layer_gb']} GB/layer). Suggested stage: {suggested}")
    _emit_ok("fit", data, human=human)


def _read_prompt(prompt: str) -> str:
    """'-' reads the prompt from stdin (pipe from an agent)."""
    if prompt == "-":
        return sys.stdin.read().strip()
    return prompt


def _prepare_pipeline(model_id: str, blocks: int, command: str):
    """Load the model, compute boundaries, split. Raises via _fail on error."""
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
    prompt: str = typer.Option(..., "--prompt", help="Prompt text ('-' reads from stdin)"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Number of tokens to generate"),
    blocks: int = typer.Option(2, "--blocks", help="Number of decoder blocks"),
):
    """Generate text by running the split pipeline in-process."""
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
    prompt: str = typer.Option("The capital of Italy is", "--prompt", help="Verification prompt ('-' = stdin)"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Number of tokens to generate"),
    blocks: int = typer.Option(2, "--blocks", help="Number of decoder blocks"),
):
    """Compare the split pipeline against the whole model (golden equivalence)."""
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
        reference = reference_generate(model, ids, max_new_tokens)        # BEFORE the split (split mutates layer_idx)
        embed, decoders, head = split_into_blocks(model, boundaries)
        pipeline = pipeline_generate(embed, decoders, head, ids, max_new_tokens)
    except Exception as e:
        _fail("selfcheck", "GENERATION_FAILED", str(e))
    data = {"model": model_id, "match": reference == pipeline, "reference": reference, "pipeline": pipeline}
    human = f"match: {data['match']}\nreference: {reference}\npipeline: {pipeline}"
    _emit_ok("selfcheck", data, human)


@app.command()
def serve(
    stages: str = typer.Option(None, "--stages", help="Stages served, e.g. 'embed,decoder:0-12'"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    host: str = typer.Option("0.0.0.0", "--host", help="Listen host"),
    port: int = typer.Option(8001, "--port", help="Listen port"),
    peers: str = typer.Option(None, "--peers", help="Seed peers for gossip discovery, comma-separated"),
    advertise: str = typer.Option(None, "--advertise", help="URL the node advertises itself with (e.g. http://IP:8001). Default http://<host>:<port>"),
    num_layers: int = typer.Option(None, "--num-layers", help="Total number of layers (for coverage). Default: from config."),
    coordinator: str = typer.Option(None, "--coordinator", help="Coordinator WS URL (e.g. ws://host:9000/node). If present, the node connects outbound instead of exposing a direct server."),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16 (bf16 for large models)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-assign layers from registry gaps + RAM capacity"),
    ram: float = typer.Option(None, "--ram", help="RAM to use for auto-assignment, GB (default: detected)"),
    reserve: float = typer.Option(0.2, "--reserve", help="Fraction of RAM reserved (auto)"),
    target: int = typer.Option(1, "--target", help="Desired replicas per range (auto; 2 = redundancy)"),
):
    """Start a BlockServer hosting the given stages (long-running process).

    Loads into RAM ONLY the assigned layers (partial loading): a node doesn't need
    resources for the whole model, just enough for its stages."""
    import uvicorn
    if auto:
        import torch
        import httpx
        from eujeno.net.capacity import probe_capacity
        from eujeno.config import parse_dtype as _pdt
        _bp = torch.finfo(_pdt(dtype)).bits // 8
        dims = model_config_dims(model_id)
        ram_gb = ram if ram is not None else (probe_capacity().get("ram_free_gb") or 4.0)
        learned = {}
        for seed in ([p.strip() for p in peers.split(",")] if peers else []):
            try:
                learned.update(httpx.get(f"{seed}/registry", timeout=5).json().get("nodes", {}))
            except Exception:
                pass
        stages = plan_auto_stages(dims, _bp, ram_gb, reserve, learned, target)
        if not stages:
            _fail("serve", "NO_GAP", "no range to cover (coverage complete or insufficient RAM)", exit_code=2)
        typer.echo(f"eujeno serve --auto: claiming stages={stages} (ram={ram_gb}GB, target={target})", err=True)
    elif stages is None:
        _fail("serve", "USAGE_ERROR", "specify --stages or --auto", exit_code=2)
    try:
        spec = parse_stages(stages)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    from eujeno.config import parse_dtype
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
        typer.echo(f"eujeno serve→coordinator {coordinator}: stages={stages} (model={model_id})", err=True)
        asyncio.run(run_node(coordinator, NodeState(model, spec)))
        return
    own_url = advertise or f"http://{host}:{port}"
    seeds = [p.strip() for p in peers.split(",")] if peers else []
    nl = num_layers if num_layers is not None else model_config_dims(model_id)["num_layers"]
    fastapi_app = create_app(model, tokenizer, spec, node_url=own_url, peers=seeds, num_layers=nl)
    typer.echo(f"eujeno serve (P2P): stages={stages} on http://{host}:{port} advertise={own_url} peers={seeds}", err=True)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command()
def coordinator(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Model ID (for tokenizer + num_layers)"),
    host: str = typer.Option("0.0.0.0", "--host", help="Listen host"),
    port: int = typer.Option(9000, "--port", help="Listen port"),
    db: str = typer.Option(None, "--db", help="SQLite job-log path (default ~/.eujeno/coordinator-jobs.db)"),
):
    """Start the coordinator-relay (must be reachable by the nodes)."""
    import uvicorn
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        num_layers = model_config_dims(model_id)["num_layers"]
    except Exception as e:
        _fail("coordinator", "MODEL_LOAD_FAILED", str(e))
    db_path = db or os.path.expanduser("~/.eujeno/coordinator-jobs.db")
    coord_app = create_coordinator_app(model_id, num_layers, tokenizer, db_path=db_path)
    typer.echo(f"eujeno coordinator: model={model_id} layers={num_layers} on http://{host}:{port}", err=True)
    uvicorn.run(coord_app, host=host, port=port, log_level="info")


@app.command()
def up(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16"),
    host: str = typer.Option("127.0.0.1", "--host", help="Coordinator host"),
    port: int = typer.Option(9000, "--port", help="Coordinator port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the commands without starting anything"),
):
    """Bring up an operational single-node network in one shot (coordinator + a node covering the whole model)."""
    nl = model_config_dims(model_id)["num_layers"]
    coord_cmd = [sys.executable, "-m", "eujeno", "coordinator", "--model", model_id,
                 "--host", host, "--port", str(port)]
    ws = f"ws://{host}:{port}/node"
    serve_cmd = [sys.executable, "-m", "eujeno", "serve", "--coordinator", ws,
                 "--stages", f"embed,decoder:0-{nl},head", "--model", model_id, "--dtype", dtype]
    if dry_run:
        _emit_ok("up", {"commands": [coord_cmd, serve_cmd],
                        "coordinator_url": f"http://{host}:{port}"},
                 human="commands:\n  " + "\n  ".join(" ".join(c) for c in [coord_cmd, serve_cmd]))
        return
    import subprocess
    import time as _t
    import httpx
    procs = [subprocess.Popen(coord_cmd)]
    _t.sleep(4)
    procs.append(subprocess.Popen(serve_cmd))
    base = f"http://{host}:{port}"
    typer.echo(f"eujeno up: starting network for {model_id} (dtype={dtype})…", err=True)
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
        typer.echo(f"READY. Query it: eujeno infer --coordinator {base} --prompt \"...\"", err=True)
        typer.echo(f"Frontend:       eujeno ui --coordinator {base}", err=True)
    else:
        typer.echo("network not operational yet (check the logs).", err=True)
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
    coordinator: str = typer.Option("http://127.0.0.1:9000", "--coordinator", help="HTTP URL of the coordinator to connect to"),
    host: str = typer.Option("127.0.0.1", "--host", help="UI host"),
    port: int = typer.Option(8500, "--port", help="UI port"),
):
    """Start the local control frontend (network dashboard + chat)."""
    import uvicorn
    from eujeno.ui.server import create_ui_app
    typer.echo(f"eujeno ui: http://{host}:{port}  (coordinator={coordinator})", err=True)
    uvicorn.run(create_ui_app(coordinator), host=host, port=port, log_level="info")


@app.command()
def infer(
    topology: str = typer.Option(None, "--topology", help="Path to the topology JSON file"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt ('-' reads from stdin)"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Number of tokens to generate"),
    peer: str = typer.Option(None, "--peer", help="[P2P] URL of any node: discovers the topology via gossip and runs directly"),
    coordinator: str = typer.Option(None, "--coordinator", help="[coordinator] HTTP URL of the coordinator: thin client"),
    mcp: bool = typer.Option(False, "--mcp", help="[coordinator/peer] use the configured MCP tools (tool-calling loop)"),
):
    """Run distributed inference over a topology of BlockServers."""
    import httpx
    from transformers import AutoTokenizer

    prompt = _read_prompt(prompt)
    if mcp:
        import httpx as _httpx
        from eujeno.mcp_config import load_servers
        from eujeno.ui.mcp import McpRegistry
        from eujeno.ui.agent import run_tool_loop
        target = coordinator or peer
        if not target:
            _fail("infer", "USAGE_ERROR", "--mcp requires --coordinator or --peer", exit_code=2)
        target = target.rstrip("/")
        reg = McpRegistry()
        for name, cfg in load_servers().items():
            reg.add(name, cfg["command"], cfg.get("args", []))
        if not reg.list_servers():
            _fail("infer", "USAGE_ERROR", "no MCP server configured (use 'eujeno mcp --add')", exit_code=2)
        try:
            tools = reg.list_tools()
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", f"MCP error: {e}")
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
        _emit_ok("infer", {"model": "eujeno", "prompt": prompt, "text": out["content"],
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
            _fail("infer", "NOT_OPERATIONAL", body.get("error", "coordinator not ready"))
        _emit_ok("infer", body, human=body["text"])
        return
    if peer:
        peer = peer.rstrip("/")
        try:
            reg = httpx.get(f"{peer}/registry", timeout=30.0).json()
        except Exception as e:
            _fail("infer", "USAGE_ERROR", f"peer unreachable: {e}", exit_code=2)
        from eujeno.net.orchestrator import distributed_generate_resilient
        from eujeno.net.generation import stop_token_ids
        try:
            tokenizer = AutoTokenizer.from_pretrained(reg["model"])
            stop_ids = stop_token_ids(tokenizer)
            def _refresh():
                return httpx.get(f"{peer}/registry", timeout=10.0).json()["nodes"]
            with httpx.Client(timeout=120.0) as client:
                result = distributed_generate_resilient(
                    reg["nodes"], reg["num_layers"], prompt, max_new_tokens, client, tokenizer,
                    stop_ids=stop_ids, refresh=_refresh)
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", str(e))
        if not result.get("ok"):
            _fail("infer", "NOT_OPERATIONAL", result.get("error", "the model is not operational on the network yet"))
        _emit_ok("infer", {"model": reg["model"], "prompt": prompt,
                           "text": result["text"], "tokens": result["tokens"],
                           "failovers": result["failovers"]}, human=result["text"])
        return
    if not topology:
        _fail("infer", "USAGE_ERROR", "specify --topology or --peer", exit_code=2)
    try:
        with open(topology) as f:
            topo = load_topology(_json.loads(f.read()))
    except Exception as e:
        _fail("infer", "USAGE_ERROR", f"topology not readable: {e}", exit_code=2)
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
    add: str = typer.Option(None, "--add", help="Name of an MCP server to add"),
    command: str = typer.Option(None, "--command", help="MCP server command (with --add)"),
    args: str = typer.Option(None, "--args", help="Command arguments, space-separated"),
    remove: str = typer.Option(None, "--remove", help="Name of an MCP server to remove"),
):
    """Configure the MCP servers (tools) usable by 'eujeno infer --mcp'. Without switches: list them."""
    from eujeno.mcp_config import load_servers, add_server, remove_server
    if add:
        if not command:
            _fail("mcp", "USAGE_ERROR", "--command is required with --add", exit_code=2)
        servers = add_server(add, command, (args or "").split())
        _emit_ok("mcp", {"servers": list(servers.keys())}, human=f"added MCP server: {add}")
        return
    if remove:
        servers = remove_server(remove)
        _emit_ok("mcp", {"servers": list(servers.keys())}, human=f"removed: {remove}")
        return
    servers = load_servers()
    tools = []
    if servers:
        from eujeno.ui.mcp import McpRegistry
        reg = McpRegistry()
        for name, cfg in servers.items():
            reg.add(name, cfg["command"], cfg.get("args", []))
        try:
            tools = [{"name": t["function"]["name"], "description": t["function"]["description"]}
                     for t in reg.list_tools()]
        except Exception as e:
            _emit_ok("mcp", {"servers": list(servers.keys()), "tools": [], "error": str(e)},
                     human=f"servers: {list(servers.keys())}  (tools not listable: {e})")
            return
    human = "\n".join([f"MCP servers: {list(servers.keys())}"] + [f"  🔧 {t['name']} — {t['description']}" for t in tools])
    _emit_ok("mcp", {"servers": list(servers.keys()), "tools": tools}, human=human or "no MCP server configured")


@app.command()
def schema():
    """Print the command+option tree in machine-readable form (for AI agents)."""
    import click
    import typer.main

    root = typer.main.get_command(app)
    commands = []
    for name, cmd in sorted(root.commands.items()):
        options = []
        for param in cmd.params:
            if isinstance(param, click.Argument):
                continue
            # Expose the real CLI flag (e.g. "--model"), not the Python
            # variable name (e.g. "model_id"): an agent uses it verbatim.
            flag = param.opts[0] if param.opts else param.name
            options.append({
                "name": flag,
                "type": getattr(param.type, "name", str(param.type)),
                "default": param.default,
                "required": bool(param.required),
            })
        commands.append({"name": name, "help": (cmd.help or "").strip(), "options": options})
    _emit_ok("schema", {"commands": commands}, human=_json.dumps({"commands": commands}, indent=2))
