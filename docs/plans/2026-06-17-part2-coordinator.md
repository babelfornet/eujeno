# Part 2 — Coordinator-relay + discovery automatica — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Inferenza distribuita che funziona **LAN e internet senza VPN**: i nodi worker si connettono in uscita a un **coordinator** pubblicamente raggiungibile che fa **discovery automatica** (registry dei blocchi) e **instrada** le attivazioni; `synapse infer` è un client sottile.

**Architecture:** Vedi [ADR-0002](../decisions/ADR-0002-connettivita-nat.md). Coordinator-relay di Milestone 0: WebSocket outbound nodo→coordinator (NAT-friendly); il coordinator tiene il registry, calcola la coverage, e guida il loop di generazione relayando ogni hop al nodo giusto. Riusa l'esecuzione a blocchi e il wire safetensors di Parte 1; cambia solo il *trasporto* (WS relay invece di POST diretti).

**Tech Stack:** Python · FastAPI WebSocket (coordinator) · `websockets` lib (client nodo) · httpx (client infer) · safetensors · asyncio · l'esistente `synapse/net/` e `synapse/model/`.

**Decisioni:** framing binario `4-byte len + JSON header + payload safetensors`; correlazione richiesta/risposta via `req_id` + Future; greedy/argmax; il coordinator possiede il tokenizer; coverage = i range decoder annunciati tassellano `[0, num_layers)`.

**Fuori scope:** libp2p nativo / hole-punching (futuro, rimuove il coordinator); failover su nodo caduto e store-and-forward durevole (Parte 3); multi-coordinator; auth.

---

## File Structure

```
pyproject.toml                  # + websockets
synapse/net/
  framing.py                    # pack()/unpack() — header+payload in un frame
  node_exec.py                  # NodeState + handle_request() (esecuzione hop, testabile)
  node.py                       # run_node() — client WS verso il coordinator
  discovery.py                  # build_chain() — topologia+coverage dal registry
  coordinator.py                # create_coordinator_app() — WS /node, /registry, POST /infer
synapse/cli.py                  # + coordinator ; serve --coordinator ; infer --coordinator
tests/
  test_framing.py               # round-trip (veloce)
  test_discovery.py             # build_chain (veloce)
  test_node_exec.py             # handle_request greedy == reference (slow)
  test_coordinator_e2e.py       # coordinator + 2 nodi reali, /infer == reference (slow)
  test_cli_coordinator.py       # `synapse infer --coordinator` end-to-end (slow)
docs/
  examples/coordinator.md       # quickstart NAT/internet
```

---

## Task 1: framing (header + payload in un frame)

**Files:** create `synapse/net/framing.py`, `tests/test_framing.py`.

- [ ] **Step 1: test `tests/test_framing.py`**
```python
from synapse.net.framing import pack, unpack


def test_roundtrip_header_and_payload():
    header = {"op": "decode", "block_key": "0-12", "job_id": "j1", "req_id": "r3"}
    payload = b"\x00\x01\x02binarydata"
    header2, payload2 = unpack(pack(header, payload))
    assert header2 == header
    assert payload2 == payload


def test_roundtrip_empty_payload():
    header2, payload2 = unpack(pack({"op": "end"}))
    assert header2 == {"op": "end"}
    assert payload2 == b""
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/synapse/.venv/bin/python -m pytest tests/test_framing.py -v` → ImportError.

- [ ] **Step 3: implementa `synapse/net/framing.py`**
```python
import json
import struct


def pack(header: dict, payload: bytes = b"") -> bytes:
    """Un frame = uint32 big-endian (lunghezza header JSON) + header + payload."""
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload


def unpack(data: bytes):
    """Inverso di pack(). Ritorna (header: dict, payload: bytes)."""
    n = struct.unpack(">I", data[:4])[0]
    header = json.loads(data[4:4 + n].decode("utf-8"))
    return header, data[4 + n:]
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_framing.py -v` → 2 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/framing.py tests/test_framing.py && git commit -m "feat(net): framing header+payload per il relay WebSocket"
```

---

## Task 2: discovery `build_chain` (topologia + coverage dal registry)

**Files:** create `synapse/net/discovery.py`, `tests/test_discovery.py`.

- [ ] **Step 1: test `tests/test_discovery.py`**
```python
from synapse.net.discovery import build_chain


def _reg(**nodes):
    return nodes


def test_build_chain_two_nodes_full_coverage():
    reg = {
        "A": {"embed": True, "head": False, "decoders": ["0-12"]},
        "B": {"embed": False, "head": True, "decoders": ["12-24"]},
    }
    chain = build_chain(reg, 24)
    assert chain is not None
    embed_c, decoders, head_c = chain
    assert embed_c == "A"
    assert head_c == "B"
    assert decoders == [("0-12", "A"), ("12-24", "B")]


def test_build_chain_incomplete_coverage_returns_none():
    reg = {"A": {"embed": True, "head": True, "decoders": ["0-12"]}}  # manca 12-24
    assert build_chain(reg, 24) is None


def test_build_chain_missing_embed_returns_none():
    reg = {"B": {"embed": False, "head": True, "decoders": ["0-12", "12-24"]}}
    assert build_chain(reg, 24) is None
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_discovery.py -v` → ImportError.

- [ ] **Step 3: implementa `synapse/net/discovery.py`**
```python
def build_chain(registry: dict, num_layers: int):
    """Dal registry {conn_id: {'embed','head','decoders':[block_key]}} costruisce
    (embed_conn, [(block_key, conn)...], head_conn) che tassella [0, num_layers).
    Ritorna None se la coverage è incompleta o manca embed/head."""
    embed_c = next((cid for cid, s in registry.items() if s.get("embed")), None)
    head_c = next((cid for cid, s in registry.items() if s.get("head")), None)
    if embed_c is None or head_c is None:
        return None

    ranges = []
    for cid, s in registry.items():
        for bk in s.get("decoders", []):
            lo, hi = (int(x) for x in bk.split("-"))
            ranges.append((lo, hi, bk, cid))
    ranges.sort()

    chain = []
    cursor = 0
    for lo, hi, bk, cid in ranges:
        if lo == cursor and hi > cursor:
            chain.append((bk, cid))
            cursor = hi
    if cursor != num_layers:
        return None
    return embed_c, chain, head_c
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_discovery.py -v` → 3 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/discovery.py tests/test_discovery.py && git commit -m "feat(net): build_chain (topologia + coverage dal registry del coordinator)"
```

---

## Task 3: `NodeState` + `handle_request` (esecuzione hop)

**Files:** create `synapse/net/node_exec.py`, `tests/test_node_exec.py`.

- [ ] **Step 1: test `tests/test_node_exec.py`**
```python
import pytest
import torch
from synapse.net.node_exec import NodeState, handle_request
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.topology import StageSpec
from synapse.model.generate import reference_generate


@pytest.mark.slow
def test_handle_request_greedy_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA di NodeState (remap)

    state = NodeState(model, StageSpec(embed=True, head=True, decoders=[(0, 24)]))
    L = ids.shape[1]
    cache_position = torch.arange(L)
    cur = ids
    generated = []
    for step in range(6):
        _, p = handle_request(state, {"op": "embed", "job_id": "j"}, encode_tensors({"input_ids": cur}))
        h = decode_tensors(p)["hidden_states"]
        _, p = handle_request(state, {"op": "decode", "block_key": "0-24", "job_id": "j"},
                              encode_tensors({"hidden_states": h, "cache_position": cache_position}))
        h = decode_tensors(p)["hidden_states"]
        rh, _ = handle_request(state, {"op": "head", "job_id": "j"}, encode_tensors({"hidden_states": h}))
        generated.append(rh["token_id"])
        cur = torch.tensor([[rh["token_id"]]])
        cache_position = torch.tensor([L + step])

    assert generated == reference
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_node_exec.py -m slow -v` → ImportError.

- [ ] **Step 3: implementa `synapse/net/node_exec.py`**
```python
from synapse.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from synapse.net.wire import encode_tensors, decode_tensors


class NodeState:
    """Stato locale di un nodo worker: blocchi serviti + KV-cache per-job."""
    def __init__(self, model, stages):
        self.embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
        self.head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
        self.prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
        self.jobs = {}   # job_id -> {block_key: DecoderBlock}

    def stages_dict(self) -> dict:
        return {"embed": self.embed_block is not None,
                "head": self.head_block is not None,
                "decoders": list(self.prepared.keys())}


def handle_request(state: NodeState, header: dict, payload: bytes):
    """Esegue un hop. Ritorna (resp_header: dict, resp_payload: bytes)."""
    op = header["op"]
    if op == "embed":
        t = decode_tensors(payload)
        h = state.embed_block.run_block(t["input_ids"])
        return {"ok": True}, encode_tensors({"hidden_states": h})
    if op == "decode":
        block_key = header["block_key"]
        job = state.jobs.setdefault(header["job_id"], {})
        block = job.get(block_key)
        if block is None:
            layers, rotary = state.prepared[block_key]
            block = DecoderBlock(layers, rotary)
            job[block_key] = block
        t = decode_tensors(payload)
        h = block.run_block(t["hidden_states"], t["cache_position"])
        return {"ok": True}, encode_tensors({"hidden_states": h})
    if op == "head":
        t = decode_tensors(payload)
        logits = state.head_block.run_block(t["hidden_states"])
        return {"ok": True, "token_id": int(logits[:, -1, :].argmax(-1).item())}, b""
    if op == "end":
        state.jobs.pop(header["job_id"], None)
        return {"ok": True}, b""
    return {"ok": False, "error": f"op sconosciuta: {op}"}, b""
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_node_exec.py -m slow -v` → PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/node_exec.py tests/test_node_exec.py && git commit -m "feat(net): NodeState + handle_request (esecuzione hop per il relay)"
```

---

## Task 4: coordinator + node client + golden distribuito via relay

**Files:** modify `pyproject.toml`; create `synapse/net/coordinator.py`, `synapse/net/node.py`, `tests/test_coordinator_e2e.py`.

- [ ] **Step 1: aggiungi `websockets` a `pyproject.toml`** (lista `dependencies`): `"websockets>=12"`. Poi `cd /Users/alberto/Projects/AI/synapse && .venv/bin/pip install -e ".[dev]"`. (Se la rete è giù e `websockets` non è importabile, BLOCKED.)

- [ ] **Step 2: test `tests/test_coordinator_e2e.py`**
```python
import socket
import threading
import time
import asyncio

import pytest
import httpx
import uvicorn

from synapse.net.coordinator import create_coordinator_app
from synapse.net.node import run_node
from synapse.net.node_exec import NodeState
from synapse.net.topology import StageSpec
from synapse.model.generate import reference_generate


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve_uvicorn(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


def _run_node_thread(ws_url, state):
    def _loop():
        asyncio.run(run_node(ws_url, state))
    threading.Thread(target=_loop, daemon=True).start()


@pytest.mark.slow
def test_two_nodes_via_coordinator_match_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA dei NodeState

    port = _free_port()
    app = create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", num_layers=24, tokenizer=tokenizer)
    server = _serve_uvicorn(app, port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    try:
        _run_node_thread(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
        _run_node_thread(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))

        # attende che entrambi i nodi siano registrati e la coverage sia completa
        with httpx.Client(timeout=30.0) as client:
            for _ in range(200):
                reg = client.get(f"http://127.0.0.1:{port}/registry").json()
                if len(reg["nodes"]) == 2:
                    break
                time.sleep(0.05)
            r = client.post(f"http://127.0.0.1:{port}/infer",
                            json={"prompt": "La capitale dell'Italia è", "max_new_tokens": 6})
            data = r.json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference
    finally:
        server.should_exit = True
```

- [ ] **Step 3: run FAIL** — `... pytest tests/test_coordinator_e2e.py -m slow -v` → ImportError.

- [ ] **Step 4: implementa `synapse/net/coordinator.py`**
```python
import asyncio

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

from synapse.net.framing import pack, unpack
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.discovery import build_chain


def create_coordinator_app(model_id: str, num_layers: int, tokenizer):
    """Coordinator-relay: i nodi si connettono via WS e annunciano gli stage; POST /infer
    guida la generazione relayando ogni hop al nodo giusto."""
    app = FastAPI()
    conns = {}        # conn_id -> {"ws", "stages", "pending": {req_id: Future}}
    counter = {"n": 0}

    def _next_id(prefix):
        counter["n"] += 1
        return f"{prefix}{counter['n']}"

    async def _call(conn_id, header, payload=b""):
        c = conns[conn_id]
        req_id = _next_id("r")
        fut = asyncio.get_event_loop().create_future()
        c["pending"][req_id] = fut
        await c["ws"].send_bytes(pack({**header, "req_id": req_id}, payload))
        return await fut   # (resp_header, resp_payload)

    @app.websocket("/node")
    async def node_ws(ws: WebSocket):
        await ws.accept()
        announce, _ = unpack(await ws.receive_bytes())
        conn_id = _next_id("c")
        conns[conn_id] = {"ws": ws, "stages": announce["stages"], "pending": {}}
        try:
            while True:
                rh, rp = unpack(await ws.receive_bytes())
                fut = conns[conn_id]["pending"].pop(rh.get("req_id"), None)
                if fut is not None and not fut.done():
                    fut.set_result((rh, rp))
        except WebSocketDisconnect:
            conns.pop(conn_id, None)

    @app.get("/registry")
    async def registry():
        return {"num_layers": num_layers,
                "nodes": [{"conn": cid, "stages": c["stages"]} for cid, c in conns.items()]}

    @app.post("/infer")
    async def infer(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 8))
        chain = build_chain({cid: c["stages"] for cid, c in conns.items()}, num_layers)
        if chain is None:
            return {"ok": False, "error": "modello non operativo: coverage incompleta"}
        embed_c, decoders, head_c = chain

        ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = ids.shape[1]
        cache_position = torch.arange(seq_len)
        cur = ids
        tokens = []
        job_id = _next_id("job")
        try:
            for step in range(max_new):
                _, p = await _call(embed_c, {"op": "embed", "job_id": job_id},
                                   encode_tensors({"input_ids": cur}))
                h = decode_tensors(p)["hidden_states"]
                for block_key, cid in decoders:
                    _, p = await _call(cid, {"op": "decode", "block_key": block_key, "job_id": job_id},
                                       encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                    h = decode_tensors(p)["hidden_states"]
                rh, _ = await _call(head_c, {"op": "head", "job_id": job_id},
                                    encode_tensors({"hidden_states": h}))
                tokens.append(rh["token_id"])
                cur = torch.tensor([[rh["token_id"]]])
                cache_position = torch.tensor([seq_len + step])
        finally:
            for cid in {embed_c, head_c, *(c for _, c in decoders)}:
                try:
                    await _call(cid, {"op": "end", "job_id": job_id})
                except Exception:
                    pass
        return {"ok": True, "model": model_id, "prompt": prompt,
                "text": tokenizer.decode(tokens), "tokens": tokens}

    return app
```

- [ ] **Step 5: implementa `synapse/net/node.py`**
```python
import asyncio

import websockets

from synapse.net.framing import pack, unpack
from synapse.net.node_exec import handle_request


async def run_node(coordinator_ws_url: str, state):
    """Si connette (outbound, NAT-friendly) al coordinator, annuncia gli stage e serve
    gli hop relayati. Il calcolo torch gira in un executor per non bloccare il loop."""
    async with websockets.connect(coordinator_ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_event_loop()
        async for message in ws:
            header, payload = unpack(message)
            resp_header, resp_payload = await loop.run_in_executor(
                None, handle_request, state, header, payload)
            await ws.send(pack({**resp_header, "req_id": header.get("req_id")}, resp_payload))
```

- [ ] **Step 6: run PASS** — `... pytest tests/test_coordinator_e2e.py -m slow -v` → PASS (2 nodi via relay == reference). Se i nodi non si registrano, aumenta l'attesa; verifica che `websockets.connect` usi l'URL `ws://`.

- [ ] **Step 7: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add pyproject.toml synapse/net/coordinator.py synapse/net/node.py tests/test_coordinator_e2e.py && git commit -m "feat(net): coordinator-relay + node WS client (golden distribuito via relay)"
```

---

## Task 5: CLI `coordinator` + `serve --coordinator` + `infer --coordinator`

**Files:** modify `synapse/cli.py`; create `tests/test_cli_coordinator.py`.

- [ ] **Step 1: test `tests/test_cli_coordinator.py`**
```python
import json
import socket
import threading
import time
import asyncio

import pytest
import uvicorn

from typer.testing import CliRunner
from synapse.cli import app as cli_app
from synapse.net.coordinator import create_coordinator_app
from synapse.net.node import run_node
from synapse.net.node_exec import NodeState
from synapse.net.topology import StageSpec
from synapse.model.generate import reference_generate

runner = CliRunner()


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve_uvicorn(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


def _run_node_thread(ws_url, state):
    threading.Thread(target=lambda: asyncio.run(run_node(ws_url, state)), daemon=True).start()


@pytest.mark.slow
def test_cli_infer_via_coordinator(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    port = _free_port()
    server = _serve_uvicorn(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    try:
        _run_node_thread(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
        _run_node_thread(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))
        import httpx
        with httpx.Client(timeout=30.0) as client:
            for _ in range(200):
                if len(client.get(f"http://127.0.0.1:{port}/registry").json()["nodes"]) == 2:
                    break
                time.sleep(0.05)

        result = runner.invoke(cli_app, ["--json", "infer", "--coordinator", f"http://127.0.0.1:{port}",
                                         "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        server.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_cli_coordinator.py -m slow -v` → FAIL (manca `--coordinator`).

- [ ] **Step 3: modifica `synapse/cli.py`**

Aggiungi import vicino agli altri `from synapse.net...`:
```python
from synapse.net.node_exec import NodeState
from synapse.net.node import run_node
from synapse.net.coordinator import create_coordinator_app
from synapse.model.loader import model_config_dims
```
(se `model_config_dims` è già importato, non duplicarlo.)

Aggiungi il comando `coordinator` (dopo `serve`):
```python
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
```

Modifica `serve` per supportare la modalità coordinator (connessione in uscita). Aggiungi un'opzione `--coordinator` e, se presente, avvia il client nodo invece del server HTTP. Sostituisci il corpo di `serve` con:
```python
@app.command()
def serve(
    stages: str = typer.Option(..., "--stages", help="Stage serviti, es. 'embed,decoder:0-12'"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    coordinator: str = typer.Option(None, "--coordinator", help="URL WS del coordinator (es. ws://host:9000/node). Se assente, avvia un server HTTP diretto (modalità LAN/topologia statica)."),
    host: str = typer.Option("0.0.0.0", "--host", help="[modalità diretta] host di ascolto"),
    port: int = typer.Option(8001, "--port", help="[modalità diretta] porta di ascolto"),
):
    """Avvia un nodo worker. Con --coordinator si connette in uscita (NAT-friendly);
    senza, espone un BlockServer HTTP diretto (richiede raggiungibilità diretta)."""
    try:
        spec = parse_stages(stages)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    try:
        model, tokenizer = load_full_model(model_id, DTYPE, DEVICE)
        model.eval()
    except Exception as e:
        _fail("serve", "MODEL_LOAD_FAILED", str(e))

    if coordinator:
        import asyncio
        state = NodeState(model, spec)
        typer.echo(f"synapse serve→coordinator {coordinator}: stages={stages} (model={model_id})", err=True)
        asyncio.run(run_node(coordinator, state))
    else:
        import uvicorn
        fastapi_app = create_app(model, tokenizer, spec)
        typer.echo(f"synapse serve (diretto): stages={stages} su http://{host}:{port}", err=True)
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
```

Modifica `infer` per supportare `--coordinator` (client sottile) accanto al `--topology` esistente. Sostituisci la firma e l'inizio di `infer` così che `--topology` e `--coordinator` siano alternativi:
```python
@app.command()
def infer(
    prompt: str = typer.Option(..., "--prompt", help="Prompt ('-' legge da stdin)"),
    topology: str = typer.Option(None, "--topology", help="[modalità diretta] file JSON di topologia statica"),
    coordinator: str = typer.Option(None, "--coordinator", help="[modalità coordinator] URL HTTP del coordinator"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
):
    """Inferenza distribuita: via coordinator (--coordinator) o topologia statica (--topology)."""
    import httpx
    prompt = _read_prompt(prompt)
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
    if not topology:
        _fail("infer", "USAGE_ERROR", "specificare --coordinator oppure --topology", exit_code=2)
    # ---- modalità topologia statica (Parte 1) ----
    from transformers import AutoTokenizer
    from synapse.net.orchestrator import distributed_generate
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
    _emit_ok("infer", {"model": topo.model, "prompt": prompt, **result}, human=result["text"])
```
> Nota: questo sostituisce il comando `infer` di Parte 1 mantenendone la modalità `--topology`. Assicurati che gli import duplicati (`AutoTokenizer`, `distributed_generate`) non siano già a livello modulo in modo conflittuale; vanno bene come import locali.

- [ ] **Step 4: run PASS** — `... pytest tests/test_cli_coordinator.py -m slow -v` → PASS. Verifica anche `cd /Users/alberto/Projects/AI/synapse && .venv/bin/synapse --help` elenca `coordinator`.

- [ ] **Step 5: assicura che i test di Parte 1 (`tests/test_cli_infer.py`) passino ancora** (modalità `--topology` invariata):
`... pytest tests/test_cli_infer.py -m slow -v` → PASS.

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/cli.py tests/test_cli_coordinator.py && git commit -m "feat(cli): comando coordinator + serve/infer in modalità coordinator (NAT-friendly)"
```

---

## Task 6: quickstart NAT/internet + suite + ROADMAP

**Files:** create `docs/examples/coordinator.md`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: crea `docs/examples/coordinator.md`** con il quickstart:
```markdown
# Quickstart — coordinator (LAN e internet, senza VPN)

I nodi worker si connettono **in uscita** al coordinator: funzionano dietro NAT senza port-forwarding. Solo il **coordinator** deve essere raggiungibile (IP pubblico / VPS / un solo port-forward).

```bash
# 1) Coordinator (su una macchina raggiungibile dagli altri; es. IP pubblico 203.0.113.5)
synapse coordinator --model Qwen/Qwen2.5-0.5B-Instruct --port 9000

# 2) Nodo A (qualsiasi rete, dietro NAT) — embedding + primi 12 layer
synapse serve --coordinator ws://203.0.113.5:9000/node --stages "embed,decoder:0-12"

# 3) Nodo B (altra rete) — ultimi 12 layer + head
synapse serve --coordinator ws://203.0.113.5:9000/node --stages "decoder:12-24,head"

# 4) Inferenza (client sottile, da qualunque rete)
synapse --json infer --coordinator http://203.0.113.5:9000 --prompt "La capitale dell'Italia è"
```

Il coordinator calcola la coverage: finché embed + tutti i range decoder + head non sono coperti, `infer` risponde `NOT_OPERATIONAL`. In LAN, metti il coordinator su un IP locale. Con una VPN, usa l'IP della VPN.
```

- [ ] **Step 2: aggiungi un richiamo nel `README.md`** (dopo la sezione "Quickstart multi-nodo"): una frase + link a `docs/examples/coordinator.md` per la modalità coordinator (LAN/internet senza VPN).

- [ ] **Step 3: aggiorna `docs/ROADMAP.md`** — sotto "Discovery & Routing" segna discovery automatica via coordinator-relay come fatta (link a [ADR-0002](./decisions/ADR-0002-connettivita-nat.md) e a questo piano), e aggiorna la riga "Ultimo aggiornamento". Nota: failover e libp2p nativo restano da fare.

- [ ] **Step 4: suite completa** — `/Users/alberto/Projects/AI/synapse/.venv/bin/python -m pytest -q -p no:warnings` → tutti PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add docs/examples/coordinator.md README.md docs/ROADMAP.md && git commit -m "docs: quickstart coordinator (LAN/internet senza VPN); ROADMAP discovery"
```

---

## Self-Review

**Coverage (ADR-0002 + PRD Parte 2):** discovery automatica via registry (Task 4 coordinator + Task 2 build_chain) ✓; transport NAT-friendly outbound WS (Task 4 node.py) ✓; coverage gate ✓; CLI `coordinator`/`serve --coordinator`/`infer --coordinator` ✓; modalità diretta Parte 1 conservata ✓; quickstart internet ✓. Failover/libp2p esplicitamente fuori scope.

**Placeholder scan:** nessun TODO/TBD; codice completo.

**Type consistency:** `pack/unpack`, `NodeState.stages_dict()`, `handle_request(state, header, payload)->(header,payload)`, `build_chain(registry, num_layers)->(embed,decoders,head)|None`, `create_coordinator_app(model_id, num_layers, tokenizer)`, `run_node(ws_url, state)` coerenti tra i task. Reference catturato PRIMA di `NodeState` (che chiama `prepare_decoder_block`, muta `layer_idx`).
```
