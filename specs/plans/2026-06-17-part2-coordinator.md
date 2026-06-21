# Part 2 — Coordinator-relay + automatic discovery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Distributed inference that works **on LAN and over the internet without a VPN**: worker nodes connect outbound to a publicly reachable **coordinator** that performs **automatic discovery** (block registry) and **routes** activations; `eujeno infer` is a thin client.

**Architecture:** See [ADR-0002](../decisions/ADR-0002-nat-connectivity.md). Milestone 0 coordinator-relay: outbound WebSocket node→coordinator (NAT-friendly); the coordinator holds the registry, computes coverage, and drives the generation loop by relaying each hop to the right node. It reuses the block execution and the safetensors wire from Part 1; only the *transport* changes (WS relay instead of direct POSTs).

**Tech Stack:** Python · FastAPI WebSocket (coordinator) · `websockets` lib (node client) · httpx (infer client) · safetensors · asyncio · the existing `eujeno/net/` and `eujeno/model/`.

**Decisions:** binary framing `4-byte len + JSON header + safetensors payload`; request/response correlation via `req_id` + Future; greedy/argmax; the coordinator owns the tokenizer; coverage = the announced decoder ranges tile `[0, num_layers)`.

**Out of scope:** native libp2p / hole-punching (future, removes the coordinator); failover on a downed node and durable store-and-forward (Part 3); multi-coordinator; auth.

---

## File Structure

```
pyproject.toml                  # + websockets
eujeno/net/
  framing.py                    # pack()/unpack() — header+payload in one frame
  node_exec.py                  # NodeState + handle_request() (hop execution, testable)
  node.py                       # run_node() — WS client toward the coordinator
  discovery.py                  # build_chain() — topology+coverage from the registry
  coordinator.py                # create_coordinator_app() — WS /node, /registry, POST /infer
eujeno/cli.py                  # + coordinator ; serve --coordinator ; infer --coordinator
tests/
  test_framing.py               # round-trip (fast)
  test_discovery.py             # build_chain (fast)
  test_node_exec.py             # handle_request greedy == reference (slow)
  test_coordinator_e2e.py       # coordinator + 2 real nodes, /infer == reference (slow)
  test_cli_coordinator.py       # `eujeno infer --coordinator` end-to-end (slow)
docs/
  examples/coordinator.md       # NAT/internet quickstart
```

---

## Task 1: framing (header + payload in one frame)

**Files:** create `eujeno/net/framing.py`, `tests/test_framing.py`.

- [ ] **Step 1: test `tests/test_framing.py`**
```python
from eujeno.net.framing import pack, unpack


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

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_framing.py -v` → ImportError.

- [ ] **Step 3: implement `eujeno/net/framing.py`**
```python
import json
import struct


def pack(header: dict, payload: bytes = b"") -> bytes:
    """One frame = uint32 big-endian (JSON header length) + header + payload."""
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload


def unpack(data: bytes):
    """Inverse of pack(). Returns (header: dict, payload: bytes)."""
    n = struct.unpack(">I", data[:4])[0]
    header = json.loads(data[4:4 + n].decode("utf-8"))
    return header, data[4 + n:]
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_framing.py -v` → 2 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/framing.py tests/test_framing.py && git commit -m "feat(net): framing header+payload for the WebSocket relay"
```

---

## Task 2: discovery `build_chain` (topology + coverage from the registry)

**Files:** create `eujeno/net/discovery.py`, `tests/test_discovery.py`.

- [ ] **Step 1: test `tests/test_discovery.py`**
```python
from eujeno.net.discovery import build_chain


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
    reg = {"A": {"embed": True, "head": True, "decoders": ["0-12"]}}  # missing 12-24
    assert build_chain(reg, 24) is None


def test_build_chain_missing_embed_returns_none():
    reg = {"B": {"embed": False, "head": True, "decoders": ["0-12", "12-24"]}}
    assert build_chain(reg, 24) is None
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_discovery.py -v` → ImportError.

- [ ] **Step 3: implement `eujeno/net/discovery.py`**
```python
def build_chain(registry: dict, num_layers: int):
    """From the registry {conn_id: {'embed','head','decoders':[block_key]}} builds
    (embed_conn, [(block_key, conn)...], head_conn) that tiles [0, num_layers).
    Returns None if coverage is incomplete or embed/head is missing."""
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
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/discovery.py tests/test_discovery.py && git commit -m "feat(net): build_chain (topology + coverage from the coordinator registry)"
```

---

## Task 3: `NodeState` + `handle_request` (hop execution)

**Files:** create `eujeno/net/node_exec.py`, `tests/test_node_exec.py`.

- [ ] **Step 1: test `tests/test_node_exec.py`**
```python
import pytest
import torch
from eujeno.net.node_exec import NodeState, handle_request
from eujeno.net.wire import encode_tensors, decode_tensors
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


@pytest.mark.slow
def test_handle_request_greedy_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # BEFORE NodeState (remap)

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

- [ ] **Step 3: implement `eujeno/net/node_exec.py`**
```python
from eujeno.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from eujeno.net.wire import encode_tensors, decode_tensors


class NodeState:
    """Local state of a worker node: served blocks + per-job KV-cache."""
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
    """Executes one hop. Returns (resp_header: dict, resp_payload: bytes)."""
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
    return {"ok": False, "error": f"unknown op: {op}"}, b""
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_node_exec.py -m slow -v` → PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/node_exec.py tests/test_node_exec.py && git commit -m "feat(net): NodeState + handle_request (hop execution for the relay)"
```

---

## Task 4: coordinator + node client + distributed golden via relay

**Files:** modify `pyproject.toml`; create `eujeno/net/coordinator.py`, `eujeno/net/node.py`, `tests/test_coordinator_e2e.py`.

- [ ] **Step 1: add `websockets` to `pyproject.toml`** (`dependencies` list): `"websockets>=12"`. Then `cd /Users/alberto/Projects/AI/eujeno && .venv/bin/pip install -e ".[dev]"`. (If the network is down and `websockets` is not importable, BLOCKED.)

- [ ] **Step 2: test `tests/test_coordinator_e2e.py`**
```python
import socket
import threading
import time
import asyncio

import pytest
import httpx
import uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


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
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # BEFORE the NodeStates

    port = _free_port()
    app = create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", num_layers=24, tokenizer=tokenizer)
    server = _serve_uvicorn(app, port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    try:
        _run_node_thread(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
        _run_node_thread(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))

        # waits until both nodes are registered and coverage is complete
        with httpx.Client(timeout=30.0) as client:
            for _ in range(200):
                reg = client.get(f"http://127.0.0.1:{port}/registry").json()
                if len(reg["nodes"]) == 2:
                    break
                time.sleep(0.05)
            r = client.post(f"http://127.0.0.1:{port}/infer",
                            json={"prompt": "The capital of Italy is", "max_new_tokens": 6})
            data = r.json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference
    finally:
        server.should_exit = True
```

- [ ] **Step 3: run FAIL** — `... pytest tests/test_coordinator_e2e.py -m slow -v` → ImportError.

- [ ] **Step 4: implement `eujeno/net/coordinator.py`**
```python
import asyncio

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

from eujeno.net.framing import pack, unpack
from eujeno.net.wire import encode_tensors, decode_tensors
from eujeno.net.discovery import build_chain


def create_coordinator_app(model_id: str, num_layers: int, tokenizer):
    """Coordinator-relay: nodes connect via WS and announce their stages; POST /infer
    drives generation by relaying each hop to the right node."""
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
            return {"ok": False, "error": "model not operational: incomplete coverage"}
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

- [ ] **Step 5: implement `eujeno/net/node.py`**
```python
import asyncio

import websockets

from eujeno.net.framing import pack, unpack
from eujeno.net.node_exec import handle_request


async def run_node(coordinator_ws_url: str, state):
    """Connects (outbound, NAT-friendly) to the coordinator, announces its stages and serves
    the relayed hops. The torch computation runs in an executor so it does not block the loop."""
    async with websockets.connect(coordinator_ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_event_loop()
        async for message in ws:
            header, payload = unpack(message)
            resp_header, resp_payload = await loop.run_in_executor(
                None, handle_request, state, header, payload)
            await ws.send(pack({**resp_header, "req_id": header.get("req_id")}, resp_payload))
```

- [ ] **Step 6: run PASS** — `... pytest tests/test_coordinator_e2e.py -m slow -v` → PASS (2 nodes via relay == reference). If the nodes do not register, increase the wait; verify that `websockets.connect` uses the `ws://` URL.

- [ ] **Step 7: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add pyproject.toml eujeno/net/coordinator.py eujeno/net/node.py tests/test_coordinator_e2e.py && git commit -m "feat(net): coordinator-relay + node WS client (distributed golden via relay)"
```

---

## Task 5: CLI `coordinator` + `serve --coordinator` + `infer --coordinator`

**Files:** modify `eujeno/cli.py`; create `tests/test_cli_coordinator.py`.

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
from eujeno.cli import app as cli_app
from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate

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
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
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
                                         "--prompt", "The capital of Italy is", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        server.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_cli_coordinator.py -m slow -v` → FAIL (`--coordinator` missing).

- [ ] **Step 3: modify `eujeno/cli.py`**

Add imports near the other `from eujeno.net...`:
```python
from eujeno.net.node_exec import NodeState
from eujeno.net.node import run_node
from eujeno.net.coordinator import create_coordinator_app
from eujeno.model.loader import model_config_dims
```
(if `model_config_dims` is already imported, do not duplicate it.)

Add the `coordinator` command (after `serve`):
```python
@app.command()
def coordinator(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Model ID (for tokenizer + num_layers)"),
    host: str = typer.Option("0.0.0.0", "--host", help="Listen host"),
    port: int = typer.Option(9000, "--port", help="Listen port"),
):
    """Start the coordinator-relay (must be reachable by the nodes)."""
    import uvicorn
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        num_layers = model_config_dims(model_id)["num_layers"]
    except Exception as e:
        _fail("coordinator", "MODEL_LOAD_FAILED", str(e))
    coord_app = create_coordinator_app(model_id, num_layers, tokenizer)
    typer.echo(f"eujeno coordinator: model={model_id} layers={num_layers} on http://{host}:{port}", err=True)
    uvicorn.run(coord_app, host=host, port=port, log_level="info")
```

Modify `serve` to support coordinator mode (outbound connection). Add a `--coordinator` option and, if present, start the node client instead of the HTTP server. Replace the body of `serve` with:
```python
@app.command()
def serve(
    stages: str = typer.Option(..., "--stages", help="Served stages, e.g. 'embed,decoder:0-12'"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    coordinator: str = typer.Option(None, "--coordinator", help="Coordinator WS URL (e.g. ws://host:9000/node). If absent, starts a direct HTTP server (LAN/static-topology mode)."),
    host: str = typer.Option("0.0.0.0", "--host", help="[direct mode] listen host"),
    port: int = typer.Option(8001, "--port", help="[direct mode] listen port"),
):
    """Start a worker node. With --coordinator it connects outbound (NAT-friendly);
    without it, exposes a direct HTTP BlockServer (requires direct reachability)."""
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
        typer.echo(f"eujeno serve→coordinator {coordinator}: stages={stages} (model={model_id})", err=True)
        asyncio.run(run_node(coordinator, state))
    else:
        import uvicorn
        fastapi_app = create_app(model, tokenizer, spec)
        typer.echo(f"eujeno serve (direct): stages={stages} on http://{host}:{port}", err=True)
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
```

Modify `infer` to support `--coordinator` (thin client) alongside the existing `--topology`. Replace the signature and the start of `infer` so that `--topology` and `--coordinator` are mutually exclusive:
```python
@app.command()
def infer(
    prompt: str = typer.Option(..., "--prompt", help="Prompt ('-' reads from stdin)"),
    topology: str = typer.Option(None, "--topology", help="[direct mode] static-topology JSON file"),
    coordinator: str = typer.Option(None, "--coordinator", help="[coordinator mode] coordinator HTTP URL"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Number of tokens to generate"),
):
    """Distributed inference: via coordinator (--coordinator) or static topology (--topology)."""
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
            _fail("infer", "NOT_OPERATIONAL", body.get("error", "coordinator not ready"))
        _emit_ok("infer", body, human=body["text"])
        return
    if not topology:
        _fail("infer", "USAGE_ERROR", "specify --coordinator or --topology", exit_code=2)
    # ---- static topology mode (Part 1) ----
    from transformers import AutoTokenizer
    from eujeno.net.orchestrator import distributed_generate
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
    _emit_ok("infer", {"model": topo.model, "prompt": prompt, **result}, human=result["text"])
```
> Note: this replaces the Part 1 `infer` command while preserving its `--topology` mode. Make sure the duplicate imports (`AutoTokenizer`, `distributed_generate`) are not already at module level in a conflicting way; they are fine as local imports.

- [ ] **Step 4: run PASS** — `... pytest tests/test_cli_coordinator.py -m slow -v` → PASS. Also verify that `cd /Users/alberto/Projects/AI/eujeno && .venv/bin/eujeno --help` lists `coordinator`.

- [ ] **Step 5: ensure the Part 1 tests (`tests/test_cli_infer.py`) still pass** (`--topology` mode unchanged):
`... pytest tests/test_cli_infer.py -m slow -v` → PASS.

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py tests/test_cli_coordinator.py && git commit -m "feat(cli): coordinator command + serve/infer in coordinator mode (NAT-friendly)"
```

---

## Task 6: NAT/internet quickstart + suite + ROADMAP

**Files:** create `docs/examples/coordinator.md`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: create `docs/examples/coordinator.md`** with the quickstart:
```markdown
# Quickstart — coordinator (LAN and internet, without a VPN)

Worker nodes connect **outbound** to the coordinator: they work behind NAT without port-forwarding. Only the **coordinator** has to be reachable (public IP / VPS / a single port-forward).

```bash
# 1) Coordinator (on a machine reachable by the others; e.g. public IP 203.0.113.5)
eujeno coordinator --model Qwen/Qwen2.5-0.5B-Instruct --port 9000

# 2) Node A (any network, behind NAT) — embedding + first 12 layers
eujeno serve --coordinator ws://203.0.113.5:9000/node --stages "embed,decoder:0-12"

# 3) Node B (another network) — last 12 layers + head
eujeno serve --coordinator ws://203.0.113.5:9000/node --stages "decoder:12-24,head"

# 4) Inference (thin client, from any network)
eujeno --json infer --coordinator http://203.0.113.5:9000 --prompt "The capital of Italy is"
```

The coordinator computes coverage: until embed + all decoder ranges + head are covered, `infer` responds `NOT_OPERATIONAL`. On a LAN, put the coordinator on a local IP. With a VPN, use the VPN IP.
```

- [ ] **Step 2: add a pointer in `README.md`** (after the "Multi-node quickstart" section): one sentence + a link to `docs/examples/coordinator.md` for the coordinator mode (LAN/internet without a VPN).

- [ ] **Step 3: update `docs/ROADMAP.md`** — under "Discovery & Routing" mark automatic discovery via coordinator-relay as done (link to [ADR-0002](../decisions/ADR-0002-nat-connectivity.md) and to this plan), and update the "Last updated" line. Note: failover and native libp2p remain to be done.

- [ ] **Step 4: full suite** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest -q -p no:warnings` → all PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add docs/examples/coordinator.md README.md docs/ROADMAP.md && git commit -m "docs: coordinator quickstart (LAN/internet without a VPN); ROADMAP discovery"
```

---

## Self-Review

**Coverage (ADR-0002 + Part 2 PRD):** automatic discovery via registry (Task 4 coordinator + Task 2 build_chain) ✓; NAT-friendly outbound WS transport (Task 4 node.py) ✓; coverage gate ✓; CLI `coordinator`/`serve --coordinator`/`infer --coordinator` ✓; Part 1 direct mode preserved ✓; internet quickstart ✓. Failover/libp2p explicitly out of scope.

**Placeholder scan:** no TODO/TBD; code complete.

**Type consistency:** `pack/unpack`, `NodeState.stages_dict()`, `handle_request(state, header, payload)->(header,payload)`, `build_chain(registry, num_layers)->(embed,decoders,head)|None`, `create_coordinator_app(model_id, num_layers, tokenizer)`, `run_node(ws_url, state)` consistent across the tasks. Reference captured BEFORE `NodeState` (which calls `prepare_decoder_block`, mutating `layer_idx`).
```
