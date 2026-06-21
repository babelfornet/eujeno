# Part 1 Networking — distributed inference over HTTP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the split inference pipeline **across multiple processes/machines via HTTP**, exactly reproducing the generation of the whole model, launchable from the `eujeno` CLI (`serve` + `infer`) on 2-3 nodes.

**Architecture:** Milestone 0 of [ADR-0001](../decisions/ADR-0001-implementation-forks.md): an **orchestrator** (entry node) drives autoregressive generation by calling **BlockServers** (FastAPI) via HTTP; activations travel as **safetensors bytes**. Each BlockServer hosts one or more *stages* (`embed`, `decoder:lo-hi`, `head`), keeps the **per-job KV-cache in memory**, and exposes stateless endpoints for embed/head and stateful ones for decode. The topology (which URL serves which stage) is a **static JSON file** in this slice; the DHT discovery that self-organizes the nodes comes in Part 2.

**Tech Stack:** Python · FastAPI + uvicorn (server) · httpx (client) · safetensors (wire) · the existing `eujeno/model/` (loader, blocks, generate) · pytest.

**Decisions in this slice:**
- **Loading:** each node loads the whole model but serves only its stages (simple and correct for the small PoC model). Real partial-loading = a later optimization.
- **Sampling:** greedy (argmax). The `head` node returns the `token_id` directly (saves bandwidth vs returning the logits).
- **Determinism:** fp32/CPU like the foundation, so distributed == `reference_generate`.

**Out of scope (upcoming slices):** discovery/DHT (Part 2), durable store-and-forward + failover (Part 3), real partial-loading, batching/concurrency, authentication.

---

## File Structure

```
pyproject.toml                  # MODIFY: + fastapi, uvicorn, httpx
eujeno/
  model/blocks.py               # MODIFY: + prepare_decoder_block()
  net/
    __init__.py                 # NEW (empty)
    wire.py                     # NEW: encode_tensors/decode_tensors (safetensors)
    topology.py                 # NEW: parse_stages (serve) + Topology/load_topology (infer)
    server.py                   # NEW: create_app() FastAPI + per-job state
    orchestrator.py             # NEW: distributed_generate() + run_server_in_thread()
  cli.py                        # MODIFY: + serve, infer commands
tests/
  test_wire.py                  # round-trip (fast)
  test_topology.py              # parsing (fast)
  test_prepare_block.py         # prepare_decoder_block (slow)
  test_server.py                # one app with all stages via TestClient (slow)
  test_orchestrator.py          # 2 servers in thread, distributed == reference (slow)
  test_cli_infer.py             # `eujeno infer` against 2 servers (slow)
docs/
  examples/topology.localhost.json   # NEW: example topology
```

---

## Task 1: dependencies + `net` package + wire

**Files:** modify `pyproject.toml`; create `eujeno/net/__init__.py`, `eujeno/net/wire.py`, `tests/test_wire.py`.

- [ ] **Step 1: add dependencies in `pyproject.toml`**

In the `[project] dependencies` list add:
```toml
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
```
Then reinstall:
```bash
cd /Users/alberto/Projects/AI/eujeno && .venv/bin/pip install -e ".[dev]"
```
(If the network is blocked and the packages cannot be installed, report BLOCKED.)

- [ ] **Step 2: create `eujeno/net/__init__.py`** (empty file).

- [ ] **Step 3: write the test `tests/test_wire.py`**

```python
import torch
from eujeno.net.wire import encode_tensors, decode_tensors


def test_roundtrip_preserves_tensors_and_dtype():
    tensors = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "hidden_states": torch.randn(1, 3, 8, dtype=torch.float32),
    }
    back = decode_tensors(encode_tensors(tensors))
    assert torch.equal(back["input_ids"], tensors["input_ids"])
    assert back["input_ids"].dtype == torch.long
    assert torch.equal(back["hidden_states"], tensors["hidden_states"])
```

- [ ] **Step 4: run it to see it fail**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_wire.py -v`
Expected: ImportError on `eujeno.net.wire`.

- [ ] **Step 5: implement `eujeno/net/wire.py`**

```python
import safetensors.torch


def encode_tensors(tensors: dict) -> bytes:
    """Serialize a name->Tensor dict into safetensors bytes (for the HTTP body)."""
    return safetensors.torch.save({k: v.contiguous() for k, v in tensors.items()})


def decode_tensors(data: bytes) -> dict:
    """Deserialize safetensors bytes into a name->Tensor dict."""
    return safetensors.torch.load(data)
```

- [ ] **Step 6: run it to see it pass**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_wire.py -v`
Expected: 1 passed.

- [ ] **Step 7: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add pyproject.toml eujeno/net/__init__.py eujeno/net/wire.py tests/test_wire.py && git commit -m "feat(net): HTTP dependencies + safetensors wire for activations"
```

---

## Task 2: topology (`parse_stages` + `Topology`)

**Files:** create `eujeno/net/topology.py`, `tests/test_topology.py`.

- [ ] **Step 1: write `tests/test_topology.py`**

```python
import pytest
from eujeno.net.topology import parse_stages, StageSpec, Topology, load_topology


def test_parse_stages_all_kinds():
    s = parse_stages("embed,decoder:0-12,head")
    assert s.embed is True
    assert s.head is True
    assert s.decoders == [(0, 12)]


def test_parse_stages_multiple_decoders():
    s = parse_stages("decoder:0-8,decoder:8-16")
    assert s.embed is False and s.head is False
    assert s.decoders == [(0, 8), (8, 16)]


def test_parse_stages_rejects_garbage():
    with pytest.raises(ValueError):
        parse_stages("frobnicate")


def test_load_topology_resolves_stages():
    topo = load_topology({
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "embed": "http://a:1",
        "decoders": [{"block": "0-12", "url": "http://a:1"}, {"block": "12-24", "url": "http://b:2"}],
        "head": "http://b:2",
    })
    assert topo.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert topo.embed == "http://a:1"
    assert topo.head == "http://b:2"
    assert topo.decoders == [("0-12", "http://a:1"), ("12-24", "http://b:2")]
    assert set(topo.all_urls()) == {"http://a:1", "http://b:2"}
```

- [ ] **Step 2: run it to see it fail**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: ImportError on `eujeno.net.topology`.

- [ ] **Step 3: implement `eujeno/net/topology.py`**

```python
from dataclasses import dataclass, field


@dataclass
class StageSpec:
    """Which stages a node serves (for `eujeno serve`)."""
    embed: bool = False
    head: bool = False
    decoders: list = field(default_factory=list)   # list[tuple[int, int]]


def parse_stages(spec: str) -> StageSpec:
    """Parse a string like 'embed,decoder:0-12,head' into a StageSpec."""
    out = StageSpec()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "embed":
            out.embed = True
        elif token == "head":
            out.head = True
        elif token.startswith("decoder:"):
            rng = token[len("decoder:"):]
            try:
                lo, hi = rng.split("-")
                out.decoders.append((int(lo), int(hi)))
            except ValueError:
                raise ValueError(f"invalid decoder range: {token!r} (expected decoder:LO-HI)")
        else:
            raise ValueError(f"unrecognized stage: {token!r}")
    return out


@dataclass
class Topology:
    """stage->URL map for distributed inference (for `eujeno infer`)."""
    model: str
    embed: str
    head: str
    decoders: list   # list[tuple[block_key, url]]

    def all_urls(self) -> list:
        seen = []
        for url in [self.embed, *[u for _, u in self.decoders], self.head]:
            if url not in seen:
                seen.append(url)
        return seen


def load_topology(data: dict) -> Topology:
    """Build a Topology from a dict (e.g. loaded from JSON)."""
    decoders = [(d["block"], d["url"]) for d in data["decoders"]]
    return Topology(model=data["model"], embed=data["embed"], head=data["head"], decoders=decoders)
```

- [ ] **Step 4: run it to see it pass**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: 4 passed.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/topology.py tests/test_topology.py && git commit -m "feat(net): stage parsing + Topology model for distributed inference"
```

---

## Task 3: `prepare_decoder_block` (shared layers, per-job cache)

> The server shares the layer modules across jobs but keeps a separate KV-cache per job. We therefore need a way to prepare the layers (slice + remap `layer_idx` to local indices) ONCE, and then create a per-job `DecoderBlock` (with its own cache) on top of those layers.

**Files:** modify `eujeno/model/blocks.py`; create `tests/test_prepare_block.py`.

- [ ] **Step 1: write `tests/test_prepare_block.py`**

```python
import pytest
import torch
from eujeno.model.blocks import prepare_decoder_block, DecoderBlock


@pytest.mark.slow
def test_prepare_returns_local_indexed_layers(full_model):
    model, _ = full_model
    layers, rotary = prepare_decoder_block(model, 0, 12)
    assert len(layers) == 12
    assert [layer.self_attn.layer_idx for layer in layers] == list(range(12))   # local indices 0..11
    # a DecoderBlock built on top runs without errors
    block = DecoderBlock(layers, rotary)
    h = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float32)
    out = block.run_block(h, torch.arange(3))
    assert out.shape == h.shape
```

- [ ] **Step 2: run it to see it fail**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_prepare_block.py -m slow -v`
Expected: ImportError/AttributeError on `prepare_decoder_block`.

- [ ] **Step 3: implement in `eujeno/model/blocks.py`**

Add at the end of the file:
```python
def prepare_decoder_block(model, lo: int, hi: int):
    """Prepare the decoder layers [lo, hi) to be served: slice them and remap
    layer_idx to local 0-based indices (ONCE). Returns (layers, rotary_emb).
    Build a DecoderBlock(layers, rotary_emb) PER JOB to get separate caches.

    WARNING: mutates layer.self_attn.layer_idx like split_into_blocks. Capture
    any references to the whole model BEFORE calling this function."""
    inner = model.model
    layers = inner.layers[lo:hi]
    for local_idx, layer in enumerate(layers):
        layer.self_attn.layer_idx = local_idx
    return layers, inner.rotary_emb
```

- [ ] **Step 4: run it to see it pass**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_prepare_block.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/model/blocks.py tests/test_prepare_block.py && git commit -m "feat(model): prepare_decoder_block (shared layers, per-job DecoderBlock cache)"
```

---

## Task 4: `BlockServer` (FastAPI)

**Files:** create `eujeno/net/server.py`, `tests/test_server.py`.

- [ ] **Step 1: write `tests/test_server.py`**

```python
import pytest
import torch
from fastapi.testclient import TestClient
from eujeno.net.wire import encode_tensors, decode_tensors
from eujeno.net.topology import StageSpec
from eujeno.net.server import create_app
from eujeno.model.generate import reference_generate


@pytest.mark.slow
def test_single_node_serving_all_stages_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # BEFORE create_app (remap)

    app = create_app(model, tokenizer, StageSpec(embed=True, head=True, decoders=[(0, 24)]))
    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True

    # greedy loop via HTTP (a single node serving all stages)
    L = ids.shape[1]
    cache_position = torch.arange(L)
    cur_ids = ids
    generated = []
    for step in range(6):
        r = client.post("/embed", params={"job_id": "j"}, content=encode_tensors({"input_ids": cur_ids}))
        h = decode_tensors(r.content)["hidden_states"]
        r = client.post("/decode/0-24", params={"job_id": "j"},
                        content=encode_tensors({"hidden_states": h, "cache_position": cache_position}))
        h = decode_tensors(r.content)["hidden_states"]
        r = client.post("/head", params={"job_id": "j"}, content=encode_tensors({"hidden_states": h}))
        token_id = r.json()["token_id"]
        generated.append(token_id)
        cur_ids = torch.tensor([[token_id]])
        cache_position = torch.tensor([L + step])

    assert generated == reference
```

- [ ] **Step 2: run it to see it fail**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_server.py -m slow -v`
Expected: ImportError on `eujeno.net.server`.

- [ ] **Step 3: implement `eujeno/net/server.py`**

```python
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from eujeno.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from eujeno.net.wire import encode_tensors, decode_tensors

_OCTET = "application/octet-stream"


def create_app(model, tokenizer, stages):
    """Create the FastAPI app of a BlockServer that serves the given `stages`, on top of an
    ALREADY-loaded `model` (shared across jobs in this process)."""
    app = FastAPI()
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}   # job_id -> {block_key: DecoderBlock}  (per-job KV-cache)

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "model": getattr(model.config, "_name_or_path", "?"),
            "stages": {"embed": embed_block is not None, "head": head_block is not None,
                       "decoders": list(prepared.keys())},
        }

    @app.post("/embed")
    async def embed(job_id: str, request: Request):
        if embed_block is None:
            return JSONResponse({"error": "this node does not serve the embed stage"}, status_code=400)
        t = decode_tensors(await request.body())
        h = embed_block.run_block(t["input_ids"])
        return Response(encode_tensors({"hidden_states": h}), media_type=_OCTET)

    @app.post("/decode/{block_key}")
    async def decode(block_key: str, job_id: str, request: Request):
        if block_key not in prepared:
            return JSONResponse({"error": f"block {block_key} not served"}, status_code=400)
        t = decode_tensors(await request.body())
        job = jobs.setdefault(job_id, {})
        block = job.get(block_key)
        if block is None:
            layers, rotary = prepared[block_key]
            block = DecoderBlock(layers, rotary)   # own cache per (job, block)
            job[block_key] = block
        h = block.run_block(t["hidden_states"], t["cache_position"])
        return Response(encode_tensors({"hidden_states": h}), media_type=_OCTET)

    @app.post("/head")
    async def head(job_id: str, request: Request):
        if head_block is None:
            return JSONResponse({"error": "this node does not serve the head stage"}, status_code=400)
        t = decode_tensors(await request.body())
        logits = head_block.run_block(t["hidden_states"])
        token_id = int(logits[:, -1, :].argmax(-1).item())
        return JSONResponse({"token_id": token_id})

    @app.delete("/job/{job_id}")
    async def end_job(job_id: str):
        jobs.pop(job_id, None)
        return {"ok": True}

    return app
```

- [ ] **Step 4: run it to see it pass**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_server.py -m slow -v`
Expected: PASS (the tokens generated via HTTP match the reference).

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/server.py tests/test_server.py && git commit -m "feat(net): FastAPI BlockServer (embed/decode/head, per-job KV-cache)"
```

---

## Task 5: orchestrator + distributed golden on 2 nodes

**Files:** create `eujeno/net/orchestrator.py`, `tests/test_orchestrator.py`.

- [ ] **Step 1: write `tests/test_orchestrator.py`**

```python
import socket
import threading
import time

import pytest
import httpx
import uvicorn

from eujeno.net.topology import StageSpec, Topology
from eujeno.net.server import create_app
from eujeno.net.orchestrator import distributed_generate
from eujeno.model.generate import reference_generate


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):           # wait for startup (max ~10s)
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "the uvicorn server did not start"
    return server


@pytest.mark.slow
def test_two_node_distributed_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # BEFORE the create_app calls

    p1, p2 = _free_port(), _free_port()
    app1 = create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]))
    app2 = create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]))
    s1, s2 = _serve(app1, p1), _serve(app2, p2)
    try:
        topo = Topology(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            embed=f"http://127.0.0.1:{p1}",
            head=f"http://127.0.0.1:{p2}",
            decoders=[("0-12", f"http://127.0.0.1:{p1}"), ("12-24", f"http://127.0.0.1:{p2}")],
        )
        with httpx.Client(timeout=60.0) as client:
            result = distributed_generate(topo, "The capital of Italy is", 6, client, tokenizer)
        assert result["tokens"] == reference
        assert isinstance(result["text"], str) and result["text"]
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: run it to see it fail**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_orchestrator.py -m slow -v`
Expected: ImportError on `eujeno.net.orchestrator`.

- [ ] **Step 3: implement `eujeno/net/orchestrator.py`**

```python
import torch

from eujeno.net.wire import encode_tensors, decode_tensors


def distributed_generate(topology, prompt: str, max_new_tokens: int, client, tokenizer,
                         job_id: str = "job") -> dict:
    """Entry node (Milestone 0): drives autoregressive greedy generation by calling
    the topology's BlockServers via HTTP. Returns {'text', 'tokens'}."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    cache_position = torch.arange(seq_len)
    cur_ids = ids
    tokens = []
    try:
        for step in range(max_new_tokens):
            r = client.post(f"{topology.embed}/embed", params={"job_id": job_id},
                            content=encode_tensors({"input_ids": cur_ids}))
            r.raise_for_status()
            h = decode_tensors(r.content)["hidden_states"]

            for block_key, url in topology.decoders:
                r = client.post(f"{url}/decode/{block_key}", params={"job_id": job_id},
                                content=encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                r.raise_for_status()
                h = decode_tensors(r.content)["hidden_states"]

            r = client.post(f"{topology.head}/head", params={"job_id": job_id},
                            content=encode_tensors({"hidden_states": h}))
            r.raise_for_status()
            token_id = r.json()["token_id"]

            tokens.append(token_id)
            cur_ids = torch.tensor([[token_id]])
            cache_position = torch.tensor([seq_len + step])
    finally:
        for url in topology.all_urls():       # free the per-job KV-cache on the nodes
            try:
                client.delete(f"{url}/job/{job_id}")
            except Exception:
                pass

    return {"text": tokenizer.decode(tokens), "tokens": tokens}
```

- [ ] **Step 4: run it to see it pass**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_orchestrator.py -m slow -v`
Expected: PASS — distributed inference on 2 real nodes (uvicorn) matches the reference.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/orchestrator.py tests/test_orchestrator.py && git commit -m "feat(net): distributed orchestrator (golden on 2 real nodes)"
```

---

## Task 6: CLI commands `serve` and `infer`

**Files:** modify `eujeno/cli.py`; create `tests/test_cli_infer.py`.

- [ ] **Step 1: write `tests/test_cli_infer.py`**

```python
import json
import socket
import threading
import time

import pytest
import uvicorn

from typer.testing import CliRunner
from eujeno.cli import app as cli_app
from eujeno.net.topology import StageSpec
from eujeno.net.server import create_app
from eujeno.model.generate import reference_generate

runner = CliRunner()


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


@pytest.mark.slow
def test_cli_infer_against_two_nodes(full_model, tmp_path):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    p1, p2 = _free_port(), _free_port()
    s1 = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)])), p1)
    s2 = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)])), p2)
    try:
        topo = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "embed": f"http://127.0.0.1:{p1}",
            "decoders": [{"block": "0-12", "url": f"http://127.0.0.1:{p1}"},
                         {"block": "12-24", "url": f"http://127.0.0.1:{p2}"}],
            "head": f"http://127.0.0.1:{p2}",
        }
        topo_file = tmp_path / "topo.json"
        topo_file.write_text(json.dumps(topo))

        result = runner.invoke(cli_app, ["--json", "infer", "--topology", str(topo_file),
                                         "--prompt", "The capital of Italy is", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: run it to see it fail**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_cli_infer.py -m slow -v`
Expected: FAIL (`infer` command does not exist).

- [ ] **Step 3: implement in `eujeno/cli.py`**

Add the imports near the other `from eujeno...` ones:
```python
import json as _json2   # (if _json already exists as json, reuse _json; do NOT redefine)
from eujeno.net.topology import parse_stages, load_topology
from eujeno.net.server import create_app
from eujeno.net.orchestrator import distributed_generate
```
> Note: the `cli.py` module already imports `json` as `_json`. To read the topology file use `_json.loads(...)`. Do NOT add a second json import; remove the `import json as _json2` line if you already have `_json`.

Add the two commands (after `selfcheck`, before `schema`):
```python
@app.command()
def serve(
    stages: str = typer.Option(..., "--stages", help="Served stages, e.g. 'embed,decoder:0-12'"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    host: str = typer.Option("0.0.0.0", "--host", help="Listen host"),
    port: int = typer.Option(8001, "--port", help="Listen port"),
):
    """Start a BlockServer that hosts the given stages (long-running process)."""
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
    typer.echo(f"eujeno serve: stages={stages} on http://{host}:{port}  (model={model_id})", err=True)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command()
def infer(
    topology: str = typer.Option(..., "--topology", help="Path to the topology JSON file"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt ('-' reads from stdin)"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Number of tokens to generate"),
):
    """Run distributed inference over a topology of BlockServers."""
    import httpx
    from transformers import AutoTokenizer

    prompt = _read_prompt(prompt)
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
```

- [ ] **Step 4: run it to see it pass**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_cli_infer.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py tests/test_cli_infer.py && git commit -m "feat(cli): serve (BlockServer) and infer (distributed inference) commands"
```

---

## Task 7: example topology + multi-node quickstart + suite

**Files:** create `docs/examples/topology.localhost.json`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: create `docs/examples/topology.localhost.json`**

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "embed": "http://127.0.0.1:8001",
  "decoders": [
    {"block": "0-12", "url": "http://127.0.0.1:8001"},
    {"block": "12-24", "url": "http://127.0.0.1:8002"}
  ],
  "head": "http://127.0.0.1:8002"
}
```

- [ ] **Step 2: add a "Multi-node quickstart" section to `README.md`**

Insert it before the "## Documentation" section:
```markdown
## Multi-node quickstart (PoC)

Distributed inference of a model across 2 nodes (here on localhost; on a LAN replace the IPs in the topology file).

```bash
pip install -e .

# Node A (serves the embedding + first 12 layers)
eujeno serve --stages "embed,decoder:0-12" --port 8001

# Node B (serves the last 12 layers + the head) — another terminal/machine
eujeno serve --stages "decoder:12-24,head" --port 8002

# Entry: runs inference across the two nodes
eujeno --json infer --topology docs/examples/topology.localhost.json --prompt "The capital of Italy is"
```

On 3 machines: start one `eujeno serve` per node with different layer ranges, copy `topology.localhost.json` filling in the **real IP:port** of each node, and run `eujeno infer` pointing at that file. All machines must be able to reach each other over the network (LAN/VPN) and will have downloaded the model from Hugging Face on first start.
```

- [ ] **Step 3: update `docs/ROADMAP.md`**

Under "Peer Node" in Phase 1, check off the network transport:
```markdown
  - [x] Network transport (FastAPI + safetensors) + distributed orchestrator (Milestone 0) — `serve`/`infer` commands, distributed golden on 2 nodes
```
and update the "Last updated" line with the date and a note.

- [ ] **Step 4: run the ENTIRE suite**

Run: `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest -q`
Expected: all tests PASS (foundation + CLI + net).

- [ ] **Step 5: manual 2-node smoke test (localhost)**

```bash
cd /Users/alberto/Projects/AI/eujeno
.venv/bin/eujeno serve --stages "embed,decoder:0-12" --port 8001 &
.venv/bin/eujeno serve --stages "decoder:12-24,head" --port 8002 &
sleep 60   # wait for the model to load on both
.venv/bin/eujeno --json infer --topology docs/examples/topology.localhost.json --prompt "The capital of Italy is" --max-new-tokens 8
kill %1 %2
```
Expected: JSON envelope with a plausible `data.text` (e.g. mentions Roma).

- [ ] **Step 6: commit**

```bash
cd /Users/alberto/Projects/AI/eujeno && git add docs/examples/topology.localhost.json README.md docs/ROADMAP.md && git commit -m "docs: multi-node quickstart + example topology; ROADMAP network transport"
```

---

## Self-Review (performed by the plan author)

**Spec coverage (PRD Part 1 §transport + ADR Milestone 0):**
- HTTP transport of safetensors activations → Task 1 (wire) + Task 4 (server) ✓
- Per-stage execution (embed/decoder/head) with per-job KV-cache → Task 3 (prepare) + Task 4 (server) ✓
- Orchestrator-driven entry node (Milestone 0) → Task 5 ✓
- Distributed golden (== whole model) → Task 4 (single-node) + Task 5 (2 nodes) ✓
- CLI `serve`/`infer` (single words) → Task 6 ✓
- Static topology → Task 2 + Task 7 (example) ✓
- Quickstart runnable across multiple nodes → Task 7 ✓

**Placeholder scan:** no TODO/TBD; code complete. (Only note: in Task 6 the `json as _json2` import is explicitly NOT to be used — the instruction is to reuse `_json`, already present in cli.py.)

**Type consistency:** `parse_stages -> StageSpec(embed,head,decoders)`, `Topology(model,embed,head,decoders).all_urls()`, `create_app(model, tokenizer, stages)`, `distributed_generate(topology, prompt, max_new_tokens, client, tokenizer, job_id)`, `prepare_decoder_block(model, lo, hi) -> (layers, rotary)` used consistently across the tasks. The reference (`reference_generate`) is always captured BEFORE `create_app`/`prepare_decoder_block` (which mutate `layer_idx`), as per the foundation.
```
