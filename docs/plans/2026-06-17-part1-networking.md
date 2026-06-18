# Part 1 Networking — distributed inference over HTTP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eseguire la pipeline di inferenza splittata **attraverso più processi/macchine via HTTP**, riproducendo esattamente la generazione del modello intero, lanciabile dalla CLI `axyn` (`serve` + `infer`) su 2-3 nodi.

**Architecture:** Milestone 0 dell'[ADR-0001](../decisions/ADR-0001-implementation-forks.md): un **orchestrator** (entry node) guida la generazione autoregressiva chiamando dei **BlockServer** (FastAPI) via HTTP; le attivazioni viaggiano come **safetensors bytes**. Ogni BlockServer ospita uno o più *stage* (`embed`, `decoder:lo-hi`, `head`), mantiene la **KV-cache per-job in memoria**, ed espone endpoint stateless per embed/head e stateful per decode. La topologia (quale URL serve quale stage) è un **file JSON statico** in questo slice; la discovery DHT che auto-organizza i nodi arriva in Parte 2.

**Tech Stack:** Python · FastAPI + uvicorn (server) · httpx (client) · safetensors (wire) · l'esistente `axyn/model/` (loader, blocks, generate) · pytest.

**Decisioni di questo slice:**
- **Caricamento:** ogni nodo carica il modello intero ma serve solo i suoi stage (semplice e corretto per il modello piccolo del PoC). Partial-loading reale = ottimizzazione successiva.
- **Sampling:** greedy (argmax). Il nodo `head` ritorna direttamente il `token_id` (risparmia banda vs ritornare i logits).
- **Determinismo:** fp32/CPU come la foundation, così il distribuito == `reference_generate`.

**Fuori scope (prossimi slice):** discovery/DHT (Parte 2), store-and-forward durevole + failover (Parte 3), partial-loading reale, batching/concorrenza, autenticazione.

---

## File Structure

```
pyproject.toml                  # MODIFICA: + fastapi, uvicorn, httpx
axyn/
  model/blocks.py               # MODIFICA: + prepare_decoder_block()
  net/
    __init__.py                 # NUOVO (vuoto)
    wire.py                     # NUOVO: encode_tensors/decode_tensors (safetensors)
    topology.py                 # NUOVO: parse_stages (serve) + Topology/load_topology (infer)
    server.py                   # NUOVO: create_app() FastAPI + stato per-job
    orchestrator.py             # NUOVO: distributed_generate() + run_server_in_thread()
  cli.py                        # MODIFICA: + comandi serve, infer
tests/
  test_wire.py                  # round-trip (veloce)
  test_topology.py              # parsing (veloce)
  test_prepare_block.py         # prepare_decoder_block (slow)
  test_server.py                # un app con tutti gli stage via TestClient (slow)
  test_orchestrator.py          # 2 server in thread, distribuito == reference (slow)
  test_cli_infer.py             # `axyn infer` contro 2 server (slow)
docs/
  examples/topology.localhost.json   # NUOVO: topologia di esempio
```

---

## Task 1: dipendenze + `net` package + wire

**Files:** modify `pyproject.toml`; create `axyn/net/__init__.py`, `axyn/net/wire.py`, `tests/test_wire.py`.

- [ ] **Step 1: aggiungi dipendenze in `pyproject.toml`**

Nella lista `[project] dependencies` aggiungi:
```toml
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
```
Poi reinstalla:
```bash
cd /Users/alberto/Projects/AI/axyn && .venv/bin/pip install -e ".[dev]"
```
(Se la rete è bloccata e i pacchetti non sono installabili, riporta BLOCKED.)

- [ ] **Step 2: crea `axyn/net/__init__.py`** (file vuoto).

- [ ] **Step 3: scrivi il test `tests/test_wire.py`**

```python
import torch
from axyn.net.wire import encode_tensors, decode_tensors


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

- [ ] **Step 4: esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_wire.py -v`
Expected: ImportError su `axyn.net.wire`.

- [ ] **Step 5: implementa `axyn/net/wire.py`**

```python
import safetensors.torch


def encode_tensors(tensors: dict) -> bytes:
    """Serializza un dict nome->Tensor in bytes safetensors (per il body HTTP)."""
    return safetensors.torch.save({k: v.contiguous() for k, v in tensors.items()})


def decode_tensors(data: bytes) -> dict:
    """Deserializza bytes safetensors in un dict nome->Tensor."""
    return safetensors.torch.load(data)
```

- [ ] **Step 6: esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_wire.py -v`
Expected: 1 passed.

- [ ] **Step 7: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add pyproject.toml axyn/net/__init__.py axyn/net/wire.py tests/test_wire.py && git commit -m "feat(net): dipendenze HTTP + wire safetensors per le attivazioni"
```

---

## Task 2: topology (`parse_stages` + `Topology`)

**Files:** create `axyn/net/topology.py`, `tests/test_topology.py`.

- [ ] **Step 1: scrivi `tests/test_topology.py`**

```python
import pytest
from axyn.net.topology import parse_stages, StageSpec, Topology, load_topology


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

- [ ] **Step 2: esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: ImportError su `axyn.net.topology`.

- [ ] **Step 3: implementa `axyn/net/topology.py`**

```python
from dataclasses import dataclass, field


@dataclass
class StageSpec:
    """Quali stage serve un nodo (per `axyn serve`)."""
    embed: bool = False
    head: bool = False
    decoders: list = field(default_factory=list)   # list[tuple[int, int]]


def parse_stages(spec: str) -> StageSpec:
    """Parsa una stringa tipo 'embed,decoder:0-12,head' in uno StageSpec."""
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
                raise ValueError(f"range decoder non valido: {token!r} (atteso decoder:LO-HI)")
        else:
            raise ValueError(f"stage non riconosciuto: {token!r}")
    return out


@dataclass
class Topology:
    """Mappa stage->URL per l'inferenza distribuita (per `axyn infer`)."""
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
    """Costruisce una Topology da un dict (es. caricato da JSON)."""
    decoders = [(d["block"], d["url"]) for d in data["decoders"]]
    return Topology(model=data["model"], embed=data["embed"], head=data["head"], decoders=decoders)
```

- [ ] **Step 4: esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_topology.py -v`
Expected: 4 passed.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/topology.py tests/test_topology.py && git commit -m "feat(net): parsing stage + modello Topology per inferenza distribuita"
```

---

## Task 3: `prepare_decoder_block` (layer condivisi, cache per-job)

> Il server condivide i moduli dei layer tra i job ma tiene una KV-cache separata per job. Serve quindi un modo per preparare i layer (slice + remap `layer_idx` a indici locali) UNA volta, e creare poi una `DecoderBlock` per-job (con la sua cache) sopra quei layer.

**Files:** modify `axyn/model/blocks.py`; create `tests/test_prepare_block.py`.

- [ ] **Step 1: scrivi `tests/test_prepare_block.py`**

```python
import pytest
import torch
from axyn.model.blocks import prepare_decoder_block, DecoderBlock


@pytest.mark.slow
def test_prepare_returns_local_indexed_layers(full_model):
    model, _ = full_model
    layers, rotary = prepare_decoder_block(model, 0, 12)
    assert len(layers) == 12
    assert [layer.self_attn.layer_idx for layer in layers] == list(range(12))   # indici locali 0..11
    # una DecoderBlock costruita sopra ci gira senza errori
    block = DecoderBlock(layers, rotary)
    h = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float32)
    out = block.run_block(h, torch.arange(3))
    assert out.shape == h.shape
```

- [ ] **Step 2: esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_prepare_block.py -m slow -v`
Expected: ImportError/AttributeError su `prepare_decoder_block`.

- [ ] **Step 3: implementa in `axyn/model/blocks.py`**

Aggiungi in fondo al file:
```python
def prepare_decoder_block(model, lo: int, hi: int):
    """Prepara i layer decoder [lo, hi) per essere serviti: li affetta e rimappa
    layer_idx a indici locali 0-based (UNA volta). Ritorna (layers, rotary_emb).
    Costruisci una DecoderBlock(layers, rotary_emb) PER JOB per avere cache separate.

    ATTENZIONE: muta layer.self_attn.layer_idx come split_into_blocks. Cattura
    eventuali riferimenti al modello intero PRIMA di chiamare questa funzione."""
    inner = model.model
    layers = inner.layers[lo:hi]
    for local_idx, layer in enumerate(layers):
        layer.self_attn.layer_idx = local_idx
    return layers, inner.rotary_emb
```

- [ ] **Step 4: esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_prepare_block.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/model/blocks.py tests/test_prepare_block.py && git commit -m "feat(model): prepare_decoder_block (layer condivisi, cache DecoderBlock per-job)"
```

---

## Task 4: `BlockServer` (FastAPI)

**Files:** create `axyn/net/server.py`, `tests/test_server.py`.

- [ ] **Step 1: scrivi `tests/test_server.py`**

```python
import pytest
import torch
from fastapi.testclient import TestClient
from axyn.net.wire import encode_tensors, decode_tensors
from axyn.net.topology import StageSpec
from axyn.net.server import create_app
from axyn.model.generate import reference_generate


@pytest.mark.slow
def test_single_node_serving_all_stages_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA di create_app (remap)

    app = create_app(model, tokenizer, StageSpec(embed=True, head=True, decoders=[(0, 24)]))
    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True

    # loop greedy via HTTP (un solo nodo che serve tutti gli stage)
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

- [ ] **Step 2: esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_server.py -m slow -v`
Expected: ImportError su `axyn.net.server`.

- [ ] **Step 3: implementa `axyn/net/server.py`**

```python
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from axyn.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from axyn.net.wire import encode_tensors, decode_tensors

_OCTET = "application/octet-stream"


def create_app(model, tokenizer, stages):
    """Crea l'app FastAPI di un BlockServer che serve gli `stages` dati, sopra un
    `model` GIA' caricato (condiviso tra i job in questo processo)."""
    app = FastAPI()
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}   # job_id -> {block_key: DecoderBlock}  (KV-cache per-job)

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
            return JSONResponse({"error": "questo nodo non serve lo stage embed"}, status_code=400)
        t = decode_tensors(await request.body())
        h = embed_block.run_block(t["input_ids"])
        return Response(encode_tensors({"hidden_states": h}), media_type=_OCTET)

    @app.post("/decode/{block_key}")
    async def decode(block_key: str, job_id: str, request: Request):
        if block_key not in prepared:
            return JSONResponse({"error": f"blocco {block_key} non servito"}, status_code=400)
        t = decode_tensors(await request.body())
        job = jobs.setdefault(job_id, {})
        block = job.get(block_key)
        if block is None:
            layers, rotary = prepared[block_key]
            block = DecoderBlock(layers, rotary)   # cache propria per (job, blocco)
            job[block_key] = block
        h = block.run_block(t["hidden_states"], t["cache_position"])
        return Response(encode_tensors({"hidden_states": h}), media_type=_OCTET)

    @app.post("/head")
    async def head(job_id: str, request: Request):
        if head_block is None:
            return JSONResponse({"error": "questo nodo non serve lo stage head"}, status_code=400)
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

- [ ] **Step 4: esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_server.py -m slow -v`
Expected: PASS (i token generati via HTTP coincidono col riferimento).

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/server.py tests/test_server.py && git commit -m "feat(net): BlockServer FastAPI (embed/decode/head, KV-cache per-job)"
```

---

## Task 5: orchestrator + golden distribuito su 2 nodi

**Files:** create `axyn/net/orchestrator.py`, `tests/test_orchestrator.py`.

- [ ] **Step 1: scrivi `tests/test_orchestrator.py`**

```python
import socket
import threading
import time

import pytest
import httpx
import uvicorn

from axyn.net.topology import StageSpec, Topology
from axyn.net.server import create_app
from axyn.net.orchestrator import distributed_generate
from axyn.model.generate import reference_generate


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
    for _ in range(200):           # attende lo startup (max ~10s)
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "il server uvicorn non è partito"
    return server


@pytest.mark.slow
def test_two_node_distributed_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA dei create_app

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
            result = distributed_generate(topo, "La capitale dell'Italia è", 6, client, tokenizer)
        assert result["tokens"] == reference
        assert isinstance(result["text"], str) and result["text"]
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_orchestrator.py -m slow -v`
Expected: ImportError su `axyn.net.orchestrator`.

- [ ] **Step 3: implementa `axyn/net/orchestrator.py`**

```python
import torch

from axyn.net.wire import encode_tensors, decode_tensors


def distributed_generate(topology, prompt: str, max_new_tokens: int, client, tokenizer,
                         job_id: str = "job") -> dict:
    """Entry node (Milestone 0): guida la generazione greedy autoregressiva chiamando
    i BlockServer della topologia via HTTP. Ritorna {'text', 'tokens'}."""
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
        for url in topology.all_urls():       # libera la KV-cache per-job sui nodi
            try:
                client.delete(f"{url}/job/{job_id}")
            except Exception:
                pass

    return {"text": tokenizer.decode(tokens), "tokens": tokens}
```

- [ ] **Step 4: esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_orchestrator.py -m slow -v`
Expected: PASS — l'inferenza distribuita su 2 nodi reali (uvicorn) coincide col riferimento.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/orchestrator.py tests/test_orchestrator.py && git commit -m "feat(net): orchestrator distribuito (golden su 2 nodi reali)"
```

---

## Task 6: comandi CLI `serve` e `infer`

**Files:** modify `axyn/cli.py`; create `tests/test_cli_infer.py`.

- [ ] **Step 1: scrivi `tests/test_cli_infer.py`**

```python
import json
import socket
import threading
import time

import pytest
import uvicorn

from typer.testing import CliRunner
from axyn.cli import app as cli_app
from axyn.net.topology import StageSpec
from axyn.net.server import create_app
from axyn.model.generate import reference_generate

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
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
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
                                         "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli_infer.py -m slow -v`
Expected: FAIL (comando `infer` inesistente).

- [ ] **Step 3: implementa in `axyn/cli.py`**

Aggiungi gli import vicino agli altri `from axyn...`:
```python
import json as _json2   # (se _json già esiste come json, riusa _json; NON ridefinire)
from axyn.net.topology import parse_stages, load_topology
from axyn.net.server import create_app
from axyn.net.orchestrator import distributed_generate
```
> Nota: il modulo `cli.py` importa già `json` come `_json`. Per leggere il file topologia usa `_json.loads(...)`. NON aggiungere un secondo import di json; rimuovi la riga `import json as _json2` se hai già `_json`.

Aggiungi i due comandi (dopo `selfcheck`, prima di `schema`):
```python
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
    typer.echo(f"axyn serve: stages={stages} su http://{host}:{port}  (model={model_id})", err=True)
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
```

- [ ] **Step 4: esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli_infer.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py tests/test_cli_infer.py && git commit -m "feat(cli): comandi serve (BlockServer) e infer (inferenza distribuita)"
```

---

## Task 7: esempio topologia + quickstart multi-nodo + suite

**Files:** create `docs/examples/topology.localhost.json`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: crea `docs/examples/topology.localhost.json`**

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

- [ ] **Step 2: aggiungi una sezione "Quickstart multi-nodo" a `README.md`**

Inserisci prima della sezione "## Documentazione":
```markdown
## Quickstart multi-nodo (PoC)

Inferenza distribuita di un modello su 2 nodi (qui in localhost; su LAN sostituisci gli IP nel file topologia).

```bash
pip install -e .

# Nodo A (serve embedding + primi 12 layer)
axyn serve --stages "embed,decoder:0-12" --port 8001

# Nodo B (serve gli ultimi 12 layer + la testa) — altro terminale/macchina
axyn serve --stages "decoder:12-24,head" --port 8002

# Entry: esegue l'inferenza attraverso i due nodi
axyn --json infer --topology docs/examples/topology.localhost.json --prompt "La capitale dell'Italia è"
```

Su 3 macchine: avvia un `axyn serve` per nodo con range di layer diversi, copia `topology.localhost.json` mettendo gli **IP:porta reali** di ogni nodo, e lancia `axyn infer` puntando a quel file. Tutte le macchine devono raggiungersi sulla rete (LAN/VPN) e avranno scaricato il modello da Hugging Face al primo avvio.
```

- [ ] **Step 3: aggiorna `docs/ROADMAP.md`**

Sotto "Peer Node" nella Fase 1, spunta il transport di rete:
```markdown
  - [x] Transport di rete (FastAPI + safetensors) + orchestrator distribuito (Milestone 0) — comandi `serve`/`infer`, golden distribuito su 2 nodi
```
e aggiorna la riga "Ultimo aggiornamento" con la data e una nota.

- [ ] **Step 4: esegui l'INTERA suite**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest -q`
Expected: tutti i test PASS (foundation + CLI + net).

- [ ] **Step 5: smoke test manuale a 2 nodi (localhost)**

```bash
cd /Users/alberto/Projects/AI/axyn
.venv/bin/axyn serve --stages "embed,decoder:0-12" --port 8001 &
.venv/bin/axyn serve --stages "decoder:12-24,head" --port 8002 &
sleep 60   # attende il caricamento del modello su entrambi
.venv/bin/axyn --json infer --topology docs/examples/topology.localhost.json --prompt "La capitale dell'Italia è" --max-new-tokens 8
kill %1 %2
```
Expected: envelope JSON con `data.text` plausibile (es. menziona Roma).

- [ ] **Step 6: commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add docs/examples/topology.localhost.json README.md docs/ROADMAP.md && git commit -m "docs: quickstart multi-nodo + topologia di esempio; ROADMAP transport di rete"
```

---

## Self-Review (eseguito dall'autore del piano)

**Spec coverage (PRD Parte 1 §transport + ADR Milestone 0):**
- Transport HTTP attivazioni safetensors → Task 1 (wire) + Task 4 (server) ✓
- Esecuzione per-stage (embed/decoder/head) con KV-cache per-job → Task 3 (prepare) + Task 4 (server) ✓
- Orchestrator-driven entry node (Milestone 0) → Task 5 ✓
- Golden distribuito (== modello intero) → Task 4 (single-node) + Task 5 (2 nodi) ✓
- CLI `serve`/`infer` (parole singole) → Task 6 ✓
- Topologia statica → Task 2 + Task 7 (esempio) ✓
- Quickstart eseguibile su più nodi → Task 7 ✓

**Placeholder scan:** nessun TODO/TBD; codice completo. (Unica nota: in Task 6 l'import `json as _json2` è esplicitamente da NON usare — istruzione di riusare `_json` già presente in cli.py.)

**Type consistency:** `parse_stages -> StageSpec(embed,head,decoders)`, `Topology(model,embed,head,decoders).all_urls()`, `create_app(model, tokenizer, stages)`, `distributed_generate(topology, prompt, max_new_tokens, client, tokenizer, job_id)`, `prepare_decoder_block(model, lo, hi) -> (layers, rotary)` usati coerentemente tra i task. Il riferimento (`reference_generate`) è sempre catturato PRIMA di `create_app`/`prepare_decoder_block` (che mutano `layer_idx`), come da foundation.
```
