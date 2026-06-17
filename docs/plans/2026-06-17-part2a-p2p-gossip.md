# Part 2a — P2P puro: discovery via gossip (decentralizzato) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Discovery **automatica e decentralizzata** (nessun server centrale): i nodi `synapse serve` si scoprono via **gossip** tra peer e si auto-annunciano; `synapse infer --peer <qualsiasi-nodo>` costruisce la topologia da solo ed esegue sul **transport diretto** di Parte 1.

**Architecture:** [ADR-0002](../decisions/ADR-0002-connettivita-nat.md) Modalità A. Ogni BlockServer tiene un **Registry** (url→stage, con TTL) e un loop di **gossip pull**: refresh della propria entry + fetch del `/registry` dei seed peer + merge + prune. La coverage e la topologia si calcolano dal registry con `build_chain`. Transport attivazioni = HTTP diretto (`distributed_generate` di Parte 1). Funziona dove i nodi sono mutuamente raggiungibili (LAN/VPN/IP pubblici).

**Tech Stack:** Python · FastAPI (lifespan background task) · httpx (async per gossip, sync per infer) · l'esistente `synapse/net/{server,orchestrator,topology}.py`.

**Fuori scope:** NAT traversal senza VPN (→ Modalità B coordinator, o libp2p futuro); failover/durabilità (Parte 3).

---

## File Structure

```
synapse/net/discovery.py        # NUOVO: Registry (gossip state) + build_chain (coverage)
synapse/net/server.py           # MODIFICA: create_app + Registry, GET /registry, gossip loop (lifespan)
synapse/cli.py                  # MODIFICA: serve --peers/--advertise ; infer --peer
tests/
  test_discovery.py             # Registry + build_chain (veloce)
  test_gossip_e2e.py            # 2 server reali: il registry converge (slow)
  test_infer_peer.py            # infer --peer == reference (slow)
docs/examples/p2p.md            # NUOVO: quickstart P2P puro
```

---

## Task 1: `Registry` + `build_chain` (logica pura)

**Files:** create `synapse/net/discovery.py`, `tests/test_discovery.py`.

- [ ] **Step 1: test `tests/test_discovery.py`**
```python
from synapse.net.discovery import Registry, build_chain


def test_build_chain_full_coverage():
    reg = {
        "http://a": {"embed": True, "head": False, "decoders": ["0-12"]},
        "http://b": {"embed": False, "head": True, "decoders": ["12-24"]},
    }
    chain = build_chain(reg, 24)
    assert chain == ("http://a", [("0-12", "http://a"), ("12-24", "http://b")], "http://b")


def test_build_chain_incomplete_returns_none():
    reg = {"http://a": {"embed": True, "head": True, "decoders": ["0-12"]}}
    assert build_chain(reg, 24) is None


def test_registry_merge_and_prune_with_ttl():
    r = Registry()
    r.upsert("http://a", {"embed": True, "head": False, "decoders": ["0-24"]}, now=100.0, ttl=60.0)
    # merge di un peer appreso
    r.merge({"http://b": {"head": True, "embed": False, "decoders": []}}, now=100.0, ttl=60.0)
    assert set(r.stages_by_url(now=120.0).keys()) == {"http://a", "http://b"}
    # dopo la scadenza (oltre now+ttl) spariscono se non rinfrescati
    r.prune(now=200.0)
    assert r.stages_by_url(now=200.0) == {}


def test_registry_refresh_extends_expiry():
    r = Registry()
    r.upsert("http://a", {"embed": True, "head": True, "decoders": ["0-24"]}, now=100.0, ttl=60.0)
    r.upsert("http://a", {"embed": True, "head": True, "decoders": ["0-24"]}, now=150.0, ttl=60.0)
    assert "http://a" in r.stages_by_url(now=200.0)   # rinfrescata a 150 -> scade a 210
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/synapse/.venv/bin/python -m pytest tests/test_discovery.py -v` → ImportError.

- [ ] **Step 3: implementa `synapse/net/discovery.py`**
```python
class Registry:
    """Stato di discovery decentralizzato: url -> {stages, expiry}. TTL relativo:
    le entry apprese scadono a now+ttl se non rinfrescate dal gossip."""
    def __init__(self):
        self.entries = {}   # url -> {"stages": dict, "expiry": float}

    def upsert(self, url: str, stages: dict, now: float, ttl: float) -> None:
        self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def merge(self, stages_by_url: dict, now: float, ttl: float) -> None:
        for url, stages in stages_by_url.items():
            self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def prune(self, now: float) -> None:
        self.entries = {u: e for u, e in self.entries.items() if e["expiry"] > now}

    def stages_by_url(self, now: float) -> dict:
        return {u: e["stages"] for u, e in self.entries.items() if e["expiry"] > now}


def build_chain(stages_by_url: dict, num_layers: int):
    """Da {url: {'embed','head','decoders':[block_key]}} costruisce
    (embed_url, [(block_key, url)...], head_url) che tassella [0, num_layers).
    Ritorna None se la coverage è incompleta o manca embed/head."""
    embed = next((u for u, s in stages_by_url.items() if s.get("embed")), None)
    head = next((u for u, s in stages_by_url.items() if s.get("head")), None)
    if embed is None or head is None:
        return None
    ranges = []
    for u, s in stages_by_url.items():
        for bk in s.get("decoders", []):
            lo, hi = (int(x) for x in bk.split("-"))
            ranges.append((lo, hi, bk, u))
    ranges.sort()
    chain = []
    cursor = 0
    for lo, hi, bk, u in ranges:
        if lo == cursor and hi > cursor:
            chain.append((bk, u))
            cursor = hi
    if cursor != num_layers:
        return None
    return embed, chain, head
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_discovery.py -v` → 4 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/discovery.py tests/test_discovery.py && git commit -m "feat(net): Registry gossip + build_chain (discovery decentralizzata)"
```

---

## Task 2: gossip nel BlockServer (`/registry` + loop)

**Files:** modify `synapse/net/server.py`; create `tests/test_gossip_e2e.py`.

- [ ] **Step 1: test `tests/test_gossip_e2e.py`**
```python
import socket
import threading
import time

import pytest
import httpx
import uvicorn

from synapse.net.topology import StageSpec
from synapse.net.server import create_app


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
def test_registry_converges_via_gossip(full_model):
    model, tokenizer = full_model
    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    # nodo 1 conosce nodo 2 come seed e viceversa
    app1 = create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                      node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3, ttl=30.0)
    app2 = create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                      node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3, ttl=30.0)
    s1, s2 = _serve(app1, p1), _serve(app2, p2)
    try:
        with httpx.Client(timeout=10.0) as client:
            converged = False
            for _ in range(100):   # ~ alcuni round di gossip
                reg = client.get(f"{u1}/registry").json()
                if set(reg["nodes"].keys()) == {u1, u2}:
                    converged = True
                    break
                time.sleep(0.1)
            assert converged, reg
            assert reg["num_layers"] == 24
            assert reg["nodes"][u2]["head"] is True
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_gossip_e2e.py -m slow -v` → TypeError (create_app non accetta i nuovi kwargs).

- [ ] **Step 3: modifica `synapse/net/server.py`**

Aggiorna gli import in cima:
```python
import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from synapse.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.discovery import Registry
```
Sostituisci la firma e l'inizio di `create_app` per accettare i parametri di gossip (opzionali: senza di essi il comportamento di Parte 1 è invariato) e registrare se stesso + avviare il loop:
```python
def create_app(model, tokenizer, stages, node_url=None, peers=None,
               num_layers=None, gossip_interval=2.0, ttl=30.0):
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}
    own_stages = {"embed": stages.embed, "head": stages.head, "decoders": list(prepared.keys())}
    registry = Registry()
    if node_url:
        registry.upsert(node_url, own_stages, now=time.time(), ttl=ttl)

    async def _gossip_loop():
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                now = time.time()
                if node_url:
                    registry.upsert(node_url, own_stages, now=now, ttl=ttl)
                for peer in (peers or []):
                    try:
                        resp = await client.get(f"{peer}/registry")
                        registry.merge(resp.json().get("nodes", {}), now=now, ttl=ttl)
                    except Exception:
                        pass
                registry.prune(now)
                await asyncio.sleep(gossip_interval)

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(_gossip_loop()) if node_url else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/registry")
    async def get_registry():
        return {"num_layers": num_layers, "model": getattr(model.config, "_name_or_path", "?"),
                "nodes": registry.stages_by_url(time.time())}
```
Il RESTO di `create_app` (gli endpoint `/health`, `/embed`, `/decode/{block_key}`, `/head`, `DELETE /job/{job_id}` e `return app`) resta **identico** a prima — lasciali invariati sotto la definizione di `get_registry`.

- [ ] **Step 4: run PASS** — `... pytest tests/test_gossip_e2e.py -m slow -v` → PASS (il registry converge a entrambi i nodi via gossip).
Verifica che i test di Parte 1 (`tests/test_server.py`, `tests/test_orchestrator.py`, `tests/test_cli_infer.py`) passino ancora (create_app retro-compatibile): `... pytest tests/test_server.py tests/test_orchestrator.py -m slow -v`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/server.py tests/test_gossip_e2e.py && git commit -m "feat(net): gossip discovery nel BlockServer (/registry + loop, retro-compatibile)"
```

---

## Task 3: `synapse serve --peers/--advertise` + `synapse infer --peer`

**Files:** modify `synapse/cli.py`; create `tests/test_infer_peer.py`.

- [ ] **Step 1: test `tests/test_infer_peer.py`**
```python
import json
import socket
import threading
import time

import pytest
import uvicorn

from typer.testing import CliRunner
from synapse.cli import app as cli_app
from synapse.net.topology import StageSpec
from synapse.net.server import create_app
from synapse.model.generate import reference_generate

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
def test_infer_peer_autodiscovers_and_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    s1 = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3), p1)
    s2 = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3), p2)
    try:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            for _ in range(100):
                if set(client.get(f"{u1}/registry").json()["nodes"].keys()) == {u1, u2}:
                    break
                time.sleep(0.1)
        result = runner.invoke(cli_app, ["--json", "infer", "--peer", u1,
                                         "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_infer_peer.py -m slow -v` → FAIL (manca `--peer`).

- [ ] **Step 3: modifica `synapse/cli.py`**

Aggiungi import vicino agli altri `from synapse.net...`:
```python
from synapse.net.discovery import build_chain
from synapse.net.topology import Topology
```
Estendi il comando `serve` con le opzioni di gossip (modalità diretta): aggiungi i parametri e passali a `create_app`. Aggiungi alla firma di `serve` (nel ramo diretto, non-coordinator):
```python
    peers: str = typer.Option(None, "--peers", help="Seed peer per la discovery gossip, separati da virgola (es. http://altro:8001)"),
    advertise: str = typer.Option(None, "--advertise", help="URL con cui questo nodo si annuncia (es. http://IP:8001). Default: http://<host>:<port>"),
    num_layers: int = typer.Option(None, "--num-layers", help="Numero totale di layer del modello (per la coverage). Default: dal config."),
```
e nel ramo diretto (`else:` di `serve`, quello che fa `create_app` + `uvicorn.run`) sostituisci con:
```python
    else:
        import uvicorn
        own_url = advertise or f"http://{host}:{port}"
        seeds = [p.strip() for p in peers.split(",")] if peers else []
        nl = num_layers if num_layers is not None else model_config_dims(model_id)["num_layers"]
        fastapi_app = create_app(model, tokenizer, spec, node_url=own_url, peers=seeds, num_layers=nl)
        typer.echo(f"synapse serve (P2P): stages={stages} su http://{host}:{port} advertise={own_url} peers={seeds}", err=True)
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
```
(`model_config_dims` è già importato in cli.py.)

Estendi il comando `infer` con la modalità `--peer` (auto-discovery via gossip + transport diretto). Aggiungi l'opzione `peer` alla firma di `infer`:
```python
    peer: str = typer.Option(None, "--peer", help="[P2P] URL di un nodo qualsiasi: scopre la topologia via gossip ed esegue diretto"),
```
e in cima al corpo di `infer`, dopo `prompt = _read_prompt(prompt)`, prima del ramo `--topology`, aggiungi il ramo `--peer`:
```python
    if peer:
        import httpx
        from transformers import AutoTokenizer
        from synapse.net.orchestrator import distributed_generate
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
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_infer_peer.py -m slow -v` → PASS. Verifica che i test infer di Parte 1 (`tests/test_cli_infer.py`) passino ancora.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/cli.py tests/test_infer_peer.py && git commit -m "feat(cli): serve --peers/--advertise + infer --peer (P2P puro, auto-discovery)"
```

---

## Task 4: quickstart P2P + suite + ROADMAP

**Files:** create `docs/examples/p2p.md`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: crea `docs/examples/p2p.md`**
```markdown
# Quickstart — P2P puro (decentralizzato, nessun server centrale)

Ogni nodo è uguale: si scoprono via gossip (un seed basta) e l'inferenza va diretta nodo-a-nodo. Richiede che i nodi si raggiungano (LAN, VPN, o IP pubblici). Per NAT-senza-VPN usa invece la modalità coordinator (vedi coordinator.md).

```bash
# Nodo A — embedding + primi 12 layer (primo nodo, nessun seed)
synapse serve --stages "embed,decoder:0-12" --port 8001 --advertise http://192.168.1.10:8001

# Nodo B — ultimi 12 layer + head, conosce A come seed
synapse serve --stages "decoder:12-24,head" --port 8001 \
  --advertise http://192.168.1.11:8001 --peers http://192.168.1.10:8001

# Inferenza: punta a UN nodo qualsiasi; scopre il resto da solo
synapse --json infer --peer http://192.168.1.10:8001 --prompt "La capitale dell'Italia è"
```

Finché la coverage non è completa (embed + tutti i decoder + head), `infer` risponde `NOT_OPERATIONAL`. Aggiungi nodi con range diversi e la rete si compone progressivamente.
```

- [ ] **Step 2: aggiorna `README.md`** — nella sezione Quickstart, distingui **P2P puro** (link `docs/examples/p2p.md`) e **coordinator** (link `docs/examples/coordinator.md`), spiegando in una riga quando usare l'uno o l'altro.

- [ ] **Step 3: aggiorna `docs/ROADMAP.md`** — sotto "Discovery & Routing" segna la discovery P2P via gossip come fatta (link a questo piano e ad [ADR-0002](./decisions/ADR-0002-connettivita-nat.md)); aggiorna la riga "Ultimo aggiornamento". Failover e libp2p restano da fare.

- [ ] **Step 4: suite completa** — `/Users/alberto/Projects/AI/synapse/.venv/bin/python -m pytest -q -p no:warnings` → tutti PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add docs/examples/p2p.md README.md docs/ROADMAP.md && git commit -m "docs: quickstart P2P puro; ROADMAP discovery gossip"
```

---

## Self-Review

**Coverage (ADR-0002 Modalità A):** discovery decentralizzata via gossip (Task 1 Registry + Task 2 loop) ✓; auto-annuncio + coverage gate (build_chain) ✓; transport diretto riusato da Parte 1 ✓; CLI `serve --peers/--advertise` + `infer --peer` ✓; retro-compatibilità modalità statica Parte 1 ✓.

**Placeholder scan:** nessun TODO/TBD; codice completo. `create_app` resta retro-compatibile (nuovi kwargs opzionali).

**Type consistency:** `Registry.upsert/merge/prune/stages_by_url(now, ttl)`, `build_chain(stages_by_url, num_layers)->(embed,decoders,head)|None`, `create_app(..., node_url, peers, num_layers, gossip_interval, ttl)`, `Topology(model, embed, head, decoders)` coerenti. `build_chain` ritorna `(embed_url, [(block_key,url)], head_url)`, consumato per costruire `Topology` e poi `distributed_generate` (Parte 1).
```
