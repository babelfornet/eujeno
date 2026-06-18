# Part 3 — Failover & ridondanza (coordinator) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Se un nodo cade durante un job, il traffico si **reindirizza automaticamente** su un holder ridondante e la generazione completa correttamente — nella modalità coordinator.

**Architecture:** [ADR-0001](../decisions/ADR-0001-implementation-forks.md) Fork C (failover = re-dispatch su holder ridondante). Implementazione di Milestone 0: il coordinator guida la generazione; se un hop fallisce (nodo disconnesso → la Future pendente solleva `ConnectionError`), esclude quel nodo, ricalcola la catena dai nodi rimasti (`build_chain(..., exclude)`) e **riavvia la generazione da capo** con un nuovo `job_id`, fino a K failover. Richiede **ridondanza**: ≥2 nodi che servono lo stesso blocco. (Il re-dispatch per-hop con replay del prefisso e lo store-and-forward durevole su SQLite restano un approfondimento successivo; riavviare da capo è semplice, corretto e accettabile sotto il framing async.)

**Tech Stack:** Python · l'esistente `axyn/net/{coordinator,discovery,node,node_exec}.py` · asyncio · pytest.

**Fuori scope:** failover per-hop con replay del prefisso (preserva il progresso); store-and-forward durevole SQLite; failover nella modalità P2P diretta (follow-up); failover del coordinator stesso.

---

## File Structure

```
axyn/net/discovery.py        # MODIFICA: build_chain(..., exclude=None)
axyn/net/coordinator.py      # MODIFICA: failover loop (escludi nodo caduto, ricalcola, riavvia)
tests/
  test_discovery.py             # MODIFICA: + test build_chain con exclude e ridondanza
  test_failover_e2e.py          # NUOVO: nodo che crasha mid-hop -> completa via ridondante (slow)
docs/examples/coordinator.md    # MODIFICA: nota su ridondanza + failover
docs/ROADMAP.md
```

---

## Task 1: `build_chain(exclude)` — consapevole di ridondanza

**Files:** modify `axyn/net/discovery.py`; modify `tests/test_discovery.py`.

- [ ] **Step 1: aggiungi i test in coda a `tests/test_discovery.py`**
```python
def test_build_chain_excludes_failed_node_uses_redundant():
    reg = {
        "A": {"embed": True, "head": False, "decoders": ["0-12"]},
        "B": {"embed": False, "head": True, "decoders": ["12-24"]},
        "C": {"embed": False, "head": True, "decoders": ["12-24"]},  # ridondante con B
    }
    # senza esclusioni: copertura ok
    assert build_chain(reg, 24) is not None
    # escludendo B, deve usare C per 12-24 e head
    chain = build_chain(reg, 24, exclude={"B"})
    assert chain is not None
    _, decoders, head = chain
    assert ("12-24", "C") in decoders
    assert head == "C"


def test_build_chain_exclude_breaks_coverage_returns_none():
    reg = {
        "A": {"embed": True, "head": True, "decoders": ["0-12"]},
        "B": {"embed": False, "head": False, "decoders": ["12-24"]},
    }
    assert build_chain(reg, 24, exclude={"B"}) is None   # senza B manca 12-24
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_discovery.py -v` → TypeError (build_chain non accetta `exclude`).

- [ ] **Step 3: modifica `build_chain` in `axyn/net/discovery.py`**

Sostituisci la firma e l'inizio della funzione `build_chain` con:
```python
def build_chain(stages_by_url: dict, num_layers: int, exclude=None):
    """Da {url: {'embed','head','decoders':[block_key]}} costruisce
    (embed_url, [(block_key, url)...], head_url) che tassella [0, num_layers),
    ignorando gli id in `exclude`. Ritorna None se la coverage è incompleta."""
    exclude = exclude or set()
    items = {u: s for u, s in stages_by_url.items() if u not in exclude}
    embed = next((u for u, s in items.items() if s.get("embed")), None)
    head = next((u for u, s in items.items() if s.get("head")), None)
    if embed is None or head is None:
        return None
    ranges = []
    for u, s in items.items():
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
(È identica a prima ma con il filtro `exclude` applicato all'inizio.)

- [ ] **Step 4: run PASS** — `... pytest tests/test_discovery.py -v` → tutti passati (i 4 esistenti + 2 nuovi).

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/discovery.py tests/test_discovery.py && git commit -m "feat(net): build_chain con exclude (ridondanza-aware per il failover)"
```

---

## Task 2: failover nel coordinator + e2e con nodo che crasha

**Files:** modify `axyn/net/coordinator.py`; create `tests/test_failover_e2e.py`.

- [ ] **Step 1: scrivi `tests/test_failover_e2e.py`**
```python
import socket
import threading
import time
import asyncio

import pytest
import httpx
import uvicorn
import websockets

from axyn.net.coordinator import create_coordinator_app
from axyn.net.node import run_node
from axyn.net.node_exec import NodeState, handle_request
from axyn.net.framing import pack, unpack
from axyn.net.topology import StageSpec
from axyn.model.generate import reference_generate


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


async def _run_flaky_node(ws_url, state):
    """Annuncia, serve gli hop, ma CHIUDE la connessione alla prima 'decode' (crash simulato)."""
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_running_loop()
        async for message in ws:
            header, payload = unpack(message)
            if header["op"] == "decode":
                await ws.close()
                return
            rh, rp = await loop.run_in_executor(None, handle_request, state, header, payload)
            await ws.send(pack({**rh, "req_id": header.get("req_id")}, rp))


def _thread(coro_factory):
    threading.Thread(target=lambda: asyncio.run(coro_factory()), daemon=True).start()


def _registry_count(client, base):
    return len(client.get(f"{base}/registry").json()["nodes"])


@pytest.mark.slow
def test_failover_completes_via_redundant_node(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=30.0) as client:
            # ordine di connessione deterministico: A, poi B (flaky), poi C (ridondante)
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)]))))
            for _ in range(200):
                if _registry_count(client, base) == 1:
                    break
                time.sleep(0.05)
            _thread(lambda: _run_flaky_node(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)]))))
            for _ in range(200):
                if _registry_count(client, base) == 2:
                    break
                time.sleep(0.05)
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)]))))
            for _ in range(200):
                if _registry_count(client, base) == 3:
                    break
                time.sleep(0.05)

            # B (flaky) viene scelto per 12-24/head, crasha alla decode, failover su C
            r = client.post(f"{base}/infer", json={"prompt": "La capitale dell'Italia è", "max_new_tokens": 6})
            data = r.json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference
        assert data["failovers"] >= 1     # ha effettivamente fatto failover
    finally:
        server.should_exit = True
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_failover_e2e.py -m slow -v`. Expected: FAIL (`data` non ha `failovers`, oppure l'infer si appende/erra perché manca la logica di failover).

- [ ] **Step 3: modifica `axyn/net/coordinator.py`**

(a) Aggiungi una costante e una eccezione vicino all'inizio del modulo (dopo gli import):
```python
MAX_FAILOVERS = 5


class _NodeFailure(Exception):
    def __init__(self, conn_id):
        super().__init__(conn_id)
        self.conn_id = conn_id
```

(b) Modifica `_call` perché segnali il nodo fallito: sostituisci il corpo di `_call` con:
```python
    async def _call(conn_id, header, payload=b""):
        if conn_id not in conns:
            raise _NodeFailure(conn_id)
        c = conns[conn_id]
        req_id = _next_id("r")
        fut = asyncio.get_running_loop().create_future()
        c["pending"][req_id] = fut
        try:
            await c["ws"].send_bytes(pack({**header, "req_id": req_id}, payload))
            return await fut
        except Exception:
            raise _NodeFailure(conn_id)
```

(c) Sostituisci INTERAMENTE l'endpoint `@app.post("/infer")` con una versione con failover che usa una funzione interna di generazione:
```python
    async def _run_generation(chain, prompt, max_new, job_id):
        embed_c, decoders, head_c = chain
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = ids.shape[1]
        cache_position = torch.arange(seq_len)
        cur = ids
        tokens = []
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
        # cleanup best-effort della KV-cache sui nodi usati
        for cid in {embed_c, head_c, *(c for _, c in decoders)}:
            try:
                await _call(cid, {"op": "end", "job_id": job_id})
            except _NodeFailure:
                pass
        return tokens

    @app.post("/infer")
    async def infer(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 8))
        excluded = set()
        last_failed = None
        for attempt in range(MAX_FAILOVERS + 1):
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is None:
                return {"ok": False, "error": "modello non operativo: coverage incompleta",
                        "excluded": sorted(excluded)}
            try:
                tokens = await _run_generation(chain, prompt, max_new, _next_id("job"))
                return {"ok": True, "model": model_id, "prompt": prompt,
                        "text": tokenizer.decode(tokens), "tokens": tokens, "failovers": attempt}
            except _NodeFailure as e:
                excluded.add(e.conn_id)        # escludi il nodo caduto e riprova da capo
                last_failed = e.conn_id
        return {"ok": False, "error": f"troppi failover (ultimo nodo fallito: {last_failed})"}
```
(Rimuovi la vecchia definizione di `infer`; tutto il resto di `create_app` resta invariato.)

- [ ] **Step 4: run PASS** — `... pytest tests/test_failover_e2e.py -m slow -v` → PASS (failover su nodo ridondante, tokens == reference, `failovers >= 1`).
Verifica nessuna regressione: `... pytest tests/test_coordinator_e2e.py tests/test_cli_coordinator.py -m slow -v` → PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/coordinator.py tests/test_failover_e2e.py && git commit -m "feat(net): failover nel coordinator (escludi nodo caduto, ri-instrada su ridondante)"
```

---

## Task 3: docs ridondanza/failover + suite + ROADMAP

**Files:** modify `docs/examples/coordinator.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: aggiungi a `docs/examples/coordinator.md`** una sezione "Ridondanza e failover":
```markdown
## Ridondanza e failover

Avvia **più nodi che servono lo stesso blocco** per la resilienza: se un nodo cade durante un job, il coordinator lo esclude e **riavvia la generazione** sui nodi rimasti (serve almeno un holder per ogni blocco).

```bash
# blocco 12-24 + head serviti da DUE nodi (B e C): se B cade, il job continua su C
axyn serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # nodo B
axyn serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # nodo C (ridondante)
```

La risposta di `infer` include `"failovers": N` (quanti reinstradamenti sono serviti). Se nessun nodo ridondante copre il blocco caduto, `infer` risponde `NOT_OPERATIONAL`.

> Nota: in questo Milestone 0 il failover **riavvia** la generazione da capo (semplice e corretto). Il re-dispatch per-hop con replay del prefisso, che preserva il progresso, è un approfondimento successivo.
```

- [ ] **Step 2: aggiorna `docs/ROADMAP.md`** — sotto "Discovery & Routing", segna il failover (coordinator) come fatto; sotto "Queue & Load Balancing"/Parte 3 nota che store-and-forward durevole + failover per-hop restano da fare. Aggiorna la riga "Ultimo aggiornamento".

- [ ] **Step 3: suite completa** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest -q -p no:warnings` → tutti PASS.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add docs/examples/coordinator.md docs/ROADMAP.md && git commit -m "docs: ridondanza e failover (coordinator); ROADMAP Parte 3"
```

---

## Self-Review

**Coverage (ADR-0001 Fork C, livello Milestone 0):** ridondanza (più holder per blocco) via `build_chain(exclude)` ✓; failover automatico su caduta nodo (escludi + ricalcola + riavvia) ✓; coverage gate quando la ridondanza non basta ✓; e2e con nodo che crasha mid-hop ✓. Store-and-forward durevole e re-dispatch per-hop esplicitamente fuori scope (follow-up).

**Placeholder scan:** nessun TODO/TBD; codice completo.

**Type consistency:** `build_chain(stages, num_layers, exclude=None)`, `_NodeFailure(conn_id)`, `_run_generation(chain, prompt, max_new, job_id)`, `_call` solleva `_NodeFailure` su nodo assente/errore; risposta `infer` con `failovers`. Reference catturato prima dei `NodeState` nel test.
```
