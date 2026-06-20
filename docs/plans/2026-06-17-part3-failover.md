# Part 3 — Failover & redundancy (coordinator) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** If a node goes down during a job, traffic is **automatically rerouted** to a redundant holder and generation completes correctly — in coordinator mode.

**Architecture:** [ADR-0001](../decisions/ADR-0001-implementation-forks.md) Fork C (failover = re-dispatch to a redundant holder). Milestone 0 implementation: the coordinator drives generation; if a hop fails (disconnected node → the pending Future raises `ConnectionError`), it excludes that node, recomputes the chain from the remaining nodes (`build_chain(..., exclude)`) and **restarts generation from scratch** with a new `job_id`, up to K failovers. It requires **redundancy**: ≥2 nodes serving the same block. (Per-hop re-dispatch with prefix replay and durable store-and-forward on SQLite remain a later deep-dive; restarting from scratch is simple, correct and acceptable under the async framing.)

**Tech Stack:** Python · the existing `eujeno/net/{coordinator,discovery,node,node_exec}.py` · asyncio · pytest.

**Out of scope:** per-hop failover with prefix replay (preserves progress); durable SQLite store-and-forward; failover in direct P2P mode (follow-up); failover of the coordinator itself.

---

## File Structure

```
eujeno/net/discovery.py        # MODIFY: build_chain(..., exclude=None)
eujeno/net/coordinator.py      # MODIFY: failover loop (exclude downed node, recompute, restart)
tests/
  test_discovery.py             # MODIFY: + test build_chain with exclude and redundancy
  test_failover_e2e.py          # NEW: node crashing mid-hop -> completes via redundant (slow)
docs/examples/coordinator.md    # MODIFY: note on redundancy + failover
docs/ROADMAP.md
```

---

## Task 1: `build_chain(exclude)` — redundancy-aware

**Files:** modify `eujeno/net/discovery.py`; modify `tests/test_discovery.py`.

- [ ] **Step 1: add the tests at the end of `tests/test_discovery.py`**
```python
def test_build_chain_excludes_failed_node_uses_redundant():
    reg = {
        "A": {"embed": True, "head": False, "decoders": ["0-12"]},
        "B": {"embed": False, "head": True, "decoders": ["12-24"]},
        "C": {"embed": False, "head": True, "decoders": ["12-24"]},  # redundant with B
    }
    # without exclusions: coverage ok
    assert build_chain(reg, 24) is not None
    # excluding B, it must use C for 12-24 and head
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
    assert build_chain(reg, 24, exclude={"B"}) is None   # without B, 12-24 is missing
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_discovery.py -v` → TypeError (build_chain does not accept `exclude`).

- [ ] **Step 3: modify `build_chain` in `eujeno/net/discovery.py`**

Replace the signature and the start of the `build_chain` function with:
```python
def build_chain(stages_by_url: dict, num_layers: int, exclude=None):
    """From {url: {'embed','head','decoders':[block_key]}} builds
    (embed_url, [(block_key, url)...], head_url) that tiles [0, num_layers),
    ignoring the ids in `exclude`. Returns None if coverage is incomplete."""
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
(It is identical to before but with the `exclude` filter applied at the start.)

- [ ] **Step 4: run PASS** — `... pytest tests/test_discovery.py -v` → all passed (the 4 existing + 2 new).

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/discovery.py tests/test_discovery.py && git commit -m "feat(net): build_chain with exclude (redundancy-aware for failover)"
```

---

## Task 2: failover in the coordinator + e2e with a crashing node

**Files:** modify `eujeno/net/coordinator.py`; create `tests/test_failover_e2e.py`.

- [ ] **Step 1: write `tests/test_failover_e2e.py`**
```python
import socket
import threading
import time
import asyncio

import pytest
import httpx
import uvicorn
import websockets

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState, handle_request
from eujeno.net.framing import pack, unpack
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


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
    """Announces, serves the hops, but CLOSES the connection on the first 'decode' (simulated crash)."""
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
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=30.0) as client:
            # deterministic connection order: A, then B (flaky), then C (redundant)
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

            # B (flaky) is chosen for 12-24/head, crashes on decode, failover to C
            r = client.post(f"{base}/infer", json={"prompt": "The capital of Italy is", "max_new_tokens": 6})
            data = r.json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference
        assert data["failovers"] >= 1     # it actually performed a failover
    finally:
        server.should_exit = True
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_failover_e2e.py -m slow -v`. Expected: FAIL (`data` has no `failovers`, or the infer hangs/errors because the failover logic is missing).

- [ ] **Step 3: modify `eujeno/net/coordinator.py`**

(a) Add a constant and an exception near the start of the module (after the imports):
```python
MAX_FAILOVERS = 5


class _NodeFailure(Exception):
    def __init__(self, conn_id):
        super().__init__(conn_id)
        self.conn_id = conn_id
```

(b) Modify `_call` so it signals the failed node: replace the body of `_call` with:
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

(c) Replace the ENTIRE `@app.post("/infer")` endpoint with a failover-capable version that uses an internal generation function:
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
        # best-effort cleanup of the KV-cache on the used nodes
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
                return {"ok": False, "error": "model not operational: incomplete coverage",
                        "excluded": sorted(excluded)}
            try:
                tokens = await _run_generation(chain, prompt, max_new, _next_id("job"))
                return {"ok": True, "model": model_id, "prompt": prompt,
                        "text": tokenizer.decode(tokens), "tokens": tokens, "failovers": attempt}
            except _NodeFailure as e:
                excluded.add(e.conn_id)        # exclude the downed node and retry from scratch
                last_failed = e.conn_id
        return {"ok": False, "error": f"too many failovers (last failed node: {last_failed})"}
```
(Remove the old `infer` definition; everything else in `create_app` stays unchanged.)

- [ ] **Step 4: run PASS** — `... pytest tests/test_failover_e2e.py -m slow -v` → PASS (failover to a redundant node, tokens == reference, `failovers >= 1`).
Verify no regression: `... pytest tests/test_coordinator_e2e.py tests/test_cli_coordinator.py -m slow -v` → PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/coordinator.py tests/test_failover_e2e.py && git commit -m "feat(net): failover in the coordinator (exclude downed node, reroute to redundant)"
```

---

## Task 3: redundancy/failover docs + suite + ROADMAP

**Files:** modify `docs/examples/coordinator.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: add to `docs/examples/coordinator.md`** a "Redundancy and failover" section:
```markdown
## Redundancy and failover

Start **multiple nodes serving the same block** for resilience: if a node goes down during a job, the coordinator excludes it and **restarts generation** on the remaining nodes (at least one holder per block is required).

```bash
# block 12-24 + head served by TWO nodes (B and C): if B goes down, the job continues on C
eujeno serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # node B
eujeno serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # node C (redundant)
```

The `infer` response includes `"failovers": N` (how many reroutes were needed). If no redundant node covers the downed block, `infer` responds `NOT_OPERATIONAL`.

> Note: in this Milestone 0, failover **restarts** generation from scratch (simple and correct). Per-hop re-dispatch with prefix replay, which preserves progress, is a later deep-dive.
```

- [ ] **Step 2: update `docs/ROADMAP.md`** — under "Discovery & Routing", mark failover (coordinator) as done; under "Queue & Load Balancing"/Part 3 note that durable store-and-forward + per-hop failover remain to be done. Update the "Last updated" line.

- [ ] **Step 3: full suite** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest -q -p no:warnings` → all PASS.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add docs/examples/coordinator.md docs/ROADMAP.md && git commit -m "docs: redundancy and failover (coordinator); ROADMAP Part 3"
```

---

## Self-Review

**Coverage (ADR-0001 Fork C, Milestone 0 level):** redundancy (multiple holders per block) via `build_chain(exclude)` ✓; automatic failover on node down (exclude + recompute + restart) ✓; coverage gate when redundancy is not enough ✓; e2e with a node crashing mid-hop ✓. Durable store-and-forward and per-hop re-dispatch explicitly out of scope (follow-up).

**Placeholder scan:** no TODO/TBD; complete code.

**Type consistency:** `build_chain(stages, num_layers, exclude=None)`, `_NodeFailure(conn_id)`, `_run_generation(chain, prompt, max_new, job_id)`, `_call` raises `_NodeFailure` on absent node/error; `infer` response with `failovers`. Reference captured before the `NodeState`s in the test.
```
