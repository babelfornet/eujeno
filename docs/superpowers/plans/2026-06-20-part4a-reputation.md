# Part 4a — Lightweight Reputation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each connection gets a `reputation` that rises on successful contribution and biases routing toward high-reputation holders (de-prioritizing low-reputation ones), exposed in `/registry`. PRD Part 4 acceptance #1 (rise) + #2 (routing).

**Architecture:** Extend 3d's `build_chain` load post-pass with an optional `reputation` map (reputation primary, load secondary, default-preserving). The coordinator tracks `conns[cid]["reputation"]` (neutral cold-start), rewards the winning chain on success, keeps a best-effort failover penalty hook, threads reputation into `build_chain`, and exposes it in `/registry`.

**Tech Stack:** Python (pure), FastAPI coordinator, pytest. Spec: `docs/superpowers/specs/2026-06-20-part4a-reputation-design.md`.

## Global Constraints

- `build_chain(stages_by_url, num_layers, exclude=None, load=None, reputation=None)` — default (`load=None, reputation=None`) byte-for-byte unchanged; 3d (`load` only) unchanged (reputation term constant). Selection key: `(-R.get(u,0.0), L.get(u,0), order[u])`.
- Reputation: `REP_INITIAL=1.0`, `REP_REWARD=0.5`, `REP_PENALTY=2.0`, clamp `[REP_MIN=0.0, REP_MAX=10.0]`. In-memory per-conn (no persistence in the PoC).
- The failover penalty is best-effort (`if cid in conns`); the disconnect failure mode makes it a no-op (documented) — do not over-test it.

---

### Task 1: `build_chain` reputation-aware selection + unit tests

**Files:**
- Modify: `eujeno/net/discovery.py`
- Test: `tests/test_load_balancing.py` (add reputation tests; reuses the existing `_s` helper)

**Interfaces:**
- Produces: `build_chain(stages_by_url, num_layers, exclude=None, load=None, reputation=None)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_load_balancing.py`:

```python
def test_prefers_higher_reputation_head():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, reputation={"b": 0.0, "c": 5.0})[2] == "c"
    assert build_chain(s, 24, reputation={"b": 5.0, "c": 0.0})[2] == "b"


def test_reputation_is_primary_over_load():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    # c has higher reputation; it wins even though it is more loaded
    assert build_chain(s, 24, load={"b": 0, "c": 9}, reputation={"b": 0.0, "c": 5.0})[2] == "c"


def test_equal_reputation_falls_back_to_load():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"b": 3, "c": 0}, reputation={"b": 1.0, "c": 1.0})[2] == "c"


def test_reputation_none_matches_load_only_path():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    a = build_chain(s, 24, load={"b": 3, "c": 0})
    b = build_chain(s, 24, load={"b": 3, "c": 0}, reputation=None)
    assert a == b


def test_prefers_higher_reputation_decoder_replica():
    s = {"a": _s(embed=True), "b": _s(decoders=["0-24"]), "c": _s(decoders=["0-24"]), "d": _s(head=True)}
    assert build_chain(s, 24, reputation={"b": 0.0, "c": 9.0})[1] == [("0-24", "c")]
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_load_balancing.py -q` → FAIL (no `reputation` kwarg).

- [ ] **Step 3: Implement** in `eujeno/net/discovery.py`. Change the signature:

```python
def build_chain(stages_by_url: dict, num_layers: int, exclude=None, load=None):
```
→
```python
def build_chain(stages_by_url: dict, num_layers: int, exclude=None, load=None, reputation=None):
```

Replace the post-pass block (currently `if load is not None: ...`) with:

```python
    if load is not None or reputation is not None:
        order = {u: i for i, u in enumerate(items)}
        L = load or {}
        R = reputation or {}
        def _least(cands):
            return min(cands, key=lambda u: (-R.get(u, 0.0), L.get(u, 0), order[u]))
        embed = _least([u for u, s in items.items() if s.get("embed")])
        head = _least([u for u, s in items.items() if s.get("head")])
        chain = [(bk, _least([u for u, s in items.items() if bk in s.get("decoders", [])]))
                 for bk, u in chain]
    return embed, chain, head
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_load_balancing.py tests/test_discovery.py -q` → PASS (new reputation tests + existing load/discovery tests green).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/discovery.py tests/test_load_balancing.py
git commit -m "feat(net): build_chain reputation-aware routing (prefer high-reputation holder)"
```

---

### Task 2: coordinator reputation tracking + reward/penalty + registry + e2e

**Files:**
- Modify: `eujeno/net/coordinator.py`
- Test: `tests/test_reputation_e2e.py` (new)

**Interfaces:**
- Consumes: `build_chain(..., reputation=...)` (Task 1).

- [ ] **Step 1: Write the failing e2e** — create `tests/test_reputation_e2e.py`:

```python
import socket, threading, time, asyncio
import pytest, httpx, uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


def _thread(coro_factory):
    threading.Thread(target=lambda: asyncio.run(coro_factory()), daemon=True).start()


@pytest.mark.slow
def test_reputation_rises_on_success_and_is_exposed(full_model, tmp_path):
    model, tokenizer = full_model
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer,
                                           db_path=str(tmp_path / "j.db")), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, head=True, decoders=[(0, 24)]))))
        with httpx.Client(timeout=120.0) as client:
            for _ in range(200):
                if client.get(f"{base}/registry").json()["nodes"]:
                    break
                time.sleep(0.05)
            before = client.get(f"{base}/registry").json()["nodes"][0]
            assert "reputation" in before
            assert before["reputation"] == 1.0   # REP_INITIAL, freshly connected
            r = client.post(f"{base}/infer", json={"prompt": "The capital of France is", "max_new_tokens": 5}).json()
            assert r["ok"] is True
            after = client.get(f"{base}/registry").json()["nodes"][0]
        assert after["reputation"] > 1.0          # rose after a successful generation
    finally:
        server.should_exit = True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_reputation_e2e.py -q` → FAIL (registry has no `reputation`). (@pytest.mark.slow.)

- [ ] **Step 3: Implement in `eujeno/net/coordinator.py`.**

(a) Add constants near `COVERAGE_POLL_INTERVAL = 0.5`:

```python
REP_INITIAL = 1.0
REP_REWARD = 0.5
REP_PENALTY = 2.0
REP_MIN = 0.0
REP_MAX = 10.0
```

(b) Registration — change:
```python
        conns[conn_id] = {"ws": ws, "stages": announce["stages"], "pending": {}, "load": 0}
```
to:
```python
        conns[conn_id] = {"ws": ws, "stages": announce["stages"], "pending": {}, "load": 0, "reputation": REP_INITIAL}
```

(c) `/registry` — change:
```python
                "nodes": [{"conn": cid, "stages": c["stages"], "load": c["load"]} for cid, c in conns.items()]}
```
to:
```python
                "nodes": [{"conn": cid, "stages": c["stages"], "load": c["load"], "reputation": c["reputation"]}
                          for cid, c in conns.items()]}
```

(d) `_await_coverage` — change the `build_chain` call:
```python
            chain = build_chain(stages, num_layers, load={cid: c["load"] for cid, c in conns.items()})
```
to:
```python
            chain = build_chain(stages, num_layers,
                                load={cid: c["load"] for cid, c in conns.items()},
                                reputation={cid: c["reputation"] for cid, c in conns.items()})
```

(e) `_generate_with_failover` — reward on success and penalize (best-effort) on failover. In the `try`, before `return {...}, None`, add the reward; in the `except _NodeFailure`, add the penalty. The success branch becomes:

```python
            try:
                tokens, prompt_len, finish_reason = await _run_generation(
                    chain, prompt, max_new, sampling, _next_id("job"),
                    on_token=lambda pos, tok: _store_safe(store.append_token, job_id, tok, pos),
                    resume_tokens=resume_tokens)
                for cid in chain_conns:
                    if cid in conns:
                        conns[cid]["reputation"] = min(REP_MAX, conns[cid]["reputation"] + REP_REWARD)
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt, "finish_reason": finish_reason}, None
            except _NodeFailure as e:
                excluded.add(e.conn_id)
                last_failed = e.conn_id
                if e.conn_id in conns:  # best-effort penalty hook (no-op when the node disconnected)
                    conns[e.conn_id]["reputation"] = max(REP_MIN, conns[e.conn_id]["reputation"] - REP_PENALTY)
                try:
                    j = store.get_job(job_id)
                    resume_tokens = (j or {}).get("tokens", []) or []
                except Exception:
                    resume_tokens = []
            finally:
                for cid in chain_conns:
                    if cid in conns:
                        conns[cid]["load"] = max(0, conns[cid]["load"] - 1)
```

(The `finally` load-decrement is unchanged from 3d.)

- [ ] **Step 4: Run the e2e** — `.venv/bin/python -m pytest tests/test_reputation_e2e.py -q` → PASS.

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q` → all green (existing coordinator/failover/load e2e assert counts/tokens/load, not reputation, so the added field is compatible).

- [ ] **Step 6: Commit**

```bash
git add eujeno/net/coordinator.py tests/test_reputation_e2e.py
git commit -m "feat(net): per-conn reputation (rise on success, routing input) + /registry field"
```

---

## Self-Review notes

- **Spec coverage:** reputation field + cold-start (Task 2 a/b) · `/registry` exposes it (Task 2 c) · reputation-aware routing, default-preserving (Task 1) · reward on success / best-effort penalty hook (Task 2 e) · acceptance #1 rise (Task 2 e2e) · acceptance #2 routing de-prioritization (Task 1 unit tests). The fall-on-failover is intentionally not e2e-tested (Decision 2 in the spec). All covered.
- **Placeholder scan:** none — complete code in every step.
- **Type consistency:** `build_chain(..., load=None, reputation=None)` defined (Task 1) and called with both maps (Task 2d). `conns[cid]["reputation"]` float initialized `REP_INITIAL` (Task 2b), read in registry (Task 2c) and routing (Task 2d), updated in `_generate_with_failover` (Task 2e). `chain_conns` already in scope from the 3d load block.
