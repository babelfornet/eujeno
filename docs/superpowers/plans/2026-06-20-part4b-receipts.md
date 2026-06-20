# Part 4b — Hop Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-peer contribution (hops, bytes, compute time) for each job in the durable log as the ledger hook — PRD Part 4 acceptance #3.

**Architecture:** `jobstore` gains a `receipts` table + `add_receipts`/`get_receipts` (UPSERT-accumulate per `(job_id, peer_id)`). The coordinator times+sizes each embed/decode/head `_call`, accumulates per peer during a generation, and persists the winning attempt's receipts at completion; a `GET /jobs/{id}/receipts` exposes them.

**Tech Stack:** Python stdlib `sqlite3`, FastAPI coordinator, pytest. Spec: `docs/superpowers/specs/2026-06-20-part4b-receipts-design.md`.

## Global Constraints

- Aggregated per `(job_id, peer_id)`: `{hops, bytes, t_compute}`. Only ACKed (successful) hops counted; winning attempt only.
- Persist via `_store_safe` (best-effort). No per-hop SQLite writes (accumulate in memory, write once at finish).
- No change to existing endpoints/shapes or the node protocol.

---

### Task 1: jobstore `receipts` table + `add_receipts`/`get_receipts` + tests

**Files:**
- Modify: `eujeno/net/jobstore.py`
- Test: `tests/test_jobstore.py` (add three tests)

**Interfaces:**
- Produces: `add_receipts(job_id, receipts: dict)` (`receipts = {peer_id: {"hops","bytes","t_compute"}}`, UPSERT-accumulate); `get_receipts(job_id) -> list[dict]` (each `{peer_id, hops, bytes, t_compute}`).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_jobstore.py`:

```python
def test_add_and_get_receipts_roundtrip(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.add_receipts("j1", {"c1": {"hops": 3, "bytes": 100, "t_compute": 0.5},
                          "c2": {"hops": 1, "bytes": 40, "t_compute": 0.1}})
    r = {x["peer_id"]: x for x in s.get_receipts("j1")}
    assert r["c1"]["hops"] == 3 and r["c1"]["bytes"] == 100 and abs(r["c1"]["t_compute"] - 0.5) < 1e-9
    assert r["c2"]["hops"] == 1


def test_add_receipts_accumulates(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.add_receipts("j1", {"c1": {"hops": 2, "bytes": 10, "t_compute": 0.2}})
    s.add_receipts("j1", {"c1": {"hops": 3, "bytes": 5, "t_compute": 0.3}})
    r = s.get_receipts("j1")
    assert len(r) == 1
    assert r[0]["hops"] == 5 and r[0]["bytes"] == 15 and abs(r[0]["t_compute"] - 0.5) < 1e-9


def test_get_receipts_unknown_job_is_empty(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    assert s.get_receipts("nope") == []
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_jobstore.py -q` → FAIL (`add_receipts`/`get_receipts` missing; no `receipts` table).

- [ ] **Step 3: Implement** in `eujeno/net/jobstore.py`.

(a) Append a second table to the `_SCHEMA` string (after the `jobs(...)` table's closing `);` and before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS receipts (
  job_id     TEXT,
  peer_id    TEXT,
  hops       INTEGER,
  bytes      INTEGER,
  t_compute  REAL,
  PRIMARY KEY (job_id, peer_id)
);
```

(b) Add the two methods to the `JobStore` class (e.g. after `recent_jobs`):

```python
    def add_receipts(self, job_id, receipts):
        for peer_id, r in (receipts or {}).items():
            self._conn.execute(
                "INSERT INTO receipts (job_id, peer_id, hops, bytes, t_compute) VALUES (?,?,?,?,?) "
                "ON CONFLICT(job_id, peer_id) DO UPDATE SET hops=hops+excluded.hops, "
                "bytes=bytes+excluded.bytes, t_compute=t_compute+excluded.t_compute",
                (job_id, peer_id, int(r.get("hops", 0)), int(r.get("bytes", 0)), float(r.get("t_compute", 0.0))))
        self._conn.commit()

    def get_receipts(self, job_id):
        rows = self._conn.execute(
            "SELECT peer_id, hops, bytes, t_compute FROM receipts WHERE job_id=? ORDER BY peer_id",
            (job_id,)).fetchall()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_jobstore.py -q` → PASS (15 passed).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/jobstore.py tests/test_jobstore.py
git commit -m "feat(net): jobstore receipts table (add_receipts/get_receipts, accumulating)"
```

---

### Task 2: coordinator receipt measurement + persist + endpoint + e2e

**Files:**
- Modify: `eujeno/net/coordinator.py`
- Test: `tests/test_receipts_e2e.py` (new)

**Interfaces:**
- Consumes: `store.add_receipts`/`store.get_receipts` (Task 1).
- Produces: `_run_generation(..., receipts=None)`; route `GET /jobs/{job_id}/receipts`.

- [ ] **Step 1: Write the failing e2e** — create `tests/test_receipts_e2e.py`:

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
def test_receipts_recorded_for_completed_job(full_model, tmp_path):
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
            r = client.post(f"{base}/infer", json={"prompt": "The capital of France is", "max_new_tokens": 5}).json()
            assert r["ok"] is True
            jid = client.get(f"{base}/jobs").json()["jobs"][0]["job_id"]
            receipts = client.get(f"{base}/jobs/{jid}/receipts").json()["receipts"]
        assert len(receipts) >= 1
        total_hops = sum(x["hops"] for x in receipts)
        assert total_hops > 0
        assert all(x["bytes"] > 0 for x in receipts)
        assert all(x["t_compute"] >= 0 for x in receipts)
    finally:
        server.should_exit = True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_receipts_e2e.py -q` → FAIL (no `/jobs/{id}/receipts` route / no receipts recorded). (@pytest.mark.slow.)

- [ ] **Step 3: Implement in `eujeno/net/coordinator.py`.**

(a) Add a `receipts=None` param to `_run_generation` and a timed wrapper. Change the signature:

```python
    async def _run_generation(chain, prompt, max_new, sampling, job_id, on_token=None, resume_tokens=None):
```
→
```python
    async def _run_generation(chain, prompt, max_new, sampling, job_id, on_token=None, resume_tokens=None, receipts=None):
```

Immediately after `embed_c, decoders, head_c = chain` (first line of the function body), add the timed call helper:

```python
        async def _rc(cid, header, payload=b""):
            t0 = time.monotonic()
            rh, rp = await _call(cid, header, payload)
            if receipts is not None:
                r = receipts.setdefault(cid, {"hops": 0, "bytes": 0, "t_compute": 0.0})
                r["hops"] += 1
                r["bytes"] += len(payload) + (len(rp) if rp else 0)
                r["t_compute"] += time.monotonic() - t0
            return rh, rp
```

Then replace the three generation `_call`s (embed, decode, head) with `_rc` (leave the `end`-cleanup `_call` as `_call`):

```python
            _, p = await _rc(embed_c, {"op": "embed", "job_id": job_id},
                             encode_tensors({"input_ids": cur}))
            h = decode_tensors(p)["hidden_states"]
            for block_key, cid in decoders:
                _, p = await _rc(cid, {"op": "decode", "block_key": block_key, "job_id": job_id},
                                 encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                h = decode_tensors(p)["hidden_states"]
            rh, _ = await _rc(head_c, {"op": "head", "job_id": job_id, "topk": topk},
                              encode_tensors({"hidden_states": h}))
```

(b) In `_generate_with_failover`, capture and persist receipts for the winning attempt. After `chain_conns = {...}` add `attempt_receipts = {}`; pass `receipts=attempt_receipts` to `_run_generation`; persist on success. The relevant section becomes:

```python
            embed_c, decoders, head_c = chain
            chain_conns = {embed_c, head_c, *(cid for _, cid in decoders)}
            attempt_receipts = {}
            for cid in chain_conns:
                if cid in conns:
                    conns[cid]["load"] += 1
            try:
                tokens, prompt_len, finish_reason = await _run_generation(
                    chain, prompt, max_new, sampling, _next_id("job"),
                    on_token=lambda pos, tok: _store_safe(store.append_token, job_id, tok, pos),
                    resume_tokens=resume_tokens, receipts=attempt_receipts)
                for cid in chain_conns:
                    if cid in conns:
                        conns[cid]["reputation"] = min(REP_MAX, conns[cid]["reputation"] + REP_REWARD)
                _store_safe(store.add_receipts, job_id, attempt_receipts)
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt, "finish_reason": finish_reason}, None
```

(the `except`/`finally` blocks are unchanged.)

(c) Add the read route near the other `/jobs` routes (e.g. after `get_job`):

```python
    @app.get("/jobs/{job_id}/receipts")
    async def get_receipts(job_id: str):
        return {"receipts": store.get_receipts(job_id)}
```

- [ ] **Step 4: Run the e2e** — `.venv/bin/python -m pytest tests/test_receipts_e2e.py -q` → PASS.

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q` → all green.

- [ ] **Step 6: Commit**

```bash
git add eujeno/net/coordinator.py tests/test_receipts_e2e.py
git commit -m "feat(net): record per-peer hop receipts (bytes/compute) + /jobs/{id}/receipts"
```

---

## Self-Review notes

- **Spec coverage:** receipts table + add/get accumulate (Task 1) · coordinator measures bytes/t_compute/hops per peer for ACKed hops only (Task 2a `_rc`, replaces embed/decode/head `_call`s, not `end`) · persists winning attempt at completion (Task 2b) · `GET /jobs/{id}/receipts` (Task 2c) · acceptance #3 e2e (Task 2) + jobstore unit tests (Task 1). All covered.
- **Placeholder scan:** none — complete code in every step.
- **Type consistency:** `add_receipts(job_id, {peer:{"hops","bytes","t_compute"}})` shape produced by `_rc`'s `receipts.setdefault(cid, {"hops","bytes","t_compute"})` matches what `add_receipts` reads (`r.get("hops"/"bytes"/"t_compute")`). `_run_generation(..., receipts=None)` defined and called with `receipts=attempt_receipts`. Route path `/jobs/{job_id}/receipts` distinct from `/jobs/{job_id}`.
