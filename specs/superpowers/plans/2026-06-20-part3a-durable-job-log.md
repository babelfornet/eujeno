# Part 3a — Durable Job Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the coordinator a durable, crash-safe SQLite job log so a distributed generation's state is reconstructible from disk and survives a coordinator restart without losing or doubling tokens — with no change to inference behavior.

**Architecture:** A standalone `eujeno/net/jobstore.py` (SQLite WAL, single `jobs` table, no network/torch deps) is unit-tested in isolation, then wired into `eujeno/net/coordinator.py`: one durable row per `/infer` and `/v1/chat/completions` request, updated per token step, finalized on DONE/FAILED, with stale `RUNNING` rows marked `INTERRUPTED` at startup. A read API (`GET /jobs/{id}`, `GET /jobs`) and a `coordinator --db PATH` flag are added.

**Tech Stack:** Python stdlib `sqlite3` (WAL mode), FastAPI, Typer, pytest. Spec: `docs/superpowers/specs/2026-06-20-part3a-durable-job-log-design.md`.

## Global Constraints

- Orchestrator-driven, coordinator-side substrate only. No per-hop `stages`/`outbox`/activation blobs (deferred peer-driven target).
- Per-token-step granularity; idempotency key is `(job_id, position)`.
- No auto-resume in 3a: on restart, `RUNNING` → `INTERRUPTED` (option A). Auto-resume is 3b.
- `create_coordinator_app(model_id, num_layers, tokenizer, db_path=None)` — `db_path=None` means `":memory:"` (ephemeral; keeps existing test callers unchanged). The CLI resolves the on-disk default `~/.eujeno/coordinator-jobs.db`.
- Inference output must not change: existing golden/e2e tests stay green.
- License header on new files: `# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>` / `# SPDX-License-Identifier: Apache-2.0`.

---

### Task 1: `jobstore.py` (durable job log) + unit tests

**Files:**
- Create: `eujeno/net/jobstore.py`
- Test: `tests/test_jobstore.py`

**Interfaces:**
- Produces (consumed by Task 2):
  - `JobStore(path: str)` — opens/creates DB; WAL on for file paths; creates schema.
  - `create_job(job_id, model_id, prompt, sampling: dict, prompt_len: int) -> None`
  - `append_token(job_id, token_id: int, position: int) -> None` — idempotent on `(job_id, position)`
  - `reset_progress(job_id) -> None` — clears tokens/position (used per failover re-attempt)
  - `finish(job_id, result: str, finish_reason: str) -> None`
  - `fail(job_id, error: str) -> None`
  - `recover() -> int` — `RUNNING` → `INTERRUPTED`, returns count
  - `get_job(job_id) -> dict | None`, `recent_jobs(limit=50) -> list[dict]` (dicts include `tokens: list[int]` and `sampling: dict`)
  - `close() -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_jobstore.py`:

```python
import json
from eujeno.net.jobstore import JobStore


def test_create_append_finish_roundtrip(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "hello", {"temperature": 0.0}, prompt_len=3)
    s.append_token("j1", 10, 0)
    s.append_token("j1", 20, 1)
    s.finish("j1", "ten twenty", "stop")
    j = s.get_job("j1")
    assert j["status"] == "DONE"
    assert j["tokens"] == [10, 20]
    assert j["position"] == 2
    assert j["result"] == "ten twenty"
    assert j["finish_reason"] == "stop"
    assert j["prompt_len"] == 3
    assert j["sampling"] == {"temperature": 0.0}


def test_append_is_idempotent_on_position(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    s.append_token("j1", 10, 0)        # same (job, position) again
    assert s.get_job("j1")["tokens"] == [10]   # no double


def test_reset_progress_clears_tokens(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    s.reset_progress("j1")
    j = s.get_job("j1")
    assert j["tokens"] == [] and j["position"] == 0 and j["status"] == "RUNNING"


def test_fail_sets_status_and_error(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.fail("j1", "too many failovers")
    j = s.get_job("j1")
    assert j["status"] == "FAILED" and j["error"] == "too many failovers"


def test_recover_marks_running_interrupted(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("run", "m", "p", {}, 1)            # stays RUNNING
    s.create_job("done", "m", "p", {}, 1); s.finish("done", "x", "stop")
    n = s.recover()
    assert n == 1
    assert s.get_job("run")["status"] == "INTERRUPTED"
    assert s.get_job("done")["status"] == "DONE"    # untouched


def test_durable_across_reopen(tmp_path):
    path = str(tmp_path / "j.db")
    s = JobStore(path)
    s.create_job("j1", "m", "p", {}, 1); s.append_token("j1", 7, 0); s.finish("j1", "seven", "stop")
    s.close()
    s2 = JobStore(path)                              # reopen
    assert s2.get_job("j1")["tokens"] == [7]
    assert s2.get_job("j1")["status"] == "DONE"


def test_recent_jobs_orders_newest_first(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("a", "m", "p", {}, 1)
    s.create_job("b", "m", "p", {}, 1)
    ids = [j["job_id"] for j in s.recent_jobs(limit=10)]
    assert set(ids) == {"a", "b"} and len(ids) == 2


def test_get_missing_job_returns_none(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    assert s.get_job("nope") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_jobstore.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eujeno.net.jobstore'`

- [ ] **Step 3: Implement `eujeno/net/jobstore.py`**

```python
# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Durable job log for the coordinator: a small SQLite(WAL) store of distributed
generation jobs. Single responsibility, no network/torch deps."""

import json
import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id        TEXT PRIMARY KEY,
  model_id      TEXT,
  status        TEXT,
  prompt        TEXT,
  sampling_json TEXT,
  prompt_len    INTEGER,
  position      INTEGER,
  tokens_json   TEXT,
  result        TEXT,
  finish_reason TEXT,
  error         TEXT,
  created_at    REAL,
  updated_at    REAL
);
"""


class JobStore:
    """Durable per-coordinator job log. status: RUNNING|DONE|FAILED|INTERRUPTED."""

    def __init__(self, path):
        self.path = path
        if path != ":memory:" and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_job(self, job_id, model_id, prompt, sampling, prompt_len):
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, model_id, status, prompt, sampling_json, "
            "prompt_len, position, tokens_json, result, finish_reason, error, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, model_id, "RUNNING", prompt, json.dumps(sampling or {}), int(prompt_len),
             0, json.dumps([]), None, None, None, now, now))
        self._conn.commit()

    def append_token(self, job_id, token_id, position):
        row = self._conn.execute("SELECT tokens_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return
        toks = json.loads(row["tokens_json"] or "[]")
        if position < len(toks):
            toks[position] = int(token_id)        # re-apply same position -> idempotent, no double
        elif position == len(toks):
            toks.append(int(token_id))
        else:
            return                                # out-of-order beyond next: ignore (not expected)
        self._conn.execute("UPDATE jobs SET tokens_json=?, position=?, updated_at=? WHERE job_id=?",
                           (json.dumps(toks), len(toks), time.time(), job_id))
        self._conn.commit()

    def reset_progress(self, job_id):
        self._conn.execute("UPDATE jobs SET tokens_json=?, position=0, updated_at=? WHERE job_id=?",
                           (json.dumps([]), time.time(), job_id))
        self._conn.commit()

    def finish(self, job_id, result, finish_reason):
        self._conn.execute("UPDATE jobs SET status=?, result=?, finish_reason=?, updated_at=? WHERE job_id=?",
                           ("DONE", result, finish_reason, time.time(), job_id))
        self._conn.commit()

    def fail(self, job_id, error):
        self._conn.execute("UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
                           ("FAILED", str(error), time.time(), job_id))
        self._conn.commit()

    def recover(self):
        cur = self._conn.execute(
            "UPDATE jobs SET status='INTERRUPTED', updated_at=? WHERE status='RUNNING'", (time.time(),))
        self._conn.commit()
        return cur.rowcount

    def get_job(self, job_id):
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def recent_jobs(self, limit=50):
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row):
        d = dict(row)
        d["sampling"] = json.loads(d.pop("sampling_json") or "{}")
        d["tokens"] = json.loads(d.pop("tokens_json") or "[]")
        return d

    def close(self):
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jobstore.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/jobstore.py tests/test_jobstore.py
git commit -m "feat(net): durable JobStore (SQLite WAL job log) + unit tests"
```

---

### Task 2: Wire JobStore into the coordinator + CLI flag + read API + e2e

**Files:**
- Modify: `eujeno/net/coordinator.py`
- Modify: `eujeno/cli.py` (coordinator command: add `--db`)
- Test: `tests/test_coordinator_jobs_e2e.py`

**Interfaces:**
- Consumes: `JobStore` (Task 1).
- Produces: `create_coordinator_app(model_id, num_layers, tokenizer, db_path=None)`; HTTP `GET /jobs/{job_id}` and `GET /jobs?limit=N`.

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_coordinator_jobs_e2e.py` (mirrors the existing coordinator e2e harness):

```python
import socket, threading, time, asyncio
import pytest, httpx, uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.net.jobstore import JobStore


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started: break
        time.sleep(0.05)
    assert server.started
    return server


def _run_node_thread(ws_url, state):
    threading.Thread(target=lambda: asyncio.run(run_node(ws_url, state)), daemon=True).start()


@pytest.mark.slow
def test_job_is_persisted_and_reconstructible(full_model, tmp_path):
    model, tokenizer = full_model
    db = str(tmp_path / "jobs.db")
    port = _free_port()
    app = create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", num_layers=24, tokenizer=tokenizer, db_path=db)
    _serve(app, port)
    # one node covering the whole model
    state = NodeState(model, [StageSpec("embed"), StageSpec("decoder", 0, 24), StageSpec("head")])
    _run_node_thread(f"ws://127.0.0.1:{port}/node", state)
    for _ in range(200):
        r = httpx.get(f"http://127.0.0.1:{port}/registry").json()
        if r["nodes"]: break
        time.sleep(0.05)

    resp = httpx.post(f"http://127.0.0.1:{port}/infer",
                      json={"prompt": "The capital of France is", "max_new_tokens": 5}, timeout=120).json()
    assert resp["ok"] is True
    tokens = resp["tokens"]

    # reconstructible via the read API
    api = httpx.get(f"http://127.0.0.1:{port}/jobs").json()
    assert len(api["jobs"]) >= 1
    jid = api["jobs"][0]["job_id"]
    one = httpx.get(f"http://127.0.0.1:{port}/jobs/{jid}").json()
    assert one["status"] == "DONE"
    assert one["tokens"] == tokens

    # reconstructible from a freshly-opened DB (durability)
    s2 = JobStore(db)
    assert s2.get_job(jid)["tokens"] == tokens
    assert s2.get_job(jid)["status"] == "DONE"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_coordinator_jobs_e2e.py -q`
Expected: FAIL — `create_coordinator_app()` has no `db_path` kwarg / no `/jobs` route.

- [ ] **Step 3: Wire JobStore into `coordinator.py`**

Add the import near the others (after line 16 `from eujeno.net.tools import extract_tool_calls`):

```python
from eujeno.net.jobstore import JobStore
```

Change the signature and add the store (replace `def create_coordinator_app(model_id: str, num_layers: int, tokenizer):` and the line `app = FastAPI()`):

```python
def create_coordinator_app(model_id: str, num_layers: int, tokenizer, db_path=None):
    """Coordinator-relay: nodes connect via WS and announce their stages; POST /infer
    drives generation by relaying each hop to the right node. Jobs are persisted to a
    durable SQLite job log (db_path=None -> in-memory, used by tests)."""
    app = FastAPI()
    store = JobStore(db_path if db_path is not None else ":memory:")
    store.recover()
```

Replace `_run_generation`'s signature line and the token-append site to accept a callback. Change:
`    async def _run_generation(chain, prompt, max_new, sampling, job_id):`
to:
`    async def _run_generation(chain, prompt, max_new, sampling, job_id, on_token=None):`
and immediately after `tokens.append(tok)` (currently line 119) insert:

```python
            if on_token is not None:
                on_token(len(tokens) - 1, tok)
```

Replace `_generate_with_failover` (lines 129-143) so it takes the durable `job_id`, resets per attempt, and streams tokens to the store:

```python
    async def _generate_with_failover(prompt, max_new, sampling, job_id):
        excluded = set()
        last_failed = None
        for attempt in range(MAX_FAILOVERS + 1):
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is None:
                return None, {"error": "model not operational: incomplete coverage", "excluded": sorted(excluded)}
            store.reset_progress(job_id)
            try:
                tokens, prompt_len, finish_reason = await _run_generation(
                    chain, prompt, max_new, sampling, _next_id("job"),
                    on_token=lambda pos, tok: store.append_token(job_id, tok, pos))
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt, "finish_reason": finish_reason}, None
            except _NodeFailure as e:
                excluded.add(e.conn_id)
                last_failed = e.conn_id
        return None, {"error": f"too many failovers (last failed node: {last_failed})"}
```

Replace the `/infer` handler (lines 145-156) to create/finish/fail the durable job:

```python
    @app.post("/infer")
    async def infer(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 8))
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        job_id = _next_id("job")
        prompt_len = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        store.create_job(job_id, model_id, prompt, sampling, prompt_len)
        result, err = await _generate_with_failover(prompt, max_new, sampling, job_id)
        if err is not None:
            store.fail(job_id, err["error"])
            return {"ok": False, **err}
        text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
        store.finish(job_id, text, result["finish_reason"])
        return {"ok": True, "model": model_id, "prompt": prompt,
                "text": text, "tokens": result["tokens"], "failovers": result["failovers"]}
```

Replace the `/v1/chat/completions` handler's generation section. After computing `prompt` (the line `prompt = "\n".join(...)` block ends at line 175), replace the rest of the function body (lines 176 onward, through the `return {...}` ) with:

```python
        job_id = _next_id("job")
        prompt_len = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        store.create_job(job_id, model_id, prompt, sampling, prompt_len)
        result, err = await _generate_with_failover(prompt, max_new, sampling, job_id)
        if err is not None:
            store.fail(job_id, err["error"])
            return JSONResponse({"error": {"message": err["error"], "type": "not_operational"}}, status_code=503)
        text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
        store.finish(job_id, text, result["finish_reason"])
        content, tool_calls = extract_tool_calls(text)
        message = {"role": "assistant", "content": (content if content else None)}
        finish_reason = result["finish_reason"]
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return {
            "id": "chatcmpl-" + _next_id("oa"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": result["prompt_len"],
                      "completion_tokens": len(result["tokens"]),
                      "total_tokens": result["prompt_len"] + len(result["tokens"])},
        }
```

Add the read API just before `return app` (end of `create_coordinator_app`):

```python
    @app.get("/jobs")
    async def list_jobs(limit: int = 50):
        return {"jobs": store.recent_jobs(limit)}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        j = store.get_job(job_id)
        if j is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return j
```

- [ ] **Step 4: Add the `--db` flag to the CLI `coordinator` command**

In `eujeno/cli.py`, add `import os` if not present (check the top of the file first). In the `coordinator` command, add a `db` option and pass it through. Add the option parameter to the `coordinator(...)` signature:

```python
    db: str = typer.Option(None, "--db", help="SQLite job-log path (default ~/.eujeno/coordinator-jobs.db)"),
```

and change the `create_coordinator_app(...)` call (currently `eujeno/cli.py:339`) from
`    coord_app = create_coordinator_app(model_id, num_layers, tokenizer)`
to:

```python
    db_path = db or os.path.expanduser("~/.eujeno/coordinator-jobs.db")
    coord_app = create_coordinator_app(model_id, num_layers, tokenizer, db_path=db_path)
```

- [ ] **Step 5: Run the new e2e + the full suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/test_coordinator_jobs_e2e.py -q`
Expected: PASS (1 passed).

Run: `.venv/bin/python -m pytest -q`
Expected: all green (the existing coordinator/openai/failover e2e tests still pass — they call `create_coordinator_app(...)` without `db_path`, so they use an in-memory store).

- [ ] **Step 6: Commit**

```bash
git add eujeno/net/coordinator.py eujeno/cli.py tests/test_coordinator_jobs_e2e.py
git commit -m "feat(net): persist jobs to the durable log + /jobs read API + coordinator --db"
```

---

## Self-Review notes

- **Spec coverage:** durable SQLite WAL substrate + `jobs` table (Task 1) · per-token-step persistence with idempotent `(job_id, position)` (Task 1 `append_token` + Task 2 `on_token`) · create/finish/fail per request (Task 2) · `recover()` RUNNING→INTERRUPTED at startup (Task 1 + wired in Task 2) · read API `GET /jobs[/{id}]` (Task 2) · `coordinator --db` default `~/.eujeno/coordinator-jobs.db` (Task 2) · acceptance #1 reconstructible (Task 2 e2e) and #4 idempotent/restart-safe (Task 1 `append_token`/`recover` tests). No inference change (existing e2e stay green). All covered.
- **Placeholder scan:** none — every step has complete code.
- **Type consistency:** `JobStore` method names/signatures identical between Task 1 (definition) and Task 2 (use): `create_job(job_id, model_id, prompt, sampling, prompt_len)`, `append_token(job_id, token_id, position)`, `reset_progress(job_id)`, `finish(job_id, result, finish_reason)`, `fail(job_id, error)`, `recover()`, `get_job`, `recent_jobs`. `create_coordinator_app(..., db_path=None)` consistent across coordinator def, CLI call, and e2e test.
