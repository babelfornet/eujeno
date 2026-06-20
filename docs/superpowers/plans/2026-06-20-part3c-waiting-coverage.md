# Part 3c — WAITING_COVERAGE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When coverage is incomplete, a job parks durably as `WAITING_COVERAGE` and the request waits (up to a timeout) for a node to cover the gap, then resumes and completes — PRD Part 3 acceptance #3.

**Architecture:** `jobstore` gains `set_status` and `recover()` also clears `WAITING_COVERAGE`. The coordinator gains a `coverage_timeout` and an `_await_coverage` poll loop that replaces the immediate "incomplete coverage" failure: it parks the job, polls `build_chain` until a chain is available (then marks RUNNING and proceeds, resuming via 3b) or times out.

**Tech Stack:** Python stdlib `sqlite3`/`asyncio`/`time`, FastAPI coordinator, pytest. Spec: `docs/superpowers/specs/2026-06-20-part3c-waiting-coverage-design.md`.

## Global Constraints

- `create_coordinator_app(model_id, num_layers, tokenizer, db_path=None, coverage_timeout=120.0)` — existing callers (no `coverage_timeout`) keep the default.
- Park/resume reuses 3b: on coverage, generation resumes from persisted tokens.
- `set_status` writes go through `_store_safe` (best-effort). Timeout → normal error envelope + job `FAILED`.
- Poll interval 0.5 s; wait uses `time.monotonic()`.
- No happy-path behavior change (complete coverage → no parking).

---

### Task 1: jobstore `set_status` + `recover()` covers WAITING_COVERAGE

**Files:**
- Modify: `eujeno/net/jobstore.py`
- Test: `tests/test_jobstore.py` (add two tests)

**Interfaces:**
- Produces: `set_status(job_id, status) -> None`; `recover()` now flips both `RUNNING` and `WAITING_COVERAGE` → `INTERRUPTED`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_jobstore.py`:

```python
def test_set_status_changes_status(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.set_status("j1", "WAITING_COVERAGE")
    assert s.get_job("j1")["status"] == "WAITING_COVERAGE"


def test_recover_marks_waiting_coverage_interrupted(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("w", "m", "p", {}, 1); s.set_status("w", "WAITING_COVERAGE")
    s.create_job("r", "m", "p", {}, 1)                       # RUNNING
    s.create_job("d", "m", "p", {}, 1); s.finish("d", "x", "stop")
    n = s.recover()
    assert n == 2
    assert s.get_job("w")["status"] == "INTERRUPTED"
    assert s.get_job("r")["status"] == "INTERRUPTED"
    assert s.get_job("d")["status"] == "DONE"
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_jobstore.py -q` → FAIL (`set_status` missing; `recover` returns 1 not 2).

- [ ] **Step 3: Implement** in `eujeno/net/jobstore.py`. Add the method (e.g. right after `fail`):

```python
    def set_status(self, job_id, status):
        self._conn.execute("UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
                           (status, time.time(), job_id))
        self._conn.commit()
```

Change `recover` to also match `WAITING_COVERAGE`:

```python
    def recover(self):
        cur = self._conn.execute(
            "UPDATE jobs SET status='INTERRUPTED', updated_at=? "
            "WHERE status IN ('RUNNING', 'WAITING_COVERAGE')", (time.time(),))
        self._conn.commit()
        return cur.rowcount
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_jobstore.py -q` → PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/jobstore.py tests/test_jobstore.py
git commit -m "feat(net): jobstore set_status + recover() clears WAITING_COVERAGE"
```

---

### Task 2: coordinator `_await_coverage` (park & resume) + e2e

**Files:**
- Modify: `eujeno/net/coordinator.py`
- Test: `tests/test_waiting_coverage_e2e.py`

**Interfaces:**
- Consumes: `store.set_status` (Task 1), `build_chain`, `_run_generation` (resume-aware).
- Produces: `create_coordinator_app(..., coverage_timeout=120.0)`.

- [ ] **Step 1: Write the failing e2e** — create `tests/test_waiting_coverage_e2e.py`:

```python
import socket, threading, time, asyncio
import pytest, httpx, uvicorn

from eujeno.net.coordinator import create_coordinator_app
from eujeno.net.node import run_node
from eujeno.net.node_exec import NodeState
from eujeno.net.topology import StageSpec
from eujeno.model.generate import reference_generate


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


def _wait_count(client, base, n):
    for _ in range(200):
        if len(client.get(f"{base}/registry").json()["nodes"]) == n:
            return
        time.sleep(0.05)
    raise AssertionError(f"registry never reached {n} nodes")


@pytest.mark.slow
def test_job_parks_waiting_coverage_then_resumes(full_model, tmp_path):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)
    db = str(tmp_path / "jobs.db")
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer, db_path=db), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    result = {}

    def _infer():
        with httpx.Client(timeout=120.0) as c:
            result["data"] = c.post(f"{base}/infer", json={"prompt": prompt, "max_new_tokens": 6}).json()

    try:
        # only embed+decoders -> the head block is uncovered
        _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 24)]))))
        with httpx.Client(timeout=10.0) as client:
            _wait_count(client, base, 1)
            t = threading.Thread(target=_infer, daemon=True); t.start()
            parked = False
            for _ in range(400):
                jobs = client.get(f"{base}/jobs").json()["jobs"]
                if jobs and jobs[0]["status"] == "WAITING_COVERAGE":
                    parked = True; break
                time.sleep(0.05)
            assert parked, "job never entered WAITING_COVERAGE"
            # provide the head -> coverage completes -> job resumes
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(head=True))))
            t.join(timeout=120)
            jid = client.get(f"{base}/jobs").json()["jobs"][0]["job_id"]
            detail = client.get(f"{base}/jobs/{jid}").json()
        assert result["data"]["ok"] is True, result
        assert result["data"]["tokens"] == reference
        assert detail["status"] == "DONE"
    finally:
        server.should_exit = True


@pytest.mark.slow
def test_coverage_timeout_fails(full_model, tmp_path):
    model, tokenizer = full_model
    db = str(tmp_path / "jobs.db")
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer,
                                           db_path=db, coverage_timeout=2.0), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 24)]))))  # no head, ever
        with httpx.Client(timeout=30.0) as client:
            _wait_count(client, base, 1)
            data = client.post(f"{base}/infer", json={"prompt": "ciao", "max_new_tokens": 4}).json()
            jobs = client.get(f"{base}/jobs").json()["jobs"]
        assert data["ok"] is False
        assert "coverage timeout" in data.get("error", "")
        assert jobs[0]["status"] == "FAILED"
    finally:
        server.should_exit = True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_waiting_coverage_e2e.py -q` → FAIL (no `coverage_timeout` kwarg; today it returns "incomplete coverage" immediately, so the job never parks and the request returns an error instead of waiting).

- [ ] **Step 3: Implement in `eujeno/net/coordinator.py`.**

Add a module constant near `MAX_FAILOVERS = 5`:

```python
COVERAGE_POLL_INTERVAL = 0.5
```

Change the signature:
`def create_coordinator_app(model_id: str, num_layers: int, tokenizer, db_path=None):`
→
`def create_coordinator_app(model_id: str, num_layers: int, tokenizer, db_path=None, coverage_timeout=120.0):`

Add the `_await_coverage` helper immediately before `_generate_with_failover`:

```python
    async def _await_coverage(excluded, job_id):
        """Return a complete chain (excluding dead nodes) once available, parking the
        job durably as WAITING_COVERAGE while it waits; return None on timeout."""
        start = time.monotonic()
        marked = False
        while True:
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is not None:
                if marked:
                    _store_safe(store.set_status, job_id, "RUNNING")
                return chain
            if not marked:
                _store_safe(store.set_status, job_id, "WAITING_COVERAGE")
                marked = True
            if time.monotonic() - start >= coverage_timeout:
                return None
            await asyncio.sleep(COVERAGE_POLL_INTERVAL)
```

Replace the top of the `_generate_with_failover` loop — change:

```python
        for attempt in range(MAX_FAILOVERS + 1):
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is None:
                return None, {"error": "model not operational: incomplete coverage", "excluded": sorted(excluded)}
```

to:

```python
        for attempt in range(MAX_FAILOVERS + 1):
            chain = await _await_coverage(excluded, job_id)
            if chain is None:
                return None, {"error": "coverage timeout: model not operational", "excluded": sorted(excluded)}
```

(The rest of `_generate_with_failover` — the `try` running `_run_generation` and the `_NodeFailure` resume handling — is unchanged.)

- [ ] **Step 4: Run the new e2e** — `.venv/bin/python -m pytest tests/test_waiting_coverage_e2e.py -q` → PASS (2 passed). (`@pytest.mark.slow`; loads the 0.5B model; ~1-3 min.)

- [ ] **Step 5: Full suite** — `.venv/bin/python -m pytest -q` → all green (no regressions; complete-coverage path unchanged).

- [ ] **Step 6: Commit**

```bash
git add eujeno/net/coordinator.py tests/test_waiting_coverage_e2e.py
git commit -m "feat(net): WAITING_COVERAGE — park & resume on incomplete coverage (acc #3)"
```

---

## Self-Review notes

- **Spec coverage:** durable park `WAITING_COVERAGE` + `set_status` (Task 1) · `recover()` clears it (Task 1) · `_await_coverage` poll loop with TTL, marks WAITING_COVERAGE then RUNNING (Task 2) · timeout → FAILED error (Task 2) · resume on coverage via existing 3b path (unchanged tail of `_generate_with_failover`) · acceptance #3 park-then-resume e2e + timeout e2e (Task 2). All covered.
- **Placeholder scan:** none — complete code in every step.
- **Type consistency:** `set_status(job_id, status)` defined (Task 1) and called as `_store_safe(store.set_status, job_id, "WAITING_COVERAGE"|"RUNNING")` (Task 2). `coverage_timeout` param threaded into `_await_coverage` via closure. `_await_coverage(excluded, job_id)` returns a chain or None, consumed correctly.
