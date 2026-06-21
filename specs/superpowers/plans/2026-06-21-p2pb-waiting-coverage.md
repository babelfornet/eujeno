# P2Pb — WAITING_COVERAGE for pure-P2P Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `infer --peer` waits for coverage (a node filling the gap) up to a timeout instead of failing immediately.

**Architecture:** Add a coverage-resolution poll-loop (`coverage_timeout`/`poll_interval`) to `distributed_generate_resilient`; expose `--wait-coverage` on `infer`. `coverage_timeout=0` keeps the current fail-fast behavior.

**Tech Stack:** Python, httpx, torch, FastAPI/Starlette (test), pytest. Spec: `docs/superpowers/specs/2026-06-21-p2pb-waiting-coverage-design.md`.

## Global Constraints

- `coverage_timeout=0.0` (default) ⇒ byte-for-byte current behavior. Sync `time.monotonic()`/`time.sleep`.
- Coverage-wait uses the existing `refresh()` callback to learn newly-joined nodes.
- The rest of `distributed_generate_resilient` (failover/resume/EOS) is unchanged.

---

### Task 1: coverage-wait in the resilient orchestrator + CLI flag + e2e

**Files:**
- Modify: `eujeno/net/orchestrator.py` (`distributed_generate_resilient`)
- Modify: `eujeno/cli.py` (`infer` — add `--wait-coverage`, pass `coverage_timeout`)
- Test: `tests/test_p2p_waiting_coverage_e2e.py` (new)

**Interfaces:**
- Produces: `distributed_generate_resilient(..., coverage_timeout=0.0, poll_interval=0.5)`.

- [ ] **Step 1: Write the failing e2e** — create `tests/test_p2p_waiting_coverage_e2e.py`:

```python
import socket, threading, time
import pytest, httpx, uvicorn

from eujeno.net.server import create_app
from eujeno.net.topology import StageSpec
from eujeno.net.orchestrator import distributed_generate_resilient
from eujeno.net.generation import stop_token_ids
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


@pytest.mark.slow
def test_p2p_waits_for_coverage_then_completes(full_model):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 24)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3), pA)
    holder = {}

    def _run():
        with httpx.Client(timeout=90.0) as client:
            holder["r"] = distributed_generate_resilient(
                {uA: {"embed": True, "head": False, "decoders": ["0-24"]}}, 24, prompt, 6, client, tokenizer,
                stop_ids=stop_token_ids(tokenizer), coverage_timeout=30,
                refresh=lambda: httpx.get(f"{uA}/registry", timeout=10).json()["nodes"])

    t = threading.Thread(target=_run, daemon=True); t.start()
    time.sleep(2.0)
    assert t.is_alive(), "should be waiting for coverage (head uncovered), not returned"
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    t.join(timeout=60)
    try:
        assert not t.is_alive(), "did not complete after coverage arrived"
        assert holder["r"]["ok"] is True, holder
        assert holder["r"]["tokens"] == reference
    finally:
        sA.should_exit = sB.should_exit = True


@pytest.mark.slow
def test_p2p_coverage_timeout(full_model):
    model, tokenizer = full_model
    pA = _free_port()
    uA = f"http://127.0.0.1:{pA}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 24)]),
                           node_url=uA, peers=[], num_layers=24, gossip_interval=0.3), pA)
    try:
        with httpx.Client(timeout=30.0) as client:
            result = distributed_generate_resilient(
                {uA: {"embed": True, "head": False, "decoders": ["0-24"]}}, 24, "ciao", 4, client, tokenizer,
                stop_ids=set(), coverage_timeout=2,
                refresh=lambda: httpx.get(f"{uA}/registry", timeout=5).json()["nodes"])
        assert result["ok"] is False
        assert "coverage timeout" in result["error"]
    finally:
        sA.should_exit = True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_p2p_waiting_coverage_e2e.py -q` → FAIL (no `coverage_timeout` kwarg; today returns incomplete-coverage immediately so the thread is not alive after 2s). (@pytest.mark.slow.)

- [ ] **Step 3: Add coverage-wait to `eujeno/net/orchestrator.py`.** Replace the `distributed_generate_resilient` signature and the top-of-attempt refresh/build/return block. New signature:

```python
def distributed_generate_resilient(stages_by_url, num_layers, prompt, max_new_tokens, client,
                                   tokenizer, stop_ids=None, job_id_prefix="job",
                                   refresh=None, max_failovers=5, coverage_timeout=0.0,
                                   poll_interval=0.5):
```

Add `import time` at the top of the function body (after the existing `from eujeno.net.discovery import build_chain`). Define a coverage resolver right after `finish_reason = "length"`:

```python
    def _resolve_chain():
        nonlocal stages_by_url
        start = time.monotonic()
        while True:
            if refresh is not None:
                try:
                    fresh = refresh()
                    if fresh:
                        stages_by_url = fresh
                except Exception:
                    pass
            chain = build_chain(stages_by_url, num_layers, exclude=excluded)
            if chain is not None:
                return chain
            if time.monotonic() - start >= coverage_timeout:
                return None
            time.sleep(poll_interval)
```

Replace the current top of the `for attempt ...` loop — i.e. replace this block:

```python
        if refresh is not None:
            try:
                fresh = refresh()
                if fresh:
                    stages_by_url = fresh
            except Exception:
                pass
        chain = build_chain(stages_by_url, num_layers, exclude=excluded)
        if chain is None:
            return {"ok": False, "error": "incomplete coverage: model not operational",
                    "tokens": tokens, "failovers": attempt}
```

with:

```python
        chain = _resolve_chain()
        if chain is None:
            err = ("coverage timeout: model not operational" if coverage_timeout > 0
                   else "incomplete coverage: model not operational")
            return {"ok": False, "error": err, "tokens": tokens, "failovers": attempt}
```

(The rest of the function — `embed_url, decoders, head_url = chain` onward, including generation, resume, EOS, cleanup, failover `except`, and the final "too many failovers" return — is unchanged.)

- [ ] **Step 4: Add the CLI flag in `eujeno/cli.py`.** Add to the `infer` command signature (near the other options):

```python
    wait_coverage: int = typer.Option(0, "--wait-coverage", help="[P2P] seconds to wait for full coverage before failing"),
```

In the `--peer` branch, pass it to the call — change the `distributed_generate_resilient(...)` invocation to include `coverage_timeout=wait_coverage`:

```python
                result = distributed_generate_resilient(
                    reg["nodes"], reg["num_layers"], prompt, max_new_tokens, client, tokenizer,
                    stop_ids=stop_ids, refresh=_refresh, coverage_timeout=wait_coverage)
```

- [ ] **Step 5: Run the e2e** — `.venv/bin/python -m pytest tests/test_p2p_waiting_coverage_e2e.py -q` → PASS (2 passed).

- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest -q` → all green (existing P2P/coordinator tests unaffected; `coverage_timeout` defaults 0).

- [ ] **Step 7: Commit**

```bash
git add eujeno/net/orchestrator.py eujeno/cli.py tests/test_p2p_waiting_coverage_e2e.py
git commit -m "feat(net): WAITING_COVERAGE for pure-P2P (infer --peer waits for coverage) + --wait-coverage"
```

---

## Self-Review notes

- **Spec coverage:** coverage poll-wait with refresh + monotonic timeout (Task 1 `_resolve_chain`) · `coverage_timeout=0` preserves fail-fast · CLI `--wait-coverage` · park-then-resume + timeout e2e. All covered.
- **Placeholder scan:** none — complete code.
- **Type consistency:** `_resolve_chain` uses `nonlocal stages_by_url`; returns a chain or None; the generation/failover tail is untouched (still consumes `embed_url, decoders, head_url`). `coverage_timeout` default 0.0 keeps existing callers (P2Pa e2e, the failover test) unchanged.
