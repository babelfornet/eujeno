# Part 3d — Load Balancing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spread concurrent requests across redundant holders — the coordinator tracks per-connection `load` (active generations) and `build_chain` prefers the least-loaded holder for each stage. PRD Part 3 §6.

**Architecture:** `build_chain` gains an optional `load` post-pass (default path unchanged). The coordinator counts active generations per connection in `conns[cid]["load"]`, exposes it in `/registry`, threads it into `build_chain`, and inc/decrements it around each generation attempt.

**Tech Stack:** Python (pure), FastAPI coordinator, pytest. Spec: `docs/superpowers/specs/2026-06-20-part3d-load-balancing-design.md`.

## Global Constraints

- `build_chain(stages_by_url, num_layers, exclude=None, load=None)` — `load=None` is byte-for-byte the current behavior (all existing tests preserved). With a `load` map, prefer least-loaded; ties broken by holder insertion order.
- Load lifecycle: increment chain conns at attempt start, decrement in `finally` (no leak on failover/exception).
- `/registry` node entries gain a numeric `load`. No other shape change.

---

### Task 1: `build_chain` load-aware selection + unit tests

**Files:**
- Modify: `eujeno/net/discovery.py`
- Test: `tests/test_load_balancing.py` (new)

**Interfaces:**
- Produces: `build_chain(stages_by_url, num_layers, exclude=None, load=None)`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_load_balancing.py`:

```python
from eujeno.net.discovery import build_chain


def _s(embed=False, head=False, decoders=()):
    return {"embed": embed, "head": head, "decoders": list(decoders)}


def test_default_path_unchanged_with_redundancy():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load=None) == build_chain(s, 24)
    e, chain, h = build_chain(s, 24)
    assert e == "a" and chain == [("0-24", "a")] and h == "b"   # first head, insertion order


def test_prefers_least_loaded_head():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"b": 5, "c": 0})[2] == "c"
    assert build_chain(s, 24, load={"b": 0, "c": 5})[2] == "b"


def test_prefers_least_loaded_embed():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(embed=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"a": 7, "b": 0})[0] == "b"


def test_prefers_least_loaded_decoder_replica():
    s = {"a": _s(embed=True), "b": _s(decoders=["0-24"]), "c": _s(decoders=["0-24"]), "d": _s(head=True)}
    assert build_chain(s, 24, load={"b": 9, "c": 0})[1] == [("0-24", "c")]


def test_load_tie_is_deterministic_insertion_order():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"b": 0, "c": 0})[2] == "b"


def test_incomplete_coverage_still_none_with_load():
    s = {"a": _s(embed=True, decoders=["0-12"]), "b": _s(head=True)}   # 12-24 missing
    assert build_chain(s, 24, load={"a": 0, "b": 0}) is None
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/test_load_balancing.py -q` → FAIL (`build_chain()` has no `load` kwarg).

- [ ] **Step 3: Implement** in `eujeno/net/discovery.py`. Change the signature and append the load post-pass just before `return embed, chain, head`. Replace:

```python
def build_chain(stages_by_url: dict, num_layers: int, exclude=None):
```

with:

```python
def build_chain(stages_by_url: dict, num_layers: int, exclude=None, load=None):
```

and change the final `return embed, chain, head` to:

```python
    if load is not None:
        order = {u: i for i, u in enumerate(items)}
        def _least(cands):
            return min(cands, key=lambda u: (load.get(u, 0), order[u]))
        embed = _least([u for u, s in items.items() if s.get("embed")])
        head = _least([u for u, s in items.items() if s.get("head")])
        chain = [(bk, _least([u for u, s in items.items() if bk in s.get("decoders", [])]))
                 for bk, u in chain]
    return embed, chain, head
```

(`items` is the exclude-filtered dict already built at the top of the function; `order` preserves insertion order for tie-breaks. The default `load=None` path is untouched.)

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/python -m pytest tests/test_load_balancing.py tests/test_discovery.py -q` → PASS (new load tests + existing discovery tests all green).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/discovery.py tests/test_load_balancing.py
git commit -m "feat(net): build_chain load-aware selection (prefer least-loaded replica)"
```

---

### Task 2: coordinator load tracking + registry + e2e assertion

**Files:**
- Modify: `eujeno/net/coordinator.py`
- Test: `tests/test_coordinator_jobs_e2e.py` (add a registry-load assertion to the existing test)

**Interfaces:**
- Consumes: `build_chain(..., load=...)` (Task 1).

- [ ] **Step 1: Add the failing assertion** — in `tests/test_coordinator_jobs_e2e.py`, inside `test_job_is_persisted_and_reconstructible`, after the `/infer` response is obtained and before the `with`-block closes, add:

```python
            reg = httpx.get(f"http://127.0.0.1:{port}/registry").json()
            assert all("load" in n for n in reg["nodes"]), reg
            assert all(n["load"] == 0 for n in reg["nodes"]), reg   # decremented back after completion
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_coordinator_jobs_e2e.py -q` → FAIL (registry entries have no `load` key). (@pytest.mark.slow.)

- [ ] **Step 3: Implement in `eujeno/net/coordinator.py`.**

(a) At registration, add `"load": 0`. Change:

```python
        conns[conn_id] = {"ws": ws, "stages": announce["stages"], "pending": {}}
```

to:

```python
        conns[conn_id] = {"ws": ws, "stages": announce["stages"], "pending": {}, "load": 0}
```

(b) Expose it in `/registry`. Change:

```python
                "nodes": [{"conn": cid, "stages": c["stages"]} for cid, c in conns.items()]}
```

to:

```python
                "nodes": [{"conn": cid, "stages": c["stages"], "load": c["load"]} for cid, c in conns.items()]}
```

(c) Thread load into `build_chain` inside `_await_coverage`. Change:

```python
            chain = build_chain(stages, num_layers)
```

to:

```python
            chain = build_chain(stages, num_layers, load={cid: c["load"] for cid, c in conns.items()})
```

(d) Inc/decrement load around each attempt in `_generate_with_failover`. Replace the body of the `for attempt ...` loop (from `chain = await _await_coverage(...)` through the `_NodeFailure` handler) with:

```python
        for attempt in range(MAX_FAILOVERS + 1):
            chain = await _await_coverage(excluded, job_id)
            if chain is None:
                return None, {"error": "coverage timeout: model not operational", "excluded": sorted(excluded)}
            embed_c, decoders, head_c = chain
            chain_conns = {embed_c, head_c, *(cid for _, cid in decoders)}
            for cid in chain_conns:
                if cid in conns:
                    conns[cid]["load"] += 1
            try:
                tokens, prompt_len, finish_reason = await _run_generation(
                    chain, prompt, max_new, sampling, _next_id("job"),
                    on_token=lambda pos, tok: _store_safe(store.append_token, job_id, tok, pos),
                    resume_tokens=resume_tokens)
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt, "finish_reason": finish_reason}, None
            except _NodeFailure as e:
                excluded.add(e.conn_id)
                last_failed = e.conn_id
                try:
                    j = store.get_job(job_id)
                    resume_tokens = (j or {}).get("tokens", []) or []
                except Exception:
                    resume_tokens = []
            finally:
                for cid in chain_conns:
                    if cid in conns:
                        conns[cid]["load"] = max(0, conns[cid]["load"] - 1)
        return None, {"error": f"too many failovers (last failed node: {last_failed})"}
```

(The `return` inside the `try` runs `finally` first, so load is decremented on success, failover, and exhaustion alike.)

- [ ] **Step 4: Run the e2e + full suite** — `.venv/bin/python -m pytest tests/test_coordinator_jobs_e2e.py -q` → PASS; then `.venv/bin/python -m pytest -q` → all green (existing coordinator/failover/openai e2e assert node counts/tokens, not registry shape, so the added `load` field is compatible).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/coordinator.py tests/test_coordinator_jobs_e2e.py
git commit -m "feat(net): track per-conn load, expose in /registry, balance chains across replicas"
```

---

## Self-Review notes

- **Spec coverage:** coordinator-tracked `load` (Task 2 a/d) · `/registry` exposes load (Task 2 b) · least-loaded selection, default-preserving (Task 1) · load lifecycle inc/dec with `finally` (Task 2 d) · unit tests for selection + default-unchanged + tie + incomplete (Task 1) · registry-load + decremented-to-0 e2e assertion (Task 2). All covered.
- **Placeholder scan:** none — complete code in every step.
- **Type consistency:** `build_chain(..., load=None)` defined (Task 1) and called with `load={cid: c["load"] ...}` (Task 2c); `conns[cid]["load"]` int initialized at registration (Task 2a), inc/dec (Task 2d), read in registry (Task 2b). `chain` unpacks `(embed_c, decoders, head_c)` with `decoders` a list of `(block_key, cid)` — matches `_run_generation`.
