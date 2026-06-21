# Part 3b — Failover Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a node failure mid-generation, resume from the durable job log's tokens-so-far (prefix replay on the surviving holders) instead of restarting from the prompt — realizing PRD Part 3 acceptance #2.

**Architecture:** Harden `jobstore.append_token` to be a true idempotent no-op for the same token (and log a warning on a differing rewrite). Extend the coordinator's `_run_generation` with a `resume_tokens` parameter (prefill `prompt + resume_tokens` to rebuild KV, then continue) and change `_generate_with_failover` to read the persisted tokens on failure and resume instead of `reset_progress`.

**Tech Stack:** Python stdlib `sqlite3`/`logging`, FastAPI/WebSocket coordinator, torch, pytest. Spec: `docs/superpowers/specs/2026-06-20-part3b-failover-resume-design.md`.

## Global Constraints

- Resume, don't restart: `_generate_with_failover` no longer calls `reset_progress`; it carries `resume_tokens` across attempts, read from the durable store on failure (read wrapped → `[]` on error).
- Greedy (temperature 0) determinism: a resumed continuation must reproduce the same sequence → `== golden`.
- No request/response shape change, no node-protocol change. `prompt_len` stays the prompt length.
- Existing `tests/test_failover_e2e.py` must stay green.

---

### Task 1: `append_token` idempotency hardening + tests

**Files:**
- Modify: `eujeno/net/jobstore.py`
- Test: `tests/test_jobstore.py` (add two tests)

**Interfaces:**
- Unchanged signature `append_token(job_id, token_id, position)`; new behavior: same token at an existing position → no-op; different token at an existing position → `log.warning` then overwrite.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_jobstore.py`:

```python
def test_append_same_token_same_position_is_strict_noop(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    before = s.get_job("j1")["updated_at"]
    s.append_token("j1", 10, 0)        # exact same -> no-op
    j = s.get_job("j1")
    assert j["tokens"] == [10]
    assert j["updated_at"] == before   # no write happened


def test_append_different_token_at_existing_position_warns(tmp_path, caplog):
    import logging
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    with caplog.at_level(logging.WARNING, logger="eujeno.jobstore"):
        s.append_token("j1", 99, 0)
    assert s.get_job("j1")["tokens"] == [99]
    assert any("rewritten" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_jobstore.py -q`
Expected: FAIL — the no-op test fails (current code rewrites + bumps `updated_at`) and the warning test fails (no warning emitted).

- [ ] **Step 3: Implement the change in `eujeno/net/jobstore.py`**

Add to the imports at the top (after `import json`):

```python
import logging
```

After the imports / before `_SCHEMA`, add the module logger:

```python
log = logging.getLogger("eujeno.jobstore")
```

Replace the `append_token` method body's branch logic. Change the current:

```python
        toks = json.loads(row["tokens_json"] or "[]")
        if position < len(toks):
            toks[position] = int(token_id)        # re-apply same position -> idempotent, no double
        elif position == len(toks):
            toks.append(int(token_id))
        else:
            return                                # out-of-order beyond next: ignore (not expected)
```

to:

```python
        toks = json.loads(row["tokens_json"] or "[]")
        if position < len(toks):
            if toks[position] == int(token_id):
                return                            # strict idempotent no-op: same token, same position
            log.warning("append_token: job %s position %d rewritten %s -> %s",
                        job_id, position, toks[position], int(token_id))
            toks[position] = int(token_id)
        elif position == len(toks):
            toks.append(int(token_id))
        else:
            return                                # out-of-order beyond next: ignore (not expected)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jobstore.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add eujeno/net/jobstore.py tests/test_jobstore.py
git commit -m "feat(net): strict idempotent append_token (no-op on same token, warn on rewrite)"
```

---

### Task 2: Coordinator resume-from-persisted-state failover + e2e

**Files:**
- Modify: `eujeno/net/coordinator.py` (`_run_generation`, `_generate_with_failover`)
- Test: `tests/test_failover_resume_e2e.py`

**Interfaces:**
- Consumes: `JobStore.get_job`/`append_token` (Task 1), `_store_safe`, `build_chain`.
- Produces: `_run_generation(chain, prompt, max_new, sampling, job_id, on_token=None, resume_tokens=None)`.

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_failover_resume_e2e.py`:

```python
import socket, threading, time, asyncio
import pytest, httpx, uvicorn, websockets

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


def _thread(coro_factory):
    threading.Thread(target=lambda: asyncio.run(coro_factory()), daemon=True).start()


def _wait_count(client, base, n):
    for _ in range(200):
        if len(client.get(f"{base}/registry").json()["nodes"]) == n:
            return
        time.sleep(0.05)
    raise AssertionError(f"registry never reached {n} nodes")


async def _run_flaky_after(ws_url, state, die_after_decodes):
    """Serve normally, but close the connection on the Nth 'decode' op (crash after some tokens)."""
    seen = {"n": 0}
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(pack({"type": "announce", "stages": state.stages_dict()}))
        loop = asyncio.get_running_loop()
        async for message in ws:
            header, payload = unpack(message)
            if header["op"] == "decode":
                seen["n"] += 1
                if seen["n"] >= die_after_decodes:
                    await ws.close()
                    return
            rh, rp = await loop.run_in_executor(None, handle_request, state, header, payload)
            await ws.send(pack({**rh, "req_id": header.get("req_id")}, rp))


@pytest.mark.slow
def test_failover_resumes_from_persisted_tokens(full_model, tmp_path):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    db = str(tmp_path / "jobs.db")
    port = _free_port()
    server = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer, db_path=db), port)
    ws_url = f"ws://127.0.0.1:{port}/node"
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=60.0) as client:
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)]))))
            _wait_count(client, base, 1)
            # tail node that dies on the 4th decode -> ~3 tokens already persisted when it fails
            _thread(lambda: _run_flaky_after(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])), 4))
            _wait_count(client, base, 2)
            # redundant tail node to fail over to
            _thread(lambda: run_node(ws_url, NodeState(model, StageSpec(head=True, decoders=[(12, 24)]))))
            _wait_count(client, base, 3)

            data = client.post(f"{base}/infer", json={"prompt": prompt, "max_new_tokens": 6}).json()
            jobs = client.get(f"{base}/jobs").json()["jobs"]
            detail = client.get(f"{base}/jobs/{jobs[0]['job_id']}").json()
        assert data["ok"] is True, data
        assert data["tokens"] == reference          # resume reproduced the golden sequence
        assert data["failovers"] >= 1
        assert detail["status"] == "DONE"
        assert detail["tokens"] == reference         # durable log drove a correct resume
    finally:
        server.should_exit = True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_failover_resume_e2e.py -q`
Expected: FAIL — current `_run_generation` has no `resume_tokens`; failover still restarts (and may also assert-fail on the durable-log expectations).

- [ ] **Step 3: Add `resume_tokens` to `_run_generation`**

In `eujeno/net/coordinator.py`, replace the whole `_run_generation` function with:

```python
    async def _run_generation(chain, prompt, max_new, sampling, job_id, on_token=None, resume_tokens=None):
        embed_c, decoders, head_c = chain
        temperature = float(sampling.get("temperature", 0.0) or 0.0)
        top_p = float(sampling.get("top_p", 1.0) or 1.0)
        rep = float(sampling.get("repetition_penalty", 1.0) or 1.0)
        do_sample = temperature > 0.0
        generator = None
        if do_sample:
            seed = sampling.get("seed")
            seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
            generator = torch.Generator().manual_seed(seed)
        topk = 100 if do_sample else 1

        resume_tokens = list(resume_tokens or [])
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = ids.shape[1]
        if resume_tokens:
            cur = torch.cat([ids, torch.tensor([resume_tokens], dtype=ids.dtype)], dim=1)
            cache_position = torch.arange(seq_len + len(resume_tokens))
        else:
            cur = ids
            cache_position = torch.arange(seq_len)
        tokens = list(resume_tokens)
        finish_reason = "length"
        for _ in range(max_new - len(resume_tokens)):
            _, p = await _call(embed_c, {"op": "embed", "job_id": job_id},
                               encode_tensors({"input_ids": cur}))
            h = decode_tensors(p)["hidden_states"]
            for block_key, cid in decoders:
                _, p = await _call(cid, {"op": "decode", "block_key": block_key, "job_id": job_id},
                                   encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                h = decode_tensors(p)["hidden_states"]
            rh, _ = await _call(head_c, {"op": "head", "job_id": job_id, "topk": topk},
                                encode_tensors({"hidden_states": h}))
            tok = sample_token(rh["topk_ids"], rh["topk_logits"], tokens, temperature, top_p, rep, generator) if do_sample else rh["token_id"]
            if tok in stop_ids:
                finish_reason = "stop"
                break
            tokens.append(tok)
            if on_token is not None:
                on_token(len(tokens) - 1, tok)
            cur = torch.tensor([[tok]])
            cache_position = torch.tensor([seq_len + len(tokens) - 1])
        for cid in {embed_c, head_c, *(c for _, c in decoders)}:
            try:
                await _call(cid, {"op": "end", "job_id": job_id})
            except _NodeFailure:
                pass
        return tokens, seq_len, finish_reason
```

(The only changes vs the current version: the `resume_tokens` parameter; the initial `cur`/`cache_position` prefix handling; the loop bound `max_new - len(resume_tokens)`; `tokens = list(resume_tokens)`; and `cache_position = [seq_len + len(tokens) - 1]` — which equals the old `seq_len + step` on the non-resume path.)

- [ ] **Step 4: Change `_generate_with_failover` to resume from the durable log**

Replace the whole `_generate_with_failover` function with:

```python
    async def _generate_with_failover(prompt, max_new, sampling, job_id):
        excluded = set()
        last_failed = None
        resume_tokens = []
        for attempt in range(MAX_FAILOVERS + 1):
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is None:
                return None, {"error": "model not operational: incomplete coverage", "excluded": sorted(excluded)}
            try:
                tokens, prompt_len, finish_reason = await _run_generation(
                    chain, prompt, max_new, sampling, _next_id("job"),
                    on_token=lambda pos, tok: _store_safe(store.append_token, job_id, tok, pos),
                    resume_tokens=resume_tokens)
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt, "finish_reason": finish_reason}, None
            except _NodeFailure as e:
                excluded.add(e.conn_id)
                last_failed = e.conn_id
                try:                                   # re-dispatch from the persisted progress
                    j = store.get_job(job_id)
                    resume_tokens = (j or {}).get("tokens", []) or []
                except Exception:
                    resume_tokens = []
        return None, {"error": f"too many failovers (last failed node: {last_failed})"}
```

(Removes the `_store_safe(store.reset_progress, job_id)` line; adds the `resume_tokens` carry + the wrapped read of persisted tokens on failure.)

- [ ] **Step 5: Run the new e2e + the full suite**

Run: `.venv/bin/python -m pytest tests/test_failover_resume_e2e.py tests/test_failover_e2e.py -q`
Expected: PASS (2 passed) — the new resume test and the existing failover test both green.

Run: `.venv/bin/python -m pytest -q`
Expected: all green (no regressions).

- [ ] **Step 6: Commit**

```bash
git add eujeno/net/coordinator.py tests/test_failover_resume_e2e.py
git commit -m "feat(net): failover resumes from persisted tokens (prefix replay) instead of restart"
```

---

## Self-Review notes

- **Spec coverage:** resume-not-restart (Task 2 `_generate_with_failover`) · prefix replay rebuilds KV (Task 2 `_run_generation` resume_tokens) · read-the-log-on-failure wrapped (Task 2) · idempotency hardening / deferred 3a Minor (Task 1) · acceptance #2 node-crash-mid-job → completes == golden (Task 2 e2e) · existing failover test stays green (Task 2 Step 5). All covered.
- **Placeholder scan:** none — complete code in every step.
- **Type consistency:** `_run_generation(..., on_token=None, resume_tokens=None)` signature used consistently; `_generate_with_failover` passes `resume_tokens=resume_tokens`; `store.get_job(...)["tokens"]` matches the JobStore dict shape from 3a. `cache_position = seq_len + len(tokens) - 1` is equivalent to the old `seq_len + step` on the non-resume path (verified: after appending token k, len(tokens)=k+1 ⇒ seq_len+k).
