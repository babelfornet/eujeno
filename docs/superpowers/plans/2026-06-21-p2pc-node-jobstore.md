# P2Pc — Per-node Job Log + Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A serve node acting as the `/v1/chat/completions` entry records the job + per-peer receipts in its own durable `JobStore`, exposed via `/jobs[/{id}]` and `/jobs/{id}/receipts`.

**Architecture:** Reuse the `JobStore` (Part 3a/4b) per node. `create_app(db_path=None)` → `:memory:` (default) or a file (`serve --db`). The `/v1` entry logs `create_job`/`finish`/`fail` and accumulates per-peer receipts in its hop closures, persisted at completion. Best-effort writes.

**Tech Stack:** Python, FastAPI, httpx, pytest. Spec: `docs/superpowers/specs/2026-06-21-p2pc-node-jobstore-design.md`.

## Global Constraints

- `create_app(..., db_path=None)` → `JobStore(":memory:")` by default (existing callers unchanged; new endpoints additive). `serve --db PATH` opts into on-disk durability.
- Entry-side coarse log (create→finish/fail; no per-token). Only embed/decode/head hops billed (not `DELETE /job`). Persistence is best-effort (`_store_safe`).
- `/v1` response shape unchanged.

---

### Task 1: per-node JobStore + receipts in the `/v1` entry + endpoints + cli `--db` + e2e

**Files:**
- Modify: `eujeno/net/server.py`
- Modify: `eujeno/cli.py` (`serve` — add `--db`)
- Test: `tests/test_p2p_node_jobstore_e2e.py` (new)

**Interfaces:**
- Produces: `create_app(model, tokenizer, stages, node_url=None, peers=None, num_layers=None, gossip_interval=2.0, ttl=30.0, db_path=None)`; node routes `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/receipts`.

- [ ] **Step 1: Write the failing e2e** — create `tests/test_p2p_node_jobstore_e2e.py`:

```python
import socket, threading, time
import pytest, httpx, uvicorn

from eujeno.net.server import create_app
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


@pytest.mark.slow
def test_p2p_node_logs_job_and_receipts(full_model, tmp_path):
    model, tokenizer = full_model
    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    # A is the entry node (has a durable db); both cover the model together
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3,
                           db_path=str(tmp_path / "nodeA.db")), pA)
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    try:
        with httpx.Client(timeout=120.0) as client:
            for _ in range(100):
                if set(client.get(f"{uA}/registry").json()["nodes"].keys()) == {uA, uB}:
                    break
                time.sleep(0.1)
            resp = client.post(f"{uA}/v1/chat/completions",
                               json={"messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 5}).json()
            assert resp["choices"][0]["message"] is not None, resp
            jobs = client.get(f"{uA}/jobs").json()["jobs"]
            assert len(jobs) >= 1
            jid = jobs[0]["job_id"]
            detail = client.get(f"{uA}/jobs/{jid}").json()
            receipts = client.get(f"{uA}/jobs/{jid}/receipts").json()["receipts"]
        assert detail["status"] == "DONE"
        assert detail["result"] is not None
        assert len(receipts) >= 1
        assert sum(r["hops"] for r in receipts) > 0
        assert all(r["bytes"] > 0 for r in receipts)
    finally:
        sA.should_exit = sB.should_exit = True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_p2p_node_jobstore_e2e.py -q` → FAIL (`create_app` has no `db_path`; no `/jobs` route). (@pytest.mark.slow.)

- [ ] **Step 3: Wire `JobStore` into `eujeno/net/server.py`.**

(a) Top of file: add `import logging` (with the other stdlib imports) and `from eujeno.net.jobstore import JobStore` (with the other `eujeno.net` imports).

(b) Signature — change:
```python
def create_app(model, tokenizer, stages, node_url=None, peers=None,
               num_layers=None, gossip_interval=2.0, ttl=30.0):
```
to:
```python
def create_app(model, tokenizer, stages, node_url=None, peers=None,
               num_layers=None, gossip_interval=2.0, ttl=30.0, db_path=None):
```

(c) After `_entry_job = {"n": 0}`, add the store + best-effort helper:
```python
    store = JobStore(db_path if db_path is not None else ":memory:")
    store.recover()

    def _store_safe(fn, *args):
        try:
            fn(*args)
        except Exception as e:
            logging.getLogger("eujeno.node").warning("jobstore write failed: %s", e)
```

(d) Replace the body of `v1_chat` from `_entry_job["n"] += 1` through the final `return {...}` with the instrumented version:
```python
        _entry_job["n"] += 1
        job_id = f"entry{_entry_job['n']}"
        prompt_len0 = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        _store_safe(store.create_job, job_id, _model_id, prompt, sampling, prompt_len0)
        receipts = {}

        def _rcacc(url, sent, recv, dt):
            r = receipts.setdefault(url, {"hops": 0, "bytes": 0, "t_compute": 0.0})
            r["hops"] += 1
            r["bytes"] += sent + recv
            r["t_compute"] += dt

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async def run_embed(cur):
                    payload = encode_tensors({"input_ids": cur}); t0 = time.monotonic()
                    r = await client.post(f"{embed_url}/embed", params={"job_id": job_id}, content=payload)
                    _rcacc(embed_url, len(payload), len(r.content), time.monotonic() - t0)
                    return decode_tensors(r.content)["hidden_states"]

                async def run_decoders(h, cache_position):
                    for bk, url in decoders:
                        payload = encode_tensors({"hidden_states": h, "cache_position": cache_position}); t0 = time.monotonic()
                        r = await client.post(f"{url}/decode/{bk}", params={"job_id": job_id}, content=payload)
                        _rcacc(url, len(payload), len(r.content), time.monotonic() - t0)
                        h = decode_tensors(r.content)["hidden_states"]
                    return h

                async def run_head(h, topk):
                    payload = encode_tensors({"hidden_states": h}); t0 = time.monotonic()
                    r = await client.post(f"{head_url}/head", params={"job_id": job_id, "topk": topk}, content=payload)
                    _rcacc(head_url, len(payload), len(r.content), time.monotonic() - t0)
                    return r.json()

                tokens, prompt_len, finish_reason = await generate_tokens(
                    tokenizer, prompt, max_new, sampling, stop_ids, run_embed, run_decoders, run_head)
                for url in {embed_url, head_url, *(u for _, u in decoders)}:
                    try:
                        await client.delete(f"{url}/job/{job_id}")
                    except Exception:
                        pass
        except Exception as e:
            _store_safe(store.fail, job_id, str(e))
            return JSONResponse({"error": {"message": str(e), "type": "generation_failed"}}, status_code=502)

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        _store_safe(store.finish, job_id, text, finish_reason)
        _store_safe(store.add_receipts, job_id, receipts)
        content, tool_calls = extract_tool_calls(text)
        message = {"role": "assistant", "content": (content if content else None)}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return {"id": "chatcmpl-" + job_id, "object": "chat.completion", "created": int(time.time()),
                "model": _model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(tokens),
                          "total_tokens": prompt_len + len(tokens)}}
```

(e) Add the read endpoints just before `return app`:
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

    @app.get("/jobs/{job_id}/receipts")
    async def get_receipts(job_id: str):
        return {"receipts": store.get_receipts(job_id)}
```

- [ ] **Step 4: Add `--db` to the CLI `serve` command in `eujeno/cli.py`.** Add to the `serve` signature:
```python
    db: str = typer.Option(None, "--db", help="[P2P] SQLite job-log path for this node (default: in-memory)"),
```
and change the `create_app(...)` call (`eujeno/cli.py:320`) to pass it:
```python
    fastapi_app = create_app(model, tokenizer, spec, node_url=own_url, peers=seeds, num_layers=nl, db_path=db)
```

- [ ] **Step 5: Run the e2e** — `.venv/bin/python -m pytest tests/test_p2p_node_jobstore_e2e.py -q` → PASS.

- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest -q` → all green (existing P2P entry/gossip/infer-peer tests use default `:memory:` and the unchanged `/v1` shape).

- [ ] **Step 7: Commit** (ONLY these files; do NOT `git add -A` — leave untracked `web/`/`.github/` alone)
```bash
git add eujeno/net/server.py eujeno/cli.py tests/test_p2p_node_jobstore_e2e.py
git commit -m "feat(net): per-node job log + receipts on P2P serve nodes (/jobs, /jobs/{id}/receipts) + serve --db"
```

---

## Self-Review notes

- **Spec coverage:** per-node JobStore (`db_path`, default `:memory:`) + recover (Task 1c) · entry-side create/finish/fail (Task 1d) · per-peer receipts via instrumented closures, persisted at completion (Task 1d) · `/jobs[/{id}]` + `/jobs/{id}/receipts` (Task 1e) · `serve --db` (Task 1 step 4) · acceptance-#3 e2e (Task 1 step 1). All covered.
- **Placeholder scan:** none — complete code.
- **Type consistency:** `JobStore` reused (Part 3a/4b: `create_job/finish/fail/add_receipts/get_receipts/get_job/recent_jobs/recover`). receipts shape `{url: {hops,bytes,t_compute}}` matches `add_receipts`. New routes `/jobs*` distinct from the existing `DELETE /job/{job_id}`. `create_app(db_path=None)` keeps all existing callers (tests, cli) working (in-memory) except the cli which now passes `db_path=db`.
