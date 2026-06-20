# P2Pa — Pure-P2P Failover + EOS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pure-P2P inference (`infer --peer`, no coordinator) fault-tolerant — re-route around a dead peer and resume from tokens-so-far (prefix replay), and stop at EOS.

**Architecture:** A new `distributed_generate_resilient` in the orchestrator builds the chain from the gossip registry, drives greedy HTTP generation, and on a hop failure excludes the dead peer URL, rebuilds (`build_chain(exclude=...)`), and resumes (prefill prompt+tokens). A shared `stop_token_ids` helper provides EOS. `infer --peer` uses it.

**Tech Stack:** Python, httpx, torch, FastAPI/Starlette (test middleware), pytest. Spec: `docs/superpowers/specs/2026-06-20-p2pa-failover-design.md`.

## Global Constraints

- `distributed_generate` (static `--topology`) stays unchanged. Only the registry-driven `--peer` path gains failover.
- Resume = prefix replay (prefill `prompt + tokens_so_far`, fresh `job_id`), same scheme as Part 3b; cache_position `= [seq_len + len(tokens) - 1]` per step (matches the non-resume path).
- Greedy (the `--peer` direct path stays greedy; the `/v1/chat/completions` node entry already samples).
- `max_failovers = 5`.

---

### Task 1: `stop_token_ids` + `distributed_generate_resilient` + e2e

**Files:**
- Modify: `eujeno/net/generation.py` (add `stop_token_ids`)
- Modify: `eujeno/net/orchestrator.py` (add `distributed_generate_resilient`)
- Test: `tests/test_p2p_failover_e2e.py` (new)

**Interfaces:**
- Produces: `stop_token_ids(tokenizer) -> set[int]`; `distributed_generate_resilient(stages_by_url, num_layers, prompt, max_new_tokens, client, tokenizer, stop_ids=None, job_id_prefix="job", refresh=None, max_failovers=5) -> dict` (`{"ok","text","tokens","failovers","finish_reason"}` or `{"ok":False,"error","tokens","failovers"}`).

- [ ] **Step 1: Write the failing e2e** — create `tests/test_p2p_failover_e2e.py`:

```python
import socket, threading, time
import pytest, httpx, uvicorn
from starlette.responses import JSONResponse

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


class _FlakyDecode:
    """ASGI wrapper: 503 on /decode after `die_after` calls; everything else (lifespan,
    gossip, /embed, /head, /registry) passes through normally."""
    def __init__(self, app, die_after):
        self.app = app; self.n = 0; self.die_after = die_after

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path", "").startswith("/decode"):
            self.n += 1
            if self.n >= self.die_after:
                await JSONResponse({"error": "dead"}, status_code=503)(scope, receive, send)
                return
        await self.app(scope, receive, send)


@pytest.mark.slow
def test_p2p_failover_reroutes_and_resumes(full_model):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    pA, pB, pC = _free_port(), _free_port(), _free_port()
    uA, uB, uC = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}", f"http://127.0.0.1:{pC}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB, uC], num_layers=24, gossip_interval=0.3), pA)
    sB = _serve(_FlakyDecode(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                             node_url=uB, peers=[uA, uC], num_layers=24, gossip_interval=0.3), die_after=4), pB)
    sC = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uC, peers=[uA, uB], num_layers=24, gossip_interval=0.3), pC)
    try:
        # ordered so build_chain picks B (flaky) first, then fails over to C
        stages = {uA: {"embed": True, "head": False, "decoders": ["0-12"]},
                  uB: {"embed": False, "head": True, "decoders": ["12-24"]},
                  uC: {"embed": False, "head": True, "decoders": ["12-24"]}}
        with httpx.Client(timeout=60.0) as client:
            result = distributed_generate_resilient(stages, 24, prompt, 6, client, tokenizer,
                                                    stop_ids=stop_token_ids(tokenizer))
        assert result["ok"] is True, result
        assert result["tokens"] == reference
        assert result["failovers"] >= 1            # B died mid-generation, C resumed
    finally:
        sA.should_exit = sB.should_exit = sC.should_exit = True


@pytest.mark.slow
def test_p2p_stops_at_eos(full_model):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3), pA)
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    try:
        stages = {uA: {"embed": True, "head": False, "decoders": ["0-12"]},
                  uB: {"embed": False, "head": True, "decoders": ["12-24"]}}
        with httpx.Client(timeout=60.0) as client:
            # make the very first token a stop token -> generation must stop immediately
            result = distributed_generate_resilient(stages, 24, prompt, 6, client, tokenizer,
                                                    stop_ids={reference[0]})
        assert result["ok"] is True
        assert result["tokens"] == []
        assert result["finish_reason"] == "stop"
        assert stop_token_ids(tokenizer) and tokenizer.eos_token_id in stop_token_ids(tokenizer)
    finally:
        sA.should_exit = sB.should_exit = True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest tests/test_p2p_failover_e2e.py -q` → FAIL (`distributed_generate_resilient` / `stop_token_ids` don't exist). (@pytest.mark.slow; loads 0.5B; a few min.)

- [ ] **Step 3: Add `stop_token_ids` to `eujeno/net/generation.py`** (after the imports / before `generate_tokens`):

```python
def stop_token_ids(tokenizer):
    """EOS + chat-end special tokens to stop generation on."""
    ids = set()
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id:
            ids.add(int(i))
    return ids
```

- [ ] **Step 4: Add `distributed_generate_resilient` to `eujeno/net/orchestrator.py`** (keep `distributed_generate` as-is; add this function below it):

```python
def distributed_generate_resilient(stages_by_url, num_layers, prompt, max_new_tokens, client,
                                   tokenizer, stop_ids=None, job_id_prefix="job",
                                   refresh=None, max_failovers=5):
    """Pure-P2P entry: greedy distributed generation with failover. On a peer hop failure,
    exclude that peer, rebuild the chain from the gossip registry, and resume from the
    tokens already produced (prefix replay). Stops at EOS (stop_ids)."""
    from eujeno.net.discovery import build_chain
    stop_ids = stop_ids or set()
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    tokens = []
    excluded = set()
    finish_reason = "length"
    for attempt in range(max_failovers + 1):
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
        embed_url, decoders, head_url = chain
        job_id = f"{job_id_prefix}{attempt}"
        current = None
        try:
            if tokens:
                cur_ids = torch.cat([ids, torch.tensor([tokens], dtype=ids.dtype)], dim=1)
                cache_position = torch.arange(seq_len + len(tokens))
            else:
                cur_ids = ids
                cache_position = torch.arange(seq_len)
            while len(tokens) < max_new_tokens:
                current = embed_url
                r = client.post(f"{embed_url}/embed", params={"job_id": job_id},
                                content=encode_tensors({"input_ids": cur_ids})); r.raise_for_status()
                h = decode_tensors(r.content)["hidden_states"]
                for block_key, url in decoders:
                    current = url
                    r = client.post(f"{url}/decode/{block_key}", params={"job_id": job_id},
                                    content=encode_tensors({"hidden_states": h, "cache_position": cache_position})); r.raise_for_status()
                    h = decode_tensors(r.content)["hidden_states"]
                current = head_url
                r = client.post(f"{head_url}/head", params={"job_id": job_id},
                                content=encode_tensors({"hidden_states": h})); r.raise_for_status()
                token_id = r.json()["token_id"]
                if token_id in stop_ids:
                    finish_reason = "stop"
                    break
                tokens.append(token_id)
                cur_ids = torch.tensor([[token_id]])
                cache_position = torch.tensor([seq_len + len(tokens) - 1])
            for url in {embed_url, head_url, *(u for _, u in decoders)}:
                try:
                    client.delete(f"{url}/job/{job_id}")
                except Exception:
                    pass
            return {"ok": True, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "tokens": tokens, "failovers": attempt, "finish_reason": finish_reason}
        except Exception:
            if current is not None:
                excluded.add(current)
    return {"ok": False, "error": "too many failovers", "tokens": tokens, "failovers": max_failovers}
```

- [ ] **Step 5: Run the e2e** — `.venv/bin/python -m pytest tests/test_p2p_failover_e2e.py -q` → PASS (2 passed).

- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest -q` → all green.

- [ ] **Step 7: Commit**

```bash
git add eujeno/net/generation.py eujeno/net/orchestrator.py tests/test_p2p_failover_e2e.py
git commit -m "feat(net): resilient P2P generation (failover + resume + EOS), stop_token_ids helper"
```

---

### Task 2: wire `infer --peer` to the resilient path

**Files:**
- Modify: `eujeno/cli.py` (the `infer` `--peer` branch)

**Interfaces:**
- Consumes: `distributed_generate_resilient`, `stop_token_ids` (Task 1).

- [ ] **Step 1: Implement** — in `eujeno/cli.py`, change the `if peer:` branch. Replace the current body (which builds a `Topology` and calls `distributed_generate(topo, ...)`) with:

```python
    if peer:
        peer = peer.rstrip("/")
        try:
            reg = httpx.get(f"{peer}/registry", timeout=30.0).json()
        except Exception as e:
            _fail("infer", "USAGE_ERROR", f"peer unreachable: {e}", exit_code=2)
        from eujeno.net.orchestrator import distributed_generate_resilient
        from eujeno.net.generation import stop_token_ids
        try:
            tokenizer = AutoTokenizer.from_pretrained(reg["model"])
            stop_ids = stop_token_ids(tokenizer)
            def _refresh():
                return httpx.get(f"{peer}/registry", timeout=10.0).json()["nodes"]
            with httpx.Client(timeout=120.0) as client:
                result = distributed_generate_resilient(
                    reg["nodes"], reg["num_layers"], prompt, max_new_tokens, client, tokenizer,
                    stop_ids=stop_ids, refresh=_refresh)
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", str(e))
        if not result.get("ok"):
            _fail("infer", "NOT_OPERATIONAL", result.get("error", "the model is not operational on the network yet"))
        _emit_ok("infer", {"model": reg["model"], "prompt": prompt,
                           "text": result["text"], "tokens": result["tokens"],
                           "failovers": result["failovers"]}, human=result["text"])
        return
```

(`distributed_generate` and the `Topology` import remain used by the `--topology` path below, unchanged.)

- [ ] **Step 2: Verify the existing P2P CLI path still works** — `.venv/bin/python -m pytest tests/test_infer_peer.py tests/test_gossip_e2e.py -q` → PASS (the happy `infer --peer` still auto-discovers and matches reference; now also EOS-aware — `max_new_tokens=6` is short enough not to hit EOS for the test prompt).

- [ ] **Step 3: Full suite** — `.venv/bin/python -m pytest -q` → all green.

- [ ] **Step 4: Commit**

```bash
git add eujeno/cli.py
git commit -m "feat(cli): infer --peer uses resilient P2P generation (failover + EOS)"
```

---

## Self-Review notes

- **Spec coverage:** failover re-route + resume (Task 1 `distributed_generate_resilient`) · EOS stop + `stop_token_ids` (Task 1) · `infer --peer` wired with refresh (Task 2) · static `--topology` path untouched · failover e2e (re-route + prefix-replay resume via FlakyDecode) + EOS e2e (Task 1) · regression test_infer_peer/gossip (Task 2). All covered.
- **Placeholder scan:** none — complete code in every step.
- **Type consistency:** `distributed_generate_resilient(stages_by_url, num_layers, ...)` defined (Task 1) and called with `reg["nodes"], reg["num_layers"]` (Task 2); returns `{"ok","text","tokens","failovers","finish_reason"}`. `stop_token_ids(tokenizer)` defined + used in both. `build_chain(..., exclude=...)` exists (Part 1). cache_position scheme matches 3b.
