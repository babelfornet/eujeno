# OpenAI-compatible API on the coordinator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Expose an **OpenAI-compatible** endpoint on the coordinator (`/v1/chat/completions` + `/v1/models`) so that any OpenAI-compatible client/agent can query Eujeno. Includes a **chat template** (messages → prompt) and **sampling** (temperature/top_p/repetition_penalty), which also fix the quality issues of greedy decoding.

**Architecture:** The `head` node exposes the **top-k** logits (in addition to the argmax, for greedy backward compatibility). The coordinator samples (pure helper `sample_token`) and drives generation with the decoding parameters; it applies the tokenizer's chat template to the `messages`; it reuses the existing failover loop. The `/v1/*` endpoints map request/response to the OpenAI format.

**Tech Stack:** Python · FastAPI · torch · transformers (chat template) · the existing `eujeno/net/{coordinator,node_exec,framing,wire,discovery}.py`.

**Out of scope (follow-up):** SSE streaming; Anthropic `/v1/messages` endpoint / LiteLLM config for Claude Code; queue & load balancing for many concurrent agents.

---

## File Structure

```
eujeno/net/sampling.py         # NEW: sample_token() (pure, testable)
eujeno/net/node_exec.py        # MOD: head also returns topk_ids/topk_logits
eujeno/net/coordinator.py      # MOD: generation refactor + sampling + chat template + /v1 endpoints
tests/
  test_sampling.py              # deterministic sample_token (fast)
  test_openai_e2e.py            # /v1/chat/completions over 2 nodes (slow)
docs/examples/agents.md         # NEW: connecting OpenAI clients / Claude Code
docs/ROADMAP.md
```

---

## Task 1: `sample_token` + head top-k

**Files:** create `eujeno/net/sampling.py`; modify `eujeno/net/node_exec.py`; create `tests/test_sampling.py`; (head already covered by test_node_exec — stays green).

- [ ] **Step 1: test `tests/test_sampling.py`**
```python
import torch
from eujeno.net.sampling import sample_token


def test_greedy_returns_argmax_when_temperature_zero():
    ids = [10, 20, 30]
    logits = [1.0, 5.0, 2.0]
    out = sample_token(ids, logits, generated_ids=[], temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0, generator=None)
    assert out == 20   # maximum logit


def test_sampling_is_deterministic_with_seed():
    ids = [10, 20, 30, 40]
    logits = [2.0, 2.0, 2.0, 2.0]
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_token(ids, logits, [], 1.0, 1.0, 1.0, g1)
    b = sample_token(ids, logits, [], 1.0, 1.0, 1.0, g2)
    assert a == b and a in ids


def test_repetition_penalty_demotes_generated_tokens():
    ids = [10, 20]
    logits = [5.0, 1.0]
    # with a strong penalty on token 10 (already generated), 20 must become the greedy argmax
    out = sample_token(ids, logits, generated_ids=[10], temperature=0.0,
                       top_p=1.0, repetition_penalty=10.0, generator=None)
    assert out == 20
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_sampling.py -v` → ImportError.

- [ ] **Step 3: implement `eujeno/net/sampling.py`**
```python
import torch


def sample_token(topk_ids, topk_logits, generated_ids, temperature, top_p,
                 repetition_penalty, generator) -> int:
    """Pick the next token from the head node's top-k candidates.
    temperature<=0 -> greedy (argmax). Otherwise: repetition penalty, temperature,
    top_p nucleus, multinomial sampling (deterministic if `generator` has a seed)."""
    logits = torch.tensor(topk_logits, dtype=torch.float32)
    ids = list(topk_ids)
    if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
        gen = set(generated_ids)
        for i, tid in enumerate(ids):
            if tid in gen:
                logits[i] = logits[i] / repetition_penalty if logits[i] > 0 else logits[i] * repetition_penalty
    if temperature is None or temperature <= 0:
        return ids[int(torch.argmax(logits))]
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = (cum - sp) <= top_p
    keep[0] = True
    sp = sp * keep
    sp = sp / sp.sum()
    choice = int(torch.multinomial(sp, 1, generator=generator).item())
    return ids[int(si[choice])]
```

- [ ] **Step 4: modify the `head` branch of `handle_request` in `eujeno/net/node_exec.py`**

Replace the `if op == "head":` block with the following (it also returns the top-k, keeping `token_id` for greedy backward compatibility):
```python
    if op == "head":
        t = decode_tensors(payload)
        logits = state.head_block.run_block(t["hidden_states"])[:, -1, :]
        k = int(header.get("topk", 1))
        k = min(k, logits.shape[-1])
        vals, idx = torch.topk(logits[0], k=k)
        ids = idx.tolist()
        return {"ok": True, "token_id": ids[0],
                "topk_ids": ids, "topk_logits": vals.tolist()}, b""
```
Add `import torch` at the top of `node_exec.py` if not already present.

- [ ] **Step 5: run PASS** — `... pytest tests/test_sampling.py tests/test_node_exec.py -m "slow or not slow" -v`. Expected: sampling 3 passed; test_node_exec still green (token_id unchanged).

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/sampling.py eujeno/net/node_exec.py tests/test_sampling.py && git commit -m "feat(net): sample_token + head exposes top-k logits (sampling)"
```

---

## Task 2: coordinator — parametric generation + sampling in /infer

**Files:** modify `eujeno/net/coordinator.py`; create test in `tests/test_openai_e2e.py` (sampling part).

- [ ] **Step 1: add a deterministic sampling test to `tests/test_openai_e2e.py`**
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
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started: break
        time.sleep(0.05)
    assert srv.started
    return srv


def _node(ws, state):
    threading.Thread(target=lambda: asyncio.run(run_node(ws, state)), daemon=True).start()


def _two_node_coordinator(full_model):
    model, tokenizer = full_model
    port = _free_port()
    srv = _serve(create_coordinator_app("Qwen/Qwen2.5-0.5B-Instruct", 24, tokenizer), port)
    ws = f"ws://127.0.0.1:{port}/node"
    _node(ws, NodeState(model, StageSpec(embed=True, decoders=[(0, 12)])))
    _node(ws, NodeState(model, StageSpec(head=True, decoders=[(12, 24)])))
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=30.0) as c:
        for _ in range(200):
            if len(c.get(f"{base}/registry").json()["nodes"]) == 2: break
            time.sleep(0.05)
    return srv, base


@pytest.mark.slow
def test_infer_sampling_seeded_is_reproducible(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            body = {"prompt": "Hi", "max_new_tokens": 6, "temperature": 0.8, "top_p": 0.9, "seed": 42}
            a = c.post(f"{base}/infer", json=body).json()
            b = c.post(f"{base}/infer", json=body).json()
        assert a["ok"] and b["ok"]
        assert a["tokens"] == b["tokens"]      # same seed -> same output
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_openai_e2e.py::test_infer_sampling_seeded_is_reproducible -m slow -v` → fails (sampling not implemented; the output may not be reproducible or the parameters are ignored).

- [ ] **Step 3: refactor in `eujeno/net/coordinator.py`**

Add at the top of the module: `import random` and `from eujeno.net.sampling import sample_token`.

Replace `_run_generation` with a version that accepts the decoding parameters and samples via the head node:
```python
    async def _run_generation(chain, prompt, max_new, sampling, job_id):
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
            rh, _ = await _call(head_c, {"op": "head", "job_id": job_id, "topk": topk},
                                encode_tensors({"hidden_states": h}))
            if do_sample:
                tok = sample_token(rh["topk_ids"], rh["topk_logits"], tokens,
                                   temperature, top_p, rep, generator)
            else:
                tok = rh["token_id"]
            tokens.append(tok)
            cur = torch.tensor([[tok]])
            cache_position = torch.tensor([seq_len + step])
        for cid in {embed_c, head_c, *(c for _, c in decoders)}:
            try:
                await _call(cid, {"op": "end", "job_id": job_id})
            except _NodeFailure:
                pass
        return tokens, seq_len
```

Extract the failover loop into a reusable function and have /infer use both (add it above the /infer endpoint):
```python
    async def _generate_with_failover(prompt, max_new, sampling):
        excluded = set()
        last_failed = None
        for attempt in range(MAX_FAILOVERS + 1):
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is None:
                return None, {"error": "model not operational: incomplete coverage", "excluded": sorted(excluded)}
            try:
                tokens, prompt_len = await _run_generation(chain, prompt, max_new, sampling, _next_id("job"))
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt}, None
            except _NodeFailure as e:
                excluded.add(e.conn_id)
                last_failed = e.conn_id
        return None, {"error": f"too many failovers (last failed node: {last_failed})"}
```

Replace the `/infer` endpoint with:
```python
    @app.post("/infer")
    async def infer(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 8))
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        result, err = await _generate_with_failover(prompt, max_new, sampling)
        if err is not None:
            return {"ok": False, **err}
        return {"ok": True, "model": model_id, "prompt": prompt,
                "text": tokenizer.decode(result["tokens"]), "tokens": result["tokens"],
                "failovers": result["failovers"]}
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_openai_e2e.py::test_infer_sampling_seeded_is_reproducible -m slow -v` (reproducible sampling). Then NO REGRESSION on greedy/failover: `... pytest tests/test_coordinator_e2e.py tests/test_cli_coordinator.py tests/test_failover_e2e.py -m slow -v` → PASS (greedy stays the default: temperature not passed → token_id).

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/coordinator.py tests/test_openai_e2e.py && git commit -m "feat(net): coordinator with parametric sampling + refactored failover"
```

---

## Task 3: endpoint OpenAI `/v1/models` + `/v1/chat/completions`

**Files:** modify `eujeno/net/coordinator.py`; modify `tests/test_openai_e2e.py`.

- [ ] **Step 1: add to `tests/test_openai_e2e.py`**
```python
@pytest.mark.slow
def test_openai_chat_completions(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            models = c.get(f"{base}/v1/models").json()
            assert models["object"] == "list" and len(models["data"]) >= 1
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "eujeno",
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "max_tokens": 8,
            })
            body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)
        assert body["usage"]["completion_tokens"] >= 1
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_openai_e2e.py::test_openai_chat_completions -m slow -v` → 404 (endpoint does not exist).

- [ ] **Step 3: add the endpoints in `eujeno/net/coordinator.py`**

Add `import time` at the top. Add before `return app`:
```python
    @app.get("/v1/models")
    async def list_models():
        return {"object": "list",
                "data": [{"id": "eujeno", "object": "model", "owned_by": "eujeno"},
                         {"id": model_id, "object": "model", "owned_by": "eujeno"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        max_new = int(body.get("max_tokens", 256))
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(m.get("content", "") for m in messages)
        result, err = await _generate_with_failover(prompt, max_new, sampling)
        if err is not None:
            return JSONResponse({"error": {"message": err["error"], "type": "not_operational"}}, status_code=503)
        text = tokenizer.decode(result["tokens"])
        return {
            "id": "chatcmpl-" + _next_id("oa"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": result["prompt_len"],
                      "completion_tokens": len(result["tokens"]),
                      "total_tokens": result["prompt_len"] + len(result["tokens"])},
        }
```
(Is `JSONResponse` already imported in coordinator.py? If not, add `from fastapi.responses import JSONResponse`.)

- [ ] **Step 4: run PASS** — `... pytest tests/test_openai_e2e.py -m slow -v` → all OpenAI tests pass. No regression: `... pytest -q -p no:warnings` → all green.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/coordinator.py tests/test_openai_e2e.py && git commit -m "feat(net): endpoint OpenAI /v1/models + /v1/chat/completions (chat template + sampling)"
```

---

## Task 4: docs "connecting agents" + ROADMAP

**Files:** create `docs/examples/agents.md`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: create `docs/examples/agents.md`** with:
```markdown
# Connecting AI agents to Eujeno (OpenAI-compatible API)

When the model is OPERATIONAL, the coordinator exposes an OpenAI-compatible API: point any OpenAI client/SDK to `http://THE_COORDINATOR:9000/v1`.

## OpenAI SDK (Python)
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="anything")
r = client.chat.completions.create(
    model="eujeno",
    messages=[{"role": "user", "content": "Write a haiku about the sea"}],
    temperature=0.8, top_p=0.9, max_tokens=80,
)
print(r.choices[0].message.content)
```

## curl
```bash
curl -s http://127.0.0.1:9000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "eujeno",
  "messages": [{"role":"user","content":"Hi!"}],
  "temperature": 0.7, "max_tokens": 64
}'
```

## Claude Code and Anthropic clients
Claude Code speaks the Anthropic API, not OpenAI. Put **LiteLLM** in front of it as a gateway (it translates Anthropic↔OpenAI) pointing it to `http://THE_COORDINATOR:9000/v1`, then:
```bash
ANTHROPIC_BASE_URL=http://LITELLM:4000 claude
```
(SSE streaming and a native Anthropic `/v1/messages` endpoint are the next steps.)

## Many agents
Every request is a job on the network. For many concurrent agents it is worth adding a queue + block replicas (Part 3) and, for quality, splitting a larger model across more nodes.
```

- [ ] **Step 2:** in `README.md`, in the Quickstart section, add a line linking `docs/examples/agents.md` ("Connecting AI agents via the OpenAI API").

- [ ] **Step 3:** in `docs/ROADMAP.md`, add under Phase 1 an `[x]` entry "OpenAI-compatible API (`/v1/chat/completions`, chat template + sampling)" with a link to this plan; note that SSE streaming and Anthropic/LiteLLM are still to do. Update "Last updated".

- [ ] **Step 4: full suite** — `... pytest -q -p no:warnings` → all green.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add docs/examples/agents.md README.md docs/ROADMAP.md && git commit -m "docs: connecting agents via the OpenAI API; ROADMAP /v1"
```

---

## Self-Review

**Coverage:** sampling (Task 1) ✓; chat template (Task 3, apply_chat_template) ✓; /v1/models + /v1/chat/completions OpenAI format with usage (Task 3) ✓; greedy/failover unchanged by default (Task 2, temperature not passed → token_id) ✓; docs for OpenAI SDK/curl/Claude Code (Task 4) ✓. SSE streaming + Anthropic/LiteLLM explicitly follow-up.

**Placeholder scan:** no TODO; code complete.

**Type consistency:** `sample_token(topk_ids, topk_logits, generated_ids, temperature, top_p, repetition_penalty, generator)->int`; head returns `token_id`+`topk_ids`+`topk_logits`; `_run_generation(chain, prompt, max_new, sampling, job_id)->(tokens, prompt_len)`; `_generate_with_failover(prompt, max_new, sampling)->(result|None, err|None)`; used consistently in /infer and /v1/chat/completions.
```
