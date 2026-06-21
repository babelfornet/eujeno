# EOS-stop + Tool/Function calling (for MCP agents) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Make Eujeno's OpenAI API usable by agents/MCP hosts: (1) generation **stops at EOS/end-of-turn** and the output is clean (no special tokens, no post-EOS text); (2) `/v1/chat/completions` accepts `tools` and returns `tool_calls` (function calling), so an agent can have the model call MCP tools.

**Architecture:** The MCP tools are executed by the agent/host; Eujeno only needs to be a backend with tool-calling. The coordinator computes the `stop_ids` from the tokenizer and stops generation; it applies the chat template with `tools` (Qwen2.5 has the native tool-use template); it parses the `<tool_call>{...}</tool_call>` output into OpenAI-format `tool_calls`. Decodes with `skip_special_tokens=True`.

**Tech Stack:** Python · the existing `eujeno/net/coordinator.py` · transformers (chat template with tools) · regex/json.

**Reality note:** the 0.5B model does not emit reliable tool-calls — the *parser* tests are deterministic (logic), while the e2e with `tools` only verifies that the endpoint accepts the tools and responds well-formed. For real tool-calling a 7B+ model is needed (identical infra).

**Out of scope:** SSE streaming; Anthropic `/v1/messages` endpoint.

---

## File Structure
```
eujeno/net/tools.py            # NEW: extract_tool_calls() (pure)
eujeno/net/coordinator.py      # MOD: stop_ids + finish_reason + skip_special_tokens + tools in chat_completions
tests/test_tools.py             # NEW: parser (fast)
tests/test_openai_e2e.py        # MOD: clean stop + smoke tools (slow)
docs/examples/agents.md         # MOD: tool/MCP section
docs/ROADMAP.md
```

---

## Task 1: stop at EOS + clean decode

**Files:** modify `eujeno/net/coordinator.py`; modify `tests/test_openai_e2e.py`.

- [ ] **Step 1: test (append to `tests/test_openai_e2e.py`)**
```python
@pytest.mark.slow
def test_chat_output_has_no_special_tokens(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "eujeno",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 64,
            }).json()
        content = r["choices"][0]["message"]["content"]
        assert "<|im_end|>" not in content and "<|endoftext|>" not in content
        assert r["choices"][0]["finish_reason"] in ("stop", "length")
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_openai_e2e.py::test_chat_output_has_no_special_tokens -m slow -v`. (May fail because the output contains special tokens / there is no correct finish_reason.)

- [ ] **Step 3: modify `eujeno/net/coordinator.py`**

(a) Right after receiving `tokenizer` in `create_coordinator_app`, compute the stop ids:
```python
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))
```

(b) In `_run_generation`, stop when a stop token comes out and also return the `finish_reason`. Replace the body of the generation loop and the return with:
```python
        finish_reason = "length"
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
            if tok in stop_ids:                 # end-of-turn: stop without emitting the special token
                finish_reason = "stop"
                break
            tokens.append(tok)
            cur = torch.tensor([[tok]])
            cache_position = torch.tensor([seq_len + step])
        for cid in {embed_c, head_c, *(c for _, c in decoders)}:
            try:
                await _call(cid, {"op": "end", "job_id": job_id})
            except _NodeFailure:
                pass
        return tokens, seq_len, finish_reason
```
(c) `_generate_with_failover` must propagate `finish_reason`: where it does `tokens, prompt_len = await _run_generation(...)`, change to `tokens, prompt_len, finish_reason = await _run_generation(...)` and include `"finish_reason": finish_reason` in the result dict.

(d) In `/infer` and `/v1/chat/completions` decode with `skip_special_tokens=True`:
- `/infer`: `"text": tokenizer.decode(result["tokens"], skip_special_tokens=True)`.
- `/v1/chat/completions`: `text = tokenizer.decode(result["tokens"], skip_special_tokens=True)` and use `result["finish_reason"]` in the choice's `finish_reason` field (instead of a fixed "stop").

- [ ] **Step 4: run PASS** — the new test passes; no regression: `... pytest tests/test_openai_e2e.py tests/test_coordinator_e2e.py tests/test_failover_e2e.py -m slow -v`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/coordinator.py tests/test_openai_e2e.py && git commit -m "fix(net): stop at end-of-turn (EOS) + decode skip_special_tokens + finish_reason"
```

---

## Task 2: parser `extract_tool_calls`

**Files:** create `eujeno/net/tools.py`, `tests/test_tools.py`.

- [ ] **Step 1: test `tests/test_tools.py`**
```python
import json
from eujeno.net.tools import extract_tool_calls


def test_single_tool_call():
    text = 'Checking the weather.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>'
    content, calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Rome"}
    assert "<tool_call>" not in content


def test_no_tool_calls():
    content, calls = extract_tool_calls("Hi, how are you?")
    assert calls == []
    assert content == "Hi, how are you?"


def test_multiple_tool_calls():
    text = '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    content, calls = extract_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"x": 1}


def test_malformed_tool_call_ignored():
    content, calls = extract_tool_calls("<tool_call>non-json</tool_call> rest")
    assert calls == []
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_tools.py -v` → ImportError.

- [ ] **Step 3: implement `eujeno/net/tools.py`**
```python
import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str):
    """Extracts the tool calls from the Qwen2.5 format (<tool_call>{json}</tool_call>) and
    converts them to the OpenAI format. Returns (content_without_toolcall, [tool_call...])."""
    calls = []
    for i, block in enumerate(_TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": json.dumps(obj.get("arguments", {})),
            },
        })
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, calls
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_tools.py -v` → 4 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/tools.py tests/test_tools.py && git commit -m "feat(net): extract_tool_calls (Qwen2.5 tool-call parsing -> OpenAI format)"
```

---

## Task 3: `tools` in `/v1/chat/completions`

**Files:** modify `eujeno/net/coordinator.py`; modify `tests/test_openai_e2e.py`.

- [ ] **Step 1: test (append to `tests/test_openai_e2e.py`)**
```python
@pytest.mark.slow
def test_chat_completions_accepts_tools(full_model):
    srv, base = _two_node_coordinator(full_model)
    tools = [{
        "type": "function",
        "function": {"name": "get_weather", "description": "Weather for a city",
                     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}},
    }]
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "eujeno",
                "messages": [{"role": "user", "content": "What's the weather in Rome?"}],
                "tools": tools, "max_tokens": 64,
            }).json()
        choice = r["choices"][0]
        # well-formed endpoint: either text content or tool_calls; consistent finish_reason
        assert "message" in choice
        if choice["message"].get("tool_calls"):
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["tool_calls"][0]["type"] == "function"
        else:
            assert isinstance(choice["message"].get("content"), str)
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_openai_e2e.py::test_chat_completions_accepts_tools -m slow -v` (the `tools` parameter is not handled; it could break in the template or ignore it).

- [ ] **Step 3: modify `chat_completions` in `eujeno/net/coordinator.py`**

Add `from eujeno.net.tools import extract_tool_calls` at the top. Replace the body of `chat_completions` with:
```python
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        max_new = int(body.get("max_tokens", 256))
        tools = body.get("tools")
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        try:
            prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                   add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(m.get("content", "") or "" for m in messages)
        result, err = await _generate_with_failover(prompt, max_new, sampling)
        if err is not None:
            return JSONResponse({"error": {"message": err["error"], "type": "not_operational"}}, status_code=503)
        text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
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

- [ ] **Step 4: run PASS** — `... pytest tests/test_openai_e2e.py -m slow -v` → all pass. Full suite: `... pytest -q -p no:warnings`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/coordinator.py tests/test_openai_e2e.py && git commit -m "feat(net): /v1/chat/completions accepts tools and returns tool_calls (function calling)"
```

---

## Task 4: docs tool/MCP + ROADMAP

**Files:** modify `docs/examples/agents.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: add a "Tool calling / MCP" section to `docs/examples/agents.md`**
```markdown
## Tool calling (and MCP tools)

`/v1/chat/completions` accepts the `tools` parameter (OpenAI format) and, if the model decides to call a tool, returns `tool_calls` with `finish_reason: "tool_calls"`. The **MCP tools are executed by the agent/host** (Claude Code, etc.): the model decides *which* tool to call, the agent executes it and sends the result back as a `role: "tool"` message.

```python
tools = [{"type":"function","function":{
  "name":"get_weather","description":"Weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r = client.chat.completions.create(model="eujeno",
      messages=[{"role":"user","content":"What's the weather in Rome?"}], tools=tools)
# r.choices[0].message.tool_calls -> [{function:{name:"get_weather", arguments:'{"city":"Rome"}'}}]
```

Note: reliable tool-calling requires a capable model (7B+). With Qwen 0.5B it only serves to verify the mechanism.
```

- [ ] **Step 2: update `docs/ROADMAP.md`** — under the OpenAI API add `[x]` "stop at EOS + tool/function calling (`tools`/`tool_calls`) — basis for MCP agents"; update "Last updated".

- [ ] **Step 3: suite** — `... pytest -q -p no:warnings` → green.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add docs/examples/agents.md docs/ROADMAP.md && git commit -m "docs: tool calling / MCP in the OpenAI API; ROADMAP"
```

---

## Self-Review

**Coverage:** EOS-stop + clean output (Task 1, stop_ids + skip_special_tokens + finish_reason) ✓; tool-call parsing (Task 2, pure, tested) ✓; `tools`→`tool_calls` in the OpenAI endpoint (Task 3) ✓; MCP/tool docs (Task 4) ✓. Streaming/Anthropic out of scope.

**Placeholder scan:** no TODO; complete code.

**Type consistency:** `_run_generation` now returns `(tokens, seq_len, finish_reason)`; the `_generate_with_failover` result includes `finish_reason`; `/infer` and `/v1/chat/completions` updated accordingly; `extract_tool_calls(text) -> (content, [tool_call])`. Greedy default unchanged (the coordinator/failover tests do not pass sampling).
```
