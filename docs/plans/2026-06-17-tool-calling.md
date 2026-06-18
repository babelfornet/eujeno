# EOS-stop + Tool/Function calling (per agenti MCP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Rendere l'API OpenAI di Axyn usabile da agenti/MCP host: (1) la generazione si **ferma all'EOS/fine-turno** e l'output è pulito (niente token speciali né testo post-EOS); (2) `/v1/chat/completions` accetta `tools` e ritorna `tool_calls` (function calling), così un agente può far chiamare i tool MCP al modello.

**Architecture:** I tool MCP li esegue l'agente/host; Axyn deve solo essere un backend con tool-calling. Il coordinator calcola gli `stop_ids` dal tokenizer e interrompe la generazione; applica il chat template con `tools` (Qwen2.5 ha il template tool-use nativo); fa il parsing dell'output `<tool_call>{...}</tool_call>` in `tool_calls` formato OpenAI. Decodifica con `skip_special_tokens=True`.

**Tech Stack:** Python · l'esistente `axyn/net/coordinator.py` · transformers (chat template con tools) · regex/json.

**Nota di realtà:** il modello da 0.5B non emette tool-call affidabili — i test del *parser* sono deterministici (logica), mentre l'e2e con `tools` verifica solo che l'endpoint accetti i tool e risponda well-formed. Per tool-calling reale serve un modello 7B+ (infra identica).

**Fuori scope:** streaming SSE; endpoint Anthropic `/v1/messages`.

---

## File Structure
```
axyn/net/tools.py            # NUOVO: extract_tool_calls() (puro)
axyn/net/coordinator.py      # MOD: stop_ids + finish_reason + skip_special_tokens + tools nel chat_completions
tests/test_tools.py             # NUOVO: parser (veloce)
tests/test_openai_e2e.py        # MOD: stop pulito + smoke tools (slow)
docs/examples/agents.md         # MOD: sezione tool/MCP
docs/ROADMAP.md
```

---

## Task 1: stop all'EOS + decode pulito

**Files:** modify `axyn/net/coordinator.py`; modify `tests/test_openai_e2e.py`.

- [ ] **Step 1: test (append a `tests/test_openai_e2e.py`)**
```python
@pytest.mark.slow
def test_chat_output_has_no_special_tokens(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "axyn",
                "messages": [{"role": "user", "content": "Di' ciao."}],
                "max_tokens": 64,
            }).json()
        content = r["choices"][0]["message"]["content"]
        assert "<|im_end|>" not in content and "<|endoftext|>" not in content
        assert r["choices"][0]["finish_reason"] in ("stop", "length")
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_openai_e2e.py::test_chat_output_has_no_special_tokens -m slow -v`. (Può fallire perché l'output contiene token speciali / non c'è finish_reason corretto.)

- [ ] **Step 3: modifica `axyn/net/coordinator.py`**

(a) Subito dopo aver ricevuto `tokenizer` in `create_coordinator_app`, calcola gli stop ids:
```python
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))
```

(b) In `_run_generation`, interrompi quando esce un token di stop e ritorna anche il `finish_reason`. Sostituisci il corpo del loop di generazione e il return con:
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
            if tok in stop_ids:                 # fine-turno: stop senza emettere il token speciale
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
(c) `_generate_with_failover` deve propagare `finish_reason`: dove fa `tokens, prompt_len = await _run_generation(...)`, cambia in `tokens, prompt_len, finish_reason = await _run_generation(...)` e includi `"finish_reason": finish_reason` nel dict result.

(d) In `/infer` e `/v1/chat/completions` decodifica con `skip_special_tokens=True`:
- `/infer`: `"text": tokenizer.decode(result["tokens"], skip_special_tokens=True)`.
- `/v1/chat/completions`: `text = tokenizer.decode(result["tokens"], skip_special_tokens=True)` e usa `result["finish_reason"]` nel campo `finish_reason` della choice (al posto di "stop" fisso).

- [ ] **Step 4: run PASS** — il nuovo test passa; nessuna regressione: `... pytest tests/test_openai_e2e.py tests/test_coordinator_e2e.py tests/test_failover_e2e.py -m slow -v`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/coordinator.py tests/test_openai_e2e.py && git commit -m "fix(net): stop alla fine-turno (EOS) + decode skip_special_tokens + finish_reason"
```

---

## Task 2: parser `extract_tool_calls`

**Files:** create `axyn/net/tools.py`, `tests/test_tools.py`.

- [ ] **Step 1: test `tests/test_tools.py`**
```python
import json
from axyn.net.tools import extract_tool_calls


def test_single_tool_call():
    text = 'Controllo il meteo.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Roma"}}\n</tool_call>'
    content, calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Roma"}
    assert "<tool_call>" not in content


def test_no_tool_calls():
    content, calls = extract_tool_calls("Ciao, come va?")
    assert calls == []
    assert content == "Ciao, come va?"


def test_multiple_tool_calls():
    text = '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    content, calls = extract_tool_calls(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"x": 1}


def test_malformed_tool_call_ignored():
    content, calls = extract_tool_calls("<tool_call>non-json</tool_call> resto")
    assert calls == []
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_tools.py -v` → ImportError.

- [ ] **Step 3: implementa `axyn/net/tools.py`**
```python
import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str):
    """Estrae i tool call dal formato Qwen2.5 (<tool_call>{json}</tool_call>) e li
    converte nel formato OpenAI. Ritorna (content_senza_toolcall, [tool_call...])."""
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
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/tools.py tests/test_tools.py && git commit -m "feat(net): extract_tool_calls (parsing tool-call Qwen2.5 -> formato OpenAI)"
```

---

## Task 3: `tools` in `/v1/chat/completions`

**Files:** modify `axyn/net/coordinator.py`; modify `tests/test_openai_e2e.py`.

- [ ] **Step 1: test (append a `tests/test_openai_e2e.py`)**
```python
@pytest.mark.slow
def test_chat_completions_accepts_tools(full_model):
    srv, base = _two_node_coordinator(full_model)
    tools = [{
        "type": "function",
        "function": {"name": "get_weather", "description": "Meteo di una città",
                     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}},
    }]
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "axyn",
                "messages": [{"role": "user", "content": "Che tempo fa a Roma?"}],
                "tools": tools, "max_tokens": 64,
            }).json()
        choice = r["choices"][0]
        # endpoint well-formed: o content testuale o tool_calls; finish_reason coerente
        assert "message" in choice
        if choice["message"].get("tool_calls"):
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["tool_calls"][0]["type"] == "function"
        else:
            assert isinstance(choice["message"].get("content"), str)
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_openai_e2e.py::test_chat_completions_accepts_tools -m slow -v` (il parametro `tools` non è gestito; potrebbe rompersi nel template o ignorarlo).

- [ ] **Step 3: modifica `chat_completions` in `axyn/net/coordinator.py`**

Aggiungi `from axyn.net.tools import extract_tool_calls` in cima. Sostituisci il corpo di `chat_completions` con:
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

- [ ] **Step 4: run PASS** — `... pytest tests/test_openai_e2e.py -m slow -v` → tutti passano. Full suite: `... pytest -q -p no:warnings`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/coordinator.py tests/test_openai_e2e.py && git commit -m "feat(net): /v1/chat/completions accetta tools e ritorna tool_calls (function calling)"
```

---

## Task 4: docs tool/MCP + ROADMAP

**Files:** modify `docs/examples/agents.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: aggiungi a `docs/examples/agents.md` una sezione "Tool calling / MCP"**
```markdown
## Tool calling (e tool MCP)

`/v1/chat/completions` accetta il parametro `tools` (formato OpenAI) e, se il modello decide di chiamare un tool, ritorna `tool_calls` con `finish_reason: "tool_calls"`. I **tool MCP li esegue l'agente/host** (Claude Code, ecc.): il modello decide *quale* tool chiamare, l'agente lo esegue e rimanda il risultato come messaggio `role: "tool"`.

```python
tools = [{"type":"function","function":{
  "name":"get_weather","description":"Meteo di una città",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r = client.chat.completions.create(model="axyn",
      messages=[{"role":"user","content":"Che tempo fa a Roma?"}], tools=tools)
# r.choices[0].message.tool_calls -> [{function:{name:"get_weather", arguments:'{"city":"Roma"}'}}]
```

Nota: il tool-calling affidabile richiede un modello capace (7B+). Con Qwen 0.5B serve solo a verificare il meccanismo.
```

- [ ] **Step 2: aggiorna `docs/ROADMAP.md`** — sotto l'API OpenAI aggiungi `[x]` "stop all'EOS + tool/function calling (`tools`/`tool_calls`) — base per agenti MCP"; aggiorna "Ultimo aggiornamento".

- [ ] **Step 3: suite** — `... pytest -q -p no:warnings` → verde.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add docs/examples/agents.md docs/ROADMAP.md && git commit -m "docs: tool calling / MCP nell'API OpenAI; ROADMAP"
```

---

## Self-Review

**Coverage:** EOS-stop + output pulito (Task 1, stop_ids + skip_special_tokens + finish_reason) ✓; parsing tool-call (Task 2, puro testato) ✓; `tools`→`tool_calls` nell'endpoint OpenAI (Task 3) ✓; docs MCP/tool (Task 4) ✓. Streaming/Anthropic fuori scope.

**Placeholder scan:** nessun TODO; codice completo.

**Type consistency:** `_run_generation` ora ritorna `(tokens, seq_len, finish_reason)`; `_generate_with_failover` result include `finish_reason`; `/infer` e `/v1/chat/completions` aggiornati di conseguenza; `extract_tool_calls(text) -> (content, [tool_call])`. Default greedy invariato (i test coordinator/failover non passano sampling).
```
