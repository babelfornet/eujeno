# API OpenAI-compatibile sul coordinator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Esporre sul coordinator un endpoint **OpenAI-compatibile** (`/v1/chat/completions` + `/v1/models`) così che qualsiasi client/agente OpenAI-compatibile possa interrogare Axyn. Include **chat template** (messages → prompt) e **sampling** (temperature/top_p/repetition_penalty), che risolvono anche i problemi di qualità del greedy.

**Architecture:** Il nodo `head` espone i logits **top-k** (oltre all'argmax, per retro-compatibilità greedy). Il coordinator campiona (helper puro `sample_token`) e guida la generazione con i parametri di decoding; applica il chat template del tokenizer ai `messages`; riusa il loop con failover esistente. Gli endpoint `/v1/*` mappano richiesta/risposta nel formato OpenAI.

**Tech Stack:** Python · FastAPI · torch · transformers (chat template) · l'esistente `axyn/net/{coordinator,node_exec,framing,wire,discovery}.py`.

**Fuori scope (follow-up):** streaming SSE; endpoint Anthropic `/v1/messages` / config LiteLLM per Claude Code; queue & load balancing per molti agenti concorrenti.

---

## File Structure

```
axyn/net/sampling.py         # NUOVO: sample_token() (puro, testabile)
axyn/net/node_exec.py        # MOD: head ritorna anche topk_ids/topk_logits
axyn/net/coordinator.py      # MOD: refactor generazione + sampling + chat template + /v1 endpoints
tests/
  test_sampling.py              # sample_token deterministico (veloce)
  test_openai_e2e.py            # /v1/chat/completions su 2 nodi (slow)
docs/examples/agents.md         # NUOVO: collegare client OpenAI / Claude Code
docs/ROADMAP.md
```

---

## Task 1: `sample_token` + head top-k

**Files:** create `axyn/net/sampling.py`; modify `axyn/net/node_exec.py`; create `tests/test_sampling.py`; (head già coperto da test_node_exec — resta verde).

- [ ] **Step 1: test `tests/test_sampling.py`**
```python
import torch
from axyn.net.sampling import sample_token


def test_greedy_returns_argmax_when_temperature_zero():
    ids = [10, 20, 30]
    logits = [1.0, 5.0, 2.0]
    out = sample_token(ids, logits, generated_ids=[], temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0, generator=None)
    assert out == 20   # logit massimo


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
    # con forte penalty sul token 10 (gia' generato), 20 deve diventare l'argmax greedy
    out = sample_token(ids, logits, generated_ids=[10], temperature=0.0,
                       top_p=1.0, repetition_penalty=10.0, generator=None)
    assert out == 20
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_sampling.py -v` → ImportError.

- [ ] **Step 3: implementa `axyn/net/sampling.py`**
```python
import torch


def sample_token(topk_ids, topk_logits, generated_ids, temperature, top_p,
                 repetition_penalty, generator) -> int:
    """Sceglie il prossimo token dai candidati top-k del nodo head.
    temperature<=0 -> greedy (argmax). Altrimenti: repetition penalty, temperature,
    nucleo top_p, campionamento multinomiale (deterministico se `generator` ha un seed)."""
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

- [ ] **Step 4: modifica il ramo `head` di `handle_request` in `axyn/net/node_exec.py`**

Sostituisci il blocco `if op == "head":` con (ritorna anche i top-k, mantenendo `token_id` per retro-compatibilità greedy):
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
Aggiungi `import torch` in cima a `node_exec.py` se non già presente.

- [ ] **Step 5: run PASS** — `... pytest tests/test_sampling.py tests/test_node_exec.py -m "slow or not slow" -v`. Expected: sampling 3 passed; test_node_exec ancora verde (token_id invariato).

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/sampling.py axyn/net/node_exec.py tests/test_sampling.py && git commit -m "feat(net): sample_token + head espone top-k logits (sampling)"
```

---

## Task 2: coordinator — generazione parametrica + sampling in /infer

**Files:** modify `axyn/net/coordinator.py`; create test in `tests/test_openai_e2e.py` (parte sampling).

- [ ] **Step 1: aggiungi a `tests/test_openai_e2e.py` un test di sampling deterministico**
```python
import socket, threading, time, asyncio
import pytest, httpx, uvicorn
from axyn.net.coordinator import create_coordinator_app
from axyn.net.node import run_node
from axyn.net.node_exec import NodeState
from axyn.net.topology import StageSpec


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
            body = {"prompt": "Ciao", "max_new_tokens": 6, "temperature": 0.8, "top_p": 0.9, "seed": 42}
            a = c.post(f"{base}/infer", json=body).json()
            b = c.post(f"{base}/infer", json=body).json()
        assert a["ok"] and b["ok"]
        assert a["tokens"] == b["tokens"]      # stesso seed -> stesso output
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_openai_e2e.py::test_infer_sampling_seeded_is_reproducible -m slow -v` → fallisce (sampling non implementato; output potrebbe non essere riproducibile o i parametri ignorati).

- [ ] **Step 3: refactor in `axyn/net/coordinator.py`**

Aggiungi in cima al modulo: `import random` e `from axyn.net.sampling import sample_token`.

Sostituisci `_run_generation` con una versione che accetta i parametri di decoding e campiona via il nodo head:
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

Estrai il loop di failover in una funzione riusabile e fai usare entrambe a /infer (aggiungi sopra l'endpoint /infer):
```python
    async def _generate_with_failover(prompt, max_new, sampling):
        excluded = set()
        last_failed = None
        for attempt in range(MAX_FAILOVERS + 1):
            stages = {cid: c["stages"] for cid, c in conns.items() if cid not in excluded}
            chain = build_chain(stages, num_layers)
            if chain is None:
                return None, {"error": "modello non operativo: coverage incompleta", "excluded": sorted(excluded)}
            try:
                tokens, prompt_len = await _run_generation(chain, prompt, max_new, sampling, _next_id("job"))
                return {"tokens": tokens, "prompt_len": prompt_len, "failovers": attempt}, None
            except _NodeFailure as e:
                excluded.add(e.conn_id)
                last_failed = e.conn_id
        return None, {"error": f"troppi failover (ultimo nodo fallito: {last_failed})"}
```

Sostituisci l'endpoint `/infer` con:
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

- [ ] **Step 4: run PASS** — `... pytest tests/test_openai_e2e.py::test_infer_sampling_seeded_is_reproducible -m slow -v` (sampling riproducibile). Poi NESSUNA REGRESSIONE sul greedy/failover: `... pytest tests/test_coordinator_e2e.py tests/test_cli_coordinator.py tests/test_failover_e2e.py -m slow -v` → PASS (il greedy resta default: temperature non passata → token_id).

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/coordinator.py tests/test_openai_e2e.py && git commit -m "feat(net): coordinator con sampling parametrico + failover refactored"
```

---

## Task 3: endpoint OpenAI `/v1/models` + `/v1/chat/completions`

**Files:** modify `axyn/net/coordinator.py`; modify `tests/test_openai_e2e.py`.

- [ ] **Step 1: aggiungi a `tests/test_openai_e2e.py`**
```python
@pytest.mark.slow
def test_openai_chat_completions(full_model):
    srv, base = _two_node_coordinator(full_model)
    try:
        with httpx.Client(timeout=60.0) as c:
            models = c.get(f"{base}/v1/models").json()
            assert models["object"] == "list" and len(models["data"]) >= 1
            r = c.post(f"{base}/v1/chat/completions", json={
                "model": "axyn",
                "messages": [{"role": "user", "content": "Di' ciao in una parola"}],
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

- [ ] **Step 2: run FAIL** — `... pytest tests/test_openai_e2e.py::test_openai_chat_completions -m slow -v` → 404 (endpoint inesistente).

- [ ] **Step 3: aggiungi gli endpoint in `axyn/net/coordinator.py`**

Aggiungi `import time` in cima. Aggiungi prima di `return app`:
```python
    @app.get("/v1/models")
    async def list_models():
        return {"object": "list",
                "data": [{"id": "axyn", "object": "model", "owned_by": "axyn"},
                         {"id": model_id, "object": "model", "owned_by": "axyn"}]}

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
(`JSONResponse` è già importato in coordinator.py? Se non lo è, aggiungi `from fastapi.responses import JSONResponse`.)

- [ ] **Step 4: run PASS** — `... pytest tests/test_openai_e2e.py -m slow -v` → tutti i test OpenAI passano. Nessuna regressione: `... pytest -q -p no:warnings` → tutto verde.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/net/coordinator.py tests/test_openai_e2e.py && git commit -m "feat(net): endpoint OpenAI /v1/models + /v1/chat/completions (chat template + sampling)"
```

---

## Task 4: docs "collegare agenti" + ROADMAP

**Files:** create `docs/examples/agents.md`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: crea `docs/examples/agents.md`** con:
```markdown
# Collegare agenti AI a Axyn (API OpenAI-compatibile)

Quando il modello è OPERATIVO, il coordinator espone un'API OpenAI-compatibile: punta qualsiasi client/SDK OpenAI a `http://IL_COORDINATOR:9000/v1`.

## SDK OpenAI (Python)
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="qualsiasi")
r = client.chat.completions.create(
    model="axyn",
    messages=[{"role": "user", "content": "Scrivi un haiku sul mare"}],
    temperature=0.8, top_p=0.9, max_tokens=80,
)
print(r.choices[0].message.content)
```

## curl
```bash
curl -s http://127.0.0.1:9000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "axyn",
  "messages": [{"role":"user","content":"Ciao!"}],
  "temperature": 0.7, "max_tokens": 64
}'
```

## Claude Code e client Anthropic
Claude Code parla l'API Anthropic, non OpenAI. Mettici davanti **LiteLLM** come gateway (traduce Anthropic↔OpenAI) puntandolo a `http://IL_COORDINATOR:9000/v1`, poi:
```bash
ANTHROPIC_BASE_URL=http://LITELLM:4000 claude
```
(Lo streaming SSE e un endpoint Anthropic nativo `/v1/messages` sono i prossimi passi.)

## Tanti agenti
Ogni richiesta è un job sulla rete. Per molti agenti concorrenti conviene aggiungere coda + repliche dei blocchi (Parte 3) e, per qualità, splittare un modello più grande su più nodi.
```

- [ ] **Step 2:** in `README.md`, nella sezione Quickstart, aggiungi una riga che linka `docs/examples/agents.md` ("Collegare agenti AI via API OpenAI").

- [ ] **Step 3:** in `docs/ROADMAP.md`, aggiungi sotto Fase 1 una voce `[x]` "API OpenAI-compatibile (`/v1/chat/completions`, chat template + sampling)" con link a questo piano; nota che streaming SSE e Anthropic/LiteLLM restano da fare. Aggiorna "Ultimo aggiornamento".

- [ ] **Step 4: suite completa** — `... pytest -q -p no:warnings` → tutto verde.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add docs/examples/agents.md README.md docs/ROADMAP.md && git commit -m "docs: collegare agenti via API OpenAI; ROADMAP /v1"
```

---

## Self-Review

**Coverage:** sampling (Task 1) ✓; chat template (Task 3, apply_chat_template) ✓; /v1/models + /v1/chat/completions formato OpenAI con usage (Task 3) ✓; greedy/failover invariati di default (Task 2, temperature non passata → token_id) ✓; docs per OpenAI SDK/curl/Claude Code (Task 4) ✓. Streaming SSE + Anthropic/LiteLLM esplicitamente follow-up.

**Placeholder scan:** nessun TODO; codice completo.

**Type consistency:** `sample_token(topk_ids, topk_logits, generated_ids, temperature, top_p, repetition_penalty, generator)->int`; head ritorna `token_id`+`topk_ids`+`topk_logits`; `_run_generation(chain, prompt, max_new, sampling, job_id)->(tokens, prompt_len)`; `_generate_with_failover(prompt, max_new, sampling)->(result|None, err|None)`; usati coerentemente in /infer e /v1/chat/completions.
```
