# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import random
import time

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from eujeno.net.framing import pack, unpack
from eujeno.net.wire import encode_tensors, decode_tensors
from eujeno.net.discovery import build_chain
from eujeno.net.sampling import sample_token
from eujeno.net.tools import extract_tool_calls
from eujeno.net.jobstore import JobStore

log = logging.getLogger("eujeno.coordinator")

MAX_FAILOVERS = 5
COVERAGE_POLL_INTERVAL = 0.5


class _NodeFailure(Exception):
    def __init__(self, conn_id):
        super().__init__(conn_id)
        self.conn_id = conn_id


def create_coordinator_app(model_id: str, num_layers: int, tokenizer, db_path=None, coverage_timeout=120.0):
    """Coordinator-relay: nodes connect via WS and announce their stages; POST /infer
    drives generation by relaying each hop to the right node. Jobs are persisted to a
    durable SQLite job log (db_path=None -> in-memory, used by tests)."""
    app = FastAPI()
    store = JobStore(db_path if db_path is not None else ":memory:")
    try:
        store.recover()
    except Exception as _e:
        log.warning("jobstore recover failed (continuing): %s", _e)

    def _store_safe(fn, *args):
        try:
            fn(*args)
        except Exception as e:                      # durability is best-effort; never break inference
            log.warning("jobstore write failed: %s", e)

    conns = {}        # conn_id -> {"ws", "stages", "pending": {req_id: Future}}
    counter = {"n": 0}

    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        _tid = tokenizer.convert_tokens_to_ids(_tok)
        if isinstance(_tid, int) and _tid >= 0 and _tid != tokenizer.unk_token_id:
            stop_ids.add(int(_tid))

    def _next_id(prefix):
        counter["n"] += 1
        return f"{prefix}{counter['n']}"

    async def _call(conn_id, header, payload=b""):
        if conn_id not in conns:
            raise _NodeFailure(conn_id)
        c = conns[conn_id]
        req_id = _next_id("r")
        fut = asyncio.get_running_loop().create_future()
        c["pending"][req_id] = fut
        try:
            await c["ws"].send_bytes(pack({**header, "req_id": req_id}, payload))
            return await fut
        except Exception:
            raise _NodeFailure(conn_id)

    @app.websocket("/node")
    async def node_ws(ws: WebSocket):
        await ws.accept()
        announce, _ = unpack(await ws.receive_bytes())
        conn_id = _next_id("c")
        conns[conn_id] = {"ws": ws, "stages": announce["stages"], "pending": {}}
        try:
            while True:
                rh, rp = unpack(await ws.receive_bytes())
                fut = conns[conn_id]["pending"].pop(rh.get("req_id"), None)
                if fut is not None and not fut.done():
                    fut.set_result((rh, rp))
        except WebSocketDisconnect:
            pass
        finally:
            c = conns.pop(conn_id, None)
            if c is not None:   # fail the pending requests instead of leaving them hanging
                for fut in c["pending"].values():
                    if not fut.done():
                        fut.set_exception(ConnectionError(f"node {conn_id} disconnected"))

    @app.get("/registry")
    async def registry():
        return {"num_layers": num_layers,
                "nodes": [{"conn": cid, "stages": c["stages"]} for cid, c in conns.items()]}

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

    async def _generate_with_failover(prompt, max_new, sampling, job_id):
        excluded = set()
        last_failed = None
        resume_tokens = []
        for attempt in range(MAX_FAILOVERS + 1):
            chain = await _await_coverage(excluded, job_id)
            if chain is None:
                return None, {"error": "coverage timeout: model not operational", "excluded": sorted(excluded)}
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

    @app.post("/infer")
    async def infer(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 8))
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        job_id = _next_id("job")
        prompt_len = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        _store_safe(store.create_job, job_id, model_id, prompt, sampling, prompt_len)
        result, err = await _generate_with_failover(prompt, max_new, sampling, job_id)
        if err is not None:
            _store_safe(store.fail, job_id, err["error"])
            return {"ok": False, **err}
        text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
        _store_safe(store.finish, job_id, text, result["finish_reason"])
        return {"ok": True, "model": model_id, "prompt": prompt,
                "text": text, "tokens": result["tokens"], "failovers": result["failovers"]}

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
        tools = body.get("tools")
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        try:
            prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                   add_generation_prompt=True)
        except Exception:
            prompt = "\n".join((m.get("content") or "") for m in messages)
        job_id = _next_id("job")
        prompt_len = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        _store_safe(store.create_job, job_id, model_id, prompt, sampling, prompt_len)
        result, err = await _generate_with_failover(prompt, max_new, sampling, job_id)
        if err is not None:
            _store_safe(store.fail, job_id, err["error"])
            return JSONResponse({"error": {"message": err["error"], "type": "not_operational"}}, status_code=503)
        text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
        _store_safe(store.finish, job_id, text, result["finish_reason"])
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

    @app.get("/jobs")
    async def list_jobs(limit: int = 50):
        return {"jobs": store.recent_jobs(limit)}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        j = store.get_job(job_id)
        if j is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return j

    return app
