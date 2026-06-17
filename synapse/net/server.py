import asyncio
import time
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from synapse.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.discovery import Registry, build_chain
from synapse.net.generation import generate_tokens
from synapse.net.tools import extract_tool_calls

_OCTET = "application/octet-stream"


def create_app(model, tokenizer, stages, node_url=None, peers=None,
               num_layers=None, gossip_interval=2.0, ttl=30.0):
    """Crea l'app FastAPI di un BlockServer. Con node_url/peers attiva la gossip
    discovery decentralizzata (Modalità A); senza, comportamento di Parte 1."""
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}
    own_stages = {"embed": stages.embed, "head": stages.head, "decoders": list(prepared.keys())}
    registry = Registry()
    if node_url:
        registry.upsert(node_url, own_stages, now=time.time(), ttl=ttl)

    stop_ids = set()
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    if tokenizer is not None:
        for _t in ("<|im_end|>", "<|endoftext|>"):
            _i = tokenizer.convert_tokens_to_ids(_t)
            if isinstance(_i, int) and _i >= 0 and _i != tokenizer.unk_token_id:
                stop_ids.add(int(_i))
    _model_id = getattr(model.config, "_name_or_path", "synapse")
    _entry_job = {"n": 0}

    async def _gossip_loop():
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                now = time.time()
                if node_url:
                    registry.upsert(node_url, own_stages, now=now, ttl=ttl)
                for peer in (peers or []):
                    try:
                        resp = await client.get(f"{peer}/registry")
                        registry.merge(resp.json().get("nodes", {}), now=now, ttl=ttl)
                    except Exception:
                        pass
                registry.prune(now)
                await asyncio.sleep(gossip_interval)

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(_gossip_loop()) if node_url else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/registry")
    async def get_registry():
        return {"num_layers": num_layers, "model": getattr(model.config, "_name_or_path", "?"),
                "nodes": registry.stages_by_url(time.time())}

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "model": getattr(model.config, "_name_or_path", "?"),
            "stages": {"embed": embed_block is not None, "head": head_block is not None,
                       "decoders": list(prepared.keys())},
        }

    @app.post("/embed")
    async def embed(job_id: str, request: Request):
        if embed_block is None:
            return JSONResponse({"error": "questo nodo non serve lo stage embed"}, status_code=400)
        t = decode_tensors(await request.body())
        h = embed_block.run_block(t["input_ids"])
        return Response(encode_tensors({"hidden_states": h}), media_type=_OCTET)

    @app.post("/decode/{block_key}")
    async def decode(block_key: str, job_id: str, request: Request):
        if block_key not in prepared:
            return JSONResponse({"error": f"blocco {block_key} non servito"}, status_code=400)
        t = decode_tensors(await request.body())
        job = jobs.setdefault(job_id, {})
        block = job.get(block_key)
        if block is None:
            layers, rotary = prepared[block_key]
            block = DecoderBlock(layers, rotary)   # cache propria per (job, blocco)
            job[block_key] = block
        h = block.run_block(t["hidden_states"], t["cache_position"])
        return Response(encode_tensors({"hidden_states": h}), media_type=_OCTET)

    @app.post("/head")
    async def head(job_id: str, request: Request, topk: int = 1):
        if head_block is None:
            return JSONResponse({"error": "questo nodo non serve lo stage head"}, status_code=400)
        t = decode_tensors(await request.body())
        logits = head_block.run_block(t["hidden_states"])[:, -1, :]
        k = min(int(topk), logits.shape[-1])
        vals, idx = torch.topk(logits[0], k=k)
        ids = idx.tolist()
        return JSONResponse({"token_id": ids[0], "topk_ids": ids, "topk_logits": vals.tolist()})

    @app.delete("/job/{job_id}")
    async def end_job(job_id: str):
        jobs.pop(job_id, None)
        return {"ok": True}

    @app.get("/v1/models")
    async def v1_models():
        return {"object": "list", "data": [{"id": "synapse", "object": "model", "owned_by": "synapse"},
                                            {"id": _model_id, "object": "model", "owned_by": "synapse"}]}

    @app.post("/v1/chat/completions")
    async def v1_chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        max_new = int(body.get("max_tokens", 256))
        tools = body.get("tools")
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        chain = build_chain(registry.stages_by_url(time.time()), num_layers)
        if chain is None:
            return JSONResponse({"error": {"message": "rete non operativa", "type": "not_operational"}}, status_code=503)
        embed_url, decoders, head_url = chain
        try:
            prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join((m.get("content") or "") for m in messages)
        _entry_job["n"] += 1
        job_id = f"entry{_entry_job['n']}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async def run_embed(cur):
                r = await client.post(f"{embed_url}/embed", params={"job_id": job_id},
                                      content=encode_tensors({"input_ids": cur}))
                return decode_tensors(r.content)["hidden_states"]

            async def run_decoders(h, cache_position):
                for bk, url in decoders:
                    r = await client.post(f"{url}/decode/{bk}", params={"job_id": job_id},
                                          content=encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                    h = decode_tensors(r.content)["hidden_states"]
                return h

            async def run_head(h, topk):
                r = await client.post(f"{head_url}/head", params={"job_id": job_id, "topk": topk},
                                      content=encode_tensors({"hidden_states": h}))
                return r.json()

            tokens, prompt_len, finish_reason = await generate_tokens(
                tokenizer, prompt, max_new, sampling, stop_ids, run_embed, run_decoders, run_head)
            for url in {embed_url, head_url, *(u for _, u in decoders)}:
                try:
                    await client.delete(f"{url}/job/{job_id}")
                except Exception:
                    pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
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

    return app
