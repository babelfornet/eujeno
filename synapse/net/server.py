import asyncio
import time
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from synapse.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.discovery import Registry

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

    return app
