from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from synapse.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from synapse.net.wire import encode_tensors, decode_tensors

_OCTET = "application/octet-stream"


def create_app(model, tokenizer, stages):
    """Crea l'app FastAPI di un BlockServer che serve gli `stages` dati, sopra un
    `model` GIA' caricato (condiviso tra i job in questo processo)."""
    app = FastAPI()
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}   # job_id -> {block_key: DecoderBlock}  (KV-cache per-job)

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
    async def head(job_id: str, request: Request):
        if head_block is None:
            return JSONResponse({"error": "questo nodo non serve lo stage head"}, status_code=400)
        t = decode_tensors(await request.body())
        logits = head_block.run_block(t["hidden_states"])
        token_id = int(logits[:, -1, :].argmax(-1).item())
        return JSONResponse({"token_id": token_id})

    @app.delete("/job/{job_id}")
    async def end_job(job_id: str):
        jobs.pop(job_id, None)
        return {"ok": True}

    return app
