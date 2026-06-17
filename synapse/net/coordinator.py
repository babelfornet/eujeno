import asyncio
import random

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

from synapse.net.framing import pack, unpack
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.discovery import build_chain
from synapse.net.sampling import sample_token


MAX_FAILOVERS = 5


class _NodeFailure(Exception):
    def __init__(self, conn_id):
        super().__init__(conn_id)
        self.conn_id = conn_id


def create_coordinator_app(model_id: str, num_layers: int, tokenizer):
    """Coordinator-relay: i nodi si connettono via WS e annunciano gli stage; POST /infer
    guida la generazione relayando ogni hop al nodo giusto."""
    app = FastAPI()
    conns = {}        # conn_id -> {"ws", "stages", "pending": {req_id: Future}}
    counter = {"n": 0}

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
            if c is not None:   # fai fallire le richieste pendenti invece di lasciarle appese
                for fut in c["pending"].values():
                    if not fut.done():
                        fut.set_exception(ConnectionError(f"nodo {conn_id} disconnesso"))

    @app.get("/registry")
    async def registry():
        return {"num_layers": num_layers,
                "nodes": [{"conn": cid, "stages": c["stages"]} for cid, c in conns.items()]}

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

    return app
