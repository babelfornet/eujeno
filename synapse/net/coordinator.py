import asyncio

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

from synapse.net.framing import pack, unpack
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.discovery import build_chain


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
        c = conns[conn_id]
        req_id = _next_id("r")
        fut = asyncio.get_event_loop().create_future()
        c["pending"][req_id] = fut
        await c["ws"].send_bytes(pack({**header, "req_id": req_id}, payload))
        return await fut   # (resp_header, resp_payload)

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
            conns.pop(conn_id, None)

    @app.get("/registry")
    async def registry():
        return {"num_layers": num_layers,
                "nodes": [{"conn": cid, "stages": c["stages"]} for cid, c in conns.items()]}

    @app.post("/infer")
    async def infer(request: Request):
        body = await request.json()
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 8))
        chain = build_chain({cid: c["stages"] for cid, c in conns.items()}, num_layers)
        if chain is None:
            return {"ok": False, "error": "modello non operativo: coverage incompleta"}
        embed_c, decoders, head_c = chain

        ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = ids.shape[1]
        cache_position = torch.arange(seq_len)
        cur = ids
        tokens = []
        job_id = _next_id("job")
        try:
            for step in range(max_new):
                _, p = await _call(embed_c, {"op": "embed", "job_id": job_id},
                                   encode_tensors({"input_ids": cur}))
                h = decode_tensors(p)["hidden_states"]
                for block_key, cid in decoders:
                    _, p = await _call(cid, {"op": "decode", "block_key": block_key, "job_id": job_id},
                                       encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                    h = decode_tensors(p)["hidden_states"]
                rh, _ = await _call(head_c, {"op": "head", "job_id": job_id},
                                    encode_tensors({"hidden_states": h}))
                tokens.append(rh["token_id"])
                cur = torch.tensor([[rh["token_id"]]])
                cache_position = torch.tensor([seq_len + step])
        finally:
            for cid in {embed_c, head_c, *(c for _, c in decoders)}:
                try:
                    await _call(cid, {"op": "end", "job_id": job_id})
                except Exception:
                    pass
        return {"ok": True, "model": model_id, "prompt": prompt,
                "text": tokenizer.decode(tokens), "tokens": tokens}

    return app
