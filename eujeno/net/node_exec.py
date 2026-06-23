# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import torch
from eujeno.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from eujeno.net.wire import encode_tensors, decode_tensors


class NodeState:
    """Local state of a worker node: served blocks + per-job KV-cache."""
    def __init__(self, model, stages, device="cpu"):
        self.device = device
        self.embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
        self.head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
        self.prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
        self.jobs = {}   # job_id -> {block_key: DecoderBlock}

    def stages_dict(self) -> dict:
        return {"embed": self.embed_block is not None,
                "head": self.head_block is not None,
                "decoders": list(self.prepared.keys())}


def handle_request(state, header: dict, payload: bytes):
    """Run a hop. Returns (resp_header: dict, resp_payload: bytes)."""
    if getattr(state, "is_mlx", False):       # MLX-backed node: route to the MLX backend
        from eujeno.model.mlx_backend import handle_request_mlx
        return handle_request_mlx(state, header, payload)
    op = header["op"]
    dev = state.device                       # inputs arrive on CPU from the wire;
    # move them onto the node's device (mps/cuda) and bring outputs back to CPU
    # for serialization. On CPU these .to() calls are no-ops.
    if op == "embed":
        t = decode_tensors(payload)
        h = state.embed_block.run_block(t["input_ids"].to(dev))
        return {"ok": True}, encode_tensors({"hidden_states": h.to("cpu")})
    if op == "decode":
        block_key = header["block_key"]
        job = state.jobs.setdefault(header["job_id"], {})
        block = job.get(block_key)
        if block is None:
            layers, rotary = state.prepared[block_key]
            block = DecoderBlock(layers, rotary)
            job[block_key] = block
        t = decode_tensors(payload)
        h = block.run_block(t["hidden_states"].to(dev), t["cache_position"].to(dev))
        return {"ok": True}, encode_tensors({"hidden_states": h.to("cpu")})
    if op == "head":
        t = decode_tensors(payload)
        logits = state.head_block.run_block(t["hidden_states"].to(dev))[:, -1, :]
        k = min(int(header.get("topk", 1)), logits.shape[-1])
        vals, idx = torch.topk(logits[0], k=k)
        ids = idx.tolist()
        return {"ok": True, "token_id": ids[0],
                "topk_ids": ids, "topk_logits": vals.tolist()}, b""
    if op == "chain":
        # Run a maximal run of this node's consecutive blocks (embed / decoder(s) / head)
        # in ONE round-trip, keeping the activation ON-DEVICE between steps — no per-block
        # WS hop and no per-block CPU<->device copy. The coordinator only sends a chain op
        # to nodes that advertised the "chain" cap, so this is purely additive.
        job = state.jobs.setdefault(header["job_id"], {})
        t = decode_tensors(payload)
        cache_position = t.get("cache_position")
        if cache_position is not None:
            cache_position = cache_position.to(dev)
        h = None
        for st in header["steps"]:
            kind = st["op"]
            if kind == "embed":
                h = state.embed_block.run_block(t["input_ids"].to(dev))
            elif kind == "decode":
                bk = st["block_key"]
                block = job.get(bk)
                if block is None:
                    layers, rotary = state.prepared[bk]
                    block = DecoderBlock(layers, rotary)
                    job[bk] = block
                src = h if h is not None else t["hidden_states"].to(dev)
                h = block.run_block(src, cache_position)
            elif kind == "head":
                logits = state.head_block.run_block(h)[:, -1, :]
                k = min(int(header.get("topk", 1)), logits.shape[-1])
                vals, idx = torch.topk(logits[0], k=k)
                ids = idx.tolist()
                return {"ok": True, "token_id": ids[0],
                        "topk_ids": ids, "topk_logits": vals.tolist()}, b""
        return {"ok": True}, encode_tensors({"hidden_states": h.to("cpu")})
    if op == "end":
        state.jobs.pop(header["job_id"], None)
        return {"ok": True}, b""
    return {"ok": False, "error": f"unknown op: {op}"}, b""
