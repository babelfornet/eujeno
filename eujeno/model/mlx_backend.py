# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""MLX (Apple Silicon) backend — a node runs its blocks via a 4-bit MLX-quantized model.

Selected with `serve --backend mlx`. A node can host the whole model OR a contiguous slab of
layers (multi-node). The node holds the (quantized) model, keeps a per-job MLX KV cache, and
serves the coordinator's ops:
  - a fused `chain` (its consecutive blocks: [embed?] → decoder slab → [head?]) in one call,
    keeping the activation as an MLX array on-device between steps;
  - or a single `embed` / `decode` / `head` op when it hosts just one block.
At the wire boundary the hidden state is converted MLX(bf16) ↔ torch(bf16) safetensors, so an
MLX node interoperates with the torch coordinator and with torch/MLX peer nodes. Apple-only.

DEPENDENCY NOTE: install via the `mlx` extra (`pip install -e '.[mlx]'`), pinned to mlx-lm<0.30
— mlx-lm 0.30+ requires transformers>=5, which conflicts with eujeno's transformers==4.46.3 pin.
With mlx-lm 0.29.x the MLX and torch backends coexist in ONE env (transformers 4.46.3 / numpy 1.26).
Single-node ~68 tok/s end-to-end (vs ~10-11 torch bf16). NOTE: the model is loaded whole and the
slab is sliced from it (partial loading of MLX checkpoints is future work); multi-node MLX wins
when slabs live on SEPARATE Apple machines (one Mac GPU split across processes just timeshares).
"""
import numpy as np

from eujeno.net.wire import decode_tensors, encode_tensors


class MlxNodeState:
    """MLX-backed node state. Mirrors NodeState's stages_dict()/jobs surface so `node.run_node`
    drives it unchanged. Holds the whole quantized model; serves whichever stages it was assigned
    (slicing the slab from the full model). jobs: job_id -> full MLX prompt cache (one per layer)."""
    is_mlx = True

    def __init__(self, model_id: str, stages):
        from mlx_lm import load
        self.model, _ = load(model_id)
        self.stages = stages
        self.jobs = {}

    def stages_dict(self) -> dict:
        return {"embed": self.stages.embed, "head": self.stages.head,
                "decoders": [f"{lo}-{hi}" for (lo, hi) in self.stages.decoders]}


def _job_cache(state, job_id):
    """Per-job MLX prompt cache (one entry per MODEL layer). A decoder block [lo,hi) uses the
    slice cache[lo:hi]; positions/rope stay consistent because each layer's cache advances once
    per token regardless of which node owns it."""
    cache = state.jobs.get(job_id)
    if cache is None:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(state.model)
        state.jobs[job_id] = cache
    return cache


def _to_mx(t):
    """torch hidden states (bf16, from the wire) -> MLX bfloat16 array."""
    import mlx.core as mx
    import torch
    return mx.array(t.detach().to(torch.float32).cpu().numpy()).astype(mx.bfloat16)


def _to_torch(a):
    """MLX array -> torch bf16 tensor for the safetensors wire."""
    import mlx.core as mx
    import torch
    return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)


def _apply_head(model, h):
    h = model.model.norm(h)
    if getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def _topk_response(ln, topk):
    k = min(int(topk), int(ln.shape[-1]))
    if k <= 1:
        tid = int(ln.argmax())
        return {"ok": True, "token_id": tid, "topk_ids": [tid], "topk_logits": [float(ln[tid])]}, b""
    idx = np.argpartition(-ln, k - 1)[:k]
    idx = idx[np.argsort(-ln[idx])]
    return {"ok": True, "token_id": int(idx[0]),
            "topk_ids": [int(i) for i in idx],
            "topk_logits": [float(ln[i]) for i in idx]}, b""


def handle_request_mlx(state: MlxNodeState, header: dict, payload: bytes):
    """Run an op (or a fused chain of ops) on an MLX node, returning the torch `head`/`decode`
    response shape so it is wire-compatible with the torch coordinator and peers."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    op = header["op"]
    if op == "end":
        state.jobs.pop(header["job_id"], None)
        return {"ok": True}, b""

    if op == "chain":
        steps = header["steps"]
    elif op == "decode":
        steps = [{"op": "decode", "block_key": header["block_key"]}]
    elif op in ("embed", "head"):
        steps = [{"op": op}]
    else:
        return {"ok": False, "error": f"mlx backend: unknown op '{op}'"}, b""

    t = decode_tensors(payload) if payload else {}
    job_cache = _job_cache(state, header["job_id"])
    model = state.model
    h = None
    for st in steps:
        kind = st["op"]
        if kind == "embed":
            ids = mx.array(t["input_ids"].cpu().numpy().astype(np.int32))
            h = model.model.embed_tokens(ids)
        elif kind == "decode":
            lo, hi = (int(x) for x in st["block_key"].split("-"))
            if h is None:                       # slab is the entry block -> hidden from the wire
                h = _to_mx(t["hidden_states"])
            layer_caches = job_cache[lo:hi]
            mask = create_attention_mask(h, layer_caches[0])
            for layer, c in zip(model.model.layers[lo:hi], layer_caches):
                h = layer(h, mask, c)
        elif kind == "head":
            if h is None:
                h = _to_mx(t["hidden_states"])
            logits = _apply_head(model, h)[:, -1, :].astype(mx.float32)
            mx.eval(logits)
            return _topk_response(np.array(logits[0]), header.get("topk", 1))
        else:
            return {"ok": False, "error": f"mlx backend: unknown step '{kind}'"}, b""

    mx.eval(h)                                  # no head in this segment -> forward hidden states
    return {"ok": True}, encode_tensors({"hidden_states": _to_torch(h)})
