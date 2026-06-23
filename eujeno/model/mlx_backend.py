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
import glob
import os

import numpy as np

from eujeno.net.wire import decode_tensors, encode_tensors


def load_partial_mlx(model_id: str, stages):
    """Partial loader for MLX (mirrors loader.load_partial_model for torch): builds the MLX
    model structure but loads into memory ONLY the weights of the assigned layers (+ embed/head
    if served), and drops every other module so its (random, lazy) init never materializes on
    eval. RAM/VRAM scales with the slab, not the whole model — the point of the architecture.
    Returns (model, num_layers)."""
    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import _get_classes, load_config

    model_path = Path(snapshot_download(model_id))
    config = load_config(model_path)
    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    num_layers = len(model.model.layers)

    tie = bool(getattr(model.args, "tie_word_embeddings", False))
    prefixes = []
    if stages.embed or (stages.head and tie):
        prefixes.append("model.embed_tokens.")
    if stages.head:
        prefixes.append("model.norm.")
        if not tie:
            prefixes.append("lm_head.")
    served = set()
    for (lo, hi) in stages.decoders:
        served.update(range(lo, hi))
        for i in range(lo, hi):
            prefixes.append(f"model.layers.{i}.")

    # load ONLY the assigned weights from the safetensors shards
    weights = {}
    for wf in glob.glob(os.path.join(str(model_path), "model*.safetensors")):
        for k, v in mx.load(wf).items():
            if any(k.startswith(p) for p in prefixes):
                weights[k] = v
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # quantize per the checkpoint's config; the predicate only quantizes a module when its
    # scales were loaded (i.e. it's part of our slab) — same logic mlx_lm.load_model uses.
    q = config.get("quantization")
    if q:
        def class_predicate(p, m):
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    mode=q.get("mode", "affine"), class_predicate=class_predicate)

    model.load_weights(list(weights.items()), strict=False)

    # drop the modules we don't serve so eval doesn't materialize their random init
    class _Drop(nn.Module):
        pass
    for i in range(num_layers):
        if i not in served:
            model.model.layers[i] = _Drop()
    if not (stages.embed or (stages.head and tie)):
        model.model.embed_tokens = _Drop()
    if not stages.head:
        model.model.norm = _Drop()
        if hasattr(model, "lm_head"):
            model.lm_head = _Drop()

    mx.eval(model.parameters())
    model.eval()
    return model, num_layers


class MlxNodeState:
    """MLX-backed node state. Mirrors NodeState's stages_dict()/jobs surface so `node.run_node`
    drives it unchanged. Partial-loads ONLY its assigned layers; jobs: job_id -> per-layer MLX
    KV cache (one entry per MODEL layer; only the served slab's entries are ever used)."""
    is_mlx = True

    def __init__(self, model_id: str, stages):
        self.model, self.num_layers = load_partial_mlx(model_id, stages)
        self.stages = stages
        self.jobs = {}

    def stages_dict(self) -> dict:
        return {"embed": self.stages.embed, "head": self.stages.head,
                "decoders": [f"{lo}-{hi}" for (lo, hi) in self.stages.decoders]}


def _job_cache(state, job_id):
    """Per-job KV cache, one entry per MODEL layer. A decoder block [lo,hi) uses the slice
    cache[lo:hi]; positions/rope stay consistent because each layer's cache advances once per
    token regardless of which node owns it. (Built directly, not via make_prompt_cache, since
    dropped non-slab modules would trip that helper.)"""
    cache = state.jobs.get(job_id)
    if cache is None:
        from mlx_lm.models.cache import KVCache
        cache = [KVCache() for _ in range(state.num_layers)]
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
