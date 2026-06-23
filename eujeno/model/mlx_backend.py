# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""MLX (Apple Silicon) backend — a node runs its blocks via a 4-bit MLX-quantized model.

Selected with `serve --backend mlx`. The node holds the (quantized) model and serves the
coordinator's fused `chain` op as a SINGLE MLX forward step with a per-job KV cache:
embed → decoder layers → head all stay as MLX arrays on-device, no per-block hop. The wire
protocol (safetensors hidden states / token ids) is unchanged, so an MLX node interoperates
with the existing coordinator. Apple-only; `mlx-lm` is an optional dependency.

Scope (prototype): single-node deployments, where the whole model lives on one MLX node and
the coordinator emits one `chain` op per token. Multi-node MLX slabs (per-block torch↔mlx
hidden-state conversion at the wire boundary) are future work — see the feat/mlx-backend branch.

DEPENDENCY ISOLATION (important): `mlx-lm` pulls transformers>=5 + numpy>=2, which CONFLICT
with eujeno's torch path (pinned transformers==4.46.3, numpy<2). An MLX node must therefore
run in its OWN environment (it doesn't use the torch stack at all). Do NOT `pip install mlx-lm`
into a torch-backend venv — it silently upgrades transformers/numpy and breaks torch nodes.
Measured: ~59 tok/s end-to-end through the coordinator (vs ~10-11 torch bf16), output coherent.
"""
import numpy as np

from eujeno.net.wire import decode_tensors


class MlxNodeState:
    """MLX-backed node state. Mirrors NodeState's stages_dict()/jobs surface so that
    `node.run_node` drives it unchanged. Holds the whole quantized model + a per-job
    MLX KV cache (list of per-layer caches)."""
    is_mlx = True

    def __init__(self, model_id: str, stages):
        from mlx_lm import load
        self.model, _ = load(model_id)
        self.stages = stages
        self.jobs = {}   # job_id -> mlx prompt cache

    def stages_dict(self) -> dict:
        return {"embed": self.stages.embed, "head": self.stages.head,
                "decoders": [f"{lo}-{hi}" for (lo, hi) in self.stages.decoders]}


def handle_request_mlx(state: MlxNodeState, header: dict, payload: bytes):
    """Serve one op on an MLX node. Single-node receives the coordinator's fused `chain`
    (embed → decoder slab → head); we run it as one MLX forward with the per-job cache and
    return the top-k, matching the torch `head`/`chain` response shape."""
    import mlx.core as mx
    op = header["op"]
    if op == "end":
        state.jobs.pop(header["job_id"], None)
        return {"ok": True}, b""
    if op == "chain":
        steps = header["steps"]
        if steps[0]["op"] != "embed" or steps[-1]["op"] != "head":
            return {"ok": False,
                    "error": "mlx backend: chain must start at embed and end at head"}, b""
        cache = state.jobs.get(header["job_id"])
        if cache is None:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(state.model)
            state.jobs[header["job_id"]] = cache
        t = decode_tensors(payload)
        ids = mx.array(t["input_ids"].cpu().numpy().astype(np.int32))
        logits = state.model(ids, cache=cache)
        last = logits[:, -1, :].astype(mx.float32)
        mx.eval(last)
        ln = np.array(last[0])
        k = min(int(header.get("topk", 1)), int(ln.shape[-1]))
        if k <= 1:
            tid = int(ln.argmax())
            return {"ok": True, "token_id": tid,
                    "topk_ids": [tid], "topk_logits": [float(ln[tid])]}, b""
        idx = np.argpartition(-ln, k - 1)[:k]
        idx = idx[np.argsort(-ln[idx])]
        return {"ok": True, "token_id": int(idx[0]),
                "topk_ids": [int(i) for i in idx],
                "topk_logits": [float(ln[i]) for i in idx]}, b""
    return {"ok": False,
            "error": f"mlx backend: unsupported op '{op}' (single-node deployment uses chain)"}, b""
