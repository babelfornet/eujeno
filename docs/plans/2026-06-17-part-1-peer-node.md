# Part 1 — Peer Node & Layer Execution (foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove in a single process that an LLM can be split into blocks of layers and executed hop-by-hop, reproducing **exactly** the generation of the whole model, with a serializable per-block KV-cache that survives a byte round-trip (handoff/restart simulation).

**Architecture:** The full model is loaded once (reference), then `split_into_blocks` divides it into an EMBED block, N DECODER blocks (contiguous slabs of layers with `layer_idx` remapped to local indices + a local `DynamicCache` KV-cache) and a HEAD block. `pipeline_generate` runs a greedy generation through the blocks and must produce the **same** token sequence as `reference_generate`. Two pure serialization primitives (KV-cache and hop payload, both via safetensors) prepare the network transport of Part 3.

**Tech Stack:** Python 3.11 · PyTorch (fp32, CPU) · Hugging Face `transformers==4.46.3` (stable `DynamicCache` API) · `safetensors` · `accelerate` · `huggingface_hub` · `pytest`. Test model: `Qwen/Qwen2.5-0.5B-Instruct` (ungated, 24 layers, hidden 896).

**Scope (this plan):** build-order steps 1, 2, 4 of [ADR-0001](../decisions/ADR-0001-implementation-forks.md) + cache/payload serialization primitives ([PRD Part 1](../prd/part-1-peer-node.md) §3).
**Out of scope (later plans):** real partial-loading via `init_empty_weights`/`load_checkpoint_in_model` (a memory concern, goes with the wire format), FastAPI transport, DHT/routing, SQLite persistence. Here the blocks live in the same process.

> **Version note:** the code mirrors the `Model.forward` of `transformers==4.46.3` (decoder layer signature with `position_embeddings`). If the engineer uses a different version and the golden test fails, align the layer call with the `forward` of the installed model — the golden test (Task 6) is the safety net that makes this deterministic. See [ADR-0001](../decisions/ADR-0001-implementation-forks.md) Q2.

---

## File Structure

```
pyproject.toml                 # project + pinned dependencies + pytest config
axyn/
  __init__.py
  config.py                    # DEFAULT_MODEL_ID, DTYPE, DEVICE
  model/
    __init__.py
    loader.py                  # load_full_model(), model_dims()
    masking.py                 # build_causal_mask()
    cache.py                   # cache_to_bytes() / cache_from_bytes()
    payload.py                 # HopPayload + to_bytes()/from_bytes()
    blocks.py                  # BlockRunner (EMBED/DECODER/HEAD), split_into_blocks()
    generate.py                # reference_generate(), pipeline_generate()
tests/
  conftest.py                  # full_model fixture (session-scoped)
  test_loader.py
  test_masking.py
  test_cache.py
  test_payload.py
  test_blocks.py
  test_golden.py               # THE golden test
  test_resilience.py           # cache round-trip mid-generation
```

Responsibility per file (one each): `loader` loads · `masking` builds masks · `cache`/`payload` serialize · `blocks` runs layers · `generate` orchestrates greedy generation.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `axyn/__init__.py`, `axyn/model/__init__.py`
- Create: `axyn/config.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "axyn"
version = "0.0.1"
description = "Decentralized peer-to-peer LLM inference network"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "transformers==4.46.3",
    "accelerate>=1.0",
    "safetensors>=0.4.5",
    "huggingface_hub>=0.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
markers = ["slow: downloads/runs the model (slow)"]
addopts = "-q"

[tool.setuptools.packages.find]
include = ["axyn*"]
```

- [ ] **Step 2: Create the package inits and `config.py`**

`axyn/__init__.py`: empty file.
`axyn/model/__init__.py`: empty file.

`axyn/config.py`:
```python
import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float32   # fp32 on CPU for determinism (see ADR-0001 Fork D)
DEVICE = "cpu"
```

- [ ] **Step 3: Create the environment and install**

Run:
```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```
Expected: installation completes without errors; `python -c "import transformers, torch, safetensors, accelerate; print('ok')"` prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml axyn/__init__.py axyn/model/__init__.py axyn/config.py
git commit -m "chore: axyn package scaffolding + pinned dependencies"
```

---

## Task 1: Model loader

**Files:**
- Create: `axyn/model/loader.py`
- Test: `tests/conftest.py`, `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:
```python
import pytest
import torch
from axyn.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from axyn.model.loader import load_full_model

@pytest.fixture(scope="session")
def full_model():
    torch.manual_seed(0)
    model, tokenizer = load_full_model(DEFAULT_MODEL_ID, DTYPE, DEVICE)
    model.eval()
    return model, tokenizer
```

`tests/test_loader.py`:
```python
import pytest
from axyn.model.loader import model_dims

@pytest.mark.slow
def test_loads_with_expected_dims(full_model):
    model, tokenizer = full_model
    dims = model_dims(model)
    assert dims["num_layers"] == 24
    assert dims["hidden_size"] == 896
    assert tokenizer is not None
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_loader.py -m slow -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` on `axyn.model.loader`.

- [ ] **Step 3: Implement `loader.py`**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_full_model(model_id: str, dtype: torch.dtype, device: str):
    """Load full model + tokenizer. Used as the reference and as the
    source from which the blocks are extracted (Task 5)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


def model_dims(model) -> dict:
    cfg = model.config
    return {
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    }
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_loader.py -m slow -v`
Expected: PASS (the first run downloads ~1GB of the model).

- [ ] **Step 5: Commit**

```bash
git add axyn/model/loader.py tests/conftest.py tests/test_loader.py
git commit -m "feat(model): full model loader + dimension introspection"
```

---

## Task 2: Causal mask builder

**Files:**
- Create: `axyn/model/masking.py`
- Test: `tests/test_masking.py`

- [ ] **Step 1: Write the failing test**

`tests/test_masking.py`:
```python
import torch
from axyn.model.masking import build_causal_mask


def test_prefill_mask_is_lower_triangular():
    cache_position = torch.arange(3)        # prefill of 3 tokens, kv_len=3
    mask = build_causal_mask(cache_position, kv_len=3, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 3, 3)
    neg = torch.finfo(torch.float32).min
    # query 0 sees only key 0
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] == neg
    assert mask[0, 0, 0, 2] == neg
    # query 2 sees all
    assert torch.all(mask[0, 0, 2, :] == 0.0)


def test_decode_step_attends_to_all_past():
    cache_position = torch.tensor([5])      # 1 query token at position 5, kv_len=6
    mask = build_causal_mask(cache_position, kv_len=6, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 1, 6)
    assert torch.all(mask[0, 0, 0, :] == 0.0)   # attends to all 6 past positions
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_masking.py -v`
Expected: FAIL with `ImportError` on `axyn.model.masking`.

- [ ] **Step 3: Implement `masking.py`**

```python
import torch


def build_causal_mask(cache_position: torch.Tensor, kv_len: int,
                      dtype: torch.dtype, device: str) -> torch.Tensor:
    """4D additive causal mask [1,1,q_len,kv_len] for batch=1 without padding.
    cache_position holds the absolute positions of the query tokens."""
    q_len = cache_position.shape[0]
    key_pos = torch.arange(kv_len, device=device)
    allowed = key_pos[None, :] <= cache_position[:, None].to(device)   # [q_len, kv_len] bool
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_masking.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add axyn/model/masking.py tests/test_masking.py
git commit -m "feat(model): causal mask builder for block execution"
```

---

## Task 3: KV-cache serialization (shared primitive #2/3)

**Files:**
- Create: `axyn/model/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cache.py`:
```python
import torch
from transformers import DynamicCache
from axyn.model.cache import cache_to_bytes, cache_from_bytes


def _make_cache(num_layers, seq=4, heads=2, head_dim=8):
    legacy = tuple(
        (torch.randn(1, heads, seq, head_dim), torch.randn(1, heads, seq, head_dim))
        for _ in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(legacy)


def test_cache_roundtrip_preserves_tensors():
    cache = _make_cache(num_layers=3)
    data = cache_to_bytes(cache)
    restored = cache_from_bytes(data)
    orig, back = cache.to_legacy_cache(), restored.to_legacy_cache()
    assert len(back) == 3
    for (k0, v0), (k1, v1) in zip(orig, back):
        assert torch.equal(k0, k1)
        assert torch.equal(v0, v1)


def test_cache_roundtrip_preserves_seq_length():
    cache = _make_cache(num_layers=2, seq=7)
    restored = cache_from_bytes(cache_to_bytes(cache))
    assert restored.get_seq_length() == 7
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL with `ImportError` on `axyn.model.cache`.

- [ ] **Step 3: Implement `cache.py`**

```python
import safetensors.torch
from transformers import DynamicCache


def cache_to_bytes(cache: DynamicCache) -> bytes:
    """Serialize a (per-block) DynamicCache into safetensors bytes."""
    legacy = cache.to_legacy_cache()
    tensors = {}
    for i, (key, value) in enumerate(legacy):
        tensors[f"key_{i}"] = key.contiguous()
        tensors[f"value_{i}"] = value.contiguous()
    return safetensors.torch.save(tensors)


def cache_from_bytes(data: bytes) -> DynamicCache:
    tensors = safetensors.torch.load(data)
    num_layers = len(tensors) // 2
    legacy = tuple(
        (tensors[f"key_{i}"], tensors[f"value_{i}"]) for i in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(legacy)
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_cache.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add axyn/model/cache.py tests/test_cache.py
git commit -m "feat(model): KV-cache serialization via safetensors (round-trip)"
```

---

## Task 4: Hop payload serialization (shared primitive #1 on the wire)

**Files:**
- Create: `axyn/model/payload.py`
- Test: `tests/test_payload.py`

- [ ] **Step 1: Write the failing test**

`tests/test_payload.py`:
```python
import torch
from axyn.model.payload import HopPayload


def test_payload_roundtrip():
    p = HopPayload(
        job_id="job-abc",
        hop=2,
        token_position=5,
        hidden_states=torch.randn(1, 1, 896),
        position_ids=torch.tensor([[5]]),
        cache_position=torch.tensor([5]),
        attention_mask=None,
    )
    back = HopPayload.from_bytes(p.to_bytes())
    assert back.job_id == "job-abc"
    assert back.hop == 2
    assert back.token_position == 5
    assert torch.equal(back.hidden_states, p.hidden_states)
    assert torch.equal(back.position_ids, p.position_ids)
    assert torch.equal(back.cache_position, p.cache_position)
    assert back.attention_mask is None
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_payload.py -v`
Expected: FAIL with `ImportError` on `axyn.model.payload`.

- [ ] **Step 3: Implement `payload.py`**

```python
import json
from dataclasses import dataclass

import torch
import safetensors.torch


@dataclass
class HopPayload:
    """Payload of a hop on the wire (Part 1 §3). The KV-cache does NOT travel
    here: it stays local to the holder (session affinity, Part 3)."""
    job_id: str
    hop: int
    token_position: int
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def to_bytes(self) -> bytes:
        header = {"job_id": self.job_id, "hop": self.hop, "token_position": self.token_position}
        header_bytes = json.dumps(header).encode("utf-8")
        tensors = {
            "_header": torch.tensor(list(header_bytes), dtype=torch.uint8),
            "hidden_states": self.hidden_states.contiguous(),
            "position_ids": self.position_ids.contiguous(),
            "cache_position": self.cache_position.contiguous(),
        }
        if self.attention_mask is not None:
            tensors["attention_mask"] = self.attention_mask.contiguous()
        return safetensors.torch.save(tensors)

    @classmethod
    def from_bytes(cls, data: bytes) -> "HopPayload":
        t = safetensors.torch.load(data)
        header = json.loads(bytes(t["_header"].tolist()).decode("utf-8"))
        return cls(
            job_id=header["job_id"],
            hop=header["hop"],
            token_position=header["token_position"],
            hidden_states=t["hidden_states"],
            position_ids=t["position_ids"],
            cache_position=t["cache_position"],
            attention_mask=t.get("attention_mask"),
        )
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add axyn/model/payload.py tests/test_payload.py
git commit -m "feat(model): HopPayload serialization via safetensors (round-trip)"
```

---

## Task 5: BlockRunner + split_into_blocks

**Files:**
- Create: `axyn/model/blocks.py`
- Test: `tests/test_blocks.py`

> **Design note:** `split_into_blocks` **mutates** `layer.self_attn.layer_idx`, remapping it to local indices (0-based per block), so each block uses a local `DynamicCache` for its own layers only (forward-compatible with the distributed model). For this reason the tests and the golden test (Task 6) always capture the reference from the whole model **before** calling `split_into_blocks`.

- [ ] **Step 1: Write the failing test**

`tests/test_blocks.py`:
```python
import pytest
import torch
from axyn.model.blocks import split_into_blocks


@pytest.mark.slow
def test_embed_block_matches_model_embedding(full_model):
    model, tokenizer = full_model
    ids = tokenizer("Hello world", return_tensors="pt").input_ids
    expected = model.model.embed_tokens(ids)
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    out = embed.run_block(ids)
    assert torch.equal(out, expected)


@pytest.mark.slow
def test_head_block_matches_model_head(full_model):
    model, tokenizer = full_model
    h = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float32)
    expected = model.lm_head(model.model.norm(h))
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    out = head.run_block(h)
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.slow
def test_decoder_blocks_cover_all_layers(full_model):
    model, _ = full_model
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    assert len(decoders) == 2
    assert sum(len(d.layers) for d in decoders) == 24
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_blocks.py -m slow -v`
Expected: FAIL with `ImportError` on `axyn.model.blocks`.

- [ ] **Step 3: Implement `blocks.py`**

```python
import torch
from transformers import DynamicCache

from .masking import build_causal_mask


class EmbedBlock:
    """First block: input_ids -> hidden_states."""
    def __init__(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @torch.no_grad()
    def run_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class DecoderBlock:
    """Contiguous slab of layers [lo, hi). Keeps a LOCAL KV-cache for its
    own layers only (0-based remapped indices)."""
    def __init__(self, layers, rotary_emb):
        self.layers = layers
        self.rotary_emb = rotary_emb
        self.cache = DynamicCache()

    @torch.no_grad()
    def run_block(self, hidden_states: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        position_ids = cache_position.unsqueeze(0)
        past_len = self.cache.get_seq_length()
        kv_len = past_len + hidden_states.shape[1]
        attn_mask = build_causal_mask(cache_position, kv_len, hidden_states.dtype, hidden_states.device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        return hidden_states


class HeadBlock:
    """Last block: hidden_states -> logits (final norm + lm_head)."""
    def __init__(self, norm, lm_head):
        self.norm = norm
        self.lm_head = lm_head

    @torch.no_grad()
    def run_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))


def split_into_blocks(model, boundaries: list[int]):
    """Split a loaded model into (EmbedBlock, [DecoderBlock...], HeadBlock).

    boundaries: decoder layer boundaries, e.g. [0, 12, 24] -> two slabs [0:12),[12:24).

    WARNING: mutates layer.self_attn.layer_idx to local indices. Capture every
    reference from the whole model BEFORE calling this function.
    """
    inner = model.model
    embed = EmbedBlock(inner.embed_tokens)
    head = HeadBlock(inner.norm, model.lm_head)

    decoders = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        layers = inner.layers[lo:hi]
        for local_idx, layer in enumerate(layers):
            layer.self_attn.layer_idx = local_idx   # remap to the block's local index
        decoders.append(DecoderBlock(layers, inner.rotary_emb))

    return embed, decoders, head
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_blocks.py -m slow -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add axyn/model/blocks.py tests/test_blocks.py
git commit -m "feat(model): BlockRunner (EMBED/DECODER/HEAD) + split_into_blocks"
```

---

## Task 6: Golden test — distributed pipeline vs whole model equivalence

**Files:**
- Create: `axyn/model/generate.py`
- Test: `tests/test_golden.py`

- [ ] **Step 1: Write the failing test**

`tests/test_golden.py`:
```python
import pytest
import torch
from axyn.model.generate import reference_generate, pipeline_generate
from axyn.model.blocks import split_into_blocks


@pytest.mark.slow
def test_pipeline_matches_reference_generation(full_model):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids

    # 1) Reference: capture BEFORE the split (split mutates the layer_idx)
    reference = reference_generate(model, ids, max_new_tokens=8)

    # 2) In-process distributed pipeline
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    pipeline = pipeline_generate(embed, decoders, head, ids, max_new_tokens=8)

    assert pipeline == reference, f"divergence: {pipeline} != {reference}"
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_golden.py -m slow -v`
Expected: FAIL with `ImportError` on `axyn.model.generate`.

- [ ] **Step 3: Implement `generate.py`**

```python
import torch
from transformers import DynamicCache


@torch.no_grad()
def reference_generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    """Greedy decode with the whole model (reference). Deterministic."""
    cache = DynamicCache()
    seq_len = input_ids.shape[1]
    cur = input_ids
    cache_position = torch.arange(seq_len)
    generated: list[int] = []
    for step in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                    cache_position=cache_position)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        cur = next_id
        cache = out.past_key_values
        cache_position = torch.tensor([seq_len + step])
    return generated


@torch.no_grad()
def pipeline_generate(embed, decoders, head, input_ids: torch.Tensor,
                      max_new_tokens: int) -> list[int]:
    """Greedy decode through the split blocks, with a per-block KV-cache
    (session affinity). Must reproduce reference_generate."""
    seq_len = input_ids.shape[1]
    cur_ids = input_ids
    cache_position = torch.arange(seq_len)
    generated: list[int] = []
    for step in range(max_new_tokens):
        h = embed.run_block(cur_ids)
        for d in decoders:
            h = d.run_block(h, cache_position)
        logits = head.run_block(h)[:, -1, :]
        next_id = logits.argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        cur_ids = next_id
        cache_position = torch.tensor([seq_len + step])
    return generated
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_golden.py -m slow -v`
Expected: PASS. If it fails with divergent tokens, it is the decoder layer signature mismatch described in the version note: align `DecoderBlock.run_block` with the `forward` of the installed model.

- [ ] **Step 5: Commit**

```bash
git add axyn/model/generate.py tests/test_golden.py
git commit -m "feat(model): golden test distributed pipeline vs whole model equivalence"
```

---

## Task 7: Capstone — KV-cache survives a mid-generation round-trip

> De-risks directly the ADR's **risk #1** ("KV-cache correctness across hops and restart/failover"): halfway through generation we serialize the per-block caches to bytes and reload them (simulating a holder handoff/restart), then continue and require the **same** sequence.

**Files:**
- Modify: `axyn/model/blocks.py` (add cache get/set)
- Test: `tests/test_resilience.py`

- [ ] **Step 1: Write the failing test**

`tests/test_resilience.py`:
```python
import pytest
import torch
from axyn.model.generate import reference_generate
from axyn.model.blocks import split_into_blocks
from axyn.model.cache import cache_to_bytes, cache_from_bytes


@pytest.mark.slow
def test_generation_survives_cache_serialization_midway(full_model):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=8)

    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    seq_len = ids.shape[1]
    cache_position = torch.arange(seq_len)
    cur_ids = ids
    generated = []
    for step in range(8):
        h = embed.run_block(cur_ids)
        for d in decoders:
            h = d.run_block(h, cache_position)
        next_id = head.run_block(h)[:, -1, :].argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        cur_ids = next_id
        cache_position = torch.tensor([seq_len + step])

        if step == 3:  # simulate handoff/restart: serialize and reload each cache
            for d in decoders:
                d.set_cache(cache_from_bytes(cache_to_bytes(d.get_cache())))

    assert generated == reference
```

- [ ] **Step 2: Run the test to see it fail**

Run: `pytest tests/test_resilience.py -m slow -v`
Expected: FAIL with `AttributeError` on `DecoderBlock.set_cache`/`get_cache`.

- [ ] **Step 3: Add cache get/set in `blocks.py`**

In `DecoderBlock`, after `run_block`, add:
```python
    def get_cache(self):
        return self.cache

    def set_cache(self, cache):
        self.cache = cache
```

- [ ] **Step 4: Run the test to see it pass**

Run: `pytest tests/test_resilience.py -m slow -v`
Expected: PASS — generation is identical even after the cache's byte round-trip.

- [ ] **Step 5: Commit**

```bash
git add axyn/model/blocks.py tests/test_resilience.py
git commit -m "test(model): KV-cache survives mid-generation serialization"
```

---

## Task 8: Full suite & ROADMAP update

- [ ] **Step 1: Run the whole suite**

Run: `pytest -v` (unit) then `pytest -m slow -v` (model).
Expected: all tests PASS.

- [ ] **Step 2: Update the ROADMAP**

In `docs/ROADMAP.md`, under "Phase 1 — PoC Implementation", check off the first item and note the completion of the Part 1 foundation block (golden test + cache/payload serialization green).

- [ ] **Step 3: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: Part 1 foundation completed (golden test green)"
```

---

## Self-Review (performed by the plan author)

**Spec coverage (PRD Part 1):**
- §3 `run_block` (EMBED/DECODER/HEAD) → Task 5 ✓
- §3 KV-cache serializable per `(job_id, stage)` → Task 3 + Task 7 ✓
- §3 safetensors payload → Task 4 ✓
- §6 golden_test equivalence vs whole model → Task 6 ✓
- §6 lossless cache round-trip / generation resumed from a persisted cache → Task 7 ✓
- §2 partial-loading via `init_empty_weights`/`load_checkpoint_in_model` → **explicitly deferred** (later plan, goes with the wire format). Intentional gap, documented in the Scope.

**Placeholder scan:** no TODO/TBD; every step has complete code.

**Type consistency:** `split_into_blocks(model, boundaries)` → `(EmbedBlock, list[DecoderBlock], HeadBlock)` used consistently in Task 5/6/7; `cache_to_bytes`/`cache_from_bytes`, `HopPayload.to_bytes`/`from_bytes`, `run_block`, `get_cache`/`set_cache` consistent across the tasks.
