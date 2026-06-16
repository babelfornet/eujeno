# Part 1 — Peer Node & Layer Execution (foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provare in un singolo processo che un modello LLM può essere splittato in blocchi di layer ed eseguito hop-by-hop riproducendo **esattamente** la generazione del modello intero, con KV-cache per-blocco serializzabile che sopravvive a un round-trip su byte (simulazione di handoff/restart).

**Architecture:** Si carica il modello completo una volta (riferimento), poi `split_into_blocks` lo divide in un blocco EMBED, N blocchi DECODER (slab contigui di layer con `layer_idx` rimappato a indici locali + KV-cache locale `DynamicCache`) e un blocco HEAD. `pipeline_generate` esegue una generazione greedy attraverso i blocchi e deve produrre la **stessa** sequenza di token di `reference_generate`. Due primitivi puri di serializzazione (KV-cache e payload di hop, entrambi via safetensors) preparano il transport di rete della Parte 3.

**Tech Stack:** Python 3.11 · PyTorch (fp32, CPU) · Hugging Face `transformers==4.46.3` (API `DynamicCache` stabile) · `safetensors` · `accelerate` · `huggingface_hub` · `pytest`. Modello di test: `Qwen/Qwen2.5-0.5B-Instruct` (ungated, 24 layer, hidden 896).

**Scope (questo piano):** build-order step 1, 2, 4 dell'[ADR-0001](../decisions/ADR-0001-implementation-forks.md) + primitivi di serializzazione cache/payload ([PRD Parte 1](../prd/part-1-peer-node.md) §3).
**Fuori scope (piani successivi):** partial-loading reale via `init_empty_weights`/`load_checkpoint_in_model` (concern di memoria, va col wire format), transport FastAPI, DHT/routing, persistenza SQLite. Qui i blocchi vivono nello stesso processo.

> **Nota di versione:** il codice mirror la `Model.forward` di `transformers==4.46.3` (firma del decoder layer con `position_embeddings`). Se l'engineer usa una versione diversa e il golden test fallisce, allineare la chiamata del layer alla `forward` del modello installato — il golden test (Task 6) è la rete di sicurezza che rende questo deterministico. Vedi [ADR-0001](../decisions/ADR-0001-implementation-forks.md) Q2.

---

## File Structure

```
pyproject.toml                 # progetto + dipendenze pinnate + config pytest
synapse/
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
  conftest.py                  # fixture full_model (session-scoped)
  test_loader.py
  test_masking.py
  test_cache.py
  test_payload.py
  test_blocks.py
  test_golden.py               # IL golden test
  test_resilience.py           # round-trip cache mid-generazione
```

Responsabilità per file (una sola per file): `loader` carica · `masking` costruisce maschere · `cache`/`payload` serializzano · `blocks` esegue layer · `generate` orchestra la generazione greedy.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `synapse/__init__.py`, `synapse/model/__init__.py`
- Create: `synapse/config.py`

- [ ] **Step 1: Crea `pyproject.toml`**

```toml
[project]
name = "synapse"
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
markers = ["slow: scarica/esegue il modello (lento)"]
addopts = "-q"

[tool.setuptools.packages.find]
include = ["synapse*"]
```

- [ ] **Step 2: Crea i package init e `config.py`**

`synapse/__init__.py`: file vuoto.
`synapse/model/__init__.py`: file vuoto.

`synapse/config.py`:
```python
import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float32   # fp32 su CPU per determinismo (vedi ADR-0001 Fork D)
DEVICE = "cpu"
```

- [ ] **Step 3: Crea l'ambiente e installa**

Run:
```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```
Expected: installazione completata senza errori; `python -c "import transformers, torch, safetensors, accelerate; print('ok')"` stampa `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml synapse/__init__.py synapse/model/__init__.py synapse/config.py
git commit -m "chore: scaffolding pacchetto synapse + dipendenze pinnate"
```

---

## Task 1: Model loader

**Files:**
- Create: `synapse/model/loader.py`
- Test: `tests/conftest.py`, `tests/test_loader.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/conftest.py`:
```python
import pytest
import torch
from synapse.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from synapse.model.loader import load_full_model

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
from synapse.model.loader import model_dims

@pytest.mark.slow
def test_loads_with_expected_dims(full_model):
    model, tokenizer = full_model
    dims = model_dims(model)
    assert dims["num_layers"] == 24
    assert dims["hidden_size"] == 896
    assert tokenizer is not None
```

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_loader.py -m slow -v`
Expected: FAIL con `ModuleNotFoundError`/`ImportError` su `synapse.model.loader`.

- [ ] **Step 3: Implementa `loader.py`**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_full_model(model_id: str, dtype: torch.dtype, device: str):
    """Carica modello completo + tokenizer. Usato come riferimento e come
    sorgente da cui estrarre i blocchi (Task 5)."""
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

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_loader.py -m slow -v`
Expected: PASS (la prima esecuzione scarica ~1GB del modello).

- [ ] **Step 5: Commit**

```bash
git add synapse/model/loader.py tests/conftest.py tests/test_loader.py
git commit -m "feat(model): loader del modello completo + introspezione dimensioni"
```

---

## Task 2: Causal mask builder

**Files:**
- Create: `synapse/model/masking.py`
- Test: `tests/test_masking.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_masking.py`:
```python
import torch
from synapse.model.masking import build_causal_mask


def test_prefill_mask_is_lower_triangular():
    cache_position = torch.arange(3)        # prefill di 3 token, kv_len=3
    mask = build_causal_mask(cache_position, kv_len=3, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 3, 3)
    neg = torch.finfo(torch.float32).min
    # query 0 vede solo key 0
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] == neg
    assert mask[0, 0, 0, 2] == neg
    # query 2 vede tutte
    assert torch.all(mask[0, 0, 2, :] == 0.0)


def test_decode_step_attends_to_all_past():
    cache_position = torch.tensor([5])      # 1 query token in posizione 5, kv_len=6
    mask = build_causal_mask(cache_position, kv_len=6, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 1, 6)
    assert torch.all(mask[0, 0, 0, :] == 0.0)   # attende a tutte le 6 posizioni passate
```

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_masking.py -v`
Expected: FAIL con `ImportError` su `synapse.model.masking`.

- [ ] **Step 3: Implementa `masking.py`**

```python
import torch


def build_causal_mask(cache_position: torch.Tensor, kv_len: int,
                      dtype: torch.dtype, device: str) -> torch.Tensor:
    """Maschera additiva causale 4D [1,1,q_len,kv_len] per batch=1 senza padding.
    cache_position contiene le posizioni assolute dei token di query."""
    q_len = cache_position.shape[0]
    key_pos = torch.arange(kv_len, device=device)
    allowed = key_pos[None, :] <= cache_position[:, None].to(device)   # [q_len, kv_len] bool
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]
```

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_masking.py -v`
Expected: PASS (2 test).

- [ ] **Step 5: Commit**

```bash
git add synapse/model/masking.py tests/test_masking.py
git commit -m "feat(model): builder maschera causale per esecuzione blocchi"
```

---

## Task 3: KV-cache serialization (primitivo condiviso #2/3)

**Files:**
- Create: `synapse/model/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_cache.py`:
```python
import torch
from transformers import DynamicCache
from synapse.model.cache import cache_to_bytes, cache_from_bytes


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

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL con `ImportError` su `synapse.model.cache`.

- [ ] **Step 3: Implementa `cache.py`**

```python
import safetensors.torch
from transformers import DynamicCache


def cache_to_bytes(cache: DynamicCache) -> bytes:
    """Serializza una DynamicCache (per-blocco) in bytes safetensors."""
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

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_cache.py -v`
Expected: PASS (2 test).

- [ ] **Step 5: Commit**

```bash
git add synapse/model/cache.py tests/test_cache.py
git commit -m "feat(model): serializzazione KV-cache via safetensors (round-trip)"
```

---

## Task 4: Hop payload serialization (primitivo condiviso #1 sul filo)

**Files:**
- Create: `synapse/model/payload.py`
- Test: `tests/test_payload.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_payload.py`:
```python
import torch
from synapse.model.payload import HopPayload


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

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_payload.py -v`
Expected: FAIL con `ImportError` su `synapse.model.payload`.

- [ ] **Step 3: Implementa `payload.py`**

```python
import json
from dataclasses import dataclass

import torch
import safetensors.torch


@dataclass
class HopPayload:
    """Payload di un hop sul filo (Parte 1 §3). La KV-cache NON viaggia qui:
    resta locale all'holder (session affinity, Parte 3)."""
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

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse/model/payload.py tests/test_payload.py
git commit -m "feat(model): serializzazione HopPayload via safetensors (round-trip)"
```

---

## Task 5: BlockRunner + split_into_blocks

**Files:**
- Create: `synapse/model/blocks.py`
- Test: `tests/test_blocks.py`

> **Nota di design:** `split_into_blocks` **muta** `layer.self_attn.layer_idx` rimappandolo a indici locali (0-based per blocco), così ogni blocco usa una `DynamicCache` locale per i soli suoi layer (forward-compatibile col modello distribuito). Per questo i test e il golden test (Task 6) catturano sempre il riferimento dal modello intero **prima** di chiamare `split_into_blocks`.

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_blocks.py`:
```python
import pytest
import torch
from synapse.model.blocks import split_into_blocks


@pytest.mark.slow
def test_embed_block_matches_model_embedding(full_model):
    model, tokenizer = full_model
    ids = tokenizer("Ciao mondo", return_tensors="pt").input_ids
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

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_blocks.py -m slow -v`
Expected: FAIL con `ImportError` su `synapse.model.blocks`.

- [ ] **Step 3: Implementa `blocks.py`**

```python
import torch
from transformers import DynamicCache

from .masking import build_causal_mask


class EmbedBlock:
    """Primo blocco: input_ids -> hidden_states."""
    def __init__(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @torch.no_grad()
    def run_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class DecoderBlock:
    """Slab contiguo di layer [lo, hi). Mantiene una KV-cache LOCALE per i
    soli suoi layer (indici rimappati 0-based)."""
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
    """Ultimo blocco: hidden_states -> logits (final norm + lm_head)."""
    def __init__(self, norm, lm_head):
        self.norm = norm
        self.lm_head = lm_head

    @torch.no_grad()
    def run_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))


def split_into_blocks(model, boundaries: list[int]):
    """Divide un modello caricato in (EmbedBlock, [DecoderBlock...], HeadBlock).

    boundaries: confini dei layer decoder, es. [0, 12, 24] -> due slab [0:12),[12:24).

    ATTENZIONE: muta layer.self_attn.layer_idx a indici locali. Catturare ogni
    riferimento dal modello intero PRIMA di chiamare questa funzione.
    """
    inner = model.model
    embed = EmbedBlock(inner.embed_tokens)
    head = HeadBlock(inner.norm, model.lm_head)

    decoders = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        layers = inner.layers[lo:hi]
        for local_idx, layer in enumerate(layers):
            layer.self_attn.layer_idx = local_idx   # rimappa a indice locale del blocco
        decoders.append(DecoderBlock(layers, inner.rotary_emb))

    return embed, decoders, head
```

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_blocks.py -m slow -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add synapse/model/blocks.py tests/test_blocks.py
git commit -m "feat(model): BlockRunner (EMBED/DECODER/HEAD) + split_into_blocks"
```

---

## Task 6: Golden test — equivalenza pipeline distribuita vs modello intero

**Files:**
- Create: `synapse/model/generate.py`
- Test: `tests/test_golden.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_golden.py`:
```python
import pytest
import torch
from synapse.model.generate import reference_generate, pipeline_generate
from synapse.model.blocks import split_into_blocks


@pytest.mark.slow
def test_pipeline_matches_reference_generation(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids

    # 1) Riferimento: cattura PRIMA dello split (split muta i layer_idx)
    reference = reference_generate(model, ids, max_new_tokens=8)

    # 2) Pipeline distribuita in-process
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    pipeline = pipeline_generate(embed, decoders, head, ids, max_new_tokens=8)

    assert pipeline == reference, f"divergenza: {pipeline} != {reference}"
```

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_golden.py -m slow -v`
Expected: FAIL con `ImportError` su `synapse.model.generate`.

- [ ] **Step 3: Implementa `generate.py`**

```python
import torch
from transformers import DynamicCache


@torch.no_grad()
def reference_generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    """Greedy decode col modello intero (riferimento). Deterministico."""
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
    """Greedy decode attraverso i blocchi splittati, con KV-cache per-blocco
    (session affinity). Deve riprodurre reference_generate."""
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

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_golden.py -m slow -v`
Expected: PASS. Se fallisce con token divergenti, è il mismatch di firma del decoder layer descritto nella nota di versione: allineare `DecoderBlock.run_block` alla `forward` del modello installato.

- [ ] **Step 5: Commit**

```bash
git add synapse/model/generate.py tests/test_golden.py
git commit -m "feat(model): golden test equivalenza pipeline distribuita vs modello intero"
```

---

## Task 7: Capstone — KV-cache sopravvive a round-trip mid-generazione

> De-risca direttamente il **rischio #1** dell'ADR ("correttezza KV-cache tra hop e restart/failover"): a metà generazione serializziamo le cache per-blocco su byte e le ricarichiamo (simulando handoff/restart di un holder), poi continuiamo e pretendiamo la **stessa** sequenza.

**Files:**
- Modify: `synapse/model/blocks.py` (aggiungi get/set della cache)
- Test: `tests/test_resilience.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_resilience.py`:
```python
import pytest
import torch
from synapse.model.generate import reference_generate
from synapse.model.blocks import split_into_blocks
from synapse.model.cache import cache_to_bytes, cache_from_bytes


@pytest.mark.slow
def test_generation_survives_cache_serialization_midway(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
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

        if step == 3:  # simula handoff/restart: serializza e ricarica ogni cache
            for d in decoders:
                d.set_cache(cache_from_bytes(cache_to_bytes(d.get_cache())))

    assert generated == reference
```

- [ ] **Step 2: Esegui il test per vederlo fallire**

Run: `pytest tests/test_resilience.py -m slow -v`
Expected: FAIL con `AttributeError` su `DecoderBlock.set_cache`/`get_cache`.

- [ ] **Step 3: Aggiungi get/set della cache in `blocks.py`**

In `DecoderBlock`, dopo `run_block`, aggiungi:
```python
    def get_cache(self):
        return self.cache

    def set_cache(self, cache):
        self.cache = cache
```

- [ ] **Step 4: Esegui il test per vederlo passare**

Run: `pytest tests/test_resilience.py -m slow -v`
Expected: PASS — la generazione è identica anche dopo il round-trip su byte della cache.

- [ ] **Step 5: Commit**

```bash
git add synapse/model/blocks.py tests/test_resilience.py
git commit -m "test(model): KV-cache sopravvive a serializzazione mid-generazione"
```

---

## Task 8: Suite completa & aggiornamento ROADMAP

- [ ] **Step 1: Esegui l'intera suite**

Run: `pytest -v` (unit) poi `pytest -m slow -v` (modello).
Expected: tutti i test PASS.

- [ ] **Step 2: Aggiorna la ROADMAP**

In `docs/ROADMAP.md`, sotto "Fase 1 — Implementazione PoC", spunta il primo punto e annota il completamento del blocco fondante della Parte 1 (golden test + serializzazione cache/payload verdi).

- [ ] **Step 3: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: Parte 1 foundation completata (golden test verde)"
```

---

## Self-Review (eseguito dall'autore del piano)

**Spec coverage (PRD Parte 1):**
- §3 `run_block` (EMBED/DECODER/HEAD) → Task 5 ✓
- §3 KV-cache serializzabile per `(job_id, stage)` → Task 3 + Task 7 ✓
- §3 payload safetensors → Task 4 ✓
- §6 golden_test equivalenza vs modello intero → Task 6 ✓
- §6 round-trip cache senza perdita / generazione ripresa da cache persistita → Task 7 ✓
- §2 partial-loading via `init_empty_weights`/`load_checkpoint_in_model` → **deferred dichiarato** (piano successivo, va col wire format). Gap intenzionale e documentato nello Scope.

**Placeholder scan:** nessun TODO/TBD; ogni step ha codice completo.

**Type consistency:** `split_into_blocks(model, boundaries)` → `(EmbedBlock, list[DecoderBlock], HeadBlock)` usato coerentemente in Task 5/6/7; `cache_to_bytes`/`cache_from_bytes`, `HopPayload.to_bytes`/`from_bytes`, `run_block`, `get_cache`/`set_cache` coerenti tra i task.
