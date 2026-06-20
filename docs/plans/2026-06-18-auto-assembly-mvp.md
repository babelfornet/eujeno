# Auto-assembly MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Nodes self-assign the layers to cover by reading the coverage gaps from the registry + their own RAM capacity, without manual assignment (`serve --auto`).

**Architecture:** Three pure functions (capacity, gaps, decision) + the wiring in `serve --auto`: at startup the node probes its RAM, queries a seed `/registry`, picks the neediest range that fits, then loads ONLY those layers and serves. Capacity is advertised (additively) in the gossip record. The replica `target` is parametric (≥2 ⇒ redundancy at startup). Implements slices 1-3 of [ADR-0003](../decisions/ADR-0003-capacity-aware-allocation.md). Runtime failover-reload + reward ledger = next plan.

**Tech Stack:** Python · Typer · the existing `eujeno/net/{discovery,server}.py`, `eujeno/cli.py`, `model_config_dims`, `parse_stages`, `parse_dtype`. Optional dependency `psutil` (stdlib fallback).

---

## File Structure
```
eujeno/net/capacity.py     # NEW: fit_layers (extracted from cli.fit) + probe_capacity
eujeno/net/discovery.py    # MOD: coverage_gaps()
eujeno/net/allocator.py    # NEW: choose_stages() (pure decision)
eujeno/net/server.py       # MOD: capacity in the own_stages record
eujeno/cli.py              # MOD: fit uses fit_layers; serve --auto/--ram/--reserve
tests/test_capacity.py      # NEW
tests/test_coverage_gaps.py # NEW
tests/test_allocator.py     # NEW
tests/test_serve_auto.py    # NEW (unit test of the selection path, without starting the server)
```

---

## Task 1: `net/capacity.py` — fit_layers (estratto) + probe_capacity

**Files:** Create `eujeno/net/capacity.py`, `tests/test_capacity.py`; Modify `eujeno/cli.py` (fit uses fit_layers).

- [ ] **Step 1: test `tests/test_capacity.py`**
```python
from eujeno.net.capacity import fit_layers, probe_capacity

DIMS = {"num_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
        "num_key_value_heads": 4, "intermediate_size": 18944, "vocab_size": 152064}


def test_fit_layers_bf16_more_than_fp32():
    bf16 = fit_layers(DIMS, 2, 8.0)["max_decoder_layers"]
    fp32 = fit_layers(DIMS, 4, 8.0)["max_decoder_layers"]
    assert bf16 >= 2 * fp32 - 1
    assert fp32 > 0


def test_fit_layers_caps_at_num_layers():
    r = fit_layers(DIMS, 2, 999.0)
    assert r["max_decoder_layers"] == 28
    assert r["fits_whole_model"] is True


def test_probe_capacity_shape():
    c = probe_capacity()
    assert "cpu_count" in c and c["cpu_count"] >= 1
    assert "ram_free_gb" in c and "ram_total_gb" in c
```

- [ ] **Step 2: run FAIL** — `.venv/bin/python -m pytest tests/test_capacity.py -v` → ImportError.

- [ ] **Step 3: create `eujeno/net/capacity.py`**
```python
"""Estimate a node's capacity: how many layers it can hold given the RAM, plus a resource probe."""
import os

_GB = 1024 ** 3


def fit_layers(dims: dict, bytes_per_param: int, ram_gb: float, reserve: float = 0.2) -> dict:
    """Given the model (dims), the size in bytes of a parameter and the available
    RAM in GB, estimate how many decoder layers the node can hold."""
    hidden = dims["hidden_size"]
    nl = dims["num_layers"]
    heads = dims["num_attention_heads"]
    kv = dims.get("num_key_value_heads") or heads
    inter = dims.get("intermediate_size") or (4 * hidden)
    vocab = dims.get("vocab_size") or 0
    kv_dim = hidden * kv / heads
    params_layer = 2 * hidden ** 2 + 2 * hidden * kv_dim + 3 * hidden * inter
    ram_layer = params_layer * bytes_per_param
    ram_embed_head = vocab * hidden * bytes_per_param
    usable = ram_gb * (1 - reserve) * _GB
    max_layers = max(0, int(usable // ram_layer)) if ram_layer > 0 else 0
    return {
        "ram_per_layer_gb": round(ram_layer / _GB, 3),
        "ram_embed_head_gb": round(ram_embed_head / _GB, 3),
        "max_decoder_layers": min(max_layers, nl),
        "fits_whole_model": (nl * ram_layer + ram_embed_head) <= usable,
    }


def probe_capacity() -> dict:
    """Total/free RAM (GB) and CPU count. Uses psutil if present, otherwise stdlib."""
    cpu = os.cpu_count() or 1
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"ram_total_gb": round(vm.total / _GB, 2),
                "ram_free_gb": round(vm.available / _GB, 2), "cpu_count": cpu}
    except Exception:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            free = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            return {"ram_total_gb": round(total / _GB, 2),
                    "ram_free_gb": round(free / _GB, 2), "cpu_count": cpu}
        except (ValueError, OSError, AttributeError):
            return {"ram_total_gb": None, "ram_free_gb": None, "cpu_count": cpu}
```

- [ ] **Step 4: refactor `cli.py::fit`** to use `fit_layers` (do NOT change the emitted `data` fields). Replace the computation block inside `fit` (from `bytes_per = torch.finfo(...)` up to the `data` dict) with:
```python
    import torch
    from eujeno.net.capacity import fit_layers
    bytes_per = torch.finfo(_dt).bits // 8
    d = model_config_dims(model_id)
    nl = d["num_layers"]
    r = fit_layers(d, bytes_per, ram, reserve)
    k = r["max_decoder_layers"]
    fits_whole = r["fits_whole_model"]
    if fits_whole:
        suggested = f"embed,decoder:0-{nl},head"
    elif k > 0:
        suggested = f"decoder:0-{k}"
    else:
        suggested = ""
    data = {
        "model": model_id, "dtype": dtype, "num_layers": nl, "hidden_size": d["hidden_size"],
        "ram_gb": ram, "reserve": reserve,
        "ram_per_layer_gb": r["ram_per_layer_gb"],
        "ram_embed_head_gb": r["ram_embed_head_gb"],
        "max_decoder_layers": k, "fits_whole_model": fits_whole,
        "suggested_stages": suggested,
    }
```
(Leave the rest of the `fit` signature, the dtype validation and the construction of `human`/`_emit_ok` unchanged.)

- [ ] **Step 5: run PASS** — `.venv/bin/python -m pytest tests/test_capacity.py tests/test_cli_fit.py -v` → all green (the refactor does not break the existing `fit` tests).

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/capacity.py eujeno/cli.py tests/test_capacity.py && git commit -m "feat(net): capacity.fit_layers + probe_capacity (fit CLI refactored)"
```

---

## Task 2: `discovery.coverage_gaps()`

**Files:** Modify `eujeno/net/discovery.py`; Create `tests/test_coverage_gaps.py`.

- [ ] **Step 1: test `tests/test_coverage_gaps.py`**
```python
from eujeno.net.discovery import coverage_gaps

A = {"embed": True, "head": False, "decoders": ["0-12"]}
B = {"embed": False, "head": True, "decoders": ["12-24"]}


def test_full_coverage_no_gaps():
    g = coverage_gaps({"a": A, "b": B}, 24, target=1)
    assert g["decoder_gaps"] == []
    assert g["embed_replicas"] == 1 and g["head_replicas"] == 1


def test_missing_middle_range():
    g = coverage_gaps({"a": A}, 24, target=1)
    assert g["decoder_gaps"] == [{"lo": 12, "hi": 24, "replicas": 0}]


def test_under_replicated_with_target_2():
    g = coverage_gaps({"a": A, "b": B}, 24, target=2)
    # everything covered only once -> the whole range is under-replicated
    assert g["decoder_gaps"] == [{"lo": 0, "hi": 24, "replicas": 1}]
    assert g["embed_replicas"] == 1 and g["head_replicas"] == 1
```

- [ ] **Step 2: run FAIL** — `.venv/bin/python -m pytest tests/test_coverage_gaps.py -v` → ImportError.

- [ ] **Step 3: append a `eujeno/net/discovery.py`**
```python
def coverage_gaps(stages_by_url: dict, num_layers: int, target: int = 1) -> dict:
    """Decoder ranges with replicas < target (uncovered or under-replicated), plus the
    replica count of embed/head. `stages_by_url`: {url: {'embed','head','decoders'}}."""
    cover = [0] * num_layers
    for s in stages_by_url.values():
        for bk in s.get("decoders", []):
            lo, hi = (int(x) for x in bk.split("-"))
            for i in range(max(0, lo), min(hi, num_layers)):
                cover[i] += 1
    gaps = []
    i = 0
    while i < num_layers:
        if cover[i] < target:
            j = i
            while j < num_layers and cover[j] < target:
                j += 1
            gaps.append({"lo": i, "hi": j, "replicas": min(cover[i:j])})
            i = j
        else:
            i += 1
    return {
        "decoder_gaps": gaps,
        "embed_replicas": sum(1 for s in stages_by_url.values() if s.get("embed")),
        "head_replicas": sum(1 for s in stages_by_url.values() if s.get("head")),
        "target": target,
    }
```

- [ ] **Step 4: run PASS** — `.venv/bin/python -m pytest tests/test_coverage_gaps.py -v` → 3 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/discovery.py tests/test_coverage_gaps.py && git commit -m "feat(net): coverage_gaps (uncovered/under-replicated ranges)"
```

---

## Task 3: `net/allocator.py` — choose_stages() (pure decision)

**Files:** Create `eujeno/net/allocator.py`, `tests/test_allocator.py`.

- [ ] **Step 1: test `tests/test_allocator.py`**
```python
from eujeno.net.allocator import choose_stages


def gaps(decoder_gaps, e=0, h=0, target=1):
    return {"decoder_gaps": decoder_gaps, "embed_replicas": e, "head_replicas": h, "target": target}


def test_takes_neediest_decoder_gap_capped_by_capacity():
    g = gaps([{"lo": 12, "hi": 24, "replicas": 0}], e=1, h=1)
    # can hold only 5 layers
    assert choose_stages(g, max_decoder_layers=5, num_layers=24, take_embed_head=False) == "decoder:12-17"


def test_claims_embed_head_when_uncovered_and_capable():
    g = gaps([{"lo": 0, "hi": 24, "replicas": 0}], e=0, h=0)
    s = choose_stages(g, max_decoder_layers=99, num_layers=24, take_embed_head=True)
    assert s == "embed,decoder:0-24,head"


def test_prefers_lower_replication_first():
    g = gaps([{"lo": 0, "hi": 6, "replicas": 1}, {"lo": 6, "hi": 12, "replicas": 0}], e=1, h=1, target=2)
    assert choose_stages(g, max_decoder_layers=99, num_layers=12, take_embed_head=False) == "decoder:6-12"


def test_no_gaps_returns_empty():
    assert choose_stages(gaps([], e=1, h=1), max_decoder_layers=10, num_layers=24, take_embed_head=False) == ""
```

- [ ] **Step 2: run FAIL** — `.venv/bin/python -m pytest tests/test_allocator.py -v` → ImportError.

- [ ] **Step 3: create `eujeno/net/allocator.py`**
```python
"""Self-assignment decision: given the gap picture (coverage_gaps) and the
node's capacity, choose the stage spec to claim. Pure function."""


def choose_stages(gaps: dict, max_decoder_layers: int, num_layers: int,
                  take_embed_head: bool) -> str:
    """Returns a stage spec for parse_stages (e.g. 'embed,decoder:12-17,head'),
    or '' if there is nothing useful/possible to claim."""
    target = gaps.get("target", 1)
    parts = []
    if take_embed_head and gaps.get("embed_replicas", 0) < target:
        parts.append("embed")
    decoder_gaps = sorted(gaps.get("decoder_gaps", []),
                          key=lambda g: (g["replicas"], -(g["hi"] - g["lo"])))
    if decoder_gaps and max_decoder_layers > 0:
        g = decoder_gaps[0]
        hi = min(g["hi"], g["lo"] + max_decoder_layers)
        parts.append(f"decoder:{g['lo']}-{hi}")
    if take_embed_head and gaps.get("head_replicas", 0) < target:
        parts.append("head")
    return ",".join(parts)
```

- [ ] **Step 4: run PASS** — `.venv/bin/python -m pytest tests/test_allocator.py -v` → 4 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/allocator.py tests/test_allocator.py && git commit -m "feat(net): allocator.choose_stages (self-assignment decision)"
```

---

## Task 4: `serve --auto` + capacity in the record

**Files:** Modify `eujeno/net/server.py` (capacity in the record), `eujeno/cli.py` (serve --auto). Create `tests/test_serve_auto.py`.

- [ ] **Step 1: test `tests/test_serve_auto.py`** — tests the pure planning function `plan_auto_stages` (no server started).
```python
from eujeno.cli import plan_auto_stages

DIMS = {"num_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
        "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936}


def test_plan_first_node_claims_whole_when_capable():
    # empty registry, roomy node -> takes everything
    s = plan_auto_stages(DIMS, bytes_per=4, ram_gb=64.0, reserve=0.2,
                         stages_by_url={}, target=1)
    assert s == "embed,decoder:0-24,head"


def test_plan_second_node_fills_remaining_gap():
    # one node covers embed+0-12; the second, small one takes the middle gap
    existing = {"a": {"embed": True, "head": False, "decoders": ["0-12"]}}
    s = plan_auto_stages(DIMS, bytes_per=4, ram_gb=0.6, reserve=0.2,
                         stages_by_url=existing, target=1)
    assert s.startswith("decoder:12-")
```

- [ ] **Step 2: run FAIL** — `.venv/bin/python -m pytest tests/test_serve_auto.py -v` → ImportError (`plan_auto_stages`).

- [ ] **Step 3a: in `eujeno/cli.py` add the pure function `plan_auto_stages`** (next to the other helpers, NOT inside a command):
```python
def plan_auto_stages(dims: dict, bytes_per: int, ram_gb: float, reserve: float,
                     stages_by_url: dict, target: int) -> str:
    """Decide the stage spec to claim by combining capacity (fit) and gaps."""
    from eujeno.net.capacity import fit_layers
    from eujeno.net.discovery import coverage_gaps
    from eujeno.net.allocator import choose_stages
    nl = dims["num_layers"]
    fit = fit_layers(dims, bytes_per, ram_gb, reserve)
    gaps = coverage_gaps(stages_by_url, nl, target=target)
    # a "roomy" node (one that holds the whole model) also offers itself for embed/head
    take_eh = fit["fits_whole_model"] or fit["max_decoder_layers"] >= nl
    return choose_stages(gaps, fit["max_decoder_layers"], nl, take_embed_head=take_eh)
```

- [ ] **Step 3b: in `eujeno/cli.py` make `--stages` optional and add `--auto/--ram/--reserve` to `serve`.** Change the `serve` signature:
  - `stages: str = typer.Option(None, "--stages", ...)` (was `...`, required).
  - add:
    ```python
    auto: bool = typer.Option(False, "--auto", help="Auto-assign layers from registry gaps + RAM capacity"),
    ram: float = typer.Option(None, "--ram", help="RAM to use for auto-assignment, GB (default: detected)"),
    reserve: float = typer.Option(0.2, "--reserve", help="Reserved RAM fraction (auto)"),
    target: int = typer.Option(1, "--target", help="Desired replicas per range (auto; 2 = redundancy)"),
    ```
  Right after `import uvicorn` (before `parse_stages`), insert the auto branch:
```python
    if auto:
        import torch, httpx
        from eujeno.net.capacity import probe_capacity, fit_layers
        from eujeno.config import parse_dtype as _pdt
        _bp = torch.finfo(_pdt(dtype)).bits // 8
        dims = model_config_dims(model_id)
        ram_gb = ram if ram is not None else (probe_capacity().get("ram_free_gb") or 4.0)
        learned = {}
        for seed in ([p.strip() for p in peers.split(",")] if peers else []):
            try:
                learned.update(httpx.get(f"{seed}/registry", timeout=5).json().get("nodes", {}))
            except Exception:
                pass
        stages = plan_auto_stages(dims, _bp, ram_gb, reserve, learned, target)
        if not stages:
            _fail("serve", "NO_GAP", "no range to cover (full coverage or insufficient RAM)", exit_code=2)
        typer.echo(f"eujeno serve --auto: claiming stages={stages} (ram={ram_gb}GB, target={target})", err=True)
    elif stages is None:
        _fail("serve", "USAGE_ERROR", "specify --stages or --auto", exit_code=2)
```
  (The rest of `serve` continues unchanged: `parse_stages(stages)`, `load_partial_model`, `create_app`. The `--coordinator` path remains valid with `--auto` too.)

- [ ] **Step 3c: in `eujeno/net/server.py` advertise the capacity in the record.** After the `own_stages = {...}` line (line ~27) add:
```python
    from eujeno.net.capacity import probe_capacity
    own_stages["capacity"] = probe_capacity()
```
  (It's additive: `build_chain` and `coverage_gaps` only read `embed`/`head`/`decoders`; `capacity` travels in the gossip and will be used by the allocator.)

- [ ] **Step 4: run PASS** — `.venv/bin/python -m pytest tests/test_serve_auto.py -v` → 2 passed. Verify CLI: `.venv/bin/eujeno serve --help | grep -q auto && echo ok`.

- [ ] **Step 5: full suite** — `.venv/bin/python -m pytest -q -p no:warnings` → green (no regressions; in particular `serve` with explicit `--stages` keeps working).

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py eujeno/net/server.py tests/test_serve_auto.py && git commit -m "feat(cli): serve --auto (layer self-assignment from capacity + gaps) + capacity in record"
```

---

## Task 5: docs (CLAUDE.md + P2P example)

**Files:** Modify `CLAUDE.md`, `docs/examples/p2p.md`. (Done by the controller.)

- [ ] **Step 1:** in `CLAUDE.md` add a row to the commands table for `serve --auto` and a note: "in P2P, start the nodes with `--auto --peers <seed>` and they split the layers among themselves (use `--target 2` for redundancy)".
- [ ] **Step 2:** in `docs/examples/p2p.md` add an "Auto-assembly" section with the example `eujeno serve --auto --peers ...`.
- [ ] **Step 3:** commit `docs: serve --auto (P2P auto-assembly)`.

---

## Self-Review

**Coverage ADR-0003:** slice 1 (capacity primitive) → Task 1 ✓; slice 2 (advertisement + gaps) → Task 2 + Task 4.3c ✓; slice 3 (allocator + serve --auto) → Task 3 + Task 4 ✓. The parametric `target` covers redundancy *at startup* (more nodes reach the target). **Out of scope (plan 2):** runtime re-assignment/reload on failure (slice 4) and reward ledger (slice 5).

**Placeholder scan:** complete code in every step; docs in Task 5.

**Type consistency:** `fit_layers(dims, bytes_per_param:int, ram_gb, reserve) -> {ram_per_layer_gb, ram_embed_head_gb, max_decoder_layers, fits_whole_model}`; `coverage_gaps(stages_by_url, num_layers, target) -> {decoder_gaps:[{lo,hi,replicas}], embed_replicas, head_replicas, target}`; `choose_stages(gaps, max_decoder_layers, num_layers, take_embed_head) -> str`; `plan_auto_stages(dims, bytes_per, ram_gb, reserve, stages_by_url, target) -> str`. The names and shapes match across the tasks.
