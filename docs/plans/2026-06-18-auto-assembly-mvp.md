# Auto-assembly MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** I nodi si auto-assegnano i layer da coprire leggendo i buchi di coverage dal registry + la propria capacità RAM, senza assegnazione manuale (`serve --auto`).

**Architecture:** Tre funzioni pure (capacità, gaps, decisione) + il wiring in `serve --auto`: all'avvio il nodo sonda la RAM, interroga un seed `/registry`, sceglie il range più bisognoso che ci sta, poi carica SOLO quei layer e serve. La capacità viene annunciata (additiva) nel record di gossip. `target` di replica è parametrico (≥2 ⇒ ridondanza a startup). Realizza le slice 1-3 di [ADR-0003](../decisions/ADR-0003-allocazione-capacity-aware.md). Failover-reload a runtime + reward ledger = piano successivo.

**Tech Stack:** Python · Typer · l'esistente `synapse/net/{discovery,server}.py`, `synapse/cli.py`, `model_config_dims`, `parse_stages`, `parse_dtype`. Dipendenza opzionale `psutil` (fallback stdlib).

---

## File Structure
```
synapse/net/capacity.py     # NUOVO: fit_layers (estratto da cli.fit) + probe_capacity
synapse/net/discovery.py    # MOD: coverage_gaps()
synapse/net/allocator.py    # NUOVO: choose_stages() (decisione pura)
synapse/net/server.py       # MOD: capacity nel record own_stages
synapse/cli.py              # MOD: fit usa fit_layers; serve --auto/--ram/--reserve
tests/test_capacity.py      # NUOVO
tests/test_coverage_gaps.py # NUOVO
tests/test_allocator.py     # NUOVO
tests/test_serve_auto.py    # NUOVO (unit del path di scelta, senza avviare il server)
```

---

## Task 1: `net/capacity.py` — fit_layers (estratto) + probe_capacity

**Files:** Create `synapse/net/capacity.py`, `tests/test_capacity.py`; Modify `synapse/cli.py` (fit usa fit_layers).

- [ ] **Step 1: test `tests/test_capacity.py`**
```python
from synapse.net.capacity import fit_layers, probe_capacity

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

- [ ] **Step 3: create `synapse/net/capacity.py`**
```python
"""Stima capacità di un nodo: quanti layer regge data la RAM, e probe risorse."""
import os

_GB = 1024 ** 3


def fit_layers(dims: dict, bytes_per_param: int, ram_gb: float, reserve: float = 0.2) -> dict:
    """Dato il modello (dims), la dimensione in byte di un parametro e la RAM
    disponibile in GB, stima quanti layer decoder regge il nodo."""
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
    """RAM totale/libera (GB) e numero di CPU. Usa psutil se presente, altrimenti stdlib."""
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

- [ ] **Step 4: refactor `cli.py::fit`** per usare `fit_layers` (NON cambiare i campi `data` emessi). Sostituisci il blocco di calcolo dentro `fit` (da `bytes_per = torch.finfo(...)` fino al dict `data`) con:
```python
    import torch
    from synapse.net.capacity import fit_layers
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
(Lascia invariati il resto della firma `fit`, la validazione dtype e la costruzione di `human`/`_emit_ok`.)

- [ ] **Step 5: run PASS** — `.venv/bin/python -m pytest tests/test_capacity.py tests/test_cli_fit.py -v` → tutti verdi (il refactor non rompe i test esistenti di `fit`).

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/capacity.py synapse/cli.py tests/test_capacity.py && git commit -m "feat(net): capacity.fit_layers + probe_capacity (fit CLI rifattorizzato)"
```

---

## Task 2: `discovery.coverage_gaps()`

**Files:** Modify `synapse/net/discovery.py`; Create `tests/test_coverage_gaps.py`.

- [ ] **Step 1: test `tests/test_coverage_gaps.py`**
```python
from synapse.net.discovery import coverage_gaps

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
    # tutto coperto solo 1 volta -> l'intero range è sotto-replicato
    assert g["decoder_gaps"] == [{"lo": 0, "hi": 24, "replicas": 1}]
    assert g["embed_replicas"] == 1 and g["head_replicas"] == 1
```

- [ ] **Step 2: run FAIL** — `.venv/bin/python -m pytest tests/test_coverage_gaps.py -v` → ImportError.

- [ ] **Step 3: append a `synapse/net/discovery.py`**
```python
def coverage_gaps(stages_by_url: dict, num_layers: int, target: int = 1) -> dict:
    """Range decoder con replica < target (scoperti o sotto-replicati), più il
    numero di replica di embed/head. `stages_by_url`: {url: {'embed','head','decoders'}}."""
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
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/discovery.py tests/test_coverage_gaps.py && git commit -m "feat(net): coverage_gaps (range scoperti/sotto-replicati)"
```

---

## Task 3: `net/allocator.py` — choose_stages() (decisione pura)

**Files:** Create `synapse/net/allocator.py`, `tests/test_allocator.py`.

- [ ] **Step 1: test `tests/test_allocator.py`**
```python
from synapse.net.allocator import choose_stages


def gaps(decoder_gaps, e=0, h=0, target=1):
    return {"decoder_gaps": decoder_gaps, "embed_replicas": e, "head_replicas": h, "target": target}


def test_takes_neediest_decoder_gap_capped_by_capacity():
    g = gaps([{"lo": 12, "hi": 24, "replicas": 0}], e=1, h=1)
    # regge solo 5 layer
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

- [ ] **Step 3: create `synapse/net/allocator.py`**
```python
"""Decisione di auto-assegnazione: dato il quadro dei buchi (coverage_gaps) e la
capacità del nodo, sceglie lo stage spec da rivendicare. Funzione pura."""


def choose_stages(gaps: dict, max_decoder_layers: int, num_layers: int,
                  take_embed_head: bool) -> str:
    """Ritorna uno stage spec per parse_stages (es. 'embed,decoder:12-17,head'),
    o '' se non c'è nulla di utile/possibile da rivendicare."""
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
cd /Users/alberto/Projects/AI/synapse && git add synapse/net/allocator.py tests/test_allocator.py && git commit -m "feat(net): allocator.choose_stages (decisione di auto-assegnazione)"
```

---

## Task 4: `serve --auto` + capacity nel record

**Files:** Modify `synapse/net/server.py` (capacity nel record), `synapse/cli.py` (serve --auto). Create `tests/test_serve_auto.py`.

- [ ] **Step 1: test `tests/test_serve_auto.py`** — testa la funzione pura di pianificazione `plan_auto_stages` (niente server avviato).
```python
from synapse.cli import plan_auto_stages

DIMS = {"num_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
        "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936}


def test_plan_first_node_claims_whole_when_capable():
    # registry vuoto, nodo capiente -> prende tutto
    s = plan_auto_stages(DIMS, bytes_per=4, ram_gb=64.0, reserve=0.2,
                         stages_by_url={}, target=1)
    assert s == "embed,decoder:0-24,head"


def test_plan_second_node_fills_remaining_gap():
    # un nodo copre embed+0-12; il secondo, piccolo, prende il buco centrale
    existing = {"a": {"embed": True, "head": False, "decoders": ["0-12"]}}
    s = plan_auto_stages(DIMS, bytes_per=4, ram_gb=0.6, reserve=0.2,
                         stages_by_url=existing, target=1)
    assert s.startswith("decoder:12-")
```

- [ ] **Step 2: run FAIL** — `.venv/bin/python -m pytest tests/test_serve_auto.py -v` → ImportError (`plan_auto_stages`).

- [ ] **Step 3a: in `synapse/cli.py` aggiungi la funzione pura `plan_auto_stages`** (vicino agli altri helper, NON dentro un comando):
```python
def plan_auto_stages(dims: dict, bytes_per: int, ram_gb: float, reserve: float,
                     stages_by_url: dict, target: int) -> str:
    """Decide lo stage spec da rivendicare combinando capacità (fit) e buchi (gaps)."""
    from synapse.net.capacity import fit_layers
    from synapse.net.discovery import coverage_gaps
    from synapse.net.allocator import choose_stages
    nl = dims["num_layers"]
    fit = fit_layers(dims, bytes_per, ram_gb, reserve)
    gaps = coverage_gaps(stages_by_url, nl, target=target)
    # un nodo "capiente" (regge l'intero modello) si offre anche per embed/head
    take_eh = fit["fits_whole_model"] or fit["max_decoder_layers"] >= nl
    return choose_stages(gaps, fit["max_decoder_layers"], nl, take_embed_head=take_eh)
```

- [ ] **Step 3b: in `synapse/cli.py` rendi `--stages` opzionale e aggiungi `--auto/--ram/--reserve` a `serve`.** Cambia la firma di `serve`:
  - `stages: str = typer.Option(None, "--stages", ...)` (era `...` obbligatorio).
  - aggiungi:
    ```python
    auto: bool = typer.Option(False, "--auto", help="Auto-assegna i layer dai buchi del registry + capacità RAM"),
    ram: float = typer.Option(None, "--ram", help="RAM da usare per l'auto-assegnazione, GB (default: rilevata)"),
    reserve: float = typer.Option(0.2, "--reserve", help="Frazione RAM riservata (auto)"),
    target: int = typer.Option(1, "--target", help="Replica desiderata per range (auto; 2 = ridondanza)"),
    ```
  Subito dopo `import uvicorn` (prima di `parse_stages`), inserisci il ramo auto:
```python
    if auto:
        import torch, httpx
        from synapse.net.capacity import probe_capacity, fit_layers
        from synapse.config import parse_dtype as _pdt
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
            _fail("serve", "NO_GAP", "nessun range da coprire (coverage completa o RAM insufficiente)", exit_code=2)
        typer.echo(f"synapse serve --auto: rivendico stages={stages} (ram={ram_gb}GB, target={target})", err=True)
    elif stages is None:
        _fail("serve", "USAGE_ERROR", "specifica --stages oppure --auto", exit_code=2)
```
  (Il resto di `serve` prosegue invariato: `parse_stages(stages)`, `load_partial_model`, `create_app`. Il path `--coordinator` resta valido anche con `--auto`.)

- [ ] **Step 3c: in `synapse/net/server.py` annuncia la capacità nel record.** Dopo la riga `own_stages = {...}` (riga ~27) aggiungi:
```python
    from synapse.net.capacity import probe_capacity
    own_stages["capacity"] = probe_capacity()
```
  (È additivo: `build_chain` e `coverage_gaps` leggono solo `embed`/`head`/`decoders`; `capacity` viaggia nel gossip e sarà usato dall'allocatore.)

- [ ] **Step 4: run PASS** — `.venv/bin/python -m pytest tests/test_serve_auto.py -v` → 2 passed. Verifica CLI: `.venv/bin/synapse serve --help | grep -q auto && echo ok`.

- [ ] **Step 5: suite completa** — `.venv/bin/python -m pytest -q -p no:warnings` → verde (nessuna regressione; in particolare `serve` con `--stages` esplicito continua a funzionare).

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/cli.py synapse/net/server.py tests/test_serve_auto.py && git commit -m "feat(cli): serve --auto (auto-assegnazione layer da capacità + gaps) + capacity nel record"
```

---

## Task 5: docs (CLAUDE.md + esempio P2P)

**Files:** Modify `CLAUDE.md`, `docs/examples/p2p.md`. (Lo fa il controller.)

- [ ] **Step 1:** in `CLAUDE.md` aggiungi una riga alla tabella comandi per `serve --auto` e una nota: "in P2P, avvia i nodi con `--auto --peers <seed>` e si dividono i layer da soli (usa `--target 2` per ridondanza)".
- [ ] **Step 2:** in `docs/examples/p2p.md` aggiungi una sezione "Auto-assemblaggio" con l'esempio `synapse serve --auto --peers ...`.
- [ ] **Step 3:** commit `docs: serve --auto (auto-assemblaggio P2P)`.

---

## Self-Review

**Coverage ADR-0003:** slice 1 (capacity primitive) → Task 1 ✓; slice 2 (advertisement + gaps) → Task 2 + Task 4.3c ✓; slice 3 (allocatore + serve --auto) → Task 3 + Task 4 ✓. `target` parametrico copre la ridondanza *a startup* (più nodi raggiungono target). **Fuori scope (piano 2):** ri-assegnazione/reload a runtime su guasto (slice 4) e reward ledger (slice 5).

**Placeholder scan:** codice completo in ogni step; doc nel Task 5.

**Type consistency:** `fit_layers(dims, bytes_per_param:int, ram_gb, reserve) -> {ram_per_layer_gb, ram_embed_head_gb, max_decoder_layers, fits_whole_model}`; `coverage_gaps(stages_by_url, num_layers, target) -> {decoder_gaps:[{lo,hi,replicas}], embed_replicas, head_replicas, target}`; `choose_stages(gaps, max_decoder_layers, num_layers, take_embed_head) -> str`; `plan_auto_stages(dims, bytes_per, ram_gb, reserve, stages_by_url, target) -> str`. I nomi e le forme combaciano tra i task.
