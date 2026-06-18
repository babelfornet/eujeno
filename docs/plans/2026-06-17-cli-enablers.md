# CLI enabler: --dtype, models, up, CLAUDE.md — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Rendere fluido il caso d'uso "un AI (Claude Code) configura/usa il nodo da CLI": `--dtype` (modelli grandi in bf16), `axyn models` + check compatibilità (sapere quali modelli usare), `axyn up` (bring-up in un comando), e un `CLAUDE.md` che insegna a un agente a pilotare la CLI.

**Architecture:** `parse_dtype` in `config.py`; `--dtype` su `serve`/`generate`/`selfcheck`. `axyn models` (lista curata) + campo `compatible` in `model --info` (riconosce l'architettura `qwen2`/`llama`). `axyn up --model X [--dtype]` avvia coordinator + un nodo serve che copre tutti i layer (sottoprocessi), attende l'operatività e stampa gli endpoint (`--dry-run` per testabilità/anteprima). `CLAUDE.md` + sezione Getting started nel README.

**Tech Stack:** Python · Typer · l'esistente `axyn/{config,cli}.py`, `model_config_dims`, `NodeManager`.

---

## File Structure
```
axyn/config.py               # MOD: parse_dtype + SUPPORTED_ARCHS
axyn/cli.py                  # MOD: --dtype (serve/generate/selfcheck), models, model --info compatible, up
tests/test_dtype.py             # NUOVO: parse_dtype (veloce)
tests/test_cli_models.py        # NUOVO: models + up --dry-run (veloce)
CLAUDE.md                       # NUOVO
README.md                       # MOD: Getting started
docs/examples/agents.md         # MOD (models/up)
```

---

## Task 1: `--dtype` (bf16/fp16 per modelli grandi)

**Files:** modify `axyn/config.py`, `axyn/cli.py`; create `tests/test_dtype.py`.

- [ ] **Step 1: test `tests/test_dtype.py`**
```python
import torch
import pytest
from axyn.config import parse_dtype


def test_parse_known():
    assert parse_dtype("float32") is torch.float32
    assert parse_dtype("bf16") is torch.bfloat16
    assert parse_dtype("bfloat16") is torch.bfloat16
    assert parse_dtype("fp16") is torch.float16


def test_parse_unknown_raises():
    with pytest.raises(ValueError):
        parse_dtype("int4")
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_dtype.py -v` → ImportError.

- [ ] **Step 3: in `axyn/config.py`** aggiungi:
```python
_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}

SUPPORTED_ARCHS = {"qwen2", "llama"}


def parse_dtype(name: str):
    key = str(name).lower()
    if key not in _DTYPES:
        raise ValueError(f"dtype non valido: {name!r} (usa float32/bfloat16/float16)")
    return _DTYPES[key]
```

- [ ] **Step 4: in `axyn/cli.py` aggiungi `--dtype` al comando `serve`** — opzione e uso. Aggiungi alla firma di `serve`:
```python
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16 (bf16 per modelli grandi)"),
```
e dove `serve` carica il modello (`load_partial_model(model_id, spec, DTYPE, DEVICE)`), sostituisci `DTYPE` con il dtype scelto:
```python
    from axyn.config import parse_dtype
    try:
        _dtype = parse_dtype(dtype)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    ...
        model, tokenizer = load_partial_model(model_id, spec, _dtype, DEVICE)
```
(Assicurati che `parse_dtype` sia importato; il default `"float32"` mantiene il comportamento attuale.)

- [ ] **Step 5: run PASS** — `... pytest tests/test_dtype.py -v` → 2 passed. Verifica che la CLI importi: `.venv/bin/axyn serve --help | grep -q dtype && echo ok`.

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/config.py axyn/cli.py tests/test_dtype.py && git commit -m "feat(cli): --dtype (bf16/fp16) su serve per modelli grandi"
```

---

## Task 2: `axyn models` + check compatibilità in `model --info`

**Files:** modify `axyn/cli.py`; create `tests/test_cli_models.py`.

- [ ] **Step 1: test `tests/test_cli_models.py`**
```python
import json
from typer.testing import CliRunner
from axyn.cli import app

runner = CliRunner()


def test_models_lists_examples():
    r = runner.invoke(app, ["--json", "models"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    assert "qwen2" in data["supported_architectures"]
    assert any("Qwen2.5" in m for m in data["examples"])
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_cli_models.py -v` → no `models` command.

- [ ] **Step 3: in `axyn/cli.py` aggiungi il comando `models`** (dopo `model`):
```python
@app.command()
def models():
    """Elenca i modelli/famiglie compatibili (architettura Llama/Qwen2)."""
    from axyn.config import SUPPORTED_ARCHS
    examples = [
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
    ]
    data = {"supported_architectures": sorted(SUPPORTED_ARCHS), "examples": examples,
            "note": "Architettura Llama/Qwen2 (decoder-only). Verifica con 'axyn model --info --model <id>'. "
                    "Per modelli grandi usa --dtype bfloat16 e/o splitta su piu nodi."}
    human = "Modelli compatibili (Llama/Qwen2):\n" + "\n".join(f"  - {m}" for m in examples)
    _emit_ok("models", data, human=human)
```
Inoltre, nel comando `model` (`--info`) aggiungi i campi compatibilità. Dove costruisce `data` con i dim del modello, aggiungi `architecture` e `compatible`:
```python
        from axyn.config import SUPPORTED_ARCHS
        arch = getattr(__import__("transformers").AutoConfig.from_pretrained(model_id), "model_type", "?")
        data["architecture"] = arch
        data["compatible"] = arch in SUPPORTED_ARCHS
```
> Nota: `model --info` carica gia `AutoConfig` via `model_config_dims`. Se preferisci evitare un secondo `from_pretrained`, fai restituire `model_type` da `model_config_dims` (aggiungi `"model_type": cfg.model_type` al dict in `loader.py`) e usalo qui. Scegli l'opzione piu pulita; l'importante e che `model --info --json` includa `architecture` e `compatible`.

- [ ] **Step 4: run PASS** — `... pytest tests/test_cli_models.py -v` → PASS. Verifica `.venv/bin/axyn --help | grep -q models`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py axyn/model/loader.py tests/test_cli_models.py && git commit -m "feat(cli): comando 'models' + check compatibilita in 'model --info'"
```

---

## Task 3: `axyn up` (bring-up in un comando)

**Files:** modify `axyn/cli.py`; modify `tests/test_cli_models.py` (append).

- [ ] **Step 1: append a `tests/test_cli_models.py`**
```python
def test_up_dry_run_prints_commands(monkeypatch):
    # evita il download config: stub model_config_dims
    import axyn.cli as cli
    monkeypatch.setattr(cli, "model_config_dims", lambda mid: {"num_layers": 24})
    r = runner.invoke(app, ["--json", "up", "--model", "X", "--dtype", "bfloat16", "--dry-run"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    cmds = " ".join(" ".join(c) for c in data["commands"])
    assert "coordinator" in cmds and "serve" in cmds
    assert "embed,decoder:0-24,head" in cmds
    assert "bfloat16" in cmds
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_cli_models.py::test_up_dry_run_prints_commands -v` → no `up`.

- [ ] **Step 3: in `axyn/cli.py` aggiungi il comando `up`** (dopo `coordinator`):
```python
@app.command()
def up(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host del coordinator"),
    port: int = typer.Option(9000, "--port", help="Porta del coordinator"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Stampa i comandi senza avviare nulla"),
):
    """Avvia in un colpo una rete operativa a nodo singolo (coordinator + un nodo che copre tutto il modello)."""
    nl = model_config_dims(model_id)["num_layers"]
    coord_cmd = [sys.executable, "-m", "axyn", "coordinator", "--model", model_id,
                 "--host", host, "--port", str(port)]
    ws = f"ws://{host}:{port}/node"
    serve_cmd = [sys.executable, "-m", "axyn", "serve", "--coordinator", ws,
                 "--stages", f"embed,decoder:0-{nl},head", "--model", model_id, "--dtype", dtype]
    if dry_run:
        _emit_ok("up", {"commands": [coord_cmd, serve_cmd],
                        "coordinator_url": f"http://{host}:{port}"},
                 human="comandi:\n  " + "\n  ".join(" ".join(c) for c in [coord_cmd, serve_cmd]))
        return
    import subprocess
    import time as _t
    import httpx
    procs = [subprocess.Popen(coord_cmd)]
    _t.sleep(4)
    procs.append(subprocess.Popen(serve_cmd))
    base = f"http://{host}:{port}"
    typer.echo(f"axyn up: avvio rete per {model_id} (dtype={dtype})…", err=True)
    operational = False
    for _ in range(120):
        try:
            reg = httpx.get(f"{base}/registry", timeout=5).json()
            if reg.get("nodes"):
                operational = True
                break
        except Exception:
            pass
        _t.sleep(2)
    if operational:
        typer.echo(f"PRONTO. Interroga: axyn infer --coordinator {base} --prompt \"...\"", err=True)
        typer.echo(f"Frontend:        axyn ui --coordinator {base}", err=True)
    else:
        typer.echo("rete non ancora operativa (controlla i log).", err=True)
    try:
        procs[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_cli_models.py -v` → PASS. Verifica `.venv/bin/axyn --help | grep -q up`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py tests/test_cli_models.py && git commit -m "feat(cli): comando 'up' (bring-up rete a nodo singolo in un comando)"
```

---

## Task 4: CLAUDE.md + Getting started + docs + suite

**Files:** create `CLAUDE.md`; modify `README.md`, `docs/examples/agents.md`. (Lo fa il controller, non un subagent.)

- [ ] **Step 1:** crea `CLAUDE.md` (guida per un agente: cos'e Axyn, install, comandi chiave con esempi — model/models/up/serve/coordinator/infer/ui/mcp — modelli supportati, workflow tipici a/b/c/d, nota dtype/memoria).
- [ ] **Step 2:** aggiungi al `README.md` una sezione "## Installazione / Getting started" in cima (clone → venv → `pip install -e .` → `axyn --help`; quickstart `axyn up`).
- [ ] **Step 3:** in `docs/examples/agents.md` cita `axyn models` e `axyn up`.
- [ ] **Step 4:** suite completa `... pytest -q -p no:warnings` → verde.
- [ ] **Step 5:** commit `docs: CLAUDE.md + Getting started (install + up + models)`.

---

## Self-Review

**Coverage:** `--dtype` (Task 1) ✓; `models` + compatibilita `model --info` (Task 2) ✓; `up` bring-up (Task 3) ✓; CLAUDE.md + Getting started (Task 4) ✓. Risponde a "come lancio la CLI" (README install) e "quali modelli" (models + compatible).

**Placeholder scan:** codice completo; CLAUDE.md scritto nel Task 4.

**Type consistency:** `parse_dtype(str)->torch.dtype`; `serve --dtype` usa `load_partial_model(..., _dtype, ...)`; `models` ritorna `{supported_architectures, examples, note}`; `model --info` aggiunge `architecture`/`compatible`; `up` costruisce comandi `python -m axyn coordinator|serve` con stage `embed,decoder:0-N,head` e `--dtype`.
```
