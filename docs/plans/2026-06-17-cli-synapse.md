# CLI `axyn` (AI-native) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un entry-point eseguibile `axyn` per tutte le operazioni del progetto, progettato per essere usato direttamente da un agente AI (output JSON con envelope stabile, stream puliti, exit code deterministici, non-interattivo, auto-descrittivo).

**Architecture:** La CLI è un layer di presentazione sottile in `axyn/cli.py` (Typer): ogni comando chiama il codice esistente in `axyn/model/` e formatta l'output tramite un unico helper-envelope condiviso (`_emit_ok`/`_fail`) che rispetta il flag globale `--json`. Due helper puri riusabili (`compute_boundaries`, `model_config_dims`) vengono aggiunti a `axyn/model/` con i loro test. L'entry-point è registrato in `pyproject.toml`.

**Tech Stack:** Python 3.12 · Typer · esistente `axyn/model/` (loader, blocks, generate) · transformers 4.46.3 · pytest + `typer.testing.CliRunner`.

**Spec di riferimento:** [docs/prd/cli.md](../prd/cli.md). Build sulla foundation di [docs/prd/part-1-peer-node.md](../prd/part-1-peer-node.md).

---

## File Structure

```
pyproject.toml                 # MODIFICA: + dipendenza typer, + [project.scripts] axyn
axyn/
  model/
    blocks.py                  # MODIFICA: + compute_boundaries()
    loader.py                  # MODIFICA: + model_config_dims()
  cli.py                       # NUOVO: app Typer, envelope, comandi
tests/
  test_boundaries.py           # NUOVO: compute_boundaries (puro, veloce)
  test_config_dims.py          # NUOVO: model_config_dims (slow)
  test_cli.py                  # NUOVO: comandi via CliRunner
```

Responsabilità: gli helper puri stanno in `model/` (riusabili, testabili senza CLI); `cli.py` è solo presentazione (parsing + envelope + chiamate). Nessuna logica di dominio in `cli.py`.

---

## Task 1: `compute_boundaries` (helper puro)

**Files:**
- Modify: `axyn/model/blocks.py` (aggiungi funzione in fondo)
- Test: `tests/test_boundaries.py`

- [ ] **Step 1: Scrivi i test che falliscono**

`tests/test_boundaries.py`:
```python
import pytest
from axyn.model.blocks import compute_boundaries


def test_even_split():
    assert compute_boundaries(24, 2) == [0, 12, 24]


def test_uneven_split_is_contiguous_and_covers_all():
    b = compute_boundaries(24, 5)
    assert b[0] == 0 and b[-1] == 24
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))   # strettamente crescente
    assert len(b) == 6                                        # 5 blocchi -> 6 confini
    sizes = [b[i + 1] - b[i] for i in range(len(b))[:-1]] if False else [b[i+1]-b[i] for i in range(5)]
    assert max(sizes) - min(sizes) <= 1                       # il più uniforme possibile


def test_single_block():
    assert compute_boundaries(24, 1) == [0, 24]


def test_one_block_per_layer():
    assert compute_boundaries(3, 3) == [0, 1, 2, 3]


def test_rejects_too_many_blocks():
    with pytest.raises(ValueError):
        compute_boundaries(4, 5)


def test_rejects_non_positive_blocks():
    with pytest.raises(ValueError):
        compute_boundaries(24, 0)
```

- [ ] **Step 2: Esegui per vederli fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_boundaries.py -v`
Expected: FAIL con `ImportError`/`AttributeError` su `compute_boundaries`.

- [ ] **Step 3: Implementa in `axyn/model/blocks.py`**

Aggiungi in fondo al file:
```python
def compute_boundaries(num_layers: int, n_blocks: int) -> list[int]:
    """Divide num_layers in n_blocks blocchi decoder contigui il più possibile
    uguali. Ritorna i confini, es. (24, 2) -> [0, 12, 24]. Copre sempre
    [0, num_layers] in modo strettamente crescente."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks deve essere >= 1, ricevuto {n_blocks}")
    if n_blocks > num_layers:
        raise ValueError(f"n_blocks ({n_blocks}) non può superare num_layers ({num_layers})")
    base, extra = divmod(num_layers, n_blocks)
    boundaries = [0]
    for i in range(n_blocks):
        size = base + (1 if i < extra else 0)   # i primi `extra` blocchi hanno un layer in più
        boundaries.append(boundaries[-1] + size)
    return boundaries
```

- [ ] **Step 4: Esegui per vederli passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_boundaries.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/model/blocks.py tests/test_boundaries.py && git commit -m "feat(model): compute_boundaries per split uniforme dei layer in N blocchi"
```

---

## Task 2: `model_config_dims` (helper puro, solo config)

**Files:**
- Modify: `axyn/model/loader.py` (aggiungi funzione)
- Test: `tests/test_config_dims.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`tests/test_config_dims.py`:
```python
import pytest
from axyn.model.loader import model_config_dims


@pytest.mark.slow
def test_config_dims_without_loading_weights():
    dims = model_config_dims("Qwen/Qwen2.5-0.5B-Instruct")
    assert dims["num_layers"] == 24
    assert dims["hidden_size"] == 896
    assert "num_attention_heads" in dims
    assert "num_key_value_heads" in dims
```

- [ ] **Step 2: Esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_config_dims.py -m slow -v`
Expected: FAIL con `ImportError`/`AttributeError` su `model_config_dims`.

- [ ] **Step 3: Implementa in `axyn/model/loader.py`**

Aggiungi in fondo (riusa la stessa forma del dict di `model_dims`):
```python
def model_config_dims(model_id: str) -> dict:
    """Dimensioni del modello dalla sola AutoConfig (NIENTE pesi scaricati)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return {
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    }
```

- [ ] **Step 4: Esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_config_dims.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/model/loader.py tests/test_config_dims.py && git commit -m "feat(model): model_config_dims (dims da AutoConfig senza pesi)"
```

---

## Task 3: Entry-point + envelope + comando `version`

**Files:**
- Modify: `pyproject.toml` (dipendenza typer + scripts)
- Create: `axyn/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Aggiungi `typer` e l'entry-point in `pyproject.toml`**

Nella tabella `[project] dependencies` aggiungi `"typer>=0.12"`. Dopo la sezione `[project.optional-dependencies]` aggiungi:
```toml
[project.scripts]
axyn = "axyn.cli:app"
```
Poi reinstalla per registrare lo script ed installare typer:
```bash
cd /Users/alberto/Projects/AI/axyn && .venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 2: Scrivi il test che fallisce**

`tests/test_cli.py`:
```python
import json
from typer.testing import CliRunner
from axyn.cli import app

runner = CliRunner()


def test_version_json_is_valid_envelope():
    result = runner.invoke(app, ["--json", "version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)          # stdout deve essere JSON puro
    assert payload["ok"] is True
    assert payload["command"] == "version"
    assert "version" in payload["data"]


def test_version_text_mode():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "axyn" in result.stdout
```

- [ ] **Step 3: Esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL con `ImportError` su `axyn.cli`.

- [ ] **Step 4: Implementa `axyn/cli.py`**

```python
import json as _json
import os
import sys

import typer

from axyn.config import DEFAULT_MODEL_ID

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Axyn — rete di inferenza LLM decentralizzata.")

# Stato globale impostato dal callback (modalità output).
_state = {"json": False}


def _json_enabled(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("AXYN_JSON", "").lower() in ("1", "true", "yes")


@app.callback()
def _main(json_out: bool = typer.Option(False, "--json", "-j", help="Output come envelope JSON")):
    """Opzioni globali. Imposta la modalità output e silenzia il rumore su stdout."""
    _state["json"] = _json_enabled(json_out)
    # Disciplina degli stream: niente progress bar / log su stdout.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


def _emit_ok(command: str, data: dict, human: str) -> None:
    if _state["json"]:
        typer.echo(_json.dumps({"ok": True, "command": command, "data": data}))
    else:
        typer.echo(human)


def _fail(command: str, code: str, message: str, exit_code: int = 1):
    if _state["json"]:
        typer.echo(_json.dumps({"ok": False, "command": command, "error": {"code": code, "message": message}}))
    else:
        typer.echo(f"error[{code}]: {message}", err=True)
    raise typer.Exit(exit_code)


@app.command()
def version():
    """Stampa la versione del pacchetto."""
    from importlib.metadata import version as _pkg_version
    try:
        v = _pkg_version("axyn")
    except Exception:
        v = "0.0.1"
    _emit_ok("version", {"version": v}, human=f"axyn {v}")
```

- [ ] **Step 5: Esegui per vederlo passare + verifica lo script installato**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: 2 passed.
Poi verifica l'entry-point reale:
```bash
cd /Users/alberto/Projects/AI/axyn && .venv/bin/axyn version && .venv/bin/axyn --json version
```
Expected: prima riga `axyn 0.0.1`; seconda riga `{"ok": true, "command": "version", "data": {"version": "0.0.1"}}`.

- [ ] **Step 6: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add pyproject.toml axyn/cli.py tests/test_cli.py && git commit -m "feat(cli): entry-point axyn + envelope JSON + comando version"
```

---

## Task 4: Comando `model` (`--info`)

**Files:**
- Modify: `axyn/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Aggiungi i test (in coda a `tests/test_cli.py`)**

```python
def test_model_info_json():
    result = runner.invoke(app, ["--json", "model", "--info", "--blocks", "2"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["num_layers"] == 24
    assert payload["data"]["boundaries"] == [0, 12, 24]


def test_model_invalid_blocks_returns_error_envelope():
    result = runner.invoke(app, ["--json", "model", "--info", "--blocks", "999"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_BOUNDARIES"
```

Nota: questi test scaricano la config del modello (piccola). Marcali `slow`:
aggiungi `import pytest` in cima al file (se non già presente) e decora entrambe le funzioni con `@pytest.mark.slow`.

- [ ] **Step 2: Esegui per vederli fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py -m slow -v`
Expected: FAIL (comando `model` inesistente → exit 2 / no such command).

- [ ] **Step 3: Implementa il comando in `axyn/cli.py`**

Aggiungi gli import in cima (sotto quelli esistenti):
```python
from axyn.model.blocks import compute_boundaries
from axyn.model.loader import model_config_dims
```
Aggiungi il comando dopo `version`:
```python
@app.command()
def model(
    info: bool = typer.Option(False, "--info", help="Mostra dimensioni e split proposto"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    blocks: int = typer.Option(2, "--blocks", help="Numero di blocchi decoder"),
):
    """Operazioni sul modello. In v1: --info (default) mostra dims + split proposto."""
    try:
        dims = model_config_dims(model_id)
    except Exception as e:
        _fail("model", "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(dims["num_layers"], blocks)
    except ValueError as e:
        _fail("model", "INVALID_BOUNDARIES", str(e))
    data = {"model": model_id, **dims, "blocks": blocks, "boundaries": boundaries}
    human = (
        f"model: {model_id}\n"
        f"layers: {dims['num_layers']}  hidden: {dims['hidden_size']}  "
        f"heads: {dims['num_attention_heads']} (kv {dims['num_key_value_heads']})\n"
        f"blocchi: {blocks}  confini: {boundaries}"
    )
    _emit_ok("model", data, human)
```

- [ ] **Step 4: Esegui per vederli passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py -m slow -v`
Expected: i due nuovi test passano.

- [ ] **Step 5: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py tests/test_cli.py && git commit -m "feat(cli): comando 'model --info' (dims + split, errore INVALID_BOUNDARIES)"
```

---

## Task 5: Comandi `generate` e `selfcheck` (+ stdin)

**Files:**
- Modify: `axyn/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Aggiungi i test (in coda a `tests/test_cli.py`)**

```python
@pytest.mark.slow
def test_generate_json_produces_text():
    result = runner.invoke(app, ["--json", "generate", "--prompt", "La capitale dell'Italia è", "--max-new-tokens", "8"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert isinstance(payload["data"]["text"], str) and payload["data"]["text"]
    assert len(payload["data"]["tokens"]) == 8


@pytest.mark.slow
def test_generate_reads_prompt_from_stdin():
    result = runner.invoke(app, ["--json", "generate", "--prompt", "-", "--max-new-tokens", "4"], input="La capitale dell'Italia è")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data"]["prompt"] == "La capitale dell'Italia è"


@pytest.mark.slow
def test_selfcheck_reports_match():
    result = runner.invoke(app, ["--json", "selfcheck", "--max-new-tokens", "8"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["data"]["match"] is True
    assert payload["data"]["reference"] == payload["data"]["pipeline"]
```

- [ ] **Step 2: Esegui per vederli fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py -m slow -v -k "generate or selfcheck"`
Expected: FAIL (comandi inesistenti).

- [ ] **Step 3: Implementa in `axyn/cli.py`**

Aggiungi questi import in cima (sotto gli altri):
```python
from axyn.config import DTYPE, DEVICE
from axyn.model.loader import load_full_model, model_dims
from axyn.model.blocks import split_into_blocks
from axyn.model.generate import reference_generate, pipeline_generate
```
Aggiungi un helper per lo stdin e i due comandi, dopo `model`:
```python
def _read_prompt(prompt: str) -> str:
    """'-' legge il prompt da stdin (pipe da un agente)."""
    if prompt == "-":
        return sys.stdin.read().strip()
    return prompt


def _prepare_pipeline(model_id: str, blocks: int, command: str):
    """Carica il modello, calcola i confini, splitta. Solleva via _fail su errore."""
    try:
        model, tokenizer = load_full_model(model_id, DTYPE, DEVICE)
        model.eval()
    except Exception as e:
        _fail(command, "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(model_dims(model)["num_layers"], blocks)
    except ValueError as e:
        _fail(command, "INVALID_BOUNDARIES", str(e))
    embed, decoders, head = split_into_blocks(model, boundaries)
    return model, tokenizer, embed, decoders, head


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", help="Testo del prompt ('-' legge da stdin)"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
    blocks: int = typer.Option(2, "--blocks", help="Numero di blocchi decoder"),
):
    """Genera testo eseguendo la pipeline splittata in-process."""
    prompt = _read_prompt(prompt)
    model, tokenizer, embed, decoders, head = _prepare_pipeline(model_id, blocks, "generate")
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        tokens = pipeline_generate(embed, decoders, head, ids, max_new_tokens)
        text = tokenizer.decode(tokens)
    except Exception as e:
        _fail("generate", "GENERATION_FAILED", str(e))
    data = {"model": model_id, "prompt": prompt, "text": text, "tokens": tokens}
    _emit_ok("generate", data, human=text)


```

> **Nota correttezza (importante):** `generate` usa `_prepare_pipeline` (che carica e splitta). `selfcheck` invece **NON** deve usarlo: il riferimento va catturato con `reference_generate(model, ...)` **prima** di `split_into_blocks` (che muta `layer_idx`). Quindi `selfcheck` carica il modello, fa il riferimento, POI splitta:

```python
@app.command()
def selfcheck(
    prompt: str = typer.Option("La capitale dell'Italia è", "--prompt", help="Prompt di verifica ('-' = stdin)"),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="ID del modello Hugging Face"),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", help="Numero di token da generare"),
    blocks: int = typer.Option(2, "--blocks", help="Numero di blocchi decoder"),
):
    """Confronta la pipeline splittata col modello intero (golden equivalence)."""
    prompt = _read_prompt(prompt)
    try:
        model, tokenizer = load_full_model(model_id, DTYPE, DEVICE)
        model.eval()
    except Exception as e:
        _fail("selfcheck", "MODEL_LOAD_FAILED", str(e))
    try:
        boundaries = compute_boundaries(model_dims(model)["num_layers"], blocks)
    except ValueError as e:
        _fail("selfcheck", "INVALID_BOUNDARIES", str(e))
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        reference = reference_generate(model, ids, max_new_tokens)        # PRIMA dello split
        embed, decoders, head = split_into_blocks(model, boundaries)      # muta layer_idx
        pipeline = pipeline_generate(embed, decoders, head, ids, max_new_tokens)
    except Exception as e:
        _fail("selfcheck", "GENERATION_FAILED", str(e))
    data = {"model": model_id, "match": reference == pipeline, "reference": reference, "pipeline": pipeline}
    human = f"match: {data['match']}\nreference: {reference}\npipeline: {pipeline}"
    _emit_ok("selfcheck", data, human)
```
(`generate` può usare `_prepare_pipeline` senza problemi: non gli serve il riferimento non-mutato.)

- [ ] **Step 4: Esegui per vederli passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py -m slow -v -k "generate or selfcheck"`
Expected: i 3 nuovi test passano.

- [ ] **Step 5: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py tests/test_cli.py && git commit -m "feat(cli): comandi generate e selfcheck (+ prompt da stdin)"
```

---

## Task 6: Comando `schema` (auto-descrizione per AI)

**Files:**
- Modify: `axyn/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Aggiungi il test (in coda a `tests/test_cli.py`)**

```python
def test_schema_lists_commands():
    result = runner.invoke(app, ["--json", "schema"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = {c["name"] for c in payload["data"]["commands"]}
    assert {"version", "model", "generate", "selfcheck", "schema"} <= names
    # ogni comando elenca le sue opzioni con nome
    model_cmd = next(c for c in payload["data"]["commands"] if c["name"] == "model")
    opt_names = {o["name"] for o in model_cmd["options"]}
    assert "info" in opt_names and "blocks" in opt_names
```
(Questo test è veloce: nessun modello.)

- [ ] **Step 2: Esegui per vederlo fallire**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py::test_schema_lists_commands -v`
Expected: FAIL (comando `schema` inesistente).

- [ ] **Step 3: Implementa `schema` in `axyn/cli.py`**

Aggiungi in fondo:
```python
@app.command()
def schema():
    """Stampa l'albero comandi+opzioni in forma machine-readable (per agenti AI)."""
    import click
    import typer.main

    root = typer.main.get_command(app)
    commands = []
    for name, cmd in sorted(root.commands.items()):
        options = []
        for param in cmd.params:
            if isinstance(param, click.Argument):
                continue
            options.append({
                "name": param.name,
                "type": getattr(param.type, "name", str(param.type)),
                "default": param.default,
                "required": bool(param.required),
            })
        commands.append({"name": name, "help": (cmd.help or "").strip(), "options": options})
    _emit_ok("schema", {"commands": commands}, human=_json.dumps({"commands": commands}, indent=2))
```

- [ ] **Step 4: Esegui per vederlo passare**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py::test_schema_lists_commands -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py tests/test_cli.py && git commit -m "feat(cli): comando schema (auto-descrizione machine-readable)"
```

---

## Task 7: Verifica stream puliti + suite completa + ROADMAP

**Files:**
- Test: `tests/test_cli.py`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Aggiungi il test di disciplina degli stream (in coda a `tests/test_cli.py`)**

```python
@pytest.mark.slow
def test_json_stdout_is_pure_json_no_progress_noise():
    # mix_stderr=False separa stdout da stderr: stdout deve essere SOLO l'envelope JSON.
    isolated = CliRunner(mix_stderr=False)
    result = isolated.invoke(app, ["--json", "model", "--info"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)   # nessuna barra di progresso / warning mescolati
    assert payload["ok"] is True
```

- [ ] **Step 2: Esegui il nuovo test**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_cli.py::test_json_stdout_is_pure_json_no_progress_noise -m slow -v`
Expected: PASS. Se fallisce perché stdout contiene rumore, verifica che `_main` (callback) imposti `HF_HUB_DISABLE_PROGRESS_BARS` e `transformers.logging.set_verbosity_error()`.

- [ ] **Step 3: Esegui l'INTERA suite**

Run: `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest -q`
Expected: tutti i test PASS (foundation Parte 1 + helper + CLI).

- [ ] **Step 4: Smoke test manuale dell'entry-point reale**

```bash
cd /Users/alberto/Projects/AI/axyn
.venv/bin/axyn --help
.venv/bin/axyn --json schema
.venv/bin/axyn --json model --info
echo "La capitale dell'Italia è" | .venv/bin/axyn --json generate --prompt - --max-new-tokens 6
```
Expected: ogni comando esce 0; le invocazioni `--json` stampano envelope JSON valido su stdout.

- [ ] **Step 5: Aggiorna `docs/ROADMAP.md`**

Spunta la voce CLI in "Fase 1" (`[ ]` → `[x]`) e annota: comandi `version`/`model --info`/`generate`/`selfcheck`/`schema` implementati, output JSON + exit code, suite verde.

- [ ] **Step 6: Commit**

```bash
cd /Users/alberto/Projects/AI/axyn && git add tests/test_cli.py docs/ROADMAP.md && git commit -m "test(cli): stream JSON puri + suite completa verde; ROADMAP CLI completata"
```

---

## Self-Review (eseguito dall'autore del piano)

**Spec coverage (docs/prd/cli.md):**
- §4 packaging/entry-point → Task 3 ✓
- §5.1 modalità output `--json` + env `AXYN_JSON` → Task 3 (`_json_enabled`, callback) ✓
- §5.2 envelope ok/error → Task 3 (`_emit_ok`/`_fail`) ✓
- §5.3 stream puliti → Task 3 (callback) + Task 7 (test) ✓
- §5.4 exit code (0/1/2) → Task 3/4/5 (`_fail` exit 1; Typer exit 2 per usage) ✓
- §5.5 non-interattiva + stdin → Task 5 (`_read_prompt`) ✓
- §5.6 schema → Task 6 ✓
- §6 comandi version/model/generate/selfcheck/schema → Task 3/4/5/6 ✓
- §7 helper compute_boundaries/model_config_dims → Task 1/2 ✓
- §8 codici errore USAGE_ERROR/INVALID_BOUNDARIES/MODEL_LOAD_FAILED/GENERATION_FAILED → Task 4/5 (mappati in `_fail`; USAGE_ERROR è gestito da Typer con exit 2) ✓
- §9 testing → ogni task ha test; Task 7 stream + suite ✓
- §10 criteri di accettazione → coperti da Task 3-7 + smoke test Task 7 ✓

**Placeholder scan:** nessun TODO/TBD; ogni step ha codice completo.

**Type consistency:** `compute_boundaries(num_layers, n_blocks) -> list[int]` e `model_config_dims(model_id) -> dict` usati coerentemente; envelope `_emit_ok(command, data, human)` / `_fail(command, code, message, exit_code=1)` coerenti in tutti i comandi; il dict `data` di ogni comando combacia con lo schema della PRD §6. Nota risolta nel piano: in `selfcheck` il riferimento è catturato prima di `split_into_blocks` (mutazione `layer_idx`), come da foundation Parte 1.
