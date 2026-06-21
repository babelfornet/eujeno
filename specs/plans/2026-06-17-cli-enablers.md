# CLI enabler: --dtype, models, up, CLAUDE.md — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Make the use case "an AI (Claude Code) configures/uses the node from the CLI" smooth: `--dtype` (large models in bf16), `eujeno models` + compatibility check (knowing which models to use), `eujeno up` (one-command bring-up), and a `CLAUDE.md` that teaches an agent to drive the CLI.

**Architecture:** `parse_dtype` in `config.py`; `--dtype` on `serve`/`generate`/`selfcheck`. `eujeno models` (curated list) + `compatible` field in `model --info` (recognizes the `qwen2`/`llama` architecture). `eujeno up --model X [--dtype]` starts a coordinator + a serve node that covers all layers (subprocesses), waits for operationality and prints the endpoints (`--dry-run` for testability/preview). `CLAUDE.md` + Getting started section in the README.

**Tech Stack:** Python · Typer · the existing `eujeno/{config,cli}.py`, `model_config_dims`, `NodeManager`.

---

## File Structure
```
eujeno/config.py               # MOD: parse_dtype + SUPPORTED_ARCHS
eujeno/cli.py                  # MOD: --dtype (serve/generate/selfcheck), models, model --info compatible, up
tests/test_dtype.py             # NEW: parse_dtype (fast)
tests/test_cli_models.py        # NEW: models + up --dry-run (fast)
CLAUDE.md                       # NEW
README.md                       # MOD: Getting started
docs/examples/agents.md         # MOD (models/up)
```

---

## Task 1: `--dtype` (bf16/fp16 for large models)

**Files:** modify `eujeno/config.py`, `eujeno/cli.py`; create `tests/test_dtype.py`.

- [ ] **Step 1: test `tests/test_dtype.py`**
```python
import torch
import pytest
from eujeno.config import parse_dtype


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

- [ ] **Step 3: in `eujeno/config.py`** add:
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
        raise ValueError(f"invalid dtype: {name!r} (use float32/bfloat16/float16)")
    return _DTYPES[key]
```

- [ ] **Step 4: in `eujeno/cli.py` add `--dtype` to the `serve` command** — option and usage. Add to the `serve` signature:
```python
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16 (bf16 for large models)"),
```
and where `serve` loads the model (`load_partial_model(model_id, spec, DTYPE, DEVICE)`), replace `DTYPE` with the chosen dtype:
```python
    from eujeno.config import parse_dtype
    try:
        _dtype = parse_dtype(dtype)
    except ValueError as e:
        _fail("serve", "USAGE_ERROR", str(e), exit_code=2)
    ...
        model, tokenizer = load_partial_model(model_id, spec, _dtype, DEVICE)
```
(Make sure `parse_dtype` is imported; the `"float32"` default keeps the current behavior.)

- [ ] **Step 5: run PASS** — `... pytest tests/test_dtype.py -v` → 2 passed. Verify the CLI imports: `.venv/bin/eujeno serve --help | grep -q dtype && echo ok`.

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/config.py eujeno/cli.py tests/test_dtype.py && git commit -m "feat(cli): --dtype (bf16/fp16) on serve for large models"
```

---

## Task 2: `eujeno models` + compatibility check in `model --info`

**Files:** modify `eujeno/cli.py`; create `tests/test_cli_models.py`.

- [ ] **Step 1: test `tests/test_cli_models.py`**
```python
import json
from typer.testing import CliRunner
from eujeno.cli import app

runner = CliRunner()


def test_models_lists_examples():
    r = runner.invoke(app, ["--json", "models"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)["data"]
    assert "qwen2" in data["supported_architectures"]
    assert any("Qwen2.5" in m for m in data["examples"])
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_cli_models.py -v` → no `models` command.

- [ ] **Step 3: in `eujeno/cli.py` add the `models` command** (after `model`):
```python
@app.command()
def models():
    """Lists the compatible models/families (Llama/Qwen2 architecture)."""
    from eujeno.config import SUPPORTED_ARCHS
    examples = [
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
    ]
    data = {"supported_architectures": sorted(SUPPORTED_ARCHS), "examples": examples,
            "note": "Llama/Qwen2 architecture (decoder-only). Check with 'eujeno model --info --model <id>'. "
                    "For large models use --dtype bfloat16 and/or split across more nodes."}
    human = "Compatible models (Llama/Qwen2):\n" + "\n".join(f"  - {m}" for m in examples)
    _emit_ok("models", data, human=human)
```
Also, in the `model` command (`--info`) add the compatibility fields. Where it builds `data` with the model dims, add `architecture` and `compatible`:
```python
        from eujeno.config import SUPPORTED_ARCHS
        arch = getattr(__import__("transformers").AutoConfig.from_pretrained(model_id), "model_type", "?")
        data["architecture"] = arch
        data["compatible"] = arch in SUPPORTED_ARCHS
```
> Note: `model --info` already loads `AutoConfig` via `model_config_dims`. If you prefer to avoid a second `from_pretrained`, make `model_config_dims` return `model_type` (add `"model_type": cfg.model_type` to the dict in `loader.py`) and use it here. Choose the cleaner option; what matters is that `model --info --json` includes `architecture` and `compatible`.

- [ ] **Step 4: run PASS** — `... pytest tests/test_cli_models.py -v` → PASS. Verify `.venv/bin/eujeno --help | grep -q models`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py eujeno/model/loader.py tests/test_cli_models.py && git commit -m "feat(cli): 'models' command + compatibility check in 'model --info'"
```

---

## Task 3: `eujeno up` (one-command bring-up)

**Files:** modify `eujeno/cli.py`; modify `tests/test_cli_models.py` (append).

- [ ] **Step 1: append to `tests/test_cli_models.py`**
```python
def test_up_dry_run_prints_commands(monkeypatch):
    # avoid the config download: stub model_config_dims
    import eujeno.cli as cli
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

- [ ] **Step 3: in `eujeno/cli.py` add the `up` command** (after `coordinator`):
```python
@app.command()
def up(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model", help="Hugging Face model ID"),
    dtype: str = typer.Option("float32", "--dtype", help="float32 | bfloat16 | float16"),
    host: str = typer.Option("127.0.0.1", "--host", help="Coordinator host"),
    port: int = typer.Option(9000, "--port", help="Coordinator port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the commands without starting anything"),
):
    """Starts an operational single-node network in one shot (coordinator + a node that covers the whole model)."""
    nl = model_config_dims(model_id)["num_layers"]
    coord_cmd = [sys.executable, "-m", "eujeno", "coordinator", "--model", model_id,
                 "--host", host, "--port", str(port)]
    ws = f"ws://{host}:{port}/node"
    serve_cmd = [sys.executable, "-m", "eujeno", "serve", "--coordinator", ws,
                 "--stages", f"embed,decoder:0-{nl},head", "--model", model_id, "--dtype", dtype]
    if dry_run:
        _emit_ok("up", {"commands": [coord_cmd, serve_cmd],
                        "coordinator_url": f"http://{host}:{port}"},
                 human="commands:\n  " + "\n  ".join(" ".join(c) for c in [coord_cmd, serve_cmd]))
        return
    import subprocess
    import time as _t
    import httpx
    procs = [subprocess.Popen(coord_cmd)]
    _t.sleep(4)
    procs.append(subprocess.Popen(serve_cmd))
    base = f"http://{host}:{port}"
    typer.echo(f"eujeno up: starting network for {model_id} (dtype={dtype})…", err=True)
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
        typer.echo(f"READY. Query: eujeno infer --coordinator {base} --prompt \"...\"", err=True)
        typer.echo(f"Frontend:     eujeno ui --coordinator {base}", err=True)
    else:
        typer.echo("network not operational yet (check the logs).", err=True)
    try:
        procs[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_cli_models.py -v` → PASS. Verify `.venv/bin/eujeno --help | grep -q up`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py tests/test_cli_models.py && git commit -m "feat(cli): 'up' command (one-command single-node network bring-up)"
```

---

## Task 4: CLAUDE.md + Getting started + docs + suite

**Files:** create `CLAUDE.md`; modify `README.md`, `docs/examples/agents.md`. (Done by the controller, not a subagent.)

- [ ] **Step 1:** create `CLAUDE.md` (guide for an agent: what Eujeno is, install, key commands with examples — model/models/up/serve/coordinator/infer/ui/mcp — supported models, typical workflows a/b/c/d, dtype/memory note).
- [ ] **Step 2:** add an "## Installation / Getting started" section at the top of the `README.md` (clone → venv → `pip install -e .` → `eujeno --help`; `eujeno up` quickstart).
- [ ] **Step 3:** in `docs/examples/agents.md` mention `eujeno models` and `eujeno up`.
- [ ] **Step 4:** full suite `... pytest -q -p no:warnings` → green.
- [ ] **Step 5:** commit `docs: CLAUDE.md + Getting started (install + up + models)`.

---

## Self-Review

**Coverage:** `--dtype` (Task 1) ✓; `models` + `model --info` compatibility (Task 2) ✓; `up` bring-up (Task 3) ✓; CLAUDE.md + Getting started (Task 4) ✓. Answers "how do I launch the CLI" (README install) and "which models" (models + compatible).

**Placeholder scan:** complete code; CLAUDE.md written in Task 4.

**Type consistency:** `parse_dtype(str)->torch.dtype`; `serve --dtype` uses `load_partial_model(..., _dtype, ...)`; `models` returns `{supported_architectures, examples, note}`; `model --info` adds `architecture`/`compatible`; `up` builds `python -m eujeno coordinator|serve` commands with stage `embed,decoder:0-N,head` and `--dtype`.
```
