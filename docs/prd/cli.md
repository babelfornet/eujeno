# PRD — `axyn` CLI (AI-native)

> References: [ADR-0001](../decisions/ADR-0001-implementation-forks.md), [PRD Part 1](./part-1-peer-node.md), [Part 1 plan](../plans/2026-06-17-part-1-peer-node.md). Vision: [00-vision-architecture.md](../00-vision-architecture.md).
>
> **Status:** designed; not yet implemented.

## 1. Purpose

Provide a single executable entry point — the `axyn` command — for all project operations, so there is **no need to invoke Python on individual files**. The CLI is designed to be **usable directly by an AI/agent**: structured output, clean streams, non-interactive, self-describing.

## 2. In scope / Out of scope

**In scope:** `axyn` entry point; **Typer** framework; `version`, `model` (with `--info`), `generate`, `selfcheck`, `schema` commands; **JSON** vs text output modes; deterministic exit codes; reading the prompt from stdin; two reusable helpers (`compute_boundaries`, `model_config_dims`).

**Out of scope (deferred):** `node serve/join`, `dht` subcommands, any networking (Parts 2-3); authentication; persistent file-based configuration; quantization/advanced runtime options.

## 3. Principles

1. **Thin presentation layer** — no business logic in the CLI. Each command calls the code in `axyn/model/` and formats the output. The logic stays reusable and testable in isolation.
2. **AI-native** — the reference consumer is an automated agent, not just a human (see §5).
3. **YAGNI** — only expose what the current code already does; future commands are appended to the same `app`.
4. **Single-word commands** — wherever possible each command is a single word (`version`, `model`, `generate`, `selfcheck`, `schema`); variants/actions are expressed with **switches**, not with compound names or nested subcommands. E.g.: `axyn model --info` (not `model-info`). This keeps the surface flat, predictable, and easy for an agent to compose.

## 4. Packaging & invocation

- Entry point in `pyproject.toml`:
  ```toml
  [project.scripts]
  axyn = "axyn.cli:app"
  ```
  After `pip install -e .`, the `axyn` command is on the PATH.
- New dependency: `typer`.
- Module: `axyn/cli.py` with the `app` object (Typer) and one function per command.

## 5. AI-native contract (requirements)

### 5.1 Output mode
- **Global** flag `--json / -j` (default: human **text**). When active, every command emits exclusively a JSON envelope on stdout.
- Also honored via the `AXYN_JSON=1` environment variable (convenient for agents that set the env once).

### 5.2 JSON envelope (stable)
Success:
```json
{ "ok": true, "command": "<name>", "data": { /* command-specific */ } }
```
Error:
```json
{ "ok": false, "command": "<name>", "error": { "code": "<CODE>", "message": "<text>" } }
```
In JSON mode, **errors too** are envelopes (`ok:false`) on stdout, with a non-zero exit code. The agent always reads only `ok` + `data`/`error.code`.

### 5.3 Stream discipline
- In `--json` mode, **stdout carries ONLY the JSON envelope** (one line / one object). Everything else — Hugging Face progress bars, warnings, transformers logs — goes to **stderr** or is silenced:
  - `HF_HUB_DISABLE_PROGRESS_BARS=1`
  - `transformers.logging.set_verbosity_error()`
- In text mode the output is formatted for humans; logs stay on stderr.

### 5.4 Deterministic exit codes
| Code | Meaning |
|------|---------|
| `0` | success |
| `1` | runtime error (e.g. model loading, generation) |
| `2` | usage/argument error (Typer/Click default) |

### 5.5 Non-interactivity & input
- No interactive prompts, ever. Every input is passed via a flag.
- `--prompt -` (or an absent prompt in `generate`/`selfcheck`) reads the prompt from **stdin**, for piping from an agent.

### 5.6 Self-description
- `axyn schema` (respects `--json`) prints the command+option tree as JSON, so an AI can discover the capabilities without parsing the human help.
- The standard Typer `--help` options remain available for humans.

## 6. Commands & `data` schema

Shared defaults: `--model` defaults from `config.py` (`Qwen/Qwen2.5-0.5B-Instruct`), `--blocks 2`, `--max-new-tokens 8`.

| Command | Options | `data` (JSON mode) |
|---------|---------|--------------------|
| `version` | — | `{ "version": "0.0.1" }` |
| `model` | `--info`, `--model`, `--blocks` | `{ "model": str, "num_layers": int, "hidden_size": int, "num_attention_heads": int, "num_key_value_heads": int, "blocks": int, "boundaries": [int, ...] }` |
| `generate` | `--model`, `--prompt`, `--max-new-tokens`, `--blocks` | `{ "model": str, "prompt": str, "text": str, "tokens": [int, ...] }` |
| `selfcheck` | `--model`, `--prompt`, `--max-new-tokens`, `--blocks` | `{ "model": str, "match": bool, "reference": [int, ...], "pipeline": [int, ...] }` |
| `schema` | — | `{ "commands": [ { "name": str, "help": str, "options": [ { "name", "type", "default", "required" } ] } ] }` |

- `model` exposes the model operations via action switches. In v1 the only action is **`--info`** (dims + proposed split); if no action switch is passed, `model` runs `--info` by default. Future actions (e.g. `--download`) will be new switches on the same command.
- `model --info` uses `model_config_dims` (only `AutoConfig`, **no weights**) → fast, does not download the whole model.
- `generate` uses `load_full_model` → `split_into_blocks(compute_boundaries(...))` → `pipeline_generate` → `tokenizer.decode`.
- `selfcheck` runs `reference_generate` **and** `pipeline_generate` and compares the token lists: `match` is the key signal for an AI.

## 7. Reusable helpers (in `axyn/model/`)

- **`compute_boundaries(num_layers: int, n_blocks: int) -> list[int]`** in `blocks.py` — splits the layers into `n_blocks` contiguous blocks that are as equal as possible (e.g. `24, 2 → [0, 12, 24]`; `24, 5 → [0,5,10,15,20,24]`). Validates the inputs: `n_blocks ≥ 1`, `n_blocks ≤ num_layers`; the result always covers `[0, num_layers]` contiguously and strictly increasing (closes the "footgun" flagged in the Part 1 review).
- **`model_config_dims(model_id: str) -> dict`** in `loader.py` — reads `AutoConfig.from_pretrained` and returns `{num_layers, hidden_size, num_attention_heads, num_key_value_heads}` without downloading the weights.

## 8. Error handling (stable codes)

| `error.code` | When | Exit |
|--------------|------|------|
| `USAGE_ERROR` | missing/invalid arguments | 2 (Typer) |
| `INVALID_BOUNDARIES` | `--blocks` out of range vs num_layers | 1 |
| `MODEL_LOAD_FAILED` | model download/loading failed | 1 |
| `GENERATION_FAILED` | error during inference | 1 |

A top-level handler catches the known exceptions, maps them to a `code`, and — in JSON mode — emits the `ok:false` envelope; in text mode it prints a readable message on stderr. Unexpected exceptions become a generic `code` with exit `1` (never a stack trace on stdout in JSON mode).

## 9. Testing

- **Unit (fast):** `compute_boundaries` (uniform split, edge cases, validation that raises on invalid inputs). The `schema` output is valid JSON with the expected structure.
- **Slow (with the small model):** CLI tests via `typer.testing.CliRunner`:
  - `version` (fast, no model): exit 0, and with `--json` produces a valid envelope with `data.version`.
  - `selfcheck --json`: exit 0, **stdout is valid JSON** (parsable), `data.match == true`.
  - `generate --json`: exit 0, `data.text` non-empty, `stdout` parsable.
  - An error case (`--blocks 999`): exit `1`, `ok:false` envelope with `error.code == "INVALID_BOUNDARIES"`.
- **Stream discipline:** a test verifies that in `--json` mode stdout is pure JSON (no progress bars / warnings mixed in).

## 10. Acceptance criteria

1. After `pip install -e .`, `axyn --help` works and `axyn version --json` prints `{"ok": true, ...}`.
2. `axyn generate --json --prompt "..."` produces on stdout **only** valid parsable JSON, with the generated text.
3. `axyn selfcheck --json` reports `match: true` on the default model (pipeline vs reference equivalence).
4. A usage or runtime error produces a non-zero exit code and — in `--json` mode — an `ok:false` envelope with a stable `error.code`.
5. `axyn schema --json` lists the commands and their options in machine-readable form.

## 11. Dependencies

- **Part 1** (`loader`, `blocks`, `generate`): the CLI is a direct consumer of it.
- New runtime dependency: `typer`.

## 12. Open questions

- Auto-enable JSON when stdout is not a TTY (convenient for agents) vs. only an explicit flag/env? (For now: explicit, more predictable.)
- Expose `selfcheck` also as a generic health check (more prompts, thresholds) or keep it minimal? (For now: minimal.)

## 13. Future commands (placeholder, not implemented)

In Parts 2-3, single-word commands with switches (consistent with principle §3.4) will be appended to the same `app`: `axyn serve` (start peer + register blocks), `axyn join` (join the network + self-assignment), `axyn dht` with action switches (`--coverage`, `--peers`) to inspect the registry and coverage. They will inherit the same AI-native contract (JSON envelope, exit codes, clean streams).
