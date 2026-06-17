# PRD — CLI `synapse` (AI-native)

> Riferimenti: [ADR-0001](../decisions/ADR-0001-implementation-forks.md), [PRD Parte 1](./part-1-peer-node.md), [piano Parte 1](../plans/2026-06-17-part-1-peer-node.md). Visione: [00-vision-architecture.md](../00-vision-architecture.md).
>
> **Stato:** progettata; non ancora implementata.

## 1. Scopo

Fornire un singolo entry-point eseguibile — il comando `synapse` — per tutte le operazioni del progetto, così da **non dover invocare Python su file singoli**. La CLI è progettata per essere **usabile direttamente da un AI/agente**: output strutturato, stream puliti, non-interattiva, auto-descrittiva.

## 2. In scope / Fuori scope

**In scope:** entry-point `synapse`; framework **Typer**; comandi `version`, `model-info`, `generate`, `selfcheck`, `schema`; modalità output **JSON** vs testo; exit code deterministici; lettura prompt da stdin; due helper riusabili (`compute_boundaries`, `model_config_dims`).

**Fuori scope (deferred):** sottocomandi `node serve/join`, `dht`, qualsiasi networking (Parte 2-3); autenticazione; configurazione persistente su file; quantizzazione/opzioni di runtime avanzate.

## 3. Principi

1. **Layer di presentazione sottile** — nessuna logica di business nella CLI. Ogni comando chiama il codice in `synapse/model/` e formatta l'output. La logica resta riusabile e testabile in isolamento.
2. **AI-native** — il consumatore di riferimento è un agente automatico, non solo un umano (vedi §5).
3. **YAGNI** — si espone solo ciò che il codice attuale già fa; i comandi futuri si appendono allo stesso `app`.

## 4. Packaging & invocazione

- Entry-point in `pyproject.toml`:
  ```toml
  [project.scripts]
  synapse = "synapse.cli:app"
  ```
  Dopo `pip install -e .`, il comando `synapse` è nel PATH.
- Nuova dipendenza: `typer`.
- Modulo: `synapse/cli.py` con l'oggetto `app` (Typer) e una funzione per comando.

## 5. Contratto AI-native (requisiti)

### 5.1 Modalità output
- Flag **globale** `--json / -j` (default: **testo** umano). Quando attivo, ogni comando emette esclusivamente un envelope JSON su stdout.
- Onorato anche via variabile d'ambiente `SYNAPSE_JSON=1` (comodo per agenti che settano l'env una volta).

### 5.2 Envelope JSON (stabile)
Successo:
```json
{ "ok": true, "command": "<nome>", "data": { /* specifico del comando */ } }
```
Errore:
```json
{ "ok": false, "command": "<nome>", "error": { "code": "<CODICE>", "message": "<testo>" } }
```
In modalità JSON, **anche gli errori** sono envelope (`ok:false`) su stdout, con exit code non-zero. L'agente legge sempre e solo `ok` + `data`/`error.code`.

### 5.3 Disciplina degli stream
- In modalità `--json`, **stdout porta SOLO l'envelope JSON** (una riga / un oggetto). Tutto il resto — progress bar di Hugging Face, warning, log di transformers — va su **stderr** o viene silenziato:
  - `HF_HUB_DISABLE_PROGRESS_BARS=1`
  - `transformers.logging.set_verbosity_error()`
- In modalità testo l'output è formattato per umani; i log restano su stderr.

### 5.4 Exit code deterministici
| Code | Significato |
|------|-------------|
| `0` | successo |
| `1` | errore runtime (es. caricamento modello, generazione) |
| `2` | errore d'uso/argomenti (default Typer/Click) |

### 5.5 Non-interattività & input
- Nessun prompt interattivo, mai. Ogni input passa da flag.
- `--prompt -` (oppure prompt assente in `generate`/`selfcheck`) legge il prompt da **stdin**, per pipe da un agente.

### 5.6 Auto-descrizione
- `synapse schema` (rispetta `--json`) stampa l'albero comandi+opzioni come JSON, così un AI scopre le capacità senza parsare l'help umano.
- Restano disponibili gli `--help` standard di Typer per gli umani.

## 6. Comandi & schema `data`

Default condivisi: `--model` default da `config.py` (`Qwen/Qwen2.5-0.5B-Instruct`), `--blocks 2`, `--max-new-tokens 8`.

| Comando | Opzioni | `data` (modalità JSON) |
|---------|---------|------------------------|
| `version` | — | `{ "version": "0.0.1" }` |
| `model-info` | `--model`, `--blocks` | `{ "model": str, "num_layers": int, "hidden_size": int, "num_attention_heads": int, "num_key_value_heads": int, "blocks": int, "boundaries": [int, ...] }` |
| `generate` | `--model`, `--prompt`, `--max-new-tokens`, `--blocks` | `{ "model": str, "prompt": str, "text": str, "tokens": [int, ...] }` |
| `selfcheck` | `--model`, `--prompt`, `--max-new-tokens`, `--blocks` | `{ "model": str, "match": bool, "reference": [int, ...], "pipeline": [int, ...] }` |
| `schema` | — | `{ "commands": [ { "name": str, "help": str, "options": [ { "name", "type", "default", "required" } ] } ] }` |

- `model-info` usa `model_config_dims` (solo `AutoConfig`, **niente pesi**) → veloce, non scarica il modello intero.
- `generate` usa `load_full_model` → `split_into_blocks(compute_boundaries(...))` → `pipeline_generate` → `tokenizer.decode`.
- `selfcheck` esegue `reference_generate` **e** `pipeline_generate` e confronta le liste di token: `match` è il segnale chiave per un AI.

## 7. Helper riusabili (in `synapse/model/`)

- **`compute_boundaries(num_layers: int, n_blocks: int) -> list[int]`** in `blocks.py` — divide i layer in `n_blocks` blocchi contigui il più possibile uguali (es. `24, 2 → [0, 12, 24]`; `24, 5 → [0,5,10,15,20,24]`). Valida gli input: `n_blocks ≥ 1`, `n_blocks ≤ num_layers`; il risultato copre sempre `[0, num_layers]` in modo contiguo e strettamente crescente (chiude il "footgun" segnalato nella review di Parte 1).
- **`model_config_dims(model_id: str) -> dict`** in `loader.py` — legge `AutoConfig.from_pretrained` e ritorna `{num_layers, hidden_size, num_attention_heads, num_key_value_heads}` senza scaricare i pesi.

## 8. Gestione errori (codici stabili)

| `error.code` | Quando | Exit |
|--------------|--------|------|
| `USAGE_ERROR` | argomenti mancanti/invalidi | 2 (Typer) |
| `INVALID_BOUNDARIES` | `--blocks` fuori range vs num_layers | 1 |
| `MODEL_LOAD_FAILED` | download/caricamento modello fallito | 1 |
| `GENERATION_FAILED` | errore durante l'inferenza | 1 |

Un handler top-level cattura le eccezioni note, le mappa a un `code`, e — in modalità JSON — emette l'envelope `ok:false`; in modalità testo stampa un messaggio leggibile su stderr. Le eccezioni inattese diventano `code` generico con exit `1` (mai stack trace su stdout in modalità JSON).

## 9. Testing

- **Unit (veloci):** `compute_boundaries` (split uniforme, casi limite, validazione che solleva su input invalidi). `schema` output è JSON valido con la struttura attesa.
- **Slow (col modello piccolo):** test CLI via `typer.testing.CliRunner`:
  - `version` (veloce, no modello): exit 0, e con `--json` produce envelope valido con `data.version`.
  - `selfcheck --json`: exit 0, **stdout è JSON valido** (parsabile), `data.match == true`.
  - `generate --json`: exit 0, `data.text` non vuoto, `stdout` parsabile.
  - Un caso di errore (`--blocks 999`): exit `1`, envelope `ok:false` con `error.code == "INVALID_BOUNDARIES"`.
- **Disciplina stream:** un test verifica che in `--json` lo stdout sia JSON puro (nessuna barra di progresso / warning mescolati).

## 10. Criteri di accettazione

1. Dopo `pip install -e .`, `synapse --help` funziona e `synapse version --json` stampa `{"ok": true, ...}`.
2. `synapse generate --json --prompt "..."` produce su stdout **solo** JSON valido parsabile, con il testo generato.
3. `synapse selfcheck --json` riporta `match: true` sul modello di default (equivalenza pipeline vs reference).
4. Un errore d'uso o di runtime produce exit code non-zero e — in `--json` — un envelope `ok:false` con un `error.code` stabile.
5. `synapse schema --json` elenca i comandi e le loro opzioni in forma machine-readable.

## 11. Dipendenze

- **Parte 1** (`loader`, `blocks`, `generate`): la CLI ne è consumatrice diretta.
- Nuova dipendenza runtime: `typer`.

## 12. Domande aperte

- Auto-attivazione JSON quando stdout non è una TTY (comodo per agenti) vs solo flag/env esplicito? (Per ora: esplicito, più prevedibile.)
- Esporre `selfcheck` anche come check di salute generico (più prompt, soglie) o tenerlo minimale? (Per ora: minimale.)

## 13. Comandi futuri (placeholder, non implementati)

Allo stesso `app` si appenderanno, in Parte 2-3: `synapse node serve` / `synapse node join` (avvio peer, registrazione blocchi), `synapse dht ...` (ispezione registry/coverage). Erediteranno lo stesso contratto AI-native (envelope JSON, exit code, stream puliti).
