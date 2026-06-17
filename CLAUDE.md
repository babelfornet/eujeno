# CLAUDE.md — guida per agenti AI alla CLI `synapse`

Questo file insegna a un agente (Claude Code o altro) a **pilotare Synapse dalla CLI**. Synapse è una rete di inferenza LLM decentralizzata: un modello è splittato in blocchi di layer (`embed`, `decoder:lo-hi`, `head`) ospitati da nodi diversi; il modello è **operativo** solo quando i blocchi coprono tutto (`embed` + tutti i range decoder + `head`).

> Modello mentale: "BOINC/SETI@home per i layer di un LLM". Inferenza asincrona store-and-forward, tollerante a latenze alte. Ogni nodo è un peer simmetrico.

## Installazione (dopo `git clone`)

```bash
./bin/synapse --help            # bootstrap: crea .venv + installa al primo avvio, poi esegue
```

`bin/synapse` è auto-bootstrap: la prima volta crea `.venv` e fa `pip install -e .`, poi inoltra ogni comando. In alternativa, manuale:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e .
synapse --help
```

## Output AI-native

Ogni comando supporta `--json` (flag globale, va **prima** del comando): emette `{"ok": true|false, "command": "...", "data": {...}}` su stdout. Usa sempre `--json` quando consumi l'output a livello di codice. Senza `--json` l'output è human-readable.

```bash
synapse --json model --info --model Qwen/Qwen2.5-0.5B-Instruct
```

## Comandi chiave

| Comando | A cosa serve |
|---|---|
| `synapse models` | Elenca i modelli/famiglie **compatibili** (Llama/Qwen2) con esempi. |
| `synapse model --info --model <id>` | Dimensioni del modello (num_layers, hidden, ...) + `architecture` + `compatible`. Usalo per **decidere lo split**. |
| `synapse up --model <id> [--dtype bfloat16]` | Bring-up in un comando: avvia coordinator + un nodo che copre tutti i layer. `--dry-run` stampa i comandi senza avviare. |
| `synapse serve --stages "<spec>" ...` | Avvia un nodo che ospita certi blocchi. `--dtype` per modelli grandi. |
| `synapse coordinator --port 9000` | Avvia un coordinator (relay per nodi dietro NAT). |
| `synapse infer --coordinator <url> --prompt "..."` | Inferenza one-shot sulla rete. `--peer <url>` in P2P puro. |
| `synapse ui --coordinator <url>` | Dashboard locale (stato rete, chat, MCP). |
| `synapse mcp --add <name> --command <cmd> --args "..."` | Configura server MCP; `synapse infer --mcp` li usa nel loop di tool-calling. |
| `synapse selfcheck` | Verifica ambiente/modello. |
| `synapse schema` | Schema macchina-leggibile di tutti i comandi/flag. |

## Quali modelli posso usare?

```bash
synapse --json models                                  # lista curata (Llama/Qwen2)
synapse --json model --info --model <id>               # controlla compatible:true e num_layers
```

Compatibili: architetture **decoder-only Llama/Qwen2**. Esempi: `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B,32B,72B}-Instruct`, `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`.

## Decidere lo split (layer ↔ RAM)

Ogni nodo carica **solo i layer assegnati** (partial loading): la RAM richiesta è ~proporzionale al numero di layer ospitati, non all'intero modello. Stima rapida della RAM per blocco:

```
bytes_per_param = 4 (float32) | 2 (bfloat16/float16)
ram_layer ≈ params_per_layer × bytes_per_param
ram_nodo  ≈ Σ ram_layer dei layer ospitati (+ embed/head se assegnati)
```

`synapse model --info` dà `num_layers` e `hidden_size` per ricavare `params_per_layer`. Per modelli grandi usa `--dtype bfloat16` (dimezza la RAM) e/o splitta su più nodi. Coverage completa = `embed` + tutti i range `decoder:0-N` contigui + `head`.

## Workflow tipici

**a) Configura il mio nodo per un modello e avvia tutto (single-box):**
```bash
synapse models                                         # scegli un modello compatibile
synapse up --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
```

**b) Unisciti a una rete esistente con i miei layer:**
```bash
synapse serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head" \
  --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
```

**c) Interroga il modello distribuito:**
```bash
synapse --json infer --coordinator http://IP:9000 --prompt "Spiega la fotosintesi"
```

**d) Avvia il frontend:**
```bash
synapse ui --coordinator http://IP:9000      # poi apri http://127.0.0.1:8500
```

## Note operative
- **Memoria:** un 7B in float32 ≈ 28GB; in bfloat16 ≈ 14GB. Splitta su più nodi o usa `--dtype bfloat16`.
- **NAT senza VPN:** usa la modalità coordinator (i nodi si connettono in uscita). In LAN/VPN/IP pubblici va bene il P2P puro (`--peer`).
- **Operatività:** finché la coverage non è completa, `infer` risponde `NOT_OPERATIONAL`. Aggiungi nodi con i range mancanti.
- **Modelli OpenAI/Anthropic client:** il coordinator espone `/v1/chat/completions` (OpenAI). Per Claude Code metti **LiteLLM** davanti (vedi `docs/examples/agents.md`).
