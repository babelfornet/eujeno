# Test 3 nodi — 1 host + 2 container Docker

Un modello piccolo (Qwen2.5-0.5B, 24 layer) diviso su **3 nodi**: uno sull'host (questa macchina) e due in container Docker. Ogni nodo carica in RAM **solo i suoi layer** (partial loading). Si usa la modalità **coordinator** (i container si connettono in uscita all'host).

Ripartizione dei 24 layer:

| Nodo | Dove | Stage |
|------|------|-------|
| coordinator | host | (instrada, non esegue layer) |
| host-node | host | `embed,decoder:0-8` |
| node-mid | container | `decoder:8-16` |
| node-tail | container | `decoder:16-24,head` |

## Avvio

```bash
# 0) (una volta) scarica il modello sull'host così i container lo riusano dal cache condiviso
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

# 1) Coordinator sull'host (porta 9000, raggiungibile dai container)
synapse coordinator --model Qwen/Qwen2.5-0.5B-Instruct --port 9000 &

# 2) Nodo sull'host: embedding + primi 8 layer
synapse serve --coordinator ws://127.0.0.1:9000/node --stages "embed,decoder:0-8" &

# 3) Due nodi nei container (build automatico la prima volta)
docker compose -f docker/compose.yaml up --build -d

# 4) Aspetta che tutti e 3 i nodi siano registrati e la coverage completa
curl -s http://127.0.0.1:9000/registry        # deve elencare 3 nodi

# 5) Testa un prompt da questa macchina
synapse --json infer --coordinator http://127.0.0.1:9000 --prompt "La capitale dell'Italia è"
```

## Verifica del vantaggio di memoria

```bash
docker stats --no-stream    # RAM dei container: ognuno carica solo i suoi layer, non il modello intero
```

## Stop

```bash
docker compose -f docker/compose.yaml down
kill %1 %2     # coordinator + host-node
```

> I container condividono il cache Hugging Face dell'host (`~/.cache/huggingface` montato su `/hf`), quindi non riscaricano il modello. Ogni nodo materializza in RAM solo i layer assegnati; gli altri restano sul device `meta` (zero memoria).
