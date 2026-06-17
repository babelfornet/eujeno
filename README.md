# Synapse

**Rete di inferenza per LLM completamente decentralizzata e peer-to-peer.** Nessun server centrale: ogni nodo è un peer simmetrico che ospita ed esegue uno o più *blocchi* di layer di un modello open-source (da Hugging Face). Le domande degli utenti vengono instradate come **job durevoli** attraverso la rete di nodi responsabili dei vari blocchi.

> **Idea guida:** Synapse non è "Petals in tempo reale". È **"BOINC / SETI@home per i layer di un LLM"** — tollera latenze altissime (ore, giorni, settimane) e tratta l'inferenza come un job asincrono che avanza hop-by-hop in *store-and-forward*. Questa rinuncia al real-time rende failover e accodamento più semplici, non più difficili.

## Stato

🚧 **PoC in costruzione.** Funziona già l'inferenza distribuita su più nodi via HTTP (orchestrator-driven, Milestone 0): un modello viene splittato in blocchi di layer ospitati da `synapse serve` su nodi diversi, e `synapse infer` esegue la generazione attraversando la rete — riproducendo esattamente il modello intero. **Prossimi passi:** discovery DHT (auto-organizzazione dei nodi), queue/store-and-forward durevole con failover. Incentivi a token rimandati (progettati su carta).

**Obiettivo del primo PoC:** inferenza distribuita di un modello **1–3B** su **2–3 nodi reali**, con discovery DHT, queue asincrona e failover automatico.

## Quickstart multi-nodo (PoC)

Inferenza distribuita di un modello su 2 nodi (qui in localhost; su LAN sostituisci gli IP nel file topologia).

```bash
pip install -e .

# Nodo A (serve embedding + primi 12 layer)
synapse serve --stages "embed,decoder:0-12" --port 8001

# Nodo B (serve gli ultimi 12 layer + la testa) — altro terminale/macchina
synapse serve --stages "decoder:12-24,head" --port 8002

# Entry: esegue l'inferenza attraverso i due nodi
synapse --json infer --topology docs/examples/topology.localhost.json --prompt "La capitale dell'Italia è"
```

Su 3 macchine: avvia un `synapse serve` per nodo con range di layer diversi, copia `docs/examples/topology.localhost.json` mettendo gli **IP:porta reali** di ogni nodo, e lancia `synapse infer` puntando a quel file. Le macchine devono raggiungersi sulla rete (LAN/VPN) e scaricano il modello da Hugging Face al primo avvio.

## Documentazione

Tutto in [`docs/`](./docs/):

- **[docs/README.md](./docs/README.md)** — indice e mappa dei documenti
- **[docs/ROADMAP.md](./docs/ROADMAP.md)** — stato del progetto, milestone, backlog
- **[docs/00-vision-architecture.md](./docs/00-vision-architecture.md)** — visione e architettura completa (diagrammi)
- **[docs/decisions/](./docs/decisions/)** — Architecture Decision Records
- **[docs/prd/](./docs/prd/)** — PRD per i 5 sottosistemi

## Architettura in breve

Ogni nodo simmetrico esegue: **Layer Executor** (blocchi transformer su HF/PyTorch) · **DHT Agent** (discovery: chi-ha-quale-blocco) · **Router** (instradamento + failover) · **Job Store durevole** (SQLite + blob, store-and-forward) · **Allocator** (auto-assegnazione blocchi scoperti) · **Reputation/Verifier** (light).

Il modello diventa **operativo** solo quando ogni blocco è coperto da ≥1 nodo; prima le richieste si accodano. Man mano che i nodi si aggiungono, il modello **si compone progressivamente** nella rete.

## Stack (PoC)

Python · Hugging Face `transformers` + `accelerate` + `safetensors` · `hivemind.DHT` (discovery) · SQLite (job store durevole) · FastAPI/uvicorn (transport attivazioni). Dettagli e motivazioni in [ADR-0001](./docs/decisions/ADR-0001-implementation-forks.md).

## Licenza

Da definire prima della pubblicazione.
