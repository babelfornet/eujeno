# Synapse — Roadmap & Stato

> Documento vivo. Traccia **cosa è stato fatto** e **cosa resta da fare**. Aggiornato a ogni passo significativo.
>
> **Ultimo aggiornamento:** 2026-06-17 — Parte 2 completa (P2P puro + coordinator-relay NAT) e **failover automatico** nel coordinator (caduta nodo → re-instradamento su ridondante).

## Legenda

- `[x]` completato · `[~]` in corso · `[ ]` da fare · `⏸` rimandato (deferred)

---

## Decisioni di scope (bloccate)

| Tema | Scelta |
|------|--------|
| Obiettivo prima fase | **PoC eseguibile davvero** su 2–3 nodi reali |
| Incentivi token/crypto | **Rimandati** nel PoC (progettati su carta, non implementati) |
| Ordine di lavoro | **Architettura intera (PRD) prima, poi codice** |
| Modello target iniziale | **Piccolo, 1B–3B** (es. Llama 3.2 1B/3B, Qwen2.5 0.5–1.5B) |
| Runtime | **Python** (HF transformers / PyTorch) |

---

## Fase 0 — Brainstorming & Architettura  `[~]`

- [x] Esplorazione contesto progetto (greenfield)
- [x] Reality-check di scope + decomposizione in Parti 0–5
- [x] Forcelle fondamentali (obiettivo / incentivi / primo modulo / modello)
- [x] Spina dorsale architetturale (Parte 0) — **approvata**
- [x] Scaffolding documentazione (`docs/`)
- [x] **Team di agent**: confronto delle 5 strade implementative contese (workflow `synapse-impl-forks`, 9 agent)
- [x] **[ADR-0001](./decisions/ADR-0001-implementation-forks.md)**: decisioni sulle 5 forcelle implementative
- [x] PRD **[Parte 1](./prd/part-1-peer-node.md)** — Peer Node & Layer Execution
- [x] PRD **[Parte 2](./prd/part-2-discovery-routing.md)** — Discovery & Routing
- [x] PRD **[Parte 3](./prd/part-3-queue-load-balancing.md)** — Queue & Load Balancing
- [x] PRD **[Parte 4](./prd/part-4-incentives-reputation.md)** — Incentivi & Reputazione *(reputazione light + token su carta)*
- [x] PRD **[Parte 5](./prd/part-5-security-bft.md)** — Sicurezza & Byzantine Fault Tolerance *(verifica light + BFT su carta)*
- [~] Spec self-review + review utente delle PRD

## Fase 1 — Implementazione PoC  `[ ]`

> Si parte solo dopo l'approvazione delle PRD. Ogni modulo: plan → build → verify.

- [~] **Peer Node** — [piano](./plans/2026-06-17-part-1-peer-node.md)
  - [x] **Foundation single-process** (build-order step 1-2-4): split del modello in blocchi (EMBED/DECODER/HEAD), `run_block`, KV-cache per-blocco serializzabile, **golden test** (la pipeline distribuita riproduce esattamente `model.generate`), **capstone** (KV-cache sopravvive a round-trip su byte mid-generazione). 12 test verdi su `Qwen2.5-0.5B`.
  - [x] **Partial-loading reale** (`init_empty_weights` + caricamento selettivo dai shard safetensors): un nodo materializza in RAM SOLO i layer assegnati, il resto resta su `meta` (zero memoria). `serve` lo usa. Golden test: 2 nodi parziali == modello intero. → gira su macchine con poca RAM/GPU.
  - [x] **Test 3 nodi (1 host + 2 container Docker)** — [docker/](../docker/) · [quickstart](./examples/docker.md)
  - [x] **Transport di rete** (FastAPI + safetensors) + **orchestrator distribuito** (Milestone 0) — [piano](./plans/2026-06-17-part1-networking.md). Comandi `serve`/`infer`; golden distribuito su 2 nodi reali verde. Topologia statica (la discovery DHT arriva in Parte 2).
- [x] **CLI `synapse`** (AI-native) — entry-point per tutte le operazioni — [PRD](./prd/cli.md) · [piano](./plans/2026-06-17-cli-synapse.md). Comandi a parola singola implementati: `version`, `model --info`, `generate`, `selfcheck`, `schema`; output JSON con envelope stabile, exit code deterministici, prompt da stdin, stream puliti. Suite verde.
- [~] **Discovery & Routing** — due modalità ([ADR-0002](./decisions/ADR-0002-connettivita-nat.md))
  - [x] **P2P puro**: discovery via **gossip** (registry decentralizzato + coverage), `serve --peers/--advertise` + `infer --peer` — [piano](./plans/2026-06-17-part2a-p2p-gossip.md). Nessun server centrale; per LAN/VPN/IP pubblici.
  - [x] **Coordinator-relay** (opt-in, NAT-senza-VPN) — [piano](./plans/2026-06-17-part2-coordinator.md) · [quickstart](./examples/coordinator.md). Nodi via WebSocket in uscita; golden via relay verde. Comandi `coordinator`, `serve --coordinator`, `infer --coordinator`.
  - [x] **Failover automatico** su nodo caduto (coordinator): ridondanza + re-instradamento su holder ridondante — [piano](./plans/2026-06-17-part3-failover.md). e2e: nodo che crasha mid-hop → job completa via ridondante.
  - [ ] Failover per-hop con replay prefisso + store-and-forward durevole SQLite (Parte 3 piena) · failover P2P-diretto · libp2p nativo per P2P-su-NAT (futuro)
- [x] **API OpenAI-compatibile** (`/v1/chat/completions` + `/v1/models`) sul coordinator: chat template + sampling (temperature/top_p/repetition_penalty/seed) — [piano](./plans/2026-06-17-openai-api.md) · [guida agenti](./examples/agents.md). Collega client/agenti OpenAI. *(streaming SSE + Anthropic/LiteLLM per Claude Code = prossimi passi)*
- [ ] Plan + build **Queue & Load Balancing**: job store durevole, store-and-forward, scheduling su holder ridondanti (per molti agenti concorrenti)
- [ ] Plan + build **Reputazione minimale** (token ⏸ rimandati)
- [ ] Integrazione end-to-end su 2–3 nodi + test di failover
- [x] Setup repo GitHub privato → [albertoferrazzoli/synapse](https://github.com/albertoferrazzoli/synapse) (pubblico al primo funzionamento)

### Limiti noti della foundation Parte 1 (da affrontare nei moduli successivi)

- `build_causal_mask` assume **batch=1, no padding, no sliding-window** (corretto per Qwen2.5-0.5B e il PoC single-stream). Batch>1 / left-padding / SWA fuori scope per ora.
- `split_into_blocks` **muta `layer.self_attn.layer_idx` in place**: ok perché ogni nodo reale carica la propria copia del modello; nei test l'isolamento è garantito da `conftest.py` che ripristina gli indici. Da irrobustire: validare che i `boundaries` coprano `[0, num_layers]` in modo contiguo.
- Failover KV-cache: alla morte di un holder mid-generazione si ricomputa il prefisso (O(seq_len)); checkpoint periodico per-blocco rimandato (vedi [ADR-0001](./decisions/ADR-0001-implementation-forks.md) Q3).
- API `run_block`: la foundation la implementa **stateful** (la KV-cache è stato del blocco, costruisce maschera/position_ids internamente e ritorna solo `hidden_states`) — scelta comoda in-process. La [PRD Parte 1](./prd/part-1-peer-node.md) §3 descrive la forma **pura** `(activation, kv) -> (activation, kv)`. Da riconciliare quando arriva il wire transport (Parte 3): o si ripristina la firma pura o si aggiorna la PRD.

## Milestone — "Modello operativo"

Il sistema è **operativo** solo quando **ogni blocco di layer è coperto da ≥1 nodo**. Prima di allora le richieste vengono **accodate**. Criteri di successo del PoC:

1. Un modello 1–3B viene splittato e distribuito su ≥2 nodi.
2. Una domanda utente produce una risposta corretta attraversando la pipeline distribuita.
3. Se un nodo cade durante un job, il traffico viene reindirizzato a un holder ridondante e il job completa.
4. Un nuovo nodo che entra si auto-assegna un blocco scoperto e la coverage si aggiorna.

---

## Backlog / Deferred  `⏸`

- ⏸ Sistema di incentivi a **token/crypto** (ledger on-chain, settlement, proof-of-compute completo)
- ⏸ **BFT completo** (commit-reveal, consenso sugli output, slashing economico)
- ⏸ Modelli **grandi (70B+)** e sharding tensor-parallel avanzato
- ⏸ Reputazione avanzata + sybil resistance economica
- ⏸ Repo **pubblico** + onboarding community
