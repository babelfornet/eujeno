# Synapse — Roadmap & Stato

> Documento vivo. Traccia **cosa è stato fatto** e **cosa resta da fare**. Aggiornato a ogni passo significativo.
>
> **Ultimo aggiornamento:** 2026-06-17 — completate Fase 0 architettura: ADR-0001 + PRD Parti 1-5.

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

- [ ] Plan + build **Peer Node**: download modello HF, sharding in blocchi, esecuzione layer, RPC inferenza
- [ ] Plan + build **Discovery & Routing**: registry DHT, allocazione dinamica blocchi, failover
- [ ] Plan + build **Queue & Load Balancing**: job store durevole, store-and-forward, scheduling su holder ridondanti
- [ ] Plan + build **Reputazione minimale** (token ⏸ rimandati)
- [ ] Integrazione end-to-end su 2–3 nodi + test di failover
- [ ] Setup repo GitHub (privato → pubblico al primo funzionamento)

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
