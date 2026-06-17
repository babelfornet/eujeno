# Synapse — Documentazione

**Synapse** è una rete di inferenza per Large Language Model **completamente decentralizzata e peer-to-peer**: nessun server centrale, ogni nodo è un peer simmetrico che ospita ed esegue uno o più *blocchi* di layer di un modello open-source (scaricato da Hugging Face). Le domande degli utenti vengono instradate come **job durevoli** attraverso la rete di nodi responsabili dei vari blocchi.

> **Idea guida:** Synapse non è "Petals in tempo reale". È **"BOINC / SETI@home per i layer di un LLM"** — tollera latenze altissime (ore, giorni, settimane) e tratta l'inferenza come un job asincrono che avanza hop-by-hop in *store-and-forward*.

## Mappa dei documenti

| Documento | Contenuto |
|-----------|-----------|
| [ROADMAP.md](./ROADMAP.md) | **Stato del progetto**: passi eseguiti / da eseguire, milestone, backlog. Punto di partenza per capire "a che punto siamo". |
| [00-vision-architecture.md](./00-vision-architecture.md) | **Parte 0** — Visione, obiettivi, principi, mappa dei componenti, flusso dati. Lo scheletro architetturale approvato. |
| [decisions/](./decisions/) | **ADR** (Architecture Decision Records): decisioni tecniche motivate (es. quale substrate P2P, quale runtime). |
| [prd/](./prd/) | **PRD** per ciascun sottosistema (Parti 1–5): Peer Node, Discovery & Routing, Queue & Load Balancing, Incentivi & Reputazione, Sicurezza & BFT. Più la **[CLI `synapse`](./prd/cli.md)** (AI-native), entry-point per tutte le operazioni. |

## Come è organizzato il lavoro

Il problema è grande, quindi è **compartimentalizzato** in parti indipendenti. Ogni parte segue il ciclo:

```
spec (PRD)  →  plan (implementazione)  →  build (codice)  →  verify
```

La **Parte 0** è il documento ombrello che tiene insieme tutto. Le **Parti 1–5** sono i sottosistemi. Le **decisioni trasversali** (stack, librerie) vivono negli ADR.

## Stato sintetico

- **Fase corrente:** Architettura & Design (le PRD vengono scritte prima del codice).
- **Obiettivo del primo PoC:** inferenza distribuita di un modello **1–3B** su **2–3 nodi reali**, con discovery DHT, queue asincrona e failover. **Token rimandati.**
- Dettaglio aggiornato sempre in [ROADMAP.md](./ROADMAP.md).
