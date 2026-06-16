# ADR-0001 — Strade implementative per il PoC

- **Stato:** Accettato
- **Data:** 2026-06-17
- **Decisori:** team di agent `synapse-impl-forks` (9 agent: 5 specialisti per-forcella, 3 architetti di stack, 1 lead architect di sintesi) + revisione utente
- **Contesto:** [00-vision-architecture.md](../00-vision-architecture.md)

## Contesto

Cinque decisioni implementative trasversali ("forcelle") gating l'intera architettura erano contese. Un team di agent le ha confrontate in profondità (con verifica web sullo stato delle librerie a giugno 2026) e ha prodotto una raccomandazione integrata. Questo ADR cristallizza l'esito.

I criteri di valutazione, in ordine di peso: (1) time-to-runnable-PoC, (2) correttezza dell'inferenza autoregressiva distribuita + KV-cache, (3) allineamento col framing async/store-and-forward, (4) estensibilità verso le parti rimandate, (5) rischio operativo / maturità librerie.

## Decisioni per forcella

| # | Forcella | Decisione | Alternative scartate |
|---|----------|-----------|----------------------|
| **A** | Substrate P2P/DHT | **`hivemind.DHT` come piano discovery/metadati SOLO**, dietro interfaccia `DiscoveryProvider`. `bmuller/kademlia` vendored come fallback LAN/VPN. **Mai** instradare attivazioni via hivemind RPC/streaming. | `kademlia` come primario (UDP-only, no NAT traversal, dormiente dal 2021); py-libp2p (kad-DHT immaturo nel 2026); go-libp2p sidecar (1-2 settimane di bridge IPC, fase sbagliata) |
| **B** | Runtime esecuzione layer | **Block-runner sottile su HF transformers**: `init_empty_weights()` + `load_checkpoint_in_model()` per materializzare solo i layer assegnati `model.model.layers[i:j]`. Embedding e lm_head sono blocchi anch'essi. KV-cache = `DynamicCache` serializzabile che possediamo, persistita per `(job_id, stage)`. | Riuso interni Petals (frozen a 2.2.0, no Llama 3.2/Qwen2.5, KV-cache saldata a sessione hivemind live = il modello low-latency che abbiamo rilassato); forward pass custom (re-derivare ogni architettura, fase sbagliata) |
| **C** | Modello job async | **Store-and-forward come nord; entry-node orchestrator-driven come Milestone 0**, entrambi scrivono sullo stesso substrato durevole: **SQLite (WAL) job log per-nodo + blob safetensors su disco** keyed `(job_id, stage)`. Hop idempotenti (ACK-after-persist, dedup su `(job_id, stage)`). Migrazione M0→peer-driven = cancellare il loop centrale, non riscrivere la persistenza. | Orchestrator-driven come design finale (entry node = SPOF multi-day); Temporal/Ray/Celery (broker centrale = SPOF, contro la tesi decentralizzata); Redis Streams (daemon extra per nessun beneficio PoC) |
| **D** | Verifica / BFT | **Reputazione always-on** (campo `reputation` nel record DHT) **+ recompute ridondante campionato ~5-10%** (biased verso nodi nuovi/low-score), confronto attivazioni promosse a **fp32 con tolleranza** (`torch.allclose` atol~1e-2 / rtol~1e-3). **Mai hash-compare.** Verifica solo hop stateless/prefill. | Verifica di ogni stage (~2x compute, kill del vantaggio async); commit-reveal di hash (fatalmente incompatibile con non-determinismo FP tra hardware eterogeneo — nodi onesti producono byte diversi) |
| **E** | Allocazione / coverage | **Conteggio diretto chiavi DHT** per il PoC: `coverage = all(DHT.get(block_i) restituisce ≥1 holder vivo)`. Self-assign di un blocco scoperto random (o least-replicated) con backoff jitterato. Cache locale TTL 2-5s sul hot path. Mappa CRDT via gossip = upgrade v1.1. | Mappa gossip-CRDT come primario (over-engineered per 2-3 nodi); coordinatore eletto/Raft (reintroduce un punto centrale, viola la simmetria dei peer) |

## Stack integrato — i 3 primitivi condivisi

Le cinque scelte compongono **un unico seam** attorno a tre primitivi che tutto il sistema condivide:

1. **Schema record DHT** — `block:{lo-hi} → {peer_id, queue_url, block, expiry, load, reputation}` con TTL ~60s. Letto/scritto da A (discovery), D (reputation), E (coverage).
2. **Substrato durevole SQLite + safetensors** — job log per-nodo + blob attivazioni/KV su disco. Condiviso da C (store-and-forward) e dal failover-and-verify di D.
3. **Chiave di idempotenza `(job_id, stage)`** — allinea hop, KV-cache, re-dispatch e verifica su un unico code path.

> **Il taglio architetturale decisivo:** separare i due contributi di Petals. Riusiamo l'*idea* di eseguire blocchi via moduli decoder HF (Fork B) ma **scartiamo** l'RPC/streaming sincrono low-latency di Petals — esattamente la proprietà che abbiamo rilassato. hivemind serve solo i metadati; le attivazioni viaggiano sul **nostro** transport durevole.

```mermaid
graph LR
    subgraph Shared["3 primitivi condivisi"]
        REC["DHT record schema<br/>block:{lo-hi} → {...}"]
        SUB["SQLite + safetensors<br/>substrato durevole"]
        KEY["(job_id, stage)<br/>idempotency key"]
    end
    A[Fork A: discovery] --> REC
    E[Fork E: coverage] --> REC
    D[Fork D: reputation] --> REC
    C[Fork C: store-and-forward] --> SUB
    D --> SUB
    C --> KEY
    B[Fork B: run_block] --> KEY
    D --> KEY
```

## Sequencing — Milestone 0 → decentralizzazione

Si compra velocità (criterio 1) senza sacrificare l'allineamento async (criterio 3): si spedisce **prima** un entry-node orchestrator-driven che scrive sullo **stesso** substrato durevole, **poi** si decentralizza cancellando il loop centrale. Nessun broker, nessun coordinatore, nessuna dipendenza da Petals.

## Build order (prototype-first)

Questi step diventano i milestone di implementazione (vedi [ROADMAP](../ROADMAP.md)):

1. **Golden reference single-process** — carica un modello piccolo (Qwen2.5-0.5B o Llama 3.2 1B), genera, cattura logits/token di riferimento per un prompt fisso.
2. **Block-split manuale single-process** — split in 2-3 block-runner in-process chiamati in sequenza con `DynamicCache` passata a mano; assert `torch.allclose` vs step 1. **De-risk di KV-cache/RoPE/position_ids prima di qualsiasi networking.**
3. **Due processi reali su localhost** — FastAPI + safetensors in-memory, routing statico, single forward, attivazione persistita su SQLite+disco; ri-assert equality vs golden.
4. **Loop autoregressivo con session affinity** — KV-cache pinnata per `(job_id, block)`; il messaggio porta solo l'hidden state del nuovo token. Ri-assert sequenza == golden.
5. **Smoke test p2pd/hivemind.DHT** sui 2-3 nodi REALI (laptop + VM/container): store/get di un record, verifica NAT traversal + TTL liveness. **Gate prima di proseguire.**
6. **DHT lookup + self-assignment + coverage gate** — sostituisce il routing statico; prova che un job va in `WAITING_COVERAGE` quando un blocco è scoperto e riprende quando un nodo si auto-assegna.
7. **Failover store-and-forward durevole** — uccidi un holder a metà generazione, prova che lo stage si re-dispaccia dall'attivazione persistita (con recompute del prefisso) e la generazione completa correttamente.
8. **Reputazione light + recompute campionato** su hop di prefill; declassa un nodo deliberatamente difettoso.
9. **Refactor finale** — cancella il loop orchestrator, ogni nodo pull/forward peer-to-peer sullo stesso substrato SQLite (migrazione M0 → store-and-forward).

## Rischi principali & mitigazioni

| Rischio | Mitigazione |
|---------|-------------|
| **Correttezza KV-cache tra hop e restart/failover** — qui il PoC vive o muore. Off-by-one in `position_ids`/`cache_position` dopo un hop ripreso, o double-append su re-dispatch, corrompe silenziosamente la generazione. | `golden_test` single-process **per primo**, ri-eseguito a **ogni** step prima di aggiungere networking/failover. Scritture cache idempotenti keyed `(job_id, stage, token_position)`. KV-cache come oggetto serializzabile (mai handle di sessione live). |
| **Perdita KV-cache su morte mid-pipeline** → recompute O(seq_len) del prefisso; sotto churn diventa patologico. | PoC: accetta recompute-from-prompt sul solo blocco fallito. Policy di checkpointing per-blocco come item v1.1. Self-assign least-replicated per tenere i blocchi caldi a replication ≥2. |
| **`p2pd` arch-mismatch** (Apple Silicon laptop vs x86 VM) → hang silenzioso; bootstrap irraggiungibile → split-brain DHT. | Smoke test p2pd come **gate di build-order** prima del lavoro sul modello. Pin `initial_peers` a un seed noto. Interfaccia `DiscoveryProvider` come escape hatch verso kademlia su VPN flat. |
| **Non-determinismo FP** rende la verifica per uguaglianza fragile. | Mai hash-compare. Promuovi a fp32, `torch.allclose` (atol~1e-2, rtol~1e-3). Campiona solo ~5-10% di hop stateless/prefill, reputation-gated. |
| **Scope creep nel transport hand-owned** mangia il budget criterio-1. | Transport minimale: HTTP via FastAPI/uvicorn con body safetensors-bytes, model id + dtype **fissi** in v1, no negoziazione di protocollo, no quantizzazione. SQLite (non Redis/NATS) = ops a zero. |

## Domande aperte (da risolvere in fase di implementazione)

1. **Matrice nodi del PoC:** i 2-3 nodi sono su LAN/VPN flat (fallback kademlia viabile) o dietro NAT reale (NAT traversal di hivemind load-bearing)? Decide quanto conta il fallback `DiscoveryProvider`.
2. **Versione API Cache di transformers:** pinnare una versione e confermare che `DynamicCache` faccia round-trip di serializzazione pulito per architettura.
3. **Policy failover KV-cache oltre il PoC:** accettare recompute completo del prefisso, o checkpoint periodico per-blocco? Decidere la soglia trigger v1.1 (es. `seq_len > N`).
4. **Valori empirici di tolleranza fp32** (atol/rtol) sul hardware eterogeneo reale: vanno **misurati**, non assunti.
5. **Granularità blocchi & edge-node:** embedding e lm_head come blocchi standalone (più semplice) o co-locati col primo/ultimo slab decoder? Impatta coverage math e fit RAM.
6. **Firma dei record DHT:** aggiungere record firmati ora (forward-compat economico per reputation/BFT) o rimandare? hivemind lo supporta.
7. **Policy di retention/pruning dell'outbox** e limiti di backpressure per evitare crescita illimitata quando un blocco downstream resta scoperto.

## Riferimenti

- [hivemind · PyPI](https://pypi.org/project/hivemind/) (release 2026-01-03, Py3.9-3.12) · [learning-at-home/hivemind](https://github.com/learning-at-home/hivemind)
- [Petals releases](https://github.com/bigscience-workshop/petals/releases) (frozen 2.2.0)
- [accelerate big-model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [Defeating Nondeterminism in LLM Inference — Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) · [arXiv 2408.05148](https://arxiv.org/pdf/2408.05148)
