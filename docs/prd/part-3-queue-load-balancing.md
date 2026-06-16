# PRD Parte 3 — Queue & Load Balancing

> Decisioni di riferimento: [ADR-0001](../decisions/ADR-0001-implementation-forks.md) (Fork C). Visione: [00-vision-architecture.md](../00-vision-architecture.md).

## 1. Scopo

Il cuore del framing **async / store-and-forward**. Trasforma una domanda utente in un **job durevole** che avanza hop-by-hop, sopravvive alla morte di qualunque nodo, e si reindirizza localmente senza rollback globale né coordinatore. È anche dove vive il **load balancing**: richieste diverse si accodano sui blocchi specifici, massimizzando l'utilizzo della rete.

## 2. In scope (PoC) / Fuori scope

**In scope:** substrato durevole SQLite(WAL) + blob safetensors; modello a job/stage idempotenti; entry-node orchestrator-driven (Milestone 0); store-and-forward peer-driven (target); `WAITING_COVERAGE`; session affinity della KV-cache; scheduling su holder ridondanti per `load`.

**Fuori scope (deferred):** broker esterni (Temporal/Ray/Redis); checkpoint periodico della KV-cache (v1.1); priorità/fairness avanzata.

## 3. Substrato durevole (primitivo condiviso #2)

Per-nodo, **SQLite in modalità WAL** (queue crash-safe, zero ops) + blob su disco.

```sql
-- job log
jobs(
  job_id TEXT PK, model_id TEXT, status TEXT,   -- QUEUED|WAITING_COVERAGE|RUNNING|DONE|FAILED
  prompt TEXT, result TEXT, created_at, updated_at
);
-- stage = un hop nel pipeline; PK = chiave di idempotenza (primitivo #3)
stages(
  job_id TEXT, stage_idx INT, block_lo INT, block_hi INT,
  token_position INT, status TEXT,              -- PENDING|PERSISTED|ACKED|DONE
  activation_ref TEXT, kv_ref TEXT, attempt INT,
  PRIMARY KEY (job_id, stage_idx)
);
-- outbox = handoff pendenti verso il prossimo holder
outbox(
  job_id TEXT, stage_idx INT, next_block TEXT, target_peer TEXT,
  status TEXT, attempts INT,                     -- PENDING|SENT|ACKED
  PRIMARY KEY (job_id, stage_idx)
);
```
Blob su disco: `{job_id}/{stage_idx}.safetensors` (attivazione). KV-cache persistita per `(job_id, block)`.

## 4. Modello a job idempotente (Fork C)

**Invariante:** ogni hop è **idempotente**, identificato da `(job_id, stage_idx)` (+ `token_position` per la cache). Protocollo: **ACK-after-persist** + dedup in ricezione.

```mermaid
sequenceDiagram
    participant H as Holder blocco i
    participant Disk as SQLite + blob
    participant Hn as Holder blocco i+1
    H->>H: run_block (Parte 1)
    H->>Disk: PERSISTI attivazione + KV (status PERSISTED)
    H->>Hn: POST payload (safetensors)  [outbox PENDING→SENT]
    Hn->>Disk: dedup su (job_id, stage_idx); persisti
    Hn-->>H: ACK
    H->>Disk: outbox ACKED; commit-and-prune
    Note over H,Hn: holder morto a qualunque punto ⇒ ri-dispaccio dall'attivazione persistita
```

### Store-and-forward vs Milestone 0
- **Milestone 0 (orchestrator-driven):** l'entry node guida i hop in un loop lineare, **scrivendo sullo stesso substrato**. Più semplice da debuggare; SPOF per-job accettabile solo come bootstrap.
- **Target (peer-driven):** ogni nodo fa pull/forward autonomamente. Migrazione = **cancellare il loop centrale**, non riscrivere la persistenza.

### WAITING_COVERAGE
Se il prossimo blocco non ha holder vivi (coverage incompleta, Parte 2), l'attivazione **parcheggia durevolmente** (`status=WAITING_COVERAGE`) finché un nodo si auto-assegna il blocco. Le richieste **si accodano, non si perdono**.

## 5. Session affinity della KV-cache

La KV-cache resta **con** l'holder che possiede il blocco, keyed `job_id`. Nel loop autoregressivo il messaggio porta **solo** l'hidden state del nuovo token, non l'intera cache. Su morte di un holder mid-generazione: la sua cache è persa → l'holder di backup ricomputa il prefisso (O(seq_len), accettato nel PoC; vedi rischio sotto).

## 6. Load balancing

- Il record DHT espone `load` (profondità coda/utilizzo). Il router (Parte 2) preferisce holder a `load` basso e `reputation` alta.
- Più richieste indipendenti fluiscono in parallelo sui blocchi; un blocco con coda profonda segnala `load` alto → nuovi job preferiscono repliche → utilizzo massimizzato e auto-bilanciato.
- Self-assignment least-replicated (Parte 2) crea repliche dove servono.

## 7. Rischi & mitigazioni (dal team)

- **Dup delivery su ACK-loss** → token doppio. Mitigazione: idempotency key `(job_id, stage_idx, token_position)`, dedup in ricezione.
- **Attivazione orfana** se un blocco non viene mai coperto. Mitigazione: `WAITING_COVERAGE` + allarme TTL.
- **Perdita KV-cache** su morte mid-pipeline → recompute O(seq_len). Mitigazione PoC: recompute-from-prompt sul blocco fallito; policy checkpoint v1.1.
- **Crescita illimitata outbox** → commit-and-prune dopo ACK; limiti di backpressure.
- **Stato job distribuito poco osservabile** → pubblica stato coarse sul DHT.

## 8. Criteri di accettazione

1. Un job attraversa la pipeline e produce la risposta; lo stato è ricostruibile da SQLite.
2. Crash di un nodo a metà job → ri-dispaccio dall'attivazione persistita → il job completa correttamente (== golden).
3. Un job con un blocco scoperto entra in `WAITING_COVERAGE` e riprende quando il blocco viene coperto.
4. Riavvio di un processo → nessun hop perso né doppiamente applicato (idempotenza verificata).

## 9. Dipendenze

- **Parte 1:** `run_block` come step.
- **Parte 2:** lookup del prossimo holder, coverage gate, `load` nel record.
- **Parte 5:** il fan-out di verifica riusa lo stesso primitivo di persistenza/re-dispatch.

## 10. Domande aperte

- Policy failover KV-cache (recompute vs checkpoint) e soglia v1.1 (ADR-0001 Q3).
- Retention/pruning outbox e backpressure (ADR-0001 Q7).
