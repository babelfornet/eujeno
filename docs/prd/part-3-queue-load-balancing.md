# PRD Part 3 — Queue & Load Balancing

> Reference decisions: [ADR-0001](../decisions/ADR-0001-implementation-forks.md) (Fork C). Vision: [00-vision-architecture.md](../00-vision-architecture.md).

## 1. Purpose

The heart of the **async / store-and-forward** framing. It turns a user question into a **durable job** that advances hop-by-hop, survives the death of any node, and re-routes locally without a global rollback or a coordinator. It is also where **load balancing** lives: different requests queue up on the specific blocks, maximizing network utilization.

## 2. In scope (PoC) / Out of scope

**In scope:** durable substrate SQLite(WAL) + safetensors blobs; idempotent job/stage model; orchestrator-driven entry-node (Milestone 0); peer-driven store-and-forward (target); `WAITING_COVERAGE`; KV-cache session affinity; scheduling over redundant holders by `load`.

**Out of scope (deferred):** external brokers (Temporal/Ray/Redis); periodic KV-cache checkpointing (v1.1); advanced priority/fairness.

## 3. Durable substrate (shared primitive #2)

Per-node, **SQLite in WAL mode** (crash-safe queue, zero ops) + on-disk blobs.

```sql
-- job log
jobs(
  job_id TEXT PK, model_id TEXT, status TEXT,   -- QUEUED|WAITING_COVERAGE|RUNNING|DONE|FAILED
  prompt TEXT, result TEXT, created_at, updated_at
);
-- stage = one hop in the pipeline; PK = idempotency key (primitive #3)
stages(
  job_id TEXT, stage_idx INT, block_lo INT, block_hi INT,
  token_position INT, status TEXT,              -- PENDING|PERSISTED|ACKED|DONE
  activation_ref TEXT, kv_ref TEXT, attempt INT,
  PRIMARY KEY (job_id, stage_idx)
);
-- outbox = pending handoffs to the next holder
outbox(
  job_id TEXT, stage_idx INT, next_block TEXT, target_peer TEXT,
  status TEXT, attempts INT,                     -- PENDING|SENT|ACKED
  PRIMARY KEY (job_id, stage_idx)
);
```
On-disk blobs: `{job_id}/{stage_idx}.safetensors` (activation). KV-cache persisted per `(job_id, block)`.

## 4. Idempotent job model (Fork C)

**Invariant:** each hop is **idempotent**, identified by `(job_id, stage_idx)` (+ `token_position` for the cache). Protocol: **ACK-after-persist** + dedup on receipt.

```mermaid
sequenceDiagram
    participant H as Block holder i
    participant Disk as SQLite + blob
    participant Hn as Block holder i+1
    H->>H: run_block (Part 1)
    H->>Disk: PERSIST activation + KV (status PERSISTED)
    H->>Hn: POST payload (safetensors)  [outbox PENDING→SENT]
    Hn->>Disk: dedup on (job_id, stage_idx); persist
    Hn-->>H: ACK
    H->>Disk: outbox ACKED; commit-and-prune
    Note over H,Hn: holder dead at any point ⇒ re-dispatch from the persisted activation
```

### Store-and-forward vs Milestone 0
- **Milestone 0 (orchestrator-driven):** the entry node drives the hops in a linear loop, **writing to the same substrate**. Simpler to debug; per-job SPOF acceptable only as a bootstrap.
- **Target (peer-driven):** each node pulls/forwards autonomously. Migration = **deleting the central loop**, not rewriting the persistence.

### WAITING_COVERAGE
If the next block has no live holders (incomplete coverage, Part 2), the activation **parks durably** (`status=WAITING_COVERAGE`) until a node self-assigns the block. Requests **queue up, they are not lost**.

## 5. KV-cache session affinity

The KV-cache stays **with** the holder that owns the block, keyed by `job_id`. In the autoregressive loop the message carries **only** the hidden state of the new token, not the whole cache. On the death of a holder mid-generation: its cache is lost → the backup holder recomputes the prefix (O(seq_len), accepted in the PoC; see the risk below).

## 6. Load balancing

- The DHT record exposes `load` (queue depth/utilization). The router (Part 2) prefers holders with low `load` and high `reputation`.
- Multiple independent requests flow in parallel across the blocks; a block with a deep queue signals high `load` → new jobs prefer replicas → utilization is maximized and self-balanced.
- Least-replicated self-assignment (Part 2) creates replicas where needed.

## 7. Risks & mitigations (from the team)

- **Duplicate delivery on ACK-loss** → double token. Mitigation: idempotency key `(job_id, stage_idx, token_position)`, dedup on receipt.
- **Orphaned activation** if a block is never covered. Mitigation: `WAITING_COVERAGE` + TTL alarm.
- **KV-cache loss** on mid-pipeline death → O(seq_len) recompute. PoC mitigation: recompute-from-prompt on the failed block; checkpoint policy v1.1.
- **Unbounded outbox growth** → commit-and-prune after ACK; backpressure limits.
- **Poorly observable distributed job state** → publish coarse state to the DHT.

## 8. Acceptance criteria

1. A job traverses the pipeline and produces the answer; the state is reconstructible from SQLite.
2. A node crash mid-job → re-dispatch from the persisted activation → the job completes correctly (== golden).
3. A job with an uncovered block enters `WAITING_COVERAGE` and resumes when the block is covered.
4. Restart of a process → no hop lost or doubly applied (idempotency verified).

## 9. Dependencies

- **Part 1:** `run_block` as a step.
- **Part 2:** next-holder lookup, coverage gate, `load` in the record.
- **Part 5:** the verification fan-out reuses the same persistence/re-dispatch primitive.

## 10. Open questions

- KV-cache failover policy (recompute vs checkpoint) and v1.1 threshold (ADR-0001 Q3).
- Outbox retention/pruning and backpressure (ADR-0001 Q7).
