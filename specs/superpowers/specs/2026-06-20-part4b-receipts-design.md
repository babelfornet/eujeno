# Design — Part 4b: hop receipts in the job log

- **Date:** 2026-06-20
- **Status:** Approved (autonomous — user authorized completing the roadmap without per-increment questions)
- **Roadmap:** second/last increment of Part 4. [PRD Part 4](../../prd/part-4-incentives-reputation.md) §4. PRD acceptance **#3** ("every ACKed hop leaves a receipt in the job log"). Completes Part 4 (tokens remain deferred, §5).

## Goal

Record, in the durable job log, a **receipt** of each peer's contribution to a job — compute time and bytes — as the attachment point for the future token ledger (PRD §5 rewards "compute time + bandwidth"). Realizes PRD acceptance #3.

## Decisions (per PRD; our coordinator-driven architecture)

1. **Aggregated per `(job_id, peer_id)` receipts.** The PRD schema is `{job_id, stage_idx, peer_id, bytes, t_compute}` per hop. Our autoregressive relay does `max_new × blocks` hops — one SQLite row per hop would explode the log. Instead, accumulate **per peer per job**: `{job_id, peer_id, hops, bytes, t_compute}`. Each ACKed hop contributes to its peer's receipt (hops++, bytes+=, t_compute+=), so every hop *is* reflected in a receipt — at a sane granularity that is exactly what the ledger needs (total compute + bandwidth per node per job).
2. **Measured at the coordinator.** The coordinator drives every hop, so it times each `_call` round-trip (`t_compute`, a proxy incl. transport — fine for the PoC) and sizes the payloads (`bytes` = sent + received). Accumulated in-memory during a generation, persisted **once at job completion** (the winning attempt's hops — only ACKed hops count; failed-attempt hops are discarded). No per-hop SQLite writes on the hot path.
3. **Read API:** `GET /jobs/{job_id}/receipts`.

## Architecture & files

- **Modify `eujeno/net/jobstore.py`** — new `receipts` table + methods:
  - schema `receipts(job_id TEXT, peer_id TEXT, hops INT, bytes INT, t_compute REAL, PRIMARY KEY(job_id, peer_id))`.
  - `add_receipts(job_id, receipts: dict)` where `receipts = {peer_id: {"hops", "bytes", "t_compute"}}`; UPSERT-accumulate (`ON CONFLICT(job_id,peer_id) DO UPDATE SET hops=hops+excluded.hops, ...`).
  - `get_receipts(job_id) -> list[dict]` (each `{peer_id, hops, bytes, t_compute}`).
- **Modify `eujeno/net/coordinator.py`**:
  - `_run_generation(..., receipts=None)` — wrap the embed/decode/head `_call`s in a timed local that, when `receipts is not None`, accumulates `receipts.setdefault(cid, {...})` with `hops += 1`, `bytes += len(sent) + len(recv)`, `t_compute += dt`. (The `end` cleanup calls are not counted.)
  - `_generate_with_failover` — per attempt, pass a fresh `attempt_receipts = {}`; on the success path (alongside the reputation reward), `_store_safe(store.add_receipts, job_id, attempt_receipts)`.
  - new route `GET /jobs/{job_id}/receipts` → `{"receipts": store.get_receipts(job_id)}`.

No change to existing request/response shapes or the node protocol.

## Data flow

```
generation hop on conn cid: receipts[cid].hops++ ; bytes += sent+recv ; t_compute += rt
job completes -> add_receipts(job_id, receipts)  (winning attempt only)
GET /jobs/{id}/receipts -> [{peer_id, hops, bytes, t_compute}, ...]   # ledger hook
```

## Error handling

- `add_receipts` goes through `_store_safe` (best-effort; a persistence failure never breaks inference). Receipts are measured only for ACKed (successful) `_call`s — a hop that raises `_NodeFailure` is not counted (the exception propagates before the accumulate line).

## Out of scope

- Per-hop (stage_idx) granularity rows; token/credit valuation, ledger settlement, slashing (PRD §5 deferred); cryptographic proof-of-compute.

## Verification

- **jobstore unit tests** (`tests/test_jobstore.py`): `add_receipts` then `get_receipts` round-trip (fields correct); calling `add_receipts` twice for the same `(job, peer)` **accumulates** (hops/bytes/t_compute summed); `get_receipts` of an unknown job → `[]`.
- **coordinator e2e** (`tests/test_receipts_e2e.py`, `@pytest.mark.slow`): after a successful `/infer` through a single all-stages node, `GET /jobs/{id}/receipts` returns a receipt for that peer with `hops > 0`, `bytes > 0`, `t_compute >= 0` — acceptance #3.
- Full suite green; no change to existing endpoints/behavior.
