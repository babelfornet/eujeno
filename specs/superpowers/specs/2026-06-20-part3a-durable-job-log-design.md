# Design — Part 3a: durable job log + idempotent token-step persistence

- **Date:** 2026-06-20
- **Status:** Approved (brainstorming)
- **Roadmap:** first increment of Full Part 3 (Queue & Load Balancing / store-and-forward). See [PRD Part 3](../../prd/part-3-queue-load-balancing.md). Later increments: 3b per-hop failover (re-dispatch from persisted state), 3c WAITING_COVERAGE, 3d queue & load-balancing.

## Goal

Give the coordinator a **durable, crash-safe job log** so a distributed generation's state is reconstructible from disk and survives a coordinator restart without losing or doubling tokens — without changing inference behavior. This realizes PRD Part 3 acceptance criteria **#1** (job state reconstructible from SQLite) and **#4** (process restart → no hop lost or doubly applied).

## Decisions locked during brainstorming

1. **Orchestrator-driven, coordinator-side substrate.** In our coordinator-relay architecture the coordinator *is* the conductor (no peer-to-peer handoff), so the durable substrate lives at the coordinator. The PRD's peer-driven `outbox`/ACK-between-peers is the deferred "target" and is out of scope here.
2. **Per-token-step granularity.** Persist a durable job log keyed at the token-step level. No per-hop `stages`/`outbox` tables and no per-hop activation blobs (those serve the peer-driven target / mid-token failover in 3b).
3. **No auto-resume in 3a** (option A). On coordinator restart, `DONE` jobs remain readable with their result; in-flight `RUNNING` jobs are marked `INTERRUPTED` (partial tokens preserved, never doubled). Auto-resume with prefix recompute is 3b.
4. **Default DB path** `~/.eujeno/coordinator-jobs.db`, overridable via `coordinator --db PATH`.

## Architecture & files

- **Create `eujeno/net/jobstore.py`** — a small SQLite(WAL) wrapper with a single responsibility (persist/query the job log), no network/torch dependencies, unit-testable in isolation. Public API:
  - `JobStore(path)` — opens/creates the DB, enables WAL, creates the schema if absent.
  - `create_job(job_id, model_id, prompt, sampling: dict, prompt_len: int) -> None` — inserts a `RUNNING` row (position 0, empty tokens).
  - `append_token(job_id, token_id: int, position: int) -> None` — idempotent on `(job_id, position)`: sets `tokens` to the first `position+1` ids and `position`; re-applying the same position is a no-op (no double).
  - `finish(job_id, result: str, finish_reason: str) -> None` — status `DONE`.
  - `fail(job_id, error: str) -> None` — status `FAILED`.
  - `recover() -> int` — sets every `RUNNING` row to `INTERRUPTED`; returns the count. Called once at startup.
  - `get_job(job_id) -> dict | None` and `recent_jobs(limit=50) -> list[dict]` — read API.
- **Modify `eujeno/net/coordinator.py`** — `create_coordinator_app(model_id, num_layers, tokenizer, db_path=None)`:
  - construct a `JobStore` (default path if `db_path` is None), call `recover()` at startup.
  - one durable row **per request** in `/infer` and `/v1/chat/completions`: `create_job` before generation, `append_token` per step, `finish` on success, `fail` on error/exhausted-failover.
  - add `GET /jobs/{job_id}` and `GET /jobs?limit=N` (read API).
- **Modify `eujeno/cli.py`** — `coordinator` command gets a `--db PATH` option, passed to `create_coordinator_app`.

## Data flow

- `infer`/`chat_completions` create a stable user-facing `job_id` once, then call `_generate_with_failover`. The durable row uses that one `job_id` (NOT the per-failover-attempt node ids used for KV-cache keying).
- The existing full-restart failover is unchanged; if an attempt restarts generation, the log reflects the **winning** attempt: on the start of a (re)attempt the row's `position`/`tokens` are reset, and each successful token is appended. Final state is `DONE` with the complete result, or `FAILED` if all attempts fail.
- `append_token` writes the cumulative token list at each step (small: a JSON array of ints), so the row is always a complete, consistent snapshot.

## Schema (single `jobs` table, SQLite WAL)

```sql
jobs(
  job_id        TEXT PRIMARY KEY,
  model_id      TEXT,
  status        TEXT,   -- RUNNING | DONE | FAILED | INTERRUPTED
  prompt        TEXT,
  sampling_json TEXT,   -- JSON of the sampling dict
  prompt_len    INTEGER,
  position      INTEGER,-- number of tokens generated so far
  tokens_json   TEXT,   -- JSON array of generated token ids
  result        TEXT,   -- decoded text (on DONE)
  finish_reason TEXT,   -- stop | length (on DONE)
  error         TEXT,   -- on FAILED
  created_at    REAL,
  updated_at    REAL
)
```

## Error handling

- DB writes wrapped so a persistence failure never breaks inference (log a warning, continue) — durability is best-effort relative to serving the request; a corrupt/locked DB must not take the network down.
- `JobStore` uses a single connection in WAL mode; the coordinator is a single async process, so writes are serialized. `JobStore` stamps `created_at`/`updated_at` itself with `time.time()`.

## Out of scope (later increments / deferred)

- Per-hop `stages`/`outbox` tables, activation blobs, ACK-after-persist between peers (peer-driven target).
- Auto-resume / re-dispatch from persisted activation (3b).
- WAITING_COVERAGE durable parking (3c).
- `load` metric + scheduling across replicas (3d).
- KV-cache checkpointing (PRD v1.1).

## Verification

- **`jobstore` unit tests** (`tests/test_jobstore.py`, pytest, tmp_path DB):
  - create → append tokens → finish: `get_job` returns `DONE`, correct `tokens`, `position`, `result`.
  - **idempotent append**: appending the same `(job_id, position)` twice yields the same token list (no double).
  - **recover()**: a `RUNNING` row → `INTERRUPTED` after `recover()`; `DONE` rows untouched; returns the right count.
  - **durability**: write rows, close, reopen `JobStore(path)` → state intact (WAL persisted).
- **e2e** (extend coordinator e2e): run a small distributed generation through the coordinator → assert `GET /jobs/{id}` is `DONE` and its `tokens` exactly equal the returned tokens (state reconstructed from a freshly-opened DB).
- Full suite stays green; inference output unchanged (golden tests still pass).
