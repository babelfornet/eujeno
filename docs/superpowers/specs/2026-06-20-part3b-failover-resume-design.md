# Design — Part 3b: resume-from-persisted-state failover

- **Date:** 2026-06-20
- **Status:** Approved (autonomous — user authorized completing Full Part 3 without per-increment questions)
- **Roadmap:** second increment of Full Part 3. Builds on [3a durable job log](./2026-06-20-part3a-durable-job-log-design.md). PRD acceptance **#2**.

## Goal

When a node fails mid-generation, **resume from the durable job log's tokens-so-far** instead of today's full restart-from-prompt. The coordinator re-dispatches to the redundant holders, prefills `prompt + tokens_so_far` on the new chain to rebuild the KV-cache, and continues from where it left off. Realizes PRD acceptance **#2** (node crash mid-job → re-dispatch → completes correctly, `== golden`) and removes the wasteful re-emission of already-produced tokens.

## Decisions (per PRD; no open questions)

1. **Resume, don't restart.** Failover keeps the persisted tokens (the durable log is the source of truth for "what was already emitted") and continues; it no longer calls `reset_progress`.
2. **Prefix replay rebuilds KV.** The new chain's nodes have no KV for the job; resuming prefills `prompt + tokens_so_far` in one pass (the existing per-token relay already prefills the prompt; we extend the initial input to include the produced tokens), then continues autoregressively.
3. **Read-the-log on failure.** On `_NodeFailure`, `resume_tokens` is read from the durable store (`store.get_job(job_id)["tokens"]`), wrapped so a read error falls back to `[]` (= full restart, safe).
4. **Scope = in-flight node failure.** Coordinator-restart auto-resume is out of scope (the synchronous HTTP client is gone after a coordinator restart; async job submission is a later/peer-driven concern).
5. **Idempotency hardening (deferred 3a Minor, folded in here):** `append_token` becomes a true no-op when the same `(job_id, position)` gets the same token, and logs a warning if a *different* token is written to an existing position (surfacing anomalies instead of silently rewriting).

## Architecture & files

- **Modify `eujeno/net/jobstore.py`** — `append_token`: when `position < len(tokens)`, no-op if the token matches; if it differs, `log.warning(...)` then overwrite. Add a module logger.
- **Modify `eujeno/net/coordinator.py`**:
  - `_run_generation(chain, prompt, max_new, sampling, job_id, on_token=None, resume_tokens=None)` — when `resume_tokens` is non-empty: initial input = `prompt ids ⧺ resume_tokens`, `cache_position = arange(seq_len + len(resume_tokens))`, `tokens` starts as `list(resume_tokens)`, the loop runs `max_new - len(resume_tokens)` more steps, and each step's next `cache_position = [seq_len + len(tokens) - 1]` (this formula is also correct for the non-resume path, replacing the current `seq_len + step`). New tokens are appended and reported via `on_token` at positions `len(resume_tokens)…`.
  - `_generate_with_failover(prompt, max_new, sampling, job_id)` — remove the per-attempt `reset_progress`; keep a `resume_tokens` list (starts `[]`); pass it to `_run_generation`; on `_NodeFailure`, set `resume_tokens` from the persisted job (`store.get_job(job_id)`, read wrapped in try/except → `[]` on error) before the next attempt.

No change to request/response shapes, the node protocol, or `/jobs`. `prompt_len` (usage accounting) stays the prompt length.

## Data flow

```
attempt 0: chain=[A,B]; run from prompt; persist tokens t0,t1,t2 ...   ── A dies after t2
attempt 1: chain=[A',B] (A excluded); resume_tokens=[t0,t1,t2] (from durable log)
           prefill prompt+[t0,t1,t2] -> rebuild KV on A',B -> continue t3,t4,... -> DONE
```

Greedy (temperature 0, used by the golden tests) is deterministic, so the resumed continuation reproduces the same sequence a non-failing run would have produced → `== golden`.

## Error handling

- The failover read of persisted tokens is wrapped (try/except → `[]`); a store read failure degrades to a full restart, never aborts the request.
- Writes remain best-effort via `_store_safe` (3a).

## Out of scope

- Coordinator-restart auto-resume of INTERRUPTED jobs (no live client). 3c WAITING_COVERAGE. 3d queue/load-balancing. KV-cache checkpointing (PRD v1.1).

## Verification

- **jobstore unit test** (extend `tests/test_jobstore.py`): `append_token` same `(job_id, position)` + same token → no-op (already covered) and remains length-stable; a differing token at an existing position logs a warning (assert via `caplog`) and overwrites.
- **e2e** (`tests/test_failover_resume_e2e.py`, `@pytest.mark.slow`): coordinator + two redundant full-coverage nodes, greedy. Capture the reference greedy result. Kill one node mid-generation; assert the `/infer` result `== reference` (golden), `failovers >= 1`, and `GET /jobs/{id}` is `DONE` with `tokens == reference` (the durable log drove a correct resume).
- The existing `tests/test_failover_e2e.py` must stay green (correct completion on node kill).
- Full suite green; no inference-behavior change on the happy path.
