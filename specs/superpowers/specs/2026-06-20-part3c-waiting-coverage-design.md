# Design — Part 3c: WAITING_COVERAGE (park & resume on incomplete coverage)

- **Date:** 2026-06-20
- **Status:** Approved (autonomous — user authorized completing Full Part 3 without per-increment questions)
- **Roadmap:** third increment of Full Part 3. Builds on [3a durable log](./2026-06-20-part3a-durable-job-log-design.md) + [3b resume](./2026-06-20-part3b-failover-resume-design.md). PRD acceptance **#3**.

## Goal

When the model is not fully covered (a layer block has no live holder) — at job start or after a failover exhausts the redundant holders — the job must **not fail immediately**. It **parks durably** (`status=WAITING_COVERAGE`) and the request waits for coverage to be (re)established, up to a timeout; when a node covers the gap the job **resumes** (reusing 3b's resume-from-persisted-tokens) and completes. Realizes PRD acceptance **#3**: "a job with an uncovered block enters WAITING_COVERAGE and resumes when the block is covered."

## Decisions (per PRD; no open questions)

1. **Long-poll the request.** In our synchronous orchestrator model, "queue, don't lose" = the `/infer` (and `/v1/chat/completions`) request awaits coverage via an async poll loop, rather than returning an error. Acceptable for the PoC (PRD §4 "requests queue up, they are not lost").
2. **Bounded wait (TTL).** Wait up to `coverage_timeout` seconds (default 120), polling every 0.5 s; on timeout, fail with a clear `coverage timeout` error (the orphaned-activation TTL-alarm mitigation from PRD §7).
3. **Durable parking.** While waiting, the job row is `WAITING_COVERAGE`; when coverage returns it goes back to `RUNNING` and proceeds (resume from persisted tokens via 3b). The already-persisted tokens are preserved.
4. **Restart handling.** `recover()` flips both `RUNNING` and `WAITING_COVERAGE` rows to `INTERRUPTED` at startup (no live client survives a coordinator restart).
5. **Config.** `create_coordinator_app(..., coverage_timeout=120.0)`; existing callers (no kwarg) keep the default. (CLI may expose it later; out of scope here.)

## Architecture & files

- **Modify `eujeno/net/jobstore.py`** — add a `set_status(job_id, status)` method (used to mark `WAITING_COVERAGE` / back to `RUNNING`); extend `recover()` to also flip `WAITING_COVERAGE` → `INTERRUPTED`.
- **Modify `eujeno/net/coordinator.py`**:
  - `create_coordinator_app(model_id, num_layers, tokenizer, db_path=None, coverage_timeout=120.0)`.
  - In `_generate_with_failover`, replace the immediate "incomplete coverage" return with an async wait: when `build_chain(...) is None`, `_store_safe(store.set_status, job_id, "WAITING_COVERAGE")`, then poll (`await asyncio.sleep(0.5)`, recompute `stages`/`chain` excluding `excluded`) until a chain is available or `coverage_timeout` elapses. On coverage: `_store_safe(store.set_status, job_id, "RUNNING")` and proceed with the (resume-aware) generation. On timeout: return `{"error": "coverage timeout: model not operational"}`.
  - The wait recomputes `excluded`-filtered coverage each poll (a previously-excluded dead node staying dead is fine; a NEW node covering the gap makes the chain build).

No request/response shape change beyond the existing error path (which now only triggers on timeout, not instantly).

## Data flow

```
/infer -> create_job(RUNNING)
  build_chain None (no head holder yet)
    -> set_status WAITING_COVERAGE ; poll every 0.5s ...
    -> a head node joins -> build_chain OK -> set_status RUNNING
    -> _run_generation (resume_tokens from any prior persisted progress) -> DONE
  (if no coverage within coverage_timeout -> FAILED "coverage timeout")
```

## Error handling

- `set_status` writes go through `_store_safe` (best-effort, never break the request).
- Timeout produces a normal error envelope (`/infer` → `{"ok": false, error}`; `/v1/chat/completions` → 503), and the job row is `FAILED` (via the existing `_store_safe(store.fail, ...)` at the call site).

## Out of scope

- Async/non-blocking job submission (client disconnect + later pickup) — peer-driven target. Coordinator-restart resume. 3d queue/load-balancing. KV checkpointing.

## Verification

- **jobstore unit test** (extend `tests/test_jobstore.py`): `set_status` changes status; `recover()` flips `WAITING_COVERAGE` → `INTERRUPTED` (and still flips `RUNNING`), leaving `DONE` untouched.
- **e2e park-then-resume** (`tests/test_waiting_coverage_e2e.py`, `@pytest.mark.slow`): start a coordinator with only the embed+decoder node (no head → incomplete coverage); fire `/infer` in a background thread; assert the job reaches `WAITING_COVERAGE` (via `GET /jobs`); then start the head node; assert the request returns `ok` with `tokens == reference` and the job is `DONE`.
- **e2e timeout** (same file): with `coverage_timeout` set small (e.g. 2 s) and no head node, `/infer` returns an error (`ok: false`) and the job is `FAILED`.
- Full suite green; existing behavior unchanged when coverage is complete (no parking path taken).
