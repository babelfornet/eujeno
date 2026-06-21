# Design — P2Pb: WAITING_COVERAGE for the pure-P2P path

- **Date:** 2026-06-21
- **Status:** Approved (autonomous — release P2P goal; user authorized completing without per-increment questions)
- **Roadmap:** P2P parity increment. Brings Part 3c (WAITING_COVERAGE) to the pure-P2P `infer --peer` path. Builds on P2Pa (`distributed_generate_resilient`).

## Goal

When the pure-P2P network isn't fully covered, `infer --peer` should **wait for coverage** (a node joining to fill the gap) instead of failing immediately — up to a timeout — then proceed (reusing P2Pa's failover/resume). Mirrors the coordinator's 3c, in the sync client-driven orchestrator.

## Decisions (per 3c; P2P sync architecture)

1. **Poll-wait in the resilient orchestrator.** Add `coverage_timeout=0.0` + `poll_interval=0.5` to `distributed_generate_resilient`. When `build_chain` returns None, poll (refresh the registry + rebuild) every `poll_interval` until a complete chain appears or `coverage_timeout` elapses. `coverage_timeout=0` ⇒ current fail-fast behavior (byte-for-byte).
2. **Refresh drives discovery.** The wait re-fetches the registry via the existing `refresh()` callback (so a newly-joined node is picked up). Without `refresh`, the registry is static and the wait simply times out — acceptable.
3. **Monotonic clock, sync sleep.** `time.monotonic()` + `time.sleep` (the orchestrator is synchronous httpx).
4. **Error wording.** Timeout ⇒ `"coverage timeout: model not operational"`; immediate (timeout 0) ⇒ unchanged `"incomplete coverage: model not operational"`.
5. **CLI flag** `infer --wait-coverage SECONDS` (default 0 = fail fast) threads `coverage_timeout` through.

## Architecture & files

- **Modify `eujeno/net/orchestrator.py`** — `distributed_generate_resilient(..., coverage_timeout=0.0, poll_interval=0.5)`. Replace the top-of-attempt "refresh + build_chain + (None→return)" with a coverage-resolution loop:
  ```
  start = time.monotonic()
  while True:
      (refresh stages_by_url best-effort)
      chain = build_chain(stages_by_url, num_layers, exclude=excluded)
      if chain is not None: break
      if time.monotonic() - start >= coverage_timeout: break
      time.sleep(poll_interval)
  if chain is None: return {"ok": False, "error": <timeout|incomplete>, ...}
  ```
  The rest (generation + failover + EOS + resume) is unchanged. Add `import time`.
- **Modify `eujeno/cli.py`** — add `wait_coverage: int = typer.Option(0, "--wait-coverage", help="[P2P] seconds to wait for full coverage before failing")` to `infer`; pass `coverage_timeout=wait_coverage` in the `--peer` call.

No node-protocol change.

## Data flow

```
infer --peer --wait-coverage 30, network missing the head block:
  build_chain -> None ; poll (refresh /registry) every 0.5s ...
  a head node joins -> registry refresh sees it -> build_chain OK -> generate -> DONE
  (no head within 30s -> ok:False "coverage timeout")
```

## Out of scope

- Durable parking of the job (the client is a transient CLI; P2Pc handles persistence on serve nodes). Node-driven `/v1` entry waiting (separate path). NAT/libp2p.

## Verification

- **e2e park-then-resume** (`tests/test_p2p_waiting_coverage_e2e.py`, `@pytest.mark.slow`): start only node A (`embed,decoder:0-24`, head uncovered). In a background thread call `distributed_generate_resilient(..., coverage_timeout=20, refresh=<fetch A's /registry nodes>)`. Assert it's still running (not returned) after the registry shows incomplete coverage; then start node B (`head`); join the thread; assert `result["ok"]` and `result["tokens"] == reference`.
- **e2e timeout** (same file): only node A (no head ever), `coverage_timeout=2` ⇒ `result["ok"] is False` and `"coverage timeout" in result["error"]`.
- Regression: `test_infer_peer.py`/`test_p2p_failover_e2e.py` green (covered network ⇒ coverage resolves on the first build, wait never triggers; `coverage_timeout` defaults 0).
- Full suite green.
