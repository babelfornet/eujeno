# Design — Part 3d: load-balancing across redundant holders

- **Date:** 2026-06-20
- **Status:** Approved (autonomous — user authorized completing Full Part 3 without per-increment questions)
- **Roadmap:** final increment of Full Part 3. PRD Part 3 §6 (load balancing). The PRD's numbered acceptance criteria (#1–#4) are already met by 3a/3b/3c; this is the remaining in-scope enhancement.

## Goal

Spread concurrent requests across **redundant holders** instead of always hammering the same one. The coordinator tracks how many active generations currently use each connection (`load`), exposes it in `/registry`, and `build_chain` prefers the **least-loaded** holder when more than one can serve the same stage. Result: with replicas, parallel `/infer` calls self-balance.

## Decisions (per PRD; no open questions)

1. **Coordinator-tracked load (Milestone 0).** In the orchestrator model the coordinator drives every job, so it *knows* the load: `conns[cid]["load"]` = number of active generations currently routing through that connection. (The PRD's node-reported `load` in the DHT record is the peer-driven target; coordinator-tracked is the equivalent here.)
2. **Least-loaded selection, default-preserving.** `build_chain(..., load=None)` is **unchanged** (all existing behavior/tests preserved). When a `load` map is passed, a post-pass swaps each role (embed, head, each decoder block) to the least-loaded holder that provides the *same* stage, tie-broken by the holder's original order (so `load`-all-equal reproduces the default pick).
3. **Load lifecycle.** A generation increments `load` for every connection in its chain at start and decrements at end (in a `finally`), so failovers and exceptions can't leak counts.

## Architecture & files

- **Modify `eujeno/net/discovery.py`** — `build_chain(stages_by_url, num_layers, exclude=None, load=None)`. Keep the existing structure computation; if `load is not None`, post-process: `embed`, `head`, and each `(bk, url)` in the chain become the least-loaded holder among those providing that exact stage (`embed`/`head`/the block_key `bk`), tie-broken by insertion order via an index map. `load=None` ⇒ byte-for-byte current behavior.
- **Modify `eujeno/net/coordinator.py`**:
  - `conns[cid]` gains `"load": 0` at registration; `/registry` node entries include `"load": c["load"]`.
  - `_generate_with_failover` passes `load={cid: c["load"] for cid in conns}` into the `build_chain` calls (the one in `_await_coverage` and the failover path both go through `_await_coverage` → thread `load` there).
  - Wrap each generation attempt: after a chain is chosen, increment `load` for `{embed_c, head_c, *decoder cids}`; decrement them in a `finally` around `_run_generation`.

No request/response shape change (other than the added `load` field in `/registry`).

## Data flow

```
2 concurrent /infer, two redundant tail holders T1,T2 (load 0,0):
  req A: build_chain(load={...}) -> picks T1 (tie-break order); T1.load=1
  req B: build_chain(load={T1:1,T2:0}) -> picks T2 (least loaded); T2.load=1
  -> the two requests run on different tail holders in parallel
```

## Error handling

- Load inc/dec is in-memory on the single-process coordinator (no locking needed); the `finally` guarantees decrement even on `_NodeFailure`/exceptions. A dead conn is removed from `conns` on WS disconnect, so its load entry disappears with it.

## Out of scope

- Node-reported load / DHT `load` field (peer-driven target). Reputation-weighted routing (Part 4). Queue depth beyond active-generation count. Fairness/priority.

## Verification

- **`build_chain` unit tests** (`tests/test_discovery.py` or a new `tests/test_load_balancing.py`):
  - `load=None` ⇒ identical result to today (regression guard on a redundant-holder fixture).
  - with two head holders and `load={h1: 3, h2: 0}` ⇒ chain's head is `h2`; with `{h1:0,h2:3}` ⇒ `h1`. Same for a duplicated decoder block and for embed.
  - tie (equal load) ⇒ deterministic (original-order) pick.
- **coordinator unit/e2e:** `/registry` entries include a numeric `load`; a single in-flight generation shows `load >= 1` on its chain's conns and back to `0` after completion (can be a focused async test, or asserted via the e2e harness).
- Full suite green; existing `build_chain`/discovery/coordinator tests unchanged (default path untouched).
