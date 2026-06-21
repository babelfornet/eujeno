# Design — Part 4a: lightweight reputation (tracking + routing)

- **Date:** 2026-06-20
- **Status:** Approved (autonomous — user authorized completing the roadmap without per-increment questions)
- **Roadmap:** first increment of Part 4 (Incentives & Reputation). [PRD Part 4](../../prd/part-4-incentives-reputation.md). Tokens are **deferred** (PRD §5). Receipts are the sibling increment 4b. PRD acceptance **#1, #2**.

## Goal

Give each connection a `reputation` score that **rises on successful contribution** and **falls when it causes a failover/timeout**, and make the router **prefer high-reputation holders** (alongside the 3d load metric). This realizes PRD acceptance #1 (reputation rises with success, falls with timeouts/failovers) and #2 (the router de-prioritizes a low-reputation node). Tokens/ledger stay out of scope.

## Decisions (per PRD; our coordinator-driven architecture)

1. **Coordinator-tracked reputation**, mirroring 3d's `load`: `conns[cid]["reputation"]`, neutral cold-start `REP_INITIAL = 1.0`, clamped to `[0.0, REP_MAX=10.0]`.
2. **Update rules (request granularity, PoC):** on a completed generation, reward every connection in the winning chain `+REP_REWARD (0.5)`. A `−REP_PENALTY (2.0)` **penalty hook** is applied to a failing connection *if it is still in `conns`* — best-effort. **Architectural note:** in the coordinator-relay model a failover is triggered by a node **disconnecting**, and a disconnected conn is popped from `conns` before the penalty can apply, so the penalty is effectively a no-op for the disconnect failure mode (the node is gone, with no stable identity to remember it by). The penalty hook is retained because it is the attachment point for the real fall-triggers that arrive later: **timeout detection** (no per-call timeout exists today) and **recompute divergence (Part 5)**. The operationally-realized, tested behaviors in 4a are therefore **rise-on-success** and **routing use**.
3. **Routing:** extend `build_chain`'s selection to prefer **highest reputation, then lowest load** (then insertion order). Reputation is primary so a low-rep node is de-prioritized (acceptance #2); with equal reputation it reduces to 3d's least-loaded (so 3d behavior/tests are preserved).
4. **Expose** `reputation` in `/registry` (like `load`).

## Architecture & files

- **Modify `eujeno/net/discovery.py`** — `build_chain(stages_by_url, num_layers, exclude=None, load=None, reputation=None)`. The post-pass runs when `load is not None or reputation is not None`; the selection key becomes `(-R.get(u, 0.0), L.get(u, 0), order[u])` where `R = reputation or {}`, `L = load or {}`. This preserves: default (`load=None, reputation=None`) → original; 3d (`load` only) → `-0` constant ⇒ least-loaded; 4a (`load`+`reputation`) → reputation-primary.
- **Modify `eujeno/net/coordinator.py`**:
  - module constants `REP_INITIAL = 1.0`, `REP_REWARD = 0.5`, `REP_PENALTY = 2.0`, `REP_MIN = 0.0`, `REP_MAX = 10.0`.
  - registration: `conns[conn_id][... ] += "reputation": REP_INITIAL`.
  - `/registry` node entries include `"reputation": c["reputation"]`.
  - `_await_coverage` passes `reputation={cid: c["reputation"] for cid, c in conns.items()}` (plus the existing `load=`) to `build_chain`.
  - `_generate_with_failover`: on the success `return`, reward the winning chain's conns (`min(REP_MAX, rep + REP_REWARD)`); in the `_NodeFailure` branch, penalize the failing conn (`max(REP_MIN, rep - REP_PENALTY)`), guarded by `if cid in conns`.

No request/response shape change beyond the added `/registry` field.

## Data flow

```
build_chain(load=L, reputation=R): per stage, pick max-reputation then min-load holder.
success  -> each winning-chain conn: reputation += 0.5 (cap 10)
_NodeFailure(cid) -> conns[cid].reputation -= 2.0 (floor 0)   # the node that caused the failover
```

## Error handling / concurrency

- Single-process asyncio; reputation reads/writes are synchronous (no race). Reputation updates are best-effort in-memory (no persistence needed for the PoC). A disconnected conn is removed from `conns`, so its reputation disappears (cold-start on reconnect — acceptable for the PoC).

## Out of scope

- Hop receipts (4b). Token/ledger/slashing (PRD §5, deferred). Recompute-divergence penalty (Part 5). Reputation persistence/decay across coordinator restarts (PoC keeps it in-memory).

## Verification

- **`build_chain` unit tests** (extend `tests/test_load_balancing.py` or new `tests/test_reputation.py`):
  - prefers higher reputation: two head holders, `reputation={h1: 0.0, h2: 5.0}` ⇒ head `h2`; flip ⇒ `h1`.
  - equal reputation ⇒ least-loaded decides (3d preserved): `reputation={h1:1,h2:1}, load={h1:3,h2:0}` ⇒ `h2`.
  - `reputation=None` ⇒ identical to the 3d/load path (regression).
- **coordinator e2e** (`tests/test_reputation_e2e.py`, `@pytest.mark.slow`):
  - `/registry` exposes a numeric `reputation` (== `REP_INITIAL` for a freshly-connected node).
  - after a successful `/infer`, the serving node's `/registry` `reputation > REP_INITIAL` (rose) — acceptance #1 (rise).
  - (The fall-on-failover is **not** e2e-tested for the reason in Decision 2; acceptance #2 — router de-prioritizes low reputation — is covered deterministically by the unit tests above.)
- Full suite green; 3d load tests + default `build_chain` unchanged.
