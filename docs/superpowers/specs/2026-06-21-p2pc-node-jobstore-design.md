# Design — P2Pc: per-node job log + receipts (pure-P2P observability)

- **Date:** 2026-06-21
- **Status:** Approved (autonomous — release P2P goal; user authorized completing without per-increment questions)
- **Roadmap:** final P2P parity increment. Brings the durable job log (3a) + receipts (4b) to the **serve-node `/v1/chat/completions` entry** — the PRD Part 3 §3 "per-node SQLite" peer-driven substrate. Reuses `JobStore`.

## Goal

Make any pure-P2P serve node **observable** like the coordinator: when a node acts as the OpenAI entry (`/v1/chat/completions`), it records the job (status + result) and per-peer **receipts** (hops/bytes/t_compute) in its own durable `JobStore`, exposed via `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/receipts`. Realizes Part 4 acceptance #3 for the P2P path.

## Decisions (peer-driven; reuse existing parts)

1. **Per-node `JobStore`** (the Part 3a/4b class, unchanged) in `create_app`. `db_path=None` ⇒ `:memory:` (ephemeral; keeps all existing `create_app` callers/tests working and avoids multi-node-same-file clashes). `serve --db PATH` opts into on-disk durability.
2. **Coarse job log (entry-side).** The `/v1` entry node logs `create_job` before generation and `finish`/`fail` after — no per-token persistence (the node `/v1` path doesn't resume; that's the coordinator/CLI paths). Job state (prompt, result, tokens, finish_reason) is reconstructible from the node's DB.
3. **Receipts measured at the entry.** The entry's `run_embed`/`run_decoders`/`run_head` closures (which already do the HTTP hops) are instrumented to accumulate `{hops, bytes=sent+recv, t_compute=round-trip}` per peer URL; persisted once at completion via `add_receipts` (winning run).
4. **Best-effort persistence.** A `_store_safe` wrapper (like the coordinator) so a DB error never breaks serving. `recover()` at startup marks stale RUNNING→INTERRUPTED.
5. Only the embed/decode/head hops are billed (not the `DELETE /job` cleanup).

## Architecture & files

- **Modify `eujeno/net/server.py`** — `create_app(model, tokenizer, stages, node_url=None, peers=None, num_layers=None, gossip_interval=2.0, ttl=30.0, db_path=None)`:
  - `from eujeno.net.jobstore import JobStore`; `store = JobStore(db_path if db_path is not None else ":memory:")`; `store.recover()`; a `_store_safe(fn, *a)` helper (try/except + `logging`).
  - In `v1_chat`: compute `prompt_len`; `_store_safe(store.create_job, job_id, model_id, prompt, sampling, prompt_len)`; accumulate `receipts={}` in the three run_* closures (time + payload sizes per peer); on success `_store_safe(store.finish, job_id, text, finish_reason)` + `_store_safe(store.add_receipts, job_id, receipts)`; wrap the generation in try/except → `_store_safe(store.fail, job_id, str(e))` and re-raise/return 503.
  - Add `GET /jobs` (`{"jobs": store.recent_jobs(limit)}`), `GET /jobs/{job_id}` (404 if missing), `GET /jobs/{job_id}/receipts` (`{"receipts": store.get_receipts(job_id)}`).
- **Modify `eujeno/cli.py`** — `serve` command: add `db: str = typer.Option(None, "--db", help="[P2P] SQLite job-log path (default in-memory)")`; pass `db_path=db` to `create_app`.

No change to the node hop protocol or the `/registry`/gossip.

## Data flow

```
POST /v1/chat/completions on node N (entry):
  create_job(entryK)  -> run hops (accumulate receipts per peer) -> finish + add_receipts
GET /jobs/entryK            -> DONE + result/tokens
GET /jobs/entryK/receipts   -> [{peer_id, hops, bytes, t_compute}, ...]   # ledger hook, per node
```

## Out of scope

- Per-token persistence / resume in the node `/v1` path (the CLI `infer --peer` path has failover/resume; node-`/v1` failover is a later increment). Token ledger valuation (deferred). NAT/libp2p.

## Verification

- **e2e** (`tests/test_p2p_node_jobstore_e2e.py`, `@pytest.mark.slow`): two serve nodes (A embed+dec, B head+dec) with `create_app(..., db_path=<tmp>)` on the entry node A; `POST /v1/chat/completions` to A → assert a valid OpenAI reply; then `GET {A}/jobs` shows the job `DONE` with the reply text, and `GET {A}/jobs/{id}/receipts` returns per-peer receipts with `hops>0`, `bytes>0`.
- Regression: existing `test_p2p_entry_e2e.py` / `test_infer_peer.py` / `test_gossip_e2e.py` green (the `/v1` response shape is unchanged; new endpoints additive; default `:memory:`).
- Full suite green.
