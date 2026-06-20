# Design — P2Pa: failover + EOS for the pure-P2P (no-coordinator) path

- **Date:** 2026-06-20
- **Status:** Approved (autonomous — release-critical P2P goal; user authorized completing without per-increment questions)
- **Roadmap:** first increment toward the **release goal "full P2P, no coordinator, works"**. Builds on Part 2a (gossip) + reuses Part 3b's prefix-replay resume idea over HTTP.

## Goal

Make pure-P2P inference (`infer --peer`, no coordinator) **fault-tolerant**: when a peer dies mid-generation, the entry re-routes around it (rebuild the chain from the gossip registry excluding the dead peer) and **resumes from the tokens already produced** (prefix replay), and it **stops at EOS**. Today the P2P path is single-pass greedy fixed-length — one dead node fails the whole job. This is the release-critical resilience gap.

## Decisions (per the audit; pure-P2P architecture)

1. **Client/entry-driven failover.** The entry (the `infer --peer` client) already holds the gossip registry; on a hop failure it excludes the failed peer URL, rebuilds the chain (`build_chain(..., exclude=...)`), and resumes. No coordinator involved.
2. **Resume = prefix replay (reuse 3b idea over HTTP).** On a new chain, prefill `prompt + tokens_so_far` (fresh `job_id`) to rebuild the peers' KV, then continue from `len(tokens)`. Only the dead peer's hops are redone on a healthy holder.
3. **Identify the dead peer by the in-flight hop.** Track the URL of the current hop; any exception on it ⇒ exclude that URL and retry (up to `max_failovers=5`).
4. **EOS stop** in the direct path via a shared `stop_token_ids(tokenizer)` helper (same rule as the coordinator/server: `eos_token_id` + `<|im_end|>`/`<|endoftext|>`).
5. **Best-effort registry refresh.** An optional `refresh()` callback re-fetches `/registry` (from the entry peer) to learn new/redundant holders and drop stale ones; failover also works from the initially-fetched registry alone (which lists redundant holders).
6. **Keep the static `--topology` path unchanged** (no redundancy in a static file → no failover); only the registry-driven `--peer` path gains resilience.

## Architecture & files

- **Add `stop_token_ids(tokenizer) -> set[int]`** to `eujeno/net/generation.py` (the shared rule; currently inlined in `server.py`/`coordinator.py`).
- **Add `distributed_generate_resilient(...)` to `eujeno/net/orchestrator.py`:**
  ```
  distributed_generate_resilient(stages_by_url, num_layers, prompt, max_new_tokens, client,
                                 tokenizer, stop_ids=None, job_id_prefix="job",
                                 refresh=None, max_failovers=5) -> dict
  ```
  - loop over attempts: `(optionally refresh)` → `chain = build_chain(stages_by_url, num_layers, exclude=excluded)`; if None → `{"ok": False, "error": "incomplete coverage", "tokens": tokens}`.
  - prefill `prompt + tokens` (resume), then per-token relay over HTTP (`/embed`,`/decode/{bk}`,`/head`) tracking `current` URL; `token_id in stop_ids` → finish; on `Exception` → `excluded.add(current)` and continue to the next attempt.
  - success → cleanup (`DELETE /job/{id}` on the live chain) and return `{"ok": True, "text", "tokens", "failovers": attempt}`.
  - exhausted → `{"ok": False, "error": "too many failovers", "tokens": tokens}`.
  - Keep `distributed_generate` (static `--topology`) as-is.
- **Modify `eujeno/cli.py`** `infer --peer`: compute `stop_ids = stop_token_ids(tokenizer)`, call `distributed_generate_resilient(reg["nodes"], reg["num_layers"], prompt, max_new_tokens, client, tokenizer, stop_ids=stop_ids, refresh=lambda: httpx.get(f"{peer}/registry", timeout=10).json()["nodes"])`; map `ok:False` → `NOT_OPERATIONAL`/`GENERATION_FAILED`, `ok:True` → emit `{text, tokens, failovers}`.

No node-protocol change; nodes stay the same HTTP block servers.

## Data flow

```
infer --peer: fetch /registry -> resilient generate
  chain (exclude={}) -> hops over HTTP, tokens t0,t1,t2 ...   ── peer B's /decode 503s
  exclude={B}; rebuild chain (picks redundant C) ; resume: prefill prompt+[t0,t1,t2] on A,C
  continue t3.. -> EOS or max -> DONE   (failovers=1)
```

## Error handling

- Any hop exception (HTTP error, connection refused, timeout) → exclude that peer, retry. `refresh()` failures are swallowed (fall back to the cached registry). Cleanup `DELETE` failures ignored.

## Out of scope

- WAITING_COVERAGE in P2P (P2Pb). Durable job log + receipts in P2P (P2Pc). Load/reputation in P2P. NAT traversal / libp2p (future). Sampling in the direct `--peer` path stays greedy (the `/v1/chat/completions` node entry already samples; making `--peer` sample is a later nicety).

## Verification

- **unit:** `stop_token_ids(tokenizer)` includes `eos_token_id`.
- **e2e** (`tests/test_p2p_failover_e2e.py`, `@pytest.mark.slow`): three HTTP serve nodes — A `embed,decoder:0-12`; B `head,decoder:12-24` wrapped in a `FlakyDecode` ASGI middleware that 503s `/decode` after 4 calls; C `head,decoder:12-24` (healthy). Build `stages_by_url` ordered **B before C** so `build_chain` picks B first; call `distributed_generate_resilient(...)` directly. Assert `result["ok"]`, `result["tokens"] == reference`, `result["failovers"] >= 1` (B died, C resumed). This exercises re-route **and** prefix-replay resume.
- **e2e EOS:** with `stop_ids = {reference[0]}`, assert `distributed_generate_resilient(..., stop_ids=stop_ids)` returns `tokens == []` (stops immediately at the first would-be token).
- Existing `test_infer_peer.py` / `test_gossip_e2e.py` stay green (the happy path still matches reference; `infer --peer` now also EOS-aware — reference is computed without EOS at `max_new_tokens=6`, short enough not to hit EOS, so it stays equal).
- Full suite green.
