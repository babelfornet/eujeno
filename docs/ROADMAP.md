# Eujeno — Roadmap & Status

> Living document. Tracks **what has been done** and **what remains to do**. Updated at every significant step.
>
> **Last updated:** 2026-06-17 — Part 2 complete (pure P2P + coordinator-relay NAT) and **automatic failover** in the coordinator (node down → re-routing to a redundant holder).

## Legend

- `[x]` done · `[~]` in progress · `[ ]` to do · `⏸` deferred

---

## Scope decisions (locked)

| Topic | Choice |
|------|--------|
| First-phase goal | **A PoC that actually runs** on 2–3 real nodes |
| Token/crypto incentives | **Deferred** in the PoC (designed on paper, not implemented) |
| Work order | **Whole architecture (PRD) first, then code** |
| Initial target model | **Small, 1B–3B** (e.g. Llama 3.2 1B/3B, Qwen2.5 0.5–1.5B) |
| Runtime | **Python** (HF transformers / PyTorch) |

---

## Phase 0 — Brainstorming & Architecture  `[~]`

- [x] Project context exploration (greenfield)
- [x] Scope reality-check + decomposition into Parts 0–5
- [x] Fundamental forks (goal / incentives / first module / model)
- [x] Architectural backbone (Part 0) — **approved**
- [x] Documentation scaffolding (`docs/`)
- [x] **Team of agents**: comparison of the 5 contested implementation paths (`eujeno-impl-forks` workflow, 9 agents)
- [x] **[ADR-0001](./decisions/ADR-0001-implementation-forks.md)**: decisions on the 5 implementation forks
- [x] PRD **[Part 1](./prd/part-1-peer-node.md)** — Peer Node & Layer Execution
- [x] PRD **[Part 2](./prd/part-2-discovery-routing.md)** — Discovery & Routing
- [x] PRD **[Part 3](./prd/part-3-queue-load-balancing.md)** — Queue & Load Balancing
- [x] PRD **[Part 4](./prd/part-4-incentives-reputation.md)** — Incentives & Reputation *(light reputation + tokens on paper)*
- [x] PRD **[Part 5](./prd/part-5-security-bft.md)** — Security & Byzantine Fault Tolerance *(light verification + BFT on paper)*
- [~] Spec self-review + user review of the PRDs

## Phase 1 — PoC Implementation  `[ ]`

> Starts only after the PRDs are approved. Each module: plan → build → verify.

- [~] **Peer Node** — [plan](./plans/2026-06-17-part-1-peer-node.md)
  - [x] **Single-process foundation** (build-order steps 1-2-4): splitting the model into blocks (EMBED/DECODER/HEAD), `run_block`, serializable per-block KV-cache, **golden test** (the distributed pipeline reproduces `model.generate` exactly), **capstone** (the KV-cache survives a byte round-trip mid-generation). 12 green tests on `Qwen2.5-0.5B`.
  - [x] **Real partial-loading** (`init_empty_weights` + selective loading from the safetensors shards): a node materializes in RAM ONLY the assigned layers, the rest stays on `meta` (zero memory). `serve` uses it. Golden test: 2 partial nodes == the whole model. → runs on machines with little RAM/GPU.
  - [x] **3-node test (1 host + 2 Docker containers)** — [docker/](../docker/) · [quickstart](./examples/docker.md)
  - [x] **Network transport** (FastAPI + safetensors) + **distributed orchestrator** (Milestone 0) — [plan](./plans/2026-06-17-part1-networking.md). `serve`/`infer` commands; distributed golden on 2 real nodes green. Static topology (DHT discovery arrives in Part 2).
- [x] **`eujeno` CLI** (AI-native) — entry point for all operations — [PRD](./prd/cli.md) · [plan](./plans/2026-06-17-cli-eujeno.md). Single-word commands implemented: `version`, `model --info`, `generate`, `selfcheck`, `schema`; JSON output with a stable envelope, deterministic exit codes, prompt from stdin, clean streams. Green suite.
- [~] **Discovery & Routing** — two modes ([ADR-0002](./decisions/ADR-0002-nat-connectivity.md))
  - [x] **Pure P2P**: discovery via **gossip** (decentralized registry + coverage), `serve --peers/--advertise` + `infer --peer` — [plan](./plans/2026-06-17-part2a-p2p-gossip.md). No central server; for LAN/VPN/public IPs.
  - [x] **Coordinator-relay** (opt-in, NAT-without-VPN) — [plan](./plans/2026-06-17-part2-coordinator.md) · [quickstart](./examples/coordinator.md). Nodes via outbound WebSocket; golden via relay green. `coordinator`, `serve --coordinator`, `infer --coordinator` commands.
  - [x] **Automatic failover** on a node going down (coordinator): redundancy + re-routing to a redundant holder — [plan](./plans/2026-06-17-part3-failover.md). e2e: a node crashing mid-hop → the job completes via the redundant holder.
  - [x] **Durable store-and-forward + resume failover + WAITING_COVERAGE** (Part 3a/3b/3c): SQLite(WAL) job log (`coordinator --db`, idempotent on `(job_id,position)`, restart-safe, `GET /jobs[/{id}]`) — [3a](./plans/2026-06-20-part3a-durable-job-log.md); failover **resumes from persisted tokens** via prefix replay instead of restarting — [3b](./plans/2026-06-20-part3b-failover-resume.md); uncovered jobs **park** as `WAITING_COVERAGE` and resume when a node covers the gap (TTL) — [3c](./plans/2026-06-20-part3c-waiting-coverage.md).
  - [ ] direct-P2P failover · native libp2p for P2P-over-NAT (future)
- [x] **OpenAI-compatible API** (`/v1/chat/completions` + `/v1/models`) on the coordinator: chat template + sampling (temperature/top_p/repetition_penalty/seed) — [plan](./plans/2026-06-17-openai-api.md) · [agents guide](./examples/agents.md). Connects OpenAI clients/agents.
- [x] **Stop at EOS + tool/function calling** (`tools`/`tool_calls`) — [plan](./plans/2026-06-17-tool-calling.md). Foundation for MCP agents (the host runs the tools; the model decides). *(SSE streaming + Anthropic/LiteLLM for Claude Code = next steps)*
- [x] **Queue & Load Balancing** (Part 3 complete): durable job store + store-and-forward (3a/3b/3c) **+ load-balancing (3d)** — the coordinator tracks per-connection `load`, exposes it in `/registry`, and `build_chain` routes concurrent requests to the least-loaded replica — [3d](./plans/2026-06-20-part3d-load-balancing.md).
- [ ] Plan + build **Minimal reputation** (tokens ⏸ deferred)
- [ ] End-to-end integration on 2–3 nodes + failover tests
- [x] Private GitHub repo setup → [albertoferrazzoli/eujeno](https://github.com/albertoferrazzoli/eujeno) (public on first working build)

### Known limitations of the Part 1 foundation (to be addressed in later modules)

- `build_causal_mask` assumes **batch=1, no padding, no sliding-window** (correct for Qwen2.5-0.5B and the single-stream PoC). Batch>1 / left-padding / SWA out of scope for now.
- `split_into_blocks` **mutates `layer.self_attn.layer_idx` in place**: fine because each real node loads its own copy of the model; in the tests, isolation is guaranteed by `conftest.py`, which restores the indices. To harden: validate that the `boundaries` cover `[0, num_layers]` contiguously.
- KV-cache failover: when a holder dies mid-generation, the prefix is recomputed (O(seq_len)); periodic per-block checkpointing deferred (see [ADR-0001](./decisions/ADR-0001-implementation-forks.md) Q3).
- `run_block` API: the foundation implements it **stateful** (the KV-cache is block state, it builds the mask/position_ids internally and returns only `hidden_states`) — a convenient in-process choice. [PRD Part 1](./prd/part-1-peer-node.md) §3 describes the **pure** form `(activation, kv) -> (activation, kv)`. To be reconciled when the wire transport arrives (Part 3): either the pure signature is restored or the PRD is updated.

## Milestone — "Operational model"

The system is **operational** only when **every layer block is covered by ≥1 node**. Before that, requests are **queued**. PoC success criteria:

1. A 1–3B model is split and distributed across ≥2 nodes.
2. A user question produces a correct answer traversing the distributed pipeline.
3. If a node goes down during a job, traffic is redirected to a redundant holder and the job completes.
4. A new node that joins self-assigns an uncovered block and coverage updates.

---

## Backlog / Deferred  `⏸`

- ⏸ **Token/crypto** incentive system (on-chain ledger, settlement, full proof-of-compute)
- ⏸ **Full BFT** (commit-reveal, consensus on outputs, economic slashing)
- ⏸ **Large models (70B+)** and advanced tensor-parallel sharding
- ⏸ Advanced reputation + economic sybil resistance
- ⏸ **Public** repo + community onboarding
