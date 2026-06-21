# Design — Node UI rebuild + real node metrics + speed-aware routing

- **Date:** 2026-06-21
- **Status:** Approved (user requested the rebuild from `docs/design/Eujeno App.dc.html`; chose React+Vite served by every node + back every field for real; added: per-node processed-requests metric + neighbor connection-speed + router favors faster nodes)
- **Source design:** `docs/design/Eujeno App.dc.html` (a design comp in the proprietary `x-dc`/`DCLogic` mockup framework — visual + behavioral reference, not production code).

## Goal

Rebuild the entire Eujeno UI as a **React + Vite single-page app served by every node** at its own URL, faithfully matching the design (sidebar + Network/Chat/Settings pages, light/dark + accent theming, animated swarm canvas). Back **every** displayed field with real node data, and add **request-throughput accounting + neighbor latency measurement** so the **router prefers faster/more-productive nodes**.

## Pages (from the design)

- **Sidebar** (collapsible): Eujeno logo, nav (Network/Chat/Settings), bottom status card (Connected dot, peerId, "N peers · model").
- **Network**: 4 stat cards (Connected peers · Your layers · Throughput tok/s · Avg latency ms) — plus **Requests served** (new); animated **swarm-topology canvas**; **Your node** panel (status, layers, RAM used/total, region, uptime, requests served, restart); **Peers table** (peer · layers · region · latency · status), sorted fastest-first.
- **Chat**: empty-state ("Ask the swarm" + example chips); user/assistant bubbles; assistant **routing footer** ("routed through N nodes · L layers · X tok/s"); composer (Enter to send). Wired to the node's own `/v1/chat/completions` (the node is the P2Pc entry).
- **Settings**: Identity (peerId + copy, node name); Node (model, layer assignment Auto/Manual, max layers, max RAM, public port); Network (region, bandwidth limit); Privacy toggles (auto-join, contribute, allow inbound, telemetry); Save.

> Naming: the design says "VRAM"; these nodes are CPU/RAM. Label it **RAM** (used/total) in our copy; the field is real (`psutil`).

## Architecture

### Frontend — `app/` (Vite + React)
- New Vite project at repo root `app/` (separate from the externally-maintained `web/` landing site). React 18, JS (no TS, matching the repo), inline-style components mirroring the design's CSS, a small theme module (light/dark + accent CSS variables, persisted in `localStorage`), the swarm `<canvas>` animation ported from the design's `_draw` logic (driven by real peer count), and a tiny API client.
- **Builds to `eujeno/ui/static/`** (Vite `build.outDir`), committed so every node ships the bundle with **zero npm** at runtime. Maintainer runs `npm --prefix app ci && npm --prefix app run build` (documented; later wired into packaging). `base: './'` so assets load from the node root.
- View state is in-app (no router lib needed) — matches the design's `view` state.

### Backend — extend the node (`eujeno/net/server.py` `create_app` + new modules)
All UI data is served by the node itself, same-origin, under `/api/*`; chat reuses `/v1/chat/completions`.

- **`eujeno/net/nodeconfig.py`** — `NodeConfig(path)`: JSON-persisted settings + a **stable persisted peerId** (`node·<hex8>·<hex8>`, generated once). Fields (defaults match the design): `name`, `model`, `layerMode` (auto|manual), `maxLayers`, `maxRam`, `port`, `region`, `bandwidth`, `autojoin`, `contribute`, `inbound`, `telemetry`. `get()`, `update(partial)`, `peer_id`.
- **`eujeno/net/metrics.py`** — `NodeMetrics`: process-local counters + derived stats:
  - **`requests_served`** — total processed: increments on every `/v1` entry job AND every worker hop served (`/embed`,`/decode`,`/head`). Persisted in the JobStore is entry-only; this counter (in `NodeMetrics`, mirrored to a small row in the JobStore for durability) covers worker hops too.
  - **`throughput_tok_s`** — Σ completion_tokens / Σ elapsed over recent DONE jobs (from JobStore).
  - **`avg_latency_ms`** — EWMA of measured neighbor round-trip (see peer probe) — i.e. connection speed to neighbors.
  - **`active_queries`** — count of RUNNING jobs.
  - **`uptime_sec`** — `now - started_at`.
  - **RAM** used/total via `probe_capacity()` (+ process RSS for "used").
- **Peer probe loop** (in `create_app` lifespan): every ~5 s, `GET {peer}/health` for each registry peer, record EWMA round-trip latency per peer URL → `peer_latency[url]`. Status derived (online if recently reachable, syncing if known-but-unreachable-briefly, offline). Each node **advertises** its own recent `throughput_tok_s` and `region`/`name` in the registry (extend `own_stages` with `name`, `region`, `tput`, `ram`), gated by the `telemetry`/`inbound` toggles.
- **Per-peer observed hop time** → `peer_hop_time[url]` (EWMA): after each entry job, fold the job's receipts `t_compute` (real measured time to use each peer for its hop) into this EWMA, blended with `peer_latency`. This is the signal the speed-aware router consumes (lower = faster ⇒ used more). `speed[url] = 1 / max(peer_hop_time[url], ε)`, neutral default for unmeasured peers.
- **New routes:**
  - `GET /api/node` → `{ peerId, name, model, numLayers, stages, layers (human, e.g. "L0–L11,head"), status, ramUsedGb, ramTotalGb, region, uptimeSec, port, requestsServed, throughputTokS }`.
  - `GET /api/metrics` → `{ connectedPeers, throughputTokS, avgLatencyMs, activeQueries, requestsServed }`.
  - `GET /api/peers` → `{ peers: [ { peerId, url, layers, region, latencyMs, throughputTokS, status } ] }` (from registry + `peer_latency` + advertised tput; **sorted fastest-first** — lowest latency, then highest tput).
  - `GET /api/settings` / `PUT /api/settings` → `NodeConfig` (PUT accepts partial, persists, re-advertises region/name; returns full).
  - `POST /api/node/restart` → best-effort: re-probe capacity + re-broadcast registration; returns `{ ok, message }`. (True process restart needs a supervisor — documented; not a silent no-op, it re-initialises what it can.)
  - **SPA mount:** `app.mount("/", StaticFiles(directory=<static>, html=True))` **after** all API/inference routes, so `GET /` serves the dashboard and assets resolve; missing dir → a friendly placeholder route (so tests/dev without a build still work).
- **Chat routing footer:** extend the `/v1/chat/completions` response with an additive `"eujeno": { "hops": <#peers in chain>, "layers": <numLayers>, "tokS": <throughputTokS> }` (OpenAI clients ignore unknown fields); the UI renders the footer from it.

### Speed-aware routing (favor faster nodes, minimize total request time) — `eujeno/net/discovery.py`
**Goal (user):** if a peer processes a hop faster, route through it more, so the *overall* time of a request driven by this node is minimized.

`build_chain(stages_by_url, num_layers, exclude=None, load=None, reputation=None, **speed**=None)`:
- `speed` is an optional per-URL score where **higher = faster observed hop**. It is the inverse of each peer's **EWMA observed hop time** — the real round-trip-to-result time the entry has measured for that peer, which is exactly the `t_compute` already recorded per peer in the **receipts ledger** (Part 4b), blended with the `/health` probe RTT. Lower observed time ⇒ higher score ⇒ chosen more often, which directly minimizes total request latency.
- **Cold start / exploration:** a peer with no measurement yet gets a **neutral (optimistic) default** score so it is still tried and can earn a real measurement; otherwise the fastest-known peer would starve newcomers.
- The selection post-pass (reputation primary — Part 4a; load — Part 3d) gains speed: when the maps are provided, the per-stage holder is chosen by key `(-reputation, -speed, load, insertion_order)` → **high-reputation, then fastest-observed, then least-loaded**. Default path (no maps) is byte-for-byte unchanged. (Single-holder ranges are still used regardless — speed only breaks ties among redundant holders.)
- The node entry (`/v1/chat/completions`, and the resilient P2P orchestrator via an optional `speed` passthrough) builds the `speed` map from its **per-peer observed-hop-time EWMA** (updated from each job's receipts) + the `/health` probe. Coordinator can pass it too (same signature). The result is a feedback loop: faster nodes accumulate higher scores and receive proportionally more of the work.

## Data flow

```
Browser → node :PORT/            → static SPA (Vite build)
SPA → GET /api/node|metrics|peers → real node state (config, JobStore, capacity, peer probe)
SPA → POST /v1/chat/completions   → node entry drives the swarm (P2Pc) → reply + eujeno routing
peer probe loop → /health RTT → peer_latency (EWMA) → speed map → build_chain favors fast peers
```

## Error handling
- All `/api/*` handlers are best-effort: a probe/JobStore/capacity error returns sane zeros, never 500s the dashboard. Peer probe failures mark a peer `offline`/`syncing`, never crash the loop. Missing build dir → placeholder page + working `/api`. Settings writes are atomic (temp+rename).

## Out of scope (v1)
- True Mbps bandwidth shaping (the limit is persisted + advertised + used as a coarse inbound-concurrency cap; real traffic shaping is future). Remote telemetry collector (the toggle gates what's shared in gossip; no external sink). Live layer re-assignment without restart (settings persist; layer/model/port changes apply on node restart — surfaced in the UI). NAT/libp2p. Auth on `/api` (LAN/trusted-network assumption, as today).

## Verification
- **Backend unit/e2e** (`tests/`, `@pytest.mark.slow` for model-loading ones):
  - `nodeconfig` round-trips + stable peerId across reloads; `metrics` throughput/requests math; `build_chain` speed-aware selection (fast peer preferred; default path unchanged — extends `tests/test_load_balancing.py`).
  - node e2e: two serve nodes; `GET /api/node|metrics|peers` return real values (peers≥1, requestsServed grows after a `/v1` call, peer latency measured); `PUT /api/settings` persists; `/v1` response carries `eujeno` routing.
  - `GET /` serves the built SPA when present (and the placeholder when not).
- **Frontend**: `npm --prefix app run build` succeeds → `eujeno/ui/static/index.html` + `assets/` exist; a lightweight check that the bundle loads the three views (smoke). Manual visual parity against the design comp.
- Full Python suite stays green.
