# Design — Eujeno documentation site (Nextra) served at `/docs`

Date: 2026-06-21
Status: approved (pending spec review)

## Goal

Add a full product documentation site, authored in **MDX** and compiled with
**Nextra**, served from the same deploy as the existing Vite landing page, at the
URL path `/docs`. All content in **English**, comprehensive (full page tree), with
not-yet-implemented subsystems clearly marked as future ("Design & Roadmap").

## Architecture

Nextra runs on **Next.js**; the existing landing is **Vite**. They are separate
toolchains that both emit static output, so they coexist in one GitHub Pages
deploy:

- Vite landing → `web/dist` → served at `/` (apex of the Pages site).
- Nextra docs → Next.js **static export** (`output: 'export'`) → copied into
  `web/dist/docs` → served at `/docs`.

The Pages artifact uploaded by CI is the combined `web/dist`.

### Base path (the one non-relative bit)

Next.js `basePath`/`assetPrefix` are absolute (unlike Vite's relative `base:'./'`).
The Pages site is at `https://babelfornet.github.io/eujeno/`, so docs assets live
under `/eujeno/docs`. Therefore:

- `next.config.mjs`: `basePath: process.env.BASE_PATH || ''`, same for `assetPrefix`,
  `images: { unoptimized: true }`, `output: 'export'`.
- CI sets `BASE_PATH=/eujeno/docs`.
- Future apex domain (`eujeno.com`): landing at `/`, docs at `/docs` → set
  `BASE_PATH=/docs`. One-variable change, documented.

## Repo layout change (folder rename)

Per the approved naming:

- **Rename `docs/` → `specs/`** — the internal engineering specs (vision, PRDs,
  ADRs, plans, examples, superpowers). Done with `git mv docs specs`.
- **New `docs/`** — the Nextra documentation *site* (Node/Next project, isolated
  from the Vite project in `web/`).

```
web/            # Vite landing (unchanged except a "Docs" link)
docs/           # NEW: Nextra docs site (next.config.mjs, theme, MDX content)
specs/          # was docs/: vision, prd/, decisions/, plans/, examples/, ...
.github/workflows/deploy.yml   # updated: build both, combine into web/dist/docs
```

### Rename ripple (references to update `docs/` → `specs/`)

Scoped: ~12 root-relative links in `README.md`, 1 in `CLAUDE.md`, 1 internal link
in `specs/plans/2026-06-17-frontend-phase1.md`. Relative cross-links inside the
folder stay valid. The React-error URL `reactjs.org/docs/...` in a built JS bundle
is a false positive and is NOT touched. README's "Documentation" section also gains
a pointer to the published docs site.

## Build & deploy integration

Extend `.github/workflows/deploy.yml` (already Node 24 + latest Pages actions):

1. Build the Vite landing: `cd web && npm ci && npm run build` → `web/dist`.
2. Build the Nextra docs: `cd docs && npm ci && BASE_PATH=/eujeno/docs npm run build`
   → `docs/out`.
3. Combine: `cp -r docs/out web/dist/docs`.
4. Upload `web/dist` as the Pages artifact (unchanged deploy job).

`setup-node` caches both `web/package-lock.json` and `docs/package-lock.json`.

## Landing integration

Add a **"Docs"** link in the landing nav and footer pointing to `docs/` (relative).
From `/eujeno/` it resolves to `/eujeno/docs/`; on the apex domain it resolves to
`/docs/`. Host-agnostic, consistent with the landing's relative-base approach. The
Vite landing is a single anchor-routed page, so it never intercepts `/docs` requests.

## Tech choices

- **Nextra 3** (Pages Router) with `nextra-theme-docs` — the most battle-tested path
  for `output: 'export'` on static hosts. (Re-evaluate Nextra 4 only if v3 blocks us;
  exact current config verified against Nextra docs during implementation.)
- Built-in client-side search (static index) — no server needed.
- MDX content with frontmatter; `_meta.{js,json}` files drive sidebar order/labels.

## Content plan (English, comprehensive)

Sidebar tree (each leaf = one MDX page). Commands/flags are taken from the **real
CLI** (verified by running `eujeno --help`), not invented.

```
Introduction                 what Eujeno is; "BOINC for LLM layers"; PoC status
Getting Started
  Installation               git clone + ./bin/eujeno; venv; pip install -e .
  Quickstart (single node)   eujeno up --model ...
  Your first query           eujeno infer / generate
Running a Node
  Join the swarm             eujeno serve --auto --peers
  Choosing layers & RAM      eujeno fit; --stages; --dtype; RAM math
  Pure P2P mode              gossip; --advertise; --peers
  Coordinator mode (NAT)     eujeno coordinator; --coordinator
Concepts
  Architecture overview      blocks, sharding, "operational" coverage
  The network                DHT discovery, router, allocator
  Store-and-forward jobs     durable queue, failover
CLI Reference                every command + global --json envelope
Models                       compatible models; deciding the split
Integrations
  OpenAI-compatible API      /v1/chat/completions
  Agents / Claude Code       LiteLLM in front
  MCP tools                  eujeno mcp; infer --mcp
Deployment                   Docker
Design & Roadmap             incentives, reputation, security/BFT, ADRs, roadmap
                             — clearly marked FUTURE / designed-on-paper
```

Sources: `README.md`, `specs/00-vision-architecture.md`, `specs/prd/*`,
`specs/decisions/*`, `specs/examples/*`, `CLAUDE.md`.

## Non-goals

- No rewrite of the Vite landing (only a Docs link added).
- No new deploy pipeline (extend the existing workflow).
- No content describing non-existent features as working — future work is labeled.
- No custom domain wiring in this change (BASE_PATH stays `/eujeno/docs`).

## Verification

- Local: `cd docs && BASE_PATH=/eujeno/docs npm run build` produces `docs/out`;
  combined `web/dist/docs/index.html` serves under a static server with assets 200.
- CI: workflow run green; live checks `…github.io/eujeno/docs/` → 200, an internal
  page → 200, `_next` asset → 200, and the landing "Docs" link resolves.
