# Evaluation — Plugin Architecture for Eujeno

Date: 2026-06-21
Status: **Evaluation** (no code; decision document to choose what to build next)
Scope note: this is an *assessment + recommended decomposition*, not an implementation spec. Each phase below earns its own design → plan → build cycle.

## 1. Verdict

A plugin architecture is a **good fit and feasible**, but it should be **reframed**: this is not "bolt on features", it is *making Eujeno's already-modular subsystems composable, plus a thin policy layer*, governed by a **network manifest** chosen by the node that creates the network.

The single strongest justification is exactly the requested one: **one codebase serving the whole spectrum** from open/permissionless (crypto incentives, DDoS challenge-response) to closed/critical (trusted whitelist, E2E vector encryption, redundant coordinator), with the network creator selecting the posture.

Key finding: **all four plugins hook onto seams that already exist**, but the amount of genuinely-*new* logic varies sharply — from thin policy over already-built machinery to one new isolated subsystem:

- Thinnest — **incentives** and **accounting**: policy and backends over **already-implemented** reputation + receipts.
- New & small: **DDoS / admission** logic on the existing job-admission seam.
- New & isolated: **E2E vector encryption** (one clean chokepoint).
- New & optional backend: **on-chain ledger** (vs the lightweight default).
- New & substantial, **separate track**: **redundant/replicated coordinator** (an opt-in attack-resistance / availability feature for civilian-critical *and* military deployments).

The underlying machinery it builds on (receipts, reputation, routing bias, verification sampling, the admission point) is **already implemented or specced** — plugins mostly need to expose it through hooks.

## 2. Core idea — posture via a signed "network manifest"

The network-creating node publishes a **signed network manifest** declaring which plugins are enabled and their parameters. Joining nodes inherit it. This is what makes "the first node decides which plugins are enabled" concrete.

```
network-manifest (signed by the creator key)
  network_id, model_id, num_layers
  plugins:
    crypto:      { enabled, scheme, key_policy }
    accounting:  { enabled, backend: ledger|chain, ... }
    incentives:  { enabled, reward_model, requires: [verifier] }
    ddos:        { enabled, mode: whitelist|challenge|ratelimit, ... }
    coordinator: { mode: p2p|single|replicated, ... }
```

### The crux: manifest enforcement is posture-dependent

In a P2P network with no central authority, a malicious node can ignore "encryption required". So the manifest is **not uniformly enforceable** — and pretending it is would be a design lie. Enforcement model depends on the posture:

| Posture | How the manifest is enforced |
|---|---|
| **Open / permissionless** | Manifest is *advisory*; deviation is punished economically/socially via **reputation + sampled verification** (a node that skips required behavior loses reputation / fails verification and is routed around). |
| **Closed / critical / military** | **Trusted-node whitelist + mutual attestation at handshake**; nodes that do not present the manifest's required capabilities (e.g. valid key, attested binary) are **refused at connect time**. Enforcement is real because membership is gated. |

This split is itself a strong argument for the plugin design: the same manifest mechanism degrades gracefully from "enforced" (gated) to "incentivized" (open).

## 3. Per-plugin assessment

References are `file:line` into `eujeno/` (verified during exploration).

### A. Crypto plugin — E2E encryption of inter-node vectors  *(NEW, highest military value, well-isolated)*

- **Seam (one clean chokepoint):** `eujeno/net/framing.py` (`pack`/`unpack`) and `eujeno/net/wire.py` (`encode_tensors`/`decode_tensors`). Senders/receivers: `eujeno/net/node.py:12-22`, `eujeno/net/coordinator.py:70-81`, `eujeno/net/server.py:29-150`.
- **Shape:** **hop-by-hop encryption** between adjacent nodes (which is what "vectors between nodes" asks). The **routing header stays cleartext**; only the **tensor payload** is encrypted, so the coordinator relays ciphertext *blind* (it routes on the header, never sees activations).
- **Key model (posture-dependent):** PKI / pre-shared keys among whitelisted nodes (military); per-session key exchange (Noise/TLS-style) between adjacent hops (open).
- 🔴 **Hard tension — Crypto ⊥ BFT verification:** Eujeno's anti-garbage defense is *sampled recompute at the coordinator* (`eujeno/net/sampling.py:7-29`). If the coordinator can't see activations, it **can't verify** them. Resolution options, to be chosen at design time: (a) move verification **node-side** (a second holder recomputes and compares, reporting only a match/no-match bit), or (b) in encrypted mode, rely on the **trusted-node assumption** (acceptable for whitelisted/military, not for open). **Crypto and verification must be designed together.**
- 🟡 **Performance:** encrypting large activation tensors per hop costs CPU; measure. Mitigant: the store-and-forward model already tolerates high latency, so the overhead may be acceptable.

### B. Accounting plugin — receipts → ledger  *(MOSTLY EXISTS; the real decision is the backend)*

- **Seam (already implemented):** hop receipts written in `eujeno/net/jobstore.py:120` (fields at `:30-37`), exposed at `GET /jobs/{id}/receipts` (`eujeno/net/coordinator.py:305-307`). The measurement layer already records *who processed what*.
- **Decision — lightweight ledger vs blockchain → recommend: lightweight distributed ledger by default; blockchain as an optional pluggable backend.**
  - Eujeno's ethos is "BOINC, not real-time, latency-tolerant". A blockchain injects consensus latency, gas, and operational weight that fight that simplicity.
  - Receipts are already append-only and signable. A **hash-linked per-node receipt log + periodically gossiped checkpoints** gives tamper-evidence and auditability **without global consensus**.
  - A blockchain only earns its keep when you need **trustless global settlement with real money** against adversaries who would double-spend rewards — a narrow subset. Expose it behind the same `LedgerBackend` interface (`record(receipt)`, `balance(peer)`) so it's opt-in, not the default.

### C. Incentives plugin — crypto rewards  *(REFRAMES part-4; reputation exists, token reward deferred)*

- **Seam:** reputation reward/penalty already in `eujeno/net/coordinator.py:213` / `:220` (constants `:25-29`); routing bias in `eujeno/net/discovery.py:64-69`; receipts as above.
- **Reframe:** incentives are **policy over the accounting data**, not a new subsystem. Keep **measurement** (accounting — who did what, exists) separate from **reward** (incentives — convert work → tokens/credits, policy).
- 🔴 **Incentives ⊥ verification:** paying per-request *without* verification is farmable — nodes can return garbage to harvest rewards. **Incentives require the verifier to be integrated first** (`sampling.py` exists but is not yet wired into the hop path).
- **Spectrum confirmation:** in closed/military networks nodes are owned/contracted, not paid per request → incentives are the **clearest "off in military, on in open"** plugin, validating the posture model.

### D. DDoS / admission protection  *(NEW but small; clear seam; posture-dependent)*

- **Seam:** job admission at the infer entry point (`eujeno/net/coordinator.py:232-248`) and job creation (`eujeno/net/jobstore.py:55-102`).
- **Mechanisms by posture:** trusted **whitelist** (military) · **challenge-response** / proof-of-work to make spam costly (open) · per-peer **rate limits & quotas** · **reputation-gated admission** (reuse existing reputation).
- **Note on the async model:** in store-and-forward, a burst is *absorbed* by the durable queue; the real risk is **resource exhaustion** (queue/disk/compute), so quotas + reputation gating matter more than classic packet rate-limiting.

## 4. Separate track — redundant / replicated coordinator  *(opt-in attack-resistance, civilian-critical AND military)*

Currently a **single coordinator** (`eujeno/net/coordinator.py:38-309`, Milestone 0); federation is acknowledged but **not designed** (ADR-0002). Reframed per the requirement: this is an **opt-in security/availability option in both civilian-critical and military deployments** — it hardens the network against **single-relay takedown, DDoS on the relay, and censorship**.

Design space (trade-offs):

| Option | Resilience | NAT-friendliness | Complexity |
|---|---|---|---|
| **Pure P2P (exists)** | Highest — no SPOF at all | Needs reachability (LAN/VPN/public IP) | Low (built) |
| **Single coordinator (exists)** | Low — relay is a SPOF | Best (outbound only) | Low (built) |
| **Replicated coordinator cluster** | High — survives relay loss | Best | **High** — needs consensus |

Key insight: **pure P2P is already the maximally-redundant topology** (zero SPOF). The redundant coordinator is specifically for the **NAT-bound** case where a relay is required but a single one is unacceptable. A small **Raft cluster of coordinators** replicating the routing/registry state is the natural design — and the **durable SQLite job log is already the substrate** to replicate. Geo-replication adds censorship/takedown resistance.

Recommendation: for critical resilience, **prefer pure-P2P-over-VPN** where reachability allows (no relay SPOF); offer the **replicated-coordinator** as an opt-in manifest mode where a relay is unavoidable. Because it needs a consensus protocol, it is its **own project**, not part of the plugin framework.

## 5. Cross-cutting tensions & risks (design-time landmines)

1. 🔴 **Crypto ⊥ Verification** — encrypting activations from the coordinator breaks sampled-recompute BFT. Must be co-designed (node-side verification, or trusted-mode assumption).
2. 🔴 **Incentives ⊥ Verification** — rewards without verification are farmable; verifier is a prerequisite for incentives.
3. 🟠 **Plugins ⊥ the 3 inviolable primitives** — plugins must be **additive** and must not change: the DHT record schema (`eujeno/net/discovery.py`), the durable SQLite + safetensors substrate (`eujeno/net/jobstore.py:14-38`), or the `(job_id, stage)` idempotency key. Extra state goes in separate columns/tables/namespaces.
4. 🟠 **Determinism** — verification and any commit-reveal proofs are blocked on deterministic kernels (already noted in part-5). Encryption does not solve this.
5. 🟡 **Performance** — per-hop encryption of large tensors; measure before committing to defaults.
6. 🟡 **Manifest trust bootstrap** — the creator's signing key must be distributed to joiners out-of-band (or via the seed/coordinator URL); the manifest is only as trustworthy as that key distribution.

## 6. Recommended decomposition & phasing

Each item is an independent spec → plan → build cycle.

- **P0 — Plugin framework (thin).** A hook/dispatcher layer at the identified seams + the **network manifest** (creator declares enabled plugins + params; nodes inherit on join; **posture-dependent enforcement**: advisory+reputation vs whitelist+attestation). Config lives in `~/.eujeno/plugins.json` alongside the existing `node.json` / `mcp.json` precedent (`eujeno/net/nodeconfig.py`, `eujeno/mcp_config.py`). Small; unlocks everything.
- **P1 — DDoS / admission plugin.** Smallest, immediately useful, validates the framework end-to-end (whitelist + rate-limit + reputation-gated admission).
- **P2 — Accounting plugin (lightweight ledger).** Build on existing receipts; hash-linked log + gossiped checkpoints; `LedgerBackend` interface with a **blockchain adapter as opt-in**.
- **P3 — Verifier integration, then Incentives plugin.** First wire `sampling.py` into the hop path (prerequisite); then incentives as policy over accounting + reputation.
- **P4 — Crypto plugin (hop-by-hop encryption).** Designed **jointly** with the verification story; key model per posture.
- **Track R (separate, parallel) — Replicated coordinator.** Not a plugin but a network **topology mode**, so it enters the plan in *two* places: (1) its slot is **reserved in P0** as the manifest field `coordinator.mode: p2p | single | replicated`, so the option exists declaratively from day one; (2) the **implementation** is its own project, gated *only* on P0 and **independent of the plugins** (P1–P4). It is the highest-complexity new piece (needs a consensus protocol). Internal sub-phases: **R1** replicate routing/registry state across N coordinators (Raft; the durable SQLite job log is the replication substrate) → **R2** leader election + node/client failover → **R3** geo-distribution + split-brain handling.

### Dependency & ordering

```
P0  framework + manifest  (incl. the coordinator.mode slot)
 ├─ P1  DDoS / admission
 ├─ P2  Accounting (lightweight ledger)
 ├─ P3  Verifier integration → Incentives
 ├─ P4  Crypto (hop-by-hop)
 └─ Track R  Replicated coordinator     (gated only on P0; runs in parallel; lowest priority)
       R1 Raft state replication → R2 leader election + failover → R3 geo-distribution
```

- **Composition:** Track R pairs with **P1 (DDoS)** to harden the relay against takedown + flooding, and is compatible with **P4 (crypto)** because replicas relay **ciphertext blind** — no replica ever sees plaintext activations.
- **Priority:** build Track R **last / on demand**, not speculatively. Pure P2P is already maximally redundant (zero SPOF), so Track R only earns its cost where a relay is *required* (NAT-bound) **and** a single relay is unacceptable — the most expensive piece for a benefit pure-P2P partly already provides.

## 7. What to reframe vs the original proposal

1. "Plugin" overstates 3 of 4 — they are **thin policy modules** over existing seams; keep them light.
2. **Blockchain → default to a lightweight ledger**; blockchain only as an opt-in backend adapter.
3. **Redundant coordinator** → an **opt-in attack-resistance/availability option in both civilian-critical and military** contexts; note pure-P2P is already more resilient; scope it as a separate project.
4. **Crypto + BFT cannot both be naïve** — the security model (encryption vs verification) is the one place that must be designed holistically, up front.

## Appendix — seam reference (where each plugin hooks)

| Concern | Hook point(s) |
|---|---|
| Encryption | `eujeno/net/framing.py` (pack/unpack), `eujeno/net/wire.py` (encode/decode_tensors) |
| Admission / DDoS | `eujeno/net/coordinator.py:232-248` (POST /infer), `eujeno/net/jobstore.py:55-102` |
| Accounting / receipts | `eujeno/net/jobstore.py:120` (+ fields `:30-37`), `eujeno/net/coordinator.py:305-307` |
| Reputation / incentives | `eujeno/net/coordinator.py:213,220` (deltas, consts `:25-29`), `eujeno/net/discovery.py:64-69` |
| Verification (BFT) | `eujeno/net/sampling.py:7-29` |
| Discovery strategy | `eujeno/net/discovery.py:19-25` (DiscoveryProvider) |
| Config / manifest | `eujeno/net/nodeconfig.py:9-68`, `eujeno/mcp_config.py`, `eujeno/config.py` |
| Coordinator / replication | `eujeno/net/coordinator.py:38-309` (`_await_coverage:171-190`, `_generate_with_failover:192-230`) |

Inviolable primitives (do not modify): DHT record schema · durable SQLite + safetensors substrate (`jobstore.py:14-38`) · `(job_id, stage)` idempotency key. (ADR-0001.)
