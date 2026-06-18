# ADR-0002 — Cross-NAT connectivity: coordinator-relay (Milestone 0)

- **Status:** Accepted
- **Date:** 2026-06-17
- **Context:** [ADR-0001](./ADR-0001-implementation-forks.md) (Fork A discovery), [PRD Part 2](../prd/part-2-discovery-routing.md). Extends the transport of [Part 1 networking](../plans/2026-06-17-part1-networking.md).

## Context and requirement

Part 2 must provide **automatic discovery** (no more static topology files). Explicit user requirement: the system **must work both on an intranet (LAN) and over the internet**, **without forcing the use of a VPN** (those who want one can still configure it).

## Forces

1. **NAT breaks the transport, not just discovery.** The Part 1 HTTP block-servers are only reachable if they have a public IP:port (or port-forwarding). Two machines both behind NAT cannot reach each other directly.
2. **NAT traversal without a VPN requires publicly reachable infrastructure.** There is no way to connect two peers both behind NAT without *at least one* public point acting as rendezvous/relay. Even "pure" P2P systems (libp2p, BitTorrent) use bootstrap and **relay** when hole-punching fails.
3. **Native libp2p (hole-punching + relay) is the complete decentralized solution but heavy/risky** (p2pd daemon, relay, cross-arch — see ADR-0001 Fork A and Q1). It is not a quick proof.
4. **Outbound connections always traverse the NAT.** A worker behind NAT can always open an *outbound* connection to a public endpoint.

## Decision — two selectable modes

Axyn supports **two connectivity modes**, chosen by the user; the coordinator is **opt-in**, never mandatory.

### Mode A — pure P2P (decentralized, default)

**No central server.** Each node is a symmetric `axyn serve` that:
- performs **automatic gossip-based discovery**: it knows one or more *seed peers*, periodically exchanges its own registry (who-serves-which-block) with neighbors, with TTL/refresh for liveness; a new node learns the whole network transitively from a seed;
- receives activations via **direct node-to-node transport** (Part 1 HTTP).

`axyn infer --peer <any-node>` queries a peer, receives the gossiped registry, builds the topology itself (coverage gate), and runs. **It works where nodes are mutually reachable** (LAN, VPN, or public IPs/port-forwarding). For pure P2P **even behind NAT without a VPN** a transport with NAT traversal is needed (**native libp2p**: hole-punching + relay between peers) — that is the future path, behind the same interface.

### Mode B — coordinator-relay (opt-in, for internet-without-VPN right now)

A lightweight **coordinator-relay** as the Milestone 0 connectivity layer:

- A publicly reachable **`axyn coordinator`** process (on the internet: a machine with a public IP or a VPS, or a single port-forward; on a LAN: any node).
- Each **`axyn serve`** opens an **outbound WebSocket connection** to the coordinator, **announces its stages** (embed / decoder:lo-hi / head), and then serves relayed hop requests. Outbound ⇒ works behind any NAT, **with no port-forwarding on the workers**.
- The coordinator maintains the **registry** (which node serves which block), computes **coverage**, and **routes** each activation hop to the right node over its WS connection, driving the generation loop (the Milestone 0 orchestrator moved into the coordinator).
- **`axyn infer`** becomes a thin client: it sends `{prompt, max_new_tokens}` to the coordinator and receives `{text, tokens}`.

Required connectivity: **only the coordinator** must be reachable; workers and the entry need **only outbound connectivity** → works on a LAN and over the internet without a VPN. With a VPN, even the coordinator can sit on a private IP.

## Consequences

**Positive:**
- Works on an intranet **and** over the internet without a VPN or port-forwarding on the workers (requirement met).
- Real automatic discovery: nodes self-announce; the entry writes no topologies.
- Reuses the block execution and the safetensors wire already built; only *how* messages reach the nodes changes (WS relay instead of direct POSTs).
- Reliable and fast to build compared to libp2p.

**Negative / debt (knowing):**
- **Introduces a central point** (the coordinator routes all traffic): a deviation from the "fully decentralized, no central server" thesis. It is **Milestone 0**, explicitly meant to be superseded.
- The coordinator is a bottleneck and an SPOF for active sessions (mitigated in the future: multiple coordinators / federation, then libp2p).

**Decentralization path (future):** native **libp2p** transport where every node with a public IP can act as a relay and hole-punching is attempted before relaying; the coordinator becomes optional. It stays behind a transport/discovery interface so that the swap does not rewrite execution.

**Preserved modes:** the direct HTTP transport + static topology of Part 1 remains valid for **pure-LAN/VPN** use without a coordinator (more decentralized when the network permits).

## Rejected alternatives (for this slice)

| Alternative | Why not (now) |
|-------------|------------------|
| **VPN + HTTP gossip** | Excellent but **requires** the VPN; the user does not want to force it. It remains available as a mode (Part 1 direct HTTP over the VPN IP). |
| **Native libp2p/hivemind** | Complete decentralized solution but heavy/risky; not a quick proof. It is the future path. |
| **Rendezvous + hole-punching only (no relay)** | Hole-punching fails behind symmetric NATs; without a relay fallback it is not reliable. |

## References
- [ADR-0001](./ADR-0001-implementation-forks.md) Fork A + Q1 (LAN vs NAT) · [PRD Part 2](../prd/part-2-discovery-routing.md)
