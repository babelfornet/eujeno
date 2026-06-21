# Eujeno — Documentation

**Eujeno** is a **fully decentralized, peer-to-peer** inference network for Large Language Models: no central server, every node is a symmetric peer that hosts and runs one or more *blocks* of layers of an open-source model (downloaded from Hugging Face). User queries are routed as **durable jobs** through the network of nodes responsible for the various blocks.

> **Guiding idea:** Eujeno is not "real-time Petals". It is **"BOINC / SETI@home for the layers of an LLM"** — it tolerates extremely high latencies (hours, days, weeks) and treats inference as an asynchronous job that advances hop-by-hop in *store-and-forward* fashion.

## Document map

| Document | Contents |
|-----------|-----------|
| [ROADMAP.md](./ROADMAP.md) | **Project status**: steps done / to do, milestones, backlog. The starting point for understanding "where we are". |
| [00-vision-architecture.md](./00-vision-architecture.md) | **Part 0** — Vision, goals, principles, component map, data flow. The approved architectural skeleton. |
| [decisions/](./decisions/) | **ADRs** (Architecture Decision Records): motivated technical decisions (e.g. which P2P substrate, which runtime). |
| [prd/](./prd/) | **PRDs** for each subsystem (Parts 1–5): Peer Node, Discovery & Routing, Queue & Load Balancing, Incentives & Reputation, Security & BFT. Plus the **[`eujeno` CLI](./prd/cli.md)** (AI-native), the entry point for all operations. |

## How the work is organized

The problem is large, so it is **compartmentalized** into independent parts. Each part follows the cycle:

```
spec (PRD)  →  plan (implementation)  →  build (code)  →  verify
```

**Part 0** is the umbrella document that holds everything together. **Parts 1–5** are the subsystems. **Cross-cutting decisions** (stack, libraries) live in the ADRs.

## Status at a glance

- **Current phase:** Architecture & Design (the PRDs are written before the code).
- **First PoC goal:** distributed inference of a **1–3B** model across **2–3 real nodes**, with DHT discovery, asynchronous queue and failover. **Tokens deferred.**
- Up-to-date detail always in [ROADMAP.md](./ROADMAP.md).
