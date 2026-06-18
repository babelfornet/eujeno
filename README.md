# Axyn

**A fully decentralized, peer-to-peer LLM inference network.** No central server: every node is a symmetric peer that hosts and runs one or more *blocks* of layers of an open-source model (from Hugging Face). User prompts are routed as **durable jobs** across the network of nodes responsible for the various blocks.

> **Guiding idea:** Axyn is not "real-time Petals". It's **"BOINC / SETI@home for the layers of an LLM"** — it tolerates very high latencies (hours, days, weeks) and treats inference as an asynchronous job that advances hop-by-hop in *store-and-forward* fashion. Giving up real-time makes failover and queueing simpler, not harder.

## Installation / Getting started

After `git clone`, get going right away with the auto-bootstrap launcher (it creates `.venv` and installs on first run):

```bash
./bin/axyn --help                 # first run: creates .venv + pip install -e . , then runs
```

Alternatively, manual installation:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e .
axyn --help
```

Single-node quickstart (starts a coordinator + one node that covers the whole model, in a single command):

```bash
axyn models                                   # which models can I use?
axyn up --model Qwen/Qwen2.5-0.5B-Instruct    # bring-up; --dtype bfloat16 for large models
```

> **AI agents:** the CLI is AI-native (`--json` on every command). See **[CLAUDE.md](CLAUDE.md)** for the guide on driving Axyn from an agent.

## Status

🚧 **PoC under construction.** Distributed inference across multiple nodes over HTTP already works (orchestrator-driven, Milestone 0): a model is split into layer blocks hosted by `axyn serve` on different nodes, and `axyn infer` runs generation across the network — reproducing the full model exactly. **Next steps:** DHT discovery (node self-organization), durable store-and-forward queue with failover. Token incentives are deferred (designed on paper).

**Goal of the first PoC:** distributed inference of a **1–3B** model across **2–3 real nodes**, with DHT discovery, an asynchronous queue, and automatic failover.

## Multi-node quickstart (PoC)

Three ways, pick based on your network:

- **[Pure P2P](docs/examples/p2p.md)** (decentralized, recommended) — nodes discover each other via **gossip**, no central server; the entry point targets any node and discovers the topology on its own. For LAN/VPN/public IPs.
  ```bash
  axyn serve --stages "embed,decoder:0-12" --port 8001 --advertise http://127.0.0.1:8001
  axyn serve --stages "decoder:12-24,head" --port 8002 --advertise http://127.0.0.1:8002 --peers http://127.0.0.1:8001
  axyn --json infer --peer http://127.0.0.1:8001 --prompt "The capital of Italy is"
  ```
- **[Coordinator](docs/examples/coordinator.md)** (opt-in) — for machines behind NAT on different networks **without a VPN**: nodes connect outbound to a reachable coordinator.
  ```bash
  axyn coordinator --port 9000                                                  # reachable machine
  axyn serve --coordinator ws://IP:9000/node --stages "embed,decoder:0-12"      # node A (any network)
  axyn serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"      # node B (any network)
  axyn --json infer --coordinator http://IP:9000 --prompt "The capital of Italy is"
  ```
- **Static topology** — a JSON file with the IPs, direct transport, no discovery:
  ```bash
  axyn serve --stages "embed,decoder:0-12" --port 8001
  axyn serve --stages "decoder:12-24,head" --port 8002
  axyn --json infer --topology docs/examples/topology.localhost.json --prompt "The capital of Italy is"
  ```

Machines download the model from Hugging Face on first run.

When the model is operational, the coordinator exposes an **OpenAI-compatible API** (`/v1/chat/completions`): you can connect agents and OpenAI clients to it (and Claude Code via LiteLLM). See **[docs/examples/agents.md](docs/examples/agents.md)**.

**Frontend:** `axyn ui --coordinator http://IP:9000` starts a local dashboard (network status + chat) → open `http://127.0.0.1:8500`. See **[docs/examples/frontend.md](docs/examples/frontend.md)**.

## Documentation

Everything is in [`docs/`](./docs/):

- **[docs/README.md](./docs/README.md)** — index and document map
- **[docs/ROADMAP.md](./docs/ROADMAP.md)** — project status, milestones, backlog
- **[docs/00-vision-architecture.md](./docs/00-vision-architecture.md)** — full vision and architecture (diagrams)
- **[docs/decisions/](./docs/decisions/)** — Architecture Decision Records
- **[docs/prd/](./docs/prd/)** — PRDs for the 5 subsystems

## Architecture in brief

Each symmetric node runs: a **Layer Executor** (transformer blocks on HF/PyTorch) · a **DHT Agent** (discovery: who-has-which-block) · a **Router** (routing + failover) · a **durable Job Store** (SQLite + blobs, store-and-forward) · an **Allocator** (self-assignment of discovered blocks) · a **Reputation/Verifier** (lightweight).

The model becomes **operational** only once every block is covered by ≥1 node; before that, requests queue up. As nodes join, the model **progressively assembles itself** in the network.

## Stack (PoC)

Python · Hugging Face `transformers` + `accelerate` + `safetensors` · `hivemind.DHT` (discovery) · SQLite (durable job store) · FastAPI/uvicorn (activation transport). Details and rationale in [ADR-0001](./docs/decisions/ADR-0001-implementation-forks.md).

## License

[Apache-2.0](./LICENSE). "Axyn" and the Axyn logo are trademarks of the project owner — see [TRADEMARKS.md](./TRADEMARKS.md) and [NOTICE](./NOTICE).
