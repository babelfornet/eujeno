# Quickstart — coordinator-relay (LAN and internet, without VPN)

Mode B of [ADR-0002](../decisions/ADR-0002-connettivita-nat.md). Worker nodes connect **outbound** to the coordinator: they work behind NAT on different networks **without port-forwarding or VPN**. Only the **coordinator** needs to be reachable (public IP / VPS / a single port-forward; on a LAN, any node).

```bash
pip install -e .

# 1) Coordinator — on a machine reachable by the others (e.g. public IP 203.0.113.5)
axyn coordinator --model Qwen/Qwen2.5-0.5B-Instruct --port 9000

# 2) Node A (any network, behind NAT) — embedding + first 12 layers
axyn serve --coordinator ws://203.0.113.5:9000/node --stages "embed,decoder:0-12"

# 3) Node B (different network, behind NAT) — last 12 layers + head
axyn serve --coordinator ws://203.0.113.5:9000/node --stages "decoder:12-24,head"

# 4) Inference — thin client, from any network
axyn --json infer --coordinator http://203.0.113.5:9000 --prompt "The capital of Italy is"
```

- Nodes using `--coordinator` do **not** expose inbound ports: they open an outbound WebSocket to the coordinator. Nothing to configure on the workers' router.
- The coordinator computes the **coverage**: until embed + all decoder ranges + head are covered, `infer` responds `NOT_OPERATIONAL`.
- On a LAN: put the coordinator on a local IP (e.g. `ws://192.168.1.10:9000/node`). With a VPN: use the VPN IP.

## Redundancy and failover

Start **multiple nodes serving the same block** for resilience: if a node goes down during a job, the coordinator excludes it and **restarts generation** on the remaining nodes (at least one holder is needed for each block).

```bash
# block 12-24 + head served by TWO nodes (B and C): if B goes down, the job continues on C
axyn serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # node B
axyn serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # node C (redundant)
```

The `infer` response includes `"failovers": N` (how many reroutes were served). If no redundant node covers the block that went down, `infer` responds `NOT_OPERATIONAL`.

> Note: in this Milestone 0, failover **restarts** generation from scratch (simple and correct). Per-hop re-dispatch with prefix replay, which preserves progress, and durable on-disk store-and-forward are later refinements (see [ADR-0001](../decisions/ADR-0001-implementation-forks.md), Fork C).

## Trade-off

The coordinator routes all traffic → it's a **central point** (Milestone 0), unlike the [pure P2P](./p2p.md) mode, which has no central server but requires nodes to be mutually reachable. Pure P2P even behind NAT (native libp2p) is on the future path. See [ADR-0002](../decisions/ADR-0002-connettivita-nat.md).
