# Quickstart — pure P2P (decentralized, no central server)

Every node is the same: they discover each other via **gossip** (one seed is enough) and inference goes **directly** node-to-node. No coordinator. This requires the nodes to be able to reach each other (same LAN, a VPN, or public IPs/port-forwarding). For NAT-without-VPN, use the **coordinator mode** instead (see [coordinator.md](./coordinator.md), Mode B).

```bash
pip install -e .

# Node A — embedding + first 12 layers (first node, no seed)
axyn serve --stages "embed,decoder:0-12" --port 8001 --advertise http://192.168.1.10:8001

# Node B — last 12 layers + head; knows A as a seed
axyn serve --stages "decoder:12-24,head" --port 8001 \
  --advertise http://192.168.1.11:8001 --peers http://192.168.1.10:8001

# Inference: point at ANY single node; it discovers the rest on its own
axyn --json infer --peer http://192.168.1.10:8001 --prompt "The capital of Italy is"
```

- `--advertise` is the URL the node uses to announce itself to the others (it must be reachable by the other nodes, not `0.0.0.0`).
- `--peers` are the seeds from which to learn the network (comma-separated). The first node doesn't need any; the others list at least one. Knowledge propagates transitively.
- Until the **coverage** is complete (embed + all decoder ranges + head), `infer` responds `NOT_OPERATIONAL`. Add nodes with different ranges and the model **assembles progressively** across the network.

## Auto-assembly (`--auto`): the nodes split the layers among themselves

Instead of assigning stages by hand, each node can **claim** a range by reading the coverage gaps from the seed + its own RAM (see [ADR-0003](../decisions/ADR-0003-capacity-aware-allocation.md)):

```bash
# Node A (small): start it first, it takes a block that fits in its RAM
axyn serve --auto --port 8001 --advertise http://192.168.1.10:8001
# Node B (large): knows A as a seed and covers the COMPLEMENT (embed + rest + head)
axyn serve --auto --port 8001 --advertise http://192.168.1.11:8001 --peers http://192.168.1.10:8001
```

Options: `--ram <GB>` forces the memory budget (default: detected free RAM); `--reserve` the fraction kept for activations/KV-cache (default 0.2); `--target 2` aims for **≥2 nodes per range** (redundancy). The node announces its `capacity` in the gossip record.

> **Startup order:** claiming is a *one-time decision at startup*. Start the **seed(s)** first and wait for them to be operational (`curl SEED/registry` non-empty) **before** launching the other `--auto` nodes: if a node plans while the seed isn't ready yet, it sees an empty registry and claims too much. **Runtime re-balancing** (a node resizing its own range when the topology changes) is the next slice (slice 4 of ADR-0003); for now, for a guaranteed split, use explicit `--stages` or stagger the startup.

## On a LAN (quick, one machine or same WiFi)

```bash
axyn serve --stages "embed,decoder:0-12" --port 8001 --advertise http://127.0.0.1:8001
axyn serve --stages "decoder:12-24,head" --port 8002 --advertise http://127.0.0.1:8002 --peers http://127.0.0.1:8001
axyn --json infer --peer http://127.0.0.1:8001 --prompt "The capital of Italy is"
```

## Every node is queryable (symmetric entry point)

In P2P mode every `serve` node also exposes the **OpenAI API** `POST /v1/chat/completions` (and `/v1/models`): it orchestrates inference over the gossiped topology via direct transport, **without a coordinator**. Point an OpenAI client (or the dashboard) at **any peer**:

```bash
curl -s http://192.168.1.10:8001/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"axyn","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```
`axyn infer --peer` remains available as an alternative.

## When it's NOT enough

If the nodes are behind NAT on different networks **without a VPN**, the direct transport can't reach them: use the coordinator mode. Pure P2P even behind NAT (native libp2p: hole-punching + relay between peers) is on the future path — see [ADR-0002](../decisions/ADR-0002-nat-connectivity.md).
