# Quickstart — coordinator-relay (LAN e internet, senza VPN)

Modalità B di [ADR-0002](../decisions/ADR-0002-connettivita-nat.md). I nodi worker si connettono **in uscita** al coordinator: funzionano dietro NAT su reti diverse **senza port-forwarding né VPN**. Solo il **coordinator** dev'essere raggiungibile (IP pubblico / VPS / un solo port-forward; in LAN un nodo qualsiasi).

```bash
pip install -e .

# 1) Coordinator — su una macchina raggiungibile dagli altri (es. IP pubblico 203.0.113.5)
synapse coordinator --model Qwen/Qwen2.5-0.5B-Instruct --port 9000

# 2) Nodo A (qualsiasi rete, dietro NAT) — embedding + primi 12 layer
synapse serve --coordinator ws://203.0.113.5:9000/node --stages "embed,decoder:0-12"

# 3) Nodo B (altra rete, dietro NAT) — ultimi 12 layer + head
synapse serve --coordinator ws://203.0.113.5:9000/node --stages "decoder:12-24,head"

# 4) Inferenza — client sottile, da qualunque rete
synapse --json infer --coordinator http://203.0.113.5:9000 --prompt "La capitale dell'Italia è"
```

- I nodi con `--coordinator` **non** espongono porte in ingresso: aprono una WebSocket in uscita verso il coordinator. Niente da configurare sul router dei worker.
- Il coordinator calcola la **coverage**: finché embed + tutti i range decoder + head non sono coperti, `infer` risponde `NOT_OPERATIONAL`.
- In LAN: metti il coordinator su un IP locale (es. `ws://192.168.1.10:9000/node`). Con una VPN: usa l'IP della VPN.

## Trade-off

Il coordinator instrada tutto il traffico → è un **punto centrale** (Milestone 0), a differenza della modalità [P2P puro](./p2p.md) che non ha server centrale ma richiede nodi mutuamente raggiungibili. Il P2P puro anche dietro NAT (libp2p nativo) è sul percorso futuro. Vedi [ADR-0002](../decisions/ADR-0002-connettivita-nat.md).
