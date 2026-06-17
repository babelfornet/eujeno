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

## Ridondanza e failover

Avvia **più nodi che servono lo stesso blocco** per la resilienza: se un nodo cade durante un job, il coordinator lo esclude e **riavvia la generazione** sui nodi rimasti (serve almeno un holder per ogni blocco).

```bash
# blocco 12-24 + head serviti da DUE nodi (B e C): se B cade, il job continua su C
synapse serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # nodo B
synapse serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head"   # nodo C (ridondante)
```

La risposta di `infer` include `"failovers": N` (quanti reinstradamenti sono serviti). Se nessun nodo ridondante copre il blocco caduto, `infer` risponde `NOT_OPERATIONAL`.

> Nota: in questo Milestone 0 il failover **riavvia** la generazione da capo (semplice e corretto). Il re-dispatch per-hop con replay del prefisso, che preserva il progresso, e lo store-and-forward durevole su disco sono approfondimenti successivi (vedi [ADR-0001](../decisions/ADR-0001-implementation-forks.md) Fork C).

## Trade-off

Il coordinator instrada tutto il traffico → è un **punto centrale** (Milestone 0), a differenza della modalità [P2P puro](./p2p.md) che non ha server centrale ma richiede nodi mutuamente raggiungibili. Il P2P puro anche dietro NAT (libp2p nativo) è sul percorso futuro. Vedi [ADR-0002](../decisions/ADR-0002-connettivita-nat.md).
