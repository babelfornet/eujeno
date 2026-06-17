# Quickstart — P2P puro (decentralizzato, nessun server centrale)

Ogni nodo è uguale: si scoprono via **gossip** (basta un seed) e l'inferenza va **diretta** nodo-a-nodo. Nessun coordinator. Richiede che i nodi si raggiungano (stessa LAN, una VPN, o IP pubblici/port-forwarding). Per NAT-senza-VPN usa invece la **modalità coordinator** (vedi [coordinator.md](./coordinator.md), Modalità B).

```bash
pip install -e .

# Nodo A — embedding + primi 12 layer (primo nodo, nessun seed)
synapse serve --stages "embed,decoder:0-12" --port 8001 --advertise http://192.168.1.10:8001

# Nodo B — ultimi 12 layer + head; conosce A come seed
synapse serve --stages "decoder:12-24,head" --port 8001 \
  --advertise http://192.168.1.11:8001 --peers http://192.168.1.10:8001

# Inferenza: punta a UN nodo qualsiasi; scopre il resto da solo
synapse --json infer --peer http://192.168.1.10:8001 --prompt "La capitale dell'Italia è"
```

- `--advertise` è l'URL con cui il nodo si annuncia agli altri (deve essere raggiungibile dagli altri nodi, non `0.0.0.0`).
- `--peers` sono i seed da cui imparare la rete (separati da virgola). Il primo nodo non ne ha bisogno; gli altri ne indicano almeno uno. La conoscenza si propaga transitivamente.
- Finché la **coverage** non è completa (embed + tutti i range decoder + head), `infer` risponde `NOT_OPERATIONAL`. Aggiungi nodi con range diversi e il modello **si compone progressivamente** nella rete.

## In LAN (rapido, una macchina o stesso WiFi)

```bash
synapse serve --stages "embed,decoder:0-12" --port 8001 --advertise http://127.0.0.1:8001
synapse serve --stages "decoder:12-24,head" --port 8002 --advertise http://127.0.0.1:8002 --peers http://127.0.0.1:8001
synapse --json infer --peer http://127.0.0.1:8001 --prompt "La capitale dell'Italia è"
```

## Quando NON basta

Se i nodi sono dietro NAT su reti diverse **senza VPN**, il transport diretto non li raggiunge: usa la modalità coordinator. Il P2P puro anche dietro NAT (libp2p nativo: hole-punching + relay tra peer) è sul percorso futuro — vedi [ADR-0002](../decisions/ADR-0002-connettivita-nat.md).
