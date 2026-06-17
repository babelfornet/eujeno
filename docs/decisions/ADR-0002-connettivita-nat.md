# ADR-0002 — Connettività cross-NAT: coordinator-relay (Milestone 0)

- **Stato:** Accettato
- **Data:** 2026-06-17
- **Contesto:** [ADR-0001](./ADR-0001-implementation-forks.md) (Fork A discovery), [PRD Parte 2](../prd/part-2-discovery-routing.md). Estende il transport di [Parte 1 networking](../plans/2026-06-17-part1-networking.md).

## Contesto e requisito

La Parte 2 deve fornire **discovery automatica** (niente più file di topologia statici). Requisito utente esplicito: il sistema **deve funzionare sia in intranet (LAN) sia su internet**, **senza obbligare all'uso di una VPN** (chi vuole può comunque configurarla).

## Forze

1. **Il NAT rompe il transport, non solo la discovery.** I block-server HTTP di Parte 1 sono raggiungibili solo se hanno un IP:porta pubblico (o port-forwarding). Due macchine entrambe dietro NAT non si raggiungono direttamente.
2. **NAT traversal senza VPN richiede infrastruttura pubblicamente raggiungibile.** Non esiste modo di connettere due peer entrambi dietro NAT senza *almeno un* punto pubblico che faccia da rendezvous/relay. Anche i sistemi P2P "puri" (libp2p, BitTorrent) usano bootstrap e **relay** quando l'hole-punching fallisce.
3. **libp2p nativo (hole-punching + relay) è la soluzione decentralizzata completa ma pesante/rischiosa** (daemon p2pd, relay, cross-arch — vedi ADR-0001 Fork A e Q1). Non è una prova rapida.
4. **Le connessioni in uscita attraversano sempre il NAT.** Un worker dietro NAT può sempre aprire una connessione *outbound* verso un endpoint pubblico.

## Decisione — due modalità selezionabili

Synapse supporta **due modalità di connettività**, scelte dall'utente; il coordinator è **opt-in**, mai obbligatorio.

### Modalità A — P2P puro (decentralizzato, default)

**Nessun server centrale.** Ogni nodo è un `synapse serve` simmetrico che:
- esegue una **discovery automatica via gossip**: conosce uno o più *seed peer*, scambia periodicamente il proprio registry (chi-serve-quale-blocco) con i vicini, con TTL/refresh per la liveness; un nuovo nodo impara l'intera rete transitivamente da un seed;
- riceve attivazioni via **transport diretto nodo-a-nodo** (HTTP di Parte 1).

`synapse infer --peer <qualsiasi-nodo>` interroga un peer, riceve il registry gossipato, costruisce la topologia da solo (coverage gate) ed esegue. **Funziona dove i nodi sono mutuamente raggiungibili** (LAN, VPN, o IP pubblici/port-forwarding). Per il P2P puro **anche dietro NAT senza VPN** serve un transport con NAT traversal (**libp2p nativo**: hole-punching + relay tra peer) — è il percorso futuro, dietro la stessa interfaccia.

### Modalità B — coordinator-relay (opt-in, per internet-senza-VPN subito)

Un **coordinator-relay** leggero come livello di connettività di Milestone 0:

- Un processo **`synapse coordinator`** pubblicamente raggiungibile (su internet: una macchina con IP pubblico o un VPS, o un solo port-forward; su LAN: un nodo qualsiasi).
- Ogni **`synapse serve`** apre una **connessione WebSocket in uscita** verso il coordinator, **annuncia i propri stage** (embed / decoder:lo-hi / head) e poi serve le richieste di hop relayate. Outbound ⇒ funziona dietro qualsiasi NAT, **senza port-forwarding sui worker**.
- Il coordinator mantiene il **registry** (quale nodo serve quale blocco), calcola la **coverage**, e **instrada** ogni hop di attivazione verso il nodo giusto sulla sua connessione WS, guidando il loop di generazione (orchestrator Milestone 0 spostato nel coordinator).
- **`synapse infer`** diventa un client sottile: invia `{prompt, max_new_tokens}` al coordinator e riceve `{text, tokens}`.

Connettività richiesta: **solo il coordinator** deve essere raggiungibile; worker ed entry hanno bisogno **solo di connettività in uscita** → funziona LAN e internet senza VPN. Con una VPN, anche il coordinator può stare su IP privato.

## Conseguenze

**Positive:**
- Funziona su intranet **e** internet senza VPN né port-forwarding sui worker (requisito soddisfatto).
- Discovery automatica reale: i nodi si auto-annunciano; l'entry non scrive topologie.
- Riusa l'esecuzione a blocchi e il wire safetensors già fatti; cambia solo *come* i messaggi raggiungono i nodi (WS relay invece di POST diretti).
- Affidabile e veloce da costruire rispetto a libp2p.

**Negative / debito (consapevole):**
- **Introduce un punto centrale** (il coordinator instrada tutto il traffico): deviazione dalla tesi "completamente decentralizzato senza server centrale". È **Milestone 0**, esplicitamente da superare.
- Il coordinator è un collo di bottiglia e un SPOF per le sessioni attive (mitigato in futuro: più coordinator / federazione, poi libp2p).

**Percorso di de-centralizzazione (futuro):** transport nativo **libp2p** dove ogni nodo con IP pubblico può fare da relay e si tenta l'hole-punching prima del relay; il coordinator diventa opzionale. Resta dietro un'interfaccia di transport/discovery così che lo swap non riscriva l'esecuzione.

**Modalità conservate:** il transport HTTP diretto + topologia statica di Parte 1 resta valido per uso **pura-LAN/VPN** senza coordinator (più decentralizzato quando la rete lo permette).

## Alternative scartate (per questo slice)

| Alternativa | Perché no (ora) |
|-------------|------------------|
| **VPN + gossip HTTP** | Ottimo ma **richiede** la VPN; l'utente non vuole obbligarla. Resta disponibile come modalità (Parte 1 HTTP diretto su IP della VPN). |
| **libp2p/hivemind nativo** | Soluzione decentralizzata completa ma pesante/rischiosa; non una prova rapida. È il percorso futuro. |
| **Solo rendezvous + hole-punching (no relay)** | L'hole-punching fallisce dietro NAT simmetrici; senza relay di fallback non è affidabile. |

## Riferimenti
- [ADR-0001](./ADR-0001-implementation-forks.md) Fork A + Q1 (LAN vs NAT) · [PRD Parte 2](../prd/part-2-discovery-routing.md)
