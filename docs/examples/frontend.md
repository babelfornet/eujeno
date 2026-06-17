# Frontend di Synapse (`synapse ui`)

Ogni nodo può lanciare la propria dashboard locale:

```bash
synapse ui --coordinator http://IP_COORDINATOR:9000 --port 8500
# poi apri http://127.0.0.1:8500
```

Cosa offre (Fase 1):
- **Stato della rete**: nodi connessi, assemblaggio del modello sui layer (EMBED → blocchi decoder → HEAD), coverage, memoria, e se il modello è **operativo**. Un grafo mostra i nodi attorno al coordinator.
- **Chat**: interroga il modello distribuito (attiva solo quando la rete copre tutti i layer). Mostra anche come collegare altri client (CLI / cURL / OpenAI).

Il browser parla **solo** col server locale `synapse ui`, che fa da proxy al coordinator (niente problemi di CORS).

## Creare o aggiungersi a una rete dalla UI (tab "Gestione")

Dal tab **Gestione** puoi controllare il nodo locale senza CLI:
- **Coordinator bersaglio**: cambia l'URL del coordinator a cui la dashboard è collegata.
- **Crea una rete**: avvia un **coordinator** locale (scegli modello e porta); la dashboard si punta automaticamente su di esso.
- **Aggiungiti a una rete**: avvia un nodo `serve` locale che si connette a un coordinator coi tuoi **stage** (es. `embed,decoder:0-12`).
- **Nodo locale**: vedi lo stato dei processi avviati dalla UI (coordinator/worker, pid) e fermali con **Stop**.

> Sicurezza: `synapse ui` è in ascolto su `127.0.0.1` e avvia processi sulla **tua** macchina (`python -m synapse coordinator|serve`). Usalo solo in locale/fidato.

## Tool MCP (tab "MCP")

`synapse ui` fa da **host MCP**: dal tab **MCP** configuri server MCP (stdio) e i loro tool diventano usabili dal modello.

- **Aggiungi server MCP**: name + command + args (es. command `npx`, args `@modelcontextprotocol/server-filesystem /percorso`). I tool scoperti compaiono nella lista.
- **Abilita "usa tool MCP"** (toggle): in chat, le richieste passano i tool al modello; quando il modello chiama un tool, `synapse ui` lo **esegue** sul server MCP e rimanda il risultato (loop di tool-calling). Sotto la risposta vedi quali tool sono stati usati (`🔧 nome → risultato`).

> Richiede un modello che supporti il **tool-calling**: con Qwen 0.5B è dimostrativo (il meccanismo funziona, ma il modello chiama i tool in modo inaffidabile); con un 7B+ diventa utile. Per ora i server MCP sono solo **stdio**.

