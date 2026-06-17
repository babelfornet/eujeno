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

In arrivo: **creare / aggiungersi** a una rete direttamente dal frontend (Fase 2) e **configurare tool MCP** (Fase 3).
