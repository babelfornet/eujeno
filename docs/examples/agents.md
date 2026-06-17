# Collegare agenti AI a Synapse (API OpenAI-compatibile)

Quando il modello è OPERATIVO, il coordinator espone un'API **OpenAI-compatibile**: punta qualsiasi client/SDK OpenAI a `http://IL_COORDINATOR:9000/v1`.

Endpoint disponibili: `GET /v1/models`, `POST /v1/chat/completions` (con `temperature`, `top_p`, `max_tokens`, `repetition_penalty`, `seed`). Il chat template viene applicato automaticamente ai `messages`.

## SDK OpenAI (Python)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="qualsiasi")
r = client.chat.completions.create(
    model="synapse",
    messages=[{"role": "user", "content": "Scrivi un haiku sul mare"}],
    temperature=0.8, top_p=0.9, max_tokens=80,
)
print(r.choices[0].message.content)
```

## curl

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "synapse",
  "messages": [{"role":"user","content":"Ciao!"}],
  "temperature": 0.7, "max_tokens": 64
}'
```

## Claude Code e client Anthropic

Claude Code parla l'API **Anthropic**, non OpenAI. Mettici davanti **LiteLLM** come gateway (traduce Anthropic↔OpenAI) puntandolo a `http://IL_COORDINATOR:9000/v1`, poi:

```bash
ANTHROPIC_BASE_URL=http://LITELLM:4000 claude
```

Lo **streaming SSE** e un endpoint Anthropic nativo `/v1/messages` sono i prossimi passi (vedi [ROADMAP](../ROADMAP.md)).

## Tanti agenti in parallelo

Ogni richiesta è un **job** sulla rete e il coordinator gestisce job concorrenti. Per molti agenti contemporanei conviene:
- aggiungere **coda + repliche dei blocchi** (Parte 3) così le richieste si distribuiscono e c'è failover;
- per la **qualità**, splittare un **modello più grande** (es. Llama 3.x 8B/70B) su più nodi — l'infrastruttura è identica, cambiano solo dimensione e numero di nodi.

Il framing async/"BOINC" è ideale qui: molti agenti accodano e ricevono le risposte nel tempo, anche con latenze alte.
