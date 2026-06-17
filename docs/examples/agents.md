# Collegare agenti AI a Synapse (API OpenAI-compatibile)

> Un agente può anche **portare su una rete da zero**: `synapse models` elenca i modelli compatibili e `synapse up --model <id> [--dtype bfloat16]` avvia coordinator + un nodo che copre tutti i layer in un comando (`--dry-run` per anteprima). Vedi [CLAUDE.md](../../CLAUDE.md).

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

## Tool calling (e tool MCP)

`/v1/chat/completions` accetta il parametro `tools` (formato OpenAI) e, se il modello decide di chiamare un tool, ritorna `tool_calls` con `finish_reason: "tool_calls"`. I **tool MCP li esegue l'agente/host** (Claude Code, ecc.): il modello decide *quale* tool chiamare, l'agente lo esegue e rimanda il risultato come messaggio `role: "tool"`.

```python
tools = [{"type":"function","function":{
  "name":"get_weather","description":"Meteo di una città",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r = client.chat.completions.create(model="synapse",
      messages=[{"role":"user","content":"Che tempo fa a Roma?"}], tools=tools)
# r.choices[0].message.tool_calls -> [{function:{name:"get_weather", arguments:'{"city":"Roma"}'}}]
```

Nota: il tool-calling affidabile richiede un modello capace (7B+). Con Qwen 0.5B serve a verificare il meccanismo. La generazione si ferma alla fine-turno (EOS) e l'output è ripulito dai token speciali.

## Tool MCP da riga di comando

Configura i server MCP e usali nell'inferenza senza frontend:

```bash
# aggiungi un server MCP (stdio)
synapse mcp --add fs --command npx --args "@modelcontextprotocol/server-filesystem /percorso"
synapse --json mcp                 # elenca server + tool scoperti
# interroga il modello con i tool MCP (loop tool-calling)
synapse infer --coordinator http://IP:9000 --mcp --prompt "Elenca i file in /percorso"
synapse mcp --remove fs
```
La config è salvata in `~/.synapse/mcp.json` (override con `SYNAPSE_HOME`). `--mcp` richiede `--coordinator` o `--peer` (entrambi espongono `/v1`). Richiede un modello che supporti il tool-calling.

## Tanti agenti in parallelo

Ogni richiesta è un **job** sulla rete e il coordinator gestisce job concorrenti. Per molti agenti contemporanei conviene:
- aggiungere **coda + repliche dei blocchi** (Parte 3) così le richieste si distribuiscono e c'è failover;
- per la **qualità**, splittare un **modello più grande** (es. Llama 3.x 8B/70B) su più nodi — l'infrastruttura è identica, cambiano solo dimensione e numero di nodi.

Il framing async/"BOINC" è ideale qui: molti agenti accodano e ricevono le risposte nel tempo, anche con latenze alte.
