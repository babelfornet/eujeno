# Connecting AI agents to Axyn (OpenAI-compatible API)

> An agent can even **bring up a network from scratch**: `axyn models` lists the compatible models, and `axyn up --model <id> [--dtype bfloat16]` starts a coordinator plus a node covering all layers in a single command (`--dry-run` for a preview). See [CLAUDE.md](../../CLAUDE.md).

When the model is OPERATIONAL, the coordinator exposes an **OpenAI-compatible** API: point any OpenAI client/SDK at `http://YOUR_COORDINATOR:9000/v1`.

Available endpoints: `GET /v1/models`, `POST /v1/chat/completions` (with `temperature`, `top_p`, `max_tokens`, `repetition_penalty`, `seed`). The chat template is applied automatically to the `messages`.

## OpenAI SDK (Python)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="anything")
r = client.chat.completions.create(
    model="axyn",
    messages=[{"role": "user", "content": "Write a haiku about the sea"}],
    temperature=0.8, top_p=0.9, max_tokens=80,
)
print(r.choices[0].message.content)
```

## curl

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "axyn",
  "messages": [{"role":"user","content":"Hi!"}],
  "temperature": 0.7, "max_tokens": 64
}'
```

## Claude Code and Anthropic clients

Claude Code speaks the **Anthropic** API, not OpenAI. Put **LiteLLM** in front of it as a gateway (it translates Anthropic↔OpenAI) and point it at `http://YOUR_COORDINATOR:9000/v1`, then:

```bash
ANTHROPIC_BASE_URL=http://LITELLM:4000 claude
```

**SSE streaming** and a native Anthropic `/v1/messages` endpoint are next on the list (see [ROADMAP](../ROADMAP.md)).

## Tool calling (and MCP tools)

`/v1/chat/completions` accepts the `tools` parameter (OpenAI format) and, if the model decides to call a tool, it returns `tool_calls` with `finish_reason: "tool_calls"`. **MCP tools are executed by the agent/host** (Claude Code, etc.): the model decides *which* tool to call, the agent runs it and sends the result back as a `role: "tool"` message.

```python
tools = [{"type":"function","function":{
  "name":"get_weather","description":"Weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r = client.chat.completions.create(model="axyn",
      messages=[{"role":"user","content":"What's the weather in Rome?"}], tools=tools)
# r.choices[0].message.tool_calls -> [{function:{name:"get_weather", arguments:'{"city":"Rome"}'}}]
```

Note: reliable tool calling requires a capable model (7B+). With Qwen 0.5B it's only good for verifying the mechanism. Generation stops at the end of the turn (EOS) and the output is stripped of special tokens.

## MCP tools from the command line

Configure MCP servers and use them in inference without a frontend:

```bash
# add an MCP server (stdio)
axyn mcp --add fs --command npx --args "@modelcontextprotocol/server-filesystem /path"
axyn --json mcp                 # list servers + discovered tools
# query the model with the MCP tools (tool-calling loop)
axyn infer --coordinator http://IP:9000 --mcp --prompt "List the files in /path"
axyn mcp --remove fs
```
The config is saved in `~/.axyn/mcp.json` (override with `AXYN_HOME`). `--mcp` requires `--coordinator` or `--peer` (both expose `/v1`). Requires a model that supports tool calling.

## Many agents in parallel

Each request is a **job** on the network, and the coordinator handles concurrent jobs. For many simultaneous agents, it's best to:
- add a **queue + block replicas** (Part 3) so requests are distributed and there's failover;
- for **quality**, split a **larger model** (e.g. Llama 3.x 8B/70B) across multiple nodes — the infrastructure is identical, only the size and number of nodes change.

The async/"BOINC" framing is ideal here: many agents enqueue and receive their responses over time, even with high latencies.
