# Connecting AI agents to Eujeno (OpenAI-compatible API)

> An agent can even **bring up a network from scratch**: `eujeno models` lists the compatible models, and `eujeno up --model <id> [--dtype bfloat16]` starts a coordinator plus a node covering all layers in a single command (`--dry-run` for a preview). See [CLAUDE.md](../../CLAUDE.md).

When the model is OPERATIONAL, the coordinator exposes an **OpenAI-compatible** API: point any OpenAI client/SDK at `http://YOUR_COORDINATOR:9000/v1`.

Available endpoints: `GET /v1/models`, `POST /v1/chat/completions` (with `temperature`, `top_p`, `max_tokens`, `repetition_penalty`, `seed`, and `stream`). The chat template is applied automatically to the `messages`. Message `content` may be a plain string or a list of OpenAI typed parts (`[{"type":"text","text":"…"}]`) — both are accepted.

**Streaming:** pass `"stream": true` to receive Server-Sent Events (`chat.completion.chunk` deltas ending with `finish_reason` and `data: [DONE]`). Eujeno generates the whole answer store-and-forward, so the stream is emitted in a few chunks rather than token-by-token — enough for streaming clients (e.g. the [PI](https://pi.dev) coding agent, IDE agents).

## OpenAI SDK (Python)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="anything")
r = client.chat.completions.create(
    model="eujeno",
    messages=[{"role": "user", "content": "Write a haiku about the sea"}],
    temperature=0.8, top_p=0.9, max_tokens=80,
)
print(r.choices[0].message.content)
```

## curl

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "eujeno",
  "messages": [{"role":"user","content":"Hi!"}],
  "temperature": 0.7, "max_tokens": 64
}'
```

## Claude Code and Anthropic clients

Claude Code speaks the **Anthropic** API, not OpenAI. Put **LiteLLM** in front of it as a gateway (it translates Anthropic↔OpenAI) and point it at `http://YOUR_COORDINATOR:9000/v1`, then:

```bash
ANTHROPIC_BASE_URL=http://LITELLM:4000 claude
```

**SSE streaming** is supported (`stream: true`, see above). A native Anthropic `/v1/messages` endpoint is still next on the list (see [ROADMAP](../ROADMAP.md)).

## PI coding agent

[PI](https://pi.dev) is OpenAI-compatible. Add Eujeno as a provider in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "eujeno": {
      "baseUrl": "http://127.0.0.1:9000/v1",
      "api": "openai-completions",
      "apiKey": "eujeno",
      "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
      "models": [ { "id": "eujeno", "name": "Eujeno (distributed)" } ]
    }
  }
}
```

Then `pi --provider eujeno --model eujeno`. PI streams (`stream: true`) and sends structured `content` parts and a large system prompt — all handled. Note: **driving a full agent harness like PI needs a capable model** (7B+, split across nodes if needed); a 1.5B connects and converses but tends to *describe* edits instead of emitting the tool calls, so files don't actually get written.

## Tool calling (and MCP tools)

`/v1/chat/completions` accepts the `tools` parameter (OpenAI format) and, if the model decides to call a tool, it returns `tool_calls` with `finish_reason: "tool_calls"`. **MCP tools are executed by the agent/host** (Claude Code, etc.): the model decides *which* tool to call, the agent runs it and sends the result back as a `role: "tool"` message.

```python
tools = [{"type":"function","function":{
  "name":"get_weather","description":"Weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r = client.chat.completions.create(model="eujeno",
      messages=[{"role":"user","content":"What's the weather in Rome?"}], tools=tools)
# r.choices[0].message.tool_calls -> [{function:{name:"get_weather", arguments:'{"city":"Rome"}'}}]
```

Note: tool calling improves sharply with model size. **Qwen2.5-0.5B** only exercises the *mechanism* — it works, but it often collapses to empty turns or emits truncated/malformed tool JSON. **~1.5B is the practical floor** for reliable structured tool-calling (`finish_reason: "tool_calls"` on the first try); a real agent that targets small models should still keep a retry + plain-codegen fallback. Generation stops at the end of the turn (EOS) and the output is stripped of special tokens.

## A self-contained agentic example

[`code_agent.py`](./code_agent.py) is a minimal PI/codex-style code-agent (pure standard library) that drives an Eujeno-served model end-to-end: it asks the model to call a `write_file` tool, falls back to plain code-gen when a tiny model can't drive tool-calls, then **runs the generated file and self-repairs it** — on failure it feeds the code + traceback back to the model, rewrites, and retries.

```bash
eujeno up --model Qwen/Qwen2.5-1.5B-Instruct        # an operational network on :9000
python specs/examples/code_agent.py                  # generate fib.py, run it -> 55
# watch the repair loop turn a broken file green:
EUJENO_SEED=$'def average(nums):\n    return sum(nums)/len(numbers)\n\nif __name__=="__main__":\n    assert average([2,4,6])==4\n    print("OK")' \
  EUJENO_FILE=stats.py python specs/examples/code_agent.py
```

It reads `EUJENO_BASE` (default `http://127.0.0.1:9000/v1`) and `EUJENO_MODEL` (default `eujeno`); see the file header for all knobs.

## MCP tools from the command line

Configure MCP servers and use them in inference without a frontend:

```bash
# add an MCP server (stdio)
eujeno mcp --add fs --command npx --args "@modelcontextprotocol/server-filesystem /path"
eujeno --json mcp                 # list servers + discovered tools
# query the model with the MCP tools (tool-calling loop)
eujeno infer --coordinator http://IP:9000 --mcp --prompt "List the files in /path"
eujeno mcp --remove fs
```
The config is saved in `~/.eujeno/mcp.json` (override with `EUJENO_HOME`). `--mcp` requires `--coordinator` or `--peer` (both expose `/v1`). Requires a model that supports tool calling.

## Many agents in parallel

Each request is a **job** on the network, and the coordinator handles concurrent jobs. For many simultaneous agents, it's best to:
- add a **queue + block replicas** (Part 3) so requests are distributed and there's failover;
- for **quality**, split a **larger model** (e.g. Llama 3.x 8B/70B) across multiple nodes — the infrastructure is identical, only the size and number of nodes change.

The async/"BOINC" framing is ideal here: many agents enqueue and receive their responses over time, even with high latencies.
