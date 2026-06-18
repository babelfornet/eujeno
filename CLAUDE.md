# CLAUDE.md — guide for AI agents to the `axyn` CLI

This file teaches an agent (Claude Code or other) how to **drive Axyn from the CLI**. Axyn is a decentralized LLM inference network: a model is split into blocks of layers (`embed`, `decoder:lo-hi`, `head`) hosted by different nodes; the model is **operational** only when the blocks cover everything (`embed` + all decoder ranges + `head`).

> Mental model: "BOINC/SETI@home for the layers of an LLM". Asynchronous store-and-forward inference, tolerant of high latencies. Every node is a symmetric peer.

## Installation (after `git clone`)

```bash
./bin/axyn --help            # bootstrap: creates .venv + installs on first run, then executes
```

`bin/axyn` is auto-bootstrapping: the first time it creates `.venv` and runs `pip install -e .`, then forwards every command. Alternatively, manually:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e .
axyn --help
```

## AI-native output

Every command supports `--json` (a global flag, it goes **before** the command): it emits `{"ok": true|false, "command": "...", "data": {...}}` on stdout. Always use `--json` when consuming the output programmatically. Without `--json` the output is human-readable.

```bash
axyn --json model --info --model Qwen/Qwen2.5-0.5B-Instruct
```

## Key commands

| Command | What it's for |
|---|---|
| `axyn models` | Lists the **compatible** models/families (Llama/Qwen2) with examples. |
| `axyn model --info --model <id>` | Model dimensions (num_layers, hidden, ...) + `architecture` + `compatible`. Use it to **decide the split**. |
| `axyn fit --model <id> --ram <GB> [--dtype bfloat16]` | How many layers a node can hold with N GB of RAM + a **suggested stage spec** for `--stages`. |
| `axyn up --model <id> [--dtype bfloat16]` | One-command bring-up: starts a coordinator + a node that covers all layers. `--dry-run` prints the commands without starting them. |
| `axyn serve --stages "<spec>" ...` | Starts a node that hosts certain blocks. `--dtype` for large models. |
| `axyn serve --auto --peers <seed>` | **Self-assignment**: the node reads the coverage gaps from the seed + its own RAM and claims a range by itself (`--target 2` for redundancy, `--ram` to force the budget). |
| `axyn coordinator --port 9000` | Starts a coordinator (relay for nodes behind NAT). |
| `axyn infer --coordinator <url> --prompt "..."` | One-shot inference over the network. `--peer <url>` for pure P2P. |
| `axyn ui --coordinator <url>` | Local dashboard (network status, chat, MCP). |
| `axyn mcp --add <name> --command <cmd> --args "..."` | Configures MCP servers; `axyn infer --mcp` uses them in the tool-calling loop. |
| `axyn selfcheck` | Checks the environment/model. |
| `axyn schema` | Machine-readable schema of all commands/flags. |

## Which models can I use?

```bash
axyn --json models                                  # curated list (Llama/Qwen2)
axyn --json model --info --model <id>               # check compatible:true and num_layers
```

Compatible: **decoder-only Llama/Qwen2** architectures. Examples: `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B,32B,72B}-Instruct`, `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`.

## Deciding the split (layers ↔ RAM)

Each node loads **only the assigned layers** (partial loading): the required RAM is ~proportional to the number of hosted layers, not the whole model. Quick RAM estimate per block:

```
bytes_per_param = 4 (float32) | 2 (bfloat16/float16)
ram_layer ≈ params_per_layer × bytes_per_param
ram_node  ≈ Σ ram_layer of the hosted layers (+ embed/head if assigned)
```

`axyn model --info` gives `num_layers` and `hidden_size` to derive `params_per_layer`. For large models use `--dtype bfloat16` (halves the RAM) and/or split across more nodes. Full coverage = `embed` + all contiguous `decoder:0-N` ranges + `head`.

Shortcut: `axyn fit --model <id> --ram 4 --dtype bfloat16` does the math for you and prints the **suggested stage spec** (e.g. `decoder:0-7`) and how many layers you can hold. Decoder layers are ~equal to each other; the memory outliers are `embed`/`head` (the `vocab × hidden` matrix).

```bash
axyn --json fit --model Qwen/Qwen2.5-7B-Instruct --ram 4 --dtype bfloat16
# -> {"max_decoder_layers": 7, "suggested_stages": "decoder:0-7", "ram_per_layer_gb": 0.434, ...}
```

## Typical workflows

**a) Configure my node for a model and start everything (single-box):**
```bash
axyn models                                         # pick a compatible model
axyn up --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
```

**b) Join an existing network with my layers:**
```bash
axyn serve --coordinator ws://IP:9000/node --stages "decoder:12-24,head" \
  --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
```

**c) Query the distributed model:**
```bash
axyn --json infer --coordinator http://IP:9000 --prompt "Explain photosynthesis"
```

**d) Start the frontend:**
```bash
axyn ui --coordinator http://IP:9000      # then open http://127.0.0.1:8500
```

## Operational notes
- **Memory:** a 7B in float32 ≈ 28GB; in bfloat16 ≈ 14GB. Split across more nodes or use `--dtype bfloat16`.
- **NAT without VPN:** use coordinator mode (nodes connect outbound). On LAN/VPN/public IPs, pure P2P (`--peer`) is fine.
- **Operationality:** until coverage is complete, `infer` responds `NOT_OPERATIONAL`. Add nodes with the missing ranges.
- **OpenAI/Anthropic client models:** the coordinator exposes `/v1/chat/completions` (OpenAI). For Claude Code, put **LiteLLM** in front (see `docs/examples/agents.md`).
