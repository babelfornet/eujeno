# Axyn frontend (`axyn ui`)

Every node can launch its own local dashboard:

```bash
axyn ui --coordinator http://COORDINATOR_IP:9000 --port 8500
# then open http://127.0.0.1:8500
```

What it offers (Phase 1):
- **Network status**: connected nodes, how the model is assembled across the layers (EMBED → decoder blocks → HEAD), coverage, memory, and whether the model is **operational**. A graph shows the nodes around the coordinator.
- **Chat**: query the distributed model (enabled only when the network covers all layers). It also shows how to connect other clients (CLI / cURL / OpenAI).

The browser talks **only** to the local `axyn ui` server, which proxies to the coordinator (no CORS issues).

## Creating or joining a network from the UI ("Management" tab)

From the **Management** tab you can control the local node without the CLI:
- **Target coordinator**: change the URL of the coordinator the dashboard is connected to.
- **Create a network**: start a local **coordinator** (choose model and port); the dashboard points to it automatically.
- **Join a network**: start a local `serve` node that connects to a coordinator with your **stages** (e.g. `embed,decoder:0-12`).
- **Local node**: see the status of the processes started from the UI (coordinator/worker, pid) and stop them with **Stop**.

### P2P mode (without a coordinator)

In the Management tab, with "Network mode → P2P", you can create/join a network **without a coordinator**: the nodes discover each other via **gossip** and **every node is queryable** (it exposes both `/registry` and `/v1/chat/completions`). The dashboard and chat work by pointing the "Target coordinator" at the URL of **any peer**. This requires the nodes to be able to reach each other (LAN/VPN/public IPs).

> Security: `axyn ui` listens on `127.0.0.1` and starts processes on **your** machine (`python -m axyn coordinator|serve`). Use it only locally / in a trusted environment.

## MCP tools ("MCP" tab)

`axyn ui` acts as an **MCP host**: from the **MCP** tab you configure MCP servers (stdio) and their tools become usable by the model.

- **Add MCP server**: name + command + args (e.g. command `npx`, args `@modelcontextprotocol/server-filesystem /path`). The discovered tools appear in the list.
- **Enable "use MCP tools"** (toggle): in chat, requests pass the tools to the model; when the model calls a tool, `axyn ui` **runs** it on the MCP server and sends back the result (tool-calling loop). Below the response you see which tools were used (`🔧 name → result`).

> Requires a model that supports **tool calling**: with Qwen 0.5B it's demonstrative (the mechanism works, but the model calls tools unreliably); with a 7B+ it becomes useful. For now, MCP servers are **stdio** only.
