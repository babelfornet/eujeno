# Frontend Phase 3 — MCP tool configuration and usage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** From the dashboard you configure **MCP servers**; the `axyn ui` server acts as an **MCP host** (connects via stdio, discovers the tools, passes them to the model as `tools`, and when the model calls a tool it **executes** it on the MCP server and returns the result — a tool-calling loop). Available when the model supports tools.

**Architecture:** `axyn ui` as MCP host: `McpRegistry` holds the MCP server configs and, via the `mcp` SDK (stdio, per-operation connection), exposes `list_tools()` (→ OpenAI format) and `call_tool()`. A pure loop `run_tool_loop(messages, tools, call_model, call_tool, max_iters)` orchestrates: model → `tool_calls` → MCP execution → `role:"tool"` → model → … → final response. New `/api/mcp/*` endpoints and an "agent" chat that uses the MCP tools. UI: **MCP** tab.

**Tech Stack:** Python · `mcp` SDK (stdio client + FastMCP for the test server) · FastAPI · the existing `axyn/ui/*`.

**Reality:** reliable tool-calling requires a capable model (7B+); with Qwen 0.5B it serves to verify the mechanism end-to-end. The UI enables MCP only when the network is operational (the model accepts `tools`).

**Out of scope:** MCP via SSE/HTTP (stdio only for now); persistent MCP sessions (per-operation connection); streaming.

---

## File Structure
```
pyproject.toml                  # MOD: + mcp dependency
axyn/ui/mcp.py               # NEW: McpRegistry (config + list_tools + call_tool via stdio)
axyn/ui/agent.py             # NEW: run_tool_loop (pure orchestration, testable)
axyn/ui/server.py            # MOD: /api/mcp/add|list|remove + chat-agent
axyn/ui/static/index.html    # MOD: MCP tab + tool activity in chat
tests/test_agent_loop.py        # NEW: run_tool_loop with fakes (fast)
tests/test_mcp_registry.py      # NEW: integration with a small Python MCP server (slow)
tests/_mcp_echo_server.py       # NEW: test MCP server (an "echo" tool)
docs/examples/frontend.md       # MOD
```

---

## Task 1: `mcp` dependency + `McpRegistry`

**Files:** modify `pyproject.toml`; create `axyn/ui/mcp.py`, `tests/_mcp_echo_server.py`, `tests/test_mcp_registry.py`.

- [ ] **Step 1: add `mcp` to the dependencies** in `pyproject.toml` (`dependencies`): `"mcp>=1.0"`. Install: `cd /Users/alberto/Projects/AI/axyn && .venv/bin/pip install -e ".[dev]"`. If not installable, BLOCKED.

- [ ] **Step 2: create the test MCP server `tests/_mcp_echo_server.py`**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test")


@mcp.tool()
def echo(text: str) -> str:
    """Echoes back the received text."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: test `tests/test_mcp_registry.py`**
```python
import sys, os
import pytest
from axyn.ui.mcp import McpRegistry

_SERVER = os.path.join(os.path.dirname(__file__), "_mcp_echo_server.py")


@pytest.mark.slow
def test_list_and_call_tool_via_stdio():
    reg = McpRegistry()
    reg.add("echo", sys.executable, [_SERVER])
    tools = reg.list_tools()                      # OpenAI format
    names = [t["function"]["name"] for t in tools]
    assert any(n.endswith("echo") for n in names)
    full = next(n for n in names if n.endswith("echo"))
    out = reg.call_tool(full, {"text": "hi"})
    assert "echo: hi" in out
    assert reg.list_servers() == ["echo"]
    reg.remove("echo")
    assert reg.list_servers() == []
```

- [ ] **Step 4: run FAIL** — `... pytest tests/test_mcp_registry.py -m slow -v` → ImportError on `axyn.ui.mcp`.

- [ ] **Step 5: implement `axyn/ui/mcp.py`** (per-operation stdio connection; tool names prefixed `server__tool`):
```python
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpRegistry:
    """MCP host: holds the server configs and opens a stdio session per operation.
    The tool names exposed to the model are prefixed 'server__tool' to avoid collisions."""
    def __init__(self):
        self._servers = {}   # name -> {"command": str, "args": [str]}

    def add(self, name: str, command: str, args=None) -> None:
        self._servers[name] = {"command": command, "args": list(args or [])}

    def remove(self, name: str) -> None:
        self._servers.pop(name, None)

    def list_servers(self) -> list:
        return list(self._servers.keys())

    async def _session(self, name):
        cfg = self._servers[name]
        params = StdioServerParameters(command=cfg["command"], args=cfg["args"])
        return stdio_client(params)

    async def _alist_tools(self):
        out = []
        for name, cfg in self._servers.items():
            params = StdioServerParameters(command=cfg["command"], args=cfg["args"])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.list_tools()
                    for t in res.tools:
                        out.append({
                            "type": "function",
                            "function": {
                                "name": f"{name}__{t.name}",
                                "description": t.description or "",
                                "parameters": t.inputSchema or {"type": "object", "properties": {}},
                            },
                            "_server": name, "_tool": t.name,
                        })
        return out

    def list_tools(self) -> list:
        """Tools from ALL servers, in OpenAI format (with _server/_tool for routing)."""
        return asyncio.run(self._alist_tools())

    async def _acall_tool(self, full_name: str, arguments: dict) -> str:
        server, _, tool = full_name.partition("__")
        cfg = self._servers[server]
        params = StdioServerParameters(command=cfg["command"], args=cfg["args"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(tool, arguments or {})
                parts = []
                for c in res.content:
                    parts.append(getattr(c, "text", "") or "")
                return "\n".join(p for p in parts if p)

    def call_tool(self, full_name: str, arguments: dict) -> str:
        return asyncio.run(self._acall_tool(full_name, arguments))
```

- [ ] **Step 6: run PASS** — `... pytest tests/test_mcp_registry.py -m slow -v` → PASS. If the API of the installed `mcp` SDK differs (imports or names), open the installed package and adapt (the API: `mcp.client.stdio.stdio_client`, `mcp.ClientSession`, `session.initialize/list_tools/call_tool`, `tool.name/description/inputSchema`, `result.content[].text`); do NOT weaken the test.

- [ ] **Step 7: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add pyproject.toml axyn/ui/mcp.py tests/_mcp_echo_server.py tests/test_mcp_registry.py && git commit -m "feat(ui): McpRegistry (stdio MCP host: list_tools/call_tool)"
```

---

## Task 2: `run_tool_loop` (pure orchestration)

**Files:** create `axyn/ui/agent.py`, `tests/test_agent_loop.py`.

- [ ] **Step 1: test `tests/test_agent_loop.py`**
```python
import json
from axyn.ui.agent import run_tool_loop


def test_loop_executes_tool_then_finishes():
    # call_model: 1st round requests the tool, 2nd round responds
    calls = {"n": 0}

    def call_model(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c0", "type": "function",
                                    "function": {"name": "echo__echo", "arguments": json.dumps({"text": "hi"})}}]}
        return {"role": "assistant", "content": "The tool said: echo: hi"}

    executed = []

    def call_tool(name, args):
        executed.append((name, args))
        return "echo: hi"

    result = run_tool_loop([{"role": "user", "content": "use echo"}],
                           tools=[{"type": "function", "function": {"name": "echo__echo"}}],
                           call_model=call_model, call_tool=call_tool, max_iters=4)
    assert result["content"] == "The tool said: echo: hi"
    assert executed == [("echo__echo", {"text": "hi"})]
    assert any(s["role"] == "tool" for s in result["messages"])


def test_loop_no_tools_returns_immediately():
    def call_model(messages, tools):
        return {"role": "assistant", "content": "direct hi"}
    result = run_tool_loop([{"role": "user", "content": "hi"}], tools=[],
                           call_model=call_model, call_tool=lambda n, a: "", max_iters=4)
    assert result["content"] == "direct hi"
    assert result["tool_runs"] == []
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_agent_loop.py -v` → ImportError.

- [ ] **Step 3: implement `axyn/ui/agent.py`**
```python
import json


def run_tool_loop(messages, tools, call_model, call_tool, max_iters=6):
    """Tool-calling loop: calls the model; if it returns tool_calls it executes them (call_tool),
    sends the results back as role:'tool' messages, and repeats until the model gives a
    final response (or the iterations run out). call_model(messages, tools)->message dict;
    call_tool(name, args_dict)->str."""
    convo = list(messages)
    tool_runs = []
    last = {"role": "assistant", "content": ""}
    for _ in range(max_iters):
        last = call_model(convo, tools)
        tcs = last.get("tool_calls") or []
        if not tcs:
            break
        convo.append({"role": "assistant", "content": last.get("content"), "tool_calls": tcs})
        for tc in tcs:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            result = call_tool(fn["name"], args)
            tool_runs.append({"name": fn["name"], "arguments": args, "result": result})
            convo.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": str(result)})
    return {"content": last.get("content") or "", "messages": convo, "tool_runs": tool_runs}
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_agent_loop.py -v` → 2 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/agent.py tests/test_agent_loop.py && git commit -m "feat(ui): run_tool_loop (tool-calling orchestration, tested)"
```

---

## Task 3: MCP endpoints + chat-agent in the server

**Files:** modify `axyn/ui/server.py`; modify `tests/test_ui_server.py`.

- [ ] **Step 1: add a test to `tests/test_ui_server.py`**
```python
def test_mcp_add_list_remove():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    assert c.get("/api/mcp/list").json()["servers"] == []
    c.post("/api/mcp/add", json={"name": "fs", "command": "echo", "args": ["x"]})
    assert c.get("/api/mcp/list").json()["servers"] == ["fs"]
    c.post("/api/mcp/remove", json={"name": "fs"})
    assert c.get("/api/mcp/list").json()["servers"] == []
```
(NB: `/api/mcp/list` returns `{servers, tools}`; this test uses the command `echo`, which is not a real MCP server, so `list` must NOT try to connect in order to list the registered servers — `list_servers` is separate from `list_tools`. See Step 3.)

- [ ] **Step 2: run FAIL** — `... pytest tests/test_ui_server.py::test_mcp_add_list_remove -v`.

- [ ] **Step 3: modify `axyn/ui/server.py`**

Add imports at the top: `from axyn.ui.mcp import McpRegistry` and `from axyn.ui.agent import run_tool_loop` and `import json`. Inside `create_ui_app`, after `manager = NodeManager()`, add `mcp = McpRegistry()`. Add the endpoints (before `return app`):
```python
    @app.get("/api/mcp/list")
    async def mcp_list():
        servers = mcp.list_servers()
        tools = []
        if servers:
            try:
                tools = [{"name": t["function"]["name"], "description": t["function"]["description"]}
                         for t in mcp.list_tools()]
            except Exception as e:
                return {"servers": servers, "tools": [], "error": f"MCP error: {e}"}
        return {"servers": servers, "tools": tools}

    @app.post("/api/mcp/add")
    async def mcp_add(request: Request):
        body = await request.json()
        if not body.get("name") or not body.get("command"):
            return JSONResponse({"error": "name and command are required"}, status_code=400)
        mcp.add(body["name"], body["command"], body.get("args", []))
        return {"ok": True, "servers": mcp.list_servers()}

    @app.post("/api/mcp/remove")
    async def mcp_remove(request: Request):
        body = await request.json()
        mcp.remove(body.get("name", ""))
        return {"ok": True, "servers": mcp.list_servers()}
```
Add a chat with MCP tools. Extend `/api/chat`: if the body has `"use_mcp": true` and there are MCP servers, run the loop. Replace the `/api/chat` handler with:
```python
    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        use_mcp = bool(body.get("use_mcp")) and bool(mcp.list_servers())
        if not use_mcp:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    r = await client.post(f"{_coord()}/v1/chat/completions", json=body,
                                          headers={"content-type": "application/json"})
                return JSONResponse(r.json(), status_code=r.status_code)
            except Exception as e:
                return JSONResponse({"error": f"coordinator unreachable: {e}"}, status_code=502)
        # --- agent mode with MCP tools ---
        try:
            tools = mcp.list_tools()
        except Exception as e:
            return JSONResponse({"error": f"MCP error: {e}"}, status_code=502)
        clean_tools = [{"type": t["type"], "function": t["function"]} for t in tools]

        def call_model(messages, tls):
            payload = {"messages": messages, "tools": tls,
                       "max_tokens": int(body.get("max_tokens", 256)),
                       "temperature": body.get("temperature", 0.7)}
            with httpx.Client(timeout=300.0) as client:
                rr = client.post(f"{_coord()}/v1/chat/completions", json=payload)
            return rr.json()["choices"][0]["message"]

        out = await asyncio.to_thread(
            run_tool_loop, body.get("messages", []), clean_tools, call_model,
            lambda name, args: mcp.call_tool(name, args), 6)
        return {"choices": [{"message": {"role": "assistant", "content": out["content"]},
                             "finish_reason": "stop"}],
                "tool_runs": out["tool_runs"]}
```
Add `import asyncio` at the top of server.py.

- [ ] **Step 4: run PASS** — `... pytest tests/test_ui_server.py -v` (fast) and `-m slow -v` → PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/server.py tests/test_ui_server.py && git commit -m "feat(ui): MCP endpoints (add/list/remove) + chat-agent with MCP tools"
```

---

## Task 4: MCP tab in the frontend

**Files:** modify `axyn/ui/static/index.html`.

- [ ] **Step 1: extend the app** (do NOT rewrite it) with an **MCP** tab (next to Network/Chat/Management), same style:
  - **MCP servers**: a form to add a server (name, command, space-separated args) → `POST /api/mcp/add`; a list of servers from `GET /api/mcp/list` with their **tools** (name + description) and a Remove button → `POST /api/mcp/remove`. Poll/refresh after every action. If `list` returns `error`, show it.
  - **Enable tools in chat**: a "use MCP tools" toggle (app state `useMcp`). When active and there are servers, the chat sends `use_mcp:true`.
  - In the **Chat view**: when `useMcp` is active, show a "MCP active (N tools)" badge; when a response includes `tool_runs`, show below the bubble which tools were called (name + short result). The chat call already goes through `/api/chat` (add `use_mcp: useMcp` to the body).
  - Note in the UI: "requires a model that supports tool-calling".

- [ ] **Step 2: structural check** — `cd /Users/alberto/Projects/AI/axyn && .venv/bin/python -c "h=open('axyn/ui/static/index.html').read(); assert '/api/mcp/add' in h and '/api/mcp/list' in h and 'use_mcp' in h and 'MCP' in h; print('mcp tab ok', len(h))"`

- [ ] **Step 3: serve test** — `... pytest tests/test_ui_server.py::test_serves_index_html -v` → PASS.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/static/index.html && git commit -m "feat(ui): MCP tab (server + tool config, toggle in chat)"
```

---

## Task 5: docs + suite

**Files:** modify `docs/examples/frontend.md`.

- [ ] **Step 1: update `docs/examples/frontend.md`** — "MCP tools (MCP tab)" section: add an MCP server (stdio command, e.g. a filesystem/echo server), enable "use MCP tools" in chat; when the model supports tool-calling, `axyn ui` executes the tools and shows the activity. Note: a capable model is needed (7B+); with 0.5B it is demonstrative. stdio only for now.

- [ ] **Step 2: full suite** — `... pytest -q -p no:warnings` → green.

- [ ] **Step 3: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add docs/examples/frontend.md && git commit -m "docs: MCP tools in the frontend (Phase 3)"
```

---

## Self-Review

**Coverage (#4 MCP config):** McpRegistry stdio host (Task 1) ✓; tool-calling loop (Task 2) ✓; /api/mcp/* endpoints + chat-agent (Task 3) ✓; MCP UI tab + toggle/activity (Task 4) ✓; docs (Task 5) ✓. "If the model allows it": the tools are passed via `/v1/chat/completions` (function calling already implemented); with a weak model it is demonstrative. SSE/persistent out of scope.

**Placeholder scan:** registry/loop/endpoint with complete code; MCP tab specified in detail + structural check.

**Type consistency:** `McpRegistry.add(name,command,args)/remove(name)/list_servers()/list_tools()->[openai tool]/call_tool(full_name,args)->str`; `run_tool_loop(messages,tools,call_model,call_tool,max_iters)->{content,messages,tool_runs}`; endpoints `/api/mcp/add|list|remove`, `/api/chat` with `use_mcp`. The frontend consumes exactly these.
```
