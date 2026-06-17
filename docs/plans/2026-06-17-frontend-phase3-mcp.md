# Frontend Fase 3 — configurazione e uso tool MCP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Dalla dashboard si configurano **server MCP**; il server `synapse ui` fa da **host MCP** (connette via stdio, scopre i tool, li passa al modello come `tools`, e quando il modello chiama un tool lo **esegue** sul server MCP e rimanda il risultato — loop di tool-calling). Disponibile quando il modello supporta i tool.

**Architecture:** `synapse ui` host MCP: `McpRegistry` tiene le config dei server MCP e, via l'SDK `mcp` (stdio, connessione per-operazione), espone `list_tools()` (→ formato OpenAI) e `call_tool()`. Un loop puro `run_tool_loop(messages, tools, call_model, call_tool, max_iters)` orchestra: modello → `tool_calls` → esecuzione MCP → `role:"tool"` → modello → … → risposta finale. Nuovi endpoint `/api/mcp/*` e una chat "agent" che usa i tool MCP. UI: tab **MCP**.

**Tech Stack:** Python · SDK `mcp` (stdio client + FastMCP per il server di test) · FastAPI · l'esistente `synapse/ui/*`.

**Realtà:** il tool-calling affidabile richiede un modello capace (7B+); con Qwen 0.5B serve a verificare il meccanismo end-to-end. La UI abilita MCP solo quando la rete è operativa (il modello accetta `tools`).

**Fuori scope:** MCP via SSE/HTTP (solo stdio per ora); sessioni MCP persistenti (connessione per-operazione); streaming.

---

## File Structure
```
pyproject.toml                  # MOD: + dipendenza mcp
synapse/ui/mcp.py               # NUOVO: McpRegistry (config + list_tools + call_tool via stdio)
synapse/ui/agent.py             # NUOVO: run_tool_loop (orchestrazione pura, testabile)
synapse/ui/server.py            # MOD: /api/mcp/add|list|remove + chat-agent
synapse/ui/static/index.html    # MOD: tab MCP + attività tool in chat
tests/test_agent_loop.py        # NUOVO: run_tool_loop con fake (veloce)
tests/test_mcp_registry.py      # NUOVO: integrazione con un piccolo server MCP Python (slow)
tests/_mcp_echo_server.py       # NUOVO: server MCP di test (un tool "echo")
docs/examples/frontend.md       # MOD
```

---

## Task 1: dipendenza `mcp` + `McpRegistry`

**Files:** modify `pyproject.toml`; create `synapse/ui/mcp.py`, `tests/_mcp_echo_server.py`, `tests/test_mcp_registry.py`.

- [ ] **Step 1: aggiungi `mcp` alle dipendenze** in `pyproject.toml` (`dependencies`): `"mcp>=1.0"`. Installa: `cd /Users/alberto/Projects/AI/synapse && .venv/bin/pip install -e ".[dev]"`. Se non installabile, BLOCKED.

- [ ] **Step 2: crea il server MCP di test `tests/_mcp_echo_server.py`**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test")


@mcp.tool()
def echo(text: str) -> str:
    """Ripete il testo ricevuto."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: test `tests/test_mcp_registry.py`**
```python
import sys, os
import pytest
from synapse.ui.mcp import McpRegistry

_SERVER = os.path.join(os.path.dirname(__file__), "_mcp_echo_server.py")


@pytest.mark.slow
def test_list_and_call_tool_via_stdio():
    reg = McpRegistry()
    reg.add("echo", sys.executable, [_SERVER])
    tools = reg.list_tools()                      # formato OpenAI
    names = [t["function"]["name"] for t in tools]
    assert any(n.endswith("echo") for n in names)
    full = next(n for n in names if n.endswith("echo"))
    out = reg.call_tool(full, {"text": "ciao"})
    assert "echo: ciao" in out
    assert reg.list_servers() == ["echo"]
    reg.remove("echo")
    assert reg.list_servers() == []
```

- [ ] **Step 4: run FAIL** — `... pytest tests/test_mcp_registry.py -m slow -v` → ImportError su `synapse.ui.mcp`.

- [ ] **Step 5: implementa `synapse/ui/mcp.py`** (connessione stdio per-operazione; nomi tool prefissati `server__tool`):
```python
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpRegistry:
    """Host MCP: tiene le config dei server e apre una sessione stdio per operazione.
    I nomi tool esposti al modello sono prefissati 'server__tool' per evitare collisioni."""
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
        """Tool di TUTTI i server, in formato OpenAI (con _server/_tool per il routing)."""
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

- [ ] **Step 6: run PASS** — `... pytest tests/test_mcp_registry.py -m slow -v` → PASS. Se l'API dell'SDK `mcp` installato differisce (import o nomi), apri il pacchetto installato e adatta (l'API: `mcp.client.stdio.stdio_client`, `mcp.ClientSession`, `session.initialize/list_tools/call_tool`, `tool.name/description/inputSchema`, `result.content[].text`); NON indebolire il test.

- [ ] **Step 7: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add pyproject.toml synapse/ui/mcp.py tests/_mcp_echo_server.py tests/test_mcp_registry.py && git commit -m "feat(ui): McpRegistry (host MCP stdio: list_tools/call_tool)"
```

---

## Task 2: `run_tool_loop` (orchestrazione pura)

**Files:** create `synapse/ui/agent.py`, `tests/test_agent_loop.py`.

- [ ] **Step 1: test `tests/test_agent_loop.py`**
```python
import json
from synapse.ui.agent import run_tool_loop


def test_loop_executes_tool_then_finishes():
    # call_model: 1° giro chiede il tool, 2° giro risponde
    calls = {"n": 0}

    def call_model(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c0", "type": "function",
                                    "function": {"name": "echo__echo", "arguments": json.dumps({"text": "ciao"})}}]}
        return {"role": "assistant", "content": "Il tool ha detto: echo: ciao"}

    executed = []

    def call_tool(name, args):
        executed.append((name, args))
        return "echo: ciao"

    result = run_tool_loop([{"role": "user", "content": "usa echo"}],
                           tools=[{"type": "function", "function": {"name": "echo__echo"}}],
                           call_model=call_model, call_tool=call_tool, max_iters=4)
    assert result["content"] == "Il tool ha detto: echo: ciao"
    assert executed == [("echo__echo", {"text": "ciao"})]
    assert any(s["role"] == "tool" for s in result["messages"])


def test_loop_no_tools_returns_immediately():
    def call_model(messages, tools):
        return {"role": "assistant", "content": "ciao diretto"}
    result = run_tool_loop([{"role": "user", "content": "ciao"}], tools=[],
                           call_model=call_model, call_tool=lambda n, a: "", max_iters=4)
    assert result["content"] == "ciao diretto"
    assert result["tool_runs"] == []
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_agent_loop.py -v` → ImportError.

- [ ] **Step 3: implementa `synapse/ui/agent.py`**
```python
import json


def run_tool_loop(messages, tools, call_model, call_tool, max_iters=6):
    """Loop di tool-calling: chiama il modello; se ritorna tool_calls li esegue (call_tool),
    rimanda i risultati come messaggi role:'tool', e ripete finché il modello dà una
    risposta finale (o si esauriscono le iterazioni). call_model(messages, tools)->message dict;
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
cd /Users/alberto/Projects/AI/synapse && git add synapse/ui/agent.py tests/test_agent_loop.py && git commit -m "feat(ui): run_tool_loop (orchestrazione tool-calling, testata)"
```

---

## Task 3: endpoint MCP + chat-agent nel server

**Files:** modify `synapse/ui/server.py`; modify `tests/test_ui_server.py`.

- [ ] **Step 1: aggiungi test a `tests/test_ui_server.py`**
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
(NB: `/api/mcp/list` ritorna `{servers, tools}`; questo test usa command `echo` che non è un vero server MCP, quindi `list` NON deve provare a connettersi per elencare i server registrati — `list_servers` è separato da `list_tools`. Vedi Step 3.)

- [ ] **Step 2: run FAIL** — `... pytest tests/test_ui_server.py::test_mcp_add_list_remove -v`.

- [ ] **Step 3: modifica `synapse/ui/server.py`**

Aggiungi import in cima: `from synapse.ui.mcp import McpRegistry` e `from synapse.ui.agent import run_tool_loop` e `import json`. Dentro `create_ui_app`, dopo `manager = NodeManager()`, aggiungi `mcp = McpRegistry()`. Aggiungi gli endpoint (prima di `return app`):
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
                return {"servers": servers, "tools": [], "error": f"errore MCP: {e}"}
        return {"servers": servers, "tools": tools}

    @app.post("/api/mcp/add")
    async def mcp_add(request: Request):
        body = await request.json()
        if not body.get("name") or not body.get("command"):
            return JSONResponse({"error": "name e command obbligatori"}, status_code=400)
        mcp.add(body["name"], body["command"], body.get("args", []))
        return {"ok": True, "servers": mcp.list_servers()}

    @app.post("/api/mcp/remove")
    async def mcp_remove(request: Request):
        body = await request.json()
        mcp.remove(body.get("name", ""))
        return {"ok": True, "servers": mcp.list_servers()}
```
Aggiungi una chat con tool MCP. Estendi `/api/chat`: se il body ha `"use_mcp": true` e ci sono server MCP, esegui il loop. Sostituisci l'handler `/api/chat` con:
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
                return JSONResponse({"error": f"coordinator non raggiungibile: {e}"}, status_code=502)
        # --- modalità agent con tool MCP ---
        try:
            tools = mcp.list_tools()
        except Exception as e:
            return JSONResponse({"error": f"errore MCP: {e}"}, status_code=502)
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
Aggiungi `import asyncio` in cima a server.py.

- [ ] **Step 4: run PASS** — `... pytest tests/test_ui_server.py -v` (fast) e `-m slow -v` → PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/ui/server.py tests/test_ui_server.py && git commit -m "feat(ui): endpoint MCP (add/list/remove) + chat-agent con tool MCP"
```

---

## Task 4: tab MCP nel frontend

**Files:** modify `synapse/ui/static/index.html`.

- [ ] **Step 1: estendi l'app** (NON riscriverla) con un tab **MCP** (accanto a Rete/Chat/Gestione), stesso stile:
  - **Server MCP**: form per aggiungere un server (name, command, args separati da spazio) → `POST /api/mcp/add`; lista dei server da `GET /api/mcp/list` con i loro **tool** (name + descrizione) e bottone Rimuovi → `POST /api/mcp/remove`. Polling/refresh dopo ogni azione. Se `list` ritorna `error`, mostralo.
  - **Abilita tool in chat**: un interruttore "usa tool MCP" (stato app `useMcp`). Quando attivo e ci sono server, la chat invia `use_mcp:true`.
  - In **vista Chat**: quando `useMcp` è attivo, mostra un badge "MCP attivo (N tool)"; quando una risposta include `tool_runs`, mostra sotto la bolla quali tool sono stati chiamati (name + risultato breve). La chiamata chat passa già da `/api/chat` (aggiungi `use_mcp: useMcp` al body).
  - Nota nella UI: "richiede un modello che supporti il tool-calling".

- [ ] **Step 2: check strutturale** — `cd /Users/alberto/Projects/AI/synapse && .venv/bin/python -c "h=open('synapse/ui/static/index.html').read(); assert '/api/mcp/add' in h and '/api/mcp/list' in h and 'use_mcp' in h and 'MCP' in h; print('mcp tab ok', len(h))"`

- [ ] **Step 3: serve test** — `... pytest tests/test_ui_server.py::test_serves_index_html -v` → PASS.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/ui/static/index.html && git commit -m "feat(ui): tab MCP (config server + tool, toggle in chat)"
```

---

## Task 5: docs + suite

**Files:** modify `docs/examples/frontend.md`.

- [ ] **Step 1: aggiorna `docs/examples/frontend.md`** — sezione "Tool MCP (tab MCP)": aggiungi un server MCP (comando stdio, es. un server filesystem/echo), abilita "usa tool MCP" in chat; quando il modello supporta il tool-calling, il `synapse ui` esegue i tool e mostra l'attività. Nota: serve un modello capace (7B+); con 0.5B è dimostrativo. Solo stdio per ora.

- [ ] **Step 2: suite completa** — `... pytest -q -p no:warnings` → verde.

- [ ] **Step 3: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add docs/examples/frontend.md && git commit -m "docs: tool MCP nel frontend (Fase 3)"
```

---

## Self-Review

**Coverage (#4 config MCP):** McpRegistry host stdio (Task 1) ✓; loop tool-calling (Task 2) ✓; endpoint /api/mcp/* + chat-agent (Task 3) ✓; tab MCP UI + toggle/attività (Task 4) ✓; docs (Task 5) ✓. "Se il modello lo permette": i tool si passano via `/v1/chat/completions` (function calling già implementato); con modello debole è dimostrativo. SSE/persistenti fuori scope.

**Placeholder scan:** registry/loop/endpoint con codice completo; tab MCP specificato in dettaglio + check strutturale.

**Type consistency:** `McpRegistry.add(name,command,args)/remove(name)/list_servers()/list_tools()->[openai tool]/call_tool(full_name,args)->str`; `run_tool_loop(messages,tools,call_model,call_tool,max_iters)->{content,messages,tool_runs}`; endpoint `/api/mcp/add|list|remove`, `/api/chat` con `use_mcp`. Il frontend consuma esattamente questi.
```
