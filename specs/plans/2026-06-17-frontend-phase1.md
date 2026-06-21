# Frontend Phase 1 — `eujeno ui` + real dashboard (network status + chat) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Every node launches `eujeno ui`: a local control server that serves a real frontend (rebuilt from the mock) used to **view the network status** (#2) and **run inference via chat** (#3), talking only to the local server (which acts as a proxy to the coordinator, no CORS).

**Architecture:** `eujeno ui --coordinator <url> --port 8500` starts a FastAPI that (a) serves `eujeno/ui/static/index.html`, (b) exposes `/api/config`, `/api/registry` (proxy GET to coordinator `/registry`), `/api/chat` (proxy POST to coordinator `/v1/chat/completions`). The frontend (React via CDN, single file, faithful to the mock) polls `/api/registry`, computes coverage client-side, and chats via `/api/chat`. Create/join-network (#1) and MCP (#4) are later phases on the same server.

**Tech Stack:** Python · FastAPI + httpx (proxy) · React 18 + Babel standalone via CDN (single-file frontend, no build) · IBM Plex (Google Fonts). Visual reference: `frontend/_mock/Eujeno Dashboard.dc.html`.

**Out of scope (phases 2/3):** starting coordinator/serve processes from the UI; MCP config + execution; streaming.

---

## File Structure
```
eujeno/ui/__init__.py          # NEW (empty)
eujeno/ui/server.py            # NEW: create_ui_app(coordinator_url) + /api/* endpoints
eujeno/ui/static/index.html    # NEW: real frontend (Network + Chat)
eujeno/cli.py                  # MOD: `ui` command
tests/test_ui_server.py         # NEW: proxy + serve index (slow: stub coordinator in thread)
specs/examples/frontend.md       # NEW: how to launch the UI
.gitignore                      # MOD: ignore frontend/_mock/
```

---

## Task 1: `eujeno/ui/server.py` (control server + proxy)

**Files:** create `eujeno/ui/__init__.py` (empty), `eujeno/ui/server.py`; create `tests/test_ui_server.py`. (index.html is created in Task 2; for now the serve test skips it or creates a minimal file — see Step 1.)

- [ ] **Step 1: test `tests/test_ui_server.py`**
```python
import socket, threading, time
import pytest, uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from eujeno.ui.server import create_ui_app


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _stub_coordinator():
    """A fake coordinator with /registry and /v1/chat/completions."""
    app = FastAPI()

    @app.get("/registry")
    async def reg():
        return {"num_layers": 24, "model": "stub",
                "nodes": [{"conn": "c1", "stages": {"embed": True, "head": True, "decoders": ["0-24"]}}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        return {"choices": [{"message": {"role": "assistant", "content": "hi!"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}
    return app


def _serve(app, port):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started: break
        time.sleep(0.05)
    assert srv.started
    return srv


def test_config_endpoint():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    assert c.get("/api/config").json()["coordinator_url"] == "http://example:9000"


def test_serves_index_html():
    app = create_ui_app("http://example:9000")
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "Eujeno" in r.text


@pytest.mark.slow
def test_proxies_registry_and_chat():
    port = _free_port()
    srv = _serve(_stub_coordinator(), port)
    try:
        app = create_ui_app(f"http://127.0.0.1:{port}")
        c = TestClient(app)
        reg = c.get("/api/registry").json()
        assert reg["num_layers"] == 24 and len(reg["nodes"]) == 1
        chat = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).json()
        assert chat["choices"][0]["message"]["content"] == "hi!"
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_ui_server.py -v` → ImportError.

- [ ] **Step 3: create `eujeno/ui/__init__.py`** (empty) and `eujeno/ui/server.py`:
```python
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_ui_app(coordinator_url: str) -> FastAPI:
    """Local control server: serves the frontend and acts as a proxy to the coordinator."""
    app = FastAPI()
    coord = coordinator_url.rstrip("/")

    def _index_html() -> str:
        path = os.path.join(_STATIC, "index.html")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        return "<!doctype html><title>Eujeno</title><h1>Eujeno UI</h1>"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _index_html()

    @app.get("/api/config")
    async def config():
        return {"coordinator_url": coord}

    @app.get("/api/registry")
    async def registry():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{coord}/registry")
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": f"coordinator unreachable: {e}"}, status_code=502)

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{coord}/v1/chat/completions", content=body,
                                      headers={"content-type": "application/json"})
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": f"coordinator unreachable: {e}"}, status_code=502)

    return app
```

- [ ] **Step 4: run PASS** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_ui_server.py -v` → 2 fast pass; `... -m slow -v` → proxy test pass. (`test_serves_index_html` passes with the minimal fallback until the real index.html exists.)

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/ui/__init__.py eujeno/ui/server.py tests/test_ui_server.py && git commit -m "feat(ui): local control server with proxy /api/registry and /api/chat"
```

---

## Task 2: real frontend `eujeno/ui/static/index.html`

**Files:** create `eujeno/ui/static/index.html`.

> Rebuild the design of the mock `frontend/_mock/Eujeno Dashboard.dc.html` as a real **single-file** app (React 18 + Babel standalone via CDN, no build). No Claude Design runtime. The frontend talks ONLY to the local server (`/api/*`, same origin).

- [ ] **Step 1: implement `eujeno/ui/static/index.html`** with these precise requirements:

**Tech:** `<script src="https://unpkg.com/react@18/umd/react.production.min.js">`, `react-dom@18`, `@babel/standalone`; a single `<script type="text/babel">` with the app. IBM Plex Sans/Mono fonts from Google Fonts. Inline styles (like the mock).

**State & data:** on startup `GET /api/config` → coordinator_url. Poll `GET /api/registry` every 2s → `{num_layers, model, nodes:[{conn,label?,stages:{embed,head,decoders:[]},mem_mb?,status?}]}`. Show "updated Xs ago". On 502/error → banner "coordinator unreachable".

**Coverage (computed client-side, identical to build_chain):**
- `hasEmbed` = some node with `stages.embed`; `hasHead` = some node with `stages.head`.
- decoder coverage: collect the `"lo-hi"` ranges, sort by lo, verify they tile `[0, num_layers)` without gaps.
- `operational = hasEmbed && hasHead && covered`. `coveredLayers`, `coveragePct`.
- `missing`: human-readable list (e.g. `"embed"`, `"head"`, `"layer 16-24"`).

**Layout (faithful to the mock):** dark theme `#070a10`, card `#0c1421`/border `#1b2a3f`, teal accent `#2dd4bf`, amber `#f3c46b`. Header with logo (reuse the mock's SVG), **Network**/**Chat** tabs, coordinator URL, and a status badge **OPERATIONAL** (teal) / **NOT OPERATIONAL** (amber).

- **Network view:** (1) "Model assembly" strip: `EMBED → [bar with segments colored per node, with a 0…num_layers ruler, uncovered gaps in red] → HEAD`, and `coveredLayers/num_layers · pct%`. (2) if incomplete, amber alert "MISSING FROM THE NETWORK" with the missing pieces + a copyable command `eujeno serve --coordinator <ws-url>/node --stages "<missing piece>"`. (3) "The network": `N nodes · coordinator <url>`, COVERAGE %, total MEMORY, and a **2D SVG graph** (coordinator at the center, nodes around it connected, color per node, status dot) — do NOT use three.js, SVG is enough. (4) "Node details": one card per node (label/conn, status, stage chips `embed`/`decoder X-Y`/`head`, memory).

- **Chat view:** if `!operational` → "🔒 The model is not assembled" screen with the missing pieces + command. If operational → "CONNECT A CLIENT" toolbar with copy buttons (CLI `eujeno infer --coordinator <url> --prompt …`, cURL to `<url>/v1/chat/completions`, OpenAI SDK snippet); message area (user/assistant bubbles, `failovers: N` badge if present, "typing" indicator); textarea input (Enter sends, Shift+Enter newline). Sending chat → `POST /api/chat` with `{"messages":[...history...],"max_tokens":256,"temperature":0.7}`; show `choices[0].message.content`. On `503`/error show the message.

**Suggestions** in an empty chat (3 clickable example prompts).

- [ ] **Step 2: verify the server serves it** — `cd /Users/alberto/Projects/AI/eujeno && .venv/bin/python -m pytest tests/test_ui_server.py::test_serves_index_html -v` → PASS (now it serves the real index; it must contain "Eujeno" and "Model assembly"). Update the assert if needed: the test only looks for "Eujeno".

- [ ] **Step 3: manual rendering smoke test** — starting with a stub and opening the browser is NOT automatable here; as a minimal check verify the HTML is well-formed:
`cd /Users/alberto/Projects/AI/eujeno && .venv/bin/python -c "import re,sys; h=open('eujeno/ui/static/index.html').read(); assert 'react' in h.lower() and 'Model assembly' in h and '/api/registry' in h and '/api/chat' in h; print('index ok', len(h), 'bytes')"`

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/ui/static/index.html && git commit -m "feat(ui): real frontend (Network + Chat views) faithful to the mock"
```

---

## Task 3: `eujeno ui` CLI command + docs

**Files:** modify `eujeno/cli.py`; create `specs/examples/frontend.md`; modify `.gitignore`, `README.md`.

- [ ] **Step 1: add the `ui` command in `eujeno/cli.py`** (after `coordinator`):
```python
@app.command()
def ui(
    coordinator: str = typer.Option("http://127.0.0.1:9000", "--coordinator", help="HTTP URL of the coordinator to connect to"),
    host: str = typer.Option("127.0.0.1", "--host", help="UI host"),
    port: int = typer.Option(8500, "--port", help="UI port"),
):
    """Start the local control frontend (network dashboard + chat)."""
    import uvicorn
    from eujeno.ui.server import create_ui_app
    typer.echo(f"eujeno ui: http://{host}:{port}  (coordinator={coordinator})", err=True)
    uvicorn.run(create_ui_app(coordinator), host=host, port=port, log_level="info")
```

- [ ] **Step 2: include the static files in the package** — in `pyproject.toml`, under `[tool.setuptools]` add (or extend) to include the package data:
```toml
[tool.setuptools.package-data]
"eujeno.ui" = ["static/*.html"]
```
(verify that the existing `[tool.setuptools.packages.find]` section stays valid.)

- [ ] **Step 3: `.gitignore`** — add a line `frontend/_mock/` (folder extracted from the mock, not to be versioned).

- [ ] **Step 4: `specs/examples/frontend.md`**
```markdown
# Eujeno frontend (`eujeno ui`)

Every node can launch its own local dashboard:

```bash
eujeno ui --coordinator http://COORDINATOR_IP:9000 --port 8500
# open http://127.0.0.1:8500
```

What it offers (Phase 1):
- **Network status**: connected nodes, model assembly across the layers, coverage, memory, and whether the model is operational.
- **Chat**: query the distributed model (active only when the network is complete); it also shows how to connect other clients (CLI/cURL/OpenAI).

The browser only talks to the local `eujeno ui` server, which acts as a proxy to the coordinator (no CORS).

Coming soon: create/join a network from the frontend (Phase 2) and configure MCP tools (Phase 3).
```

- [ ] **Step 5:** add a line to `README.md` (Quickstart section): "**Frontend:** `eujeno ui --coordinator http://IP:9000` → network dashboard + chat (see [specs/examples/frontend.md](specs/examples/frontend.md))."

- [ ] **Step 6: reinstall + verify the command**
`cd /Users/alberto/Projects/AI/eujeno && .venv/bin/pip install -e . >/dev/null 2>&1 && .venv/bin/eujeno --help | grep -q "ui" && echo "ui command ok"`

- [ ] **Step 7: full suite** — `... pytest -q -p no:warnings` → green.

- [ ] **Step 8: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py pyproject.toml .gitignore specs/examples/frontend.md README.md && git commit -m "feat(cli): 'eujeno ui' command (local frontend) + docs"
```

---

## Self-Review

**Coverage:** #2 network status (Network view: assembly + coverage + nodes + graph, Task 2) ✓; #3 chat (Chat view via /api/chat, Task 2) ✓; local control server that serves as the base for #1/#4 (Task 1) ✓; `eujeno ui` command (Task 3) ✓. #1 create/join and #4 MCP are explicitly phases 2/3.

**Placeholder scan:** the server and CLI have complete code; the frontend (Task 2) is specified in detail with reference to the mock (large visual artifact, not TDD-able line by line) + automatic structural checks (Steps 2/3).

**Type consistency:** `create_ui_app(coordinator_url) -> FastAPI`; endpoints `/api/config` `/api/registry` `/api/chat`; the frontend consumes exactly these and the shapes `/registry` (`num_layers`,`nodes[].stages`) and `/v1/chat/completions` (`choices[0].message.content`). Consistent with the existing coordinator.
```
