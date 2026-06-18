# Frontend Fase 1 — `axyn ui` + dashboard reale (stato rete + chat) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Ogni nodo lancia `axyn ui`: un server di controllo locale che serve un frontend reale (ricostruito dal mock) con cui **vedere lo stato della rete** (#2) e **fare inferenza via chat** (#3), parlando solo col server locale (che fa da proxy al coordinator, niente CORS).

**Architecture:** `axyn ui --coordinator <url> --port 8500` avvia una FastAPI che (a) serve `axyn/ui/static/index.html`, (b) espone `/api/config`, `/api/registry` (proxy GET coordinator `/registry`), `/api/chat` (proxy POST coordinator `/v1/chat/completions`). Il frontend (React via CDN, single file, fedele al mock) fa polling di `/api/registry`, calcola la coverage lato client, e chatta via `/api/chat`. Crea/aggiungi-rete (#1) e MCP (#4) sono fasi successive sullo stesso server.

**Tech Stack:** Python · FastAPI + httpx (proxy) · React 18 + Babel standalone via CDN (frontend single-file, nessun build) · IBM Plex (Google Fonts). Riferimento visivo: `frontend/_mock/Axyn Dashboard.dc.html`.

**Fuori scope (fasi 2/3):** avvio processi coordinator/serve dal UI; config + esecuzione MCP; streaming.

---

## File Structure
```
axyn/ui/__init__.py          # NUOVO (vuoto)
axyn/ui/server.py            # NUOVO: create_ui_app(coordinator_url) + endpoint /api/*
axyn/ui/static/index.html    # NUOVO: frontend reale (Rete + Chat)
axyn/cli.py                  # MOD: comando `ui`
tests/test_ui_server.py         # NUOVO: proxy + serve index (slow: stub coordinator in thread)
docs/examples/frontend.md       # NUOVO: come lanciare la UI
.gitignore                      # MOD: ignora frontend/_mock/
```

---

## Task 1: `axyn/ui/server.py` (control server + proxy)

**Files:** create `axyn/ui/__init__.py` (empty), `axyn/ui/server.py`; create `tests/test_ui_server.py`. (index.html viene creato nel Task 2; per ora il test del serve lo salta o crea un file minimo — vedi Step 1.)

- [ ] **Step 1: test `tests/test_ui_server.py`**
```python
import socket, threading, time
import pytest, uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from axyn.ui.server import create_ui_app


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _stub_coordinator():
    """Un finto coordinator con /registry e /v1/chat/completions."""
    app = FastAPI()

    @app.get("/registry")
    async def reg():
        return {"num_layers": 24, "model": "stub",
                "nodes": [{"conn": "c1", "stages": {"embed": True, "head": True, "decoders": ["0-24"]}}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        return {"choices": [{"message": {"role": "assistant", "content": "ciao!"}, "finish_reason": "stop"}],
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
    assert "Axyn" in r.text


@pytest.mark.slow
def test_proxies_registry_and_chat():
    port = _free_port()
    srv = _serve(_stub_coordinator(), port)
    try:
        app = create_ui_app(f"http://127.0.0.1:{port}")
        c = TestClient(app)
        reg = c.get("/api/registry").json()
        assert reg["num_layers"] == 24 and len(reg["nodes"]) == 1
        chat = c.post("/api/chat", json={"messages": [{"role": "user", "content": "ciao"}]}).json()
        assert chat["choices"][0]["message"]["content"] == "ciao!"
    finally:
        srv.should_exit = True
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_ui_server.py -v` → ImportError.

- [ ] **Step 3: crea `axyn/ui/__init__.py`** (vuoto) e `axyn/ui/server.py`:
```python
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_ui_app(coordinator_url: str) -> FastAPI:
    """Server di controllo locale: serve il frontend e fa da proxy al coordinator."""
    app = FastAPI()
    coord = coordinator_url.rstrip("/")

    def _index_html() -> str:
        path = os.path.join(_STATIC, "index.html")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        return "<!doctype html><title>Axyn</title><h1>Axyn UI</h1>"

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
            return JSONResponse({"error": f"coordinator non raggiungibile: {e}"}, status_code=502)

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{coord}/v1/chat/completions", content=body,
                                      headers={"content-type": "application/json"})
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": f"coordinator non raggiungibile: {e}"}, status_code=502)

    return app
```

- [ ] **Step 4: run PASS** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_ui_server.py -v` → 2 fast pass; `... -m slow -v` → proxy test pass. (Il `test_serves_index_html` passa col fallback minimo finché non c'è il vero index.html.)

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/__init__.py axyn/ui/server.py tests/test_ui_server.py && git commit -m "feat(ui): control server locale con proxy /api/registry e /api/chat"
```

---

## Task 2: frontend reale `axyn/ui/static/index.html`

**Files:** create `axyn/ui/static/index.html`.

> Ricostruisci il design del mock `frontend/_mock/Axyn Dashboard.dc.html` come app reale **single-file** (React 18 + Babel standalone via CDN, nessun build). Niente runtime Claude Design. Il frontend parla SOLO col server locale (`/api/*`, stessa origin).

- [ ] **Step 1: implementa `axyn/ui/static/index.html`** con questi requisiti precisi:

**Tech:** `<script src="https://unpkg.com/react@18/umd/react.production.min.js">`, `react-dom@18`, `@babel/standalone`; un unico `<script type="text/babel">` con l'app. Font IBM Plex Sans/Mono da Google Fonts. Stili inline (come il mock).

**Stato & dati:** all'avvio `GET /api/config` → coordinator_url. Polling `GET /api/registry` ogni 2s → `{num_layers, model, nodes:[{conn,label?,stages:{embed,head,decoders:[]},mem_mb?,status?}]}`. Mostra "aggiornato Xs fa". Se 502/errore → banner "coordinator non raggiungibile".

**Coverage (calcolata lato client, identica a build_chain):**
- `hasEmbed` = qualche nodo con `stages.embed`; `hasHead` = qualche nodo con `stages.head`.
- copertura decoder: raccogli i range `"lo-hi"`, ordina per lo, verifica che tassellino `[0, num_layers)` senza buchi.
- `operational = hasEmbed && hasHead && coperto`. `coveredLayers`, `coveragePct`.
- `missing`: lista leggibile (es. `"embed"`, `"head"`, `"layer 16-24"`).

**Layout (fedele al mock):** tema scuro `#070a10`, card `#0c1421`/bordo `#1b2a3f`, accento teal `#2dd4bf`, ambra `#f3c46b`. Header con logo (riusa l'SVG del mock), tab **Rete**/**Chat**, URL coordinator, e un badge stato **OPERATIVO** (teal) / **NON OPERATIVO** (ambra).

- **Vista Rete:** (1) striscia "Assemblaggio del modello": `EMBED → [barra a segmenti colorati per nodo, con righello 0…num_layers, gap scoperti in rosso] → HEAD`, e `coveredLayers/num_layers · pct%`. (2) se incompleto, alert ambra "MANCANO ALLA RETE" coi pezzi mancanti + un comando copiabile `axyn serve --coordinator <ws-url>/node --stages "<pezzo mancante>"`. (3) "La rete": `N nodi · coordinator <url>`, COVERAGE %, MEMORIA totale, e un **grafo SVG 2D** (coordinator al centro, nodi attorno collegati, colore per nodo, pallino stato) — NON usare three.js, basta SVG. (4) "Dettaglio nodi": card per nodo (label/conn, stato, chip stage `embed`/`decoder X-Y`/`head`, memoria).

- **Vista Chat:** se `!operational` → schermata "🔒 Il modello non è assemblato" con i pezzi mancanti + comando. Se operational → toolbar "CONNETTI UN CLIENT" con bottoni copia (CLI `axyn infer --coordinator <url> --prompt …`, cURL su `<url>/v1/chat/completions`, snippet OpenAI SDK); area messaggi (bolle user/assistant, badge `failovers: N` se presente, indicatore "typing"); input textarea (Invio invia, Maiusc+Invio a capo). Invio chat → `POST /api/chat` con `{"messages":[...storico...],"max_tokens":256,"temperature":0.7}`; mostra `choices[0].message.content`. In caso di `503`/errore mostra il messaggio.

**Suggerimenti** in chat vuota (3 prompt d'esempio cliccabili).

- [ ] **Step 2: verifica che il server lo serva** — `cd /Users/alberto/Projects/AI/axyn && .venv/bin/python -m pytest tests/test_ui_server.py::test_serves_index_html -v` → PASS (ora serve l'index vero; deve contenere "Axyn" e "Assemblaggio"). Aggiorna l'assert se serve: il test cerca solo "Axyn".

- [ ] **Step 3: smoke manuale del rendering** — avvia con uno stub e apri il browser NON è automatizzabile qui; come check minimo verifica che l'HTML sia ben formato:
`cd /Users/alberto/Projects/AI/axyn && .venv/bin/python -c "import re,sys; h=open('axyn/ui/static/index.html').read(); assert 'react' in h.lower() and 'Assemblaggio' in h and '/api/registry' in h and '/api/chat' in h; print('index ok', len(h), 'bytes')"`

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/static/index.html && git commit -m "feat(ui): frontend reale (vista Rete + Chat) fedele al mock"
```

---

## Task 3: comando CLI `axyn ui` + docs

**Files:** modify `axyn/cli.py`; create `docs/examples/frontend.md`; modify `.gitignore`, `README.md`.

- [ ] **Step 1: aggiungi il comando `ui` in `axyn/cli.py`** (dopo `coordinator`):
```python
@app.command()
def ui(
    coordinator: str = typer.Option("http://127.0.0.1:9000", "--coordinator", help="URL HTTP del coordinator a cui collegarsi"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host della UI"),
    port: int = typer.Option(8500, "--port", help="Porta della UI"),
):
    """Avvia il frontend di controllo locale (dashboard rete + chat)."""
    import uvicorn
    from axyn.ui.server import create_ui_app
    typer.echo(f"axyn ui: http://{host}:{port}  (coordinator={coordinator})", err=True)
    uvicorn.run(create_ui_app(coordinator), host=host, port=port, log_level="info")
```

- [ ] **Step 2: includi i file statici nel package** — in `pyproject.toml`, sotto `[tool.setuptools]` aggiungi (o estendi) per includere i dati del package:
```toml
[tool.setuptools.package-data]
"axyn.ui" = ["static/*.html"]
```
(verifica che la sezione `[tool.setuptools.packages.find]` esistente resti valida.)

- [ ] **Step 3: `.gitignore`** — aggiungi una riga `frontend/_mock/` (cartella estratta dal mock, da non versionare).

- [ ] **Step 4: `docs/examples/frontend.md`**
```markdown
# Frontend di Axyn (`axyn ui`)

Ogni nodo può lanciare la propria dashboard locale:

```bash
axyn ui --coordinator http://IP_COORDINATOR:9000 --port 8500
# apri http://127.0.0.1:8500
```

Cosa offre (Fase 1):
- **Stato della rete**: nodi connessi, assemblaggio del modello sui layer, coverage, memoria, e se il modello è operativo.
- **Chat**: interroga il modello distribuito (attiva solo quando la rete è completa); mostra anche come collegare altri client (CLI/cURL/OpenAI).

Il browser parla solo col server locale `axyn ui`, che fa da proxy al coordinator (niente CORS).

In arrivo: creare/aggiungersi a una rete dal frontend (Fase 2) e configurare tool MCP (Fase 3).
```

- [ ] **Step 5:** aggiungi al `README.md` (sezione Quickstart) una riga: "**Frontend:** `axyn ui --coordinator http://IP:9000` → dashboard rete + chat (vedi [docs/examples/frontend.md](docs/examples/frontend.md))."

- [ ] **Step 6: reinstalla + verifica il comando**
`cd /Users/alberto/Projects/AI/axyn && .venv/bin/pip install -e . >/dev/null 2>&1 && .venv/bin/axyn --help | grep -q "ui" && echo "comando ui ok"`

- [ ] **Step 7: suite completa** — `... pytest -q -p no:warnings` → verde.

- [ ] **Step 8: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/cli.py pyproject.toml .gitignore docs/examples/frontend.md README.md && git commit -m "feat(cli): comando 'axyn ui' (frontend locale) + docs"
```

---

## Self-Review

**Coverage:** #2 stato rete (vista Rete: assemblaggio + coverage + nodi + grafo, Task 2) ✓; #3 chat (vista Chat via /api/chat, Task 2) ✓; server di controllo locale che fa da base per #1/#4 (Task 1) ✓; comando `axyn ui` (Task 3) ✓. #1 crea/aggiungi e #4 MCP esplicitamente fasi 2/3.

**Placeholder scan:** server e CLI hanno codice completo; il frontend (Task 2) è specificato in dettaglio con riferimento al mock (artefatto visivo grande, non TDD-abile riga per riga) + check strutturali automatici (Step 2/3).

**Type consistency:** `create_ui_app(coordinator_url) -> FastAPI`; endpoint `/api/config` `/api/registry` `/api/chat`; il frontend consuma esattamente questi e le forme `/registry` (`num_layers`,`nodes[].stages`) e `/v1/chat/completions` (`choices[0].message.content`). Coerente con coordinator esistente.
```
