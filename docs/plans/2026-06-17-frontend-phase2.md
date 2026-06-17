# Frontend Fase 2 — crea / aggiungi rete dal frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Dalla dashboard `synapse ui`, un nodo può **creare una rete** (avvia un coordinator locale) o **aggiungersi a una rete** (avvia un `serve --coordinator` coi propri stage), e gestire/fermare il processo locale — il tutto senza CLI.

**Architecture:** Il server `synapse ui` gestisce sottoprocessi tramite un `NodeManager` (Popen di `python -m synapse coordinator|serve …`). Nuovi endpoint `/api/network/create`, `/api/network/join`, `/api/node/status`, `/api/node/stop`. Il bersaglio `coordinator_url` del proxy diventa **mutabile** (`POST /api/config`): creando/aggiungendosi, la UI punta alla rete giusta. Il frontend aggiunge un pannello "Gestione rete".

**Tech Stack:** Python · subprocess · FastAPI · l'esistente `synapse/ui/*` e la CLI `synapse`. Sicurezza: la UI è in ascolto su `127.0.0.1`; i comandi sono costruiti come liste (no shell), input passato come argomenti.

**Fuori scope (Fase 3):** config + esecuzione tool MCP.

---

## File Structure
```
synapse/__main__.py             # NUOVO: abilita `python -m synapse`
synapse/ui/manager.py           # NUOVO: NodeManager (spawn/status/stop)
synapse/ui/server.py            # MOD: coordinator_url mutabile + endpoint create/join/status/stop
synapse/ui/static/index.html    # MOD: pannello "Gestione rete" (crea/aggiungi/stato/stop)
tests/test_ui_manager.py        # NUOVO: NodeManager (spawn proc banale)
tests/test_ui_server.py         # MOD: POST /api/config + /api/node/status
docs/examples/frontend.md       # MOD
```

---

## Task 1: `python -m synapse` + `NodeManager`

**Files:** create `synapse/__main__.py`, `synapse/ui/manager.py`, `tests/test_ui_manager.py`.

- [ ] **Step 1: test `tests/test_ui_manager.py`**
```python
import sys, time
from synapse.ui.manager import NodeManager


def test_start_status_stop():
    mgr = NodeManager()
    assert mgr.status() == {}
    # processo banale di lunga durata, niente modello
    mgr.start("worker", [sys.executable, "-c", "import time; time.sleep(30)"], {"stages": "decoder:0-8"})
    st = mgr.status()
    assert st["worker"]["running"] is True
    assert st["worker"]["stages"] == "decoder:0-8"
    assert isinstance(st["worker"]["pid"], int)
    mgr.stop("worker")
    time.sleep(0.3)
    assert mgr.status().get("worker", {}).get("running", False) is False or "worker" not in mgr.status()


def test_start_replaces_previous():
    mgr = NodeManager()
    mgr.start("coordinator", [sys.executable, "-c", "import time; time.sleep(30)"], {"port": 9001})
    pid1 = mgr.status()["coordinator"]["pid"]
    mgr.start("coordinator", [sys.executable, "-c", "import time; time.sleep(30)"], {"port": 9002})
    pid2 = mgr.status()["coordinator"]["pid"]
    assert pid1 != pid2
    mgr.stop("coordinator")
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/synapse/.venv/bin/python -m pytest tests/test_ui_manager.py -v` → ImportError.

- [ ] **Step 3: crea `synapse/__main__.py`**
```python
from synapse.cli import app

app()
```

- [ ] **Step 4: crea `synapse/ui/manager.py`**
```python
import subprocess


class NodeManager:
    """Gestisce i processi locali avviati dalla UI (coordinator e/o worker)."""
    def __init__(self):
        self._procs = {}   # role -> {"popen": Popen, "info": dict}

    def start(self, role: str, cmd: list, info: dict) -> None:
        self.stop(role)
        popen = subprocess.Popen(cmd)
        self._procs[role] = {"popen": popen, "info": dict(info)}

    def status(self) -> dict:
        out = {}
        for role, d in self._procs.items():
            running = d["popen"].poll() is None
            out[role] = {"running": running, "pid": d["popen"].pid, **d["info"]}
        return out

    def stop(self, role: str) -> None:
        d = self._procs.pop(role, None)
        if d is not None and d["popen"].poll() is None:
            d["popen"].terminate()
            try:
                d["popen"].wait(timeout=5)
            except Exception:
                d["popen"].kill()

    def stop_all(self) -> None:
        for role in list(self._procs.keys()):
            self.stop(role)
```

- [ ] **Step 5: run PASS** — `/Users/alberto/Projects/AI/synapse/.venv/bin/python -m pytest tests/test_ui_manager.py -v` → 2 passed. Verifica anche l'entry: `.venv/bin/python -m synapse --help | head -1`.

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/__main__.py synapse/ui/manager.py tests/test_ui_manager.py && git commit -m "feat(ui): NodeManager (spawn/status/stop processi) + entry python -m synapse"
```

---

## Task 2: endpoint create/join/status/stop + coordinator_url mutabile

**Files:** modify `synapse/ui/server.py`; modify `tests/test_ui_server.py`.

- [ ] **Step 1: aggiungi test a `tests/test_ui_server.py`**
```python
def test_config_can_be_updated():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    r = c.post("/api/config", json={"coordinator_url": "http://nuovo:9100"})
    assert r.status_code == 200
    assert c.get("/api/config").json()["coordinator_url"] == "http://nuovo:9100"


def test_node_status_empty_initially():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    assert c.get("/api/node/status").json() == {}
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_ui_server.py::test_config_can_be_updated tests/test_ui_server.py::test_node_status_empty_initially -v`.

- [ ] **Step 3: modifica `synapse/ui/server.py`**

Rendi `coordinator_url` mutabile e aggiungi gli endpoint. Sostituisci l'inizio di `create_ui_app` (fino alla def di `_index_html`) e aggiungi i nuovi endpoint:
```python
import os
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from synapse.ui.manager import NodeManager

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_ui_app(coordinator_url: str) -> FastAPI:
    """Server di controllo locale: serve il frontend, fa da proxy al coordinator,
    e gestisce i processi locali (crea/aggiungi rete)."""
    app = FastAPI()
    state = {"coordinator_url": coordinator_url.rstrip("/")}
    manager = NodeManager()

    def _coord() -> str:
        return state["coordinator_url"]
```
(Sostituisci ogni uso di `coord` negli endpoint esistenti con `_coord()`: in `/api/registry` usa `f"{_coord()}/registry"`, in `/api/chat` usa `f"{_coord()}/v1/chat/completions"`.)

Aggiorna `/api/config` per supportare GET e POST, e aggiungi gli endpoint di gestione (prima di `return app`):
```python
    @app.get("/api/config")
    async def get_config():
        return {"coordinator_url": _coord()}

    @app.post("/api/config")
    async def set_config(request: Request):
        body = await request.json()
        if body.get("coordinator_url"):
            state["coordinator_url"] = str(body["coordinator_url"]).rstrip("/")
        return {"coordinator_url": _coord()}

    @app.get("/api/node/status")
    async def node_status():
        return manager.status()

    @app.post("/api/network/create")
    async def network_create(request: Request):
        body = await request.json()
        model = body.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        port = int(body.get("port", 9000))
        cmd = [sys.executable, "-m", "synapse", "coordinator", "--model", model, "--port", str(port)]
        manager.start("coordinator", cmd, {"role": "coordinator", "port": port, "model": model,
                                           "url": f"http://127.0.0.1:{port}"})
        state["coordinator_url"] = f"http://127.0.0.1:{port}"
        return {"ok": True, "coordinator_url": _coord(), "status": manager.status()}

    @app.post("/api/network/join")
    async def network_join(request: Request):
        body = await request.json()
        coord_url = str(body.get("coordinator_url") or _coord()).rstrip("/")
        ws = coord_url.replace("http://", "ws://").replace("https://", "wss://") + "/node"
        stages = body.get("stages", "")
        model = body.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        if not stages:
            return JSONResponse({"error": "stages obbligatori (es. 'embed,decoder:0-12')"}, status_code=400)
        cmd = [sys.executable, "-m", "synapse", "serve", "--coordinator", ws, "--stages", stages, "--model", model]
        manager.start("worker", cmd, {"role": "worker", "stages": stages, "coordinator": coord_url, "model": model})
        state["coordinator_url"] = coord_url
        return {"ok": True, "coordinator_url": _coord(), "status": manager.status()}

    @app.post("/api/node/stop")
    async def node_stop(request: Request):
        body = await request.json()
        role = body.get("role")
        if role:
            manager.stop(role)
        else:
            manager.stop_all()
        return {"ok": True, "status": manager.status()}
```
Lascia invariati `/`, `/api/registry`, `/api/chat` (a parte l'uso di `_coord()`).

- [ ] **Step 4: run PASS** — `... pytest tests/test_ui_server.py -v` (fast) e `-m slow -v` (proxy) → tutti PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/ui/server.py tests/test_ui_server.py && git commit -m "feat(ui): endpoint create/join/status/stop + coordinator_url mutabile"
```

---

## Task 3: pannello "Gestione rete" nel frontend

**Files:** modify `synapse/ui/static/index.html`.

> Aggiungi all'app esistente (NON riscriverla) un pannello/modale **"Gestione rete"** raggiungibile dalla vista Rete (es. un terzo tab "Gestione" o un pulsante in header). Mantieni lo stile esistente (dark, IBM Plex, teal/ambra).

- [ ] **Step 1: estendi `synapse/ui/static/index.html`** con:
  - **Bersaglio coordinator**: campo editabile con l'URL corrente (da `GET /api/config`); "Applica" → `POST /api/config {coordinator_url}` e poi riprende il polling sul nuovo bersaglio.
  - **Crea una rete**: form (model id, default `Qwen/Qwen2.5-0.5B-Instruct`; porta, default 9000) → bottone "Crea" → `POST /api/network/create`; al successo aggiorna il bersaglio e mostra lo stato.
  - **Aggiungiti a una rete**: form (coordinator URL http, default = bersaglio corrente; stage es. `embed,decoder:0-12`) → bottone "Aggiungiti" → `POST /api/network/join`; al successo aggiorna il bersaglio.
  - **Nodo locale**: mostra `GET /api/node/status` (coordinator e/o worker: running, pid, dettagli) con bottoni **Stop** (`POST /api/node/stop {role}`). Polling dello status ogni 2s.
  - Suggerimento utile: dopo "Crea", proponi i comandi/azioni per aggiungere embed/decoder/head (puoi riusare il blocco "MANCANO" della vista Rete che già calcola i pezzi mancanti).
  - Gestisci errori (400/500) mostrando il messaggio.

- [ ] **Step 2: check strutturale** — `cd /Users/alberto/Projects/AI/synapse && .venv/bin/python -c "h=open('synapse/ui/static/index.html').read(); assert '/api/network/create' in h and '/api/network/join' in h and '/api/node/status' in h and '/api/config' in h; print('gestione rete ok', len(h))"`

- [ ] **Step 3: il server serve ancora l'index** — `... pytest tests/test_ui_server.py::test_serves_index_html -v` → PASS.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add synapse/ui/static/index.html && git commit -m "feat(ui): pannello Gestione rete (crea/aggiungi/stato/stop)"
```

---

## Task 4: docs + suite

**Files:** modify `docs/examples/frontend.md`.

- [ ] **Step 1: aggiorna `docs/examples/frontend.md`** — aggiungi una sezione "Crea o aggiungi una rete dalla UI": dal pannello *Gestione rete* puoi creare una rete (avvia un coordinator locale), aggiungerti a una rete esistente coi tuoi stage, cambiare il coordinator bersaglio, e fermare il nodo locale. Nota di sicurezza: la UI è locale (`127.0.0.1`) e avvia processi sulla tua macchina.

- [ ] **Step 2: suite completa** — `... pytest -q -p no:warnings` → verde.

- [ ] **Step 3: commit**
```bash
cd /Users/alberto/Projects/AI/synapse && git add docs/examples/frontend.md && git commit -m "docs: gestione rete (crea/aggiungi) dal frontend"
```

---

## Self-Review

**Coverage (#1 crea/aggiungi rete):** NodeManager spawn/stop (Task 1) ✓; entry `python -m synapse` (Task 1) ✓; endpoint create/join/status/stop + bersaglio mutabile (Task 2) ✓; pannello UI Gestione rete (Task 3) ✓; docs (Task 4) ✓. MCP = Fase 3.

**Placeholder scan:** server/manager con codice completo; il pannello frontend è un'estensione specificata in dettaglio (artefatto visivo) con check strutturale.

**Type consistency:** `NodeManager.start(role, cmd:list, info:dict)/status()->dict/stop(role)/stop_all()`; endpoint `/api/network/create`,`/api/network/join`,`/api/node/status`,`/api/node/stop`,`/api/config` (GET+POST); il frontend consuma esattamente questi. I comandi spawnati usano `python -m synapse <subcommand>` (entry in `__main__.py`).
```
