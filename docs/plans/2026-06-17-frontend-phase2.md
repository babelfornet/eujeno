# Frontend Phase 2 — create / join a network from the frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** From the `axyn ui` dashboard, a node can **create a network** (start a local coordinator) or **join a network** (start a `serve --coordinator` with its own stages), and manage/stop the local process — all without the CLI.

**Architecture:** The `axyn ui` server manages subprocesses via a `NodeManager` (Popen of `python -m axyn coordinator|serve …`). New endpoints `/api/network/create`, `/api/network/join`, `/api/node/status`, `/api/node/stop`. The proxy's `coordinator_url` target becomes **mutable** (`POST /api/config`): on create/join, the UI points to the right network. The frontend adds a "Network management" panel.

**Tech Stack:** Python · subprocess · FastAPI · the existing `axyn/ui/*` and the `axyn` CLI. Security: the UI listens on `127.0.0.1`; commands are built as lists (no shell), input passed as arguments.

**Out of scope (Phase 3):** MCP tool config + execution.

---

## File Structure
```
axyn/__main__.py             # NEW: enables `python -m axyn`
axyn/ui/manager.py           # NEW: NodeManager (spawn/status/stop)
axyn/ui/server.py            # MOD: mutable coordinator_url + create/join/status/stop endpoints
axyn/ui/static/index.html    # MOD: "Network management" panel (create/join/status/stop)
tests/test_ui_manager.py        # NEW: NodeManager (spawn trivial proc)
tests/test_ui_server.py         # MOD: POST /api/config + /api/node/status
docs/examples/frontend.md       # MOD
```

---

## Task 1: `python -m axyn` + `NodeManager`

**Files:** create `axyn/__main__.py`, `axyn/ui/manager.py`, `tests/test_ui_manager.py`.

- [ ] **Step 1: test `tests/test_ui_manager.py`**
```python
import sys, time
from axyn.ui.manager import NodeManager


def test_start_status_stop():
    mgr = NodeManager()
    assert mgr.status() == {}
    # trivial long-running process, no model
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

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_ui_manager.py -v` → ImportError.

- [ ] **Step 3: create `axyn/__main__.py`**
```python
from axyn.cli import app

app()
```

- [ ] **Step 4: create `axyn/ui/manager.py`**
```python
import subprocess


class NodeManager:
    """Manages the local processes started by the UI (coordinator and/or worker)."""
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

- [ ] **Step 5: run PASS** — `/Users/alberto/Projects/AI/axyn/.venv/bin/python -m pytest tests/test_ui_manager.py -v` → 2 passed. Also verify the entry point: `.venv/bin/python -m axyn --help | head -1`.

- [ ] **Step 6: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/__main__.py axyn/ui/manager.py tests/test_ui_manager.py && git commit -m "feat(ui): NodeManager (spawn/status/stop processes) + python -m axyn entry point"
```

---

## Task 2: create/join/status/stop endpoints + mutable coordinator_url

**Files:** modify `axyn/ui/server.py`; modify `tests/test_ui_server.py`.

- [ ] **Step 1: add tests to `tests/test_ui_server.py`**
```python
def test_config_can_be_updated():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    r = c.post("/api/config", json={"coordinator_url": "http://new:9100"})
    assert r.status_code == 200
    assert c.get("/api/config").json()["coordinator_url"] == "http://new:9100"


def test_node_status_empty_initially():
    app = create_ui_app("http://example:9000")
    c = TestClient(app)
    assert c.get("/api/node/status").json() == {}
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_ui_server.py::test_config_can_be_updated tests/test_ui_server.py::test_node_status_empty_initially -v`.

- [ ] **Step 3: modify `axyn/ui/server.py`**

Make `coordinator_url` mutable and add the endpoints. Replace the start of `create_ui_app` (up to the def of `_index_html`) and add the new endpoints:
```python
import os
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from axyn.ui.manager import NodeManager

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_ui_app(coordinator_url: str) -> FastAPI:
    """Local control server: serves the frontend, proxies to the coordinator,
    and manages the local processes (create/join network)."""
    app = FastAPI()
    state = {"coordinator_url": coordinator_url.rstrip("/")}
    manager = NodeManager()

    def _coord() -> str:
        return state["coordinator_url"]
```
(Replace every use of `coord` in the existing endpoints with `_coord()`: in `/api/registry` use `f"{_coord()}/registry"`, in `/api/chat` use `f"{_coord()}/v1/chat/completions"`.)

Update `/api/config` to support GET and POST, and add the management endpoints (before `return app`):
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
        cmd = [sys.executable, "-m", "axyn", "coordinator", "--model", model, "--port", str(port)]
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
            return JSONResponse({"error": "stages required (e.g. 'embed,decoder:0-12')"}, status_code=400)
        cmd = [sys.executable, "-m", "axyn", "serve", "--coordinator", ws, "--stages", stages, "--model", model]
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
Leave `/`, `/api/registry`, `/api/chat` unchanged (apart from the use of `_coord()`).

- [ ] **Step 4: run PASS** — `... pytest tests/test_ui_server.py -v` (fast) and `-m slow -v` (proxy) → all PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/server.py tests/test_ui_server.py && git commit -m "feat(ui): create/join/status/stop endpoints + mutable coordinator_url"
```

---

## Task 3: "Network management" panel in the frontend

**Files:** modify `axyn/ui/static/index.html`.

> Add to the existing app (do NOT rewrite it) a **"Network management"** panel/modal reachable from the Network view (e.g. a third "Management" tab or a header button). Keep the existing style (dark, IBM Plex, teal/amber).

- [ ] **Step 1: extend `axyn/ui/static/index.html`** with:
  - **Coordinator target**: editable field with the current URL (from `GET /api/config`); "Apply" → `POST /api/config {coordinator_url}` and then resume polling on the new target.
  - **Create a network**: form (model id, default `Qwen/Qwen2.5-0.5B-Instruct`; port, default 9000) → "Create" button → `POST /api/network/create`; on success, update the target and show the status.
  - **Join a network**: form (coordinator http URL, default = current target; stages e.g. `embed,decoder:0-12`) → "Join" button → `POST /api/network/join`; on success, update the target.
  - **Local node**: show `GET /api/node/status` (coordinator and/or worker: running, pid, details) with **Stop** buttons (`POST /api/node/stop {role}`). Poll the status every 2s.
  - Useful tip: after "Create", suggest the commands/actions to add embed/decoder/head (you can reuse the "MISSING" block of the Network view that already computes the missing pieces).
  - Handle errors (400/500) by showing the message.

- [ ] **Step 2: structural check** — `cd /Users/alberto/Projects/AI/axyn && .venv/bin/python -c "h=open('axyn/ui/static/index.html').read(); assert '/api/network/create' in h and '/api/network/join' in h and '/api/node/status' in h and '/api/config' in h; print('network management ok', len(h))"`

- [ ] **Step 3: the server still serves the index** — `... pytest tests/test_ui_server.py::test_serves_index_html -v` → PASS.

- [ ] **Step 4: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add axyn/ui/static/index.html && git commit -m "feat(ui): Network management panel (create/join/status/stop)"
```

---

## Task 4: docs + suite

**Files:** modify `docs/examples/frontend.md`.

- [ ] **Step 1: update `docs/examples/frontend.md`** — add a "Create or join a network from the UI" section: from the *Network management* panel you can create a network (start a local coordinator), join an existing network with your stages, change the target coordinator, and stop the local node. Security note: the UI is local (`127.0.0.1`) and starts processes on your machine.

- [ ] **Step 2: full suite** — `... pytest -q -p no:warnings` → green.

- [ ] **Step 3: commit**
```bash
cd /Users/alberto/Projects/AI/axyn && git add docs/examples/frontend.md && git commit -m "docs: network management (create/join) from the frontend"
```

---

## Self-Review

**Coverage (#1 create/join network):** NodeManager spawn/stop (Task 1) ✓; `python -m axyn` entry point (Task 1) ✓; create/join/status/stop endpoints + mutable target (Task 2) ✓; UI Network management panel (Task 3) ✓; docs (Task 4) ✓. MCP = Phase 3.

**Placeholder scan:** server/manager with complete code; the frontend panel is an extension specified in detail (visual artifact) with a structural check.

**Type consistency:** `NodeManager.start(role, cmd:list, info:dict)/status()->dict/stop(role)/stop_all()`; endpoints `/api/network/create`,`/api/network/join`,`/api/node/status`,`/api/node/stop`,`/api/config` (GET+POST); the frontend consumes exactly these. The spawned commands use `python -m axyn <subcommand>` (entry point in `__main__.py`).
```
