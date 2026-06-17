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

    def _index_html() -> str:
        path = os.path.join(_STATIC, "index.html")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        return "<!doctype html><title>Synapse</title><h1>Synapse UI</h1>"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _index_html()

    @app.get("/api/config")
    async def get_config():
        return {"coordinator_url": _coord()}

    @app.post("/api/config")
    async def set_config(request: Request):
        body = await request.json()
        if body.get("coordinator_url"):
            state["coordinator_url"] = str(body["coordinator_url"]).rstrip("/")
        return {"coordinator_url": _coord()}

    @app.get("/api/registry")
    async def registry():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{_coord()}/registry")
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": f"coordinator non raggiungibile: {e}"}, status_code=502)

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{_coord()}/v1/chat/completions", content=body,
                                      headers={"content-type": "application/json"})
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": f"coordinator non raggiungibile: {e}"}, status_code=502)

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

    return app
