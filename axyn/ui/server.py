# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from axyn.ui.manager import NodeManager
from axyn.ui.mcp import McpRegistry
from axyn.ui.agent import run_tool_loop

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_ui_app(coordinator_url: str) -> FastAPI:
    """Local control server: serves the frontend, proxies to the coordinator,
    and manages local processes (create/join network)."""
    app = FastAPI()
    state = {"coordinator_url": coordinator_url.rstrip("/")}
    manager = NodeManager()
    mcp = McpRegistry()

    def _coord() -> str:
        return state["coordinator_url"]

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
            return JSONResponse({"error": f"coordinator unreachable: {e}"}, status_code=502)

    @app.get("/api/mcp/list")
    async def mcp_list():
        servers = mcp.list_servers()
        tools = []
        if servers:
            try:
                tools = [{"name": t["function"]["name"], "description": t["function"]["description"]}
                         for t in await asyncio.to_thread(mcp.list_tools)]
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
        try:
            tools = await asyncio.to_thread(mcp.list_tools)
        except Exception as e:
            return JSONResponse({"error": f"MCP error: {e}"}, status_code=502)
        clean_tools = [{"type": t["type"], "function": t["function"]} for t in tools]
        coord = _coord()
        max_tokens = int(body.get("max_tokens", 256))
        temperature = body.get("temperature", 0.7)

        def call_model(messages, tls):
            payload = {"messages": messages, "tools": tls, "max_tokens": max_tokens, "temperature": temperature}
            with httpx.Client(timeout=300.0) as client:
                rr = client.post(f"{coord}/v1/chat/completions", json=payload)
            return rr.json()["choices"][0]["message"]

        out = await asyncio.to_thread(
            run_tool_loop, body.get("messages", []), clean_tools, call_model,
            lambda name, args: mcp.call_tool(name, args), 6)
        return {"choices": [{"message": {"role": "assistant", "content": out["content"]},
                             "finish_reason": "stop"}],
                "tool_runs": out["tool_runs"]}

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
            return JSONResponse({"error": "stages are required (e.g. 'embed,decoder:0-12')"}, status_code=400)
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

    @app.post("/api/network/create_p2p")
    async def network_create_p2p(request: Request):
        body = await request.json()
        advertise = str(body.get("advertise") or "http://127.0.0.1:8001").rstrip("/")
        stages = body.get("stages", "")
        model = body.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        if not stages:
            return JSONResponse({"error": "stages are required"}, status_code=400)
        port = advertise.rsplit(":", 1)[-1]
        cmd = [sys.executable, "-m", "axyn", "serve", "--advertise", advertise,
               "--stages", stages, "--model", model, "--host", "0.0.0.0", "--port", str(port)]
        manager.start("worker", cmd, {"role": "worker", "mode": "p2p", "advertise": advertise, "stages": stages})
        state["coordinator_url"] = advertise
        return {"ok": True, "coordinator_url": _coord(), "status": manager.status()}

    @app.post("/api/network/join_p2p")
    async def network_join_p2p(request: Request):
        body = await request.json()
        advertise = str(body.get("advertise") or "http://127.0.0.1:8001").rstrip("/")
        peers = str(body.get("peers") or "").strip()
        stages = body.get("stages", "")
        model = body.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        if not stages or not peers:
            return JSONResponse({"error": "stages and peers are required"}, status_code=400)
        port = advertise.rsplit(":", 1)[-1]
        cmd = [sys.executable, "-m", "axyn", "serve", "--advertise", advertise, "--peers", peers,
               "--stages", stages, "--model", model, "--host", "0.0.0.0", "--port", str(port)]
        manager.start("worker", cmd, {"role": "worker", "mode": "p2p", "advertise": advertise,
                                      "peers": peers, "stages": stages})
        state["coordinator_url"] = advertise
        return {"ok": True, "coordinator_url": _coord(), "status": manager.status()}

    return app
