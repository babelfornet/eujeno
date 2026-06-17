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
        return "<!doctype html><title>Synapse</title><h1>Synapse UI</h1>"

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
