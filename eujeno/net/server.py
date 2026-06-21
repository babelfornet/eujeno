# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from eujeno.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from eujeno.net.wire import encode_tensors, decode_tensors
from eujeno.net.discovery import Registry, build_chain
from eujeno.net.generation import generate_tokens
from eujeno.net.jobstore import JobStore
from eujeno.net.tools import extract_tool_calls
from eujeno.net.nodeconfig import NodeConfig
from eujeno.net.metrics import NodeMetrics
from eujeno.net.capacity import probe_capacity

_OCTET = "application/octet-stream"


def create_app(model, tokenizer, stages, node_url=None, peers=None,
               num_layers=None, gossip_interval=2.0, ttl=30.0, db_path=None,
               config_path=None, device="cpu"):
    """Create the FastAPI app of a BlockServer. With node_url/peers it enables
    decentralized gossip discovery (Mode A); without, Part 1 behavior."""
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}
    own_stages = {"embed": stages.embed, "head": stages.head, "decoders": list(prepared.keys())}
    own_stages["capacity"] = probe_capacity()
    registry = Registry()

    # Monotonic per-origin heartbeat, stamped on every self-advert. The relay path
    # (Registry.merge) extends an entry's TTL only on a newer hb, so a node's own
    # adverts must always carry a strictly increasing one. Seeded from wall-clock so
    # it keeps rising across restarts (a restarted node out-incarnates its old self).
    _hb = {"v": time.time()}
    _hb_lock = threading.Lock()

    def _next_hb():
        with _hb_lock:
            t = time.time()
            if t <= _hb["v"]:
                t = _hb["v"] + 1e-3
            _hb["v"] = t
            return t

    if node_url:
        registry.upsert(node_url, {**own_stages, "hb": _next_hb()}, now=time.time(), ttl=ttl)

    config = NodeConfig(config_path)
    metrics = NodeMetrics()
    peer_status = {}   # url -> "online"|"syncing"|"offline"

    stop_ids = set()
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    if tokenizer is not None:
        for _t in ("<|im_end|>", "<|endoftext|>"):
            _i = tokenizer.convert_tokens_to_ids(_t)
            if isinstance(_i, int) and _i >= 0 and _i != tokenizer.unk_token_id:
                stop_ids.add(int(_i))
    _model_id = getattr(model.config, "_name_or_path", "eujeno")
    _entry_job = {"n": 0}
    _proc = uuid.uuid4().hex[:8]
    store = JobStore(db_path if db_path is not None else ":memory:")

    def _store_safe(fn, *args):
        try:
            fn(*args)
        except Exception as e:
            logging.getLogger("eujeno.node").warning("jobstore write failed: %s", e)

    _store_safe(store.recover)

    def _get_json(url, timeout=5.0):
        """Blocking GET → JSON, used by the gossip thread (sync httpx)."""
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _ping_ms(url, timeout=3.0):
        """Blocking GET; returns round-trip latency in ms."""
        t0 = time.monotonic()
        httpx.get(url, timeout=timeout).raise_for_status()
        return (time.monotonic() - t0) * 1000

    # Discovery runs in plain daemon OS threads with *sync* httpx, NOT asyncio tasks.
    # On some hosts the asyncio transport silently stalled the gossip pull inside the
    # uvicorn event loop (loop alive, 0% CPU, awaited pulls never firing) while
    # inference still worked; a dedicated thread sidesteps the event loop entirely.
    # Registry is thread-safe (its own lock), so concurrent reads from the async
    # request handlers are safe.
    peer_fail: dict[str, int] = {}

    def _gossip_thread():
        while True:
            now = time.time()
            try:
                adv = None
                if node_url:
                    adv = {**own_stages, "name": config.get()["name"], "region": config.get()["region"]}
                    if config.get().get("telemetry"):
                        adv["tput"] = metrics.throughput_tok_s()
                    adv["hb"] = _next_hb()
                    registry.upsert(node_url, adv, now=now, ttl=ttl)
                for peer in (peers or []):
                    # pull: learn what the peer already knows
                    try:
                        registry.merge(_get_json(f"{peer}/registry").get("nodes", {}), now=now, ttl=ttl)
                    except Exception:
                        pass
                    # push: announce ourselves so the peer learns us even when we
                    # are not in its static --peers list. Without this a node that
                    # joined via a seed stays invisible to the rest of the swarm
                    # (pull-only gossip makes a new node a sink).
                    if adv is not None:
                        try:
                            httpx.post(f"{peer}/registry", json={"url": node_url, "node": adv}, timeout=5.0)
                        except Exception:
                            pass
                registry.prune(now)
                # Evict per-peer bookkeeping for nodes that just aged out, so the
                # status/fail/metrics dicts track the live swarm instead of growing.
                live = set(registry.stages_by_url(now).keys())
                for d in (peer_status, peer_fail):
                    for u in [u for u in d if u not in live]:
                        del d[u]
                metrics.retain(live)
            except Exception:
                pass
            time.sleep(gossip_interval)

    def _probe_thread():
        """Ping each peer every 5 s; record latency + update peer_status."""
        while True:
            now = time.time()
            try:
                for url in list(registry.stages_by_url(now).keys()):
                    if url == node_url:
                        continue
                    try:
                        metrics.observe_latency(url, _ping_ms(f"{url}/health"))
                        peer_status[url] = "online"
                        peer_fail[url] = 0
                    except Exception:
                        peer_fail[url] = peer_fail.get(url, 0) + 1
                        peer_status[url] = "offline" if peer_fail[url] >= 3 else "syncing"
            except Exception:
                pass
            time.sleep(5.0)

    @asynccontextmanager
    async def lifespan(_app):
        if node_url:
            threading.Thread(target=_gossip_thread, daemon=True).start()
            threading.Thread(target=_probe_thread, daemon=True).start()
        yield

    app = FastAPI(lifespan=lifespan)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ram_gb():
        try:
            c = probe_capacity()
            tot = c.get("ram_total_gb") or 0.0
            free = c.get("ram_free_gb") or 0.0
            return round(max(0.0, tot - free), 1), round(tot, 1)
        except Exception:
            return 0.0, 0.0

    def _layers_label():
        parts = []
        if stages.embed:
            parts.append("embed")
        for (lo, hi) in stages.decoders:
            parts.append(f"L{lo}–L{hi-1}")
        if stages.head:
            parts.append("head")
        return ",".join(parts) or "—"

    # ── /api/* routes ─────────────────────────────────────────────────────────

    @app.get("/api/node")
    async def api_node():
        try:
            used, tot = _ram_gb()
            clean_stages = {"embed": stages.embed, "head": stages.head, "decoders": list(prepared.keys())}
            return {
                "peerId": config.peer_id,
                "name": config.get()["name"],
                "model": getattr(model.config, "_name_or_path", "?"),
                "numLayers": num_layers,
                "stages": clean_stages,
                "layers": _layers_label(),
                "status": "serving",
                "ramUsedGb": used,
                "ramTotalGb": tot,
                "region": config.get()["region"],
                "uptimeSec": round(metrics.uptime_sec()),
                "port": config.get()["port"],
                "requestsServed": metrics.requests_served,
                "throughputTokS": metrics.throughput_tok_s(),
            }
        except Exception as e:
            logging.getLogger("eujeno.node").warning("/api/node error: %s", e)
            return {"peerId": config.peer_id, "error": str(e)}

    @app.get("/api/metrics")
    async def api_metrics():
        try:
            nodes = registry.stages_by_url(time.time())
            try:
                active = sum(1 for j in store.recent_jobs(100) if j.get("status") == "RUNNING")
            except Exception:
                active = 0
            return {
                "connectedPeers": max(0, len(nodes) - 1),
                "throughputTokS": metrics.throughput_tok_s(),
                "avgLatencyMs": metrics.avg_latency_ms(),
                "activeQueries": active,
                "requestsServed": metrics.requests_served,
            }
        except Exception as e:
            logging.getLogger("eujeno.node").warning("/api/metrics error: %s", e)
            return {"connectedPeers": 0, "throughputTokS": 0.0, "avgLatencyMs": 0,
                    "activeQueries": 0, "requestsServed": metrics.requests_served}

    @app.get("/api/peers")
    async def api_peers():
        try:
            nodes = registry.stages_by_url(time.time())
            out = []
            for url, st in nodes.items():
                if url == node_url:
                    continue
                decs = st.get("decoders") or []
                parts = []
                if st.get("embed"):
                    parts.append("embed")
                parts.extend(f"L{d}" for d in decs)
                if st.get("head"):
                    parts.append("head")
                lab = ",".join(parts) or "—"
                lat = metrics.peer_latency.get(url)
                out.append({
                    "peerId": st.get("name") or url,
                    "url": url,
                    "layers": lab,
                    "region": st.get("region") or "—",
                    "latencyMs": round(lat) if lat is not None else None,
                    "throughputTokS": st.get("tput") or 0.0,
                    "status": peer_status.get(url, "syncing"),
                })
            out.sort(key=lambda p: (
                p["latencyMs"] if p["latencyMs"] is not None else 1e9,
                -(p["throughputTokS"] or 0),
            ))
            return {"peers": out}
        except Exception as e:
            logging.getLogger("eujeno.node").warning("/api/peers error: %s", e)
            return {"peers": []}

    @app.get("/api/settings")
    async def api_settings_get():
        try:
            return config.get()
        except Exception as e:
            return {"error": str(e)}

    @app.put("/api/settings")
    async def api_settings_put(request: Request):
        try:
            body = await request.json()
            return config.update(body)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/api/node/restart")
    async def api_restart():
        try:
            own_stages["capacity"] = probe_capacity()
            adv = {**own_stages, "name": config.get()["name"], "region": config.get()["region"]}
            if config.get().get("telemetry"):
                adv["tput"] = metrics.throughput_tok_s()
            adv["hb"] = _next_hb()
            registry.upsert(node_url, adv, now=time.time(), ttl=ttl)
        except Exception as e:
            logging.getLogger("eujeno.node").warning("/api/node/restart error: %s", e)
        return {"ok": True, "message": "re-probed capacity and re-broadcast registration"}

    # ── existing routes ────────────────────────────────────────────────────────

    @app.get("/registry")
    async def get_registry():
        return {"num_layers": num_layers, "model": getattr(model.config, "_name_or_path", "?"),
                "nodes": registry.stages_by_url(time.time())}

    @app.post("/registry")
    async def post_registry(req: Request):
        """Receive a peer's self-announcement (gossip push) and merge it. This is
        what lets a node that joined via us — but isn't in our static --peers
        list — become known to the whole swarm."""
        try:
            body = await req.json()
            now = time.time()
            if isinstance(body, dict) and body.get("url"):
                registry.upsert(body["url"], body.get("node") or {}, now=now, ttl=ttl)
            elif isinstance(body, dict) and isinstance(body.get("nodes"), dict):
                registry.merge(body["nodes"], now=now, ttl=ttl)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "model": getattr(model.config, "_name_or_path", "?"),
            "stages": {"embed": embed_block is not None, "head": head_block is not None,
                       "decoders": list(prepared.keys())},
        }

    @app.post("/embed")
    async def embed(job_id: str, request: Request):
        if embed_block is None:
            return JSONResponse({"error": "this node does not serve the embed stage"}, status_code=400)
        metrics.inc_request()
        t = decode_tensors(await request.body())
        h = embed_block.run_block(t["input_ids"].to(device))
        return Response(encode_tensors({"hidden_states": h.cpu()}), media_type=_OCTET)

    @app.post("/decode/{block_key}")
    async def decode(block_key: str, job_id: str, request: Request):
        if block_key not in prepared:
            return JSONResponse({"error": f"block {block_key} not served"}, status_code=400)
        metrics.inc_request()
        t = decode_tensors(await request.body())
        job = jobs.setdefault(job_id, {})
        block = job.get(block_key)
        if block is None:
            layers, rotary = prepared[block_key]
            block = DecoderBlock(layers, rotary)   # own cache per (job, block)
            job[block_key] = block
        h = block.run_block(t["hidden_states"].to(device), t["cache_position"].to(device))
        return Response(encode_tensors({"hidden_states": h.cpu()}), media_type=_OCTET)

    @app.post("/head")
    async def head(job_id: str, request: Request, topk: int = 1):
        if head_block is None:
            return JSONResponse({"error": "this node does not serve the head stage"}, status_code=400)
        metrics.inc_request()
        t = decode_tensors(await request.body())
        logits = head_block.run_block(t["hidden_states"].to(device))[:, -1, :]
        k = min(int(topk), logits.shape[-1])
        vals, idx = torch.topk(logits[0], k=k)
        ids = idx.tolist()
        return JSONResponse({"token_id": ids[0], "topk_ids": ids, "topk_logits": vals.tolist()})

    @app.delete("/job/{job_id}")
    async def end_job(job_id: str):
        jobs.pop(job_id, None)
        return {"ok": True}

    @app.get("/v1/models")
    async def v1_models():
        return {"object": "list", "data": [{"id": "eujeno", "object": "model", "owned_by": "eujeno"},
                                            {"id": _model_id, "object": "model", "owned_by": "eujeno"}]}

    @app.post("/v1/chat/completions")
    async def v1_chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        max_new = int(body.get("max_tokens", 256))
        tools = body.get("tools")
        sampling = {k: body.get(k) for k in ("temperature", "top_p", "repetition_penalty", "seed")}
        now = time.time()
        nodes = registry.stages_by_url(now)
        speed = metrics.speed_map(list(nodes.keys()))
        chain = build_chain(nodes, num_layers, speed=speed)
        if chain is None:
            return JSONResponse({"error": {"message": "network not operational", "type": "not_operational"}}, status_code=503)
        embed_url, decoders, head_url = chain
        try:
            prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join((m.get("content") or "") for m in messages)
        _entry_job["n"] += 1
        job_id = f"entry-{_proc}-{_entry_job['n']}"
        metrics.inc_request()
        prompt_len0 = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
        _store_safe(store.create_job, job_id, _model_id, prompt, sampling, prompt_len0)
        receipts = {}

        def _rcacc(url, sent, recv, dt):
            r = receipts.setdefault(url, {"hops": 0, "bytes": 0, "t_compute": 0.0})
            r["hops"] += 1
            r["bytes"] += sent + recv
            r["t_compute"] += dt

        t0_gen = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async def run_embed(cur):
                    payload = encode_tensors({"input_ids": cur}); t0 = time.monotonic()
                    r = await client.post(f"{embed_url}/embed", params={"job_id": job_id}, content=payload)
                    _rcacc(embed_url, len(payload), len(r.content), time.monotonic() - t0)
                    return decode_tensors(r.content)["hidden_states"]

                async def run_decoders(h, cache_position):
                    for bk, url in decoders:
                        payload = encode_tensors({"hidden_states": h, "cache_position": cache_position}); t0 = time.monotonic()
                        r = await client.post(f"{url}/decode/{bk}", params={"job_id": job_id}, content=payload)
                        _rcacc(url, len(payload), len(r.content), time.monotonic() - t0)
                        h = decode_tensors(r.content)["hidden_states"]
                    return h

                async def run_head(h, topk):
                    payload = encode_tensors({"hidden_states": h}); t0 = time.monotonic()
                    r = await client.post(f"{head_url}/head", params={"job_id": job_id, "topk": topk}, content=payload)
                    _rcacc(head_url, len(payload), len(r.content), time.monotonic() - t0)
                    return r.json()

                tokens, prompt_len, finish_reason = await generate_tokens(
                    tokenizer, prompt, max_new, sampling, stop_ids, run_embed, run_decoders, run_head)
                for url in {embed_url, head_url, *(u for _, u in decoders)}:
                    try:
                        await client.delete(f"{url}/job/{job_id}")
                    except Exception:
                        pass
        except Exception as e:
            _store_safe(store.fail, job_id, str(e))
            _store_safe(store.add_receipts, job_id, receipts)
            return JSONResponse({"error": {"message": str(e), "type": "generation_failed"}}, status_code=502)

        elapsed = time.monotonic() - t0_gen
        metrics.record_job(len(tokens), elapsed)
        for url, rc in receipts.items():
            metrics.observe_hop_time(url, rc["t_compute"])

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        _store_safe(store.finish, job_id, text, finish_reason)
        _store_safe(store.add_receipts, job_id, receipts)
        content, tool_calls = extract_tool_calls(text)
        message = {"role": "assistant", "content": (content if content else None)}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        hop_urls = {embed_url, head_url, *(u for _, u in decoders)}
        eujeno_field = {
            "hops": len(hop_urls),
            "layers": num_layers,
            "tokS": metrics.throughput_tok_s(),
        }

        return {"id": "chatcmpl-" + job_id, "object": "chat.completion", "created": int(time.time()),
                "model": _model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(tokens),
                          "total_tokens": prompt_len + len(tokens)},
                "eujeno": eujeno_field}

    @app.get("/jobs")
    async def list_jobs(limit: int = 50):
        return {"jobs": store.recent_jobs(limit)}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        j = store.get_job(job_id)
        if j is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return j

    @app.get("/jobs/{job_id}/receipts")
    async def get_receipts(job_id: str):
        return {"receipts": store.get_receipts(job_id)}

    # ── SPA / placeholder — MUST be last so /api/*, /v1/*, /registry,
    #    /embed, /decode, /head, /jobs all take precedence ────────────────────
    import os as _os
    _static_dir = _os.path.join(_os.path.dirname(__file__), "..", "ui", "static")
    _index = _os.path.join(_static_dir, "index.html")
    if _os.path.exists(_index):
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=_static_dir, html=True), name="ui")
    else:
        @app.get("/")
        async def _placeholder():
            from fastapi.responses import HTMLResponse
            return HTMLResponse("<!doctype html><meta charset=utf-8>"
                                "<title>Eujeno node</title>"
                                "<body style='font-family:system-ui;padding:40px'>"
                                "<h1>Eujeno node</h1><p>Dashboard bundle not built. "
                                "Run <code>npm --prefix app run build</code>. API is live at "
                                "<code>/api/node</code>.</p></body>")

    return app
