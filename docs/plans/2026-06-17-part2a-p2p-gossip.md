# Part 2a — Pure P2P: discovery via gossip (decentralized) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** **Automatic and decentralized** discovery (no central server): `eujeno serve` nodes discover each other via **gossip** between peers and self-announce; `eujeno infer --peer <any-node>` builds the topology on its own and runs over the **direct transport** from Part 1.

**Architecture:** [ADR-0002](../decisions/ADR-0002-nat-connectivity.md) Mode A. Each BlockServer keeps a **Registry** (url→stage, with TTL) and a **gossip pull** loop: refresh of its own entry + fetch of the `/registry` from seed peers + merge + prune. Coverage and topology are computed from the registry with `build_chain`. Activation transport = direct HTTP (`distributed_generate` from Part 1). Works wherever the nodes are mutually reachable (LAN/VPN/public IPs).

**Tech Stack:** Python · FastAPI (lifespan background task) · httpx (async for gossip, sync for infer) · the existing `eujeno/net/{server,orchestrator,topology}.py`.

**Out of scope:** NAT traversal without VPN (→ Mode B coordinator, or future libp2p); failover/durability (Part 3).

---

## File Structure

```
eujeno/net/discovery.py        # NEW: Registry (gossip state) + build_chain (coverage)
eujeno/net/server.py           # MODIFY: create_app + Registry, GET /registry, gossip loop (lifespan)
eujeno/cli.py                  # MODIFY: serve --peers/--advertise ; infer --peer
tests/
  test_discovery.py             # Registry + build_chain (fast)
  test_gossip_e2e.py            # 2 real servers: the registry converges (slow)
  test_infer_peer.py            # infer --peer == reference (slow)
docs/examples/p2p.md            # NEW: pure P2P quickstart
```

---

## Task 1: `Registry` + `build_chain` (pure logic)

**Files:** create `eujeno/net/discovery.py`, `tests/test_discovery.py`.

- [ ] **Step 1: test `tests/test_discovery.py`**
```python
from eujeno.net.discovery import Registry, build_chain


def test_build_chain_full_coverage():
    reg = {
        "http://a": {"embed": True, "head": False, "decoders": ["0-12"]},
        "http://b": {"embed": False, "head": True, "decoders": ["12-24"]},
    }
    chain = build_chain(reg, 24)
    assert chain == ("http://a", [("0-12", "http://a"), ("12-24", "http://b")], "http://b")


def test_build_chain_incomplete_returns_none():
    reg = {"http://a": {"embed": True, "head": True, "decoders": ["0-12"]}}
    assert build_chain(reg, 24) is None


def test_registry_merge_and_prune_with_ttl():
    r = Registry()
    r.upsert("http://a", {"embed": True, "head": False, "decoders": ["0-24"]}, now=100.0, ttl=60.0)
    # merge of a learned peer
    r.merge({"http://b": {"head": True, "embed": False, "decoders": []}}, now=100.0, ttl=60.0)
    assert set(r.stages_by_url(now=120.0).keys()) == {"http://a", "http://b"}
    # after expiry (beyond now+ttl) they disappear if not refreshed
    r.prune(now=200.0)
    assert r.stages_by_url(now=200.0) == {}


def test_registry_refresh_extends_expiry():
    r = Registry()
    r.upsert("http://a", {"embed": True, "head": True, "decoders": ["0-24"]}, now=100.0, ttl=60.0)
    r.upsert("http://a", {"embed": True, "head": True, "decoders": ["0-24"]}, now=150.0, ttl=60.0)
    assert "http://a" in r.stages_by_url(now=200.0)   # refreshed at 150 -> expires at 210
```

- [ ] **Step 2: run FAIL** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest tests/test_discovery.py -v` → ImportError.

- [ ] **Step 3: implement `eujeno/net/discovery.py`**
```python
class Registry:
    """Decentralized discovery state: url -> {stages, expiry}. Relative TTL:
    learned entries expire at now+ttl if not refreshed by gossip."""
    def __init__(self):
        self.entries = {}   # url -> {"stages": dict, "expiry": float}

    def upsert(self, url: str, stages: dict, now: float, ttl: float) -> None:
        self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def merge(self, stages_by_url: dict, now: float, ttl: float) -> None:
        for url, stages in stages_by_url.items():
            self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def prune(self, now: float) -> None:
        self.entries = {u: e for u, e in self.entries.items() if e["expiry"] > now}

    def stages_by_url(self, now: float) -> dict:
        return {u: e["stages"] for u, e in self.entries.items() if e["expiry"] > now}


def build_chain(stages_by_url: dict, num_layers: int):
    """From {url: {'embed','head','decoders':[block_key]}} builds
    (embed_url, [(block_key, url)...], head_url) that tiles [0, num_layers).
    Returns None if coverage is incomplete or embed/head is missing."""
    embed = next((u for u, s in stages_by_url.items() if s.get("embed")), None)
    head = next((u for u, s in stages_by_url.items() if s.get("head")), None)
    if embed is None or head is None:
        return None
    ranges = []
    for u, s in stages_by_url.items():
        for bk in s.get("decoders", []):
            lo, hi = (int(x) for x in bk.split("-"))
            ranges.append((lo, hi, bk, u))
    ranges.sort()
    chain = []
    cursor = 0
    for lo, hi, bk, u in ranges:
        if lo == cursor and hi > cursor:
            chain.append((bk, u))
            cursor = hi
    if cursor != num_layers:
        return None
    return embed, chain, head
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_discovery.py -v` → 4 passed.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/discovery.py tests/test_discovery.py && git commit -m "feat(net): Registry gossip + build_chain (decentralized discovery)"
```

---

## Task 2: gossip in the BlockServer (`/registry` + loop)

**Files:** modify `eujeno/net/server.py`; create `tests/test_gossip_e2e.py`.

- [ ] **Step 1: test `tests/test_gossip_e2e.py`**
```python
import socket
import threading
import time

import pytest
import httpx
import uvicorn

from eujeno.net.topology import StageSpec
from eujeno.net.server import create_app


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


@pytest.mark.slow
def test_registry_converges_via_gossip(full_model):
    model, tokenizer = full_model
    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    # node 1 knows node 2 as a seed and vice versa
    app1 = create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                      node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3, ttl=30.0)
    app2 = create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                      node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3, ttl=30.0)
    s1, s2 = _serve(app1, p1), _serve(app2, p2)
    try:
        with httpx.Client(timeout=10.0) as client:
            converged = False
            for _ in range(100):   # ~ a few gossip rounds
                reg = client.get(f"{u1}/registry").json()
                if set(reg["nodes"].keys()) == {u1, u2}:
                    converged = True
                    break
                time.sleep(0.1)
            assert converged, reg
            assert reg["num_layers"] == 24
            assert reg["nodes"][u2]["head"] is True
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_gossip_e2e.py -m slow -v` → TypeError (create_app does not accept the new kwargs).

- [ ] **Step 3: modify `eujeno/net/server.py`**

Update the imports at the top:
```python
import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from eujeno.model.blocks import EmbedBlock, HeadBlock, DecoderBlock, prepare_decoder_block
from eujeno.net.wire import encode_tensors, decode_tensors
from eujeno.net.discovery import Registry
```
Replace the signature and the beginning of `create_app` to accept the gossip parameters (optional: without them the Part 1 behavior is unchanged) and register itself + start the loop:
```python
def create_app(model, tokenizer, stages, node_url=None, peers=None,
               num_layers=None, gossip_interval=2.0, ttl=30.0):
    embed_block = EmbedBlock(model.model.embed_tokens) if stages.embed else None
    head_block = HeadBlock(model.model.norm, model.lm_head) if stages.head else None
    prepared = {f"{lo}-{hi}": prepare_decoder_block(model, lo, hi) for (lo, hi) in stages.decoders}
    jobs = {}
    own_stages = {"embed": stages.embed, "head": stages.head, "decoders": list(prepared.keys())}
    registry = Registry()
    if node_url:
        registry.upsert(node_url, own_stages, now=time.time(), ttl=ttl)

    async def _gossip_loop():
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                now = time.time()
                if node_url:
                    registry.upsert(node_url, own_stages, now=now, ttl=ttl)
                for peer in (peers or []):
                    try:
                        resp = await client.get(f"{peer}/registry")
                        registry.merge(resp.json().get("nodes", {}), now=now, ttl=ttl)
                    except Exception:
                        pass
                registry.prune(now)
                await asyncio.sleep(gossip_interval)

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(_gossip_loop()) if node_url else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/registry")
    async def get_registry():
        return {"num_layers": num_layers, "model": getattr(model.config, "_name_or_path", "?"),
                "nodes": registry.stages_by_url(time.time())}
```
The REST of `create_app` (the `/health`, `/embed`, `/decode/{block_key}`, `/head`, `DELETE /job/{job_id}` endpoints and `return app`) stays **identical** to before — leave them unchanged below the definition of `get_registry`.

- [ ] **Step 4: run PASS** — `... pytest tests/test_gossip_e2e.py -m slow -v` → PASS (the registry converges to both nodes via gossip).
Verify that the Part 1 tests (`tests/test_server.py`, `tests/test_orchestrator.py`, `tests/test_cli_infer.py`) still pass (create_app backward-compatible): `... pytest tests/test_server.py tests/test_orchestrator.py -m slow -v`.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/net/server.py tests/test_gossip_e2e.py && git commit -m "feat(net): gossip discovery in the BlockServer (/registry + loop, backward-compatible)"
```

---

## Task 3: `eujeno serve --peers/--advertise` + `eujeno infer --peer`

**Files:** modify `eujeno/cli.py`; create `tests/test_infer_peer.py`.

- [ ] **Step 1: test `tests/test_infer_peer.py`**
```python
import json
import socket
import threading
import time

import pytest
import uvicorn

from typer.testing import CliRunner
from eujeno.cli import app as cli_app
from eujeno.net.topology import StageSpec
from eujeno.net.server import create_app
from eujeno.model.generate import reference_generate

runner = CliRunner()


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


@pytest.mark.slow
def test_infer_peer_autodiscovers_and_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("The capital of Italy is", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    p1, p2 = _free_port(), _free_port()
    u1, u2 = f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}"
    s1 = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=u1, peers=[u2], num_layers=24, gossip_interval=0.3), p1)
    s2 = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=u2, peers=[u1], num_layers=24, gossip_interval=0.3), p2)
    try:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            for _ in range(100):
                if set(client.get(f"{u1}/registry").json()["nodes"].keys()) == {u1, u2}:
                    break
                time.sleep(0.1)
        result = runner.invoke(cli_app, ["--json", "infer", "--peer", u1,
                                         "--prompt", "The capital of Italy is", "--max-new-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["data"]["tokens"] == reference
    finally:
        s1.should_exit = True
        s2.should_exit = True
```

- [ ] **Step 2: run FAIL** — `... pytest tests/test_infer_peer.py -m slow -v` → FAIL (`--peer` is missing).

- [ ] **Step 3: modify `eujeno/cli.py`**

Add imports near the other `from eujeno.net...`:
```python
from eujeno.net.discovery import build_chain
from eujeno.net.topology import Topology
```
Extend the `serve` command with the gossip options (direct mode): add the parameters and pass them to `create_app`. Add to the `serve` signature (in the direct, non-coordinator branch):
```python
    peers: str = typer.Option(None, "--peers", help="Seed peers for gossip discovery, comma-separated (e.g. http://other:8001)"),
    advertise: str = typer.Option(None, "--advertise", help="URL this node announces itself with (e.g. http://IP:8001). Default: http://<host>:<port>"),
    num_layers: int = typer.Option(None, "--num-layers", help="Total number of model layers (for coverage). Default: from config."),
```
and in the direct branch (the `else:` of `serve`, the one that does `create_app` + `uvicorn.run`) replace with:
```python
    else:
        import uvicorn
        own_url = advertise or f"http://{host}:{port}"
        seeds = [p.strip() for p in peers.split(",")] if peers else []
        nl = num_layers if num_layers is not None else model_config_dims(model_id)["num_layers"]
        fastapi_app = create_app(model, tokenizer, spec, node_url=own_url, peers=seeds, num_layers=nl)
        typer.echo(f"eujeno serve (P2P): stages={stages} on http://{host}:{port} advertise={own_url} peers={seeds}", err=True)
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
```
(`model_config_dims` is already imported in cli.py.)

Extend the `infer` command with the `--peer` mode (auto-discovery via gossip + direct transport). Add the `peer` option to the `infer` signature:
```python
    peer: str = typer.Option(None, "--peer", help="[P2P] URL of any node: discovers the topology via gossip and runs direct"),
```
and at the top of the `infer` body, after `prompt = _read_prompt(prompt)`, before the `--topology` branch, add the `--peer` branch:
```python
    if peer:
        import httpx
        from transformers import AutoTokenizer
        from eujeno.net.orchestrator import distributed_generate
        try:
            reg = httpx.get(f"{peer}/registry", timeout=30.0).json()
        except Exception as e:
            _fail("infer", "USAGE_ERROR", f"peer unreachable: {e}", exit_code=2)
        chain = build_chain(reg["nodes"], reg["num_layers"])
        if chain is None:
            _fail("infer", "NOT_OPERATIONAL", "incomplete coverage: the model is not yet operational on the network")
        embed_url, decoders, head_url = chain
        topo = Topology(model=reg["model"], embed=embed_url, head=head_url, decoders=decoders)
        try:
            tokenizer = AutoTokenizer.from_pretrained(topo.model)
            with httpx.Client(timeout=120.0) as client:
                result = distributed_generate(topo, prompt, max_new_tokens, client, tokenizer)
        except Exception as e:
            _fail("infer", "GENERATION_FAILED", str(e))
        _emit_ok("infer", {"model": topo.model, "prompt": prompt, **result}, human=result["text"])
        return
```

- [ ] **Step 4: run PASS** — `... pytest tests/test_infer_peer.py -m slow -v` → PASS. Verify that the Part 1 infer tests (`tests/test_cli_infer.py`) still pass.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add eujeno/cli.py tests/test_infer_peer.py && git commit -m "feat(cli): serve --peers/--advertise + infer --peer (pure P2P, auto-discovery)"
```

---

## Task 4: P2P quickstart + suite + ROADMAP

**Files:** create `docs/examples/p2p.md`; modify `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: create `docs/examples/p2p.md`**
```markdown
# Quickstart — pure P2P (decentralized, no central server)

Every node is equal: they discover each other via gossip (one seed is enough) and inference goes directly node-to-node. Requires the nodes to be reachable (LAN, VPN, or public IPs). For NAT-without-VPN use the coordinator mode instead (see coordinator.md).

```bash
# Node A — embedding + first 12 layers (first node, no seed)
eujeno serve --stages "embed,decoder:0-12" --port 8001 --advertise http://192.168.1.10:8001

# Node B — last 12 layers + head, knows A as a seed
eujeno serve --stages "decoder:12-24,head" --port 8001 \
  --advertise http://192.168.1.11:8001 --peers http://192.168.1.10:8001

# Inference: point at ANY node; it discovers the rest on its own
eujeno --json infer --peer http://192.168.1.10:8001 --prompt "The capital of Italy is"
```

Until coverage is complete (embed + all decoders + head), `infer` responds `NOT_OPERATIONAL`. Add nodes with different ranges and the network assembles itself progressively.
```

- [ ] **Step 2: update `README.md`** — in the Quickstart section, distinguish **pure P2P** (link `docs/examples/p2p.md`) and **coordinator** (link `docs/examples/coordinator.md`), explaining in one line when to use each.

- [ ] **Step 3: update `docs/ROADMAP.md`** — under "Discovery & Routing" mark the P2P discovery via gossip as done (link to this plan and to [ADR-0002](../decisions/ADR-0002-nat-connectivity.md)); update the "Last updated" line. Failover and libp2p remain to be done.

- [ ] **Step 4: full suite** — `/Users/alberto/Projects/AI/eujeno/.venv/bin/python -m pytest -q -p no:warnings` → all PASS.

- [ ] **Step 5: commit**
```bash
cd /Users/alberto/Projects/AI/eujeno && git add docs/examples/p2p.md README.md docs/ROADMAP.md && git commit -m "docs: pure P2P quickstart; ROADMAP gossip discovery"
```

---

## Self-Review

**Coverage (ADR-0002 Mode A):** decentralized discovery via gossip (Task 1 Registry + Task 2 loop) ✓; self-announce + coverage gate (build_chain) ✓; direct transport reused from Part 1 ✓; CLI `serve --peers/--advertise` + `infer --peer` ✓; backward-compatibility with Part 1 static mode ✓.

**Placeholder scan:** no TODO/TBD; complete code. `create_app` stays backward-compatible (new optional kwargs).

**Type consistency:** `Registry.upsert/merge/prune/stages_by_url(now, ttl)`, `build_chain(stages_by_url, num_layers)->(embed,decoders,head)|None`, `create_app(..., node_url, peers, num_layers, gossip_interval, ttl)`, `Topology(model, embed, head, decoders)` consistent. `build_chain` returns `(embed_url, [(block_key,url)], head_url)`, consumed to build `Topology` and then `distributed_generate` (Part 1).
```
