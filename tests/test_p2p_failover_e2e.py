import socket, threading, time
import pytest, httpx, uvicorn
from starlette.responses import JSONResponse

from eujeno.net.server import create_app
from eujeno.net.topology import StageSpec
from eujeno.net.orchestrator import distributed_generate_resilient
from eujeno.net.generation import stop_token_ids
from eujeno.model.generate import reference_generate


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server


class _FlakyDecode:
    """ASGI wrapper: 503 on /decode after `die_after` calls; everything else (lifespan,
    gossip, /embed, /head, /registry) passes through normally."""
    def __init__(self, app, die_after):
        self.app = app; self.n = 0; self.die_after = die_after

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path", "").startswith("/decode"):
            self.n += 1
            if self.n >= self.die_after:
                await JSONResponse({"error": "dead"}, status_code=503)(scope, receive, send)
                return
        await self.app(scope, receive, send)


@pytest.mark.slow
def test_p2p_failover_reroutes_and_resumes(full_model):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    pA, pB, pC = _free_port(), _free_port(), _free_port()
    uA, uB, uC = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}", f"http://127.0.0.1:{pC}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB, uC], num_layers=24, gossip_interval=0.3), pA)
    sB = _serve(_FlakyDecode(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                             node_url=uB, peers=[uA, uC], num_layers=24, gossip_interval=0.3), die_after=4), pB)
    sC = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uC, peers=[uA, uB], num_layers=24, gossip_interval=0.3), pC)
    try:
        # ordered so build_chain picks B (flaky) first, then fails over to C
        stages = {uA: {"embed": True, "head": False, "decoders": ["0-12"]},
                  uB: {"embed": False, "head": True, "decoders": ["12-24"]},
                  uC: {"embed": False, "head": True, "decoders": ["12-24"]}}
        with httpx.Client(timeout=60.0) as client:
            result = distributed_generate_resilient(stages, 24, prompt, 6, client, tokenizer,
                                                    stop_ids=stop_token_ids(tokenizer))
        assert result["ok"] is True, result
        assert result["tokens"] == reference
        assert result["failovers"] >= 1            # B died mid-generation, C resumed
    finally:
        sA.should_exit = sB.should_exit = sC.should_exit = True


@pytest.mark.slow
def test_p2p_stops_at_eos(full_model):
    model, tokenizer = full_model
    prompt = "La capitale dell'Italia è"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)

    pA, pB = _free_port(), _free_port()
    uA, uB = f"http://127.0.0.1:{pA}", f"http://127.0.0.1:{pB}"
    sA = _serve(create_app(model, tokenizer, StageSpec(embed=True, decoders=[(0, 12)]),
                           node_url=uA, peers=[uB], num_layers=24, gossip_interval=0.3), pA)
    sB = _serve(create_app(model, tokenizer, StageSpec(head=True, decoders=[(12, 24)]),
                           node_url=uB, peers=[uA], num_layers=24, gossip_interval=0.3), pB)
    try:
        stages = {uA: {"embed": True, "head": False, "decoders": ["0-12"]},
                  uB: {"embed": False, "head": True, "decoders": ["12-24"]}}
        with httpx.Client(timeout=60.0) as client:
            # make the very first token a stop token -> generation must stop immediately
            result = distributed_generate_resilient(stages, 24, prompt, 6, client, tokenizer,
                                                    stop_ids={reference[0]})
        assert result["ok"] is True
        assert result["tokens"] == []
        assert result["finish_reason"] == "stop"
        assert stop_token_ids(tokenizer) and tokenizer.eos_token_id in stop_token_ids(tokenizer)
    finally:
        sA.should_exit = sB.should_exit = True
