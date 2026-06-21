# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import torch

from eujeno.net.wire import encode_tensors, decode_tensors


def distributed_generate(topology, prompt: str, max_new_tokens: int, client, tokenizer,
                         job_id: str = "job") -> dict:
    """Entry node (Milestone 0): drives greedy autoregressive generation by calling
    the topology's BlockServers over HTTP. Returns {'text', 'tokens'}."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    cache_position = torch.arange(seq_len)
    cur_ids = ids
    tokens = []
    try:
        for step in range(max_new_tokens):
            r = client.post(f"{topology.embed}/embed", params={"job_id": job_id},
                            content=encode_tensors({"input_ids": cur_ids}))
            r.raise_for_status()
            h = decode_tensors(r.content)["hidden_states"]

            for block_key, url in topology.decoders:
                r = client.post(f"{url}/decode/{block_key}", params={"job_id": job_id},
                                content=encode_tensors({"hidden_states": h, "cache_position": cache_position}))
                r.raise_for_status()
                h = decode_tensors(r.content)["hidden_states"]

            r = client.post(f"{topology.head}/head", params={"job_id": job_id},
                            content=encode_tensors({"hidden_states": h}))
            r.raise_for_status()
            token_id = r.json()["token_id"]

            tokens.append(token_id)
            cur_ids = torch.tensor([[token_id]])
            cache_position = torch.tensor([seq_len + step])
    finally:
        for url in topology.all_urls():
            try:
                client.delete(f"{url}/job/{job_id}")
            except Exception:
                pass

    return {"text": tokenizer.decode(tokens), "tokens": tokens}


def distributed_generate_resilient(stages_by_url, num_layers, prompt, max_new_tokens, client,
                                   tokenizer, stop_ids=None, job_id_prefix="job",
                                   refresh=None, max_failovers=5, coverage_timeout=0.0,
                                   poll_interval=0.5):
    """Pure-P2P entry: greedy distributed generation with failover. On a peer hop failure,
    exclude that peer, rebuild the chain from the gossip registry, and resume from the
    tokens already produced (prefix replay). Stops at EOS (stop_ids)."""
    import time
    from eujeno.net.discovery import build_chain
    stop_ids = stop_ids or set()
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    tokens = []
    excluded = set()
    finish_reason = "length"

    deadline = time.monotonic() + coverage_timeout

    def _resolve_chain(deadline):
        nonlocal stages_by_url
        while True:
            chain = build_chain(stages_by_url, num_layers, exclude=excluded)
            if chain is not None:
                return chain
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval)
            if refresh is not None:
                try:
                    fresh = refresh()
                    if fresh:
                        stages_by_url = fresh
                except Exception:
                    pass

    for attempt in range(max_failovers + 1):
        chain = _resolve_chain(deadline)
        if chain is None:
            err = ("coverage timeout: model not operational" if coverage_timeout > 0
                   else "incomplete coverage: model not operational")
            return {"ok": False, "error": err, "tokens": tokens, "failovers": attempt}
        embed_url, decoders, head_url = chain
        job_id = f"{job_id_prefix}{attempt}"
        current = None
        try:
            if tokens:
                cur_ids = torch.cat([ids, torch.tensor([tokens], dtype=ids.dtype)], dim=1)
                cache_position = torch.arange(seq_len + len(tokens))
            else:
                cur_ids = ids
                cache_position = torch.arange(seq_len)
            while len(tokens) < max_new_tokens:
                current = embed_url
                r = client.post(f"{embed_url}/embed", params={"job_id": job_id},
                                content=encode_tensors({"input_ids": cur_ids})); r.raise_for_status()
                h = decode_tensors(r.content)["hidden_states"]
                for block_key, url in decoders:
                    current = url
                    r = client.post(f"{url}/decode/{block_key}", params={"job_id": job_id},
                                    content=encode_tensors({"hidden_states": h, "cache_position": cache_position})); r.raise_for_status()
                    h = decode_tensors(r.content)["hidden_states"]
                current = head_url
                r = client.post(f"{head_url}/head", params={"job_id": job_id},
                                content=encode_tensors({"hidden_states": h})); r.raise_for_status()
                token_id = r.json()["token_id"]
                if token_id in stop_ids:
                    finish_reason = "stop"
                    break
                tokens.append(token_id)
                cur_ids = torch.tensor([[token_id]])
                cache_position = torch.tensor([seq_len + len(tokens) - 1])
            for url in {embed_url, head_url, *(u for _, u in decoders)}:
                try:
                    client.delete(f"{url}/job/{job_id}")
                except Exception:
                    pass
            return {"ok": True, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "tokens": tokens, "failovers": attempt, "finish_reason": finish_reason}
        except Exception:
            if current is not None:
                excluded.add(current)
    return {"ok": False, "error": "too many failovers", "tokens": tokens, "failovers": max_failovers}
