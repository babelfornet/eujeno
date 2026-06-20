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
