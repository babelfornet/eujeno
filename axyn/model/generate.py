# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache


@torch.no_grad()
def reference_generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    """Greedy decode col modello intero (riferimento). Deterministico."""
    cache = DynamicCache()
    seq_len = input_ids.shape[1]
    cur = input_ids
    cache_position = torch.arange(seq_len)
    generated: list[int] = []
    for step in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                    cache_position=cache_position)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        cur = next_id
        cache = out.past_key_values
        cache_position = torch.tensor([seq_len + step])
    return generated


@torch.no_grad()
def pipeline_generate(embed, decoders, head, input_ids: torch.Tensor,
                      max_new_tokens: int) -> list[int]:
    """Greedy decode attraverso i blocchi splittati, con KV-cache per-blocco
    (session affinity). Deve riprodurre reference_generate."""
    seq_len = input_ids.shape[1]
    cur_ids = input_ids
    cache_position = torch.arange(seq_len)
    generated: list[int] = []
    for step in range(max_new_tokens):
        h = embed.run_block(cur_ids)
        for d in decoders:
            h = d.run_block(h, cache_position)
        logits = head.run_block(h)[:, -1, :]
        next_id = logits.argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        cur_ids = next_id
        cache_position = torch.tensor([seq_len + step])
    return generated
