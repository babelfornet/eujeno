# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import random

import torch

from eujeno.net.sampling import sample_token


def stop_token_ids(tokenizer):
    """EOS + chat-end special tokens to stop generation on."""
    ids = set()
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id:
            ids.add(int(i))
    return ids


async def generate_tokens(tokenizer, prompt, max_new, sampling, stop_ids,
                          run_embed, run_decoders, run_head):
    """Autoregressive loop with injected transport.
    run_embed(cur_ids)->hidden ; run_decoders(hidden, cache_position)->hidden (all blocks) ;
    run_head(hidden, topk)->{token_id, topk_ids, topk_logits}. Returns (tokens, prompt_len, finish_reason)."""
    temperature = float(sampling.get("temperature", 0.0) or 0.0)
    top_p = float(sampling.get("top_p", 1.0) or 1.0)
    rep = float(sampling.get("repetition_penalty", 1.0) or 1.0)
    do_sample = temperature > 0.0
    generator = None
    if do_sample:
        seed = sampling.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        generator = torch.Generator().manual_seed(seed)
    topk = 100 if do_sample else 1

    ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = ids.shape[1]
    cache_position = torch.arange(seq_len)
    cur = ids
    tokens = []
    finish_reason = "length"
    for step in range(max_new):
        h = await run_embed(cur)
        h = await run_decoders(h, cache_position)
        rh = await run_head(h, topk)
        if do_sample:
            tok = sample_token(rh["topk_ids"], rh["topk_logits"], tokens, temperature, top_p, rep, generator)
        else:
            tok = rh["token_id"]
        if tok in stop_ids:
            finish_reason = "stop"
            break
        tokens.append(tok)
        cur = torch.tensor([[tok]])
        cache_position = torch.tensor([seq_len + step])
    return tokens, seq_len, finish_reason
