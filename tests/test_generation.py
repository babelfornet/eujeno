import asyncio
import torch
from transformers import AutoTokenizer
from axyn.net.generation import generate_tokens


def _tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_greedy_loop_stops_at_eos():
    tokenizer = _tok()
    scripted = [101, 102, 103, tokenizer.eos_token_id]   # 4° token = EOS -> stop, non incluso

    async def run_embed(cur):
        return torch.zeros(1, cur.shape[1], 4)

    async def run_decoders(h, cache_position):
        return h

    calls = {"i": 0}

    async def run_head(h, topk):
        tid = scripted[calls["i"]]; calls["i"] += 1
        return {"token_id": tid, "topk_ids": [tid], "topk_logits": [1.0]}

    tokens, prompt_len, finish = asyncio.run(generate_tokens(
        tokenizer, "ciao", max_new=10, sampling={}, stop_ids={tokenizer.eos_token_id},
        run_embed=run_embed, run_decoders=run_decoders, run_head=run_head))
    assert tokens == [101, 102, 103]
    assert finish == "stop"
    assert prompt_len >= 1
