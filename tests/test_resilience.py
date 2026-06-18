import pytest
import torch
from axyn.model.generate import reference_generate
from axyn.model.blocks import split_into_blocks
from axyn.model.cache import cache_to_bytes, cache_from_bytes


@pytest.mark.slow
def test_generation_survives_cache_serialization_midway(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=8)

    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    seq_len = ids.shape[1]
    cache_position = torch.arange(seq_len)
    cur_ids = ids
    generated = []
    for step in range(8):
        h = embed.run_block(cur_ids)
        for d in decoders:
            h = d.run_block(h, cache_position)
        next_id = head.run_block(h)[:, -1, :].argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        cur_ids = next_id
        cache_position = torch.tensor([seq_len + step])

        if step == 3:  # simula handoff/restart: serializza e ricarica ogni cache
            for d in decoders:
                d.set_cache(cache_from_bytes(cache_to_bytes(d.get_cache())))

    assert generated == reference
