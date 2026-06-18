import pytest
import torch
from axyn.model.blocks import split_into_blocks


@pytest.mark.slow
def test_embed_block_matches_model_embedding(full_model):
    model, tokenizer = full_model
    ids = tokenizer("Ciao mondo", return_tensors="pt").input_ids
    expected = model.model.embed_tokens(ids)
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    out = embed.run_block(ids)
    assert torch.equal(out, expected)


@pytest.mark.slow
def test_head_block_matches_model_head(full_model):
    model, tokenizer = full_model
    h = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float32)
    expected = model.lm_head(model.model.norm(h))
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    out = head.run_block(h)
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.slow
def test_decoder_blocks_cover_all_layers(full_model):
    model, _ = full_model
    embed, decoders, head = split_into_blocks(model, boundaries=[0, 12, 24])
    assert len(decoders) == 2
    assert sum(len(d.layers) for d in decoders) == 24
