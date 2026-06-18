import pytest
import torch
from axyn.model.blocks import prepare_decoder_block, DecoderBlock


@pytest.mark.slow
def test_prepare_returns_local_indexed_layers(full_model):
    model, _ = full_model
    layers, rotary = prepare_decoder_block(model, 0, 12)
    assert len(layers) == 12
    assert [layer.self_attn.layer_idx for layer in layers] == list(range(12))   # indici locali 0..11
    block = DecoderBlock(layers, rotary)
    h = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float32)
    out = block.run_block(h, torch.arange(3))
    assert out.shape == h.shape
