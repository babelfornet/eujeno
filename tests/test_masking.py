import torch
from axyn.model.masking import build_causal_mask


def test_prefill_mask_is_lower_triangular():
    cache_position = torch.arange(3)        # prefill di 3 token, kv_len=3
    mask = build_causal_mask(cache_position, kv_len=3, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 3, 3)
    neg = torch.finfo(torch.float32).min
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] == neg
    assert mask[0, 0, 0, 2] == neg
    assert torch.all(mask[0, 0, 2, :] == 0.0)


def test_decode_step_attends_to_all_past():
    cache_position = torch.tensor([5])      # 1 query token in posizione 5, kv_len=6
    mask = build_causal_mask(cache_position, kv_len=6, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 1, 6)
    assert torch.all(mask[0, 0, 0, :] == 0.0)
