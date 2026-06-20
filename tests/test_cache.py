import torch
from transformers import DynamicCache
from eujeno.model.cache import cache_to_bytes, cache_from_bytes


def _make_cache(num_layers, seq=4, heads=2, head_dim=8):
    legacy = tuple(
        (torch.randn(1, heads, seq, head_dim), torch.randn(1, heads, seq, head_dim))
        for _ in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(legacy)


def test_cache_roundtrip_preserves_tensors():
    cache = _make_cache(num_layers=3)
    data = cache_to_bytes(cache)
    restored = cache_from_bytes(data)
    orig, back = cache.to_legacy_cache(), restored.to_legacy_cache()
    assert len(back) == 3
    for (k0, v0), (k1, v1) in zip(orig, back):
        assert torch.equal(k0, k1)
        assert torch.equal(v0, v1)


def test_cache_roundtrip_preserves_seq_length():
    cache = _make_cache(num_layers=2, seq=7)
    restored = cache_from_bytes(cache_to_bytes(cache))
    assert restored.get_seq_length() == 7
