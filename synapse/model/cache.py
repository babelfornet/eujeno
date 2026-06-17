import safetensors.torch
from transformers import DynamicCache


def cache_to_bytes(cache: DynamicCache) -> bytes:
    """Serializza una DynamicCache (per-blocco) in bytes safetensors."""
    legacy = cache.to_legacy_cache()
    tensors = {}
    for i, (key, value) in enumerate(legacy):
        tensors[f"key_{i}"] = key.contiguous()
        tensors[f"value_{i}"] = value.contiguous()
    return safetensors.torch.save(tensors)


def cache_from_bytes(data: bytes) -> DynamicCache:
    tensors = safetensors.torch.load(data)
    num_layers = len(tensors) // 2
    legacy = tuple(
        (tensors[f"key_{i}"], tensors[f"value_{i}"]) for i in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(legacy)
