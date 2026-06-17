import torch


def build_causal_mask(cache_position: torch.Tensor, kv_len: int,
                      dtype: torch.dtype, device: str) -> torch.Tensor:
    """Maschera additiva causale 4D [1,1,q_len,kv_len] per batch=1 senza padding.
    cache_position contiene le posizioni assolute dei token di query."""
    q_len = cache_position.shape[0]
    key_pos = torch.arange(kv_len, device=device)
    allowed = key_pos[None, :] <= cache_position[:, None].to(device)   # [q_len, kv_len] bool
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]
