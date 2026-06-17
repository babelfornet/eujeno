import safetensors.torch


def encode_tensors(tensors: dict) -> bytes:
    """Serializza un dict nome->Tensor in bytes safetensors (per il body HTTP)."""
    return safetensors.torch.save({k: v.contiguous() for k, v in tensors.items()})


def decode_tensors(data: bytes) -> dict:
    """Deserializza bytes safetensors in un dict nome->Tensor."""
    return safetensors.torch.load(data)
