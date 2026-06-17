import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float32   # fp32 su CPU per determinismo (vedi ADR-0001 Fork D)
DEVICE = "cpu"

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}

SUPPORTED_ARCHS = {"qwen2", "llama"}


def parse_dtype(name: str):
    key = str(name).lower()
    if key not in _DTYPES:
        raise ValueError(f"dtype non valido: {name!r} (usa float32/bfloat16/float16)")
    return _DTYPES[key]
