# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float32   # fp32 su CPU per determinismo (vedi ADR-0001 Fork D)
DEVICE = "cpu"          # library default — kept on CPU for deterministic golden tests


def auto_device() -> str:
    """Best compute device available on THIS machine (where the node runs).

    Prefers the GPU — Apple ``mps`` or CUDA — and falls back to ``cpu``. The
    library default (:data:`DEVICE`) stays ``cpu`` for deterministic tests;
    only the CLI opts into the GPU automatically when no ``--device`` is given.
    """
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_device(device) -> str:
    """Resolve a user-supplied ``--device``: ``None``/``"auto"`` → auto-detect the
    best local device; an explicit ``cpu``/``cuda``/``mps`` is used as-is."""
    if device is None or str(device).lower() == "auto":
        return auto_device()
    return device

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}

SUPPORTED_ARCHS = {"qwen2", "llama"}


def parse_dtype(name: str):
    key = str(name).lower()
    if key not in _DTYPES:
        raise ValueError(f"invalid dtype: {name!r} (use float32/bfloat16/float16)")
    return _DTYPES[key]
