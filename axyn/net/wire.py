# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import safetensors.torch


def encode_tensors(tensors: dict) -> bytes:
    """Serializes a name->Tensor dict into safetensors bytes (for the HTTP body)."""
    return safetensors.torch.save({k: v.contiguous() for k, v in tensors.items()})


def decode_tensors(data: bytes) -> dict:
    """Deserializes safetensors bytes into a name->Tensor dict."""
    return safetensors.torch.load(data)
