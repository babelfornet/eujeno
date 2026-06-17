import torch
import pytest
from synapse.config import parse_dtype


def test_parse_known():
    assert parse_dtype("float32") is torch.float32
    assert parse_dtype("bf16") is torch.bfloat16
    assert parse_dtype("bfloat16") is torch.bfloat16
    assert parse_dtype("fp16") is torch.float16


def test_parse_unknown_raises():
    with pytest.raises(ValueError):
        parse_dtype("int4")
