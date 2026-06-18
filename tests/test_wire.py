import torch
from axyn.net.wire import encode_tensors, decode_tensors


def test_roundtrip_preserves_tensors_and_dtype():
    tensors = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "hidden_states": torch.randn(1, 3, 8, dtype=torch.float32),
    }
    back = decode_tensors(encode_tensors(tensors))
    assert torch.equal(back["input_ids"], tensors["input_ids"])
    assert back["input_ids"].dtype == torch.long
    assert torch.equal(back["hidden_states"], tensors["hidden_states"])
