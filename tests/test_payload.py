import torch
from axyn.model.payload import HopPayload


def test_payload_roundtrip():
    p = HopPayload(
        job_id="job-abc",
        hop=2,
        token_position=5,
        hidden_states=torch.randn(1, 1, 896),
        position_ids=torch.tensor([[5]]),
        cache_position=torch.tensor([5]),
        attention_mask=None,
    )
    back = HopPayload.from_bytes(p.to_bytes())
    assert back.job_id == "job-abc"
    assert back.hop == 2
    assert back.token_position == 5
    assert torch.equal(back.hidden_states, p.hidden_states)
    assert torch.equal(back.position_ids, p.position_ids)
    assert torch.equal(back.cache_position, p.cache_position)
    assert back.attention_mask is None


def test_payload_roundtrip_with_attention_mask():
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    mask[0, 0, 0, 1] = torch.finfo(torch.float32).min
    p = HopPayload(
        job_id="job-xyz",
        hop=0,
        token_position=0,
        hidden_states=torch.randn(1, 3, 896),
        position_ids=torch.tensor([[0, 1, 2]]),
        cache_position=torch.tensor([0, 1, 2]),
        attention_mask=mask,
    )
    back = HopPayload.from_bytes(p.to_bytes())
    assert back.attention_mask is not None
    assert torch.equal(back.attention_mask, mask)
