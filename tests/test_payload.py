import torch
from synapse.model.payload import HopPayload


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
