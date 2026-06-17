import pytest
import torch
from synapse.net.node_exec import NodeState, handle_request
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.topology import StageSpec
from synapse.model.generate import reference_generate


@pytest.mark.slow
def test_handle_request_greedy_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA di NodeState (remap)

    state = NodeState(model, StageSpec(embed=True, head=True, decoders=[(0, 24)]))
    L = ids.shape[1]
    cache_position = torch.arange(L)
    cur = ids
    generated = []
    for step in range(6):
        _, p = handle_request(state, {"op": "embed", "job_id": "j"}, encode_tensors({"input_ids": cur}))
        h = decode_tensors(p)["hidden_states"]
        _, p = handle_request(state, {"op": "decode", "block_key": "0-24", "job_id": "j"},
                              encode_tensors({"hidden_states": h, "cache_position": cache_position}))
        h = decode_tensors(p)["hidden_states"]
        rh, _ = handle_request(state, {"op": "head", "job_id": "j"}, encode_tensors({"hidden_states": h}))
        generated.append(rh["token_id"])
        cur = torch.tensor([[rh["token_id"]]])
        cache_position = torch.tensor([L + step])

    assert generated == reference
