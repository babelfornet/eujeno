import pytest
import torch

from axyn.config import DEFAULT_MODEL_ID, DTYPE, DEVICE
from axyn.model.loader import load_partial_model
from axyn.net.topology import StageSpec
from axyn.net.node_exec import NodeState, handle_request
from axyn.net.wire import encode_tensors, decode_tensors
from axyn.model.generate import reference_generate


def _greedy_two_nodes(s1, s2, ids, max_new):
    L = ids.shape[1]
    cache_position = torch.arange(L)
    cur = ids
    out = []
    for step in range(max_new):
        _, p = handle_request(s1, {"op": "embed", "job_id": "j"}, encode_tensors({"input_ids": cur}))
        h = decode_tensors(p)["hidden_states"]
        _, p = handle_request(s1, {"op": "decode", "block_key": "0-12", "job_id": "j"},
                              encode_tensors({"hidden_states": h, "cache_position": cache_position}))
        h = decode_tensors(p)["hidden_states"]
        _, p = handle_request(s2, {"op": "decode", "block_key": "12-24", "job_id": "j"},
                              encode_tensors({"hidden_states": h, "cache_position": cache_position}))
        h = decode_tensors(p)["hidden_states"]
        rh, _ = handle_request(s2, {"op": "head", "job_id": "j"}, encode_tensors({"hidden_states": h}))
        out.append(rh["token_id"])
        cur = torch.tensor([[rh["token_id"]]])
        cache_position = torch.tensor([L + step])
    return out


@pytest.mark.slow
def test_two_partial_nodes_match_reference(full_model):
    full, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(full, ids, max_new_tokens=6)

    # ogni nodo carica SOLO i suoi stage
    spec1 = StageSpec(embed=True, decoders=[(0, 12)])
    spec2 = StageSpec(head=True, decoders=[(12, 24)])
    m1, _ = load_partial_model(DEFAULT_MODEL_ID, spec1, DTYPE, DEVICE)
    m2, _ = load_partial_model(DEFAULT_MODEL_ID, spec2, DTYPE, DEVICE)
    s1 = NodeState(m1, spec1)
    s2 = NodeState(m2, spec2)

    assert _greedy_two_nodes(s1, s2, ids, 6) == reference


@pytest.mark.slow
def test_partial_decoder_node_does_not_materialize_other_layers():
    # un nodo che serve solo decoder 0-12 NON deve avere materializzato i layer 12-24
    spec = StageSpec(decoders=[(0, 12)])
    model, _ = load_partial_model(DEFAULT_MODEL_ID, spec, DTYPE, DEVICE)
    # i layer assegnati hanno pesi reali (non meta)
    assert model.model.layers[0].self_attn.q_proj.weight.device.type != "meta"
    # un layer NON assegnato resta su meta (zero memoria)
    assert model.model.layers[20].self_attn.q_proj.weight.device.type == "meta"
