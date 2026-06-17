import pytest
import torch
from fastapi.testclient import TestClient
from synapse.net.wire import encode_tensors, decode_tensors
from synapse.net.topology import StageSpec
from synapse.net.server import create_app
from synapse.model.generate import reference_generate


@pytest.mark.slow
def test_single_node_serving_all_stages_matches_reference(full_model):
    model, tokenizer = full_model
    ids = tokenizer("La capitale dell'Italia è", return_tensors="pt").input_ids
    reference = reference_generate(model, ids, max_new_tokens=6)   # PRIMA di create_app (remap)

    app = create_app(model, tokenizer, StageSpec(embed=True, head=True, decoders=[(0, 24)]))
    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True

    L = ids.shape[1]
    cache_position = torch.arange(L)
    cur_ids = ids
    generated = []
    for step in range(6):
        r = client.post("/embed", params={"job_id": "j"}, content=encode_tensors({"input_ids": cur_ids}))
        h = decode_tensors(r.content)["hidden_states"]
        r = client.post("/decode/0-24", params={"job_id": "j"},
                        content=encode_tensors({"hidden_states": h, "cache_position": cache_position}))
        h = decode_tensors(r.content)["hidden_states"]
        r = client.post("/head", params={"job_id": "j"}, content=encode_tensors({"hidden_states": h}))
        token_id = r.json()["token_id"]
        generated.append(token_id)
        cur_ids = torch.tensor([[token_id]])
        cache_position = torch.tensor([L + step])

    assert generated == reference
