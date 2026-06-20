import torch
from eujeno.net.sampling import sample_token


def test_greedy_returns_argmax_when_temperature_zero():
    ids = [10, 20, 30]
    logits = [1.0, 5.0, 2.0]
    out = sample_token(ids, logits, generated_ids=[], temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0, generator=None)
    assert out == 20


def test_sampling_is_deterministic_with_seed():
    ids = [10, 20, 30, 40]
    logits = [2.0, 2.0, 2.0, 2.0]
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_token(ids, logits, [], 1.0, 1.0, 1.0, g1)
    b = sample_token(ids, logits, [], 1.0, 1.0, 1.0, g2)
    assert a == b and a in ids


def test_repetition_penalty_demotes_generated_tokens():
    ids = [10, 20]
    logits = [5.0, 1.0]
    out = sample_token(ids, logits, generated_ids=[10], temperature=0.0,
                       top_p=1.0, repetition_penalty=10.0, generator=None)
    assert out == 20
