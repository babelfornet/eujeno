from axyn.net.allocator import choose_stages


def gaps(decoder_gaps, e=0, h=0, target=1):
    return {"decoder_gaps": decoder_gaps, "embed_replicas": e, "head_replicas": h, "target": target}


def test_takes_neediest_decoder_gap_capped_by_capacity():
    g = gaps([{"lo": 12, "hi": 24, "replicas": 0}], e=1, h=1)
    assert choose_stages(g, max_decoder_layers=5, num_layers=24, take_embed_head=False) == "decoder:12-17"


def test_claims_embed_head_when_uncovered_and_capable():
    g = gaps([{"lo": 0, "hi": 24, "replicas": 0}], e=0, h=0)
    s = choose_stages(g, max_decoder_layers=99, num_layers=24, take_embed_head=True)
    assert s == "embed,decoder:0-24,head"


def test_prefers_lower_replication_first():
    g = gaps([{"lo": 0, "hi": 6, "replicas": 1}, {"lo": 6, "hi": 12, "replicas": 0}], e=1, h=1, target=2)
    assert choose_stages(g, max_decoder_layers=99, num_layers=12, take_embed_head=False) == "decoder:6-12"


def test_no_gaps_returns_empty():
    assert choose_stages(gaps([], e=1, h=1), max_decoder_layers=10, num_layers=24, take_embed_head=False) == ""
