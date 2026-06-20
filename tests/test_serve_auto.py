from eujeno.cli import plan_auto_stages

DIMS = {"num_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
        "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936}


def test_plan_first_node_claims_whole_when_capable():
    s = plan_auto_stages(DIMS, bytes_per=4, ram_gb=64.0, reserve=0.2,
                         stages_by_url={}, target=1)
    assert s == "embed,decoder:0-24,head"


def test_plan_second_node_fills_remaining_gap():
    existing = {"a": {"embed": True, "head": False, "decoders": ["0-12"]}}
    s = plan_auto_stages(DIMS, bytes_per=4, ram_gb=0.6, reserve=0.2,
                         stages_by_url=existing, target=1)
    assert s.startswith("decoder:12-")
