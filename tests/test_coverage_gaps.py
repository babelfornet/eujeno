from synapse.net.discovery import coverage_gaps

A = {"embed": True, "head": False, "decoders": ["0-12"]}
B = {"embed": False, "head": True, "decoders": ["12-24"]}


def test_full_coverage_no_gaps():
    g = coverage_gaps({"a": A, "b": B}, 24, target=1)
    assert g["decoder_gaps"] == []
    assert g["embed_replicas"] == 1 and g["head_replicas"] == 1


def test_missing_middle_range():
    g = coverage_gaps({"a": A}, 24, target=1)
    assert g["decoder_gaps"] == [{"lo": 12, "hi": 24, "replicas": 0}]


def test_under_replicated_with_target_2():
    g = coverage_gaps({"a": A, "b": B}, 24, target=2)
    assert g["decoder_gaps"] == [{"lo": 0, "hi": 24, "replicas": 1}]
    assert g["embed_replicas"] == 1 and g["head_replicas"] == 1
