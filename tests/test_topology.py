import pytest
from synapse.net.topology import parse_stages, StageSpec, Topology, load_topology


def test_parse_stages_all_kinds():
    s = parse_stages("embed,decoder:0-12,head")
    assert s.embed is True
    assert s.head is True
    assert s.decoders == [(0, 12)]


def test_parse_stages_multiple_decoders():
    s = parse_stages("decoder:0-8,decoder:8-16")
    assert s.embed is False and s.head is False
    assert s.decoders == [(0, 8), (8, 16)]


def test_parse_stages_rejects_garbage():
    with pytest.raises(ValueError):
        parse_stages("frobnicate")


def test_load_topology_resolves_stages():
    topo = load_topology({
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "embed": "http://a:1",
        "decoders": [{"block": "0-12", "url": "http://a:1"}, {"block": "12-24", "url": "http://b:2"}],
        "head": "http://b:2",
    })
    assert topo.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert topo.embed == "http://a:1"
    assert topo.head == "http://b:2"
    assert topo.decoders == [("0-12", "http://a:1"), ("12-24", "http://b:2")]
    assert set(topo.all_urls()) == {"http://a:1", "http://b:2"}
