from axyn.net.discovery import Registry, build_chain


def test_build_chain_full_coverage():
    reg = {
        "http://a": {"embed": True, "head": False, "decoders": ["0-12"]},
        "http://b": {"embed": False, "head": True, "decoders": ["12-24"]},
    }
    chain = build_chain(reg, 24)
    assert chain == ("http://a", [("0-12", "http://a"), ("12-24", "http://b")], "http://b")


def test_build_chain_incomplete_returns_none():
    reg = {"http://a": {"embed": True, "head": True, "decoders": ["0-12"]}}
    assert build_chain(reg, 24) is None


def test_registry_merge_and_prune_with_ttl():
    r = Registry()
    r.upsert("http://a", {"embed": True, "head": False, "decoders": ["0-24"]}, now=100.0, ttl=60.0)
    r.merge({"http://b": {"head": True, "embed": False, "decoders": []}}, now=100.0, ttl=60.0)
    assert set(r.stages_by_url(now=120.0).keys()) == {"http://a", "http://b"}
    r.prune(now=200.0)
    assert r.stages_by_url(now=200.0) == {}


def test_registry_refresh_extends_expiry():
    r = Registry()
    r.upsert("http://a", {"embed": True, "head": True, "decoders": ["0-24"]}, now=100.0, ttl=60.0)
    r.upsert("http://a", {"embed": True, "head": True, "decoders": ["0-24"]}, now=150.0, ttl=60.0)
    assert "http://a" in r.stages_by_url(now=200.0)


def test_build_chain_excludes_failed_node_uses_redundant():
    reg = {
        "A": {"embed": True, "head": False, "decoders": ["0-12"]},
        "B": {"embed": False, "head": True, "decoders": ["12-24"]},
        "C": {"embed": False, "head": True, "decoders": ["12-24"]},  # ridondante con B
    }
    assert build_chain(reg, 24) is not None
    chain = build_chain(reg, 24, exclude={"B"})
    assert chain is not None
    _, decoders, head = chain
    assert ("12-24", "C") in decoders
    assert head == "C"


def test_build_chain_exclude_breaks_coverage_returns_none():
    reg = {
        "A": {"embed": True, "head": True, "decoders": ["0-12"]},
        "B": {"embed": False, "head": False, "decoders": ["12-24"]},
    }
    assert build_chain(reg, 24, exclude={"B"}) is None
