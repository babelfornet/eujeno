from eujeno.net.discovery import Registry, build_chain


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


def test_merge_stale_hb_does_not_refresh_expiry():
    # A relayed entry carrying a heartbeat we've ALREADY seen must not extend the
    # entry's expiry — otherwise a dead node ping-ponged between live peers never
    # ages out (the "ghost" bug).
    r = Registry()
    r.merge({"X": {"decoders": [], "hb": 5.0}}, now=100.0, ttl=60.0)   # learn X@hb5
    r.merge({"X": {"decoders": [], "hb": 5.0}}, now=150.0, ttl=60.0)   # stale relay, same hb
    assert "X" not in r.stages_by_url(now=170.0)   # original expiry 160 stands → gone


def test_merge_newer_hb_refreshes_expiry():
    # A relayed entry with a fresher heartbeat (the origin is alive and still
    # advertising) MUST refresh the expiry.
    r = Registry()
    r.merge({"X": {"decoders": [], "hb": 5.0}}, now=100.0, ttl=60.0)
    r.merge({"X": {"decoders": [], "hb": 6.0}}, now=150.0, ttl=60.0)   # newer advert relayed
    assert "X" in r.stages_by_url(now=170.0)        # refreshed to 210


def test_relay_does_not_resurrect_dead_node():
    # Two live registries keep gossiping a third node X that has DIED (its hb is
    # frozen). With expiry propagation X must eventually prune everywhere.
    a, b = Registry(), Registry()
    x = {"embed": True, "head": True, "decoders": ["0-24"], "hb": 1.0}
    a.merge({"http://x": x}, now=100.0, ttl=60.0)
    b.merge({"http://x": x}, now=100.0, ttl=60.0)
    t = 105.0
    for _ in range(30):                              # X never advertises again
        a.merge(b.stages_by_url(now=t), now=t, ttl=60.0)
        b.merge(a.stages_by_url(now=t), now=t, ttl=60.0)
        a.prune(now=t)
        b.prune(now=t)
        t += 10.0
    assert "http://x" not in a.stages_by_url(now=t)
    assert "http://x" not in b.stages_by_url(now=t)


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
