from eujeno.net.discovery import build_chain


def _s(embed=False, head=False, decoders=()):
    return {"embed": embed, "head": head, "decoders": list(decoders)}


def test_default_path_unchanged_with_redundancy():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load=None) == build_chain(s, 24)
    e, chain, h = build_chain(s, 24)
    assert e == "a" and chain == [("0-24", "a")] and h == "b"   # first head, insertion order


def test_prefers_least_loaded_head():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"b": 5, "c": 0})[2] == "c"
    assert build_chain(s, 24, load={"b": 0, "c": 5})[2] == "b"


def test_prefers_least_loaded_embed():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(embed=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"a": 7, "b": 0})[0] == "b"


def test_prefers_least_loaded_decoder_replica():
    s = {"a": _s(embed=True), "b": _s(decoders=["0-24"]), "c": _s(decoders=["0-24"]), "d": _s(head=True)}
    assert build_chain(s, 24, load={"b": 9, "c": 0})[1] == [("0-24", "c")]


def test_load_tie_is_deterministic_insertion_order():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"b": 0, "c": 0})[2] == "b"


def test_incomplete_coverage_still_none_with_load():
    s = {"a": _s(embed=True, decoders=["0-12"]), "b": _s(head=True)}   # 12-24 missing
    assert build_chain(s, 24, load={"a": 0, "b": 0}) is None


def test_prefers_higher_reputation_head():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, reputation={"b": 0.0, "c": 5.0})[2] == "c"
    assert build_chain(s, 24, reputation={"b": 5.0, "c": 0.0})[2] == "b"


def test_reputation_is_primary_over_load():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    # c has higher reputation; it wins even though it is more loaded
    assert build_chain(s, 24, load={"b": 0, "c": 9}, reputation={"b": 0.0, "c": 5.0})[2] == "c"


def test_equal_reputation_falls_back_to_load():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    assert build_chain(s, 24, load={"b": 3, "c": 0}, reputation={"b": 1.0, "c": 1.0})[2] == "c"


def test_reputation_none_matches_load_only_path():
    s = {"a": _s(embed=True, decoders=["0-24"]), "b": _s(head=True), "c": _s(head=True)}
    a = build_chain(s, 24, load={"b": 3, "c": 0})
    b = build_chain(s, 24, load={"b": 3, "c": 0}, reputation=None)
    assert a == b


def test_prefers_higher_reputation_decoder_replica():
    s = {"a": _s(embed=True), "b": _s(decoders=["0-24"]), "c": _s(decoders=["0-24"]), "d": _s(head=True)}
    assert build_chain(s, 24, reputation={"b": 0.0, "c": 9.0})[1] == [("0-24", "c")]


def test_build_chain_prefers_faster_peer():
    # two redundant heads; u_fast faster (higher speed) than u_slow -> u_fast chosen
    stages = {
        "u_embed": _s(embed=True, decoders=["0-12"]),
        "u_fast":  _s(head=True, decoders=["12-24"]),
        "u_slow":  _s(head=True, decoders=["12-24"]),
    }
    chain = build_chain(stages, 24, speed={"u_fast": 9.0, "u_slow": 1.0})
    _, decoders, head = chain
    assert head == "u_fast"
    assert dict(decoders)["12-24"] == "u_fast"
    # default path unchanged when no maps given
    assert build_chain(stages, 24) is not None
