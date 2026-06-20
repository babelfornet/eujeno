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
