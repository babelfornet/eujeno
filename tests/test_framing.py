from synapse.net.framing import pack, unpack


def test_roundtrip_header_and_payload():
    header = {"op": "decode", "block_key": "0-12", "job_id": "j1", "req_id": "r3"}
    payload = b"\x00\x01\x02binarydata"
    header2, payload2 = unpack(pack(header, payload))
    assert header2 == header
    assert payload2 == payload


def test_roundtrip_empty_payload():
    header2, payload2 = unpack(pack({"op": "end"}))
    assert header2 == {"op": "end"}
    assert payload2 == b""
