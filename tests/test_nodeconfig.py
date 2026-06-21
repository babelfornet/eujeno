from eujeno.net.nodeconfig import NodeConfig, DEFAULTS

def test_defaults_and_stable_peer_id(tmp_path):
    p = str(tmp_path / "n.json")
    c = NodeConfig(p)
    pid = c.peer_id
    assert pid and pid.startswith("node·")
    assert c.get()["region"] == DEFAULTS["region"]
    # reload keeps the same peer id + persisted changes
    c.update({"name": "alpha", "region": "us-east"})
    c2 = NodeConfig(p)
    assert c2.peer_id == pid
    assert c2.get()["name"] == "alpha" and c2.get()["region"] == "us-east"

def test_peer_id_immutable_and_unknown_keys_ignored(tmp_path):
    c = NodeConfig(str(tmp_path / "n.json"))
    pid = c.peer_id
    out = c.update({"peerId": "node·hack", "bogus": 1, "maxLayers": 12})
    assert out["peerId"] == pid
    assert "bogus" not in out
    assert out["maxLayers"] == 12

def test_in_memory_without_path():
    c = NodeConfig(None)
    assert c.peer_id and c.update({"name": "x"})["name"] == "x"
