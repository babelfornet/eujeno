import synapse.mcp_config as mc


def test_add_list_remove(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))
    assert mc.load_servers() == {}
    mc.add_server("fs", "npx", ["@modelcontextprotocol/server-filesystem", "/tmp"])
    s = mc.load_servers()
    assert s["fs"]["command"] == "npx"
    assert s["fs"]["args"] == ["@modelcontextprotocol/server-filesystem", "/tmp"]
    mc.remove_server("fs")
    assert mc.load_servers() == {}


def test_persists_across_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))
    mc.add_server("echo", "python", ["server.py"])
    assert "echo" in mc.load_servers()
