import sys, time
from synapse.ui.manager import NodeManager


def test_start_status_stop():
    mgr = NodeManager()
    assert mgr.status() == {}
    mgr.start("worker", [sys.executable, "-c", "import time; time.sleep(30)"], {"stages": "decoder:0-8"})
    st = mgr.status()
    assert st["worker"]["running"] is True
    assert st["worker"]["stages"] == "decoder:0-8"
    assert isinstance(st["worker"]["pid"], int)
    mgr.stop("worker")
    time.sleep(0.3)
    assert mgr.status().get("worker", {}).get("running", False) is False or "worker" not in mgr.status()


def test_start_replaces_previous():
    mgr = NodeManager()
    mgr.start("coordinator", [sys.executable, "-c", "import time; time.sleep(30)"], {"port": 9001})
    pid1 = mgr.status()["coordinator"]["pid"]
    mgr.start("coordinator", [sys.executable, "-c", "import time; time.sleep(30)"], {"port": 9002})
    pid2 = mgr.status()["coordinator"]["pid"]
    assert pid1 != pid2
    mgr.stop("coordinator")
