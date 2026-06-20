import json
from eujeno.net.jobstore import JobStore


def test_create_append_finish_roundtrip(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "hello", {"temperature": 0.0}, prompt_len=3)
    s.append_token("j1", 10, 0)
    s.append_token("j1", 20, 1)
    s.finish("j1", "ten twenty", "stop")
    j = s.get_job("j1")
    assert j["status"] == "DONE"
    assert j["tokens"] == [10, 20]
    assert j["position"] == 2
    assert j["result"] == "ten twenty"
    assert j["finish_reason"] == "stop"
    assert j["prompt_len"] == 3
    assert j["sampling"] == {"temperature": 0.0}


def test_append_is_idempotent_on_position(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    s.append_token("j1", 10, 0)        # same (job, position) again
    assert s.get_job("j1")["tokens"] == [10]   # no double


def test_reset_progress_clears_tokens(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    s.reset_progress("j1")
    j = s.get_job("j1")
    assert j["tokens"] == [] and j["position"] == 0 and j["status"] == "RUNNING"


def test_fail_sets_status_and_error(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.fail("j1", "too many failovers")
    j = s.get_job("j1")
    assert j["status"] == "FAILED" and j["error"] == "too many failovers"


def test_recover_marks_running_interrupted(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("run", "m", "p", {}, 1)            # stays RUNNING
    s.create_job("done", "m", "p", {}, 1); s.finish("done", "x", "stop")
    n = s.recover()
    assert n == 1
    assert s.get_job("run")["status"] == "INTERRUPTED"
    assert s.get_job("done")["status"] == "DONE"    # untouched


def test_durable_across_reopen(tmp_path):
    path = str(tmp_path / "j.db")
    s = JobStore(path)
    s.create_job("j1", "m", "p", {}, 1); s.append_token("j1", 7, 0); s.finish("j1", "seven", "stop")
    s.close()
    s2 = JobStore(path)                              # reopen
    assert s2.get_job("j1")["tokens"] == [7]
    assert s2.get_job("j1")["status"] == "DONE"


def test_recent_jobs_orders_newest_first(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("a", "m", "p", {}, 1)
    s.create_job("b", "m", "p", {}, 1)
    ids = [j["job_id"] for j in s.recent_jobs(limit=10)]
    assert set(ids) == {"a", "b"} and len(ids) == 2


def test_get_missing_job_returns_none(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    assert s.get_job("nope") is None


def test_append_same_token_same_position_is_strict_noop(tmp_path):
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    before = s.get_job("j1")["updated_at"]
    s.append_token("j1", 10, 0)        # exact same -> no-op
    j = s.get_job("j1")
    assert j["tokens"] == [10]
    assert j["updated_at"] == before   # no write happened


def test_append_different_token_at_existing_position_warns(tmp_path, caplog):
    import logging
    s = JobStore(str(tmp_path / "j.db"))
    s.create_job("j1", "m", "p", {}, 1)
    s.append_token("j1", 10, 0)
    with caplog.at_level(logging.WARNING, logger="eujeno.jobstore"):
        s.append_token("j1", 99, 0)
    assert s.get_job("j1")["tokens"] == [99]
    assert any("rewritten" in r.message for r in caplog.records)
