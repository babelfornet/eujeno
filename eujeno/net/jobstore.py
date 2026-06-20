# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Durable job log for the coordinator: a small SQLite(WAL) store of distributed
generation jobs. Single responsibility, no network/torch deps."""

import json
import logging
import os
import sqlite3
import time

log = logging.getLogger("eujeno.jobstore")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id        TEXT PRIMARY KEY,
  model_id      TEXT,
  status        TEXT,
  prompt        TEXT,
  sampling_json TEXT,
  prompt_len    INTEGER,
  position      INTEGER,
  tokens_json   TEXT,
  result        TEXT,
  finish_reason TEXT,
  error         TEXT,
  created_at    REAL,
  updated_at    REAL
);
CREATE TABLE IF NOT EXISTS receipts (
  job_id     TEXT,
  peer_id    TEXT,
  hops       INTEGER,
  bytes      INTEGER,
  t_compute  REAL,
  PRIMARY KEY (job_id, peer_id)
);
"""


class JobStore:
    """Durable per-coordinator job log. status: RUNNING|DONE|FAILED|INTERRUPTED."""

    def __init__(self, path):
        self.path = path
        if path != ":memory:" and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_job(self, job_id, model_id, prompt, sampling, prompt_len):
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, model_id, status, prompt, sampling_json, "
            "prompt_len, position, tokens_json, result, finish_reason, error, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, model_id, "RUNNING", prompt, json.dumps(sampling or {}), int(prompt_len),
             0, json.dumps([]), None, None, None, now, now))
        self._conn.commit()

    def append_token(self, job_id, token_id, position):
        row = self._conn.execute("SELECT tokens_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return
        toks = json.loads(row["tokens_json"] or "[]")
        if position < len(toks):
            if toks[position] == int(token_id):
                return                            # strict idempotent no-op: same token, same position
            log.warning("append_token: job %s position %d rewritten %s -> %s",
                        job_id, position, toks[position], int(token_id))
            toks[position] = int(token_id)
        elif position == len(toks):
            toks.append(int(token_id))
        else:
            return                                # out-of-order beyond next: ignore (not expected)
        self._conn.execute("UPDATE jobs SET tokens_json=?, position=?, updated_at=? WHERE job_id=?",
                           (json.dumps(toks), len(toks), time.time(), job_id))
        self._conn.commit()

    def reset_progress(self, job_id):
        self._conn.execute("UPDATE jobs SET tokens_json=?, position=0, updated_at=? WHERE job_id=?",
                           (json.dumps([]), time.time(), job_id))
        self._conn.commit()

    def finish(self, job_id, result, finish_reason):
        self._conn.execute("UPDATE jobs SET status=?, result=?, finish_reason=?, updated_at=? WHERE job_id=?",
                           ("DONE", result, finish_reason, time.time(), job_id))
        self._conn.commit()

    def fail(self, job_id, error):
        self._conn.execute("UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
                           ("FAILED", str(error), time.time(), job_id))
        self._conn.commit()

    def set_status(self, job_id, status):
        self._conn.execute("UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
                           (status, time.time(), job_id))
        self._conn.commit()

    def recover(self):
        cur = self._conn.execute(
            "UPDATE jobs SET status='INTERRUPTED', updated_at=? "
            "WHERE status IN ('RUNNING', 'WAITING_COVERAGE')", (time.time(),))
        self._conn.commit()
        return cur.rowcount

    def get_job(self, job_id):
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def recent_jobs(self, limit=50):
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def add_receipts(self, job_id, receipts):
        for peer_id, r in (receipts or {}).items():
            self._conn.execute(
                "INSERT INTO receipts (job_id, peer_id, hops, bytes, t_compute) VALUES (?,?,?,?,?) "
                "ON CONFLICT(job_id, peer_id) DO UPDATE SET hops=hops+excluded.hops, "
                "bytes=bytes+excluded.bytes, t_compute=t_compute+excluded.t_compute",
                (job_id, peer_id, int(r.get("hops", 0)), int(r.get("bytes", 0)), float(r.get("t_compute", 0.0))))
        self._conn.commit()

    def get_receipts(self, job_id):
        rows = self._conn.execute(
            "SELECT peer_id, hops, bytes, t_compute FROM receipts WHERE job_id=? ORDER BY peer_id",
            (job_id,)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row):
        d = dict(row)
        d["sampling"] = json.loads(d.pop("sampling_json") or "{}")
        d["tokens"] = json.loads(d.pop("tokens_json") or "[]")
        return d

    def close(self):
        self._conn.close()
