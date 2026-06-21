# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Process-local node metrics: requests served, throughput, neighbor latency,
observed per-peer hop time (feeds speed-aware routing), uptime."""
import time


class NodeMetrics:
    def __init__(self, ewma_alpha=0.3):
        self.requests_served = 0
        self._alpha = ewma_alpha
        self._started = time.monotonic()
        self.peer_latency = {}    # url -> ms (EWMA)
        self.peer_hop_time = {}   # url -> seconds (EWMA)
        self._recent = []         # [(tokens, elapsed_seconds)] recent finished jobs

    def _ewma(self, store, key, val):
        prev = store.get(key)
        store[key] = val if prev is None else (1 - self._alpha) * prev + self._alpha * val

    def inc_request(self, n=1):
        self.requests_served += n

    def record_job(self, tokens, elapsed):
        if elapsed and elapsed > 0:
            self._recent.append((tokens, elapsed))
            self._recent = self._recent[-50:]

    def observe_latency(self, url, ms):
        self._ewma(self.peer_latency, url, float(ms))

    def observe_hop_time(self, url, seconds):
        if seconds and seconds > 0:
            self._ewma(self.peer_hop_time, url, float(seconds))

    def uptime_sec(self):
        return time.monotonic() - self._started

    def throughput_tok_s(self):
        if not self._recent:
            return 0.0
        tok = sum(t for t, _ in self._recent)
        el = sum(e for _, e in self._recent)
        return round(tok / el, 1) if el > 0 else 0.0

    def avg_latency_ms(self):
        vals = list(self.peer_latency.values())
        return round(sum(vals) / len(vals)) if vals else 0

    def speed_map(self, urls):
        """Per-url score, higher = faster observed hop. Unmeasured peers get a
        neutral (optimistic) default so they are still explored."""
        measured = [v for v in self.peer_hop_time.values() if v > 0]
        neutral_time = (sum(measured) / len(measured)) if measured else 0.5
        out = {}
        for u in urls:
            t = self.peer_hop_time.get(u)
            out[u] = 1.0 / max(t if t else neutral_time, 1e-3)
        return out
