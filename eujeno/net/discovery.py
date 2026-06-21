# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import threading


def _hb_newer(new_hb, cur_hb) -> bool:
    """Whether a relayed heartbeat is strictly fresher than the held one. Missing
    heartbeats sort oldest, so a node still emitting `hb` always wins over a legacy
    (None) entry, and a None relay never refreshes an entry that has an hb."""
    if new_hb is None:
        return False
    if cur_hb is None:
        return True
    return new_hb > cur_hb


class Registry:
    """Decentralized discovery state: url -> {stages, expiry}. Relative TTL:
    learned entries expire at now+ttl unless refreshed by the gossip.

    Thread-safe: the node's gossip/probe run in a daemon OS thread (writes) while
    the async request handlers read concurrently, so every accessor takes a lock."""
    def __init__(self):
        self.entries = {}   # url -> {"stages": dict, "expiry": float}
        self._lock = threading.RLock()

    def upsert(self, url: str, stages: dict, now: float, ttl: float) -> None:
        with self._lock:
            self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def merge(self, stages_by_url: dict, now: float, ttl: float) -> None:
        """Relay (gossip pull): adopt entries learned from a peer. Unlike upsert
        (which is the origin authoritatively refreshing its OWN record), a relay
        must NOT keep extending an entry's expiry on every round — otherwise a
        dead node gossiped back and forth between two live peers never ages out.
        We therefore refresh an existing entry only when the incoming heartbeat
        (`hb`, stamped by the origin and carried verbatim through relays) is newer
        than the one we hold; a frozen hb (dead origin) is left to expire."""
        with self._lock:
            for url, stages in stages_by_url.items():
                cur = self.entries.get(url)
                if cur is None or _hb_newer(stages.get("hb"), cur["stages"].get("hb")):
                    self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def prune(self, now: float) -> None:
        with self._lock:
            self.entries = {u: e for u, e in self.entries.items() if e["expiry"] > now}

    def stages_by_url(self, now: float) -> dict:
        with self._lock:
            return {u: e["stages"] for u, e in self.entries.items() if e["expiry"] > now}


def build_chain(stages_by_url: dict, num_layers: int, exclude=None, load=None, reputation=None, speed=None):
    """From {url: {'embed','head','decoders':[block_key]}} builds
    (embed_url, [(block_key, url)...], head_url) that tiles [0, num_layers),
    ignoring the ids in `exclude`. Returns None if coverage is incomplete."""
    exclude = exclude or set()
    items = {u: s for u, s in stages_by_url.items() if u not in exclude}
    embed = next((u for u, s in items.items() if s.get("embed")), None)
    head = next((u for u, s in items.items() if s.get("head")), None)
    if embed is None or head is None:
        return None
    ranges = []
    for u, s in items.items():
        for bk in s.get("decoders", []):
            lo, hi = (int(x) for x in bk.split("-"))
            ranges.append((lo, hi, bk, u))
    ranges.sort()
    chain = []
    cursor = 0
    for lo, hi, bk, u in ranges:
        if lo == cursor and hi > cursor:
            chain.append((bk, u))
            cursor = hi
    if cursor != num_layers:
        return None
    if load is not None or reputation is not None or speed is not None:
        order = {u: i for i, u in enumerate(items)}
        L = load or {}
        R = reputation or {}
        S = speed or {}
        def _least(cands):
            return min(cands, key=lambda u: (-R.get(u, 0.0), -S.get(u, 0.0), L.get(u, 0), order[u]))
        embed = _least([u for u, s in items.items() if s.get("embed")])
        head = _least([u for u, s in items.items() if s.get("head")])
        chain = [(bk, _least([u for u, s in items.items() if bk in s.get("decoders", [])]))
                 for bk, u in chain]
    return embed, chain, head


def coverage_gaps(stages_by_url: dict, num_layers: int, target: int = 1) -> dict:
    """Decoder ranges with replicas < target (uncovered or under-replicated), plus the
    replica count of embed/head. `stages_by_url`: {url: {'embed','head','decoders'}}."""
    cover = [0] * num_layers
    for s in stages_by_url.values():
        for bk in s.get("decoders", []):
            lo, hi = (int(x) for x in bk.split("-"))
            for i in range(max(0, lo), min(hi, num_layers)):
                cover[i] += 1
    gaps = []
    i = 0
    while i < num_layers:
        if cover[i] < target:
            j = i
            while j < num_layers and cover[j] < target:
                j += 1
            gaps.append({"lo": i, "hi": j, "replicas": min(cover[i:j])})
            i = j
        else:
            i += 1
    return {
        "decoder_gaps": gaps,
        "embed_replicas": sum(1 for s in stages_by_url.values() if s.get("embed")),
        "head_replicas": sum(1 for s in stages_by_url.values() if s.get("head")),
        "target": target,
    }
