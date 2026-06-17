class Registry:
    """Stato di discovery decentralizzato: url -> {stages, expiry}. TTL relativo:
    le entry apprese scadono a now+ttl se non rinfrescate dal gossip."""
    def __init__(self):
        self.entries = {}   # url -> {"stages": dict, "expiry": float}

    def upsert(self, url: str, stages: dict, now: float, ttl: float) -> None:
        self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def merge(self, stages_by_url: dict, now: float, ttl: float) -> None:
        for url, stages in stages_by_url.items():
            self.entries[url] = {"stages": stages, "expiry": now + ttl}

    def prune(self, now: float) -> None:
        self.entries = {u: e for u, e in self.entries.items() if e["expiry"] > now}

    def stages_by_url(self, now: float) -> dict:
        return {u: e["stages"] for u, e in self.entries.items() if e["expiry"] > now}


def build_chain(stages_by_url: dict, num_layers: int, exclude=None):
    """Da {url: {'embed','head','decoders':[block_key]}} costruisce
    (embed_url, [(block_key, url)...], head_url) che tassella [0, num_layers),
    ignorando gli id in `exclude`. Ritorna None se la coverage è incompleta."""
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
    return embed, chain, head
