from dataclasses import dataclass, field


@dataclass
class StageSpec:
    """Quali stage serve un nodo (per `synapse serve`)."""
    embed: bool = False
    head: bool = False
    decoders: list = field(default_factory=list)   # list[tuple[int, int]]


def parse_stages(spec: str) -> StageSpec:
    """Parsa una stringa tipo 'embed,decoder:0-12,head' in uno StageSpec."""
    out = StageSpec()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "embed":
            out.embed = True
        elif token == "head":
            out.head = True
        elif token.startswith("decoder:"):
            rng = token[len("decoder:"):]
            try:
                lo, hi = rng.split("-")
                out.decoders.append((int(lo), int(hi)))
            except ValueError:
                raise ValueError(f"range decoder non valido: {token!r} (atteso decoder:LO-HI)")
        else:
            raise ValueError(f"stage non riconosciuto: {token!r}")
    return out


@dataclass
class Topology:
    """Mappa stage->URL per l'inferenza distribuita (per `synapse infer`)."""
    model: str
    embed: str
    head: str
    decoders: list   # list[tuple[block_key, url]]

    def all_urls(self) -> list:
        seen = []
        for url in [self.embed, *[u for _, u in self.decoders], self.head]:
            if url not in seen:
                seen.append(url)
        return seen


def load_topology(data: dict) -> Topology:
    """Costruisce una Topology da un dict (es. caricato da JSON)."""
    decoders = [(d["block"], d["url"]) for d in data["decoders"]]
    return Topology(model=data["model"], embed=data["embed"], head=data["head"], decoders=decoders)
