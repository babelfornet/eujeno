# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field


@dataclass
class StageSpec:
    """Which stages a node serves (for `eujeno serve`)."""
    embed: bool = False
    head: bool = False
    decoders: list = field(default_factory=list)   # list[tuple[int, int]]


def parse_stages(spec: str) -> StageSpec:
    """Parses a string like 'embed,decoder:0-12,head' into a StageSpec."""
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
                raise ValueError(f"invalid decoder range: {token!r} (expected decoder:LO-HI)")
        else:
            raise ValueError(f"unrecognized stage: {token!r}")
    return out


@dataclass
class Topology:
    """Maps stage->URL for distributed inference (for `eujeno infer`)."""
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
    """Builds a Topology from a dict (e.g. loaded from JSON)."""
    decoders = [(d["block"], d["url"]) for d in data["decoders"]]
    return Topology(model=data["model"], embed=data["embed"], head=data["head"], decoders=decoders)
