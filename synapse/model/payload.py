import json
from dataclasses import dataclass

import torch
import safetensors.torch


@dataclass
class HopPayload:
    """Payload di un hop sul filo (Parte 1 §3). La KV-cache NON viaggia qui:
    resta locale all'holder (session affinity, Parte 3)."""
    job_id: str
    hop: int
    token_position: int
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def to_bytes(self) -> bytes:
        header = {"job_id": self.job_id, "hop": self.hop, "token_position": self.token_position}
        header_bytes = json.dumps(header).encode("utf-8")
        tensors = {
            "_header": torch.tensor(list(header_bytes), dtype=torch.uint8),
            "hidden_states": self.hidden_states.contiguous(),
            "position_ids": self.position_ids.contiguous(),
            "cache_position": self.cache_position.contiguous(),
        }
        if self.attention_mask is not None:
            tensors["attention_mask"] = self.attention_mask.contiguous()
        return safetensors.torch.save(tensors)

    @classmethod
    def from_bytes(cls, data: bytes) -> "HopPayload":
        t = safetensors.torch.load(data)
        header = json.loads(bytes(t["_header"].tolist()).decode("utf-8"))
        return cls(
            job_id=header["job_id"],
            hop=header["hop"],
            token_position=header["token_position"],
            hidden_states=t["hidden_states"],
            position_ids=t["position_ids"],
            cache_position=t["cache_position"],
            attention_mask=t.get("attention_mask"),
        )
