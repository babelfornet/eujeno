import torch
from transformers import DynamicCache

from .masking import build_causal_mask


class EmbedBlock:
    """Primo blocco: input_ids -> hidden_states."""
    def __init__(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @torch.no_grad()
    def run_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class DecoderBlock:
    """Slab contiguo di layer [lo, hi). Mantiene una KV-cache LOCALE per i
    soli suoi layer (indici rimappati 0-based)."""
    def __init__(self, layers, rotary_emb):
        self.layers = layers
        self.rotary_emb = rotary_emb
        self.cache = DynamicCache()

    @torch.no_grad()
    def run_block(self, hidden_states: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        position_ids = cache_position.unsqueeze(0)
        past_len = self.cache.get_seq_length()
        kv_len = past_len + hidden_states.shape[1]
        attn_mask = build_causal_mask(cache_position, kv_len, hidden_states.dtype, hidden_states.device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        return hidden_states

    def get_cache(self):
        return self.cache

    def set_cache(self, cache):
        self.cache = cache


class HeadBlock:
    """Ultimo blocco: hidden_states -> logits (final norm + lm_head)."""
    def __init__(self, norm, lm_head):
        self.norm = norm
        self.lm_head = lm_head

    @torch.no_grad()
    def run_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))


def split_into_blocks(model, boundaries: list[int]):
    """Divide un modello caricato in (EmbedBlock, [DecoderBlock...], HeadBlock).

    boundaries: confini dei layer decoder, es. [0, 12, 24] -> due slab [0:12),[12:24).

    ATTENZIONE: muta layer.self_attn.layer_idx a indici locali. Catturare ogni
    riferimento dal modello intero PRIMA di chiamare questa funzione.
    """
    inner = model.model
    embed = EmbedBlock(inner.embed_tokens)
    head = HeadBlock(inner.norm, model.lm_head)

    decoders = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        layers = inner.layers[lo:hi]
        for local_idx, layer in enumerate(layers):
            layer.self_attn.layer_idx = local_idx   # rimappa a indice locale del blocco
        decoders.append(DecoderBlock(layers, inner.rotary_emb))

    return embed, decoders, head


def compute_boundaries(num_layers: int, n_blocks: int) -> list[int]:
    """Divide num_layers in n_blocks blocchi decoder contigui il più possibile
    uguali. Ritorna i confini, es. (24, 2) -> [0, 12, 24]. Copre sempre
    [0, num_layers] in modo strettamente crescente."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks deve essere >= 1, ricevuto {n_blocks}")
    if n_blocks > num_layers:
        raise ValueError(f"n_blocks ({n_blocks}) non può superare num_layers ({num_layers})")
    base, extra = divmod(num_layers, n_blocks)
    boundaries = [0]
    for i in range(n_blocks):
        size = base + (1 if i < extra else 0)   # i primi `extra` blocchi hanno un layer in più
        boundaries.append(boundaries[-1] + size)
    return boundaries
