"""Decisione di auto-assegnazione: dato il quadro dei buchi (coverage_gaps) e la
capacità del nodo, sceglie lo stage spec da rivendicare. Funzione pura."""


def choose_stages(gaps: dict, max_decoder_layers: int, num_layers: int,
                  take_embed_head: bool) -> str:
    """Ritorna uno stage spec per parse_stages (es. 'embed,decoder:12-17,head'),
    o '' se non c'è nulla di utile/possibile da rivendicare."""
    target = gaps.get("target", 1)
    parts = []
    if take_embed_head and gaps.get("embed_replicas", 0) < target:
        parts.append("embed")
    decoder_gaps = sorted(gaps.get("decoder_gaps", []),
                          key=lambda g: (g["replicas"], -(g["hi"] - g["lo"])))
    if decoder_gaps and max_decoder_layers > 0:
        g = decoder_gaps[0]
        hi = min(g["hi"], g["lo"] + max_decoder_layers)
        parts.append(f"decoder:{g['lo']}-{hi}")
    if take_embed_head and gaps.get("head_replicas", 0) < target:
        parts.append("head")
    return ",".join(parts)
